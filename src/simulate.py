import pandas as pd
import numpy as np
#Simulate household-level retailer choices under switching costs and contract lock-in.
def simulate_data(
    seed_1: int = 42,
    seed_2: int = 123,
    n_nonmover: int = 5000,
    n_periods: int = 10,
    n_retailers: int = 6,
    n_mover_per_period: int = 600,
    sigma_true: float = 0.5,
    theta_p_true: float = -0.3,
    gamma_A_true: float = -1.0,
    gamma_R_true: float = -2.0,
) -> dict:
    '''
    Simulate household-level retailer choices under switching costs and contract lock-in.

    Economic setup
    --------------
    - There are `n_retailers` retailers.
    - Retailer 'r0' is the incumbent AREP.
    - Retailer 'r1' is a long-contract retailer.
    - Retailer 'r5' is the outside option with zero utility and zero price.
    - All nonmovers start with the AREP in period 0.
    - Movers enter in one specific period and make a choice only in that period.
    - Households face switching costs when leaving their current retailer:
        * `gamma_A_true` if leaving the AREP
        * `gamma_R_true` if leaving a non-AREP retailer
    - Utility includes:
        * retailer-time mean utility delta_jt
        * household-specific AREP preference v_i
        * price sensitivity theta_p_true
        * switching costs
        * i.i.d. Type-I extreme value taste shocks

    Parameters
    ----------
    seed_1 : int
        Random seed 1.
    seed_2 : int
        Random seed 2.
    n_nonmover : int
        Number of nonmover households observed over all periods.
    n_periods : int
        Number of time periods.
    n_retailers : int
        Number of retailers. Must be at least 6 because retailer identities are fixed.
    n_mover_per_period : int
        Number of movers entering in each period.
    sigma_true : float
        Standard deviation of the random coefficient on AREP.
    theta_p_true : float
        Price coefficient.
    gamma_A_true : float
        Switching cost for leaving the AREP.
    gamma_R_true : float
        Switching cost for leaving a non-AREP retailer.

    Returns
    -------
    dict
        A dictionary containing:
        - "retailers": list of retailer IDs
        - "retailer_time_df": retailer-time offers and mean utilities
        - "panel_df": simulated household-period choice panel
        - "true_params": true parameter values used in the DGP
        - "meta": metadata including retailer roles and contract lengths
        - "latent": latent household-specific random coefficients

    Notes
    -----
    This simulation is designed for a parameter-recovery exercise. It keeps
    the data-generating process transparent rather than maximally general.
    '''
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if n_retailers < 6:
        raise ValueError("n_retailers must be at least 6 because the code uses r5 as the outside option.")
    if n_nonmover < 0:
        raise ValueError("n_nonmover must be nonnegative.")
    if n_periods < 1:
        raise ValueError("n_periods must be at least 1.")
    if n_mover_per_period < 0:
        raise ValueError("n_mover_per_period must be nonnegative.")
    if sigma_true < 0:
        raise ValueError("sigma_true must be nonnegative.")

    # seeds
    #rng = np.random.default_rng(seed)
    rng_v = np.random.default_rng(seed_1)        # for v_i
    rng_off = np.random.default_rng(seed_1+1) 
    rng = np.random.default_rng(seed_2)
    # ------------------------------------------------------------------
    # Retailer identities and contract lengths
    # ------------------------------------------------------------------
    retailers = [f"r{i}" for i in range(n_retailers)]

    arep_id = "r0"
    long_contract_id = "r1"
    outside_id = "r5"

    if arep_id not in retailers or long_contract_id not in retailers or outside_id not in retailers:
        raise ValueError("Retailer roles are not consistent with n_retailers.")

    contract_len = {j: 12 for j in retailers}
    contract_len[arep_id] = 1
    contract_len[long_contract_id] = 36
    contract_len[outside_id] = 1

    # ------------------------------------------------------------------
    # Step 1: Construct household population
    # ------------------------------------------------------------------
    nonmover_ids = [f"nonmover_{i}" for i in range(n_nonmover)]
    mover_birth = {
        f"mover_{t}_{i}": t
        for t in range(n_periods)
        for i in range(n_mover_per_period)
    }
    mover_ids = list(mover_birth.keys())
    all_households = nonmover_ids + mover_ids

    # Household-specific random coefficient on AREP preference
    v_i = dict(zip(all_households, rng_v.normal(0, sigma_true, size=len(all_households))))

    def is_active(hh_id: str, t: int) -> bool:
        """
        Nonmovers are active in all periods.
        Movers are active only in their entry period.
        """
        return (hh_id in nonmover_ids) or (mover_birth.get(hh_id) == t)


    # ------------------------------------------------------------------
    # Step 2: Simulate retailer-time mean utilities and offer prices
    # ------------------------------------------------------------------
    retailer_time_rows = []

    for t in range(n_periods):
        common_shock_t = rng_off.normal(0.0, 0.3)

        for j in retailers:
            if j == outside_id:
                delta_jt = 0.0
                price_offer_jt = 0.0

            elif j == arep_id:
                delta_jt = 3.2 + common_shock_t + rng_off.normal(0.0, 0.1)
                price_offer_jt = rng_off.uniform(8, 12)

            else:
                delta_jt = 2.8 + common_shock_t + rng_off.normal(0.0, 0.1)
                price_offer_jt = rng_off.uniform(8, 12)

            retailer_time_rows.append(
                {
                    "time": t,
                    "ran_rep_id": j,
                    "delta": delta_jt,
                    "price_offer": price_offer_jt,
                }
            )

    retailer_time_df = pd.DataFrame(retailer_time_rows)
    retailer_time_map = retailer_time_df.set_index(["time", "ran_rep_id"])[["delta", "price_offer"]]

   # ------------------------------------------------------------------
    # Step 3: Simulate household choice paths
    # ------------------------------------------------------------------
    # Household state variables at the start of each period:
    # - current_retailer: current provider
    # - locked_price: contract price if household stays
    # - remaining: number of periods remaining on current contract, including current period
    state = {}

    # Initialize nonmovers in period 0 at the AREP
    initial_price_arep = float(retailer_time_map.loc[(0, arep_id), "price_offer"])
    initial_contract_len_arep = contract_len[arep_id]

    for hh_id in nonmover_ids:
        state[hh_id] = {
            "current_retailer": arep_id,
            "locked_price": initial_price_arep,
            "remaining": initial_contract_len_arep,
        }

    panel_rows = []

    for t in range(n_periods):
        for hh_id in all_households:
            if not is_active(hh_id, t):
                continue

            # Movers enter only once, with no previous retailer
            if hh_id in mover_ids and mover_birth[hh_id] == t:
                state[hh_id] = {
                    "current_retailer": None,
                    "locked_price": None,
                    "remaining": 0,
                }

            prev_retailer = state[hh_id]["current_retailer"]

            # Price faced if household stays with the current retailer
            can_stay = prev_retailer is not None
            if can_stay and state[hh_id]["remaining"] > 0:
                stay_price_at_t = state[hh_id]["locked_price"]
            elif can_stay:
                stay_price_at_t = float(retailer_time_map.loc[(t, prev_retailer), "price_offer"])
            else:
                stay_price_at_t = None

            utilities_det = []

            for j in retailers:
                delta_jt = float(retailer_time_map.loc[(t, j), "delta"])
                current_offer_jt = float(retailer_time_map.loc[(t, j), "price_offer"])

                if prev_retailer is not None and j == prev_retailer:
                    price_faced = stay_price_at_t
                else:
                    price_faced = current_offer_jt

                is_arep = 1 if j == arep_id else 0

                switch_cost_arep = 1 if (prev_retailer == arep_id and j != arep_id) else 0
                switch_cost_rep = 1 if (prev_retailer is not None and prev_retailer != arep_id and j != prev_retailer) else 0

                u_det = (
                    delta_jt
                    + v_i[hh_id] * is_arep
                    + theta_p_true * price_faced
                    + gamma_A_true * switch_cost_arep
                    + gamma_R_true * switch_cost_rep
                )
                utilities_det.append(u_det)

            utilities_det = np.asarray(utilities_det)

            # Draw i.i.d. Type-I extreme value shocks
            u_uniform = np.clip(rng.random(len(retailers)), 1e-12, 1 - 1e-12)
            gumbel_shocks = -np.log(-np.log(u_uniform))
            utilities_total = utilities_det + gumbel_shocks

            chosen_retailer = retailers[int(np.argmax(utilities_total))]

            # Update household state after the choice
            if prev_retailer is not None and chosen_retailer == prev_retailer:
                # If the contract expired at the start of the period, renew at current offer
                if state[hh_id]["remaining"] <= 0:
                    renewed_price = float(retailer_time_map.loc[(t, chosen_retailer), "price_offer"])
                    state[hh_id]["locked_price"] = renewed_price
                    state[hh_id]["remaining"] = contract_len[chosen_retailer]

                price_used = state[hh_id]["locked_price"]
                state[hh_id]["remaining"] -= 1

            else:
                # Switch to a new retailer and start a new contract
                new_price = float(retailer_time_map.loc[(t, chosen_retailer), "price_offer"])
                state[hh_id]["current_retailer"] = chosen_retailer
                state[hh_id]["locked_price"] = new_price
                state[hh_id]["remaining"] = contract_len[chosen_retailer] - 1
                price_used = new_price

            panel_rows.append(
                {
                    "hh_id": hh_id,
                    "time": t,
                    "mover": int(hh_id in mover_ids),
                    "prev_retailer": prev_retailer,
                    "stay_price_at_t": stay_price_at_t,
                    "chosen_retailer": chosen_retailer,
                    "price_used": price_used,
                    "remaining_contract_end_of_t": state[hh_id]["remaining"],
                }
            )

    panel_df = pd.DataFrame(panel_rows)

    return {
        "retailers": retailers,
        "retailer_time_df": retailer_time_df,
        "panel_df": panel_df,
        "true_params": {
            "sigma": sigma_true,
            "theta_p": theta_p_true,
            "gamma_A": gamma_A_true,
            "gamma_R": gamma_R_true,
        },
        "meta": {
            "arep_id": arep_id,
            "long_contract_id": long_contract_id,
            "outside_id": outside_id,
            "contract_len": contract_len,
            "n_nonmover": n_nonmover,
            "n_mover_per_period": n_mover_per_period,
            "n_periods": n_periods,
            "n_retailers": n_retailers,
            "seed_1": seed_1,
            "seed_2": seed_2,
        },
        "latent": {
            "v_i": v_i,
            "mover_birth": mover_birth,
        },
    }


# Example usage
# sim = simulate_data(seed=42)
# retailer_time_df = sim["retailer_time_df"]
# panel_df = sim["panel_df"]
# print(sim["true_params"])
# print(panel_df.head())
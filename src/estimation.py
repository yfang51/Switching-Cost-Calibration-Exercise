import gc
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import norm, qmc
from scipy.special import logsumexp


# ---------------------------------------------------------------------
# Simulation draws
# ---------------------------------------------------------------------
def halton_normal_draws(
    n_households: int,
    n_draws: int = 100,
    seed_3: int = 35,
    clip: float = 1e-12,
) -> np.ndarray:
    """
    Generate Halton-based quasi-random draws transformed to N(0, 1).

    Parameters
    ----------
    n_households : int
        Number of households.
    n_draws : int, default 100
        Number of simulation draws per household.
    seed : int, default 123
        Random seed for the scrambled Halton sequence.
    clip : float, default 1e-12
        Lower/upper clipping threshold before applying the inverse normal CDF.

    Returns
    -------
    np.ndarray
        Array of shape (n_households, n_draws) containing standard normal draws.
    """
    engine = qmc.Halton(d=1, scramble=True, seed =seed_3)
    u = engine.random(n_households * n_draws).reshape(n_households, n_draws)
    u = np.clip(u, clip, 1 - clip)
    return norm.ppf(u)


# ---------------------------------------------------------------------
# Core probability simulation
# ---------------------------------------------------------------------
def simulate_probability_block(
    r_lo: int,
    r_hi: int,
    v_draws: np.ndarray,
    delta_jt_vec: np.ndarray,
    arep: np.ndarray,
    sc_arep: np.ndarray,
    sc_rep: np.ndarray,
    hh_idx: np.ndarray,
    group_idx: np.ndarray,
    theta_p: float,
    gamma_A: float,
    gamma_R: float,
    price_faced: np.ndarray,
    G,
) -> np.ndarray:
    """
    Simulate choice probabilities for a block of random-coefficient draws.

    Returns
    -------
    np.ndarray
        Array of shape (n_rows, r_hi - r_lo) containing simulated probabilities.
    """
    utility_block = (
        delta_jt_vec[:, None]
        + theta_p * price_faced[:, None]
        + v_draws[hh_idx, r_lo:r_hi] * arep[:, None]
        + gamma_A * sc_arep[:, None]
        + gamma_R * sc_rep[:, None]
    )

    # Stabilize exponentiation within each draw column
    max_u = utility_block.max(axis=0, keepdims=True)
    exp_u = np.exp(utility_block - max_u)

    denom_block = G @ exp_u
    probs_block = exp_u / denom_block[group_idx, :]
    return probs_block


def simulated_choice_probabilities(
    theta_p: float,
    gamma_A: float,
    gamma_R: float,
    delta: np.ndarray,
    v_draws: np.ndarray,
    data: dict,
    jt_code: np.ndarray,
    mask_in: np.ndarray,
    block_size: int = 10,
    n_jobs: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute simulated choice probabilities.

    Parameters
    ----------
    theta_p, gamma_A, gamma_R : float
        Structural parameters.
    delta : np.ndarray
        Mean utility vector for included retailer-time cells.
    v_draws : np.ndarray
        Random coefficient draws with shape (n_households, n_draws).
    data : dict
        Output from build_estimation_data().
    jt_code : np.ndarray
        Integer code mapping each included row to an element of delta.
    mask_in : np.ndarray
        Boolean mask indicating rows whose retailer-time cells are included in delta.
    block_size : int, default 10
        Number of draws to process in each parallel block.
    n_jobs : int, default 5
        Number of parallel workers.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        - mean_probs: shape (n_rows,)
        - all_probs: shape (n_rows, n_draws)
    """
    estimation_df = data["estimation_df"]
    G = data["G"]
    group_idx = data["group_idx"]

    arep = estimation_df["AREP"].to_numpy()
    sc_arep = estimation_df["sc_AREP"].to_numpy()
    sc_rep = estimation_df["sc_REP"].to_numpy()
    hh_idx = estimation_df["hh_idx"].to_numpy()
    price_faced = estimation_df["real_price"].to_numpy()

    n_rows = len(estimation_df)

    delta_jt_vec = np.zeros(n_rows, dtype=float)
    delta_jt_vec[mask_in] = delta[jt_code[mask_in]]

    n_draws = v_draws.shape[1]
    block_ranges = [
        (r, min(r + block_size, n_draws))
        for r in range(0, n_draws, block_size)
    ]

    probability_blocks = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(simulate_probability_block)(
            r_lo,
            r_hi,
            v_draws,
            delta_jt_vec,
            arep,
            sc_arep,
            sc_rep,
            hh_idx,
            group_idx,
            theta_p,
            gamma_A,
            gamma_R,
            price_faced,
            G,
        )
        for r_lo, r_hi in block_ranges
    )

    all_probs = np.empty((n_rows, n_draws), dtype=np.float64)
    col = 0
    for block in probability_blocks:
        n_cols = block.shape[1]
        all_probs[:, col : col + n_cols] = block
        col += n_cols

    del probability_blocks
    gc.collect()

    mean_probs = all_probs.mean(axis=1)
    return mean_probs, all_probs


def predicted_market_shares(
    theta_p: float,
    gamma_A: float,
    gamma_R: float,
    delta: np.ndarray,
    v_draws: np.ndarray,
    data: dict,
    jt_code: np.ndarray,
    mask_in: np.ndarray,
    outside_id: str | None = None,
    block_size: int = 10,
    n_jobs: int = 5,
) -> np.ndarray:
    """
    Compute predicted market shares for retailer-time cells.

    Returns
    -------
    np.ndarray
        Predicted shares sorted by (time, ran_rep_id), optionally excluding outside_id.
    """
    estimation_df = data["estimation_df"]

    mean_probs, _ = simulated_choice_probabilities(
        theta_p=theta_p,
        gamma_A=gamma_A,
        gamma_R=gamma_R,
        delta=delta,
        v_draws=v_draws,
        data=data,
        jt_code=jt_code,
        mask_in=mask_in,
        block_size=block_size,
        n_jobs=n_jobs,
    )

    shares_df = estimation_df[["time", "ran_rep_id", "hh_idx"]].copy()
    shares_df["simulated_prob"] = mean_probs

    n_households_by_time = (
        shares_df.groupby("time")["hh_idx"]
        .nunique()
        .rename("N_t")
        .reset_index()
    )

    numerator = (
        shares_df.groupby(["time", "ran_rep_id"])["simulated_prob"]
        .sum()
        .rename("sum_prob")
        .reset_index()
    )

    market_share = (
        numerator.merge(n_households_by_time, on="time", how="left")
        .assign(share=lambda d: d["sum_prob"] / d["N_t"])
        .sort_values(["time", "ran_rep_id"])
    )

    if outside_id is not None:
        market_share = market_share.loc[market_share["ran_rep_id"] != outside_id]

    return market_share["share"].to_numpy()


# ---------------------------------------------------------------------
# Contraction mapping with SQUAREM acceleration
# ---------------------------------------------------------------------
def squarem_contraction_mapping(
    delta_init: np.ndarray,
    theta_p: float,
    gamma_A: float,
    gamma_R: float,
    v_draws: np.ndarray,
    data: dict,
    jt_code: np.ndarray,
    mask_in: np.ndarray,
    s_obs: np.ndarray,
    outside_id: str | None = None,
    cm_max_iter: int = 5000,
    cm_tol: float = 1e-7,
    step_min: float = 1.0,
    step_max: float = 1.0,
    step_factor: float = 4.0,
    #damping_factor: float = 1.0,
    block_size: int = 10,
    n_jobs: int = 5,
    verbose: bool = False,
) -> np.ndarray:
    """
    Recover delta via contraction mapping accelerated by SQUAREM.

    Parameters
    ----------
    delta_init : np.ndarray
        Initial guess for delta.
    s_obs : np.ndarray
        Observed shares corresponding to included retailer-time cells.

    Returns
    -------
    np.ndarray
        Estimated delta vector.
    """
    delta = delta_init.copy()
    prev_max_diff = np.inf

    for it in range(cm_max_iter):
        delta0 = delta.copy()

        # First fixed-point step
        s_pred_0 = np.clip(
            predicted_market_shares(
                theta_p=theta_p,
                gamma_A=gamma_A,
                gamma_R=gamma_R,
                delta=delta0,
                v_draws=v_draws,
                data=data,
                jt_code=jt_code,
                mask_in=mask_in,
                outside_id=outside_id,
                block_size=block_size,
                n_jobs=n_jobs,
            ),
            1e-14,
            1.0,
        )
        delta1 = delta0 + (np.log(s_obs) - np.log(s_pred_0))
        r = delta1 - delta0

        # Second fixed-point step
        s_pred_1 = np.clip(
            predicted_market_shares(
                theta_p=theta_p,
                gamma_A=gamma_A,
                gamma_R=gamma_R,
                delta=delta1,
                v_draws=v_draws,
                data=data,
                jt_code=jt_code,
                mask_in=mask_in,
                outside_id=outside_id,
                block_size=block_size,
                n_jobs=n_jobs,
            ),
            1e-14,
            1.0,
        )
        delta2 = delta1 + (np.log(s_obs) - np.log(s_pred_1))
        v = delta2 - delta1 - r

        rr = np.dot(r, r)
        vv = np.dot(v, v)

        if vv <= 1e-16:
            alpha = -1.0
        else:
            alpha = -np.sqrt(rr / vv)

        alpha = -np.maximum(step_min, np.minimum(step_max, -alpha))

        if -alpha == step_max:
            step_max *= step_factor

        # if prev_max_diff < 1e-7:
        #     alpha *= damping_factor

        delta = delta0 - 2.0 * alpha * r + (alpha**2) * v

        s_pred_final = np.clip(
            predicted_market_shares(
                theta_p=theta_p,
                gamma_A=gamma_A,
                gamma_R=gamma_R,
                delta=delta,
                v_draws=v_draws,
                data=data,
                jt_code=jt_code,
                mask_in=mask_in,
                outside_id=outside_id,
                block_size=block_size,
                n_jobs=n_jobs,
            ),
            1e-14,
            1.0,
        )

        max_diff = np.max(np.abs(np.log(s_obs) - np.log(s_pred_final)))

        if verbose:
            print(
                f"[SQUAREM iter {it:4d}] "
                f"max log-share error = {max_diff:.3e}, step = {alpha:.3f}"
            )

        if max_diff < cm_tol:
            return delta

        prev_max_diff = max_diff

    return delta


# ---------------------------------------------------------------------
# Likelihood
# ---------------------------------------------------------------------
def negative_log_likelihood(
    params: np.ndarray,
    data: dict,
    jt_code: np.ndarray,
    mask_in: np.ndarray,
    s_obs: np.ndarray,
    delta_state: dict,
    v_base_draws: np.ndarray,
    outside_id: str | None = None,
    cm_max_iter: int = 5000,
    cm_tol: float = 1e-7,
    block_size: int = 10,
    n_jobs: int = 5,
    verbose_cm: bool = True,
) -> float:
    """
    Compute the negative average log-likelihood per observed choice occasion.

    Parameters
    ----------
    params : array-like
        Parameter vector (theta_p, sigma, gamma_A, gamma_R).
    data : dict
        Output from build_estimation_data().
    jt_code : np.ndarray
        Integer code mapping included rows to delta elements.
    mask_in : np.ndarray
        Boolean mask for rows whose retailer-time cells are included in delta.
    s_obs : np.ndarray
        Observed market shares corresponding to delta.
    delta_state : dict
        Mutable dictionary storing the latest delta guess for warm starts.
        Expected key: "value".
    v_base_draws : np.ndarray
        Standard normal simulation draws with shape (n_households, n_draws).

    Returns
    -------
    float
        Negative average log-likelihood.
    """
    theta_p, sigma, gamma_A, gamma_R = params

    estimation_df = data["estimation_df"]
    choice = estimation_df["choice"].to_numpy(dtype=bool)
    hh_idx = estimation_df["hh_idx"].to_numpy(dtype=np.int64)
    # Random coefficients on AREP preference
    v_draws = sigma * v_base_draws

    delta_hat = squarem_contraction_mapping(
        delta_init=delta_state["value"],
        theta_p=theta_p,
        gamma_A=gamma_A,
        gamma_R=gamma_R,
        v_draws=v_draws,
        data=data,
        jt_code=jt_code,
        mask_in=mask_in,
        s_obs=s_obs,
        outside_id=outside_id,
        cm_max_iter=cm_max_iter,
        cm_tol=cm_tol,
        block_size=block_size,
        n_jobs=n_jobs,
        verbose=verbose_cm,
    )

    mean_probs, all_probs = simulated_choice_probabilities(
        theta_p=theta_p,
        gamma_A=gamma_A,
        gamma_R=gamma_R,
        delta=delta_hat,
        v_draws=v_draws,
        data=data,
        jt_code=jt_code,
        mask_in=mask_in,
        block_size=block_size,
        n_jobs=n_jobs,
    )
    # Keep only chosen alternatives:
    # one row per observed choice occasion, one column per simulation draw
    p_chosen = np.clip(all_probs[choice, :], 1e-300, 1.0)
    hh_chosen = hh_idx[choice]

    # Log probability of each chosen alternative, by draw
    logp = np.log(p_chosen)

    # Aggregate over all choice occasions for each household, separately by draw
    n_households = data["n_households"]
    n_draws = all_probs.shape[1]

    loglike_i_r = np.zeros((n_households, n_draws), dtype=np.float64)
    np.add.at(loglike_i_r, hh_chosen, logp)

    # Integrate over simulation draws:
    # log L_i = log( (1/R) * sum_r exp(loglike_i_r) )
    logLi = logsumexp(loglike_i_r, axis=1) - np.log(n_draws)

    # Sum over households
    loglik = logLi.sum()

    # Average by number of observed choice occasions
    n_choice_occasions = data["n_groups"]
    avg_loglik = loglik / n_choice_occasions

    # Warm start for the next optimizer step
    delta_state["value"] = delta_hat

    return -avg_loglik
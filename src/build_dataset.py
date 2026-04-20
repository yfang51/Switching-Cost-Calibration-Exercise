import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix


def build_estimation_data(
    panel_df: pd.DataFrame,
    retailer_time_df: pd.DataFrame,
    retailers: list,
    arep_id: str,
    outside_id: str | None = None,
) -> dict:
    """
    Build the expanded choice-set dataset and indexing objects used in estimation.

    Parameters
    ----------
    panel_df : pd.DataFrame
        Household-period panel produced by the simulation step. Expected columns:
        - hh_id
        - time
        - prev_retailer
        - stay_price_at_t
        - chosen_retailer
    retailer_time_df : pd.DataFrame
        Retailer-time data with one row per (time, retailer). Expected columns:
        - time
        - ran_rep_id
        - delta
        - price_offer
    retailers : list[str]
        List of retailer IDs included in the choice set.
    arep_id : str
        Retailer ID corresponding to the incumbent AREP.
    outside_id : str or None, optional
        Retailer ID corresponding to the outside option. Included for metadata
        consistency and future extensions. Not directly used in this function.

    Returns
    -------
    dict
        Dictionary containing:
        - "estimation_df": expanded long-format estimation dataset
        - "G": sparse group-by-row incidence matrix
        - "group_idx": integer group index for each expanded row
        - "n_groups": number of household-time choice situations
        - "households": array of unique household IDs
        - "household_to_index": mapping from household ID to integer index
        - "n_households": number of unique households
        - "n_rows": number of rows in the expanded dataset
        - "n_alternatives": number of retailers in the choice set

    Notes
    -----
    Each original household-time observation is expanded to all retailers in the
    choice set. The resulting dataset has one row per (household, time, retailer).

    The sparse matrix G maps each expanded row to its household-time choice group.
    This is useful for fast group-level operations such as logit denominators.
    """

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    required_panel_cols = {
        "hh_id",
        "time",
        "prev_retailer",
        "stay_price_at_t",
        "chosen_retailer",
    }
    required_retailer_time_cols = {
        "time",
        "ran_rep_id",
        "delta",
        "price_offer",
    }

    missing_panel_cols = required_panel_cols - set(panel_df.columns)
    if missing_panel_cols:
        raise ValueError(f"panel_df is missing required columns: {sorted(missing_panel_cols)}")

    missing_rt_cols = required_retailer_time_cols - set(retailer_time_df.columns)
    if missing_rt_cols:
        raise ValueError(
            f"retailer_time_df is missing required columns: {sorted(missing_rt_cols)}"
        )

    if arep_id not in retailers:
        raise ValueError("arep_id must be included in retailers.")

    if outside_id is not None and outside_id not in retailers:
        raise ValueError("outside_id must be included in retailers if provided.")

    n_alternatives = len(retailers)
    if n_alternatives < 1:
        raise ValueError("retailers must contain at least one alternative.")

       # ------------------------------------------------------------------
    # Step 1: Expand each household-time observation to the full choice set
    # ------------------------------------------------------------------
    estimation_df = panel_df.loc[panel_df.index.repeat(n_alternatives)].copy()
    estimation_df["ran_rep_id"] = np.tile(retailers, len(panel_df))

    estimation_df = estimation_df.merge(
        retailer_time_df,
        on=["time", "ran_rep_id"],
        how="left",
        validate="many_to_one",
    )

    if estimation_df[["delta", "price_offer"]].isna().any().any():
        raise ValueError(
            "Merge with retailer_time_df produced missing values. "
            "Check that retailer_time_df contains one row for every (time, ran_rep_id)."
        )

    # ------------------------------------------------------------------
    # Step 2: Construct estimation variables
    # ------------------------------------------------------------------
    estimation_df["AREP"] = (estimation_df["ran_rep_id"] == arep_id).astype(np.int8)

    estimation_df["sc_AREP"] = (
        (estimation_df["prev_retailer"] == arep_id)
        & (estimation_df["ran_rep_id"] != arep_id)
    ).astype(np.int8)

    estimation_df["sc_REP"] = (
        estimation_df["prev_retailer"].notna()
        & (estimation_df["prev_retailer"] != arep_id)
        & (estimation_df["ran_rep_id"] != estimation_df["prev_retailer"])
    ).astype(np.int8)

    # Real price faced by the household for this alternative:
    # - for the previous retailer, use the stay/renewal price at time t
    # - otherwise, use the retailer's current offer price
    estimation_df["real_price"] = np.where(
        estimation_df["prev_retailer"].notna()
        & (estimation_df["ran_rep_id"] == estimation_df["prev_retailer"]),
        estimation_df["stay_price_at_t"],
        estimation_df["price_offer"],
    )

    estimation_df["choice"] = (
        estimation_df["chosen_retailer"] == estimation_df["ran_rep_id"]
    ).astype(np.int8)

    # ------------------------------------------------------------------
    # Step 3: Construct household and group indices
    # ------------------------------------------------------------------
    households = estimation_df["hh_id"].unique()
    household_to_index = {hh_id: i for i, hh_id in enumerate(households)}
    estimation_df["hh_idx"] = estimation_df["hh_id"].map(household_to_index).astype(np.intp)

    # Each group is a unique (household, time) choice occasion
    group_keys = pd.MultiIndex.from_frame(estimation_df[["hh_id", "time"]])
    group_idx, _ = pd.factorize(group_keys, sort=False)
    group_idx = group_idx.astype(np.intp, copy=False)
    n_groups = int(group_idx.max()) + 1

    n_rows = len(estimation_df)
    n_households = len(households)

    # ------------------------------------------------------------------
    # Step 4: Sparse group-by-row incidence matrix
    # ------------------------------------------------------------------
    # G has shape (n_groups, n_rows)
    # Column c corresponds to expanded row c.
    # Row g marks which expanded rows belong to group g.
    rows = group_idx
    cols = np.arange(n_rows, dtype=np.int32)
    data = np.ones(n_rows, dtype=np.int8)

    G = coo_matrix((data, (rows, cols)), shape=(n_groups, n_rows)).tocsr()

    # ------------------------------------------------------------------
    # Step 5: Sanity checks
    # ------------------------------------------------------------------
    group_sizes = np.asarray(G.sum(axis=1)).ravel()
    if not np.all(group_sizes == n_alternatives):
        raise ValueError(
            "Each household-time group should contain exactly one row per retailer. "
            "At least one group has the wrong number of alternatives."
        )

    choice_counts = estimation_df.groupby(["hh_id", "time"])["choice"].sum().to_numpy()
    if not np.all(choice_counts == 1):
        raise ValueError(
            "Each household-time group should have exactly one chosen alternative."
        )

    return {
        "estimation_df": estimation_df,
        "G": G,
        "group_idx": group_idx,
        "n_groups": n_groups,
        "households": households,
        "household_to_index": household_to_index,
        "n_households": n_households,
        "n_rows": n_rows,
        "n_alternatives": n_alternatives,
        "meta": {
            "arep_id": arep_id,
            "outside_id": outside_id,
        },
    }


# Example usage
# sim = simulate_data(seed=42)
# est_data = build_estimation_data(
#     panel_df=sim["panel_df"],
#     retailer_time_df=sim["retailer_time_df"],
#     retailers=sim["retailers"],
#     arep_id=sim["meta"]["arep_id"],
#     outside_id=sim["meta"]["outside_id"],
# )
# estimation_df = est_data["estimation_df"]
# print(estimation_df.head())
"""Sample reproducible profiles from distinct pre-holdout category buyers."""

import numpy as np
import pandas as pd

from data.etl.config import CHANNELS, INCOME_ELASTICITY_FACTORS, SEED, income_group
from data.loaders import load_twin_responses

BEHAVIOR = [
    "weekly_buy_rate",
    "purchase_cycle_weeks",
    "avg_units",
    "avg_price_paid",
    "price_elasticity_observed",
    "promo_share",
    "coupon_rate",
    "brand_loyalty",
]
TRAITS = ("impulsiveness", "risk_aversion", "time_preference")
DEFAULT_TRAITS = {
    "deal_hunter": (0.45, 0.55, 0.30),
    "frequent_replenisher": (0.32, 0.55, 0.48),
    "high_spender": (0.50, 0.40, 0.60),
    "occasional": (0.28, 0.68, 0.35),
    "steady_shopper": (0.35, 0.52, 0.45),
}
BACKSTORIES = {
    "deal_hunter": "Checks offers before stocking up on toilet tissue.",
    "frequent_replenisher": "Restocks toilet tissue regularly as part of a shopping routine.",
    "high_spender": "Spends more on toilet tissue across their household shopping trips.",
    "occasional": "Buys toilet tissue infrequently and keeps an eye on the supply at home.",
    "steady_shopper": "Replenishes toilet tissue when it is needed during a regular shop.",
}
PROFILE_COUNT = 150
EDGE_COUNT = 15
MIN_PER_ARCHETYPE = 20
SURVEY_DRAW_SIZE = 12


def _survey_traits() -> pd.DataFrame | None:
    # The public numeric export uses 1=left and 2=right for both choice matrices.
    time_columns = [f"QID84_{row}" for row in range(4, 13)]
    risk_columns = [f"QID250_{row}" for row in range(1, 15)]
    survey = load_twin_responses(["QID32", *time_columns, *risk_columns])
    if survey is None:
        return None

    complete = survey.dropna()
    valid = (
        complete["QID32"].isin([1, 2, 3, 4, 5])
        & complete[time_columns].isin([1, 2]).all(axis=1)
        & complete[risk_columns].isin([1, 2]).all(axis=1)
    )
    complete = complete.loc[valid]
    if len(complete) < SURVEY_DRAW_SIZE:
        raise ValueError("Twin-2K-500 CSV has too few valid QID32, QID84 and QID250 responses")
    return pd.DataFrame(
        {
            "impulsiveness": (complete["QID32"] - 1) / 4,
            "risk_aversion": complete[risk_columns].eq(2).mean(axis=1),
            "time_preference": complete[time_columns].eq(2).mean(axis=1),
        }
    )


def _allocate(total, target, minimum, maximum):
    """Integer quotas nearest the proportional targets, subject to capacity and floors."""
    quota = pd.Series(minimum, index=target.index, dtype=int)
    if (quota > maximum).any() or not quota.sum() <= total <= maximum.sum():
        raise ValueError("Insufficient households or slots to meet the sampling quotas")
    for _ in range(int(total - quota.sum())):
        deficit = (target - quota).where(quota < maximum)
        quota.loc[deficit.idxmax()] += 1
    return quota


def sample_households(households: pd.DataFrame) -> pd.DataFrame:
    """Local-only stratified sample; the index identifies donors and is never published."""
    if not households.index.is_unique:
        raise ValueError("A household can occur only once in the sampling frame")
    # Stable input order makes seeded draws and distance ties independent of Parquet row order.
    pool = households.sort_index().copy()
    archetype = pool["archetype"]
    sizes = archetype.value_counts().sort_index()
    quotas = _allocate(PROFILE_COUNT, PROFILE_COUNT * sizes / len(pool), MIN_PER_ARCHETYPE, sizes)

    known = pool[["age", "income"]].notna().all(axis=1)
    cells = pool.loc[known].groupby(["age", "income"], observed=True).size()
    common_cells = cells.index[cells >= 0.02 * known.sum()]
    # Poststratify within archetype. Rare observed cells share one stratum.
    cell = pd.Series("missing", index=pool.index)
    cell.loc[known] = "other known"
    for age, income in common_cells:
        cell.loc[pool["age"].eq(age) & pool["income"].eq(income)] = f"{age} | {income}"
    strata = pd.DataFrame({"archetype": archetype, "cell": cell}).groupby(["archetype", "cell"])
    pool["_weight_stratum"] = strata.ngroup()
    distances = pool.groupby("archetype")["distance_to_center"].rank(
        ascending=False, method="first"
    )
    pool["_edge"] = distances <= np.ceil(archetype.map(sizes) * 0.10)

    groups = pool.groupby("_weight_stratum")
    counts = groups.size()
    stratum_quota = pd.Series(0, index=counts.index)
    stratum_archetype = groups["archetype"].first()
    for name, ids in stratum_archetype.groupby(stratum_archetype).groups.items():
        population = counts.loc[ids]
        slots = quotas[name]
        stratum_quota.loc[ids] = _allocate(slots, slots * population / sizes[name], 1, population)
    # Constrain the edge count across strata. Small cells may contain only tails.
    edges = groups["_edge"].sum()
    edge_min = (stratum_quota - (counts - edges)).clip(lower=0)
    edge_max = np.minimum(stratum_quota, edges)
    edge_target = stratum_quota * edges / counts
    edge_quota = _allocate(EDGE_COUNT, edge_target, edge_min, edge_max)

    rng = np.random.default_rng(SEED)
    samples = []
    for (group, edge), frame in pool.groupby(["_weight_stratum", "_edge"]):
        slots = edge_quota[group] if edge else stratum_quota[group] - edge_quota[group]
        if slots:
            chosen = frame.iloc[rng.choice(len(frame), size=slots, replace=False)].copy()
            # Archetype share / archetype quota, corrected to each demographic stratum's share.
            chosen["weight"] = counts[group] / (len(pool) * stratum_quota[group])
            samples.append(chosen)
    sample = pd.concat(samples)
    return sample.iloc[rng.permutation(len(sample))]


STRUCTURE = ["household_size", "kids_count", "home_ownership"]
DEMOGRAPHIC_FIELDS = {
    "age": "age_band",
    "income": "income_band",
    "household_size": "household_size",
    "kids_count": "kids",
    "home_ownership": "homeowner",
}


def _demographics(row, donors, rng):
    """Keep observed values and fill missing typed fields from a single compatible donor."""
    values = {field: getattr(row, field) for field in DEMOGRAPHIC_FIELDS}
    missing = [field for field, value in values.items() if pd.isna(value)]
    missing_structure = [field for field in STRUCTURE if field in missing]
    if missing_structure:
        candidates = donors
        for field in [*STRUCTURE, "age", "income"]:
            if field not in missing:
                matching = candidates[candidates[field].eq(values[field])]
                if not matching.empty:
                    candidates = matching
        if candidates.empty:
            raise ValueError("No complete household structure is available for imputation")
        donor = candidates.iloc[rng.integers(len(candidates))]
        for field in missing_structure:
            values[field] = donor[field]
    return {
        "age_band": "Unknown" if "age" in missing else values["age"],
        "income_band": "Unknown" if "income" in missing else values["income"],
        "household_size": int(str(values["household_size"]).removesuffix("+")),
        "kids": int(str(values["kids_count"]).removesuffix("+")),
        "homeowner": values["home_ownership"] in ("Homeowner", "Probable Homeowner"),
    }, {
        DEMOGRAPHIC_FIELDS[field]: "default" if field in missing else "data"
        for field in DEMOGRAPHIC_FIELDS
    }


def build_personas(households: pd.DataFrame, sample: pd.DataFrame | None = None) -> list[dict]:
    if sample is None:
        from data.etl.selection_bias import calibrate_selection

        sample, _ = calibrate_selection(households, sample_households(households))
    rng = np.random.default_rng(SEED)
    donors = households.sort_index().dropna(subset=STRUCTURE)
    survey = _survey_traits()
    surveyed = None if survey is None else survey[list(TRAITS)].to_numpy()
    profiles = []
    for slot, (_, row) in enumerate(sample.iterrows(), start=1):
        demographics, demographic_sources = _demographics(row, donors, rng)
        cluster = row["cluster_label"]
        trait_source = "default"
        trait_values = DEFAULT_TRAITS[cluster]
        if surveyed is not None:
            draws = rng.integers(len(surveyed), size=SURVEY_DRAW_SIZE)
            trait_values = tuple(round(float(value), 6) for value in surveyed[draws].mean(axis=0))
            trait_source = "data"
        prefix = "tt-edge" if row["_edge"] else "tt"
        factor = INCOME_ELASTICITY_FACTORS.get(income_group(demographics["income_band"]), 1.0)
        profiles.append(
            {
                "persona_id": f"{prefix}-{slot:03d}",
                "segment": str(row["segment"]),
                "archetype": str(row["archetype"]),
                "cluster_label": cluster,
                "weight": float(row["weight"]),
                **demographics,
                **{field: round(float(row[field]), 6) for field in BEHAVIOR},
                "price_elasticity": round(float(row["price_elasticity"]) * factor, 6),
                "traits": dict(zip(TRAITS, trait_values, strict=True)),
                "channels": list(CHANNELS),
                "backstory": BACKSTORIES[cluster],
                "source": {
                    "behavior": "data",
                    "price_elasticity": "default",
                    "price_elasticity_observed": "data",
                    "traits": trait_source,
                    "backstory": "llm_estimate",
                    **demographic_sources,
                },
            }
        )
    return profiles

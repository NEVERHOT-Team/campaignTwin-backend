"""Generate reproducible synthetic profiles from observed household strata."""

from collections import Counter

import numpy as np
import pandas as pd

from data.etl.config import CHANNELS, SEED
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
STRATUM = ["cluster_label", "age", "income"]
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
PROFILE_COUNT = 60
EDGE_COUNT = 6
MIN_PER_ARCHETYPE = 10
MIN_STRATUM_FOR_BEHAVIOR = 12
BOOTSTRAP_SIZE = 12


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
    if len(complete) < BOOTSTRAP_SIZE:
        raise ValueError("Twin-2K-500 CSV has too few valid QID32, QID84 and QID250 responses")
    return pd.DataFrame(
        {
            "impulsiveness": (complete["QID32"] - 1) / 4,
            "risk_aversion": complete[risk_columns].eq(2).mean(axis=1),
            "time_preference": complete[time_columns].eq(2).mean(axis=1),
        }
    )


def _draw_stratum(counts: pd.Series, rng: np.random.Generator) -> tuple[str, str, str]:
    position = rng.choice(len(counts), p=counts.to_numpy() / counts.sum())
    return counts.index[position]


def _sample_strata(counts: pd.Series, rng: np.random.Generator) -> list[tuple[str, str, str]]:
    # Ensure that common age × income pairs appear before filling archetype quotas.
    age_income = counts.groupby(level=["age", "income"]).sum()
    strata = [
        _draw_stratum(counts.xs(cell, level=["age", "income"], drop_level=False), rng)
        for cell in age_income.loc[age_income.ge(0.02 * age_income.sum())].index
    ]
    for cluster in counts.index.unique("cluster_label"):
        candidates = counts.xs(cluster, level="cluster_label", drop_level=False)
        reserved = sum(stratum[0] == cluster for stratum in strata)
        strata.extend(
            _draw_stratum(candidates, rng) for _ in range(max(0, MIN_PER_ARCHETYPE - reserved))
        )
    if len(strata) > PROFILE_COUNT:
        raise ValueError("The demographic coverage and archetype quotas exceed 60 personas")
    strata.extend(_draw_stratum(counts, rng) for _ in range(PROFILE_COUNT - len(strata)))
    return strata


def _sample_demographics(
    cohort: pd.DataFrame,
    demo: pd.DataFrame,
    age: str,
    income: str,
    rng: np.random.Generator,
) -> tuple[int, int, bool]:
    # Preserve the empirical household-size × kids distribution and sample ownership independently.
    by_cell = demo.loc[demo["age"].eq(age) & demo["income"].eq(income)]
    pool = by_cell if len(by_cell) >= MIN_STRATUM_FOR_BEHAVIOR else cohort
    if pool.empty:
        pool = demo
    structure = pool.groupby(["household_size", "kids_count"]).size()
    if structure.empty:
        structure = demo.groupby(["household_size", "kids_count"]).size()
    position = rng.choice(len(structure), p=structure.to_numpy() / structure.sum())
    size, kids = structure.index[position]
    owners = pool["home_ownership"].dropna()
    if owners.empty:
        owners = demo["home_ownership"].dropna()
    homeowner = rng.random() < owners.isin(["Homeowner", "Probable Homeowner"]).mean()
    return int(str(size).removesuffix("+")), int(str(kids).removesuffix("+")), bool(homeowner)


def _behavior_sample(
    pool: pd.DataFrame, used: set[tuple[float, ...]], rng: np.random.Generator
) -> dict[str, float]:
    values = pool[BEHAVIOR].to_numpy()
    for _ in range(100):
        indexes = rng.integers(len(values), size=BOOTSTRAP_SIZE)
        profile = tuple(round(float(value), 6) for value in values[indexes].mean(axis=0))
        if profile not in used:
            used.add(profile)
            return dict(zip(BEHAVIOR, profile, strict=True))
    raise ValueError("Could not sample a distinct aggregate behavior profile")


def build_personas(households: pd.DataFrame) -> list[dict]:
    rng = np.random.default_rng(SEED)
    demo = households.dropna(subset=["age", "income"])
    counts = demo.groupby(STRATUM).size()
    if set(counts.index.unique("cluster_label")) != set(households["cluster_label"]):
        raise ValueError("Every archetype needs observed age and income to build personas")

    strata = _sample_strata(counts, rng)
    slot_counts = Counter(strata)
    represented = counts.reindex(list(slot_counts)).groupby(level="cluster_label").sum()
    cluster_weights = households["cluster_label"].value_counts(normalize=True)
    edge_slots = set(rng.choice(PROFILE_COUNT, EDGE_COUNT, replace=False))
    survey = _survey_traits()
    surveyed = None if survey is None else survey[list(TRAITS)].to_numpy()

    cohorts = dict(tuple(households.groupby("cluster_label")))
    cohort_demos = dict(tuple(demo.groupby("cluster_label")))
    cells = dict(tuple(demo.groupby(STRATUM)))
    tails = {
        name: cohort.nlargest(max(BOOTSTRAP_SIZE, int(len(cohort) * 0.10)), "distance_to_center")
        for name, cohort in cohorts.items()
    }
    profiles = []
    used: set[tuple[float, ...]] = set()
    for slot, stratum in enumerate(strata):
        cluster, age, income = stratum
        cohort = cohorts[cluster]
        cohort_demo = cohort_demos[cluster]
        age_income_cell = cells[stratum]
        pool = age_income_cell if len(age_income_cell) >= MIN_STRATUM_FOR_BEHAVIOR else cohort
        edge = slot in edge_slots
        if edge:
            pool = tails[cluster]
        behavior = _behavior_sample(pool, used, rng)
        size, kids, homeowner = _sample_demographics(cohort_demo, demo, age, income, rng)

        trait_source = "default"
        trait_values = DEFAULT_TRAITS[cluster]
        if surveyed is not None:
            draws = rng.integers(len(surveyed), size=BOOTSTRAP_SIZE)
            trait_values = tuple(round(float(value), 6) for value in surveyed[draws].mean(axis=0))
            trait_source = "data"

        weight = cluster_weights[cluster] * counts[stratum]
        weight /= represented[cluster] * slot_counts[stratum]
        prefix = "tt-edge" if edge else "tt"
        profiles.append(
            {
                "persona_id": f"{prefix}-{slot + 1:02d}",
                "segment": str(cohort["segment"].iloc[0]),
                "archetype": str(cohort["archetype"].iloc[0]),
                "cluster_label": cluster,
                "weight": float(weight),
                "age_band": age,
                "income_band": income,
                "household_size": size,
                "kids": kids,
                "homeowner": homeowner,
                **behavior,
                "price_elasticity": float(cohort["price_elasticity"].iloc[0]),
                "traits": dict(zip(TRAITS, trait_values, strict=True)),
                "channels": list(CHANNELS),
                "backstory": BACKSTORIES[cluster],
                "source": {
                    "behavior": "data",
                    "price_elasticity": "default",
                    "price_elasticity_observed": "data",
                    "traits": trait_source,
                    "backstory": "llm_estimate",
                },
            }
        )
    return profiles

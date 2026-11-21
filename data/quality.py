"""Measure the published personas against the local Complete Journey calibration panel."""

import json
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from data import loaders
from data.etl.config import income_group
from data.etl.selection_bias import BEHAVIOUR_ATTRIBUTES, ks_similarity
from data.quality_history import HISTORICAL_KS, HISTORICAL_SOURCES
from engine import shopper

ENGINE_BEHAVIOUR = (
    "weekly_buy_rate",
    "purchase_cycle_weeks",
    "avg_units",
    "promo_share",
    "brand_loyalty",
)
PROFILE_ONLY = {
    "coupon_rate": "not used by engine",
    "avg_price_paid": "the engine uses a single market reference price",
}
TARGETS = {
    "ks_similarity_min": 0.85,
    "brand_loyalty_ks_min": 0.75,
    "sd_ratio_min": 0.8,
    "correlation_gap_max": 0.1,
}
AGENT_COUNT = 5_000
AGENT_SEED = 42
MISSING_PANEL = (
    "Household reference checks skipped: local data/calib/household_behavior.parquet is missing. "
    "Run python -m data.etl.build_tissue with local raw data to enable them."
)
MISSING_LOYALTY = (
    "This panel stores only the leading weekly brand. Rebuild with python -m data.etl.build_tissue "
    "for exact top-manufacturer unit share and the five-attribute engine correlation."
)


def household_attributes(panel: pd.DataFrame) -> pd.DataFrame:
    """Reduce the local weekly grid to one equally weighted observation per household."""
    if panel.empty or panel.duplicated(["household_key", "week"]).any():
        raise ValueError("Expected a nonempty panel with unique household-week observations")
    weeks = panel["week"].nunique()
    grouped = panel.groupby("household_key")
    if not grouped.size().eq(weeks).all():
        raise ValueError("The calibration panel has missing household weeks")
    purchases = panel.loc[panel["bought"]].sort_values(["household_key", "week"]).copy()
    if purchases[["price_paid", "units", "on_promo", "coupon_used"]].isna().any().any():
        raise ValueError("Missing price, units or promotion flags in a purchase week")
    if not purchases["units"].gt(0).all():
        raise ValueError("Purchase weeks require positive units")
    purchases["gap"] = purchases.groupby("household_key")["week"].diff()
    purchases["paid_total"] = purchases["price_paid"] * purchases["units"]
    by_buyer = purchases.groupby("household_key")
    count = by_buyer.size()
    if len(count) != grouped.ngroups:
        raise ValueError("Expected at least one category purchase per calibration household")
    result = pd.DataFrame(
        {
            "weekly_buy_rate": count / weeks,
            "purchase_cycle_weeks": by_buyer["gap"].median().fillna(weeks / count),
            "avg_units": by_buyer["units"].mean(),
            "avg_price_paid": by_buyer["paid_total"].sum() / by_buyer["units"].sum(),
            "promo_share": by_buyer["on_promo"].mean(),
            "coupon_rate": by_buyer["coupon_used"].mean(),
        }
    )
    if "brand_loyalty" in panel:
        if not grouped["brand_loyalty"].nunique(dropna=False).eq(1).all():
            raise ValueError("Household brand_loyalty must be constant across the panel weeks")
        result["brand_loyalty"] = grouped["brand_loyalty"].first()
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError("Calibration household attributes must be finite")
    return result


def _engine_agents(personas: pd.DataFrame) -> pd.DataFrame:
    population = shopper.build_population(n_agents=AGENT_COUNT, seed=AGENT_SEED)
    agents = pd.DataFrame(
        {
            "weekly_buy_rate": population.weekly_buy_rate,
            "purchase_cycle_weeks": population.cycle_weeks,
            "avg_units": population.avg_units,
            "promo_share": population.promo_share,
            "brand_loyalty": population.loyalty,
            "price_elasticity": population.elasticity,
        }
    )
    if len(agents) != AGENT_COUNT or not np.isfinite(agents.to_numpy()).all():
        raise ValueError("Expected a 5,000-agent engine population with finite attributes")
    persona_ids = np.asarray(population.persona_ids)[population.persona_idx]
    profiles = personas.set_index("persona_id")
    if not profiles.index.is_unique:
        raise ValueError("Engine population evaluation requires unique persona IDs")
    groups = profiles.loc[persona_ids, ["archetype", "income_band"]]
    return agents.join(groups.reset_index(drop=True))


def _mean_abs_correlation(frame, weights=None):
    values = frame.to_numpy(dtype=float)
    if len(values) < 2 or (np.ptp(values, axis=0) == 0).any():
        return None
    cov = np.cov(values, rowvar=False, aweights=weights, ddof=0)
    sd = np.sqrt(np.diag(cov))
    corr = cov / np.outer(sd, sd)
    return float(np.abs(corr[np.triu_indices(len(sd), k=1)]).mean())


def _status(value, threshold, *, maximum=False):
    if value is None:
        return "skipped"
    # Ignore floating-point noise exactly at an inclusive threshold.
    value = round(value, 12)
    passes = value <= threshold if maximum else value >= threshold
    return "pass" if passes else "fail"


def _reference_checks(personas, households, agents):
    weights = personas["weight"].to_numpy()
    unit_weights = np.ones(len(agents))
    marginal = {}
    for field in BEHAVIOUR_ATTRIBUTES:
        profile_score = engine_score = target = None
        if field in ENGINE_BEHAVIOUR:
            key = "brand_loyalty_ks_min" if field == "brand_loyalty" else "ks_similarity_min"
            target = TARGETS[key]
        if households is not None and field in households:
            reference = households[field].to_numpy()
            profile_score = ks_similarity(personas[field].to_numpy(), weights, reference)
            if target is not None:
                engine_score = ks_similarity(agents[field].to_numpy(), unit_weights, reference)
        row = {
            **HISTORICAL_KS[field],
            "profile_value": profile_score,
            "value": engine_score,
            "target": target,
        }
        if field in PROFILE_ONLY:
            row.update(status="profile_only", note=PROFILE_ONLY[field])
        else:
            row["status"] = _status(engine_score, target)
        marginal[field] = row

    # KS is symmetric. Its weighted argument is the configured persona prior.
    elasticity = agents["price_elasticity"].to_numpy()
    marginal["price_elasticity"] = {
        "averaged_60": None,
        "own_jitter_150": None,
        "profile_value": None,
        "value": ks_similarity(personas["price_elasticity"].to_numpy(), weights, elasticity),
        "target": None,
        "status": "diagnostic",
        "reference": "weighted_persona_prior",
        "engine_mean": float(elasticity.mean()),
        "engine_median": float(np.median(elasticity)),
        "note": "KS against the configured prior. Household-level slopes are unavailable.",
    }

    rates = personas["weekly_buy_rate"].to_numpy()
    center = np.average(rates, weights=weights)
    persona_sd = float(np.sqrt(np.average((rates - center) ** 2, weights=weights)))
    engine_sd = float(agents["weekly_buy_rate"].std(ddof=0))
    real_sd = None if households is None else float(households["weekly_buy_rate"].std(ddof=0))
    ratio = None if real_sd is None or real_sd == 0 else engine_sd / real_sd
    variance = {
        "persona_sd": persona_sd,
        "engine_sd": engine_sd,
        "household_sd": real_sd,
        "value": ratio,
        "status": _status(ratio, TARGETS["sd_ratio_min"]),
    }
    if real_sd == 0:
        variance["note"] = "Household weekly buy rates have zero standard deviation."

    fields = list(ENGINE_BEHAVIOUR)
    engine_corr = _mean_abs_correlation(agents[fields])
    real_corr = None
    if households is not None and set(fields).issubset(households):
        real_corr = _mean_abs_correlation(households[fields])
    gap = None if engine_corr is None or real_corr is None else abs(engine_corr - real_corr)
    correlation = {
        "attributes": fields,
        "persona_mean_abs_corr": _mean_abs_correlation(personas[fields], weights),
        "engine_mean_abs_corr": engine_corr,
        "household_mean_abs_corr": real_corr,
        "gap": gap,
        "status": _status(gap, TARGETS["correlation_gap_max"], maximum=True),
    }
    if gap is None:
        correlation["note"] = "All five attributes must be present and nonconstant in both groups."
    return marginal, variance, correlation


def _direction(frame, field, groups, left, right, *, weighted=False):
    def mean(group):
        rows = frame.loc[groups.eq(group)]
        if rows.empty:
            return None
        return float(np.average(rows[field], weights=rows["weight"] if weighted else None))

    a, b = mean(left), mean(right)
    if a is None or b is None:
        status = "skipped"
    else:
        status = "pass" if round(a, 12) < round(b, 12) else "fail"
    return {
        "field": field,
        "source": "data" if field == "price_elasticity_observed" else "default",
        "population": "weighted_profiles" if weighted else "engine_agents",
        left: a,
        right: b,
        "status": status,
    }


def _direction_checks(personas, agents):
    return {
        "deal_hunter_vs_loyal": _direction(
            agents, "price_elasticity", agents["archetype"], "deal_hunter", "loyal"
        ),
        "income_observed": _direction(
            personas,
            "price_elasticity_observed",
            personas["income_band"].map(income_group),
            "low",
            "high",
            weighted=True,
        ),
        "income_prior": _direction(
            agents, "price_elasticity", agents["income_band"].map(income_group), "low", "high"
        ),
    }


def build_eval_report() -> dict:
    personas = pd.DataFrame(loaders.load_personas())
    inputs = [*BEHAVIOUR_ATTRIBUTES, "weight", "price_elasticity", "price_elasticity_observed"]
    if personas.empty or not np.isfinite(personas[inputs].to_numpy(dtype=float)).all():
        raise ValueError("Quality evaluation requires finite persona attributes")
    if not personas["weight"].gt(0).all():
        raise ValueError("Quality evaluation requires finite persona features and positive weights")
    try:
        panel = loaders.load_household_behavior()
    except FileNotFoundError:
        households = None
        reference = {"status": "skipped", "households": None, "weeks": None, "note": MISSING_PANEL}
    else:
        households = household_attributes(panel)
        reference = {
            "status": "available",
            "households": len(households),
            "weeks": panel.week.nunique(),
        }
        if "brand_loyalty" not in households:
            reference.update(status="partial", note=MISSING_LOYALTY)
    agents = _engine_agents(personas)
    marginal, variance, correlation = _reference_checks(personas, households, agents)
    directions = _direction_checks(personas, agents)
    try:
        selection_bias = loaders.load_selection_bias()
    except FileNotFoundError:
        selection_bias = {
            "status": "skipped",
            "note": (
                "Local selection-bias audit is missing. Run python -m data.etl.build_tissue to "
                "compare demographic coverage and record the sampling factors."
            ),
        }

    library = loaders.load_library()
    sourced = library["source_url"].fillna("").str.strip().ne("")
    known = library["reaction_type"].isin(["none", "match", "undercut", "non_price"])
    evidence = {
        "total_rows": len(library),
        "with_source_url": int(sourced.sum()),
        "with_known_reaction_type": int(known.sum()),
        "with_both": int((sourced & known).sum()),
    }
    labels = Counter(row["label"] for row in loaders.load_assumptions())
    total = labels.total()
    counts = {label: labels[label] for label in ("data", "default", "llm_estimate")}
    shares = {label: 100 * count / total for label, count in counts.items()}
    results = [
        *(marginal[field] for field in ENGINE_BEHAVIOUR),
        variance,
        correlation,
        *directions.values(),
    ]
    statuses = Counter(row["status"] for row in results)
    return {
        "status": "fail" if statuses["fail"] else "skipped" if statuses["skipped"] else "pass",
        "check_counts": {name: statuses[name] for name in ("pass", "fail", "skipped")},
        "source": "data",
        "reference": reference,
        "persona_count": len(personas),
        "ks_population": {
            "agents": AGENT_COUNT,
            "seed": AGENT_SEED,
            "builder": "engine.shopper.build_population",
            "source": "default",
        },
        "historical_comparisons": {
            "commits": HISTORICAL_SOURCES.copy(),
            "method": "Archived 5,000-agent KS with seed 42 and independent uniform ±10% jitter.",
        },
        "targets": TARGETS.copy(),
        "targets_source": "default",
        "method": {
            "reference": "Pre-holdout panel from the packaged simulated Complete Journey data.",
            "weights": "Population weights for personas and equal weights for households.",
            "selection_bias": (
                "The tissue ETL records the pre-holdout comparison and fitted weighting factors "
                "in a local aggregate audit when it builds the personas."
            ),
            "marginal_fit": (
                "One minus KS at six decimal places. Five engine arrays are compared with the "
                "household reference. profile_value uses weighted profiles. Coupon rate and "
                "average price paid are profile diagnostics. Elasticity is compared with the "
                "configured prior. Only rows with numeric targets count towards empirical fit."
            ),
            "variance_ratio": "Population SD of engine buy rate over household population SD.",
            "over_correlation": "Gap in mean absolute Pearson r over ten shared behaviour pairs.",
            "direction_checks": (
                "Prior directions use the engine agents' elasticities grouped by their sampled "
                "personas. Observed slopes use weighted profiles and were fitted by archetype. "
                "Annual income groups are below USD 35k and at least USD 75k."
            ),
            "confidence_inputs": (
                "Coverage is the percentage of all assumptions labelled data. Rival evidence "
                "counts library rows with both a nonblank source URL and a known reaction type."
            ),
        },
        "selection_bias": selection_bias,
        "marginal_fit": marginal,
        "variance_ratio": variance,
        "over_correlation": correlation,
        "direction_checks": directions,
        "rival_evidence": evidence,
        "data_coverage": {"total": total, "counts": counts, "shares_pct": shares},
        "confidence_inputs": {
            "coverage_pct": shares["data"],
            "rival_evidence_n": evidence["with_both"],
            "backtest_label": None,
        },
    }


def _display_score(value: float | None) -> str:
    return "—" if value is None else f"{value:.4f}"


def main() -> None:
    report = build_eval_report()
    destination = loaders.PROCESSED / "eval_report.json"
    with TemporaryDirectory(dir=destination.parent) as staging:
        temp = Path(staging) / destination.name
        temp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temp.replace(destination)
    columns = [
        "KS attribute",
        "60 averaged",
        "150 own-jitter",
        f"{report['persona_count']} engine population",
        "target",
    ]
    ks = []
    for field, score in report["marginal_fit"].items():
        current = _display_score(score["value"])
        if field == "price_elasticity":
            current += " (vs input prior)"
        ks.append(
            [
                field,
                _display_score(score["averaged_60"]),
                _display_score(score["own_jitter_150"]),
                current,
                _display_score(score["target"]),
            ]
        )
    print(pd.DataFrame(ks, columns=columns).to_string(index=False))
    for field, note in PROFILE_ONLY.items():
        value = report["marginal_fit"][field]["profile_value"]
        print(f"{field} profile KS {_display_score(value)} ({note}).")
    rows = []
    variance = report["variance_ratio"]
    corr = report["over_correlation"]
    rows.extend(
        [
            ("Buy-rate SD ratio", variance["value"], variance["status"]),
            ("Mean |r| gap", corr["gap"], corr["status"]),
        ]
    )
    for name, check in report["direction_checks"].items():
        groups = ("deal_hunter", "loyal") if name == "deal_hunter_vs_loyal" else ("low", "high")
        values = [check[group] for group in groups]
        pair = " vs ".join("—" if v is None else f"{v:.4f}" for v in values)
        rows.append((name, pair, check["status"]))
    evidence = report["rival_evidence"]
    for name in ("with_source_url", "with_known_reaction_type", "with_both"):
        rows.append((f"Rival {name}", f"{evidence[name]}/{evidence['total_rows']}", "count"))
    for label, share in report["data_coverage"]["shares_pct"].items():
        rows.append((f"Assumptions {label}", f"{share:.2f}%", "share"))
    rows.append(("Backtest label", "pending engine", "pending"))
    print(pd.DataFrame(rows, columns=["Measure", "Result", "Status"]).to_string(index=False))
    if "note" in report["reference"]:
        print(report["reference"]["note"])


if __name__ == "__main__":
    main()

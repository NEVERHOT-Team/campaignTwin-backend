"""Measure the published personas against the local Complete Journey calibration panel."""

import json
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from data import loaders

ATTRIBUTES = (
    "weekly_buy_rate",
    "purchase_cycle_weeks",
    "avg_price_paid",
    "promo_share",
    "coupon_rate",
    "brand_loyalty",
)
TARGETS = {"ks_similarity_min": 0.9, "sd_ratio_min": 0.8, "correlation_gap_max": 0.1}
MISSING_PANEL = (
    "Household reference checks skipped: local data/calib/household_behavior.parquet is missing. "
    "Run python -m data.etl.build_tissue with local raw data to enable them."
)
MISSING_LOYALTY = (
    "This panel stores only the leading weekly brand. Rebuild with python -m data.etl.build_tissue "
    "for exact top-manufacturer unit share and the six-attribute correlation."
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


def _ecdf(values, weights, support):
    order = np.argsort(values)
    cumulative = np.insert(np.cumsum(weights[order]), 0, 0.0)
    return cumulative[np.searchsorted(values[order], support, side="right")] / cumulative[-1]


def _ks_similarity(persona, weights, real):
    # The persona ETL publishes these values to six decimals. Compare ties at that precision.
    persona, real = np.round(persona, 6), np.round(real, 6)
    support = np.union1d(persona, real)
    difference = _ecdf(persona, weights, support) - _ecdf(real, np.ones(len(real)), support)
    return float(1 - np.max(np.abs(difference)))


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


def _reference_checks(personas, households):
    weights = personas["weight"].to_numpy()
    marginal = {}
    for field in ATTRIBUTES:
        score = None
        if households is not None and field in households:
            score = _ks_similarity(
                personas[field].to_numpy(), weights, households[field].to_numpy()
            )
        marginal[field] = {"value": score, "status": _status(score, TARGETS["ks_similarity_min"])}

    rates = personas["weekly_buy_rate"].to_numpy()
    center = np.average(rates, weights=weights)
    persona_sd = float(np.sqrt(np.average((rates - center) ** 2, weights=weights)))
    real_sd = None if households is None else float(households["weekly_buy_rate"].std(ddof=0))
    ratio = None if real_sd is None or real_sd == 0 else persona_sd / real_sd
    variance = {
        "persona_sd": persona_sd,
        "household_sd": real_sd,
        "value": ratio,
        "status": _status(ratio, TARGETS["sd_ratio_min"]),
    }
    if real_sd == 0:
        variance["note"] = "Household weekly buy rates have zero standard deviation."

    persona_corr = _mean_abs_correlation(personas[list(ATTRIBUTES)], weights)
    real_corr = None
    if households is not None and set(ATTRIBUTES).issubset(households):
        real_corr = _mean_abs_correlation(households[list(ATTRIBUTES)])
    gap = None if persona_corr is None or real_corr is None else abs(persona_corr - real_corr)
    correlation = {
        "persona_mean_abs_corr": persona_corr,
        "household_mean_abs_corr": real_corr,
        "gap": gap,
        "status": _status(gap, TARGETS["correlation_gap_max"], maximum=True),
    }
    if gap is None:
        correlation["note"] = "All six attributes must be present and nonconstant in both groups."
    return marginal, variance, correlation


def _direction(personas, field, left_name, left, right_name, right):
    def weighted_mean(mask):
        selected = personas.loc[mask]
        if selected.empty:
            return None
        return float(np.average(selected[field], weights=selected["weight"]))

    a, b = weighted_mean(left), weighted_mean(right)
    if a is None or b is None:
        status = "skipped"
    else:
        status = "pass" if round(a, 12) < round(b, 12) else "fail"
    source = "data" if field == "price_elasticity_observed" else "default"
    return {"field": field, "source": source, left_name: a, right_name: b, "status": status}


def _direction_checks(personas):
    bands = personas["income_band"]
    lower = pd.to_numeric(bands.str.extract(r"^(\d+)(?:-\d+K|K\+)$", expand=False))
    low = bands.eq("Under 15K") | lower.lt(35)
    high = lower.ge(75)
    archetype = personas["archetype"]
    return {
        "deal_hunter_vs_loyal": _direction(
            personas,
            "price_elasticity",
            "deal_hunter",
            archetype.eq("deal_hunter"),
            "loyal",
            archetype.eq("loyal"),
        ),
        "income_observed": _direction(
            personas, "price_elasticity_observed", "low", low, "high", high
        ),
        "income_prior": _direction(personas, "price_elasticity", "low", low, "high", high),
    }


def build_eval_report() -> dict:
    personas = pd.DataFrame(loaders.load_personas())
    inputs = [*ATTRIBUTES, "weight", "price_elasticity", "price_elasticity_observed"]
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
    marginal, variance, correlation = _reference_checks(personas, households)
    directions = _direction_checks(personas)

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
    results = [*marginal.values(), variance, correlation, *directions.values()]
    statuses = Counter(row["status"] for row in results)
    return {
        "status": "fail" if statuses["fail"] else "skipped" if statuses["skipped"] else "pass",
        "check_counts": {name: statuses[name] for name in ("pass", "fail", "skipped")},
        "source": "data",
        "reference": reference,
        "persona_count": len(personas),
        "targets": TARGETS.copy(),
        "targets_source": "default",
        "method": {
            "reference": "Pre-holdout panel from the packaged simulated Complete Journey data.",
            "weights": "Population weights for personas and equal weights for households.",
            "marginal_fit": "One minus maximum empirical CDF distance at six-decimal precision.",
            "variance_ratio": "Population SD of persona buy rate over household population SD.",
            "over_correlation": "Absolute gap in mean absolute Pearson r over 15 distinct pairs.",
            "direction_checks": (
                "Weighted persona means. A more negative slope means greater price sensitivity. "
                "Annual income groups are below USD 35k and at least USD 75k. Observed slopes "
                "were fitted by archetype and assigned to personas."
            ),
            "confidence_inputs": (
                "Coverage is the percentage of all assumptions labelled data. Rival evidence "
                "counts library rows with both a nonblank source URL and a known reaction type."
            ),
        },
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


def main() -> None:
    report = build_eval_report()
    destination = loaders.PROCESSED / "eval_report.json"
    with TemporaryDirectory(dir=destination.parent) as staging:
        temp = Path(staging) / destination.name
        temp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temp.replace(destination)
    rows = []
    for field, score in report["marginal_fit"].items():
        rows.append((f"KS {field}", score["value"], score["status"]))
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

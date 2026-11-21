"""Compare demographic coverage and calibrate sample weights to category buyers."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

BEHAVIOUR_ATTRIBUTES = (
    "weekly_buy_rate",
    "purchase_cycle_weeks",
    "avg_units",
    "avg_price_paid",
    "promo_share",
    "coupon_rate",
    "brand_loyalty",
)
SELECTION_KS_MIN = 0.90


def _ecdf(values, weights, support):
    order = np.argsort(values)
    cumulative = np.insert(np.cumsum(weights[order]), 0, 0.0)
    return cumulative[np.searchsorted(values[order], support, side="right")] / cumulative[-1]


def ks_similarity(values, weights, reference):
    """One minus the maximum ECDF gap, comparing ties at published precision."""
    values, reference = np.round(values, 6), np.round(reference, 6)
    support = np.union1d(values, reference)
    gaps = _ecdf(values, weights, support) - _ecdf(reference, np.ones(len(reference)), support)
    return float(1 - np.max(np.abs(gaps)))


def compare_demographic_groups(households: pd.DataFrame) -> dict:
    known = households[["age", "income"]].notna().all(axis=1)
    covered, missing = households.loc[known], households.loc[~known]
    report = {
        "status": "available",
        "source": "data",
        "threshold": SELECTION_KS_MIN,
        "method": (
            "Each pre-holdout category buyer counts once. Coverage requires both age and income. "
            "KS similarity is one minus the KS statistic. Mean difference is with demographics "
            "minus without, in each attribute's original units."
        ),
        "counts": {"with_demographics": len(covered), "without_demographics": len(missing)},
        "metrics": {},
        "triggered_attributes": [],
        "needs_reweighting": False,
    }
    if covered.empty or missing.empty:
        report.update(status="skipped", note="Comparison requires buyers in both coverage groups.")
        return report

    for field in BEHAVIOUR_ATTRIBUTES:
        with_demo = covered[field].to_numpy(dtype=float)
        without_demo = missing[field].to_numpy(dtype=float)
        if not np.isfinite(with_demo).all() or not np.isfinite(without_demo).all():
            raise ValueError(f"Selection-bias comparison requires finite {field}")
        ks = ks_similarity(with_demo, np.ones(len(covered)), without_demo)
        report["metrics"][field] = {
            "ks_similarity": ks,
            "mean_with_demographics": float(with_demo.mean()),
            "mean_without_demographics": float(without_demo.mean()),
            "mean_difference": float(with_demo.mean() - without_demo.mean()),
        }
        if round(ks, 12) < SELECTION_KS_MIN:
            report["triggered_attributes"].append(field)
    report["needs_reweighting"] = bool(report["triggered_attributes"])
    return report


def _quartile_cuts(values):
    # Place cuts between distinct values, near quartiles, retaining the mass of ties.
    levels, counts = np.unique(values, return_counts=True)
    endpoints = counts.cumsum()[:-1] / len(values)
    if not len(endpoints):
        return np.array([])
    closest = np.abs(endpoints[:, None] - [0.25, 0.50, 0.75]).argmin(axis=0)
    return levels[np.unique(closest)]


def calibrate_selection(
    households: pd.DataFrame, sample: pd.DataFrame
) -> tuple[pd.DataFrame, dict]:
    """Fit aggregate behaviour-bin factors while preserving demographic stratum weights."""
    report = compare_demographic_groups(households)
    sample = sample.copy()
    fields = report["triggered_attributes"]
    if not fields:
        report["reweighting"] = {"applied": False, "factors": {}}
        return sample, report

    base = sample["weight"].to_numpy(dtype=float)
    if not np.isfinite(base).all() or not (base > 0).all() or not np.isclose(base.sum(), 1):
        raise ValueError("Selection calibration requires positive demographic weights summing to 1")
    groups, _ = pd.factorize(sample.get("_weight_stratum", sample["archetype"]))
    if np.any(groups < 0):
        raise ValueError("Every sampled buyer needs a weighting stratum")
    mass = np.bincount(groups, weights=base)
    design, targets, factors = [], [], {}
    for field in fields:
        population = households[field].to_numpy().round(6)
        cuts = _quartile_cuts(population)
        bins = np.arange(len(cuts) + 1)
        target = np.bincount(np.searchsorted(cuts, population), minlength=len(bins)) / len(
            population
        )
        membership = np.searchsorted(cuts, sample[field].round(6))[:, None] == bins
        design.append(membership)
        targets.append(target)
        factors[field] = {"upper_cuts_inclusive": cuts.tolist()}
    design = np.column_stack(design).astype(float)
    target = np.concatenate(targets)

    def balance(beta):
        # Subtract each stratum maximum before exponentiating for numerical stability.
        logits = design @ beta
        peak = np.full(len(mass), -np.inf)
        np.maximum.at(peak, groups, logits)
        tilted = base * np.exp(logits - peak[groups])
        denominator = np.bincount(groups, weights=tilted)
        weights = tilted * (mass / denominator)[groups]
        log_partition = mass @ (peak + np.log(denominator / mass))
        return weights, log_partition - target @ beta

    def objective(beta):
        weights, dual = balance(beta)
        return dual, design.T @ weights - target

    # Limit each fitted multiplier to the population-size odds. Unsupported margins
    # cannot drive the optimizer to infinity. The audit exposes any residual mismatch.
    limit = np.log(len(households))
    solution = minimize(
        objective,
        np.zeros(len(target)),
        method="L-BFGS-B",
        jac=True,
        bounds=[(-limit, limit)] * len(target),
        options={"ftol": 1e-14, "gtol": 1e-10, "maxiter": 2000},
    )
    if not solution.success:
        raise ValueError(f"Could not calibrate persona weights: {solution.message}")
    weights, _ = balance(solution.x)
    offset = 0
    for field, shares in zip(fields, targets, strict=True):
        region = slice(offset, offset + len(shares))
        factors[field].update(
            multiplier=np.exp(solution.x[region]).tolist(),
            target_share=shares.tolist(),
            before_share=(design[:, region].T @ base).tolist(),
            after_share=(design[:, region].T @ weights).tolist(),
        )
        offset += len(shares)
    sample["weight"] = weights
    report["reweighting"] = {
        "applied": True,
        "source": "data",
        "method": (
            "Calibrate flagged behaviour distributions to all buyers using exponential tilting "
            "from the demographic design weights. Multiply by each reported bin factor, then "
            "normalize within each original archetype and demographic weighting stratum to "
            "preserve its original mass. Bins follow the nearest attainable population "
            "quartiles without splitting ties. The last bin is above the final cut. Each "
            "multiplier is bounded by population size and its reciprocal. Target and achieved "
            "shares include bins with limited or zero sample support."
        ),
        "factors": factors,
        "max_abs_share_gap_before": float(np.max(np.abs(design.T @ base - target))),
        "max_abs_share_gap_after": float(np.max(np.abs(design.T @ weights - target))),
    }
    return sample, report

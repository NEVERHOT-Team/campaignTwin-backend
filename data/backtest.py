"""Engine-facing backtest scores in percentage points, using published aggregates."""

from decimal import Decimal
from math import isfinite

from data.loaders import load_backtest


def naive_benchmark() -> float:
    """Return the historical mean DiD lift computed by the backtest ETL."""
    return load_backtest()["actual"]["naive_lift_pct"]


def evaluate_backtest(simulated_lift_pct: float) -> dict:
    """Compare a lift to the inclusive holdout interval and its midpoint."""
    if not isfinite(simulated_lift_pct):
        raise ValueError("simulated_lift_pct must be finite")
    actual = load_backtest()["actual"]
    # Use decimal percentage points so exact boundary values and error ties stay inclusive.
    low, high = (Decimal(str(v)) for v in actual["lift_range_pct"])
    midpoint = (low + high) / 2
    simulated = Decimal(str(simulated_lift_pct))
    error = abs(simulated - midpoint)
    naive = actual["naive_lift_pct"]
    inside = low <= simulated <= high
    if inside:
        label = "Calibrated"
    elif error <= 2 * (high - low):
        label = "Partially calibrated"
    else:
        label = "Uncalibrated"
    return {
        "simulated": simulated_lift_pct,
        "observed_range": list(actual["lift_range_pct"]),
        "naive": naive,
        "inside_range": inside,
        "abs_error_vs_mid": float(error),
        "beats_naive": error < abs(Decimal(str(naive)) - midpoint),
        "label": label,
    }

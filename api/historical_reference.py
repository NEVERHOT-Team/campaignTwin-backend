"""Pin the aggregate Complete Journey holdout available at admission."""

import hashlib
import json
import logging
import re
from statistics import mean, median

from data import loaders
from data.etl.backtest import BASELINE, CAMPAIGN, CONTROL_PRODUCT_TYPES
from data.etl.config import BACKTEST_CAMPAIGN_ID, CATEGORY, MIN_SEGMENT_HOUSEHOLDS

logger = logging.getLogger(__name__)


# The release stores category sales and the baseline window in its aggregate description.
_SALES_PERIODS = re.compile(
    r"Complete Journey toilet-tissue coupon campaign (\d+)\. All-household category sales were "
    r"USD ([0-9.]+) in weeks (\d+)–(\d+) versus USD ([0-9.]+) in weeks (\d+)–(\d+)\."
)


# This loader version describes its protocol in prose. Require the released description
# until the data interface publishes a structured, versioned protocol identifier.
_METHOD = (
    "Difference-in-differences in percentage points. Subtract the unweighted mean "
    "control-category sales change from the toilet-tissue sales change. Each change "
    "compares total retailer sales including returns in the same eight-week windows. "
    "The reported range spans the mean-adjusted and median-adjusted lifts."
)
_SELECTION = (
    "Household paper and hygiene product types in GROCERY with zero "
    f"campaign-{BACKTEST_CAMPAIGN_ID} coupon-eligible products."
)


def _source_text(value):
    if not isinstance(value, str):
        raise ValueError("Missing historical methodology or description")
    return " ".join(value.split())


def _growth(baseline, campaign):
    if (type(baseline) not in (int, float) or type(campaign) not in (int, float)
            or baseline <= 0):
        raise ValueError("Invalid aggregate sales")
    return 100 * (campaign / baseline - 1)


def _periods(case):
    description = _SALES_PERIODS.match(_source_text(case["description"]))
    if (not description or case["source"] != "data"
            or case["campaign_id"] != str(BACKTEST_CAMPAIGN_ID)
            or case["campaign_id"] != description[1]):
        raise ValueError("Unverified campaign or population")
    _, sales, first, last, prior_sales, prior_first, prior_last = description.groups()
    start, end = case["start_week"], case["end_week"]
    if (type(start) is not int or type(end) is not int or (start, end) != (int(first), int(last))
            or (start, end) != (CAMPAIGN[0], CAMPAIGN[-1])
            or (int(prior_first), int(prior_last)) != (BASELINE[0], BASELINE[-1])):
        raise ValueError("Invalid eight-week campaign or baseline window")
    weeks = case["calibration_weeks"]
    if (not isinstance(weeks, list) or not weeks
            or any(type(w) is not int or w != i for i, w in enumerate(weeks, 1))
            or weeks[-1] != start - 1):
        raise ValueError("Invalid fit cutoff")
    return (weeks[-1], {"first": int(prior_first), "last": int(prior_last)},
            {"first": start, "last": end}, _growth(float(prior_sales), float(sales)))


def _observation(case, raw_growth):
    control = case["seasonality_control"]
    rows = control["categories"]
    if (_source_text(control["method"]) != _METHOD
            or _source_text(control["selection"]) != _SELECTION
            or not isinstance(rows, list) or len(rows) != len(CONTROL_PRODUCT_TYPES)
            or {r["product_type"] for r in rows} != set(CONTROL_PRODUCT_TYPES)):
        raise ValueError("Unverified seasonality controls")
    changes = []
    for row in rows:
        growth = _growth(row["baseline_sales_usd"], row["campaign_sales_usd"])
        if type(row["change_pct"]) not in (int, float) or round(growth, 6) != row["change_pct"]:
            raise ValueError("Control sales disagree with reported change")
        changes.append(growth)

    actual = case["actual"]
    # Gan's evaluate_backtest uses the naive comparator alongside the observed range.
    if type(actual["naive_lift_pct"]) not in (float, int):
        raise ValueError("Invalid naive lift comparator")
    adjusted = {"lift_pct_raw": raw_growth, "control_change_pct": mean(changes),
                "control_change_pct_median": median(changes),
                "lift_pct": raw_growth - mean(changes),
                "lift_pct_median": raw_growth - median(changes)}
    if any(type(actual[k]) not in (int, float) or actual[k] != round(v, 2)
           for k, v in adjusted.items()):
        raise ValueError("Adjusted observation disagrees with the aggregate sales")
    low, high = actual["lift_range_pct"]
    if (type(low) not in (int, float) or type(high) not in (int, float)
            or [low, high] != sorted([actual["lift_pct"], actual["lift_pct_median"]])):
        raise ValueError("Range disagrees with the two adjustment estimates")
    return {"low": low, "high": high, "basis": "mean_and_median_control_adjustment"}


def _reference(case, review):
    # Hash the complete loader artifact, including fields outside the API projection.
    canonical = json.dumps(case, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False, allow_nan=False)
    cutoff, baseline, campaign, growth = _periods(case)
    observed = _observation(case, growth)
    depth = case["discount_depth"]
    households = case["treated_households"]
    if (type(depth) not in (float, int) or not 0 <= depth <= 1
            or type(households) is not int or households < MIN_SEGMENT_HOUSEHOLDS
            or case["target_segments"] != []):
        raise ValueError("Unverified historical discount or target population")
    discount = review["resolved_offer"]["discount"]
    matched = ["category"]
    if discount["kind"] == "voucher":
        matched.append("offer_mechanic")
    facts = review["facts"]
    product = review["resolved_product"]
    return {
        "campaign_id": case["campaign_id"], "category": review["category"],
        "dataset": "complete_journey", "source": "data", "matched_on": matched,
        "evidence_version": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "fit_through_week": cutoff, "baseline_weeks": baseline, "campaign_weeks": campaign,
        "metric": "control_adjusted_category_sales_lift", "unit": "percentage_points",
        "observed_range": observed,
        "limitations": [
            "The packaged simulated US grocery panel measures sales across the category. "
            f"The plan covers one pack ({product['pack_label']}) on {product['channel']}.",
            f"The panel covers retail weeks {campaign['first']}–{campaign['last']}. "
            f"The submitted offer runs from {facts['start_date']} until {facts['end_date']} "
            "(end date excluded).",
            f"The median realized coupon discount was {depth:.2%}. The submitted discount "
            f"is {discount['value']} {discount['unit']}.",
            f"The panel assigned coupons to {households:,} households with no recorded segment "
            "targeting rule. The submitted offer covers all eligible buyers.",
            "The range shows how two seasonality adjustments change the observed lift. "
            "It is an observational sensitivity range, not a statistical confidence interval. "
            "Other promotions could affect sales. Category similarity alone cannot validate "
            "the submitted plan's profitability.",
        ],
    }


def pin_reference(review):
    """Select by resolved category and offer; never query the campaign examples for outcomes."""
    if review["category"] != CATEGORY or not review["resolved_product"]:
        return {"reference": None, "reason": "no_comparable_case"}
    try:
        case = loaders.load_backtest()
    except Exception:
        # The optional evidence provider cannot prevent publication of a valid live report.
        logger.exception("Historical campaign source unavailable")
        return {"reference": None, "reason": "source_unavailable"}
    try:
        reference = _reference(case, review)
    except (KeyError, TypeError, ValueError, OverflowError, RecursionError):
        logger.warning("Historical campaign source failed validation", exc_info=True)
        return {"reference": None, "reason": "source_unavailable"}
    return {"reference": reference, "reason": None}

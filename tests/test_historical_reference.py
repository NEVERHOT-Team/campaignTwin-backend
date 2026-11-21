"""Only the released, verified toilet-tissue case can contribute an observation."""

import hashlib
import json
from copy import deepcopy

import pytest

from api.v1_catalog import resolve
from api.v1_contract import contract
from data import loaders


def resolved_review(kind="voucher"):
    operation = contract()["paths"]["/api/v1/campaign-reviews"]["post"]
    examples = operation["requestBody"]["content"]["application/json"]["examples"]
    facts = deepcopy(examples["syntheticRehearsal"]["value"]["fields"])
    facts["offer"].update(kind=kind, unit="THB" if kind == "voucher" else "pct")
    return {**resolve(facts), "facts": facts}


@pytest.mark.parametrize("kind,matched", [
    ("voucher", ["category", "offer_mechanic"]), ("percent_off", ["category"]),
])
def test_released_observation(kind, matched):
    from api.historical_reference import pin_reference

    case = loaders.load_backtest()
    snapshot = pin_reference(resolved_review(kind))
    reference = snapshot["reference"]
    assert snapshot["reason"] is None
    assert reference["campaign_id"] == case["campaign_id"] == "18"
    canonical = json.dumps(case, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False, allow_nan=False)
    assert reference["evidence_version"] == hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    assert reference["matched_on"] == matched
    assert reference["fit_through_week"] == 44
    assert reference["baseline_weeks"] == {"first": 37, "last": 44}
    assert reference["campaign_weeks"] == {"first": 45, "last": 52}
    assert reference["observed_range"] == {
        "low": 1.71, "high": 6.24, "basis": "mean_and_median_control_adjustment",
    }
    notes = " ".join(reference["limitations"]).lower()
    for detail in ("simulated", "us grocery", "pack", "shopee", "weeks", "22.25%", "target",
                   "seasonality"):
        assert detail in notes


def test_unrelated_category_even_with_quoted_tissue_hint():
    from api.historical_reference import pin_reference

    review = resolved_review()
    review["category"] = "instant_coffee"
    assert pin_reference(review) == {"reference": None, "reason": "no_comparable_case"}


@pytest.mark.parametrize("field,bad_value", [
    ("campaign_id", "19"), ("source", "default"), ("description", "instant coffee"),
    ("calibration_weeks", [1, 45]), ("calibration_weeks", list(range(1, 44))),
    ("start_week", 0), ("end_week", 44),
    ("discount_depth", float("nan")), ("seasonality_control", {}),
    ("actual.lift_range_pct", [6.24, 1.71]), ("actual.lift_range_pct", [True, 6.24]),
    ("actual.lift_range_pct", [1.71, float("inf")]), ("actual.lift_range_pct", [1.71, 7]),
])
def test_damaged_evidence_is_unavailable(monkeypatch, field, bad_value):
    from api.historical_reference import pin_reference

    case = deepcopy(loaders.load_backtest())
    key = field.split(".")[-1]
    target = case["actual"] if field.startswith("actual.") else case
    target[key] = bad_value
    monkeypatch.setattr(loaders, "load_backtest", lambda: case)
    assert pin_reference(resolved_review()) == {"reference": None, "reason": "source_unavailable"}


@pytest.mark.parametrize("location,key,value", [
    ("control", "categories", "unverified"),
    ("control", "selection", ["zero campaign-18 coupon-eligible products"]),
    ("control", "method", "Use only treated households in one week for mean-adjusted and "
                          "median-adjusted sales in percentage points."),
    ("control", "selection", "Includes products eligible for campaign 18 coupons."),
    ("row", "product_type", "PAPER TOWELS"),
    ("row", "baseline_sales_usd", 1),
    ("row", "change_pct", 3.0),
    ("actual", "lift_pct_raw", 8.99),
    ("actual", "control_change_pct_median", 1.0),
])
def test_contradictory_control_evidence_is_unavailable(monkeypatch, location, key, value):
    from api.historical_reference import pin_reference

    case = deepcopy(loaders.load_backtest())
    controls = case["seasonality_control"]
    target = {"control": controls, "row": controls["categories"][0], "actual": case["actual"]}
    target[location][key] = value
    monkeypatch.setattr(loaders, "load_backtest", lambda: case)
    assert pin_reference(resolved_review()) == {"reference": None, "reason": "source_unavailable"}


def test_self_consistent_weeks_outside_released_campaign_are_unavailable(monkeypatch):
    from api.historical_reference import pin_reference

    case = deepcopy(loaders.load_backtest())
    case.update(start_week=145, end_week=152)
    description = case["description"].replace("45–52", "145–152")
    case["description"] = description.replace("37–44", "137–144")
    monkeypatch.setattr(loaders, "load_backtest", lambda: case)
    assert pin_reference(resolved_review()) == {"reference": None, "reason": "source_unavailable"}


@pytest.mark.parametrize("invalid_naive", [None, True, "unavailable"])
def test_source_requires_numeric_naive_comparator_used_by_gan(monkeypatch, invalid_naive):
    from api.historical_reference import pin_reference

    case = deepcopy(loaders.load_backtest())
    case["actual"]["naive_lift_pct"] = invalid_naive
    monkeypatch.setattr(loaders, "load_backtest", lambda: case)
    assert pin_reference(resolved_review()) == {"reference": None, "reason": "source_unavailable"}


def test_source_prose_can_be_rewrapped_without_changing_observation(monkeypatch):
    from api.historical_reference import pin_reference

    case = deepcopy(loaders.load_backtest())
    review = resolved_review()
    original = pin_reference(review)["reference"]
    case["description"] = case["description"].replace(". ", ".\n")
    for key in ("method", "selection"):
        case["seasonality_control"][key] = case["seasonality_control"][key].replace(" ", "\n  ")
    monkeypatch.setattr(loaders, "load_backtest", lambda: case)
    updated = pin_reference(review)["reference"]
    assert updated is not None
    assert updated["observed_range"] == original["observed_range"]
    assert updated["evidence_version"] != original["evidence_version"]


@pytest.mark.parametrize("alternate_id", ["19", 18, True])
def test_case_identity_must_match_released_campaign(monkeypatch, alternate_id):
    from api.historical_reference import pin_reference

    case = deepcopy(loaders.load_backtest())
    case["campaign_id"] = alternate_id
    case["description"] = case["description"].replace("campaign 18", f"campaign {alternate_id}")
    controls = case["seasonality_control"]
    controls["selection"] = controls["selection"].replace("campaign-18", f"campaign-{alternate_id}")
    monkeypatch.setattr(loaders, "load_backtest", lambda: case)
    assert pin_reference(resolved_review()) == {"reference": None, "reason": "source_unavailable"}

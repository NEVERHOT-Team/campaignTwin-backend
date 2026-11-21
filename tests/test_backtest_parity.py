"""Check the server-side historical prediction boundary against Gan's public scorer."""

import hashlib
import json
import math

from api.historical_prediction import compare_prediction
from api.historical_reference import pin_reference
from api.v1_catalog import resolve
from api.v1_contract import contract
from data.backtest import evaluate_backtest
from data.loaders import load_backtest


def test_aligned_artifact_agrees_with_released_backtest_at_inclusive_boundaries():
    intake = contract()["paths"]["/api/v1/campaign-reviews"]["post"]["requestBody"]
    examples = intake["content"]["application/json"]["examples"]
    facts = examples["syntheticRehearsal"]["value"]["fields"]
    reference = pin_reference({**resolve(facts), "facts": facts})["reference"]
    case = load_backtest()
    canonical = json.dumps(case, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                           allow_nan=False).encode("utf-8")
    first, last = case["start_week"], case["end_week"]
    duration = last - first + 1
    low, high = case["actual"]["lift_range_pct"]
    for value in (math.nextafter(low, -math.inf), low, (low + high) / 2, high,
                  math.nextafter(high, math.inf)):
        artifact = {
            "artifact_id": f"test-only-{value}", "campaign_id": case["campaign_id"],
            "evidence_version": hashlib.sha256(canonical).hexdigest(),
            "protocol": "complete_journey_tissue_holdout_v1",
            "fit_through_week": max(case["calibration_weeks"]),
            "baseline_weeks": {"first": first - duration, "last": first - 1},
            "campaign_weeks": {"first": first, "last": last},
            "metric": "control_adjusted_category_sales_lift", "unit": "percentage_points",
            "value": value,
        }
        expected = evaluate_backtest(value)
        bounds = reference["observed_range"]
        assert [bounds["low"], bounds["high"]] == expected["observed_range"]
        result = compare_prediction(reference, artifact)
        assert result["status"] == ("inside_range" if expected["inside_range"] else "outside_range")
        assert result["reason"] is None
        assert result["prediction"]["value"] == expected["simulated"]

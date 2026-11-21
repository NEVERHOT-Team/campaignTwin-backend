"""Published historical operation and the engine-owner artifact boundary."""

from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest
import yaml
from fastapi.testclient import TestClient

import db
from api.v1_contract import contract, validator
from main import app

PATH = "/api/v1/runs/{run_id}/historical-comparison"
SHARED = "./campaigntwin-api-v1.openapi.yaml"


def test_published_historical_contract_is_self_contained():
    docs = Path(__file__).resolve().parents[1] / "docs"
    draft = docs / "campaign-comparison-v1.proposed.openapi.yaml"
    proposed = yaml.safe_load(draft.read_text().replace(SHARED + "#", "#"))
    response = TestClient(app).get("/openapi.json")
    assert response.status_code == 200
    public = response.json()
    assert public["paths"][PATH] == proposed["paths"][PATH]
    for name, schema in proposed["components"]["schemas"].items():
        assert public["components"]["schemas"][name] == schema
    for path in ["/api/v1/runs/{run_id}/evaluation", "/api/v1/runs/{run_id}/input",
                 "/api/v1/runs/{run_id}/report"]:
        assert public["paths"][path] == contract()["paths"][path]


@pytest.fixture
def reference(monkeypatch, tmp_path):
    from api.historical_reference import pin_reference

    monkeypatch.setenv("DB_MODE", "json")
    monkeypatch.setattr(db, "RUNS_DIR", tmp_path)
    path = "/api/v1/campaign-reviews"
    operation = contract()["paths"][path]["post"]
    examples = operation["requestBody"]["content"]["application/json"]["examples"]
    example = examples["syntheticRehearsal"]
    review = TestClient(app).post(path, json=example["value"])
    assert review.status_code == 201, review.text
    reference = pin_reference(review.json())["reference"]
    validator("HistoricalCase").validate(reference)
    return reference


def artifact(reference, value):
    fields = ("campaign_id", "evidence_version", "metric", "unit", "fit_through_week",
              "baseline_weeks", "campaign_weeks")
    return {**{key: reference[key] for key in fields}, "artifact_id": "test-engine-artifact-1",
            "protocol": "complete_journey_tissue_holdout_v1", "value": value}


@pytest.mark.parametrize(("bound", "delta", "expected"), [
    ("low", 0, "inside_range"), ("high", 0, "inside_range"),
    ("low", -0.01, "outside_range"), ("high", 0.01, "outside_range"),
])
def test_historical_prediction_uses_inclusive_range(reference, bound, delta, expected):
    from api.historical_prediction import compare_prediction

    prediction = artifact(reference, reference["observed_range"][bound] + delta)
    validator("HistoricalPrediction").validate(prediction)
    assert compare_prediction(reference, prediction) == {
        "status": expected, "reason": None, "prediction": prediction,
    }


@pytest.mark.parametrize("override", [
    {"campaign_id": "another"}, {"evidence_version": "0" * 64}, {"metric": "profit"},
    {"unit": "THB"}, {"protocol": "all_shoppers_sales_v2"}, {"fit_through_week": 45},
    {"fit_through_week": 43}, {"baseline_weeks": {"first": 36, "last": 44}},
    {"campaign_weeks": {"first": 45, "last": 51}}, {"artifact_id": ""},
    {"value": float("nan")}, {"value": float("inf")}, {"value": True}, {"unknown": 1},
    {"value": np.float32("inf")}, {"value": Decimal("Infinity")}, {"value": complex(4, 1)},
])
def test_incompatible_engine_artifact_is_omitted(reference, override):
    from api.historical_prediction import compare_prediction

    result = compare_prediction(reference, {**artifact(reference, 4), **override})
    assert result == {"status": "reference_only", "reason": "incompatible_prediction",
                      "prediction": None}


def test_absent_prediction_preserves_reference(reference):
    from api.historical_prediction import compare_prediction

    assert compare_prediction(reference) == {"status": "reference_only",
                                             "reason": "prediction_missing", "prediction": None}

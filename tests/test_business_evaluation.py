from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

import db
from api.v1_contract import contract, validator
from api.v1_evaluation import business_evaluation
from main import app


@pytest.mark.parametrize("budget", [1000, 1])
def test_planner_can_verify_decision_against_run(monkeypatch, tmp_path, budget):
    monkeypatch.setenv("DB_MODE", "json")
    monkeypatch.setattr(db, "RUNS_DIR", tmp_path)
    reviews = "/api/v1/campaign-reviews"
    operation = contract()["paths"][reviews]["post"]
    example = operation["requestBody"]["content"]["application/json"]["examples"]
    fields = {**example["syntheticRehearsal"]["value"]["fields"], "budget_thb": budget}
    with TestClient(app) as client:
        review = client.post(reviews, json={"fields": fields}).json()
        approval = client.post(
            f"{reviews}/{review['review_id']}/approvals", json={"revision": 1, "confirmed": True}
        ).json()
        submission = client.post("/api/v1/runs", json={
            "approval_id": approval["approval_id"], "submission_id": str(uuid4()),
        }).json()
        run = submission["status_url"]
        report = client.get(run + "/report").json()
        response = client.get(run + "/evaluation")
        assert response.status_code == 200, response.text
        result = response.json()
        validator("BusinessEvaluation").validate(result)
        assert result["decision"] == report["decision"]
        for comparison, plan in zip(result["comparisons"], report["plan_results"], strict=True):
            assert comparison["scenario_results"] == plan["scenario_results"]
            assert comparison["plan_id"] == plan["input"]["plan_id"]
        metrics = {m["metric"]: m for m in report["scenario_results"][0]["metrics"]}
        gates = {g["metric"]: g for g in result["checks"] if g["scenario"] == "S1"}
        risk = gates["probability_loss"]
        assert risk["observed"] == metrics["probability_loss"]["value"]
        assert risk["threshold"] == review["evaluation"]["max_loss_probability"]
        assert (risk["status"] == "pass") == (risk["observed"] <= risk["threshold"])
        profit = gates["incremental_profit"]
        assert profit["observed"] == metrics["incremental_profit"]["value"]["p50"]
        assert (profit["status"] == "pass") == (profit["observed"] >= profit["threshold"])
        leakage = gates["subsidy_leakage"]
        if metrics["subsidy_leakage"]["status"] == "unavailable":
            assert leakage["status"] == "unavailable"
            assert leakage["observed"] is None
            assert leakage["unavailable_reason"] == "undefined_denominator"
        else:
            assert leakage["status"] == "pass"
            assert leakage["observed"] == metrics["subsidy_leakage"]["value"]["p50"]
        expected = "hold" if any(gate["status"] == "fail" for gate in gates.values()) else "proceed"
        if leakage["status"] == "unavailable":
            expected = "insufficient_evidence"
        assert result["decision"]["status"] == expected


def test_older_report_retains_results_and_explains_missing_thresholds():
    report = contract()["components"]["examples"]["HoldReport"]["value"]
    result = business_evaluation({k: v for k, v in report.items() if k != "plan_results"})
    validator("BusinessEvaluation").validate(result)
    assert result["decision"] == report["decision"]
    assert result["checks"] == []
    assert result["checks_unavailable_reason"]
    assert result["comparisons"][0]["scenario_results"] == report["scenario_results"]


@pytest.mark.parametrize("target", [
    None,
    {"metric": "incremental_roas", "unit": "ratio", "minimum": 1},
    {"metric": "incremental_profit", "unit": "USD", "minimum": 10},
])
def test_stored_target_must_support_a_thb_profit_check(tmp_path, monkeypatch, target):
    monkeypatch.setenv("DB_MODE", "json")
    monkeypatch.setattr(db, "RUNS_DIR", tmp_path)
    reviews = "/api/v1/campaign-reviews"
    endpoint = contract()["paths"][reviews]["post"]
    examples = endpoint["requestBody"]["content"]["application/json"]["examples"]
    example = examples["syntheticRehearsal"]
    with TestClient(app, raise_server_exceptions=False) as client:
        review = client.post(reviews, json=example["value"]).json()
        receipt = client.post(
            f"{reviews}/{review['review_id']}/approvals", json={"revision": 1, "confirmed": True}
        ).json()
        run = client.post("/api/v1/runs", json={
            "approval_id": receipt["approval_id"], "submission_id": str(uuid4()),
        }).json()
        with db.v1_transaction() as state:
            report = state["runs"][run["run_id"]]["report"]
            for plan in report["plan_results"]:
                plan["input"]["evaluation"]["goal_target"] = target
            validator("RunReport").validate(report)
        response = client.get(run["status_url"] + "/evaluation")
        assert response.status_code == 200
        result = response.json()
        validator("BusinessEvaluation").validate(result)
        assert result["decision"] == report["decision"]
        assert result["checks_unavailable_reason"]
        assert {c["metric"] for c in result["checks"]} == {"probability_loss", "subsidy_leakage"}

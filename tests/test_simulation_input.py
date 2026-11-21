from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

import db
from api import v1_runs
from api.v1_contract import contract
from main import app


@pytest.mark.parametrize("engine_failure", [False, True])
def test_run_input_is_the_frozen_evaluator_call(tmp_path, monkeypatch, engine_failure):
    monkeypatch.setenv("DB_MODE", "json")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setattr(db, "RUNS_DIR", tmp_path / "runs")
    evaluate = v1_runs.evaluate
    received = {}

    def capture(run_id, approval_id, review):
        received.update(run_id=run_id, approval_id=approval_id, review=review)
        if engine_failure:
            raise ValueError("Rejected at engine boundary")
        return evaluate(run_id, approval_id, review)

    monkeypatch.setattr(v1_runs, "evaluate", capture)
    with TestClient(app) as client:
        reviews = "/api/v1/campaign-reviews"
        operation = contract()["paths"][reviews]["post"]
        examples = operation["requestBody"]["content"]["application/json"]["examples"]
        review = client.post(reviews, json=examples["syntheticRehearsal"]["value"]).json()
        review_url = f"{reviews}/{review['review_id']}"
        approval = client.post(f"{review_url}/approvals", json={"revision": 1, "confirmed": True})
        request = {"approval_id": approval.json()["approval_id"], "submission_id": str(uuid4())}
        run = client.post("/api/v1/runs", json=request).json()
        edit = {"base_revision": 1, "edits": {"budget_thb": 321}}
        assert client.post(f"{review_url}/revisions", json=edit).status_code == 201
        snapshot = client.get(run["status_url"] + "/input")
        assert snapshot.status_code == 200
        assert snapshot.json() == received
        assert snapshot.json()["review"] == review
        assert client.get(f"/api/v1/runs/{uuid4()}/input").status_code == 404

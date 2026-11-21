"""Exercise historical evidence through the same review and run routes used by planners."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

import db
from api import v1_runs
from api.v1_contract import validator
from data import loaders
from main import app

REVIEWS = "/api/v1/campaign-reviews"
RUNS = "/api/v1/runs"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_MODE", "json")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setattr(db, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(db, "CACHE_DIR", tmp_path / "cache")
    with TestClient(app) as client:
        yield client


def voucher_review(client):
    content = client.get("/openapi.json").json()["paths"][REVIEWS]["post"]["requestBody"]["content"]
    example = content["application/json"]["examples"]["syntheticRehearsal"]["value"]
    review = client.post(REVIEWS, json=example).json()
    review_url = REVIEWS + "/" + review["review_id"]
    receipt = client.post(review_url + "/approvals", json={"revision": 1, "confirmed": True})
    assert receipt.status_code == 201
    return review_url, {"approval_id": receipt.json()["approval_id"], "submission_id": str(uuid4())}


def submit(client, body):
    response = client.post(RUNS, json=body)
    assert response.status_code == 202, response.text
    return response.json()["status_url"]


def comparison(client, url):
    response = client.get(url + "/historical-comparison")
    assert response.status_code == 200, response.text
    payload = response.json()
    validator("HistoricalComparison").validate(payload)
    return payload


def unavailable_source():
    raise FileNotFoundError("Test historical source outage")


def test_pin_matches_loader_and_survives_refresh_replay_edit_and_restart(client, monkeypatch):
    case = loaders.load_backtest()
    encoded = json.dumps(case, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                         allow_nan=False).encode("utf-8")
    review_url, body = voucher_review(client)
    url = submit(client, body)
    pinned = comparison(client, url)
    assert (pinned["status"], pinned["reason"]) == ("reference_only", "prediction_missing")
    assert pinned["prediction"] is None
    assert pinned["reference"]["evidence_version"] == hashlib.sha256(encoded).hexdigest()
    assert pinned["reference"]["observed_range"] == {
        "low": min(case["actual"]["lift_range_pct"]), "high": max(case["actual"]["lift_range_pct"]),
        "basis": "mean_and_median_control_adjustment",
    }
    with db.v1_transaction() as state:
        report = state["runs"][pinned["run_id"]]["report"]
    assert client.get(url + "/report").json() == report
    evaluation = client.get(url + "/evaluation").json()
    for key in ("run_id", "approval_id", "revision"):
        assert pinned[key] == report[key] == evaluation[key]
    assert evaluation["decision"] == report["decision"]

    monkeypatch.setattr(loaders, "load_backtest", unavailable_source)
    refreshed = submit(client, {**body, "submission_id": str(uuid4())})
    outage = comparison(client, refreshed)
    assert (outage["status"], outage["reason"]) == ("unavailable", "source_unavailable")
    assert outage["reference"] is None and outage["prediction"] is None
    assert client.get(refreshed + "/report").json()["decision"] == report["decision"]
    edit = {"base_revision": 1, "edits": {"budget_thb": 200}}
    assert client.post(review_url + "/revisions", json=edit).status_code == 201
    replay = client.post(RUNS, json=body)
    assert replay.status_code == 200 and replay.json()["status_url"] == url
    assert comparison(client, url) == pinned
    # Boot main:app in a separate interpreter with only the persisted JSON state.
    script = """
import json, sys
from pathlib import Path
import db
from fastapi.testclient import TestClient
from main import app
db.RUNS_DIR = Path(sys.argv[1])
with TestClient(app) as client:
    print(json.dumps([client.get(url + '/historical-comparison').json() for url in sys.argv[2:]]))
"""
    restored = subprocess.check_output(
        [sys.executable, "-c", script, str(db.RUNS_DIR), url, refreshed],
        cwd=Path(__file__).resolve().parents[1], text=True,
    )
    assert json.loads(restored) == [pinned, outage]


def test_before_snapshots_and_failed_or_unknown_runs(client, monkeypatch):
    _, body = voucher_review(client)
    url = submit(client, body)
    run_id = url.split("/")[-1]
    with db.v1_transaction() as state:
        del state["runs"][run_id]["historical_snapshot"]
    legacy = comparison(client, url)
    assert (legacy["status"], legacy["reason"]) == ("unavailable", "snapshot_missing")
    assert legacy["reference"] is None and legacy["prediction"] is None
    assert client.get(url + "/report").status_code == 200

    def evaluation_failure(*args):
        raise RuntimeError("Test evaluator outage")

    monkeypatch.setattr(v1_runs.v1_arena, "evaluate", evaluation_failure)
    failed = submit(client, {**body, "submission_id": str(uuid4())})
    for run, code, status in [(failed, "run_failed", 409), (RUNS + "/" + str(uuid4()),
                                                            "not_found", 404)]:
        response = client.get(run + "/historical-comparison")
        assert response.status_code == status
        validator("Error").validate(response.json())
        assert response.json()["code"] == code


def test_historical_evidence_requires_a_published_report(client):
    _, body = voucher_review(client)
    url = submit(client, body)
    with db.v1_transaction() as state:
        run = state["runs"][url.split("/")[-1]]
        run["report"] = None
    response = client.get(url + "/historical-comparison")
    assert response.status_code == 409
    assert response.json()["code"] == "report_pending"

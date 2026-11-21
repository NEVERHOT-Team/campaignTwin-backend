from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

import db
from agents.llm import call_llm
from api import routes
from contracts.models import (
    BriefParseResponse,
    ContextPack,
    DemoReplay,
    HealthResponse,
    Report,
    RunCreated,
    RunStatus,
    SimulationPlan,
)
from main import app
from mocks import load_mock


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("DB_MODE", "json")
    monkeypatch.setattr(db, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(db, "CACHE_DIR", tmp_path / "cache")
    with TestClient(app) as test_client:
        yield test_client


def test_health_parse_and_context(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert HealthResponse.model_validate(response.json()).model_dump() == {
        "ok": True,
        "db_mode": "json",
    }

    response = client.post("/api/brief/parse", json={"text": "Run a 40% voucher for 11.11"})
    assert response.status_code == 200
    parsed = BriefParseResponse.model_validate(response.json())
    assert parsed == BriefParseResponse.model_validate(load_mock("campaign_spec"))

    response = client.post("/api/plan/context", json=parsed.spec.model_dump(mode="json"))
    assert response.status_code == 200
    assert ContextPack.model_validate(response.json()).rivals

    assert client.post("/api/brief/parse", json={}).status_code == 422
    assert client.post("/api/plan/context", json={}).status_code == 422
    assert client.post("/api/runs", json={}).status_code == 422


def test_start_run_poll_report_and_replay(client, monkeypatch):
    progress_calls = []
    update_progress = db.update_progress

    def record_progress(run_id, progress, **kwargs):
        progress_calls.append(progress)
        update_progress(run_id, progress, **kwargs)

    monkeypatch.setattr(db, "update_progress", record_progress)
    monkeypatch.setattr(routes, "sleep", lambda _: None)

    plan = SimulationPlan.model_validate(
        {
            "spec": load_mock("campaign_spec")["spec"],
            "context": load_mock("context_pack"),
            "strategies": [],
        }
    )
    response = client.post("/api/runs", json=plan.model_dump(mode="json"))
    assert response.status_code == 200
    run_id = RunCreated.model_validate(response.json()).run_id

    for _ in range(12):
        response = client.get(f"/api/runs/{run_id}")
        assert response.status_code == 200
        status = RunStatus.model_validate(response.json())
        if status.status == "done":
            break
    assert status.status == "done"
    assert status.progress == 1
    mock_status = RunStatus.model_validate(load_mock("run_status"))
    assert status.partial_results == mock_status.partial_results
    assert progress_calls == [step / 10 for step in range(11)]
    assert db.get_run(run_id)["plan"] == plan.model_dump(mode="json")

    response = client.get(f"/api/runs/{run_id}/report")
    assert response.status_code == 200
    report = Report.model_validate(response.json())
    response = client.get("/api/demo/replay")
    assert response.status_code == 200
    replay = DemoReplay.model_validate(response.json())
    assert replay.report == report
    assert replay.status.partial_results == status.partial_results


def test_unknown_and_queued_run(client):
    run_id = str(uuid4())
    assert client.get(f"/api/runs/{run_id}").status_code == 404
    assert client.get(f"/api/runs/{run_id}/report").status_code == 404
    db.save_run(run_id, {})
    response = client.get(f"/api/runs/{run_id}")
    assert response.status_code == 200
    assert RunStatus.model_validate(response.json()).status == "queued"
    assert client.get(f"/api/runs/{run_id}/report").status_code == 409


@pytest.mark.parametrize("origin", ["http://localhost:5173", "https://ct-preview.vercel.app"])
def test_cors_preflight(client, origin):
    response = client.options(
        "/api/runs",
        headers={"Origin": origin, "Access-Control-Request-Method": "POST"},
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == origin


def test_fixture_llm_cache(client, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    messages = [{"role": "user", "content": "Fixture prompt"}]
    expected = BriefParseResponse.model_validate(load_mock("campaign_spec"))
    assert call_llm("fixture-model", messages, BriefParseResponse) == expected
    assert db.export_cache(tmp_path / "export") == 1
    assert len(list((tmp_path / "export").glob("*.json"))) == 1
    assert call_llm("fixture-model", messages, BriefParseResponse) == expected

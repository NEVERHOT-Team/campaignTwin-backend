from copy import deepcopy
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

import db
from agents import rival_policy
from api import v1_arena
from api.v1_catalog import ARENA_POLICY, ARENA_PRODUCT, arena_costs, resolve
from api.v1_contract import contract, validator
from main import app

REVIEWS = "/api/v1/campaign-reviews"


def arena_fields():
    operation = contract()["paths"][REVIEWS]["post"]
    examples = operation["requestBody"]["content"]["application/json"]["examples"]
    return deepcopy(examples["arenaRehearsal"]["value"]["fields"])


def test_catalog_resolves_the_arena_choice():
    context = resolve(arena_fields())
    assert context["blockers"] == []
    assert context["resolved_product"] == ARENA_PRODUCT
    assert context["data_versions"]["evaluation_policy"] == ARENA_POLICY
    assert context["evaluation"]["scenarios"] == ["S1", "S2", "S3", "S4"]
    assert context["resolved_offer"]["discount"]["value"] == 40
    assert {"regular_price", "cogs", "channel_commission", "payment_fee",
            "seller_fulfillment"} <= set(context["costs"])
    assert ARENA_PRODUCT in context["supported_product_choices"]


def test_catalog_reads_per_field_sources_without_upgrading_defaults():
    from data.loaders import load_market

    market = {**load_market(), "source": {"ref_price": "data", "unit_cost": "data"},
              "source_notes": {"fields": {"ref_price": "default", "payment_fee_pct": "data"}}}
    costs = arena_costs(market)
    assert costs["regular_price"]["source"] == costs["cogs"]["source"] == "default"
    assert costs["payment_fee"]["source"] == "data"


@pytest.mark.parametrize(
    ("change", "code"),
    [
        ({"offer": {"kind": "percent_off", "value": 60, "unit": "pct", "eligibility": "all"}},
         "offer_terms_unverified"),
        ({"channel": "lazada"}, "unsupported_choice"),
        ({"end_date": "2027-03-01"}, "evaluation_unsupported"),
    ],
)
def test_catalog_blocks_unsupported_arena_terms(change, code):
    facts = {**arena_fields(), **change}
    assert code in {blocker["code"] for blocker in resolve(facts)["blockers"]}


def test_retired_synthetic_choice_is_unlisted_and_cannot_be_approved():
    facts = {**arena_fields(), "product_choice_id": "demo-tissue-12-shopee",
             "offer": {"kind": "voucher", "value": 10, "unit": "THB", "eligibility": "all"},
             "budget_thb": 1000, "start_date": "2026-12-01", "end_date": "2026-12-08"}
    context = resolve(facts)
    assert context["supported_product_choices"] == [ARENA_PRODUCT]
    assert not context["resolved_product"] and context["evaluation"] is None
    assert any(b["code"] == "exact_product_required" for b in context["blockers"])


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_MODE", "json")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setattr(db, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(rival_policy, "call_llm", lambda *args, **kwargs: None)
    monkeypatch.setattr(v1_arena, "SEEDS", 2)
    monkeypatch.setattr(v1_arena, "N_AGENTS", 600)
    with TestClient(app) as test_client:
        yield test_client


def approved_run(client):
    review = client.post(REVIEWS, json={"fields": arena_fields()}).json()
    assert review["status"] == "ready_for_approval", review["blockers"]
    validator("Review").validate(review)
    approval = client.post(f"{REVIEWS}/{review['review_id']}/approvals",
                           json={"revision": review["revision"], "confirmed": True})
    assert approval.status_code == 201
    request = {"approval_id": approval.json()["approval_id"], "submission_id": str(uuid4())}
    accepted = client.post("/api/v1/runs", json=request)
    assert accepted.status_code == 202
    return accepted.json()


def test_arena_run_publishes_a_contract_valid_report(client):
    run = approved_run(client)
    status = client.get(run["status_url"]).json()
    validator("RunState").validate(status)
    assert status["status"] == "done" and status["progress"] == 1

    report = client.get(status["report_url"]).json()
    validator("RunReport").validate(report)
    assert report["data_versions"]["evaluation_policy"] == ARENA_POLICY
    assert [r["scenario"] for r in report["scenario_results"]][:3] == ["S1", "S2", "S3"]
    roles = [plan["role"] for plan in report["plan_results"]]
    assert roles[0] == "original" and roles[-1] == "baseline"
    assert set(roles) <= {"original", "conservative", "baseline"}
    original = report["plan_results"][0]
    assert original["discount_pct"] == pytest.approx(40)
    for scenario in report["scenario_results"]:
        assert len({m["metric"] for m in scenario["metrics"]}) == 6
    assert report["commentary"]

    evaluation = client.get(run["status_url"] + "/evaluation").json()
    validator("BusinessEvaluation").validate(evaluation)
    assert evaluation["checks_unavailable_reason"] is None
    assert {check["scenario"] for check in evaluation["checks"]} >= {"S1", "S2", "S3"}


def test_arena_failure_is_reported_as_a_run_error(client, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("engine crashed")

    monkeypatch.setattr(v1_arena, "evaluate", broken)
    run = approved_run(client)
    status = client.get(run["status_url"]).json()
    validator("RunState").validate(status)
    assert status["status"] == "error" and status["failure"]["retryable"] is True
    assert client.get(run["status_url"] + "/report").status_code == 409


def test_running_state_matches_the_contract(client, monkeypatch):
    from api import v1_runs

    monkeypatch.setattr(v1_runs, "run_arena", lambda run_id: None)  # never finishes
    run = approved_run(client)
    status = client.get(run["status_url"]).json()
    validator("RunState").validate(status)
    assert status["status"] == "queued" and status["poll_after_seconds"] == 2
    response = client.get(run["status_url"] + "/report")
    assert response.status_code == 409 and response.json()["code"] == "report_pending"


def test_arena_outage_cannot_reprice_a_cellox_approval(client, monkeypatch):
    from api import v1_runs

    def outage(*args, **kwargs):
        raise RuntimeError("arena unavailable")

    tried = []
    synthetic_evaluate = v1_runs.evaluate

    def fallback(run_id, approval_id, review):
        tried.append(review)
        return synthetic_evaluate(run_id, approval_id, review)

    monkeypatch.setattr(v1_arena.arena, "run", outage)
    monkeypatch.setattr(v1_runs, "evaluate", fallback)
    run = approved_run(client)
    assert len(tried) == 1
    frozen = client.get(run["status_url"] + "/input").json()["review"]
    assert tried[0] == frozen
    assert frozen["resolved_product"]["choice_id"] == "cellox-tissue-shopee"
    assert frozen["costs"]["regular_price"]["source"] == "data"
    status = client.get(run["status_url"]).json()
    assert status["failure"]["retryable"] and status["status"] == "error"
    assert client.get(run["status_url"] + "/report").status_code == 409

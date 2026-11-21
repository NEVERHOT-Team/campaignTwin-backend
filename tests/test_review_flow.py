from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

import db
from api import v1_arena
from api.v1_contract import validator
from main import app

REVIEWS = "/api/v1/campaign-reviews"
RUNS = "/api/v1/runs"


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("DB_MODE", "json")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setattr(db, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(db, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(v1_arena, "SEEDS", 2)
    monkeypatch.setattr(v1_arena, "N_AGENTS", 600)
    with TestClient(app) as test_client:
        yield test_client


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"brief": " "},
        {"fields": {}},
        {"fields": {"budget_thb": -1}},
        {"fields": {"offer": {"kind": "percent_off", "value": 110, "unit": "pct"}}},
        {"fields": {"offer": {"kind": "fixed_off", "value": 20, "unit": "pct"}}},
    ],
)
def test_invalid_intake(client, body):
    assert client.post(REVIEWS, json=body).status_code == 422


def test_blocked_review_accepts_corrections_only_at_current_revision(client):
    fields = {"product_hint": "unlisted seller SKU", "channel": "shopee", "budget_thb": 500}
    response = client.post(REVIEWS, json={"fields": fields})
    assert response.status_code == 201, response.text
    review = response.json()
    validator("Review").validate(review)
    assert review["facts"] == fields
    assert client.get(f"{REVIEWS}/{review['review_id']}").json() == review
    assert review["status"] == "blocked"
    assert review["blockers"] and review["questions"]
    url = f"{REVIEWS}/{review['review_id']}"
    approval = client.post(f"{url}/approvals", json={"confirmed": True, "revision": 1})
    assert approval.status_code == 409
    unknown = {"base_revision": 1, "confirm": ["never-issued"]}
    assert client.post(f"{url}/revisions", json=unknown).status_code == 422
    edits = {"base_revision": 1, "edits": {"product_hint": None, "budget_thb": 750}}
    response = client.post(f"{url}/revisions", json=edits)
    assert response.status_code == 201, response.text
    assert response.json()["facts"] == {
        "channel": "shopee", "budget_thb": 750, "product_choice_id": "cellox-tissue-shopee",
    }
    assert response.json()["fact_origins"]["/facts/product_choice_id"] == "default"
    assert response.json()["revision"] == 2
    assert client.post(f"{url}/revisions", json=edits).status_code == 409


def test_stage_catalog_defaults_and_market_field_provenance(client):
    review = client.post(REVIEWS, json={"fields": {"budget_thb": 500000}}).json()
    assert [row["choice_id"] for row in review["supported_product_choices"]] == [
        "cellox-tissue-shopee"
    ]
    assert review["resolved_product"] == review["supported_product_choices"][0]
    assert review["fact_origins"]["/facts/product_choice_id"] == "default"
    assert review["fact_origins"]["/facts/channel"] == "derived"
    assert review["fact_origins"]["/facts/budget_thb"] == "planner"
    assert {key: row["source"] for key, row in review["costs"].items()} == {
        "regular_price": "data", "cogs": "default", "channel_commission": "default",
        "payment_fee": "default", "seller_fulfillment": "default", "margin_floor_pct": "default",
    }
    url = f"{REVIEWS}/{review['review_id']}/revisions"
    response = client.post(url, json={"base_revision": 1, "edits": {
        "product_choice_id": "cellox-tissue-shopee", "channel": "shopee",
        "channels": ["shopee", "line"], "event": "11.11",
    }})
    assert response.status_code == 201, response.text
    updated = response.json()
    for key in ("product_choice_id", "channel", "channels", "event"):
        assert updated["fact_origins"][f"/facts/{key}"] == "planner"
    assert updated["facts"]["channels"] == ["shopee", "line"]
    assert updated["facts"]["event"] == "11.11"


@pytest.fixture
def approved(client):
    fields = {
        "product_choice_id": "cellox-tissue-shopee",
        "channel": "shopee",
        "audience_hint": "all",
        "offer": {"kind": "percent_off", "value": 10, "unit": "pct", "eligibility": "all"},
        "objective": "incremental_profit",
        "budget_thb": 1000,
        "start_date": "2026-12-01",
        "end_date": "2026-12-08",
    }
    response = client.post(REVIEWS, json={"fields": fields})
    assert response.status_code == 201, response.text
    review = response.json()
    validator("Review").validate(review)
    assert review["status"] == "ready_for_approval", review["blockers"]
    url = f"{REVIEWS}/{review['review_id']}"
    approval = {"confirmed": True, "revision": 1}
    response = client.post(f"{url}/approvals", json=approval)
    assert response.status_code == 201, response.text
    receipt = response.json()
    validator("Approval").validate(receipt)
    repeat = client.post(f"{url}/approvals", json=approval)
    assert repeat.status_code == 200 and repeat.json() == receipt
    return review, receipt


def test_approved_inputs_survive_revision_and_submission_replay(client, approved):
    review, receipt = approved
    url = f"{REVIEWS}/{review['review_id']}"
    body = {"submission_id": str(uuid4()), "approval_id": receipt["approval_id"]}
    response = client.post(RUNS, json=body)
    assert response.status_code == 202, response.text
    accepted = response.json()
    state = client.get(accepted["status_url"]).json()
    validator("RunState").validate(state)
    assert state["status"] == "done", state
    response = client.get(state["report_url"])
    assert response.status_code == 200
    report = response.json()
    validator("RunReport").validate(report)
    assert report["comparator"] == "paired_no_campaign"
    assert report["review_id"] == review["review_id"]
    assert report["approval_id"] == receipt["approval_id"]
    assert report["revision"] == 1 and report["scenario_results"]
    assert report["data_versions"] == review["data_versions"]
    next_run = client.post(RUNS, json={**body, "submission_id": str(uuid4())})
    assert next_run.status_code == 202
    next_state = client.get(next_run.json()["status_url"]).json()
    next_report = client.get(next_state["report_url"]).json()
    assert report["scenario_results"] == next_report["scenario_results"]

    revision = {"base_revision": 1, "edits": {"budget_thb": 500}}
    assert client.post(f"{url}/revisions", json=revision).status_code == 201
    replay = client.post(RUNS, json=body)
    assert replay.status_code == 200 and replay.json() == accepted
    assert client.post(RUNS, json={**body, "submission_id": str(uuid4())}).status_code == 409
    assert client.get(state["report_url"]).json() == report


def test_concurrent_submission_and_single_retry(client, approved, monkeypatch):
    _, receipt = approved
    body = {"submission_id": str(uuid4()), "approval_id": receipt["approval_id"]}
    evaluate = v1_arena.evaluate

    def unavailable(*args):
        raise RuntimeError("simulated worker interruption")

    monkeypatch.setattr(v1_arena, "evaluate", unavailable)
    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(lambda _: client.post(RUNS, json=body), range(2)))
    assert sorted(response.status_code for response in responses) == [200, 202]
    assert responses[0].json() == responses[1].json()
    accepted = responses[0].json()
    status = client.get(accepted["status_url"]).json()
    assert status["status"] == "error" and status["failure"]["retryable"]
    assert client.get(accepted["status_url"] + "/report").status_code == 409

    monkeypatch.setattr(v1_arena, "evaluate", evaluate)
    retry = {**body, "retry_of_run_id": accepted["run_id"]}
    assert client.post(RUNS, json=retry).status_code == 409
    retry["submission_id"] = str(uuid4())
    response = client.post(RUNS, json=retry)
    assert response.status_code == 202 and response.json() != accepted
    assert client.get(response.json()["status_url"]).json()["status"] == "done"
    assert client.post(RUNS, json=retry).json() == response.json()
    assert client.post(RUNS, json={**retry, "submission_id": str(uuid4())}).status_code == 409


def test_zero_budget_reaches_each_approved_plan_input(client, approved):
    review, _ = approved
    url = f"{REVIEWS}/{review['review_id']}"
    revised = client.post(f"{url}/revisions", json={"base_revision": 1, "edits": {"budget_thb": 0}})
    assert revised.status_code == 201
    receipt = client.post(f"{url}/approvals", json={"revision": 2, "confirmed": True}).json()
    response = client.post(RUNS, json={"approval_id": receipt["approval_id"],
                                       "submission_id": str(uuid4())})
    state = client.get(response.json()["status_url"]).json()
    report = client.get(state["report_url"]).json()
    plans = {plan["role"]: plan for plan in report["plan_results"]}
    assert plans.keys() == {"original", "conservative", "baseline"}
    assert all(plan["input"]["subsidy_budget"] == 0 for plan in plans.values())
    for scenario in plans["baseline"]["scenario_results"]:
        profit = next(m for m in scenario["metrics"] if m["metric"] == "incremental_profit")
        assert profit["value"] == {"p10": 0, "p50": 0, "p90": 0}


def test_brief_conflict_requires_an_explicit_revision(client):
    explicit_offer = {"kind": "percent_off", "value": 25, "unit": "pct"}
    body = {"brief": "Give all buyers 20% off.", "fields": {"offer": explicit_offer}}
    created = client.post(REVIEWS, json=body)
    assert created.status_code == 201, created.text
    review = created.json()
    assert review["facts"]["offer"] == explicit_offer
    offer = next(item for item in review["candidates"] if item["field"] == "/facts/offer")
    candidates = [offer["candidate_id"]]
    url = f"{REVIEWS}/{review['review_id']}/revisions"
    conflict = {"base_revision": 1, "edits": {"offer": explicit_offer}, "confirm": candidates}
    assert client.post(url, json=conflict).status_code == 422
    revised = client.post(url, json={"base_revision": 1, "confirm": candidates})
    assert revised.status_code == 201, revised.text
    response = revised.json()
    assert response["facts"]["offer"]["value"] == 20
    assert response["fact_origins"]["/facts/offer"] == "planner_confirmed_extraction"
    assert all(item["field"] != "/facts/offer" for item in response["candidates"])
    assert client.post(url, json={"base_revision": 2, "confirm": candidates}).status_code == 422


@pytest.mark.parametrize("override", [None, 9.99])
def test_published_voucher_walkthrough(client, override):
    schema = client.get("/openapi.json").json()
    operation = schema["paths"][REVIEWS]["post"]
    content = operation["requestBody"]["content"]["application/json"]
    payload = content["examples"]["syntheticRehearsal"]["value"]
    if override is not None:
        payload["fields"]["offer"]["value"] = override
    response = client.post(REVIEWS, json=payload)
    assert response.status_code == 201, response.text
    review = response.json()
    validator("Review").validate(review)
    url = f"{REVIEWS}/{review['review_id']}"
    response = client.post(f"{url}/approvals", json={"revision": 1, "confirmed": True})
    assert response.status_code == 201, response.text
    receipt = response.json()
    validator("Approval").validate(receipt)
    response = client.post(RUNS, json={"submission_id": str(uuid4()),
                                       "approval_id": receipt["approval_id"]})
    assert response.status_code == 202, response.text
    accepted = response.json()
    validator("RunAccepted").validate(accepted)
    state = client.get(accepted["status_url"]).json()
    validator("RunState").validate(state)
    assert state["status"] == "done", state
    report = client.get(state["report_url"]).json()
    validator("RunReport").validate(report)
    inputs = {plan["role"]: plan["input"] for plan in report["plan_results"]}
    baseline = inputs["baseline"]
    assert baseline["offer"] is None and baseline["subsidy_budget"] == 0
    assert inputs["original"]["offer"]["discount"] == payload["fields"]["offer"]
    assert inputs["original"]["product"]["choice_id"] == "cellox-tissue-shopee"
    assert report["data_versions"]["evaluation_policy"] == "arena-tissue-v1"
    for item in inputs.values():
        assert item["approval_id"] == receipt["approval_id"]
        assert item["review_id"] == review["review_id"] and item["revision"] == 1
        assert item["paired_baseline_id"] == baseline["plan_id"]
        assert item["seeds"] == baseline["seeds"]
        assert item["data_versions"] == review["data_versions"]


@pytest.mark.parametrize("invalid_turn", [
    {"answer": "   "},
    {"answer": "budget 500 THB", "edits": {"budget_thb": 700}},
    {"answer": "budget 500 THB", "confirm": ["unknown"]},
])
def test_invalid_interview_turn_preserves_revision(client, invalid_turn):
    review = client.post(REVIEWS, json={"brief": "Rehearse toilet tissue."}).json()
    url = f"{REVIEWS}/{review['review_id']}"
    response = client.post(f"{url}/revisions", json={"base_revision": 1, **invalid_turn})
    assert response.status_code == 422
    assert client.get(url).json() == review


def test_answer_preserves_confirmed_values_during_provider_outage(client):
    created = client.post(REVIEWS, json={"fields": {"budget_thb": 888, "channel": "shopee"}})
    review = created.json()
    url = f"{REVIEWS}/{review['review_id']}"
    update = {"base_revision": 1, "answer": "Make the discount 20% off."}
    response = client.post(f"{url}/revisions", json=update)
    assert response.status_code == 201, response.text
    revision = response.json()
    validator("Review").validate(revision)
    assert revision["revision"] == 2
    assert revision["agent"]["status"] == "degraded"
    assert revision["facts"] == {
        "budget_thb": 888, "channel": "shopee", "product_choice_id": "cellox-tissue-shopee",
    }
    offer = next(c for c in revision["candidates"] if c["field"] == "/facts/offer")
    assert offer["value"]["value"] == 20 and offer["brief_quote"] in update["answer"]
    assert revision["questions"] and revision["status"] == "blocked"
    assert client.post(f"{url}/revisions", json=update).status_code == 409
    assert client.get(url).json() == revision


def test_minimal_brief_and_free_text_answers_reach_simulation(client):
    created = client.post(REVIEWS, json={"brief": "I want to promote toilet tissue."})
    assert created.status_code == 201
    review = created.json()
    url = f"{REVIEWS}/{review['review_id']}"
    assert review["status"] == "blocked" and review["questions"]
    answer = ("Use the demo for all buyers, target incremental profit, "
              "budget 1000 THB, 10 THB voucher, next week.")
    response = client.post(f"{url}/revisions", json={"base_revision": 1, "answer": answer})
    assert response.status_code == 201, response.text
    proposed = response.json()
    assert proposed["status"] == "blocked" and proposed["candidates"]
    blocked = client.post(f"{url}/approvals", json={"revision": 2, "confirmed": True})
    assert blocked.status_code == 409
    response = client.post(f"{url}/revisions", json={"base_revision": 2, "answer": "confirm all"})
    assert response.status_code == 201, response.text
    ready = response.json()
    validator("Review").validate(ready)
    assert ready["status"] == "ready_for_approval", ready["blockers"]
    assert client.get(url).json() == ready
    receipt = client.post(f"{url}/approvals", json={"revision": 3, "confirmed": True})
    assert receipt.status_code == 201, receipt.text
    accepted = client.post(RUNS, json={"approval_id": receipt.json()["approval_id"],
                                       "submission_id": str(uuid4())})
    assert accepted.status_code == 202, accepted.text
    state = client.get(accepted.json()["status_url"]).json()
    assert state["status"] == "done", state
    report = client.get(state["report_url"]).json()
    validator("RunReport").validate(report)
    original = next(plan["input"] for plan in report["plan_results"] if plan["role"] == "original")
    assert original["product"]["choice_id"] == "cellox-tissue-shopee"
    assert original["subsidy_budget"] == 1000
    assert original["offer"]["discount"] == {
        "kind": "voucher", "value": 10, "unit": "THB", "eligibility": "all",
    }
    start = date.fromisoformat(ready["facts"]["start_date"])
    end = date.fromisoformat(ready["facts"]["end_date"])
    assert start.weekday() == 0 and end - start == timedelta(days=7)
    assert original["campaign_window"] == {
        "start_date": ready["facts"]["start_date"], "end_date": ready["facts"]["end_date"],
    }
    assert original["approval_id"] == receipt.json()["approval_id"]
    run_url = accepted.json()["status_url"]
    frozen = client.get(run_url + "/input").json()
    validator("RunInput").validate(frozen)
    assert frozen["review"] == ready
    evaluation = client.get(run_url + "/evaluation").json()
    validator("BusinessEvaluation").validate(evaluation)
    assert evaluation["decision"] == report["decision"]
    assert evaluation["approval_id"] == frozen["approval_id"]


def test_ai_interprets_reply_in_review_and_requires_confirmation(client, monkeypatch):
    import json

    import httpx
    from openai import OpenAI

    from agents import llm

    calls = []

    def check_strict_objects(node):
        if isinstance(node, list):
            for child in node:
                check_strict_objects(child)
        if isinstance(node, dict):
            if node.get("type") == "object":
                assert node.get("additionalProperties") is False
                assert set(node["required"]) == set(node["properties"])
            for child in node.values():
                check_strict_objects(child)

    def completion(request):
        body = json.loads(request.content)
        response_format = body["response_format"]["json_schema"]
        assert response_format["strict"]
        check_strict_objects(response_format["schema"])
        name = response_format["name"]
        calls.append((name, body["messages"]))
        context = json.loads(body["messages"][-1]["content"])
        if name == "Interview":
            fields = [q["field"] for q in context["questions"]]
            field = "/facts/budget_thb" if "/facts/budget_thb" in fields else fields[0]
            answer = {"first_field": field, "question": "Can you confirm this campaign input?"}
        elif "cap it at a grand" in context["message"]:
            assert context["pending"]
            proposals = [
                ("budget_thb", 1000, "cap it at a grand"),
                ("audience_hint", "all", "all buyers"),
                ("objective", "incremental_profit", "incremental profit"),
                ("offer", {"kind": "voucher", "unit": "THB", "value": 10}, "10 THB voucher"),
                ("start_date", "2026-12-01", "2026-12-01"),
                ("end_date", "2026-12-08", "2026-12-08"),
            ]
            answer = {"items": [{"field": field, "value_json": json.dumps(value), "quote": quote}
                                for field, value, quote in proposals]}
        else:
            answer = {"items": [{"field": "product_hint",
                                 "value_json": json.dumps("toilet tissue"),
                                 "quote": "toilet tissue"}]}
        return httpx.Response(200, json={
            "id": "test-completion", "object": "chat.completion", "created": 1,
            "model": body["model"], "choices": [{"index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": json.dumps(answer)}}],
        })

    def local_client(**options):
        transport = httpx.MockTransport(completion)
        return OpenAI(**options, http_client=httpx.Client(transport=transport))

    monkeypatch.setenv("OPENAI_API_KEY", "test-only")
    monkeypatch.setenv("MODEL_FAST", "test-model")
    monkeypatch.setattr(llm, "OpenAI", local_client)
    initial = client.post(REVIEWS, json={"brief": "Rehearse toilet tissue."}).json()
    url = f"{REVIEWS}/{initial['review_id']}"
    answer = ("Please cap it at a grand. Offer a 10 THB voucher to all buyers for incremental "
              "profit, from 2026-12-01 until 2026-12-08. Reach me at planner@example.test.")
    revised = client.post(f"{url}/revisions", json={"base_revision": 1, "answer": answer})
    assert revised.status_code == 201, revised.text
    review = revised.json()
    assert review["agent"]["status"] == "completed"
    assert "budget_thb" not in review["facts"]
    candidate = next(c for c in review["candidates"] if c["field"] == "/facts/budget_thb")
    assert candidate["value"] == 1000
    assert candidate["brief_quote"] == "cap it at a grand"
    assert {"Interpretation", "Interview"} <= {name for name, _ in calls}
    assert "planner@example.test" not in json.dumps(calls)
    response = client.post(f"{url}/revisions", json={
        "base_revision": 2, "answer": "confirm all",
    })
    assert response.status_code == 201, response.text
    ready = response.json()
    assert ready["status"] == "ready_for_approval", ready["blockers"]
    receipt = client.post(f"{url}/approvals", json={"revision": 3, "confirmed": True}).json()
    accepted = client.post(RUNS, json={"approval_id": receipt["approval_id"],
                                       "submission_id": str(uuid4())}).json()
    status = client.get(accepted["status_url"]).json()
    assert status["status"] == "done"
    report = client.get(status["report_url"]).json()
    validator("RunReport").validate(report)
    original = next(plan["input"] for plan in report["plan_results"] if plan["role"] == "original")
    assert original["subsidy_budget"] == 1000 and original["offer"]["discount"]["value"] == 10


def test_ai_cannot_invent_reply_evidence(client, monkeypatch):
    from agents import llm

    def provider(model, messages, schema):
        if schema.__name__ == "Interview":
            return schema(first_field="/facts/budget_thb", question="How much can you spend?")
        return schema(items=[{"field": "budget_thb", "value_json": "99999",
                              "quote": "a quote absent from the answer"}])

    monkeypatch.setenv("OPENAI_API_KEY", "test-only")
    monkeypatch.setenv("MODEL_FAST", "test-model")
    monkeypatch.setattr(llm, "_provider_call", provider)
    review = client.post(REVIEWS, json={"fields": {"channel": "shopee"}}).json()
    url = f"{REVIEWS}/{review['review_id']}/revisions"
    response = client.post(url, json={"base_revision": 1, "answer": "I need a voucher."})
    assert response.status_code == 201, response.text
    revised = response.json()
    assert revised["agent"]["status"] == "degraded"
    assert "budget_thb" not in revised["facts"]
    assert all(c["field"] != "/facts/budget_thb" for c in revised["candidates"])


def test_late_interview_answer_cannot_overwrite_a_newer_revision(client, monkeypatch):
    from threading import Event

    from agents import llm

    started = Event()
    release = Event()

    def slow_interpretation(model, messages, schema):
        if schema.__name__ == "Interpretation":
            started.set()
            assert release.wait(5)
        return None

    review = client.post(REVIEWS, json={"fields": {"budget_thb": 500}}).json()
    url = f"{REVIEWS}/{review['review_id']}"
    monkeypatch.setattr(llm, "call_llm", slow_interpretation)
    with ThreadPoolExecutor(max_workers=1) as pool:
        reply = pool.submit(client.post, f"{url}/revisions",
                            json={"base_revision": 1, "answer": "10 THB voucher"})
        try:
            assert started.wait(5)
            edit = client.post(f"{url}/revisions", json={
                "base_revision": 1, "edits": {"budget_thb": 600},
            })
            assert edit.status_code == 201
        finally:
            release.set()
        assert reply.result(timeout=5).status_code == 409
    assert client.get(url).json() == edit.json()


def test_requested_platform_remains_visible_and_blocks_incompatible_simulation(client):
    created = client.post(REVIEWS, json={"brief": "Use toilet tissue on Shopify, not Shopee."})
    assert created.status_code == 201
    review = created.json()
    url = f"{REVIEWS}/{review['review_id']}"
    confirmed = client.post(f"{url}/revisions", json={"base_revision": 1, "answer": "confirm all"})
    assert confirmed.status_code == 201, confirmed.text
    review = confirmed.json()
    assert review["facts"]["channel"] == "shopify"
    assert any(b["field"] == "/facts/channel" and b["code"] == "unsupported_choice"
               for b in review["blockers"])
    assert client.post(
        f"{url}/approvals", json={"revision": 2, "confirmed": True}
    ).status_code == 409
    switched = client.post(f"{url}/revisions", json={
        "base_revision": 2, "edits": {"channel": " Shopee "},
    })
    assert switched.status_code == 201
    assert switched.json()["facts"]["channel"] == "shopee"
    assert not any(b["field"] == "/facts/channel" for b in switched.json()["blockers"])


@pytest.mark.parametrize("invalid", [
    {"weekly_buy_rate": 1.01}, {"share": float("nan")}, {"households": 100_001},
    {"households": 50.5}, {"name": ""},
])
def test_unusable_market_context_blocks_approval(client, monkeypatch, invalid):
    from api import v1_catalog

    row = {"name": "buyers", "households": 120, "share": 1.0, "weekly_buy_rate": 0.2}
    monkeypatch.setattr(v1_catalog, "load_market", lambda: {"segments": [{**row, **invalid}]})
    response = client.post(REVIEWS, json={"fields": {"product_choice_id": "cellox-tissue-shopee"}})
    assert response.status_code == 201
    review = response.json()
    validator("Review").validate(review)
    assert review["segments"] == []
    assert review["data_versions"]["segments"] is None
    assert any(b["field"] == "/segments" for b in review["blockers"])

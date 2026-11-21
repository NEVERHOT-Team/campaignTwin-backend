"""Seller journeys through the opt-in application, with isolated storage and no provider key."""

from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

import db
from api.v2.app import REVIEWS, app
from api.v2.contract import contract, validator
from main import app as existing_app


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_MODE", "json")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.delenv("CAMPAIGNTWIN_EVIDENCE_SNAPSHOT", raising=False)
    monkeypatch.setattr(db, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(db, "CACHE_DIR", tmp_path / "cache")
    with TestClient(app) as client:
        yield client


def example(name):
    content = contract()["paths"][REVIEWS]["post"]["requestBody"]["content"]
    return content["application/json"]["examples"][name]["value"]["fields"].copy()


def start(api, **body):
    response = api.post(REVIEWS, json=body)
    assert response.status_code == 201, response.text
    return response.json()


def turn(api, session, **body):
    response = api.post(
        session["links"]["turn"], json={"base_revision": session["review"]["revision"], **body}
    )
    assert response.status_code == 201, response.text
    validator("Review").validate(response.json()["review"])
    return response.json()


def test_entry_points_and_legacy_compatibility(api):
    old = TestClient(existing_app)
    assert old.get("/api/v2").status_code == 404
    assert api.get("/openapi.json").json() == old.get("/openapi.json").json()
    info = api.get("/api/v2").json()
    schema = api.get(info["schema"]).json()
    assert info["create_review"] in schema["paths"]
    assert REVIEWS + "/{review_id}/turns" in schema["paths"]
    assert REVIEWS + "/{review_id}/what-if" in schema["paths"]
    assert api.get(info["docs"]).status_code == 200
    session = start(api, **info["example"])
    validator("Session").validate(session)
    assert session["review"]["facts"]["end_date"] == "2026-12-08"
    assert old.get("/api/v1/campaign-reviews/" + session["review"]["review_id"]).status_code == 404
    v1 = old.post("/api/v1/campaign-reviews", json={"fields": {"budget_thb": 0}}).json()
    assert api.get(REVIEWS + "/" + v1["review_id"]).status_code == 404


def test_rice_from_three_seller_inputs_through_real_simulation(api):
    session = start(
        api,
        fields={"budget_thb": 2500},
        selling_period={"first_day": "2026-12-01", "last_day": "2026-12-07"},
    )
    session = turn(api, session, answer="Rice")
    assert session["next_step"]["choices"][0]["label"] == "Rice"
    assert session["review"]["resolved_product"] is None
    session = turn(api, session, answer="yes")
    assert session["review"]["confirmed_quotes"]["/facts/product_choice_id"] == "Rice"
    assert session["review"]["facts"]["channel"] == "shopee"
    for choice in ["all-buyers", "profit", "catalog-voucher", "break-even"]:
        session = turn(api, session, choice=choice)
    review = session["review"]
    assert session["next_step"]["mode"] == "review"
    assert session["unit_economics"]["ordinary_contribution_thb"] == 34
    assert session["unit_economics"]["redeemed_contribution_thb"] == 15
    costs = {c["field"]: c for c in session["cost_inputs"]}
    assert costs["commission_pct"]["evidence_ref"] == "fixture-cost-sheet"
    assert review["evaluation"]["alternative_policy"] == "fixed_offer"
    assert api.get(session["links"]["self"]).json() == session
    receipt = api.post(
        session["links"]["approve"], json={"revision": review["revision"], "confirmed": True}
    ).json()
    accepted = api.post(
        session["links"]["run"],
        json={"approval_id": receipt["approval_id"], "submission_id": str(uuid4())},
    )
    assert accepted.status_code == 202
    status = api.get(accepted.json()["status_url"]).json()
    assert status["status"] == "done", status
    report = api.get(status["report_url"]).json()
    assert {p["role"] for p in report["plan_results"]} == {"original", "baseline"}


def test_blank_start_conflicts_and_contextual_amount(api):
    session = start(api)
    assert session["next_step"]["field"] == "/facts/product_choice_id"
    links = session["links"]
    for body in [{"answer": "yes"}, {"choice": "absent"}, {"edits": {"unknown": 1}}]:
        assert api.post(links["turn"], json={"base_revision": 1, **body}).status_code == 422
    assert api.get(links["self"]).json() == session
    fields = example("syntheticRehearsal")
    del fields["budget_thb"]
    session = turn(api, session, edits=fields)
    assert session["next_step"]["field"] == "/facts/budget_thb"
    session = turn(api, session, answer="1,000 THB")
    assert "budget_thb" not in session["review"]["facts"]
    assert session["next_step"]["choices"][0]["value"] == 1000
    session = turn(api, session, choice=session["next_step"]["choices"][0]["id"])
    assert session["review"]["facts"]["budget_thb"] == 1000
    session = turn(api, session, answer="Our budget is 500 THB.")
    assert session["review"]["facts"]["budget_thb"] == 1000
    assert session["next_step"]["mode"] == "confirm"
    assert (
        api.post(
            links["approve"], json={"revision": session["review"]["revision"], "confirmed": True}
        ).status_code
        == 409
    )
    session = turn(api, session, edits={"budget_thb": 0})
    assert session["unit_economics"]["max_funded_redemptions"] == 0
    assert "/facts/budget_thb" not in session["review"].get("confirmed_quotes", {})


def test_hand_calculation_and_what_if_isolation(api):
    fields = dict(
        example("syntheticRehearsal"),
        cogs_thb=55,
        order_fee_thb=5,
        paid_media_thb=100,
        profit_target_thb=25,
        budget_thb=1500,
    )
    session = start(api, fields=fields)
    case = {"base_revision": 1, "redeemed_orders": 120, "additional_orders": 20}
    url = session["links"]["what_if"]
    result = api.post(url, json=case).json()
    assert result["campaign_spend_thb"] == 1300
    assert result["incremental_profit_thb"] == -500
    assert result["minimum_additional_orders_for_target"] == 34
    for n, expected in [(33, False), (34, True)]:
        assert (
            api.post(url, json={**case, "additional_orders": n}).json()["meets_profit_target"]
            is expected
        )
    for changes in [
        {"redeemed_orders": 141},
        {"additional_orders": 121},
        {"additional_orders": -1},
        {"additional_orders": True},
        {"additional_orders": 2.5},
    ]:
        assert api.post(url, json={**case, **changes}).status_code == 422
    assert api.get(session["links"]["self"]).json() == session
    revised = turn(
        api, session, edits={"offer": {"kind": "percent_off", "unit": "pct", "value": 5}}
    )
    assert revised["unit_economics"]["redeemed_contribution_thb"] == 35
    assert api.post(url, json=case).status_code == 409
    result = api.post(
        url, json={"base_revision": 2, "redeemed_orders": 118, "additional_orders": 18}
    ).json()
    assert result["incremental_profit_thb"] == 30
    assert result["meets_profit_target"]


def test_provider_failure_and_preserving_terms(api):
    fields = example("syntheticRehearsal")
    fields["offer"] = {**fields["offer"], "min_spend_thb": 50}
    session = start(api, fields=fields)
    session = turn(api, session, answer="Change the voucher to a 5 THB voucher.")
    proposal = session["review"]["candidates"][0]
    assert proposal["value"]["min_spend_thb"] == 50
    assert session["review"]["agent"]["status"] == "degraded"
    failed = api.post(session["links"]["turn"], json={"base_revision": 2, "answer": "???"})
    assert failed.status_code == 422 and failed.json()["code"] == "answer_unresolved"
    assert api.get(session["links"]["self"]).json() == session
    assert (
        api.post(
            session["links"]["what_if"],
            json={"base_revision": 2, "redeemed_orders": 0, "additional_orders": 0},
        ).status_code
        == 409
    )


def test_race_and_stale_writes(api):
    session = start(api, fields=example("syntheticRehearsal"))
    url = session["links"]["turn"]
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(
                lambda v: api.post(url, json={"base_revision": 1, "edits": {"cogs_thb": v}}),
                [0, 55],
            )
        )
    assert sorted(r.status_code for r in results) == [201, 409]
    latest = api.get(session["links"]["self"]).json()
    assert latest["review"]["revision"] == 2
    costs = {c["field"]: c for c in latest["cost_inputs"]}
    assert costs["cogs_thb"]["origin"] == "planner"


@pytest.mark.parametrize(
    "period",
    [
        {"first_day": "2026-12-08", "last_day": "2026-12-01"},
        {"first_day": "2026-01-01", "last_day": "2026-02-12"},
        {"first_day": "9999-12-31", "last_day": "9999-12-31"},
    ],
)
def test_invalid_selling_period(api, period):
    assert api.post(REVIEWS, json={"selling_period": period}).status_code == 422


def test_nonpositive_margin_and_unreachable_target(api):
    session = start(api, fields={**example("syntheticRehearsal"), "cogs_thb": 100})
    result = api.post(
        session["links"]["what_if"],
        json={"base_revision": 1, "redeemed_orders": 10, "additional_orders": 10},
    ).json()
    assert result["incremental_profit_thb"] == -100
    assert result["minimum_additional_orders_for_target"] is None
    assert result["target_reachable_at_this_volume"] is False


def test_profit_target_and_permission_to_change_discount(api):
    session = start(api, fields=example("syntheticRehearsal"))
    assert session["next_step"]["field"] == "/commercial_inputs/profit_target_thb"
    session = turn(api, session, answer="250 THB")
    session = turn(api, session, answer="yes")
    assert session["review"]["evaluation"]["goal_target"]["minimum"] == 250
    assert session["next_step"]["field"] == "/commercial_inputs/alternative_policy"
    session = turn(api, session, choice="fixed")
    assert session["next_step"]["mode"] == "review"
    assert session["review"]["evaluation"]["alternative_policy"] == "fixed_offer"
    receipt = api.post(session["links"]["approve"], json={
        "revision": session["review"]["revision"], "confirmed": True
    })
    assert receipt.status_code == 201
    revised = turn(api, session, edits={"profit_target_thb": None})
    assert revised["next_step"]["field"] == "/commercial_inputs/profit_target_thb"
    assert api.post(session["links"]["run"], json={
        "approval_id": receipt.json()["approval_id"], "submission_id": str(uuid4())
    }).status_code == 409


def test_submission_replay_and_reports_stay_in_their_own_version(api):
    """Replay a saved v1 run after a v2 trial using the same client submission UUID."""
    from api.v1_contract import validator as legacy_validator

    legacy_reviews = "/api/v1/campaign-reviews"
    fields = example("syntheticRehearsal")
    commercial = {**fields, "order_fee_thb": 2, "profit_target_thb": 25}
    assert api.post(legacy_reviews, json={"fields": commercial}).status_code == 422
    v1_fields = {**fields, "product_choice_id": "cellox-tissue-shopee"}
    v1 = api.post(legacy_reviews, json={"fields": v1_fields}).json()
    legacy_validator("Review").validate(v1)
    consent = {"revision": 1, "confirmed": True}
    receipt1 = api.post(f"{legacy_reviews}/{v1['review_id']}/approvals", json=consent).json()
    submission_id = str(uuid4())
    request1 = {"approval_id": receipt1["approval_id"], "submission_id": submission_id}
    accepted1 = api.post("/api/v1/runs", json=request1)
    assert accepted1.status_code == 202
    status1 = accepted1.headers["Location"]
    report1 = api.get(status1 + "/report").json()
    legacy_validator("RunReport").validate(report1)
    assert report1["data_versions"]["evaluation_policy"] == "arena-tissue-v1"
    assert "recommendation" not in report1
    stored1 = (db.RUNS_DIR / "v1-workspace.json").read_bytes()

    session = start(api, fields=commercial)
    validator("Session").validate(session)
    assert api.post(session["links"]["run"], json=request1).status_code == 404
    assert api.post(f"{REVIEWS}/{v1['review_id']}/approvals", json=consent).status_code == 404
    receipt2 = api.post(session["links"]["approve"], json=consent).json()
    request2 = {"approval_id": receipt2["approval_id"], "submission_id": submission_id}
    accepted2 = api.post(session["links"]["run"], json=request2)
    assert accepted2.status_code == 202
    status2 = accepted2.headers["Location"]
    assert status2 == accepted2.json()["status_url"] and status2.startswith("/api/v2/runs/")
    assert accepted1.json()["run_id"] != accepted2.json()["run_id"]
    next_client = TestClient(app)
    for version, request, expected in [(1, request1, accepted1), (2, request2, accepted2)]:
        replay = next_client.post(f"/api/v{version}/runs", json=request)
        assert replay.status_code == 200
        assert replay.json() == expected.json()
    for suffix in ("", "/report", "/input", "/evaluation", "/historical-comparison"):
        assert next_client.get(status2.replace("/v2/", "/v1/") + suffix).status_code == 404
        assert next_client.get(status1.replace("/v1/", "/v2/") + suffix).status_code == 404
    report_url = next_client.get(status2).json()["report_url"]
    assert report_url == status2 + "/report"
    report2 = next_client.get(report_url).json()
    assert report2["data_versions"]["evaluation_policy"] == "synthetic-voucher-v2"
    assert report2["evidence"]["historical_comparison_url"] == status2 + "/historical-comparison"
    assert next_client.get(status2 + "/evaluation").json()["recommendation"] == (
        report2["recommendation"]
    )
    assert next_client.get(status1 + "/report").json() == report1
    assert (db.RUNS_DIR / "v1-workspace.json").read_bytes() == stored1


def test_v2_schema_matches_routes_sessions_and_errors(api):
    from fastapi.openapi.utils import get_openapi

    from api.v2.service import app as seller_api

    schema = api.get("/api/v2/openapi.json").json()
    runtime = get_openapi(title="Runtime", version="2", routes=seller_api.routes)
    assert set(schema["paths"]) == {"/api/v2" + p.rstrip("/") for p in runtime["paths"]}
    assert schema["servers"] == [{"url": "/"}]
    for path, op in runtime["paths"].items():
        published = schema["paths"]["/api/v2" + path.rstrip("/")]
        assert set(op) == set(published) - {"parameters"}
    session = start(api)
    validator("Start").validate(api.get("/api/v2").json()["example"])
    validator("Session").validate(session)
    step = api.post(session["links"]["turn"], json={"base_revision": 1, "choice": "absent"})
    assert step.status_code == 422
    validator("Error").validate(step.json())
    case = api.post(
        session["links"]["what_if"],
        json={"base_revision": 1, "redeemed_orders": 1, "additional_orders": 0},
    )
    assert case.status_code == 409
    validator("Error").validate(case.json())


def test_combined_server_keeps_dev_arena_choice_on_v1(api):
    path = "/api/v1/campaign-reviews"
    schema = api.get("/openapi.json").json()
    content = schema["paths"][path]["post"]["requestBody"]["content"]["application/json"]
    payload = content["examples"]["arenaRehearsal"]["value"]
    old = api.post(path, json=payload)
    assert old.status_code == 201
    arena = old.json()
    assert arena["status"] == "ready_for_approval"
    assert arena["evaluation"]["scenarios"] == ["S1", "S2", "S3", "S4"]

    # v1 promotional channels/event are not v2 seller intake fields.
    assert api.post(REVIEWS, json=payload).status_code == 422
    common = {k: v for k, v in payload["fields"].items() if k not in {"channels", "event"}}
    seller = start(api, fields=common)
    assert seller["review"]["status"] == "blocked"
    assert seller["review"]["resolved_product"] is None
    assert api.get(REVIEWS + "/" + arena["review_id"]).status_code == 404


def test_campaign_outlay_has_a_finite_wire_representation(api):
    # Use the same campaign outlay range as committed media. Unbounded integers can
    # otherwise turn into null in a successful numeric response or fail at simulation.
    fields = example("syntheticRehearsal")
    assert api.post(REVIEWS, json={"fields": {**fields, "budget_thb": 10**311}}).status_code == 422
    session = start(api, fields={**fields, "budget_thb": 10**12})
    validator("Session").validate(session)
    response = api.post(session["links"]["turn"], json={
        "base_revision": 1, "edits": {"budget_thb": 10**12 + 1}
    })
    assert response.status_code == 422
    assert api.get(session["links"]["self"]).json() == session

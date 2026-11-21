"""Regression cases for the units and commitments a seller actually stated."""

import pytest
from fastapi.testclient import TestClient

import db
from agents import review_v2
from api.v2.app import REVIEWS, app
from api.v2.contract import contract, validator


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_MODE", "json")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setattr(db, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(db, "CACHE_DIR", tmp_path / "cache")
    with TestClient(app) as client:
        yield client


def ready_fields():
    content = contract()["paths"][REVIEWS]["post"]["requestBody"]["content"]["application/json"]
    return content["examples"]["syntheticRehearsal"]["value"]["fields"].copy()


@pytest.mark.parametrize("amount", ["100 USD", "EUR 100", "$99", "1.5k THB"])
def test_unresolved_monetary_unit_cannot_change_a_saved_budget(api, amount):
    session = api.post(REVIEWS, json={"fields": ready_fields()}).json()
    response = api.post(
        session["links"]["turn"], json={"base_revision": 1, "answer": f"Our budget is {amount}."}
    )
    assert response.status_code == 422
    validator("Error").validate(response.json())
    assert api.get(session["links"]["self"]).json() == session
    assert api.post(REVIEWS, json={"brief": f"Our budget is {amount}."}).status_code == 422


@pytest.mark.parametrize("model_misreads_risk", [False, True])
def test_risk_and_commitment_survive_extraction_and_approval(api, monkeypatch, model_misreads_risk):
    if model_misreads_risk:

        def ask(model, messages, schema):
            if schema is review_v2.Interpretation:
                return schema(
                    items=[
                        {
                            "field": "offer",
                            "quote": "20%",
                            "value_json": '{"kind":"percent_off","unit":"pct","value":20}',
                        }
                    ]
                )
            return None

        monkeypatch.setattr(review_v2.llm, "call_llm", ask)
    brief = "Do not change the voucher. The maximum chance of loss is 20%."
    fields = ready_fields()
    session = api.post(REVIEWS, json={"fields": fields, "brief": brief}).json()
    candidates = session["review"]["candidates"]
    assert {c["field"]: c["value"] for c in candidates} == {
        "/commercial_inputs/alternative_policy": "fixed_offer",
        "/commercial_inputs/max_loss_probability": 0.2,
    }
    assert (
        api.post(session["links"]["approve"], json={"revision": 1, "confirmed": True}).status_code
        == 409
    )
    confirmations = [c["candidate_id"] for c in candidates]
    response = api.post(
        session["links"]["turn"], json={"base_revision": 1, "confirm": confirmations}
    )
    assert response.status_code == 201
    reviewed = response.json()["review"]
    assert reviewed["facts"]["offer"] == fields["offer"]
    assert reviewed["evaluation"]["alternative_policy"] == "fixed_offer"
    assert reviewed["evaluation"]["max_loss_probability"] == 0.2
    assert all(quote in brief for quote in reviewed["confirmed_quotes"].values())
    assert (
        api.post(session["links"]["approve"], json={"revision": 2, "confirmed": True}).status_code
        == 201
    )


def test_percentage_voucher_has_one_declared_unit(api):
    session = api.post(REVIEWS, json={"brief": "Voucher of 10%."}).json()
    offers = [c["value"] for c in session["review"]["candidates"] if c["field"] == "/facts/offer"]
    assert offers == [{"kind": "percent_off", "unit": "pct", "value": 10}]


def test_fixed_catalog_strategy_points_to_its_pinned_evidence(api):
    content = contract()["paths"][REVIEWS]["post"]["requestBody"]["content"]["application/json"]
    fields = content["examples"]["evidenceVoucher"]["value"]["fields"]
    session = api.post(REVIEWS, json={"fields": fields}).json()
    policy = next(p for p in session["policy_inputs"] if p["field"] == "alternative_policy")
    assert policy["value"] == "fixed_offer"
    bundle = session["review"]["catalog_evidence"]
    assert policy["evidence_ref"] == bundle["capability"]["evidence_ref"]
    assert any(
        source["evidence_ref"] == policy["evidence_ref"] and source["label"] == policy["origin"]
        for source in bundle["sources"]
    )


def test_revision_reply_uses_question_actually_displayed(api):
    session = api.post(REVIEWS, json={"fields": ready_fields()}).json()
    assert session["next_step"]["field"] == "/commercial_inputs/profit_target_thb"
    revisions = session["links"]["self"] + "/revisions"
    response = api.post(revisions, json={"base_revision": 1, "answer": "250 THB"})
    assert response.status_code == 201
    suggestion = response.json()["next_step"]
    assert suggestion["mode"] == "confirm"
    assert suggestion["choices"][0]["value"] == 250
    confirmed = api.post(revisions, json={"base_revision": 2, "answer": "yes"})
    assert confirmed.status_code == 201
    assert confirmed.json()["review"]["evaluation"]["goal_target"]["minimum"] == 250



@pytest.mark.parametrize("day", [0, "0", "2026-12-01T00:00:00Z"])
def test_local_selling_period_rejects_timestamps(api, day):
    response = api.post(REVIEWS, json={"selling_period": {"first_day": day, "last_day": day}})
    assert response.status_code == 422
    validator("Error").validate(response.json())

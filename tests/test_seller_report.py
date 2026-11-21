"""Commercial acceptance through the opt-in v2 seller API."""

from copy import deepcopy
from decimal import Decimal
from threading import Event, Thread
from uuid import uuid4

import numpy as np
import pytest
from fastapi.testclient import TestClient

import db
from api.v2.app import app
from api.v2.contract import contract, validator
from engine.seller_ledger import Settlement
from engine.seller_simulation import compare_plans

REVIEWS = "/api/v2/campaign-reviews"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_MODE", "json")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setattr(db, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(db, "CACHE_DIR", tmp_path / "cache")
    return TestClient(app)


def fields(**edits):
    body = contract()["paths"][REVIEWS]["post"]["requestBody"]["content"]["application/json"]
    return {**body["examples"]["syntheticRehearsal"]["value"]["fields"], **edits}


def approve(client, **settings):
    review = client.post(REVIEWS, json={"fields": fields(**settings)}).json()["review"]
    assert review["status"] == "ready_for_approval", review["blockers"]
    validator("Review").validate(review)
    url = f"{REVIEWS}/{review['review_id']}/approvals"
    return client.post(url, json={"confirmed": True, "revision": 1}).json()["approval_id"]


def reports(client, **settings):
    approval = approve(client, **settings)
    body = {"approval_id": approval, "submission_id": str(uuid4())}
    accepted = client.post("/api/v2/runs", json=body)
    assert accepted.status_code == 202
    url = accepted.json()["status_url"]
    report = client.get(url + "/report").json()
    scorecard = client.get(url + "/evaluation").json()
    validator("RunReport").validate(report)
    validator("BusinessEvaluation").validate(scorecard)
    assert scorecard["decision"] == report["decision"]
    assert scorecard["recommendation"] == report["recommendation"]
    assert scorecard["evidence"] == report["evidence"]
    for view, plan in zip(scorecard["comparisons"], report["plan_results"], strict=True):
        assert view["plan_id"] == plan["input"]["plan_id"]
        assert view["scenario_results"] == plan["scenario_results"]
    return report


def metrics(result):
    return {m["metric"]: m["value"] for m in result["metrics"]}


def test_srs_hand_calculated_seller_accounts():
    d = Decimal
    terms = Settlement(d(100), d(55), d(0), d(5), d(0), d(0), d(100))
    baseline = terms.ledger(100, 0, d(0), d(0))
    assert baseline["contribution_profit"] == 4000
    for units, discount, receipts, spend, contribution, media_return in (
        (120, 10, 10800, 1300, 3500, 8),
        (118, 5, 11210, 690, 4030, 12.1),
    ):
        row = terms.ledger(units, units, d(discount), d(100))
        assert (row["net_receipts"], row["campaign_spend"], row["contribution_profit"]) == (
            receipts,
            spend,
            contribution,
        )
        assert (row["net_receipts"] - baseline["net_receipts"]) / 100 == d(str(media_return))


@pytest.mark.parametrize("fee,status", [(2, "revise"), (7, "hold")])
def test_known_seller_fee_changes_eligibility(client, fee, status):
    report = reports(
        client,
        order_fee_thb=fee,
        profit_target_thb=25,
        min_unit_contribution_thb=32,
        discount_response=8,
    )
    ranking = report["recommendation"]
    assert ranking["status"] == status
    for item, discount in zip(ranking["assessments"], (10, 5), strict=True):
        floor = next(g for g in item["checks"] if g["metric"] == "unit_contribution")
        assert floor["observed"] == 100 - 60 - fee - discount
    assert ranking["selected_plan_id"] == (
        report["plan_results"][1]["input"]["plan_id"] if fee == 2 else None
    )


def test_cross_day_leakage_and_realized_returns(client):
    report = reports(client)
    result = report["scenario_results"][0]
    values = metrics(result)
    assert values["incremental_roas"] is None
    assert values["platform_roas"] is None
    assert values["incremental_new_buyers"] is None
    assert values["campaign_iroi"]["p50"] == pytest.approx(-0.6)
    # This fixture's same-day overlap is 90%. Reconstruct campaign-window buyers from
    # the paired draws independently, including baseline days after the budget runs out.
    segments = report["plan_results"][0]["input"]["segments"]
    rates = np.concatenate(
        [
            np.full(s["households"], 1 - (1 - s["weekly_purchase_probability"]) ** (1 / 7))
            for s in segments
        ]
    )
    for ledger in result["seed_ledgers"]:
        seed = ledger["seed"]
        draws = np.random.default_rng(seed).random((7, len(rates)))
        bought = set(np.where((draws < rates).any(axis=0))[0])
        redeemed = []
        for day, day_draws in enumerate(draws):
            orders = np.where(day_draws < rates * 1.1)[0]
            redeemed.extend(
                np.random.default_rng([seed, day, 1]).permutation(orders)[: 100 - len(redeemed)]
            )
        assert ledger["overlapping_subsidy"] == 10 * sum(idx in bought for idx in redeemed)
    assert values["subsidy_leakage"]["p50"] == pytest.approx(0.92)


def test_campaign_windows_media_stress_and_pairing(client):
    report = reports(
        client,
        paid_media_thb=100,
        discount_response=10,
        stockup_days=14,
        rival_demand_reduction_pct=30,
    )
    plans = report["plan_results"]
    assert report["evidence"]["scenario_basis"][1]["likelihood"] is None
    for plan in plans:
        for s, baseline in zip(
            plan["scenario_results"], plans[-1]["scenario_results"], strict=True
        ):
            assert s["paired_seed_count"] == 8
            assert [w["phase"] for w in s["weekly"]] == ["campaign", "post", "post"]
            assert s["weekly"][-1]["end_date"] == report["measurement_window"]["end_date"]
            m = metrics(s)
            assert (
                metrics(s["weekly"][-1])["cumulative_incremental_profit"] == m["incremental_profit"]
            )
            assert [r["seed"] for r in s["seed_ledgers"]] == list(range(8))
            for row, ref in zip(s["seed_ledgers"], baseline["seed_ledgers"], strict=True):
                assert row["campaign_spend"] <= plan["input"]["total_budget_thb"]
                assert row["incremental_units"] == row["units"] - ref["units"]
                assert (
                    row["incremental_profit"]
                    == row["contribution_profit"] - ref["contribution_profit"]
                )
            if plan["role"] != "baseline":
                assert m["post_incremental_units"]["p50"] < 0
                assert m["incremental_roas"]["p50"] == np.median(
                    [r["incremental_net_receipts"] / 100 for r in s["seed_ledgers"]]
                )
    corrupted = deepcopy(plans)
    corrupted[1]["scenario_results"][0]["seed_ledgers"].pop()
    assert compare_plans(corrupted)["reason"] == "incomplete_pairing"
    corrupted = deepcopy(plans)
    corrupted[1]["input"]["currency"] = "USD"
    assert compare_plans(corrupted)["status"] == "insufficient_evidence"
    for plan in plans:
        plan["input"]["paired_baseline_id"] = "missing/baseline"
    assert compare_plans(plans)["status"] == "insufficient_evidence"


def test_catalog_selection_pins_evidence_and_minimum_spend(client):
    settings = {
        "product_choice_id": "fixture-th-shopee-rice-fixed-voucher",
        "offer": {
            "unit": "THB",
            "kind": "voucher",
            "value": 20,
            "eligibility": "all",
            "min_spend_thb": 120,
        },
        "post_days": 7,
    }
    report = reports(client, **settings, min_unit_contribution_thb=10)
    url = f"/api/v2/runs/{report['run_id']}"
    review = client.get(url + "/input").json()["review"]
    source = review["catalog_evidence"]["economics"]
    assert review["costs"]["regular_price"]["value"] == source["regular_price"]["value"] == 120
    assert review["evaluation"]["measurement_window"]["end_date"] == "2026-12-15"
    assert review["costs"]["channel_commission"]["basis"] == "net_merchandise_after_voucher"
    assert (
        next(
            g
            for g in report["recommendation"]["assessments"][0]["checks"]
            if g["metric"] == "unit_contribution"
        )["observed"]
        == 15
    )
    assert client.get(url + "/historical-comparison").json()["reason"] == "no_comparable_case"
    for conflicting in [
        {"end_date": "2027-01-03"},
        {"offer": {**settings["offer"], "min_spend_thb": 100}},
    ]:
        payload = fields(**{**settings, **conflicting})
        assert (
            client.post(REVIEWS, json={"fields": payload}).json()["review"]["status"] == "blocked"
        )


def test_strategy_permission_and_fallback(client, monkeypatch):
    from agents import llm

    def proposer(*args):
        return args[-1](value=3.75)

    monkeypatch.setattr(llm, "call_llm", proposer)
    result = reports(client, alternative_policy="agent_bounded")
    assert result["recommendation"]["strategy"]["source"] == "llm_estimate"
    frozen = client.get(f"/api/v2/runs/{result['run_id']}/input").json()
    validator("RunInput").validate(frozen)
    assert frozen["strategy_proposal"] == result["recommendation"]["strategy"]
    assert result["plan_results"][1]["input"]["offer"]["discount"]["value"] == 3.75
    monkeypatch.setattr(llm, "call_llm", lambda *args: args[-1](value=20))
    rejected = reports(client, alternative_policy="agent_bounded")
    assert rejected["recommendation"]["strategy"]["reason"] == "candidate_outside_bounds"
    assert rejected["plan_results"][1]["input"]["offer"]["discount"]["value"] == 5
    fixed = reports(client, alternative_policy="fixed_offer", discount_response=10)
    assert [p["role"] for p in fixed["plan_results"]] == ["original", "baseline"]
    assert fixed["recommendation"]["status"] == "keep"


def test_commercial_brief_confirmations_and_product_scope(client):
    brief = (
        "Our household id: 342521. Our seller fee is 7 THB per order. We require at least "
        "5000 THB incremental profit. Commission is 5%. Also promote coffee."
    )
    review = client.post(REVIEWS, json={"brief": brief, "fields": fields()}).json()["review"]
    assert "342521" not in str(review)
    url = f"{REVIEWS}/{review['review_id']}/revisions"
    revised = client.post(url, json={"base_revision": 1, "answer": "confirm all"}).json()["review"]
    assert revised["status"] == "blocked"
    assert revised["facts"]["offer"] == fields()["offer"]
    revised = client.post(
        url, json={"base_revision": 2, "edits": {"product_scope": "single_product"}}
    ).json()["review"]
    assert revised["status"] == "ready_for_approval"
    assert revised["costs"]["payment_fee"]["value"] == 7
    assert revised["costs"]["channel_commission"]["value"] == 5
    assert revised["evaluation"]["goal_target"]["minimum"] == 5000
    assert revised["confirmed_quotes"]["/commercial_inputs/order_fee_thb"] == "seller fee is 7 THB"


def test_inflight_poll_and_abandoned_attempt(client, monkeypatch):
    from api.v2 import runs

    approval = approve(client)
    body = {"approval_id": approval, "submission_id": str(uuid4())}
    entered, release = Event(), Event()
    evaluate = runs.evaluate

    def paused(*args):
        entered.set()
        assert release.wait(10)
        return evaluate(*args)

    monkeypatch.setattr(runs, "evaluate", paused)
    worker = Thread(target=client.post, args=("/api/v2/runs",), kwargs={"json": body})
    worker.start()
    try:
        assert entered.wait(5)
        replay = client.post("/api/v2/runs", json=body)
        assert replay.status_code == 200
        url = replay.json()["status_url"]
        state = client.get(url).json()
        assert state["status"] == "running"
        validator("RunState").validate(state)
        assert client.get(url + "/evaluation").json()["code"] == "report_pending"
    finally:
        release.set()
        worker.join(10)
    assert client.get(url).json()["status"] == "done"
    with db.v2_transaction() as state:
        record = state["runs"][replay.json()["run_id"]]
        record.update(report=None, failure=None, stage="running")
    client.get(url)
    state = client.get(url).json()
    assert state["status"] == "error" and state["failure"]["retryable"]


def test_approved_weekly_rule_actually_stops_redemptions(client):
    common = dict(end_date="2026-12-15", budget_thb=10000)
    with_rule = reports(client, **common, stop_vouchers_at_utilization_pct=5)
    control = reports(client, **common)
    plan = with_rule["plan_results"][0]
    scenario = plan["scenario_results"][0]
    assert len(scenario["rule_events"]) == scenario["paired_seed_count"]
    assert all(
        e["date"] == "2026-12-08" and e["observed_pct"] >= 5 for e in scenario["rule_events"]
    )
    assert metrics(scenario["weekly"][1])["campaign_spend"]["p50"] == 0
    assert metrics(control["scenario_results"][0]["weekly"][1])["campaign_spend"]["p50"] > 0
    baseline = with_rule["plan_results"][-1]
    assert baseline["input"]["contingency_rules"] == []
    assert baseline["scenario_results"][0]["rule_events"] == []


def test_operator_catalog_uses_configured_sku_and_dated_costs(client, monkeypatch, tmp_path):
    import json

    from data.loaders import load_campaign_evidence_snapshot

    snapshot = load_campaign_evidence_snapshot()
    capability = snapshot["capabilities"][0]
    capability["capability_id"] = "seller-pack-013"
    capability["product"].update(product_id="seller-013", name="Example seller listing")
    economics = snapshot["economics"][0]
    economics["capability_id"] = capability["capability_id"]
    economics["regular_price"]["value"] = 150
    for segment in snapshot["segment_candidates"]:
        segment["capability_id"] = capability["capability_id"]
    path = tmp_path / "authorized_catalog.json"
    path.write_text(json.dumps(snapshot))
    monkeypatch.setenv("CAMPAIGNTWIN_EVIDENCE_SNAPSHOT", str(path))
    params = dict(
        product_choice_id="seller-pack-013",
        offer={
            "value": 20,
            "kind": "voucher",
            "unit": "THB",
            "eligibility": "all",
            "min_spend_thb": 120,
        },
    )
    report = reports(client, **params)
    plan = report["plan_results"][0]["input"]
    assert plan["product"]["product_id"] == "seller-013"
    for row in report["scenario_results"][0]["seed_ledgers"]:
        assert row["gross_sales"] == row["units"] * 150
    approval = approve(client, **params)
    economics["regular_price"]["value"] = 175
    path.write_text(json.dumps(snapshot))
    assert (
        client.post(
            "/api/v2/runs", json={"approval_id": approval, "submission_id": str(uuid4())}
        ).json()["code"]
        == "approval_stale"
    )
    assert client.get(f"/api/v2/runs/{report['run_id']}/report").json() == report
    economics["channel_commission"]["basis"] = "unmodeled_tax_inclusive_basis"
    path.write_text(json.dumps(snapshot))
    assert (
        client.post(REVIEWS, json={"fields": fields(**params)}).json()["review"]["status"]
        == "blocked"
    )


def test_equivalent_effective_discount_is_excluded(client):
    report = reports(
        client, offer={"kind": "percent_off", "value": 0.01, "unit": "pct", "eligibility": "all"}
    )
    assert [p["role"] for p in report["plan_results"]] == ["original", "baseline"]
    assert report["recommendation"]["alternative_shortfall"] == (
        "no_distinct_positive_subsidy_at_satang_precision"
    )


def test_fixed_catalog_guards_and_cost_windows(client):
    rice = fields(
        product_choice_id="fixture-th-shopee-rice-fixed-voucher",
        offer={
            "kind": "voucher",
            "value": 20,
            "unit": "THB",
            "eligibility": "all",
            "min_spend_thb": 120,
        },
    )
    for overrides, message in [
        ({"start_date": "2026-12-25", "end_date": "2027-01-01"}, "expire"),
        ({"min_unit_contribution_thb": 16}, "15"),
        ({"alternative_policy": "agent_bounded"}, "fixed value"),
    ]:
        review = client.post(REVIEWS, json={"fields": {**rice, **overrides}}).json()["review"]
        assert review["status"] == "blocked"
        assert any(message in b["message"] for b in review["blockers"])
    result = reports(client, **rice)
    assert len(result["plan_results"]) == 2


def test_net_profit_and_multiple_product_language(client):
    data = {
        "fields": fields(),
        "brief": "Run tissue and coffee together. We will only run if we earn 5000 THB net profit.",
    }
    review = client.post(REVIEWS, json=data).json()["review"]
    assert review["status"] == "blocked"
    assert {c["field"]: c["value"] for c in review["candidates"]}.items() >= {
        "/commercial_inputs/product_scope": "multiple_products",
        "/commercial_inputs/profit_target_thb": 5000,
    }.items()


def test_unknown_redemption_fee_preserves_evidence_and_allows_confirmed_zero(client, monkeypatch):
    from api.v2 import seller_catalog
    from data.loaders import load_campaign_evidence_snapshot

    snapshot = load_campaign_evidence_snapshot()
    fee = snapshot["economics"][0]["voucher_service_fee"]
    fee.update(status="unavailable", value=None, reason="Seller schedule missing")
    monkeypatch.setattr(seller_catalog, "load_campaign_evidence_snapshot", lambda: snapshot)
    params = fields(
        product_choice_id="fixture-th-shopee-rice-fixed-voucher",
        offer={
            "kind": "voucher",
            "value": 20,
            "unit": "THB",
            "eligibility": "all",
            "min_spend_thb": 120,
        },
    )
    review = client.post(REVIEWS, json={"fields": params}).json()["review"]
    validator("Review").validate(review)
    assert review["costs"]["voucher_service_fee"]["status"] == "unknown"
    assert any(b["field"] == "/commercial_inputs/voucher_fee_thb" for b in review["blockers"])
    review = client.post(
        f"{REVIEWS}/{review['review_id']}/revisions",
        json={"base_revision": 1, "edits": {"voucher_fee_thb": 0}},
    ).json()["review"]
    validator("Review").validate(review)
    assert review["status"] == "ready_for_approval"
    assert review["costs"]["voucher_service_fee"]["value"] == 0
    assert review["catalog_evidence"]["economics"]["voucher_service_fee"]["value"] is None

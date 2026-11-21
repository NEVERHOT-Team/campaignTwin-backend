"""Commercial bounds and coverage are pinned to the exact approved configuration."""

from agents import llm
from agents.voucher_strategy import propose_discount
from api.v2 import seller_catalog
from api.v2.catalog import resolve
from api.v2.contract import contract
from api.v2.reviews import build_review, field_pointer
from data.loaders import load_campaign_evidence_snapshot
from engine.seller_simulation import compare_plans, evaluate


def review_for(example="syntheticRehearsal", **changes):
    body = contract()["paths"]["/api/v2/campaign-reviews"]["post"]["requestBody"]
    facts = {
        **body["content"]["application/json"]["examples"][example]["value"]["fields"],
        **changes,
    }
    origins = {field_pointer(key): "planner" for key in facts}
    review = build_review(
        "review", 1, facts, origins, [], {"status": "not_requested", "reason": None}
    )
    assert review["status"] == "ready_for_approval", review["blockers"]
    return review


def test_smallest_accepted_voucher_cannot_emit_out_of_bounds_alternative():
    offer = {"kind": "voucher", "value": 0.02, "unit": "THB", "eligibility": "all"}
    review = review_for(offer=offer)
    invalid = resolve({**review["facts"], "offer": {**offer, "value": 0.01}})
    assert invalid["resolved_offer"] is None
    report = evaluate("run", "approval", review)
    assert report["recommendation"]["alternative_shortfall"] == "alternative_outside_listing_bounds"
    assert [p["role"] for p in report["plan_results"]] == ["original", "baseline"]


def test_effectively_identical_agent_offer_uses_distinct_fallback(monkeypatch):
    offer = {"kind": "percent_off", "value": 10, "unit": "pct", "eligibility": "all"}
    review = review_for(offer=offer, alternative_policy="agent_bounded")
    monkeypatch.setattr(llm, "call_llm", lambda *args: args[-1](value=9.99999))
    proposal = propose_discount(review)
    assert proposal == {"source": "default", "reason": "candidate_outside_bounds", "value": None}
    report = evaluate("run", "approval", review, proposal=proposal)
    assert report["plan_results"][1]["input"]["offer"]["discount"]["value"] == 5


def test_costs_count_planner_observations_and_assumptions(monkeypatch):
    snapshot = load_campaign_evidence_snapshot()
    # Preserve catalog references, vary evidence classes to exercise the operator snapshot path.
    refs = {row["evidence_ref"]: row for row in snapshot["sources"]}
    economics = snapshot["economics"][0]
    refs[economics["cogs"]["evidence_ref"]].update(
        label="data", evidence_class="observed_local_market"
    )
    refs[economics["voucher_service_fee"]["evidence_ref"]].update(
        label="default", evidence_class="assumption"
    )
    monkeypatch.setattr(seller_catalog, "load_campaign_evidence_snapshot", lambda: snapshot)
    review = review_for("evidenceVoucher", cogs_thb=72)
    evidence = evaluate("run", "approval", review)["evidence"]
    coverage = evidence["commercial_terms"]
    classes = evidence["commercial_evidence_classes"]
    assert sum(classes.values()) == coverage["total"] == 7
    assert coverage == {
        "total": 7,
        "planner_provided": 1,
        "catalog_data": 4,
        "assumed_or_synthetic": 2,
    }
    assert classes["business_input"] == 1
    assert classes["observed_local_market"] == 4
    assert classes["assumption"] == 2


def test_ambiguous_plan_identity_or_overstated_sample_size_withholds_ranking():
    report = evaluate("run", "approval", review_for())
    rows = report["plan_results"]
    original_id = rows[0]["input"]["plan_id"]
    rows[0]["input"]["plan_id"] = rows[1]["input"]["plan_id"]
    assert compare_plans(rows)["reason"] == "incomplete_pairing"
    rows[0]["input"]["plan_id"] = original_id
    rows[0]["scenario_results"][0]["paired_seed_count"] += 1
    assert compare_plans(rows)["reason"] == "incomplete_pairing"

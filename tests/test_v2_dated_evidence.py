"""An approved catalog listing must have its own, pre-launch evidence."""

from copy import deepcopy

import pytest

from api.v2 import seller_catalog
from api.v2.catalog import resolve
from data.loaders import load_campaign_evidence_snapshot


@pytest.fixture
def catalog(monkeypatch):
    monkeypatch.delenv("CAMPAIGNTWIN_EVIDENCE_SNAPSHOT", raising=False)
    snapshot = load_campaign_evidence_snapshot()
    monkeypatch.setattr(seller_catalog, "load_campaign_evidence_snapshot", lambda: snapshot)
    return snapshot


def rice_review(choice_id):
    return resolve({
        "product_choice_id": choice_id, "start_date": "2026-12-01", "end_date": "2026-12-08",
        "budget_thb": 1500, "objective": "incremental_profit", "audience_hint": "all",
        "offer": {"kind": "voucher", "unit": "THB", "value": 20,
                  "eligibility": "all", "min_spend_thb": 120},
    })


def test_expired_choice_cannot_borrow_another_vouchers_terms(catalog):
    current = catalog["capabilities"][0]
    other = deepcopy(current)
    other.update(capability_id="november-only", valid_from="2026-11-01",
                 valid_to_exclusive="2026-12-01")
    catalog["capabilities"].append(other)
    assert not rice_review(current["capability_id"])["blockers"]
    expired = rice_review(other["capability_id"])
    assert expired["blockers"]
    assert "catalog_evidence" not in expired


@pytest.mark.parametrize("invalid", ["observation_after_launch", "uncovered_rate", "late_source"])
def test_only_covered_prelaunch_behavior_can_drive_simulation(catalog, invalid):
    segment = catalog["segment_candidates"][0]
    source = next(item for item in catalog["sources"]
                  if item["evidence_ref"] == segment["evidence_ref"])
    if invalid == "observation_after_launch":
        source["window_end_exclusive"] = segment["window_end_exclusive"] = "2026-12-05"
    elif invalid == "uncovered_rate":
        source["window_start"] = "2026-10-30"
    else:
        source["as_of"] = "2026-12-02"
    review = rice_review(catalog["capabilities"][0]["capability_id"])
    assert review["blockers"]
    assert review["segments"] == []


def test_release_only_documented_aggregate_fields(catalog):
    segment = catalog["segment_candidates"][0]
    segment["population_count"]["example_members"] = ["sensitive-example"]
    segment["metrics"]["weekly_purchase_probability"]["denominator"]["rows"] = ["sensitive-example"]
    catalog["sources"][0]["raw_payload"] = {"members": ["sensitive-example"]}
    review = rice_review(catalog["capabilities"][0]["capability_id"])
    assert not review["blockers"]
    assert "sensitive-example" not in str(review["catalog_evidence"])


def test_unmodeled_voucher_restrictions_block_capability(catalog):
    catalog["capabilities"][0]["voucher"]["members_only"] = True
    review = rice_review(catalog["capabilities"][0]["capability_id"])
    assert review["blockers"]
    assert review["resolved_offer"] is None


def test_configured_choice_cannot_shadow_demo_identity(catalog):
    catalog["capabilities"][0]["capability_id"] = "demo-tissue-12-shopee"
    review = rice_review("demo-tissue-12-shopee")
    assert review["resolved_product"]["product_id"] == "synthetic-tissue"
    assert len({c["choice_id"] for c in review["supported_product_choices"]}) == len(
        review["supported_product_choices"]
    )


def test_private_listing_and_price_attachments_never_reach_review(catalog):
    catalog["reference"]["internal_rows"] = ["private-reference-row"]
    catalog["capabilities"][0]["product"]["transactions"] = ["private-order-row"]
    catalog["economics"][0]["cogs"]["supplier_export"] = ["private-supplier-row"]
    result = rice_review(catalog["capabilities"][0]["capability_id"])
    for private in ("private-reference-row", "private-order-row", "private-supplier-row"):
        assert private not in str(result)



def test_purchase_probability_requires_counts_of_observed_household_weeks(catalog):
    rate = catalog["segment_candidates"][0]["metrics"]["weekly_purchase_probability"]
    rate["numerator"]["value"] = 0.25
    rate["denominator"]["value"] = 1
    review = rice_review(catalog["capabilities"][0]["capability_id"])
    assert review["blockers"]
    assert review["segments"] == []

import json

import pytest

from agents.strategist import baseline
from contracts.models import CampaignSpec
from data import loaders, rival
from data.context import competitive_outlook


@pytest.fixture
def tissue_spec():
    return CampaignSpec.model_validate(loaders.load_demo_brief()["user_plan_spec"])


def test_tissue_outlook_preserves_scenario_evidence_and_campaign_sources(tissue_spec):
    outlook = competitive_outlook(tissue_spec)
    assert json.loads(json.dumps(outlook, allow_nan=False)) == outlook
    assert outlook["prior_label"] == loaders.load_rival_prior()["source"] == "data+default"
    by_id = {entry["rival"]: entry for entry in outlook["rivals"]}
    assert {key: entry["display_name"] for key, entry in by_id.items()} == {
        "764": "Scott",
        "595": "Zilk",
        "69": "House brand",
    }
    for estimate in rival.get_rival_scenarios(tissue_spec):
        entry = by_id[estimate.rival]
        assert entry["rival_score"] == estimate.rival_score
        assert set(entry["scenarios"]) == {"S1", "S2", "S3"}
        assert entry["scenarios"][estimate.scenario] == {
            "reaction": estimate.reaction,
            "likelihood": estimate.likelihood,
            "ci_low": estimate.ci_low,
            "ci_high": estimate.ci_high,
            "depth": estimate.depth,
            "delay_weeks": estimate.delay_weeks,
            "duration_weeks": estimate.duration_weeks,
            "evidence_ids": estimate.evidence_ids,
        }
    campaigns = outlook["similar_campaigns"]
    assert campaigns == rival.similar_campaigns(tissue_spec, k=5)
    assert len(campaigns) == 5
    urls = loaders.load_library().set_index("campaign_id")["source_url"]
    for campaign in campaigns:
        assert campaign["source_url"].strip()
        assert campaign["source_url"] == urls[campaign["campaign_id"]]


def test_strategy_drives_both_reactions_and_campaign_retrieval(tissue_spec):
    strategy = baseline(tissue_spec)
    outlook = competitive_outlook(tissue_spec, strategy)
    for entry in outlook["rivals"]:
        for scenario in entry["scenarios"].values():
            assert scenario["reaction"] == "none"
            assert scenario["depth"] == 0 and scenario["likelihood"] == 1

    campaigns = outlook["similar_campaigns"]
    assert campaigns == rival.similar_campaigns(tissue_spec, strategy, k=5)
    assert campaigns != rival.similar_campaigns(tissue_spec, k=5)


def test_report_preserves_scenario_specific_evidence_and_prior_label(tissue_spec, monkeypatch):
    estimates = rival.get_rival_scenarios(tissue_spec, baseline(tissue_spec))
    estimates[0] = estimates[0].model_copy(update={"evidence_ids": ["observed-response"]})
    monkeypatch.setattr(rival, "get_rival_scenarios", lambda spec, strategy: estimates)
    monkeypatch.setattr(loaders, "load_rival_prior", lambda: {"source": "data"})

    outlook = competitive_outlook(tissue_spec)
    assert outlook["prior_label"] == "data"
    scenarios = outlook["rivals"][0]["scenarios"]
    assert scenarios["S1"]["evidence_ids"] == ["observed-response"]
    assert scenarios["S2"]["evidence_ids"] == []

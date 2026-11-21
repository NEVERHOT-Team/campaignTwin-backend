from collections import defaultdict

import numpy as np
import pandas as pd
import pytest
from scipy.stats import beta

from contracts.models import CampaignSpec
from data import rival
from data.etl.build_rival_prior import (
    DEFAULT_PROBABILITY,
    EVENT_COLUMNS,
    base_rate,
    reaction_events,
)
from mocks import load_mock


@pytest.fixture
def spec():
    return CampaignSpec.model_validate(load_mock("campaign_spec")["spec"]).model_copy(
        update={
            "goal": "11.11 tissue voucher",
            "discount_pct": 20,
            "mechanic": "voucher",
            "start": "2026-11-09",
            "end": "2026-11-15",
        }
    )


def test_top_three_evidence_intervals_and_plausibility(spec):
    by_rival = defaultdict(list)
    library_ids = set(rival.load_library()["campaign_id"])
    brands = rival.load_brands().set_index("brand")
    for result in rival.get_rival_scenarios(spec):
        by_rival[result.rival].append(result)
        assert set(result.evidence_ids) <= library_ids
        assert 0 <= result.likelihood <= 1
        assert 0 <= result.ci_low <= result.ci_high <= 1
        assert 0 <= result.depth <= 1 - brands.at[result.rival, "margin_floor_pct"] / 100
        assert 0.5 <= result.rival_score <= 1
    assert len(by_rival) == 3
    assert "751" not in by_rival
    for passive, modal, aggressive in by_rival.values():
        assert [passive.scenario, modal.scenario, aggressive.scenario] == ["S1", "S2", "S3"]
        assert passive.reaction == "none" and passive.depth == 0
        assert aggressive.reaction == "undercut"
        assert modal.likelihood >= max(passive.likelihood, aggressive.likelihood)


@pytest.mark.parametrize("discount", [None, 0])
def test_no_discount_is_deterministically_passive(spec, discount):
    results = rival.get_rival_scenarios(spec.model_copy(update={"discount_pct": discount}))
    assert len(results) == 9
    for row in results:
        assert row.reaction == "none" and row.depth == row.delay_weeks == row.duration_weeks == 0
        assert row.ci_low == row.likelihood == row.ci_high == 1
        assert row.evidence_ids == []


def test_competitor_score_uses_campaign_targets_and_filters_distant_brands(spec):
    catalog = rival.load_brands().copy()
    catalog.loc[catalog["is_ours"], ["price_tier", "geo"]] = ["mid", "TH"]
    catalog.loc[~catalog["is_ours"], ["category", "target_segments", "channels", "geo"]] = [
        "other",
        "waiter",
        "other",
        "other",
    ]
    catalog.loc[
        catalog["brand"].eq("595"), ["category", "price_tier", "target_segments", "channels", "geo"]
    ] = ["toilet_tissue", "low", "explorer|loyal", "Shopee|LINE", "TH"]
    targeted = spec.model_copy(update={"target_segments": ["loyal"], "channel": ["LINE"]})
    _, selected = rival._competitors(targeted, catalog)
    assert selected["brand"].tolist() == ["595"]
    assert selected.iloc[0].score == pytest.approx(0.35 + 0.125 + 0.10 + 0.075 + 0.05)


def test_deeper_discount_does_not_lower_undercut(spec):
    previous = {}
    for discount in [1, 8, 12, 20, 25.5, 30, 40, 44, 50, 75, 100]:
        rows = rival.get_rival_scenarios(spec.model_copy(update={"discount_pct": discount}))
        current = {row.rival: row.likelihood for row in rows if row.scenario == "S3"}
        assert all(risk >= previous.get(brand, 0) - 1e-12 for brand, risk in current.items())
        previous = current


def test_linked_campaign_votes_once_and_uses_responders_terms(spec, monkeypatch):
    template = rival.load_library().iloc[0].to_dict()
    trigger = {
        **template,
        "campaign_id": "launch",
        "brand": "751",
        "category": "toilet_tissue",
        "description": spec.goal,
        "mechanic": spec.mechanic,
        "season": "11.11",
        "discount_depth": 0.20,
        "reaction_type": "",
        "reaction_to_campaign_id": "",
        "start_date": spec.start,
        "end_date": spec.end,
    }
    response = {
        **trigger,
        "campaign_id": "response",
        "brand": "764",
        "reaction_to_campaign_id": "launch",
        "reaction_type": "undercut",
        "start_date": "2026-11-12",
        "end_date": "2026-11-25",
        "reaction_delay_days": 3,
        "discount_depth": 0.99,
        "description": "A different response",
        "mechanic": "bundle",
    }
    unknown = {**response, "campaign_id": "unknown", "reaction_type": ""}
    monkeypatch.setattr(rival, "load_library", lambda: pd.DataFrame([trigger, response, unknown]))
    prior = rival.load_rival_prior()
    row = next(
        r for r in rival.get_rival_scenarios(spec) if r.rival == "764" and r.scenario == "S3"
    )
    assert row.evidence_ids == ["response"]
    alpha = 5 * prior["p"]["undercut"] + 1
    assert row.likelihood == pytest.approx(alpha / 6)
    assert [row.ci_low, row.ci_high] == pytest.approx(beta.ppf([0.05, 0.95], alpha, 6 - alpha))
    assert row.depth == pytest.approx(0.90)  # capped by this rival's margin floor
    assert row.delay_weeks == 1 and row.duration_weeks == 2


def test_all_unknown_reactions_use_category_prior(spec, monkeypatch):
    library = rival.load_library().assign(reaction_type="")
    monkeypatch.setattr(rival, "load_library", lambda: library)
    prior = rival.load_rival_prior()
    for row in rival.get_rival_scenarios(spec):
        assert row.likelihood == pytest.approx(prior["p"][row.reaction])
        assert row.evidence_ids == []
        if row.scenario == "S3":
            assert row.depth == pytest.approx(prior["depth_p90"])
            assert row.delay_weeks == prior["delay_weeks_min"]


def test_signed_event_window_thresholds_and_sparse_blend():
    grid = pd.MultiIndex.from_product([[1, 2, 3, 4, 5], range(1, 11)], names=["brand", "week"])
    weekly = pd.DataFrame({"depth": 0.0, "featured": False}, index=grid)
    weekly.loc[(1, 5), "depth"] = 0.20
    weekly.loc[(2, 8), "depth"] = 0.25
    weekly.loc[(3, 2), "depth"] = 0.27
    weekly.loc[(4, 5), "featured"] = True
    weekly.loc[(5, 9), "depth"] = 0.40
    events = reaction_events(weekly)
    triggered = events.loc[events["brand"].eq(1)].set_index("rival")
    assert triggered["reaction_type"].to_dict() == {
        2: "match",
        3: "undercut",
        4: "non_price",
        5: "none",
    }
    assert triggered.loc[3, "lag_weeks"] == -3
    assert triggered.loc[2, "lag_weeks"] == 3
    empty = base_rate(pd.DataFrame(columns=EVENT_COLUMNS))
    assert {label: empty[label] for label in DEFAULT_PROBABILITY} == DEFAULT_PROBABILITY
    small = base_rate(triggered.reset_index())
    assert small["n_events"] == 1 and small["source"] == "default"
    assert small["none"] == pytest.approx(0.95 * 0.45 + 0.05 * 0.25)


def test_neighbour_limit_and_stable_ties():
    evidence = pd.DataFrame(
        {
            "campaign_id": [str(i) for i in range(12)],
            "fixed_score": np.full(12, 0.5),
            "trigger_depth": np.full(12, 0.2),
            "reaction_type": ["match"] * 12,
        }
    )
    _, neighbours, ids = rival._posterior(evidence, 20, {"p": DEFAULT_PROBABILITY})
    assert len(neighbours) == 10 and ids == list(evidence["campaign_id"].iloc[:10])


def test_risk_floor_retrieves_current_neighbours_for_response_terms():
    evidence = pd.DataFrame(
        {
            "campaign_id": [f"c{i:02}" for i in range(20)],
            "fixed_score": [0.5] * 20,
            "trigger_depth": [0.20] * 10 + [0.80] * 10,
            "reaction_type": ["undercut"] * 10 + ["match"] * 10,
        }
    )
    prior = {"p": DEFAULT_PROBABILITY}
    shallow, _, _ = rival._posterior(evidence, 20, prior)
    deep, neighbours, ids = rival._posterior(evidence, 80, prior)
    assert deep[2] / deep.sum() == pytest.approx(shallow[2] / shallow.sum())
    assert deep.sum() == pytest.approx(5 + 10 * 0.6)
    assert set(neighbours["campaign_id"]) == set(evidence["campaign_id"].iloc[10:])
    assert set(ids) == set(evidence["campaign_id"])


def test_consecutive_trigger_weeks_and_response_depth_changes():
    index = pd.MultiIndex.from_product([[1, 2], range(1, 5)], names=["brand", "week"])
    weekly = pd.DataFrame({"depth": 0.0, "featured": False}, index=index)
    weekly.loc[1, "depth"] = 0.2
    weekly.loc[(2, 2), "depth"] = 0.05
    weekly.loc[(2, 3), "depth"] = 0.30
    events = reaction_events(weekly)
    ours = events.loc[events["brand"].eq(1)].set_index("week")
    assert list(ours.index) == [1, 2, 3, 4]
    assert ours.loc[3, "reaction_type"] == "undercut" and ours.loc[3, "lag_weeks"] == 0

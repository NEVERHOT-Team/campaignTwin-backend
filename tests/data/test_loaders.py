import json
from collections import defaultdict

import pytest

from contracts.models import BriefParseResponse
from data import loaders
from data.rival import get_rival_scenarios
from mocks import load_mock


def test_segments_and_personas():
    segments = loaders.load_segments()
    assert len(segments) == 5
    assert segments["households"].min() >= 50
    assert segments["share"].sum() == pytest.approx(1)
    assert (segments["price_elasticity"] < 0).all()
    assert set(segments["source"]) == {"default"}
    archetypes = {"loyal", "deal_hunter", "waiter", "explorer", "indifferent"}
    assert set(segments["archetype"]) == archetypes

    personas = loaders.load_personas()
    assert len(personas) == 10
    assert len({persona["persona_id"] for persona in personas}) == 10
    assert sum(persona["weight"] for persona in personas) == pytest.approx(1)
    assert {persona["segment_id"] for persona in personas} <= set(segments["segment_id"])
    for persona in personas:
        assert {"hh_size", "kids", "age_band"} <= persona["demo"].keys()
        assert {"aov_thb", "elasticity"} <= persona["behavior"].keys()
        assert "deal_proneness" in persona["psycho"]
        assert set(persona["labels"].values()) == {"default"}


def test_economics_anchors_and_backtest():
    economics = loaders.load_unit_econ()
    assert economics["currency"] == "THB"
    assert economics["price_thb"] > 0
    assert set(economics["source"].values()) == {"default"}

    anchors = loaders.load_anchors()
    assert set(anchors) == {"en", "th"}
    assert all(len(anchors[language][0]) == 5 for language in anchors)

    backtest = loaders.load_backtest()
    assert set(backtest["actual"]) == {"lift_pct", "redemption_rate", "incremental_sales"}
    assert backtest["source"] == "default"
    assert max(backtest["calibration_weeks"]) < backtest["start_week"]


def test_rival_library_prior_and_scenarios():
    library = loaders.load_library()
    assert len(library) == 8
    assert set(library["confidence"]) == {"default"}
    assert {"campaign_id", "reaction_to", "channel_mix", "source_url"} <= set(library.columns)

    prior = loaders.load_rival_prior()
    assert set(prior["p"]) == {"none", "match", "undercut", "non_price"}
    assert sum(prior["p"].values()) == pytest.approx(1)
    assert prior["source"] == "default"

    spec = BriefParseResponse.model_validate(load_mock("campaign_spec")).spec
    rivals = defaultdict(list)
    campaign_ids = set(library["campaign_id"])
    for scenario in get_rival_scenarios(spec):
        rivals[scenario.rival].append(scenario)
        assert set(scenario.evidence_ids) <= campaign_ids
    assert len(rivals) == 2
    for scenarios in rivals.values():
        assert {scenario.scenario for scenario in scenarios} == {"S1", "S2", "S3"}
        assert sum(scenario.likelihood for scenario in scenarios) == pytest.approx(1)


def test_loader_names_missing_keys(monkeypatch, tmp_path):
    path = tmp_path / "rival_prior.json"
    path.write_text(json.dumps({"p": {"none": 1}}))
    monkeypatch.setattr(loaders, "PROCESSED", tmp_path)
    loaders.load_rival_prior.cache_clear()
    with pytest.raises(ValueError, match=r"rival_prior.json missing required.*n_events"):
        loaders.load_rival_prior()
    loaders.load_rival_prior.cache_clear()

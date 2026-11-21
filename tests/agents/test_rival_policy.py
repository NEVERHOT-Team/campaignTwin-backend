import json
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from agents import llm, rival_policy
from contracts.models import ContextPack, RivalScenario, Strategy
from data import loaders
from mocks import load_mock


@pytest.fixture
def rivals():
    return ContextPack.model_validate(load_mock("context_pack")).rivals


@pytest.fixture
def spec():
    return {"brand": {"name": "Morning Vale", "category": "instant_coffee"}}


@pytest.fixture
def strategies():
    return [
        Strategy(
            id=name,
            name=name,
            discount_pct=20,
            mechanic="percent_off",
            target_segments=["deal_hunter"],
            message="A campaign message",
            channel_mix={"shopee": 1},
            timing={"weeks": [1, 2, 3, 4]},
            cap_per_user=1,
            budget_split={"discount": 0.8, "media": 0.2},
            contingency_rules=[],
            rationale="test strategy",
            is_baseline=name == "baseline",
        )
        for name in ["baseline", "plan-a", "plan-b", "plan-c", "plan-d"]
    ]


@pytest.fixture(autouse=True)
def offline_llm(monkeypatch):
    provider = Mock(
        return_value=rival_policy.CounterMove(
            reaction="undercut",
            depth=0.99,
            delay_weeks=1,
            duration_weeks=2,
            rationale="Target the same shoppers with an earlier undercut.",
        )
    )
    monkeypatch.setattr(rival_policy, "call_llm", provider)
    return provider


def test_top_three_exclude_our_brand_deduplicate_and_sort(spec, rivals):
    reference = rivals[0]
    candidates = [
        reference.model_copy(update={"rival": name, "rival_score": score})
        for name, score in [
            ("Morning Vale", 1),
            ("daily peak", 0.8),
            ("Daily Peak", 0.9),
            ("Rival C", 0.6),
            ("Rival D", 0.7),
            ("Rival E", 0.5),
        ]
    ]
    top = rival_policy.select_top_rivals(spec, candidates)
    assert [row.rival for row in top] == ["Daily Peak", "Rival D", "Rival C"]
    assert [row.rival_score for row in top] == [0.9, 0.7, 0.6]
    assert rival_policy.select_top_rivals(spec, list(reversed(candidates))) == top


def test_fixture_s1_s3_probability_intervals_remain_valid(rivals):
    assert {row.scenario for row in rivals} == {"S1", "S2", "S3"}
    for row in rivals:
        assert 0 <= row.ci_low <= row.likelihood <= row.ci_high <= 1


def test_s4_shortlist_depth_evidence_and_unweighted_outputs(spec, strategies, rivals, offline_llm):
    before = [row.model_dump() for row in rivals]
    rows = rival_policy.red_team(spec, strategies, rivals)
    assert len(rows) == 3
    assert [row.strategy_id for row in rows] == ["plan-a", "plan-b", "plan-c"]
    library = loaders.load_library()
    top_history = library[library["brand"] == "Daily Peak"]
    cap = top_history["discount_depth"].max() * 1.1
    for row in rows:
        assert isinstance(row, RivalScenario)
        assert row.rival == "Daily Peak" and row.scenario == "S4"
        assert row.depth == pytest.approx(cap)
        assert row.likelihood == row.ci_low == row.ci_high == 0
        assert row.label == "llm_estimate" and row.evidence_label == "default"
        assert set(row.evidence_ids) <= set(top_history["campaign_id"])
        assert row.rationale
        assert row.model_dump()["strategy_id"] == row.strategy_id
    assert [row.model_dump() for row in rivals] == before
    assert offline_llm.call_count == 3
    model, messages, schema = offline_llm.call_args.args
    assert model == "strong" and schema is rival_policy.CounterMove
    prompt = json.loads(messages[1]["content"])
    assert {row["scenario"] for row in prompt["S2_S3"]} == {"S2", "S3"}
    assert prompt["history_stats"]["max_observed_depth"] == 0.44


def test_bad_evidence_ids_are_dropped(spec, strategies, rivals):
    for row in rivals:
        row.evidence_ids = ["fabricated", "c02"]
    result = rival_policy.red_team(spec, strategies[:2], rivals)[0]
    assert result.evidence_ids == ["c02"]


@pytest.mark.parametrize("field,value", [("rival_margin_floor", 0.2), ("margin_floor_pct", 20)])
def test_known_rival_margin_cap_is_enforced(spec, strategies, rivals, monkeypatch, field, value):
    library = loaders.load_library().copy()
    library[field] = value
    monkeypatch.setattr(rival_policy, "load_library", lambda: library)
    result = rival_policy.red_team(spec, strategies[:2], rivals)[0]
    assert result.depth == result.depth_bound == pytest.approx(0.2)


def test_llm_failure_uses_s3_numbers_and_default_label(spec, strategies, rivals, offline_llm):
    offline_llm.return_value = None
    result = rival_policy.red_team(spec, strategies[:2], rivals)[0]
    s3 = next(row for row in rivals if row.rival == "Daily Peak" and row.scenario == "S3")
    assert (result.reaction, result.depth, result.delay_weeks, result.duration_weeks) == (
        s3.reaction,
        s3.depth,
        s3.delay_weeks,
        s3.duration_weeks,
    )
    assert result.label == "default"
    assert "S3 fallback" in result.rationale


def test_empty_library_uses_prior_bound_and_no_llm(
    spec, strategies, rivals, monkeypatch, offline_llm
):
    monkeypatch.setattr(rival_policy, "load_library", pd.DataFrame)
    policies = [row for row in rivals if row.scenario != "S3"]
    first = rival_policy.red_team(spec, strategies, policies)
    assert first == rival_policy.red_team(spec, strategies, policies)
    prior = loaders.load_rival_prior()
    for row in first:
        assert row.label == "default" and row.bound_source == "prior"
        assert row.depth == prior["depth_p90"]
        assert row.depth_bound == pytest.approx(prior["depth_p90"] * 1.1)
        assert row.delay_weeks == prior["delay_weeks_min"]
        assert row.duration_weeks == prior["duration_weeks"]
        assert row.evidence_ids == []
    offline_llm.assert_not_called()


def test_missing_rivals_default_to_context_fixture(spec, strategies, rivals):
    assert rival_policy.red_team(spec, strategies) == rival_policy.red_team(
        spec, strategies, rivals
    )


def test_empty_or_baseline_only_inputs_make_no_model_calls(spec, strategies, offline_llm):
    assert rival_policy.red_team(spec, strategies, []) == []
    assert rival_policy.red_team(spec, []) == []
    assert rival_policy.red_team(spec, strategies[:1]) == []
    offline_llm.assert_not_called()


@pytest.mark.parametrize("bad", [{"reaction": "invented"}, {"depth": float("nan")}])
def test_invalid_proposals_fall_back_cleanly(spec, strategies, rivals, offline_llm, bad):
    offline_llm.return_value = {
        "reaction": "undercut",
        "depth": 0.2,
        "delay_weeks": 0,
        "duration_weeks": 1,
        "rationale": "test",
        **bad,
    }
    row = rival_policy.red_team(spec, strategies[:2], rivals)[0]
    assert row.label == "default" and row.depth <= row.depth_bound


def test_real_call_wrapper_caches_so_identical_inputs_produce_identical_s4(
    spec,
    strategies,
    rivals,
    monkeypatch,
):
    cache = {}
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("MODEL_STRONG", "test-strong")
    monkeypatch.setattr(llm.db, "cache_get", cache.get)
    monkeypatch.setattr(
        llm.db, "cache_put", lambda key, model, prompt, result: cache.update({key: result})
    )
    parsed = rival_policy.CounterMove(
        reaction="undercut",
        depth=0.3,
        delay_weeks=0,
        duration_weeks=2,
        rationale="test",
    )
    parse = Mock(
        return_value=SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(parsed=parsed, refusal=None),
                )
            ]
        )
    )
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(parse=parse)), close=Mock()
    )
    monkeypatch.setattr(llm, "OpenAI", Mock(return_value=client))
    monkeypatch.setattr(rival_policy, "call_llm", llm.call_llm)
    first = rival_policy.red_team(spec, strategies, rivals)
    assert rival_policy.red_team(spec, strategies, rivals) == first
    assert parse.call_count == 3

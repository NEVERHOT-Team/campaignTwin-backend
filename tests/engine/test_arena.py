from copy import deepcopy

import pytest

from agents.strategist import propose
from contracts.campaign_spec import CampaignSpec
from contracts.models import ContextPack, RivalScenario, SimulationPlan, Strategy
from engine import arena, evaluator, shopper
from mocks import load_mock

SEEDS = 3
AGENTS = 2000


def rival_row(scenario, reaction="none", depth=0.0, likelihood=0.3, delay=0, duration=4):
    return RivalScenario(
        rival="764", scenario=scenario, reaction=reaction, depth=depth, delay_weeks=delay,
        duration_weeks=duration, likelihood=likelihood, ci_low=0.0, ci_high=1.0,
        evidence_ids=[], rival_score=0.8,
    )


def fixed_evidence(spec, strategy):
    return [
        rival_row("S1", likelihood=0.7),
        rival_row("S2", "match", 20.0, likelihood=0.2),
        rival_row("S3", "undercut", 40.0, likelihood=0.1),
    ]


class FakeS4(RivalScenario):
    strategy_id: str


def fake_red_team(spec, strategies, rivals):
    return [
        FakeS4(**rival_row("S4", "undercut", 45.0, likelihood=0.0).model_dump(),
               strategy_id=s.id)
        for s in strategies[:2]
    ]


@pytest.fixture(scope="module")
def plan():
    spec = CampaignSpec.model_validate(deepcopy(load_mock("campaign_spec")["spec"]))
    context = ContextPack.model_validate(load_mock("context_pack"))
    return SimulationPlan(
        spec=spec, context=context, strategies=propose(spec, context),
        scenarios=["S1", "S2", "S3", "S4"], seeds=SEEDS, n_agents=AGENTS,
    )


@pytest.fixture(scope="module")
def outcome(plan):
    return arena.run(plan, evidence_fn=fixed_evidence, red_team_fn=fake_red_team)


def test_every_strategy_gets_s1_to_s3_and_shortlist_gets_s4(plan, outcome):
    candidates = [s.id for s in plan.strategies if not s.is_baseline]
    got = {(r.strategy_id, r.scenario) for r in outcome.results}
    for sid in candidates:
        assert {(sid, s) for s in ("S1", "S2", "S3")} <= got
    assert {sid for sid, s in got if s == "S4"} == set(candidates[:2])
    assert "baseline" not in {sid for sid, _ in got}
    assert outcome.appeal_source == "appeal_fallback"
    assert (outcome.campaign_weeks, outcome.weeks) == (4, 6)


def test_results_match_the_contract_shape(outcome):
    for result in outcome.results:
        assert len(result.weekly) == outcome.weeks
        kpi = result.kpi
        assert kpi.incr_profit.p10 <= kpi.incr_profit.p50 <= kpi.incr_profit.p90
        assert 0 <= kpi.p_loss <= 1 and 0 <= kpi.p_hit_target <= 1
        assert 0 <= kpi.leakage.p50 <= 1


def test_arena_is_deterministic(plan, outcome):
    again = arena.run(plan, evidence_fn=fixed_evidence, red_team_fn=fake_red_team)
    assert [r.model_dump() for r in again.results] == [r.model_dump() for r in outcome.results]


def test_aggressive_rival_hurts_more_than_passive(outcome):
    by_key = {(r.strategy_id, r.scenario): r for r in outcome.results}
    deep = "g0-deep"
    assert by_key[(deep, "S3")].kpi.incr_profit.p50 < by_key[(deep, "S1")].kpi.incr_profit.p50


def test_summary_ranks_every_strategy_with_weighted_ev(plan, outcome):
    summary = outcome.summary
    assert [row["rank"] for row in summary] == list(range(1, len(summary) + 1))
    assert {row["strategy_id"] for row in summary} == {
        s.id for s in plan.strategies if not s.is_baseline
    }
    weights = summary[0]["weights"]
    assert set(weights) == {"S1", "S2", "S3"} and sum(weights.values()) == pytest.approx(1)
    assert weights["S1"] == pytest.approx(0.7)
    evs = [row["ev_incr_profit"] for row in summary]
    assert evs == sorted(evs, reverse=True)


def test_a_no_discount_no_media_plan_matches_the_baseline(plan):
    copy = Strategy(
        id="a-a", name="A/A check", discount_pct=0, mechanic="percent_off",
        target_segments=[], message="", channel_mix={"shopee": 1.0},
        timing={"budget_thb": 0}, cap_per_user=None, budget_split={"discount": 1.0},
        contingency_rules=[], rationale="",
    )
    aa = plan.model_copy(update={"strategies": [copy], "scenarios": ["S1"]})
    result = arena.run(aa, appeal={}, evidence_fn=fixed_evidence).results[0]
    assert result.kpi.incr_profit.p50 == pytest.approx(0)
    assert result.kpi.new_buyers.p50 == pytest.approx(0)


def test_failed_evidence_falls_back_to_context_rivals(plan):
    def broken(*_):
        raise RuntimeError("no data")

    small = plan.model_copy(update={"scenarios": ["S1"], "seeds": 1})
    result = arena.run(small, evidence_fn=broken)
    assert result.results
    some = next(iter(result.rivals.values()))
    assert some["S1"] and all(row.scenario == "S1" for row in some["S1"])


def test_schedule_and_message_style(plan):
    assert arena.schedule(plan.spec, 6) == (4, 6)
    assert arena.schedule(plan.spec, 4) == (2, 4)
    flash = next(s for s in plan.strategies if s.id == "g0-flash")
    assert arena.message_style(flash) == "urgency"


def test_fallback_appeal_covers_every_persona_and_plan(plan):
    population = shopper.build_population(500, seed=1)
    candidates = [s for s in plan.strategies if not s.is_baseline]
    table = arena.fallback_appeal(population, candidates)
    assert len(table) == len(population.persona_ids) * len(candidates)
    assert all(-1 <= value <= 1 for value in table.values())


# ---------------------------------------------------------------- evaluator units


def outcome_with(weekly, profit=None):
    rows = tuple(
        evaluator.WeekRow(week=i + 1, sales=0, profit=0, buyers=0, from_rival=0, waiting=0)
        for i in range(len(weekly))
    )
    return evaluator.SeedOutcome(
        incr_profit=sum(weekly) if profit is None else profit, incr_revenue=0, campaign_cost=0,
        iroas=None, leakage=None, new_buyers=0, weekly_incr_profit=tuple(weekly), plan_rows=rows,
    )


def test_pct_orders_and_handles_empty():
    p = evaluator.pct([3, 1, 2, 10, 5])
    assert p.p10 <= p.p50 <= p.p90
    assert evaluator.pct([]).model_dump() == {"p10": 0.0, "p50": 0.0, "p90": 0.0}


def test_breakpoint_finds_the_week_profit_stays_negative():
    good = [outcome_with([100, 50, 20, 10, 5, 5])]
    assert evaluator.breakpoint(good, 4, []).week is None
    breaks = [outcome_with([100, 50, -200, -50, -10, -10])]
    bp = evaluator.breakpoint(breaks, 4, [rival_row("S3", "undercut", 40, delay=2, duration=2)])
    assert bp.week == 3 and bp.cause == "rival_reaction" and bp.share == 1.0
    dip = evaluator.breakpoint([outcome_with([50, 10, 10, 10, -100, -50])], 4, [])
    assert dip.week == 5 and dip.cause == "post_campaign_dip"


def test_scenario_weights_normalize_and_default_to_equal():
    rows = {"S1": [rival_row("S1", likelihood=0.6)], "S2": [rival_row("S2", likelihood=0.3)],
            "S3": [rival_row("S3", likelihood=0.3)]}
    weights = evaluator.scenario_weights(rows)
    assert sum(weights.values()) == pytest.approx(1) and weights["S1"] == pytest.approx(0.5)
    zero = {s: [rival_row(s, likelihood=0)] for s in ("S1", "S2", "S3")}
    assert evaluator.scenario_weights(zero) == pytest.approx({s: 1 / 3 for s in zero})

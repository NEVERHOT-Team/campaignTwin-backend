"""Two-sided adaptation: our contingency rules and reactive rivals in the weekly loop."""

import pytest

from contracts.models import RivalScenario, Strategy
from engine import rival, rules, shopper

N = 4000
WEEKS = 6
CAMPAIGN = 4
SEED = 11
CORE = ("buyers_us", "units_us", "revenue_us", "discounted_units_us", "buyers_rival",
        "buyers_other", "from_rival", "new_buyers_us", "waiting", "exposed")


def make_strategy(discount=30.0, targets=(), rules_=(), baseline=False, sid=None):
    return Strategy(
        id="baseline" if baseline else (sid or f"plan-{discount:g}"),
        name="test",
        discount_pct=0 if baseline else discount,
        mechanic="none" if baseline else "percent_off",
        target_segments=list(targets),
        message="test",
        channel_mix={} if baseline else {"shopee": 0.5, "line": 0.5},
        timing={},
        cap_per_user=None,
        budget_split={},
        contingency_rules=list(rules_),
        rationale="test",
        is_baseline=baseline,
    )


def row(scenario="S3", reaction="undercut", depth=30.0, delay=0, duration=4):
    return RivalScenario(
        rival="764", scenario=scenario, reaction=reaction, depth=depth, delay_weeks=delay,
        duration_weeks=duration, likelihood=0.1, ci_low=0.0, ci_high=1.0,
        evidence_ids=[], rival_score=0.8,
    )


@pytest.fixture(scope="module")
def population():
    return shopper.build_population(N, seed=SEED)


def adaptive(population, strategy, rivals=None, reference=None, seed=SEED):
    return shopper.simulate(population, strategy, WEEKS, CAMPAIGN, seed, rivals, None,
                            adaptive=True, reference=reference)


# ---------------------------------------------------------------- rules


def test_parse_rules_keeps_supported_and_drops_unknown():
    parsed = rules.parse_rules([
        "if rival_undercuts then shift_budget:loyal",
        "if share_drop>5 then cap_per_user:1",
        "if week>=3 and loss then end_early",
        "if rival_undercuts then match:15",  # not supported: would change our discount cost
        "nonsense",
    ])
    assert [r.kind for r in parsed] == ["shift_budget", "cap_per_user", "end_early"]
    assert parsed[0].segment == "loyal"
    assert (parsed[1].threshold, parsed[1].cap) == (5.0, 1)
    assert parsed[2].threshold == 3.0


def test_apply_rules_fires_each_rule_once_on_its_trigger():
    parsed = rules.parse_rules([
        "if rival_undercuts then shift_budget:loyal",
        "if share_drop>5 then cap_per_user:1",
        "if week>=3 and loss then end_early",
    ])
    plan = rules.PlanState(target_segments=(), cap_per_user=None, reach=1.0)
    calm = rules.Observation(week=1, our_discount_pct=20, rival_discount_pct=0,
                             our_share_pp=30, first_share_pp=30, incr_profit_to_date=100)
    assert rules.apply_rules(parsed, plan, calm, 2) == []

    attack = rules.Observation(week=2, our_discount_pct=20, rival_discount_pct=25,
                               our_share_pp=22, first_share_pp=30, incr_profit_to_date=-5)
    fired = rules.apply_rules(parsed, plan, attack, 3)
    assert [e.action for e in fired] == ["shift_budget", "cap_per_user", "end_early"]
    assert plan.target_segments == ("loyal",) and plan.cap_per_user == 1 and plan.stopped
    assert rules.apply_rules(parsed, plan, attack, 4) == []  # once each


def test_end_early_waits_for_its_week():
    parsed = rules.parse_rules(["if week>=3 and loss then end_early"])
    plan = rules.PlanState(target_segments=(), cap_per_user=None, reach=1.0)
    losing = rules.Observation(week=1, incr_profit_to_date=-1)
    assert rules.apply_rules(parsed, plan, losing, 2) == [] and not plan.stopped
    assert rules.apply_rules(parsed, plan, losing, 3) and plan.stopped


# ---------------------------------------------------------------- rival


def test_passive_rival_never_reacts():
    book = rival.RivalBook([row("S1", "none", 0.0)])
    assert all(book.react(w, 40.0)[0] == 0 for w in range(1, 7))


def test_match_follows_our_discount_up_to_its_cap_and_ratchets():
    book = rival.RivalBook([row("S2", "match", depth=20.0, delay=1, duration=3)])
    depths = [book.react(w, d)[0] for w, d in enumerate([10, 10, 30, 0, 0, 0], 1)]
    # noticed in week 1 -> reacts in weeks 2-4: follows us (10), caps at 20, never drops
    assert depths == [0, 10, 20, 20, 0, 0]


def test_undercut_goes_one_step_deeper_than_us():
    book = rival.RivalBook([row("S3", "undercut", depth=40.0, delay=0, duration=2)])
    assert book.react(1, 10.0)[0] == pytest.approx(10 + rival.UNDERCUT_STEP_PP)
    assert book.react(2, 38.0)[0] == pytest.approx(40.0)  # capped
    assert book.react(3, 38.0)[0] == 0  # window over


def test_rival_ignores_a_campaign_it_cannot_see():
    book = rival.RivalBook([row("S3", "undercut", depth=40.0)])
    assert book.react(1, 0.0)[0] == 0  # no discount, no share gain
    assert book.react(2, 0.0, share_gain_pp=rival.TRIGGER_SHARE_GAIN_PP)[0] > 0


def test_red_team_keeps_its_designed_depth():
    book = rival.RivalBook([row("S4", "undercut", depth=33.0)])
    assert book.react(1, 10.0)[0] == pytest.approx(33.0)


def test_rival_does_not_start_a_price_war_against_an_ended_offer():
    late = rival.RivalBook([row("S3", "undercut", depth=40.0, delay=3)])
    late.react(1, 26.0)  # noticed while our offer ran
    assert late.react(4, 0.0, our_active=False)[0] == 0  # we stopped in week 3
    running = rival.RivalBook([row("S3", "undercut", depth=40.0, delay=0, duration=3)])
    assert running.react(1, 20.0)[0] == pytest.approx(25.0)
    assert running.react(2, 0.0, our_active=False)[0] == pytest.approx(25.0)  # keeps going


def test_identical_rival_moves_are_one_event():
    rows = [row("S2", "match", depth=30.0).model_copy(update={"rival": name})
            for name in ("764", "595", "69")]
    _, _, events = rival.RivalBook(rows).react(1, 20.0)
    assert [e.action for e in events] == ["noticed", "match"]
    assert events[0].detail.startswith("764, 595, 69 ")


# ---------------------------------------------------------------- closed loop


def test_adaptive_without_rules_or_rivals_matches_the_fixed_plan(population):
    plan = make_strategy(25)
    fixed = shopper.simulate(population, plan, WEEKS, CAMPAIGN, SEED)
    loop = adaptive(population, plan)
    for a, b in zip(fixed, loop, strict=True):
        assert [getattr(a, f) for f in CORE] == [getattr(b, f) for f in CORE]


def test_baseline_has_no_rival_reaction_or_rules(population):
    base = make_strategy(baseline=True)
    stats = adaptive(population, base, rivals=[row()])
    assert all(s.extra["rival_discount_pct"] == 0 and not s.extra["events"] for s in stats)


def test_rival_reacts_to_our_offer_and_is_deterministic(population):
    plan = make_strategy(20)
    first = adaptive(population, plan, rivals=[row("S3", "undercut", depth=30.0, delay=1)])
    again = adaptive(population, plan, rivals=[row("S3", "undercut", depth=30.0, delay=1)])
    assert [s.extra for s in first] == [s.extra for s in again]
    depths = [s.extra["rival_discount_pct"] for s in first]
    assert depths[0] == 0 and depths[1] == pytest.approx(25.0)
    assert any(e["side"] == "rival" for s in first for e in s.extra["events"])


def test_our_rule_responds_to_the_rival(population):
    plan = make_strategy(20, rules_=["if rival_undercuts then shift_budget:loyal"])
    passive = adaptive(population, plan, rivals=[row("S1", "none", 0.0)])
    attacked = adaptive(population, plan, rivals=[row("S3", "undercut", depth=30.0)])
    ours = [e for s in attacked for e in s.extra["events"] if e["side"] == "us"]
    assert ours and ours[0]["action"] == "shift_budget" and ours[0]["week"] == 2
    assert not [e for s in passive for e in s.extra["events"] if e["side"] == "us"]


def test_end_early_stops_discounted_sales(population):
    base = shopper.simulate(population, make_strategy(baseline=True), WEEKS, CAMPAIGN, SEED)
    deep = make_strategy(60, rules_=["if week>=2 and loss then end_early"])
    stats = adaptive(population, deep, reference=base)
    stopped = [e for s in stats for e in s.extra["events"] if e["action"] == "end_early"]
    if stopped:  # a 60% discount loses money on these economics, so the rule should fire
        week = stopped[0]["week"]
        assert all(s.discounted_units_us == 0 for s in stats[week - 1:])


def test_our_rules_stay_quiet_after_the_campaign(population):
    # The rival keeps discounting into the post window; our rule must not fire there.
    plan = make_strategy(20, rules_=["if rival_undercuts then shift_budget:loyal"])
    stats = adaptive(population, plan,
                     rivals=[row("S3", "undercut", depth=30.0, delay=CAMPAIGN - 1, duration=3)])
    fired = [e["week"] for s in stats for e in s.extra["events"] if e["side"] == "us"]
    assert all(week <= CAMPAIGN for week in fired)

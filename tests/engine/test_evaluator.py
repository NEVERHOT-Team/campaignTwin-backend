from dataclasses import replace
from warnings import catch_warnings, simplefilter

import pytest

from contracts.campaign_spec import SmartGoal
from contracts.models import SimResult, Strategy
from engine import evaluator
from engine.shopper import WeekStats


def market(campaign=1, post=0, **overrides):
    return {
        "ref_price": 100,
        "unit_cost": 60,
        "platform_fee_pct": 0,
        "payment_fee_pct": 0,
        "shipping_cost_per_order": 0,
        "weeks": {"campaign": campaign, "post": post},
        **overrides,
    }


def strategy(plan_id="offer", discount=10, **overrides):
    return Strategy(
        id=plan_id,
        name=plan_id,
        discount_pct=discount,
        mechanic="percent_off",
        target_segments=[],
        message="",
        channel_mix={},
        cap_per_user=None,
        contingency_rules=[],
        rationale="",
        **({"budget_split": {}, "timing": {}} | overrides),
    )


def week(units, *, discounted=0, would_buy=None, buyers=None, new=0):
    """Each full-price unit costs ฿100 and each discounted unit costs ฿90."""
    return WeekStats(
        buyers_us=units if buyers is None else buyers,
        units_us=units,
        revenue_us=100 * units - 10 * discounted,
        discounted_units_us=discounted,
        buyers_rival=0,
        buyers_other=0,
        from_rival=0,
        new_buyers_us=new,
        waiting=0,
        exposed=discounted,
        holdout_buyers_us=0,
        holdout_size=0,
        extra={} if would_buy is None else {"discounted_units_would_buy": would_buy},
    )


def test_one_seed_costs_and_net_returns():
    economics = market(platform_fee_pct=7, payment_fee_pct=3, shipping_cost_per_order=2)
    plan = strategy(budget_split={"discount": 0.8, "media": 0.2})
    result = evaluator.evaluate_plan(
        {7: [week(12, discounted=12, buyers=6, new=3)]},
        {7: [week(8, buyers=4, new=1)]},
        plan,
        "S1",
        economics,
        100,
    )

    assert isinstance(result, SimResult)
    # ฿1,080 - ฿108 fees - 12 × ฿60 cost - 6 × ฿2 shipping = ฿240.
    assert result.weekly[0].profit == 240
    # Baseline gross profit is ฿232. Net increment is ฿240 - ฿232 - ฿20 media.
    kpi = result.kpi
    assert kpi.incr_profit.p50 == -12
    assert kpi.incr_revenue.p50 == 280
    assert kpi.discount_cost.p50 == 120
    assert kpi.media_cost.p50 == 20
    assert kpi.campaign_cost.p50 == 140
    assert kpi.iroas.p50 == (1080 - 800) / (120 + 20)
    assert kpi.iroi.p50 == pytest.approx(-12 / 140)
    assert kpi.leakage is None
    assert kpi.leakage_label == "unavailable"
    assert kpi.new_buyers.p50 == 2
    assert kpi.p_loss == 1
    assert kpi.p_hit_target == 0
    assert evaluator.EvaluatedPlan.model_validate_json(result.model_dump_json()) == result


def test_arena_and_evaluate_plan_publish_the_same_campaign_accounting():
    plan = strategy(
        "member-days",
        is_user_plan=True,
        timing={"budget_thb": 100},
        budget_split={"media": 0.2, "discount": 0.8},
    )
    stats = [week(16, discounted=16, new=2), week(10, discounted=10, new=1), week(1)]
    base = [week(8), week(8), week(10)]
    economics = market(campaign=2, post=1)
    direct = evaluator.evaluate_plan({7: stats}, {7: base}, plan, "S2", economics, 100, target=-240)
    seed_result = evaluator.seed_outcome(plan, stats, base, 2, 100, 100, economics)
    through_arena = evaluator.sim_result(plan, "S2", [seed_result], 2, [], target_profit=-240)

    # The weekly chart reports gross earnings. After the two ฿10 media installments,
    # cumulative increments are +150, +120, -240.
    assert through_arena.weekly == direct.weekly
    assert [row.profit for row in through_arena.weekly] == [480, 300, 40]
    assert through_arena.breakpoint.week == direct.breakpoint.week == 3
    assert through_arena.breakpoint.cause == direct.breakpoint.cause == "post_campaign_dip"
    assert through_arena.kpi == direct.kpi
    # Result persistence in api.routes uses this JSON-mode dump.
    payload = through_arena.model_dump(mode="json")
    assert payload["strategy_id"] == "member-days"
    assert payload["is_user_plan"]
    assert payload["scenario"] == "S2"
    assert payload["media_cost_label"] == "data"
    assert payload["peak_week"] == 1
    metrics = payload["kpi"]
    assert metrics["incr_profit"]["p50"] == -240
    assert metrics["discount_cost"]["p50"] == 260
    assert metrics["media_cost"]["p50"] == 20
    assert metrics["campaign_cost"]["p50"] == 280
    assert metrics["iroi"]["p50"] == pytest.approx(-240 / 280)
    assert metrics["leakage"] is None
    assert metrics["leakage_label"] == "unavailable"
    assert metrics["p_hit_target"] == 1


def test_thirty_satang_timing_difference_breaks_even():
    # The same ฿0.30 arrives in different weeks; there is no loss or breakpoint.
    plan = {1: [replace(week(3), revenue_us=0.30), week(0), week(0)]}
    base = {1: [week(0), replace(week(2), revenue_us=0.20), replace(week(1), revenue_us=0.10)]}
    result = evaluator.evaluate_plan(
        plan,
        base,
        strategy(discount=0),
        "S1",
        market(campaign=1, post=2, ref_price=0.10, unit_cost=0),
        0,
        target=0,
    )
    assert result.kpi.incr_profit.p50 == 0
    assert result.kpi.p_loss == 0
    assert result.kpi.p_hit_target == 1
    assert result.breakpoint.week is None
    assert result.breakpoint.cause is None
    assert result.peak_week == 1


def test_seed_pairing_quantiles_and_weekly_medians():
    # Deliberately insert the baseline seeds in a different order.
    plan = {7: [week(8, discounted=8)], 9: [week(12, discounted=12)], 11: [week(20, discounted=20)]}
    base = {11: [week(8)], 7: [week(10)], 9: [week(9)]}
    result = evaluator.evaluate_plan(plan, base, strategy(), "S2", market(), 0, target=0)
    # Seed increments are ฿240 - ฿400, ฿360 - ฿360, and ฿600 - ฿320.
    pct = result.kpi.incr_profit
    assert [pct.p10, pct.p50, pct.p90] == pytest.approx([-128, 0, 224])
    assert result.kpi.p_loss == pytest.approx(1 / 3)
    assert result.kpi.p_hit_target == pytest.approx(2 / 3)
    assert result.media_cost_label == "default"
    assert result.weekly[0].model_dump() == {
        "week": 1,
        "sales": 1080,
        "profit": 360,
        "buyers": 12,
        "from_rival": 0,
        "waiting": 0,
    }


def test_leakage_aggregates_paired_shopper_units_across_campaign_weeks():
    result = evaluator.evaluate_plan(
        {1: [week(20, discounted=20, would_buy=3), week(5, discounted=5, would_buy=4), week(10)]},
        {1: [week(40), week(40), week(40)]},
        strategy(),
        "S1",
        market(campaign=2, post=1),
        100,
    )
    # Both baseline weeks sold 40 units. Only 7 discounted units went to paired buyers.
    assert result.kpi.leakage.p50 == pytest.approx(7 / 25)
    assert result.kpi.leakage_label == "data"
    assert result.kpi.media_cost.p50 == 0
    assert result.media_cost_label == "default"


@pytest.mark.parametrize("missing_week", [False, True])
def test_leakage_requires_counterfactuals_for_all_redeemed_units(missing_week):
    measured = week(5, discounted=5, would_buy=0)
    missing = week(5, discounted=5)
    plan = {1: [measured, measured], 2: [measured, missing if missing_week else measured]}
    base = {seed: [week(20), week(20)] for seed in plan}
    result = evaluator.evaluate_plan(plan, base, strategy(), "S1", market(campaign=2), 0)
    if missing_week:
        assert result.kpi.leakage is None
        assert result.kpi.leakage_label == "unavailable"
    else:
        assert result.kpi.leakage.p50 == 0
        assert result.kpi.leakage_label == "data"


def test_breakpoint_persists_after_recovery_to_a_higher_peak():
    plan = [week(12, discounted=12), week(12, discounted=12), week(5)]
    base = [week(8), week(8), week(10)]
    # Cumulative increments are +40, +80, -120. A fourth week takes profit to +120.
    broken = evaluator.evaluate_plan(
        {1: plan},
        {1: base},
        strategy(),
        "S1",
        market(campaign=2, post=1),
        0,
    )
    recovered = evaluator.evaluate_plan(
        {1: [*plan, week(16)]},
        {1: [*base, week(10)]},
        strategy(),
        "S1",
        market(campaign=2, post=2),
        0,
    )
    for result in [broken, recovered]:
        assert result.breakpoint.week == 3
        assert result.breakpoint.cause == "post_campaign_dip"
        assert result.peak_week == 2


@pytest.mark.parametrize("counts", [[10, 10, 10], [9, 11, 9]])
def test_never_profitable_has_no_breakpoint_or_positive_peak(counts):
    result = evaluator.evaluate_plan(
        {1: [week(n) for n in counts]},
        {1: [week(10)] * 3},
        strategy(discount=0),
        "S1",
        market(campaign=2, post=1),
        0,
    )
    assert result.breakpoint.week is None
    assert result.breakpoint.cause == "never_profitable"
    assert result.peak_week is None


def test_breakpoint_uses_median_of_cumulative_paired_profits():
    # Median cumulative increments are +200, +320, -320.
    # The second week's median increment alone is -40.
    plan = {
        seed: [week(n) for n in counts]
        for seed, counts in enumerate([[40, 10, 0], [23, 25, 10], [25, 19, 8]])
    }
    base = {seed: [week(20)] * 3 for seed in plan}
    result = evaluator.evaluate_plan(plan, base, strategy(discount=0), "S1", market(campaign=3), 0)
    assert result.breakpoint.week == 3
    assert result.peak_week == 2


def test_breakpoint_requires_strict_loss_after_a_positive_week():
    # Cumulative increments are -40, +160, 0, -40, -80.
    result = evaluator.evaluate_plan(
        {1: [week(n) for n in [9, 15, 6, 9, 9]]},
        {1: [week(10)] * 5},
        strategy(discount=0),
        "S1",
        market(campaign=4, post=1),
        0,
    )
    assert result.breakpoint.week == 4
    assert result.breakpoint.cause == "discount_cost"
    assert result.peak_week == 2


@pytest.mark.parametrize(
    "bad",
    [
        replace(week(1), units_us=-1),
        replace(week(1), discounted_units_us=-1),
        replace(week(1), revenue_us=float("nan")),
    ],
)
def test_invalid_week_stats_rejected(bad):
    with pytest.raises(ValueError):
        evaluator.evaluate_plan({1: [bad]}, {1: [week(1)]}, strategy(), "S1", market(), 0)


@pytest.mark.parametrize("base", [{2: [week(1)]}, {1: [week(1), week(1)]}])
def test_paired_runs_require_matching_seeds_and_weeks(base):
    with pytest.raises(ValueError):
        evaluator.evaluate_plan({1: [week(1)]}, base, strategy(), "S1", market(), 0)


def test_discounted_units_outside_offer_window_rejected():
    with pytest.raises(ValueError):
        evaluator.evaluate_plan(
            {1: [week(1, discounted=1), week(1, discounted=1)]},
            {1: [week(1), week(1)]},
            strategy(),
            "S1",
            market(campaign=1, post=1),
            0,
        )


def test_named_smart_goal_uses_the_target_kpi():
    result = evaluator.evaluate_plan(
        {1: [week(12, discounted=12, new=3)]},
        {1: [week(10, new=1)]},
        strategy(),
        "S1",
        market(),
        0,
        target=SmartGoal(metric="new_buyers", target=2, unit="buyers"),
    )
    assert result.kpi.incr_profit.p50 == -40
    assert result.kpi.p_hit_target == 1


def test_leakage_target_is_unavailable_without_paired_shopper_units():
    goal = SmartGoal(metric="subsidy_leakage", target=0, unit="percent")
    result = evaluator.evaluate_plan(
        {1: [week(12, discounted=12)]}, {1: [week(10)]}, strategy(), "S1", market(), 0, goal
    )
    assert result.kpi.leakage is None
    assert result.kpi.p_hit_target is None


def test_iroi_percentage_target_uses_ratio_units():
    plan, base = {1: [week(12, discounted=12)]}, {1: [week(8)]}
    # Incremental profit is ฿40 on ฿120 spend, above the 30% target.
    goal = SmartGoal(metric="iROI", target=30, unit="%")
    result = evaluator.evaluate_plan(plan, base, strategy(), "S1", market(), 0, goal)
    assert result.kpi.p_hit_target == 1
    with pytest.raises(ValueError, match="require THB"):
        evaluator.evaluate_plan(
            plan,
            base,
            strategy(),
            "S1",
            market(),
            0,
            SmartGoal(metric="incremental_profit", target=30, unit="%"),
        )


def test_explanation_compares_the_crossing_week_and_calm_scenario():
    plan = strategy()
    economics = market(post=1)
    baseline = {1: [week(8), week(10)]}
    calm = evaluator.evaluate_plan(
        {1: [week(12, discounted=12), week(10)]},
        baseline,
        plan,
        "S1",
        economics,
        0,
    )
    stress = evaluator.evaluate_plan(
        {1: [week(12, discounted=12), week(5)]},
        baseline,
        plan,
        "S3",
        economics,
        0,
    )
    quiet, undercut = evaluator.explain_breakpoints([calm, stress])
    assert quiet.breakpoint.week is None
    assert undercut.breakpoint.cause == "rival undercut"
    assert undercut.breakpoint.share == 1  # (+40 - (-160)) / 160 is capped at 1.
    # Arena finalizes the scenario set through summarize() before returning the results.
    arena_results = [calm.model_copy(deep=True), stress.model_copy(deep=True)]
    evaluator.summarize(arena_results, {})
    assert arena_results[1].breakpoint == undercut.breakpoint

    # Both scenarios break, despite positive absolute profit in the post week.
    calm_dip = evaluator.evaluate_plan(
        {1: [week(12, discounted=12), week(8)]},
        baseline,
        plan,
        "S1",
        economics,
        0,
    )
    dip, worse_dip = evaluator.explain_breakpoints([calm_dip, stress])
    assert dip.breakpoint.cause == worse_dip.breakpoint.cause == "post_campaign_dip"
    assert worse_dip.breakpoint.share == pytest.approx(0.75)  # (-40 - (-160)) / 160.

    campaign_loss = evaluator.evaluate_plan(
        {1: [week(8, discounted=8), week(10)]},
        baseline,
        plan,
        "S1",
        economics,
        0,
    )
    assert evaluator.explain_breakpoints([campaign_loss])[0].breakpoint.cause == "never_profitable"


def test_weighted_ranking_and_loss_filter_keep_the_user_plan():
    fragile = strategy("user", discount=0, is_user_plan=True)
    robust = strategy("robust", discount=0)
    baseline = strategy("baseline", discount=0, is_baseline=True)
    base = {1: [week(10)]}
    results = [
        evaluator.evaluate_plan({1: [week(units)]}, base, plan, scenario, market(), 0)
        for plan, counts in [
            (baseline, [10, 10, 10]),
            (fragile, [15, 12, 7]),
            (robust, [12, 12, 12]),
        ]
        for scenario, units in zip(("S1", "S2", "S3"), counts, strict=True)
    ]

    # Fragile earns +200, +80, -120. Robust earns +80 in every scenario.
    stress_heavy = evaluator.rank_plans(results, {"S1": 2, "S2": 2, "S3": 6}, max_loss_prob=1)
    assert stress_heavy[0].strategy_id == "robust"
    by_id = {entry.strategy_id: entry for entry in stress_heavy}
    assert by_id["robust"].score == 80
    assert by_id["user"].score == pytest.approx(-16)
    assert by_id["user"].reason
    assert not by_id["user"].filtered

    calm_weights = {"S1": 8, "S2": 1, "S3": 1}
    calm_heavy = evaluator.rank_plans(results, calm_weights, max_loss_prob=1)
    assert calm_heavy[0].strategy_id == "user"
    filtered = evaluator.rank_plans(results, calm_weights)
    assert filtered[0].strategy_id == "robust"
    assert next(entry for entry in filtered if entry.strategy_id == "user").filtered
    assert evaluator.rank_plans(results, max_loss_prob=1)[0].strategy_id == "robust"


def test_ranking_breaks_score_ties_by_downside_then_max_regret():
    # Each extra unit earns ฿40. Three candidates average 2 extra units and tie.
    gains = {
        "higher-regret": (1, 1, 4),
        "lower-regret": (4, 1, 1),
        "steady": (2, 2, 2),
        "specialist": (5, 0, 0),
    }
    results = [
        evaluator.evaluate_plan(
            {1: [week(10 + gain)]},
            {1: [week(10)]},
            strategy(plan_id, discount=0),
            scenario,
            market(),
            0,
        )
        for plan_id, units in gains.items()
        for scenario, gain in zip(("S1", "S2", "S3"), units, strict=True)
    ]
    ranked = evaluator.rank_plans(results, {"S1": 1, "S2": 1, "S3": 1})
    assert [entry.strategy_id for entry in ranked] == [
        "steady",
        "lower-regret",
        "higher-regret",
        "specialist",
    ]


def test_aa_lift_warns_only_beyond_two_percentage_points():
    baseline = {7: [week(100)]}
    with catch_warnings():
        simplefilter("error")
        assert evaluator.aa_lift(baseline, baseline) == 0
        assert evaluator.aa_lift(baseline, {7: [week(102)]}) == pytest.approx(2)
    with pytest.warns():
        assert evaluator.aa_lift(baseline, {7: [week(97)]}) == pytest.approx(-3)


@pytest.mark.parametrize(("sales", "expected"), [(0.51, 2), (0.49, -2)])
def test_aa_lift_at_exact_decimal_boundary_does_not_warn(sales, expected):
    with catch_warnings():
        simplefilter("error")
        assert evaluator.aa_lift(
            {7: [replace(week(1), revenue_us=0.50)]},
            {7: [replace(week(1), revenue_us=sales)]},
        ) == pytest.approx(expected)

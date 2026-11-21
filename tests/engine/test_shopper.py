import numpy as np
import pytest

from contracts.models import RivalScenario, Strategy
from engine import shopper

N = 6000
WEEKS = 6
CAMPAIGN = 4
SEED = 7


def make_strategy(
    discount=30.0,
    targets=(),
    baseline=False,
    cap=None,
    flash_days=None,
    sid=None,
):
    return Strategy(
        id="baseline" if baseline else (sid or f"plan-{discount:g}"),
        name="test",
        discount_pct=0 if baseline else discount,
        mechanic="none" if baseline else "percent_off",
        target_segments=list(targets),
        message="test",
        channel_mix={} if baseline else {"shopee": 0.5, "line": 0.5},
        timing={"flash_days": flash_days},
        cap_per_user=cap,
        budget_split={},
        contingency_rules=[],
        rationale="test",
        is_baseline=baseline,
    )


def undercut(depth=30.0, delay=0, duration=4):
    return RivalScenario(
        rival="764", scenario="S3", reaction="undercut", depth=depth, delay_weeks=delay,
        duration_weeks=duration, likelihood=0.1, ci_low=0.05, ci_high=0.2,
        evidence_ids=[], rival_score=0.8,
    )


@pytest.fixture(scope="module")
def population():
    return shopper.build_population(N, seed=SEED)


def run(population, strategy, rivals=None, appeal=None, seed=SEED):
    return shopper.simulate(population, strategy, WEEKS, CAMPAIGN, seed, rivals, appeal)


def total(stats, field, weeks=range(1, WEEKS + 1)):
    return sum(getattr(stats[w - 1], field) for w in weeks)


def test_population_is_deterministic_and_has_a_holdout(population):
    again = shopper.build_population(N, seed=SEED)
    assert np.array_equal(population.persona_idx, again.persona_idx)
    assert np.array_equal(population.holdout, again.holdout)
    assert 0.07 < population.holdout.mean() < 0.13
    assert set(population.preferred) <= {0, 1, 2}
    assert population.sensitivity.mean() == pytest.approx(1.0)


def test_same_seed_same_result_different_seed_differs(population):
    plan = make_strategy(30)
    assert run(population, plan) == run(population, plan)
    assert run(population, plan) != run(population, plan, seed=SEED + 1)


def test_discount_lifts_campaign_sales_against_paired_baseline(population):
    base = run(population, make_strategy(baseline=True))
    plan = run(population, make_strategy(30))
    campaign = range(1, CAMPAIGN + 1)
    assert total(plan, "buyers_us", campaign) > total(base, "buyers_us", campaign)
    assert total(plan, "units_us", campaign) > total(base, "units_us", campaign)
    assert total(plan, "exposed", campaign) > 0
    assert total(base, "exposed") == 0


def test_post_campaign_dip_from_stockpiling_and_waiting(population):
    base = run(population, make_strategy(baseline=True))
    plan = run(population, make_strategy(40))
    post = range(CAMPAIGN + 1, WEEKS + 1)
    assert total(plan, "units_us", post) < total(base, "units_us", post)
    assert total(plan, "waiting", post) > total(base, "waiting", post)


def test_holdout_never_sees_our_offer(population):
    base = run(population, make_strategy(baseline=True))
    plan = run(population, make_strategy(40))
    assert [s.holdout_buyers_us for s in plan] == [s.holdout_buyers_us for s in base]


def test_price_sensitive_segment_responds_more_than_loyal(population):
    base = run(population, make_strategy(baseline=True))
    campaign = range(1, CAMPAIGN + 1)

    def lift_per_exposed(segment):
        plan = run(population, make_strategy(30, targets=(segment,), sid=f"to-{segment}"))
        exposed = total(plan, "exposed", range(1, 2))  # unique shoppers exposed in week 1
        lift = total(plan, "buyers_us", campaign) - total(base, "buyers_us", campaign)
        return lift / max(exposed, 1)

    assert lift_per_exposed("deal_hunter") > lift_per_exposed("loyal")


def test_rival_undercut_takes_buyers_from_us(population):
    plan = make_strategy(20)
    calm = run(population, plan)
    attacked = run(population, plan, rivals=[undercut(30)])
    campaign = range(1, CAMPAIGN + 1)
    assert total(attacked, "buyers_us", campaign) < total(calm, "buyers_us", campaign)
    assert total(attacked, "buyers_rival", campaign) > total(calm, "buyers_rival", campaign)


def test_cap_per_user_limits_discounted_purchases(population):
    plan = make_strategy(40, cap=1)
    state = shopper.initial_state(population, SEED)
    for week in range(1, CAMPAIGN + 1):
        offer = shopper.offer_from_strategy(plan, week, CAMPAIGN)
        _, state = shopper.step(population, state, offer, SEED, week)
    assert state.redemptions.max() <= 1


def test_message_appeal_raises_our_share(population):
    plan = make_strategy(20, sid="msg")
    liked = {(pid, "msg"): 1.0 for pid in population.persona_ids}
    disliked = {(pid, "msg"): -1.0 for pid in population.persona_ids}
    campaign = range(1, CAMPAIGN + 1)
    assert total(run(population, plan, appeal=liked), "buyers_us", campaign) > total(
        run(population, plan, appeal=disliked), "buyers_us", campaign
    )


def test_offer_from_strategy_windows():
    plan = make_strategy(25)
    rival = undercut(35, delay=1, duration=2)
    offers = [shopper.offer_from_strategy(plan, w, CAMPAIGN, [rival]) for w in range(1, 7)]
    assert [o.active for o in offers] == [True, True, True, True, False, False]
    assert [o.rival_discount_pct for o in offers] == [0, 35, 35, 0, 0, 0]
    flash = shopper.offer_from_strategy(make_strategy(30, flash_days=2), 2, CAMPAIGN)
    assert not flash.active
    base = shopper.offer_from_strategy(make_strategy(baseline=True), 1, CAMPAIGN)
    assert not base.active and base.our_discount_pct == 0


def test_week_row_profit_is_revenue_minus_costs(population):
    stats = run(population, make_strategy(baseline=True))[0]
    market = {"platform_fee_pct": 7, "payment_fee_pct": 3, "unit_cost": 100,
              "shipping_cost_per_order": 10}
    row = shopper.week_row(1, stats, market)
    expected = stats.revenue_us * 0.9 - stats.units_us * 100 - stats.buyers_us * 10
    assert row.profit == pytest.approx(expected, abs=0.01)
    assert row.buyers == stats.buyers_us


def test_stats_are_scaled_by_households_per_agent():
    small = shopper.build_population(2000, seed=SEED, params=shopper.ShopperParams(
        households_per_agent=1.0))
    big = shopper.build_population(2000, seed=SEED, params=shopper.ShopperParams(
        households_per_agent=50.0))
    plan = make_strategy(20)
    one = shopper.simulate(small, plan, WEEKS, CAMPAIGN, SEED)
    fifty = shopper.simulate(big, plan, WEEKS, CAMPAIGN, SEED)
    assert fifty[0].revenue_us == pytest.approx(one[0].revenue_us * 50)
    assert fifty[0].buyers_us == one[0].buyers_us * 50
    assert big.households == pytest.approx(2000 * 50)
    assert shopper.ShopperParams().households_per_agent == 50.0


def media_plan(media_share, budget=80_000):
    return make_strategy(20).model_copy(update={
        "timing": {"budget_thb": budget},
        "budget_split": {"discount": 1 - media_share, "media": media_share},
    })


def test_media_budget_buys_reach_with_diminishing_returns(population):
    none = shopper.media_reach(population, media_plan(0.0))
    some = shopper.media_reach(population, media_plan(0.2))
    lots = shopper.media_reach(population, media_plan(0.2, budget=8_000_000))
    params = shopper.ShopperParams()
    assert none == pytest.approx(params.organic_reach)
    assert params.organic_reach < some < lots <= 1.0
    assert shopper.media_reach(population, make_strategy(20)) == 1.0  # no budget info
    low = run(population, media_plan(0.0))
    high = run(population, media_plan(0.2, budget=8_000_000))
    assert total(low, "exposed", range(1, 2)) < total(high, "exposed", range(1, 2))


def test_rival_undercut_keeps_hurting_after_it_ends(population):
    plan = make_strategy(20)
    calm = run(population, plan)
    attacked = run(population, plan, rivals=[undercut(30, delay=0, duration=4)])
    post = range(CAMPAIGN + 1, WEEKS + 1)
    # Some shoppers who switched during the undercut keep buying the rival.
    assert total(attacked, "buyers_us", post) < total(calm, "buyers_us", post)


def test_rival_promotion_never_raises_our_full_price_sales(population):
    # Regression: a rival discount used to pull shoppers into the category and then let
    # them buy (and stock up on) our full-price product.
    base = make_strategy(baseline=True)
    calm = run(population, base)
    attacked = run(population, base, rivals=[undercut(30, delay=0, duration=4)])
    campaign = range(1, CAMPAIGN + 1)
    assert total(attacked, "units_us", campaign) < total(calm, "units_us", campaign)
    assert total(attacked, "buyers_us", campaign) <= total(calm, "buyers_us", campaign)

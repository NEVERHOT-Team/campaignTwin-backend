from copy import deepcopy

import pytest

from agents import strategist
from agents.strategist import (
    PERCENT_MECHANICS,
    RULE_PATTERNS,
    all_distinct,
    enforce,
    propose,
    resolve_bounds,
)
from contracts.campaign_spec import CampaignSpec
from contracts.models import ContextPack, Strategy
from mocks import load_mock


def _spec(**changes) -> CampaignSpec:
    raw = deepcopy(load_mock("campaign_spec")["spec"])
    for key, value in changes.items():
        raw[key] = value
    return CampaignSpec.model_validate(raw)


@pytest.fixture
def spec() -> CampaignSpec:
    return _spec()


@pytest.fixture
def context() -> ContextPack:
    return ContextPack.model_validate(load_mock("context_pack"))


def _generated(plans: list[Strategy]) -> list[Strategy]:
    return [plan for plan in plans if not plan.is_baseline and not plan.is_user_plan]


def test_propose_returns_baseline_user_plan_and_six_generated(spec, context):
    plans = propose(spec, context)
    assert plans[0].is_baseline and plans[0].discount_pct == 0
    assert plans[1].is_user_plan
    assert sum(plan.is_baseline for plan in plans) == 1
    assert sum(plan.is_user_plan for plan in plans) == 1
    assert len(_generated(plans)) == strategist.GENERATED
    assert len({plan.id for plan in plans}) == len(plans)
    assert all(plan.id.startswith("g0-") for plan in _generated(plans))


def test_user_plan_is_kept_as_briefed(spec, context):
    user = propose(spec, context)[1]
    assert user.discount_pct == 40
    assert user.mechanic == "voucher"
    assert user.cap_per_user == 1
    assert set(user.target_segments) == {"loyal", "deal_hunter", "waiter"}
    # 40% is below break-even after unit cost, fees and shipping, so it carries a warning.
    assert "Warning" in user.rationale


def test_generated_plans_respect_every_hard_limit(spec, context):
    bounds = resolve_bounds(spec, context)
    segment_ids = {segment["segment_id"] for segment in context.segments}
    for plan in _generated(propose(spec, context)):
        assert plan.mechanic in bounds.mechanics
        assert plan.discount_pct <= min(bounds.discount_max, bounds.max_unit_discount) + 1e-9
        if plan.mechanic in PERCENT_MECHANICS:
            assert plan.discount_pct >= min(bounds.discount_min, bounds.max_unit_discount) - 1e-9
        assert sum(plan.budget_split.values()) == pytest.approx(1)
        assert set(plan.budget_split) <= {"discount", "media", "fees"}
        assert set(plan.channel_mix) <= set(spec.channels)
        assert sum(plan.channel_mix.values()) == pytest.approx(1)
        assert set(plan.target_segments) <= segment_ids
        assert bounds.budget_min - 1 <= plan.timing["budget_thb"] <= bounds.budget_max + 1
        assert all(any(p.match(rule) for p in RULE_PATTERNS) for rule in plan.contingency_rules)


def test_plans_differ_on_at_least_two_knobs(spec, context):
    plans = propose(spec, context)
    bounds = resolve_bounds(spec, context)
    assert all_distinct(plans[1:], bounds)


def test_propose_is_deterministic(spec, context):
    first = [plan.model_dump() for plan in propose(spec, context)]
    second = [plan.model_dump() for plan in propose(spec, context)]
    assert first == second


def test_failing_proposer_falls_back_to_templates(spec, context):
    def broken(*_args):
        raise TimeoutError("model unavailable")

    def empty(*_args):
        return None

    expected = [plan.model_dump() for plan in propose(spec, context)]
    assert [plan.model_dump() for plan in propose(spec, context, broken)] == expected
    assert [plan.model_dump() for plan in propose(spec, context, empty)] == expected


def test_proposer_output_is_enforced_not_trusted(spec, context):
    wild = Strategy(
        id="wild",
        name="Wild idea",
        discount_pct=90,
        mechanic="gift",
        target_segments=["loyal", "martians"],
        message="Everything must go",
        channel_mix={"tiktok_shop": 1.0},
        timing={"budget_thb": 10_000_000},
        cap_per_user=0,
        budget_split={"discount": 3, "media": 1},
        contingency_rules=["do something clever", "if week>=3 and loss then end_early"],
        rationale="LLM proposal.",
    )
    bounds = resolve_bounds(spec, context)
    fixed = enforce(wild, spec, context, bounds)
    assert fixed.mechanic in bounds.mechanics
    assert fixed.discount_pct <= bounds.max_unit_discount
    assert fixed.target_segments == ["loyal"]
    assert set(fixed.channel_mix) == set(spec.channels)
    assert fixed.timing["budget_thb"] == pytest.approx(bounds.budget_max)
    assert fixed.cap_per_user is None
    assert fixed.budget_split == pytest.approx({"discount": 0.75, "media": 0.25})
    assert fixed.contingency_rules == ["if week>=3 and loss then end_early"]
    assert "[Adjusted:" in fixed.rationale

    plans = propose(spec, context, lambda *_: [wild])
    assert plans[2].id == "g0-wild"
    assert len(_generated(plans)) == strategist.GENERATED


def test_explicit_bounds_and_locked_segments(context):
    raw = deepcopy(load_mock("campaign_spec")["spec"])
    raw["strategist_bounds"] = {
        "discount_pct": [30, 40],
        "mechanics_allowed": ["voucher", "percent_off"],
        "segments_locked": True,
    }
    raw["field_meta"]["strategist_bounds"] = {"source": "user", "confidence": 1}
    spec = CampaignSpec.model_validate(raw)
    bounds = resolve_bounds(spec, context)
    assert bounds.labels["discount_pct"] == "data"
    for plan in _generated(propose(spec, context)):
        assert plan.mechanic in {"voucher", "percent_off"}
        assert set(plan.target_segments) == {"loyal", "deal_hunter", "waiter"}


def test_must_avoid_drops_matching_plans(context):
    raw = deepcopy(load_mock("campaign_spec")["spec"])
    raw["constraints"]["must_avoid"] = ["flash"]
    spec = CampaignSpec.model_validate(raw)
    assert all("flash" not in plan.name.lower() for plan in propose(spec, context))


def test_spec_without_budget_is_rejected(context):
    raw = deepcopy(load_mock("campaign_spec")["spec"])
    raw["status"] = "draft"
    raw["budget"]["total_thb"] = None
    spec = CampaignSpec.model_validate(raw)
    with pytest.raises(ValueError, match="budget.total_thb"):
        propose(spec, context)


def test_unit_discount_limit_counts_cost_fees_and_shipping(spec, context):
    from data.loaders import load_market

    bounds = resolve_bounds(spec, context)
    market = load_market()
    price, cost = market["ref_price"], market["unit_cost"]
    fees = (market["platform_fee_pct"] + market["payment_fee_pct"]) / 100
    shipping = market["shipping_cost_per_order"]
    # At the limit, contribution per unit is break-even once shipping is spread over
    # at least one unit per order, so the limit sits below the no-shipping ceiling.
    no_shipping = 100 * (1 - cost / (price * (1 - fees)))
    with_one_unit = 100 * (1 - (cost + shipping) / (price * (1 - fees)))
    assert with_one_unit <= bounds.max_unit_discount < no_shipping
    for plan in _generated(propose(spec, context)):
        net = price * (1 - plan.discount_pct / 100)
        assert net * (1 - fees) - cost - shipping / 1.0 > -shipping  # never far below cost


def test_explicit_zero_budget_flex_locks_the_budget(context):
    raw = deepcopy(load_mock("campaign_spec")["spec"])
    raw["strategist_bounds"] = {"budget_flex_pct": 0}
    raw["field_meta"]["strategist_bounds"] = {"source": "user", "confidence": 1}
    spec = CampaignSpec.model_validate(raw)
    bounds = resolve_bounds(spec, context)
    assert bounds.labels["budget_flex_pct"] == "data"
    assert bounds.budget_min == bounds.budget_max == spec.budget.total_thb
    for plan in _generated(propose(spec, context)):
        assert plan.timing["budget_thb"] == pytest.approx(spec.budget.total_thb)


def test_media_is_capped_at_what_reaches_the_target(context):
    raw = deepcopy(load_mock("campaign_spec")["spec"])
    raw["budget"]["total_thb"] = 500_000
    spec = CampaignSpec.model_validate(raw)
    bounds = resolve_bounds(spec, context)
    plans = {plan.id: plan for plan in _generated(propose(spec, context))}
    for plan in plans.values():
        media = plan.budget_split.get("media", 0.0) * plan.timing["budget_thb"]
        assert media <= bounds.media_needed(plan.target_segments) + 1
        assert sum(plan.budget_split.values()) == pytest.approx(1)
    targeted = plans["g0-targeted"]
    # A plan for a few segments needs far less media than the 25% template share.
    assert targeted.budget_split["media"] < 0.25
    assert "media" in targeted.rationale and "enough to reach the target" in targeted.rationale
    assert bounds.media_needed(["deal_hunter"]) < bounds.media_needed([])

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
from data.loaders import load_demo_brief, load_market, load_personas
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


@pytest.mark.parametrize("margin_floor", [0, 10])
def test_tissue_discounts_protect_generated_plans_and_preserve_user_offer(context, margin_floor):
    raw = deepcopy(load_demo_brief()["user_plan_spec"])
    raw.setdefault("constraints", {})["margin_floor_pct"] = margin_floor
    raw["field_meta"]["constraints.margin_floor_pct"] = {"source": "user", "confidence": 1}
    spec = CampaignSpec.model_validate(raw)
    market = load_market()
    personas = load_personas()
    units = sum(p["weight"] * p["avg_units"] for p in personas) / sum(
        p["weight"] for p in personas
    )
    fees = (market["platform_fee_pct"] + market["payment_fee_pct"]) / 100
    cost = market["unit_cost"] + market["shipping_cost_per_order"] / units
    price = market["ref_price"]
    break_even = 100 * (1 - cost / (price * (1 - fees)))
    floor_cap = 100 * (1 - cost / (price * (1 - fees - margin_floor / 100)))

    baseline, user, *generated = propose(spec, context)
    assert (baseline.discount_pct, baseline.below_cost) == (0, False)
    assert (user.discount_pct, user.below_cost) == (40, True)
    assert len(generated) == strategist.GENERATED
    for plan in generated:
        assert plan.discount_pct <= min(break_even - 1, floor_cap) + 1e-9
        assert plan.below_cost is False
        assert price * (1 - plan.discount_pct / 100) * (1 - fees) >= cost


def test_safety_margin_preserves_tissue_templates_already_within_cap(context, monkeypatch):
    spec = CampaignSpec.model_validate(load_demo_brief()["user_plan_spec"])
    limit = strategist.break_even_discount_pct() - 1
    current = {p.id: p.discount_pct for p in _generated(propose(spec, context))}
    monkeypatch.setattr(strategist, "DISCOUNT_SAFETY_POINTS", 0)
    original = {p.id: p.discount_pct for p in _generated(propose(spec, context))}
    assert original["g0-deep"] > limit
    assert original["g0-targeted"] <= limit
    assert original["g0-flash"] <= limit
    for plan_id, discount in original.items():
        if discount <= limit:
            assert current[plan_id] == discount


def test_plans_when_even_list_price_sells_below_cost(context, monkeypatch):
    spec = CampaignSpec.model_validate(load_demo_brief()["user_plan_spec"])
    market = dict(load_market())
    market["unit_cost"] = market["ref_price"]
    monkeypatch.setattr(strategist, "load_market", lambda: market)
    plans = propose(spec, context)
    assert [(plan.id, plan.discount_pct) for plan in plans] == [("baseline", 0), ("user-plan", 40)]
    assert all(plan.below_cost for plan in plans)


@pytest.mark.parametrize(
    ("face_pct", "effective_pct", "below_cost"),
    [(10, 10, False), (26, 26, False), (40, 40, True), (120, 100, True)],
)
def test_fixed_baht_voucher_effective_discount_uses_market_price(
    context, face_pct, effective_pct, below_cost
):
    raw = deepcopy(load_demo_brief()["user_plan_spec"])
    amount = load_market()["ref_price"] * face_pct / 100
    raw["offer"]["mechanics"] = [{"type": "voucher", "value": amount, "value_unit": "thb"}]
    spec = CampaignSpec.model_validate(raw)

    baseline, user, *generated = propose(spec, context)
    assert user.mechanic == "voucher"
    assert user.discount_pct == pytest.approx(effective_pct)
    assert user.below_cost is below_cost
    assert baseline.discount_pct == 0
    assert all(plan.below_cost is False for plan in generated)


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


def test_new_buyer_voucher_is_offered_and_keeps_its_eligibility(spec, context):
    plans = {plan.id: plan for plan in propose(spec, context)}
    newbuyer = plans["g0-newbuyer"]
    assert newbuyer.timing["eligibility"] == "new_only"
    assert newbuyer.mechanic == "voucher" and newbuyer.cap_per_user == 1
    assert all("eligibility" not in plans[pid].timing for pid in plans if pid != "g0-newbuyer")

"""Strategist AI, generation 0: diverse candidate plans inside the marketer's bounds.

`propose(spec, context)` always returns the no-campaign baseline, the marketer's own plan
and up to six generated plans. Generated plans come from an optional proposer (the LLM,
wired in later) and are topped up from deterministic templates, so the demo works without
a model. Every generated plan passes `enforce()`; nothing from a proposer is trusted as is.

Strategy field conventions (shared with engine/, pending a contract decision by Big):
- discount_pct: percentage points (0-100) off list price. For free_shipping, bundle and
  gift it is the equivalent discount value.
- mechanic: a CampaignSpec MechanicType, or "none" for the baseline.
- target_segments: ContextPack segment_ids. An empty list means every shopper.
- channel_mix: {channel: share} over spec.channels, shares sum to 1.
- budget_split: {"discount", "media", "fees"} shares of the plan budget, sum to 1.
- timing: {"start_date", "end_date" (ISO), "event", "budget_thb", "flash_days"}.
  budget_thb lives here because Strategy has no budget field yet.
- contingency_rules: strings matching RULE_PATTERNS, so the engine can parse them.
"""

import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

from contracts.campaign_spec import CampaignSpec
from contracts.models import ContextPack, Strategy
from data.loaders import load_market, load_personas, load_unit_econ

GENERATED = 6
DEFAULT_BUDGET_FLEX_PCT = 10.0
DEFAULT_MAX_DISCOUNT_PCT = 50.0
# Break-even by default. market.json's margin_floor_pct caps *rival* discounts in
# data/rival.py; a floor for our own plans comes from spec.constraints.margin_floor_pct.
DEFAULT_MARGIN_FLOOR_PCT = 0.0
DEFAULT_UNITS_PER_ORDER = 1.1
DEFAULT_SPLIT = {"discount": 0.7, "media": 0.25, "fees": 0.05}
PERCENT_MECHANICS = {"percent_off", "voucher", "flash_sale", "coins_cashback"}
DEFAULT_EXTRA_MECHANICS = ("percent_off", "voucher", "free_shipping", "bundle", "flash_sale")

RULE_PATTERNS = (
    re.compile(r"^if rival_undercuts then shift_budget:[a-z0-9_]+$"),
    re.compile(r"^if share_drop>\d+(\.\d+)? then cap_per_user:\d+$"),
    re.compile(r"^if week>=\d+ and loss then end_early$"),
)

Proposer = Callable[[CampaignSpec, ContextPack, "Bounds"], list[Strategy] | None]


@dataclass(frozen=True)
class Bounds:
    discount_min: float
    discount_max: float
    max_unit_discount: float
    mechanics: tuple[str, ...]
    budget_thb: float
    budget_min: float
    budget_max: float
    segments_locked: bool
    user_segments: tuple[str, ...]
    labels: dict[str, str] = field(default_factory=dict)

    def label_note(self) -> str:
        return ", ".join(f"{name}: {label}" for name, label in sorted(self.labels.items()))


# ---------------------------------------------------------------- bounds


def _user_discount(spec: CampaignSpec) -> float | None:
    for mechanic in spec.offer.mechanics:
        if mechanic.percentage is not None:
            return float(mechanic.percentage)
    return None


def _user_segments(spec: CampaignSpec, context: ContextPack) -> tuple[str, ...]:
    hints = {hint.strip().lower() for hint in spec.audience.include_hints}
    found = []
    for segment in context.segments:
        keys = {
            str(segment.get("segment_id", "")).lower(),
            str(segment.get("archetype", "")).lower(),
        }
        if hints & keys:
            found.append(segment["segment_id"])
    return tuple(found)


def _units_per_order() -> tuple[float, str]:
    """Weighted mean units per order across personas (spreads shipping over units)."""
    try:
        personas = load_personas()
        weights = [max(float(p.get("weight", 0)), 0.0) for p in personas]
        units = [max(float(p.get("avg_units", 1.0)), 0.1) for p in personas]
        if sum(weights) > 0:
            return sum(w * u for w, u in zip(weights, units, strict=True)) / sum(weights), "data"
    except Exception:
        pass
    return DEFAULT_UNITS_PER_ORDER, "default"


def _unit_costs(spec: CampaignSpec) -> tuple[float, float, float, float, dict[str, str]]:
    """(price, unit cost, fee share, shipping per unit) from the spec, else market data.

    Same unit economics the engine uses in shopper.week_row, so a plan the strategist
    allows never loses money per unit by construction.
    """
    labels: dict[str, str] = {}
    try:
        market = load_market()
        price = float(market["ref_price"])
        cost = float(market["unit_cost"])
        fees = (float(market["platform_fee_pct"]) + float(market["payment_fee_pct"])) / 100
        shipping = float(market["shipping_cost_per_order"])
        source = "data"
    except Exception:  # older checkouts without market.json
        econ = load_unit_econ()
        price = float(econ["price_thb"])
        cost = price * (1 - float(econ["margin_pct"]) / 100)
        fees = (float(econ["shopee_commission_pct"]) + float(econ["shopee_txn_fee_pct"])) / 100
        shipping = 0.0
        source = "default"
    labels.update(price=source, unit_cost=source, fee_pct=source, shipping=source)
    economics = spec.unit_economics
    if economics and economics.avg_price_thb is not None:
        price, labels["price"] = float(economics.avg_price_thb), "data"
    if economics and economics.margin_pct is not None:
        cost, labels["unit_cost"] = price * (1 - economics.margin_pct / 100), "data"
    if economics and economics.fee_pct is not None:
        fees, labels["fee_pct"] = economics.fee_pct / 100, "data"
    units, labels["units_per_order"] = _units_per_order()
    return price, cost, fees, shipping / max(units, 0.1), labels


def _max_unit_discount(spec: CampaignSpec) -> tuple[float, dict[str, str]]:
    """Largest discount (%) whose unit contribution stays at or above the margin floor.

    contribution = price (1 - d)(1 - fees) - unit_cost - shipping_per_unit
    and it must be >= floor x price (1 - d), so
    d <= 1 - (unit_cost + shipping_per_unit) / (price (1 - fees - floor)).
    """
    price, cost, fees, shipping, labels = _unit_costs(spec)
    floor = spec.constraints.margin_floor_pct
    if floor is None:
        floor, labels["margin_floor_pct"] = DEFAULT_MARGIN_FLOOR_PCT, "default"
    else:
        labels["margin_floor_pct"] = "data"
    keep = price * (1 - fees - floor / 100)
    if keep <= 0:
        return 0.0, labels
    limit = 100 * (1 - (cost + shipping) / keep)
    return max(0.0, min(100.0, limit)), labels


def resolve_bounds(spec: CampaignSpec, context: ContextPack) -> Bounds:
    if spec.budget.total_thb is None or spec.timing.start_date is None:
        raise ValueError("Strategist needs a CampaignSpec with budget.total_thb and timing")
    if not context.segments:
        raise ValueError("Strategist needs at least one segment in the ContextPack")
    given = spec.strategist_bounds
    labels: dict[str, str] = {}
    cap = spec.budget.max_discount_pct
    if cap is None:
        cap = DEFAULT_MAX_DISCOUNT_PCT
    user_discount = _user_discount(spec)

    if given and given.discount_pct:
        low, high = given.discount_pct
        labels["discount_pct"] = "data"
    elif user_discount is not None:
        low, high = user_discount * 0.5, user_discount * 1.5
        labels["discount_pct"] = "default"
    else:
        low, high = 5.0, 30.0
        labels["discount_pct"] = "default"
    high = min(high, cap)
    low = min(low, high)

    if given and given.mechanics_allowed:
        mechanics = tuple(given.mechanics_allowed)
        labels["mechanics"] = "data"
    else:
        ordered = [m.type for m in spec.offer.mechanics] + list(DEFAULT_EXTRA_MECHANICS)
        mechanics = tuple(dict.fromkeys(ordered))
        labels["mechanics"] = "default"

    # An explicit 0 locks the budget; only an unset value falls back to the default.
    if given and "budget_flex_pct" in given.model_fields_set:
        flex, labels["budget_flex_pct"] = float(given.budget_flex_pct), "data"
    else:
        flex, labels["budget_flex_pct"] = DEFAULT_BUDGET_FLEX_PCT, "default"
    total = float(spec.budget.total_thb)

    unit_limit, margin_labels = _max_unit_discount(spec)
    labels.update(margin_labels)
    return Bounds(
        discount_min=round(low, 1),
        discount_max=round(high, 1),
        max_unit_discount=unit_limit,
        mechanics=mechanics,
        budget_thb=total,
        budget_min=total * (1 - flex / 100),
        budget_max=total * (1 + flex / 100),
        segments_locked=bool(given and given.segments_locked),
        user_segments=_user_segments(spec, context),
        labels=labels,
    )


# ---------------------------------------------------------------- shared helpers


def _equal_mix(channels: list[str]) -> dict[str, float]:
    if not channels:
        return {}
    share = 1 / len(channels)
    return {channel: share for channel in channels}


def _timing(spec: CampaignSpec, budget: float, flash_days: int | None = None) -> dict[str, Any]:
    return {
        "start_date": spec.timing.start_date.isoformat() if spec.timing.start_date else None,
        "end_date": spec.timing.end_date.isoformat() if spec.timing.end_date else None,
        "event": spec.timing.event,
        "budget_thb": round(budget, 2),
        "flash_days": flash_days,
    }


def _normalize(shares: dict[str, float], allowed: list[str] | None = None) -> dict[str, float]:
    cleaned = {
        key: max(0.0, float(value)) for key, value in shares.items()
        if (allowed is None or key in allowed) and math.isfinite(float(value))
    }
    total = sum(cleaned.values())
    if total <= 0:
        return {}
    return {key: value / total for key, value in cleaned.items()}


def _segments_by(context: ContextPack, key: str, reverse: bool = True) -> list[str]:
    ranked = sorted(context.segments, key=lambda s: float(s.get(key, 0)), reverse=reverse)
    return [segment["segment_id"] for segment in ranked]


def _segment_with_archetype(context: ContextPack, archetype: str) -> str | None:
    for segment in context.segments:
        if segment.get("archetype") == archetype:
            return segment["segment_id"]
    return None


def _message(spec: CampaignSpec, text: str) -> str:
    brand = spec.brand.name or "Our brand"
    event = spec.timing.event
    prefix = f"{brand} {event}" if event and event != "none" else brand
    return f"{prefix}: {text}"


# ---------------------------------------------------------------- fixed plans


def baseline(spec: CampaignSpec) -> Strategy:
    return Strategy(
        id="baseline",
        name="No campaign",
        discount_pct=0,
        mechanic="none",
        target_segments=[],
        message="",
        channel_mix={},
        timing=_timing(spec, 0),
        cap_per_user=None,
        budget_split={},
        contingency_rules=[],
        rationale="No-campaign twin. Every plan is measured against it on the same seeds.",
        is_baseline=True,
    )


def user_plan(spec: CampaignSpec, bounds: Bounds) -> Strategy:
    """The marketer's own plan, kept as written. Notes flag anything outside the bounds."""
    first = spec.offer.mechanics[0] if spec.offer.mechanics else None
    discount = float(first.percentage) if first and first.percentage is not None else 0.0
    notes = []
    if discount > bounds.max_unit_discount:
        notes.append(
            f"{discount:g}% is above the {bounds.max_unit_discount:.1f}% that keeps unit "
            "contribution above the margin floor"
        )
    split = DEFAULT_SPLIT
    if spec.budget.split is not None:
        split = _normalize(spec.budget.split.model_dump()) or DEFAULT_SPLIT
    message = spec.offer.message
    text = (message.headline or message.key_message) if message else None
    rationale = "The marketer's plan as briefed."
    if notes:
        rationale += " Warning: " + "; ".join(notes) + "."
    return Strategy(
        id="user-plan",
        name="Your plan",
        discount_pct=discount,
        mechanic=first.type if first else "none",
        target_segments=list(bounds.user_segments),
        message=text or _message(spec, f"{discount:g}% off"),
        channel_mix=_equal_mix(list(spec.channels)),
        timing=_timing(spec, bounds.budget_thb),
        cap_per_user=first.cap_per_user if first else None,
        budget_split=dict(split),
        contingency_rules=[],
        rationale=rationale,
        is_user_plan=True,
    )


# ---------------------------------------------------------------- templates


def template_strategies(
    spec: CampaignSpec, context: ContextPack, bounds: Bounds
) -> list[Strategy]:
    """Six archetype plans built from rules. Also the fallback when the LLM fails."""
    # Whole-number discounts that already respect the margin floor, so messages match.
    high = float(math.floor(min(bounds.discount_max, bounds.max_unit_discount)))
    low = float(math.ceil(min(bounds.discount_min, high)))
    low = min(low, high)
    mid = round((low + high) / 2)
    sensitive = _segments_by(context, "promo_response")
    # Least price-elastic segment (elasticity closest to zero) stands in for "loyal".
    loyal = (
        _segment_with_archetype(context, "loyal")
        or _segments_by(context, "price_elasticity")[0]
    )
    explorer = _segment_with_archetype(context, "explorer")
    indifferent = _segment_with_archetype(context, "indifferent")
    deal_hunter = _segment_with_archetype(context, "deal_hunter") or sensitive[0]
    channels = list(spec.channels)
    mix = _equal_mix(channels)
    flash_mix = _normalize({c: (0.6 if c == "shopee" else 0.4) for c in channels}) or mix
    new_buyers = [s for s in (explorer, indifferent) if s] or sensitive[-2:]
    shipping = "free_shipping" if "free_shipping" in bounds.mechanics else "bundle"

    specs = [
        dict(
            key="deep", name="Deep discount for everyone", discount=high,
            mechanic="percent_off", targets=[], cap=None, budget=bounds.budget_max,
            split={"discount": 0.8, "media": 0.15, "fees": 0.05}, mix=mix,
            rules=["if week>=3 and loss then end_early"],
            why="Maximise reach and volume with the deepest allowed discount.",
            text=f"{high:g}% off everything",
        ),
        dict(
            key="targeted", name="Targeted voucher for price-sensitive shoppers", discount=mid,
            mechanic="voucher", targets=sensitive[:2], cap=1, budget=bounds.budget_thb,
            split={"discount": 0.7, "media": 0.25, "fees": 0.05}, mix=mix,
            rules=[f"if rival_undercuts then shift_budget:{loyal}"],
            why="Spend the discount only where promo response is highest, one use per shopper.",
            text=f"{mid:g}% voucher, one per customer",
        ),
        dict(
            key="loyalty", name="Member voucher for loyal shoppers",
            discount=round(low + (high - low) * 0.3), mechanic="voucher",
            targets=[s for s in (loyal, explorer) if s], cap=1, budget=bounds.budget_thb * 0.9,
            split={"discount": 0.65, "media": 0.3, "fees": 0.05}, mix=mix,
            rules=["if share_drop>5 then cap_per_user:1"],
            why="Defend share with a smaller member-only reward instead of a public price cut.",
            text="members-only voucher",
        ),
        dict(
            key="shipping", name="Free shipping to win new buyers",
            discount=min(10.0, high), mechanic=shipping, targets=new_buyers, cap=None,
            budget=bounds.budget_thb, split={"discount": 0.6, "media": 0.35, "fees": 0.05},
            mix=mix, rules=[],
            why="A non-price offer that rivals rarely match and that attracts first-time buyers.",
            text="free shipping" if shipping == "free_shipping" else "bundle deal",
        ),
        dict(
            key="flash", name="Short flash sale", discount=round(high * 0.9),
            mechanic="flash_sale", targets=[deal_hunter], cap=1, budget=bounds.budget_thb * 0.8,
            split={"discount": 0.75, "media": 0.2, "fees": 0.05}, mix=flash_mix,
            rules=["if week>=3 and loss then end_early"], flash_days=2,
            why="Concentrate the discount in two days to limit pull-forward and rival response.",
            text=f"{round(high * 0.9):g}% flash sale, 48 hours only",
        ),
        dict(
            key="conservative", name="Conservative small discount", discount=low,
            mechanic="percent_off", targets=[], cap=None, budget=bounds.budget_min,
            split={"discount": 0.6, "media": 0.35, "fees": 0.05}, mix=mix, rules=[],
            why="Lowest-risk option: small discount, smallest budget.",
            text=f"{low:g}% off",
        ),
    ]

    plans = []
    for item in specs:
        mechanic = item["mechanic"]
        note = ""
        if mechanic not in bounds.mechanics:
            mechanic, note = bounds.mechanics[0], f" ({item['mechanic']} not allowed)"
        plans.append(Strategy(
            id=f"g0-{item['key']}",
            name=item["name"],
            discount_pct=float(item["discount"]),
            mechanic=mechanic,
            target_segments=list(item["targets"]),
            message=_message(spec, item["text"]),
            channel_mix=dict(item["mix"]),
            timing=_timing(spec, item["budget"], item.get("flash_days")),
            cap_per_user=item["cap"],
            budget_split=dict(item["split"]),
            contingency_rules=list(item["rules"]),
            rationale=item["why"] + note,
        ))
    return plans


# ---------------------------------------------------------------- enforcement


def violates_must_avoid(strategy: Strategy, spec: CampaignSpec) -> bool:
    haystack = f"{strategy.mechanic} {strategy.name} {strategy.message}".lower()
    return any(term.strip().lower() in haystack for term in spec.constraints.must_avoid if term)


def enforce(
    strategy: Strategy, spec: CampaignSpec, context: ContextPack, bounds: Bounds
) -> Strategy:
    """Pull a generated plan back inside every hard limit and record each change."""
    if strategy.is_baseline or strategy.is_user_plan:
        return strategy
    notes: list[str] = []
    segment_ids = {segment["segment_id"] for segment in context.segments}

    mechanic = strategy.mechanic
    if mechanic not in bounds.mechanics:
        notes.append(f"mechanic {mechanic} not allowed, used {bounds.mechanics[0]}")
        mechanic = bounds.mechanics[0]

    discount = float(strategy.discount_pct) if math.isfinite(strategy.discount_pct) else 0.0
    low = bounds.discount_min if mechanic in PERCENT_MECHANICS else 0.0
    high = min(bounds.discount_max, bounds.max_unit_discount)
    low = min(low, high)
    clamped = min(max(discount, low), high)
    if abs(clamped - discount) > 1e-9:
        reason = "margin floor" if discount > bounds.max_unit_discount else "bounds"
        notes.append(f"discount {discount:g}->{clamped:.1f} ({reason})")
    # Round for readability, then clamp again so rounding never crosses a limit.
    discount = min(max(round(clamped, 1), low), high)

    if bounds.segments_locked:
        targets = list(bounds.user_segments)
        if set(targets) != set(strategy.target_segments):
            notes.append("segments locked to the brief")
    else:
        targets = [s for s in dict.fromkeys(strategy.target_segments) if s in segment_ids]
        dropped = set(strategy.target_segments) - set(targets)
        if dropped:
            notes.append(f"unknown segments dropped: {', '.join(sorted(dropped))}")

    channels = list(spec.channels)
    mix = _normalize(strategy.channel_mix, channels)
    if not mix:
        mix = _equal_mix(channels)
        notes.append("channel mix reset to the brief's channels")

    split = _normalize(strategy.budget_split, list(DEFAULT_SPLIT))
    if not split:
        split = dict(DEFAULT_SPLIT)
        notes.append("budget split reset to default")

    timing = dict(strategy.timing)
    budget = timing.get("budget_thb")
    budget = float(budget) if isinstance(budget, (int, float)) and math.isfinite(budget) else None
    if budget is None:
        budget = bounds.budget_thb
    fixed_budget = min(max(budget, bounds.budget_min), bounds.budget_max)
    if abs(fixed_budget - budget) > 0.5:
        notes.append(f"budget {budget:,.0f}->{fixed_budget:,.0f} THB (flex)")
    base_timing = _timing(spec, fixed_budget, timing.get("flash_days"))

    cap = strategy.cap_per_user if strategy.cap_per_user and strategy.cap_per_user >= 1 else None

    rules = [
        rule for rule in strategy.contingency_rules
        if any(pattern.match(rule) for pattern in RULE_PATTERNS)
    ]
    if len(rules) != len(strategy.contingency_rules):
        notes.append(f"{len(strategy.contingency_rules) - len(rules)} unparseable rule(s) dropped")

    rationale = strategy.rationale.strip()
    if notes:
        rationale += " [Adjusted: " + "; ".join(notes) + "]"
    return strategy.model_copy(update={
        "mechanic": mechanic,
        "discount_pct": discount,
        "target_segments": targets,
        "channel_mix": mix,
        "budget_split": split,
        "timing": base_timing,
        "cap_per_user": cap,
        "contingency_rules": rules,
        "rationale": rationale,
    })


# ---------------------------------------------------------------- diversity


def knobs(strategy: Strategy, bounds: Bounds) -> tuple:
    budget = float(strategy.timing.get("budget_thb") or 0)
    return (
        round(strategy.discount_pct / 5),
        strategy.mechanic,
        frozenset(strategy.target_segments),
        strategy.cap_per_user,
        strategy.timing.get("flash_days"),
        round(budget / bounds.budget_thb, 1) if bounds.budget_thb else 0,
    )


def knob_distance(a: Strategy, b: Strategy, bounds: Bounds) -> int:
    return sum(x != y for x, y in zip(knobs(a, bounds), knobs(b, bounds), strict=True))


def all_distinct(strategies: list[Strategy], bounds: Bounds, minimum: int = 2) -> bool:
    return all(knob_distance(a, b, bounds) >= minimum for a, b in combinations(strategies, 2))


# ---------------------------------------------------------------- entry point


def propose(
    spec: CampaignSpec,
    context: ContextPack,
    proposer: Proposer | None = None,
    n: int = GENERATED,
) -> list[Strategy]:
    """Return [baseline, user plan, up to n generated plans], deterministic for fixed inputs."""
    bounds = resolve_bounds(spec, context)
    fixed = [baseline(spec), user_plan(spec, bounds)]

    proposed: list[Strategy] = []
    if proposer is not None:
        try:
            proposed = list(proposer(spec, context, bounds) or [])
        except Exception:  # the LLM path must never break the run
            proposed = []

    kept: list[Strategy] = []
    seen_ids = {plan.id for plan in fixed}
    for candidate in proposed + template_strategies(spec, context, bounds):
        if len(kept) >= n:
            break
        plan = enforce(candidate, spec, context, bounds)
        if plan.is_baseline or plan.is_user_plan or violates_must_avoid(plan, spec):
            continue
        if any(knob_distance(plan, other, bounds) < 2 for other in fixed[1:] + kept):
            continue
        plan_id = plan.id if plan.id.startswith("g0-") else f"g0-{plan.id}"
        while plan_id in seen_ids:
            plan_id += "-b"
        seen_ids.add(plan_id)
        note = f" [bounds: {bounds.label_note()}]"
        kept.append(plan.model_copy(update={"id": plan_id, "rationale": plan.rationale + note}))
    return fixed + kept

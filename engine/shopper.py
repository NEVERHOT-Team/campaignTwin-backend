"""Shopper agents: a vectorized, seeded weekly purchase model built from personas.

Each simulated shopper is a weighted copy of one persona (data/processed/personas.json)
with small jitter. Every week a shopper:

1. picks the brand they would buy (us / top rival / other) with a logit on price,
   loyalty to their usual brand, and the SSR appeal of our message (exposed shoppers only);
2. decides whether to buy: weekly_buy_rate x need (weeks since the last purchase relative
   to the purchase cycle) x waiting (shoppers who have seen a promotion hold back in weeks
   without one) x the chosen brand's price effect (price ** elasticity), so a rival's
   discount only adds rival purchases;
3. buys more units when the chosen brand is cheap, which delays the next purchase
   (stockpiling), so campaigns pull demand forward and a post-campaign dip appears;
4. may adopt the brand it bought as its new usual brand (habit x (1 - loyalty)).

No LLM calls happen here. Plan and baseline runs that share a seed draw the same random
numbers every week (common random numbers), so differences come only from the offer.
Holdout shoppers never see our offer; they still see rival prices.
All behavioural parameters in ShopperParams are labelled `default` until calibrated.
"""

import math
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from contracts.models import RivalScenario, Strategy, WeekRow
from data.loaders import load_brands, load_market, load_personas, load_unit_econ

US, RIVAL, OTHER = 0, 1, 2
DEFAULT_SHARES = (0.2, 0.3, 0.5)
DEFAULT_COST_PER_REACH = 0.32


@dataclass(frozen=True)
class ShopperParams:
    beta_price: float = 4.0  # utility per unit of relative price, times price sensitivity
    beta_loyalty: float = 2.0  # bonus for the shopper's usual brand, times brand_loyalty
    beta_appeal: float = 0.8  # utility per unit of SSR appeal, exposed shoppers only
    asc_rival: float = 0.0
    asc_other: float = 0.0
    rival_non_price_boost: float = 0.3
    need_floor: float = 0.3
    need_cap: float = 2.0
    price_mult_cap: float = 3.0
    max_weekly_prob: float = 0.9
    stockpile: float = 0.5  # extra units per unit of price-driven demand lift
    wait_per_exposure: float = 0.15  # x promo_share, once a shopper has seen a promotion
    wait_floor: float = 0.4
    jitter: float = 0.1
    holdout_share: float = 0.10
    # A shopper who buys a brand other than their usual one adopts it with probability
    # habit x (1 - brand_loyalty), so rival undercuts keep costing us after they end.
    habit: float = 0.2
    # Share of targeted shoppers who see the offer with no paid media (platform listing).
    organic_reach: float = 0.3
    # Real households per simulated shopper. WeekStats (and so revenue, profit and
    # buyers) are multiplied by it, so totals are comparable with a real campaign budget.
    households_per_agent: float = 50.0
    label: str = "default"


@dataclass(frozen=True)
class Population:
    persona_ids: tuple[str, ...]  # persona level
    persona_segments: tuple[str, ...]
    persona_channels: tuple[frozenset[str], ...]
    persona_idx: np.ndarray  # shopper level from here on
    weekly_buy_rate: np.ndarray
    cycle_weeks: np.ndarray
    avg_units: np.ndarray
    elasticity: np.ndarray
    sensitivity: np.ndarray
    promo_share: np.ndarray
    loyalty: np.ndarray
    preferred: np.ndarray
    holdout: np.ndarray
    ref_price: float
    shares: tuple[float, float, float]
    # Fixed per-shopper draw; the shopper is reached when reach_draw < offer.reach.
    reach_draw: np.ndarray | None = None
    # Real households each simulated shopper stands for. WeekStats are scaled by it, so
    # revenue and profit are market-sized and comparable with a real campaign budget.
    households_per_agent: float = 1.0
    scale_label: str = "default"

    @property
    def size(self) -> int:
        return int(self.persona_idx.size)

    @property
    def households(self) -> float:
        return self.size * self.households_per_agent


@dataclass
class ShopperState:
    weeks_since: np.ndarray
    exposures: np.ndarray
    redemptions: np.ndarray
    last_brand: np.ndarray
    ever_us: np.ndarray
    preferred: np.ndarray | None = None


@dataclass(frozen=True)
class WeekOffer:
    """What the market looks like in one week. Discounts are percentage points."""

    active: bool = False
    our_discount_pct: float = 0.0
    target_segments: tuple[str, ...] = ()  # empty = everyone
    channels: tuple[str, ...] = ()  # empty = every channel
    cap_per_user: int | None = None
    message_id: str | None = None
    rival_discount_pct: float = 0.0
    rival_non_price: bool = False
    reach: float = 1.0  # share of targeted, non-holdout shoppers who see the offer


@dataclass(frozen=True)
class WeekStats:
    buyers_us: int
    units_us: float
    revenue_us: float
    discounted_units_us: float
    buyers_rival: int
    buyers_other: int
    from_rival: int
    new_buyers_us: int
    waiting: int
    exposed: int
    holdout_buyers_us: int
    holdout_size: int
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------- population


def market_shares() -> tuple[float, float, float]:
    """(us, top rival, everyone else) from brands.csv, or a labelled default."""
    try:
        brands = load_brands()
        ours = float(brands.loc[brands["is_ours"], "market_share"].iloc[0])
        rival = float(brands.loc[~brands["is_ours"], "market_share"].max())
    except Exception:
        return DEFAULT_SHARES
    other = max(0.0, 1.0 - ours - rival)
    total = ours + rival + other
    return (ours / total, rival / total, other / total)


def build_population(
    n_agents: int = 5000,
    seed: int = 0,
    params: ShopperParams | None = None,
    personas: list[dict[str, Any]] | None = None,
    ref_price: float | None = None,
    shares: tuple[float, float, float] | None = None,
) -> Population:
    """Sample n_agents shoppers. Each stands for params.households_per_agent real households
    (an engine multiplier, label default); personas.json is never rescaled."""
    params = params or ShopperParams()
    personas = personas if personas is not None else load_personas()
    if not personas or n_agents <= 0:
        raise ValueError("A population needs personas and n_agents > 0")
    if ref_price is None:
        ref_price = float(load_market()["ref_price"])
    households_per_agent = max(float(params.households_per_agent), 1e-9)
    scale_label = params.label
    shares = shares or market_shares()
    rng = np.random.default_rng([seed, 1])

    weights = np.array([max(float(p.get("weight", 0)), 0.0) for p in personas])
    if weights.sum() > 0:
        weights = weights / weights.sum()
    else:
        weights = np.full(len(personas), 1 / len(personas))
    idx = rng.choice(len(personas), size=n_agents, p=weights)

    def column(key: str, default: float) -> np.ndarray:
        values = np.array([float(p.get(key, default)) for p in personas])
        return values[idx]

    def jitter() -> np.ndarray:
        return np.clip(rng.normal(1.0, params.jitter, n_agents), 0.5, 1.5)

    elasticity = column("price_elasticity", -1.5) * jitter()
    sensitivity = np.abs(elasticity) / max(np.abs(elasticity).mean(), 1e-9)
    return Population(
        persona_ids=tuple(str(p["persona_id"]) for p in personas),
        persona_segments=tuple(str(p.get("segment") or p.get("archetype")) for p in personas),
        persona_channels=tuple(
            frozenset(str(c).casefold() for c in p.get("channels", [])) for p in personas
        ),
        persona_idx=idx,
        weekly_buy_rate=np.clip(column("weekly_buy_rate", 0.05) * jitter(), 0.0, 1.0),
        cycle_weeks=np.maximum(column("purchase_cycle_weeks", 8.0) * jitter(), 1.0),
        avg_units=np.maximum(column("avg_units", 1.0), 0.1),
        elasticity=elasticity,
        sensitivity=sensitivity,
        promo_share=np.clip(column("promo_share", 0.5), 0.0, 1.0),
        loyalty=np.clip(column("brand_loyalty", 0.5), 0.0, 1.0),
        preferred=rng.choice(3, size=n_agents, p=np.asarray(shares) / sum(shares)),
        holdout=rng.random(n_agents) < params.holdout_share,
        ref_price=float(ref_price),
        shares=tuple(float(s) for s in shares),
        reach_draw=rng.random(n_agents),  # drawn last so earlier draws stay unchanged
        households_per_agent=households_per_agent,
        scale_label=scale_label,
    )


def initial_state(population: Population, seed: int = 0) -> ShopperState:
    rng = np.random.default_rng([seed, 2])
    n = population.size
    return ShopperState(
        weeks_since=rng.uniform(0, population.cycle_weeks),
        exposures=np.zeros(n, dtype=int),
        redemptions=np.zeros(n, dtype=int),
        last_brand=population.preferred.copy(),
        ever_us=np.zeros(n, dtype=bool),
        preferred=population.preferred.copy(),
    )


def appeal_vector(
    population: Population, table: dict[tuple[str, str], float] | None, message_id: str | None
) -> np.ndarray:
    """Per-shopper SSR appeal for one message (0 where unknown)."""
    if not table or message_id is None:
        return np.zeros(population.size)
    per_persona = np.array(
        [float(table.get((pid, message_id), 0.0)) for pid in population.persona_ids]
    )
    return np.clip(per_persona[population.persona_idx], -1.0, 1.0)


# ---------------------------------------------------------------- weekly step


def _exposure(population: Population, offer: WeekOffer) -> np.ndarray:
    if not offer.active or offer.our_discount_pct < 0:
        return np.zeros(population.size, dtype=bool)
    targets = {s.casefold() for s in offer.target_segments}
    channels = {c.casefold() for c in offer.channels}
    persona_ok = np.array([
        (not targets or segment.casefold() in targets)
        and (not channels or bool(persona_channels & channels))
        for segment, persona_channels in zip(
            population.persona_segments, population.persona_channels, strict=True
        )
    ])
    exposed = persona_ok[population.persona_idx] & ~population.holdout
    if offer.reach < 1.0 and population.reach_draw is not None:
        exposed &= population.reach_draw < offer.reach
    return exposed


def step(
    population: Population,
    state: ShopperState,
    offer: WeekOffer,
    seed: int,
    week: int,
    params: ShopperParams | None = None,
    appeal: np.ndarray | None = None,
) -> tuple[WeekStats, ShopperState]:
    """Advance one week. Same (seed, week) -> same random draws for every offer."""
    params = params or ShopperParams()
    n = population.size
    rng = np.random.default_rng([seed, 100 + week])
    u_buy = rng.random(n)
    u_brand = rng.random(n)
    u_habit = rng.random(n)  # drawn every week so plan and baseline stay paired
    appeal = np.zeros(n) if appeal is None else np.asarray(appeal, dtype=float)
    preferred = population.preferred if state.preferred is None else state.preferred

    exposed = _exposure(population, offer)
    eligible = exposed.copy()
    if offer.cap_per_user is not None:
        eligible &= state.redemptions < offer.cap_per_user
    d_us = np.where(eligible, np.clip(offer.our_discount_pct, 0, 100), 0.0)
    d_rival = float(np.clip(offer.rival_discount_pct, 0, 100))
    p_us = 1 - d_us / 100
    p_rival = 1 - d_rival / 100

    need = np.clip(state.weeks_since / population.cycle_weeks, params.need_floor, params.need_cap)
    promo_now = (d_us > 0) | (d_rival > 0)
    # Having seen a promotion teaches a shopper to hold back in weeks without one. One
    # campaign is one lesson: the effect does not grow with each week of the same campaign.
    waiting_factor = np.where(
        ~promo_now & (state.exposures > 0),
        np.clip(1 - params.wait_per_exposure * population.promo_share, params.wait_floor, 1.0),
        1.0,
    )
    p_base = np.clip(
        population.weekly_buy_rate * need * waiting_factor, 0, params.max_weekly_prob
    )

    # The brand a shopper would pick this week is decided first; the purchase decision
    # then uses that brand's price. So a rival's discount only adds rival purchases and
    # never makes anyone buy (or stock up on) more of our product.
    loyal = params.beta_loyalty * population.loyalty
    price = params.beta_price * population.sensitivity
    u_us = (
        -price * p_us
        + loyal * (preferred == US)
        + params.beta_appeal * appeal * exposed
    )
    u_rival = (
        params.asc_rival
        - price * p_rival
        + loyal * (preferred == RIVAL)
        + params.rival_non_price_boost * offer.rival_non_price
    )
    u_other = params.asc_other - price * 1.0 + loyal * (preferred == OTHER)
    utility = np.stack([u_us, u_rival, u_other], axis=1)
    utility -= utility.max(axis=1, keepdims=True)
    probs = np.exp(utility)
    probs /= probs.sum(axis=1, keepdims=True)
    brand = (u_brand[:, None] > np.cumsum(probs, axis=1)).sum(axis=1).clip(0, 2)

    chosen_price = np.where(brand == US, p_us, np.where(brand == RIVAL, p_rival, 1.0))
    chosen_mult = np.clip(np.power(np.maximum(chosen_price, 1e-6), population.elasticity), 1.0,
                          params.price_mult_cap)
    buy = u_buy < np.clip(p_base * chosen_mult, 0, params.max_weekly_prob)
    units = np.where(buy, population.avg_units * (1 + params.stockpile * (chosen_mult - 1)), 0.0)
    ours = buy & (brand == US)
    discounted = ours & (d_us > 0)

    scale = population.households_per_agent

    def count(mask: np.ndarray) -> int:
        return int(round(float(mask.sum()) * scale))

    stats = WeekStats(
        buyers_us=count(ours),
        units_us=float(units[ours].sum()) * scale,
        revenue_us=float((units * population.ref_price * p_us)[ours].sum()) * scale,
        discounted_units_us=float(units[discounted].sum()) * scale,
        buyers_rival=count(buy & (brand == RIVAL)),
        buyers_other=count(buy & (brand == OTHER)),
        from_rival=count(ours & (state.last_brand == RIVAL)),
        new_buyers_us=count(ours & ~state.ever_us & (population.preferred != US)),
        waiting=count(~buy & (waiting_factor < 1)),
        exposed=count(exposed),
        holdout_buyers_us=count(ours & population.holdout),
        holdout_size=count(population.holdout),
        extra={"households_per_agent": scale} if scale != 1.0 else {},
    )

    switch = (
        buy
        & (brand != preferred)
        & (u_habit < params.habit * (1 - population.loyalty))
    )
    extra_supply = (units / population.avg_units - 1) * population.cycle_weeks
    new_state = ShopperState(
        weeks_since=np.where(buy, -np.maximum(extra_supply, 0.0), state.weeks_since) + 1.0,
        # Only real discounts teach shoppers to wait for the next promotion.
        exposures=state.exposures + (exposed & (d_us > 0)).astype(int),
        redemptions=state.redemptions + discounted.astype(int),
        last_brand=np.where(buy, brand, state.last_brand),
        ever_us=state.ever_us | ours,
        preferred=np.where(switch, brand, preferred),
    )
    return stats, new_state


# ---------------------------------------------------------------- strategy glue


def _active_weeks(strategy: Strategy, campaign_weeks: int) -> int:
    flash_days = strategy.timing.get("flash_days") if strategy.timing else None
    if isinstance(flash_days, (int, float)) and flash_days > 0:
        return min(campaign_weeks, max(1, math.ceil(flash_days / 7)))
    return campaign_weeks


def offer_from_strategy(
    strategy: Strategy,
    week: int,
    campaign_weeks: int,
    rivals: list[RivalScenario] | None = None,
    reach: float = 1.0,
) -> WeekOffer:
    """One week of a plan under one scenario's rival policies (week 1 = campaign start).

    Rival depth is in percentage points. When several rivals are active, the deepest
    discount sets the rival price, as agreed with data/rival.py. `reach` comes from
    media_reach() and limits how many targeted shoppers see the offer.
    """
    active = not strategy.is_baseline and 1 <= week <= _active_weeks(strategy, campaign_weeks)
    rival_depth = 0.0
    non_price = False
    for row in rivals or []:
        start = 1 + int(row.delay_weeks)
        end = start + max(int(row.duration_weeks), 1) - 1
        if row.reaction == "none" or not start <= week <= end:
            continue
        if row.reaction in {"match", "undercut"}:
            rival_depth = max(rival_depth, float(row.depth))
        elif row.reaction == "non_price":
            non_price = True
    return WeekOffer(
        active=active,
        our_discount_pct=0.0 if strategy.is_baseline else float(strategy.discount_pct),
        target_segments=tuple(strategy.target_segments),
        channels=tuple(strategy.channel_mix),
        cap_per_user=strategy.cap_per_user,
        message_id=None if strategy.is_baseline else strategy.id,
        rival_discount_pct=rival_depth,
        rival_non_price=non_price,
        reach=float(np.clip(reach, 0.0, 1.0)),
    )


def _cost_per_reach() -> float:
    try:
        return float(load_unit_econ()["media_cost_per_reach_thb"])
    except Exception:
        return DEFAULT_COST_PER_REACH


def media_reach(
    population: Population,
    strategy: Strategy,
    params: ShopperParams | None = None,
    cost_per_reach: float | None = None,
) -> float:
    """Share of targeted shoppers who see the offer, bought with the plan's media budget.

    reach = 1 - (1 - organic) * exp(-impressions / targeted households), with
    impressions = media budget / cost per reach (unit_economics.json). Plans without a
    budget or a media share keep full reach, as before media was modelled.
    """
    params = params or ShopperParams()
    if strategy.is_baseline:
        return 0.0
    budget = (strategy.timing or {}).get("budget_thb")
    share = (strategy.budget_split or {}).get("media")
    if not isinstance(budget, (int, float)) or budget <= 0 or share is None:
        return 1.0
    probe = offer_from_strategy(strategy, 1, 1)
    targeted = float(_exposure(population, replace(probe, active=True)).sum())
    targeted *= population.households_per_agent
    if targeted <= 0:
        return 0.0
    cpr = cost_per_reach if cost_per_reach is not None else _cost_per_reach()
    impressions = float(budget) * float(share) / max(cpr, 1e-9)
    return float(1 - (1 - params.organic_reach) * math.exp(-impressions / targeted))


def simulate(
    population: Population,
    strategy: Strategy,
    weeks: int,
    campaign_weeks: int,
    seed: int,
    rivals: list[RivalScenario] | None = None,
    appeal: dict[tuple[str, str], float] | None = None,
    params: ShopperParams | None = None,
    *,
    adaptive: bool = False,
    reference: list[WeekStats] | None = None,
    market: dict[str, Any] | None = None,
) -> list[WeekStats]:
    """Run one plan for `weeks` weeks (campaign weeks first, then the post window).

    adaptive=False keeps the scheduled rival windows of offer_from_strategy().
    adaptive=True closes the loop: each week our contingency rules (engine.rules) react to
    last week's observations, and rivals (engine.rival) react to our posted offer. Pass the
    paired no-campaign run as `reference` so `end_early` can see incremental profit.
    Adaptive runs record what happened in WeekStats.extra: events, our and rival posted
    discounts.
    """
    if adaptive:
        return _simulate_adaptive(population, strategy, weeks, campaign_weeks, seed, rivals,
                                  appeal, params, reference, market)
    state = initial_state(population, seed)
    message = None if strategy.is_baseline else strategy.id
    vector = appeal_vector(population, appeal, message)
    reach = media_reach(population, strategy, params)
    stats = []
    for week in range(1, weeks + 1):
        offer = offer_from_strategy(strategy, week, campaign_weeks, rivals, reach)
        week_stats, state = step(population, state, offer, seed, week, params, vector)
        stats.append(week_stats)
    return stats


def _share_pp(stats: WeekStats) -> float | None:
    total = stats.buyers_us + stats.buyers_rival + stats.buyers_other
    return 100.0 * stats.buyers_us / total if total else None


def _simulate_adaptive(
    population: Population,
    strategy: Strategy,
    weeks: int,
    campaign_weeks: int,
    seed: int,
    rivals: list[RivalScenario] | None,
    appeal: dict[tuple[str, str], float] | None,
    params: ShopperParams | None,
    reference: list[WeekStats] | None,
    market: dict[str, Any] | None,
) -> list[WeekStats]:
    from engine.rival import RivalBook
    from engine.rules import Observation, PlanState, apply_rules, parse_rules

    state = initial_state(population, seed)
    message = None if strategy.is_baseline else strategy.id
    vector = appeal_vector(population, appeal, message)
    reach = media_reach(population, strategy, params)
    rules = [] if strategy.is_baseline else parse_rules(strategy.contingency_rules)
    plan = PlanState(tuple(strategy.target_segments), strategy.cap_per_user, reach)
    book = RivalBook(None if strategy.is_baseline else rivals)
    needs_profit = reference is not None and any(r.kind == "end_early" for r in rules)
    market = market or (load_market() if needs_profit else None)
    active_weeks = _active_weeks(strategy, campaign_weeks)
    media = 0.0
    if not strategy.is_baseline:
        budget = (strategy.timing or {}).get("budget_thb")
        share = (strategy.budget_split or {}).get("media", 0.0)
        if isinstance(budget, (int, float)) and budget > 0:
            media = float(budget) * float(share or 0.0)

    obs = Observation(week=0)
    first_share: float | None = None
    share_gain: float | None = None
    incr = 0.0
    stats: list[WeekStats] = []
    for week in range(1, weeks + 1):
        events = apply_rules(rules, plan, obs, week) if rules else []
        if any(e.action == "shift_budget" for e in events):
            shifted = strategy.model_copy(update={"target_segments": list(plan.target_segments)})
            plan.reach = media_reach(population, shifted, params)
        scheduled = offer_from_strategy(strategy, week, campaign_weeks, None, plan.reach)
        on = scheduled.active and not plan.stopped
        posted = scheduled.our_discount_pct if on else 0.0
        rival_pct, non_price, rival_events = book.react(week, posted, share_gain)
        offer = replace(
            scheduled,
            active=on,
            target_segments=plan.target_segments,
            cap_per_user=plan.cap_per_user,
            rival_discount_pct=rival_pct,
            rival_non_price=non_price,
        )
        week_stats, state = step(population, state, offer, seed, week, params, vector)
        share = _share_pp(week_stats)
        if week == 1:
            first_share = share
        base_share = (_share_pp(reference[week - 1])
                      if reference is not None and week <= len(reference) else None)
        share_gain = share - base_share if share is not None and base_share is not None else None
        if needs_profit and market is not None and week <= len(reference):
            incr += week_row(week, week_stats, market).profit
            incr -= week_row(week, reference[week - 1], market).profit
        spent = media * min(week, active_weeks) / active_weeks if active_weeks else 0.0
        obs = Observation(
            week=week,
            our_discount_pct=posted,
            rival_discount_pct=rival_pct,
            our_share_pp=share,
            first_share_pp=first_share,
            incr_profit_to_date=(incr - spent) if needs_profit else None,
        )
        log = [e.as_dict() for e in events + rival_events]
        stats.append(replace(week_stats, extra={
            **week_stats.extra,
            "events": log,
            "our_discount_pct": posted,
            "rival_discount_pct": rival_pct,
            "rival_non_price": non_price,
        }))
    return stats


def week_row(week: int, stats: WeekStats, market: dict[str, Any] | None = None) -> WeekRow:
    """Contract row. Profit is gross: revenue - unit cost - platform/payment fees - shipping."""
    market = market or load_market()
    fees = (float(market["platform_fee_pct"]) + float(market["payment_fee_pct"])) / 100
    profit = (
        stats.revenue_us * (1 - fees)
        - stats.units_us * float(market["unit_cost"])
        - stats.buyers_us * float(market["shipping_cost_per_order"])
    )
    return WeekRow(
        week=week,
        sales=round(stats.revenue_us, 2),
        profit=round(profit, 2),
        buyers=stats.buyers_us,
        from_rival=stats.from_rival,
        waiting=stats.waiting,
    )

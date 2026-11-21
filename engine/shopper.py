"""Shopper agents: a vectorized, seeded weekly purchase model built from personas.

Each simulated shopper is a weighted copy of one persona (data/processed/personas.json)
with small jitter. Every week a shopper:

1. decides whether to buy the category: weekly_buy_rate x need (weeks since the last
   purchase relative to the purchase cycle) x price effect (best price ** elasticity)
   x waiting (shoppers who have seen promotions hold back in weeks without one);
2. if buying, picks a brand (us / top rival / other) with a logit on price,
   loyalty to their usual brand, and the SSR appeal of our message (exposed shoppers only);
3. buys more units when the price is low, which delays the next purchase (stockpiling),
   so campaigns pull demand forward and a post-campaign dip appears.

No LLM calls happen here. Plan and baseline runs that share a seed draw the same random
numbers every week (common random numbers), so differences come only from the offer.
Holdout shoppers never see our offer; they still see rival prices.
All behavioural parameters in ShopperParams are labelled `default` until calibrated.
"""

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from contracts.models import RivalScenario, Strategy, WeekRow
from data.loaders import load_brands, load_market, load_personas

US, RIVAL, OTHER = 0, 1, 2
DEFAULT_SHARES = (0.2, 0.3, 0.5)


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
    wait_per_exposure: float = 0.15
    wait_floor: float = 0.4
    jitter: float = 0.1
    holdout_share: float = 0.10
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

    @property
    def size(self) -> int:
        return int(self.persona_idx.size)


@dataclass
class ShopperState:
    weeks_since: np.ndarray
    exposures: np.ndarray
    redemptions: np.ndarray
    last_brand: np.ndarray
    ever_us: np.ndarray


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
    params = params or ShopperParams()
    personas = personas if personas is not None else load_personas()
    if not personas or n_agents <= 0:
        raise ValueError("A population needs personas and n_agents > 0")
    if ref_price is None:
        ref_price = float(load_market()["ref_price"])
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
    return persona_ok[population.persona_idx] & ~population.holdout


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
    appeal = np.zeros(n) if appeal is None else np.asarray(appeal, dtype=float)

    exposed = _exposure(population, offer)
    eligible = exposed.copy()
    if offer.cap_per_user is not None:
        eligible &= state.redemptions < offer.cap_per_user
    d_us = np.where(eligible, np.clip(offer.our_discount_pct, 0, 100), 0.0)
    d_rival = float(np.clip(offer.rival_discount_pct, 0, 100))
    p_us = 1 - d_us / 100
    p_rival = 1 - d_rival / 100

    need = np.clip(state.weeks_since / population.cycle_weeks, params.need_floor, params.need_cap)
    best = np.minimum(np.minimum(p_us, p_rival), 1.0)
    price_mult = np.clip(np.power(np.maximum(best, 1e-6), population.elasticity), 1.0,
                         params.price_mult_cap)
    promo_now = (d_us > 0) | (d_rival > 0)
    waiting_factor = np.where(
        ~promo_now & (state.exposures > 0),
        np.clip(1 - params.wait_per_exposure * population.promo_share * state.exposures,
                params.wait_floor, 1.0),
        1.0,
    )
    p_buy = np.clip(
        population.weekly_buy_rate * need * price_mult * waiting_factor, 0, params.max_weekly_prob
    )
    buy = u_buy < p_buy

    loyal = params.beta_loyalty * population.loyalty
    price = params.beta_price * population.sensitivity
    u_us = (
        -price * p_us
        + loyal * (population.preferred == US)
        + params.beta_appeal * appeal * exposed
    )
    u_rival = (
        params.asc_rival
        - price * p_rival
        + loyal * (population.preferred == RIVAL)
        + params.rival_non_price_boost * offer.rival_non_price
    )
    u_other = params.asc_other - price * 1.0 + loyal * (population.preferred == OTHER)
    utility = np.stack([u_us, u_rival, u_other], axis=1)
    utility -= utility.max(axis=1, keepdims=True)
    probs = np.exp(utility)
    probs /= probs.sum(axis=1, keepdims=True)
    brand = (u_brand[:, None] > np.cumsum(probs, axis=1)).sum(axis=1).clip(0, 2)

    units = np.where(buy, population.avg_units * (1 + params.stockpile * (price_mult - 1)), 0.0)
    ours = buy & (brand == US)
    discounted = ours & (d_us > 0)

    stats = WeekStats(
        buyers_us=int(ours.sum()),
        units_us=float(units[ours].sum()),
        revenue_us=float((units * population.ref_price * p_us)[ours].sum()),
        discounted_units_us=float(units[discounted].sum()),
        buyers_rival=int((buy & (brand == RIVAL)).sum()),
        buyers_other=int((buy & (brand == OTHER)).sum()),
        from_rival=int((ours & (state.last_brand == RIVAL)).sum()),
        new_buyers_us=int((ours & ~state.ever_us & (population.preferred != US)).sum()),
        waiting=int((~buy & (waiting_factor < 1)).sum()),
        exposed=int(exposed.sum()),
        holdout_buyers_us=int((ours & population.holdout).sum()),
        holdout_size=int(population.holdout.sum()),
    )

    extra_supply = (units / population.avg_units - 1) * population.cycle_weeks
    new_state = ShopperState(
        weeks_since=np.where(buy, -np.maximum(extra_supply, 0.0), state.weeks_since) + 1.0,
        # Only real discounts teach shoppers to wait for the next promotion.
        exposures=state.exposures + (exposed & (d_us > 0)).astype(int),
        redemptions=state.redemptions + discounted.astype(int),
        last_brand=np.where(buy, brand, state.last_brand),
        ever_us=state.ever_us | ours,
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
) -> WeekOffer:
    """One week of a plan under one scenario's rival policies (week 1 = campaign start).

    Rival depth is in percentage points. When several rivals are active, the deepest
    discount sets the rival price, as agreed with data/rival.py.
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
    )


def simulate(
    population: Population,
    strategy: Strategy,
    weeks: int,
    campaign_weeks: int,
    seed: int,
    rivals: list[RivalScenario] | None = None,
    appeal: dict[tuple[str, str], float] | None = None,
    params: ShopperParams | None = None,
) -> list[WeekStats]:
    """Run one plan for `weeks` weeks (campaign weeks first, then the post window)."""
    state = initial_state(population, seed)
    message = None if strategy.is_baseline else strategy.id
    vector = appeal_vector(population, appeal, message)
    stats = []
    for week in range(1, weeks + 1):
        offer = offer_from_strategy(strategy, week, campaign_weeks, rivals)
        week_stats, state = step(population, state, offer, seed, week, params, vector)
        stats.append(week_stats)
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

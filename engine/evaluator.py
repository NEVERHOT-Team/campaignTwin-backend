"""Turn paired plan-vs-baseline shopper runs into SimResult KPIs. Deterministic, no LLM.

Every plan seed is compared with the no-campaign baseline run on the same seed, so the
difference comes only from the offer (common random numbers). Profit is gross weekly
profit from engine.shopper.week_row (revenue after discount, minus unit cost, platform and
payment fees, shipping) minus the plan's media spend. The discount itself is already
inside revenue, so it is not subtracted twice.

Approximations (documented, label `default`):
- leakage = baseline units bought by shoppers the plan exposes during the campaign,
  divided by the plan's discounted units. The shopper model reports aggregates per week,
  so "would have bought anyway" is estimated from the exposed share of baseline buyers.
- breakpoint = first week after which the median cumulative incremental profit stays
  below zero to the end of the measurement window.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from contracts.models import KPI, Breakpoint, Pct, RivalScenario, SimResult, Strategy, WeekRow
from engine.shopper import WeekStats, week_row


@dataclass(frozen=True)
class SeedOutcome:
    incr_profit: float
    incr_revenue: float
    campaign_cost: float
    iroas: float | None
    leakage: float | None
    new_buyers: float
    weekly_incr_profit: tuple[float, ...]
    plan_rows: tuple[WeekRow, ...]


def pct(values: list[float]) -> Pct:
    clean = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not clean:
        return Pct(p10=0.0, p50=0.0, p90=0.0)
    p10, p50, p90 = np.percentile(clean, [10, 50, 90])
    return Pct(p10=round(float(p10), 4), p50=round(float(p50), 4), p90=round(float(p90), 4))


def plan_budget(strategy: Strategy) -> float:
    value = (strategy.timing or {}).get("budget_thb")
    return float(value) if isinstance(value, (int, float)) and value > 0 else 0.0


def media_cost(strategy: Strategy) -> float:
    if strategy.is_baseline:
        return 0.0
    return plan_budget(strategy) * float((strategy.budget_split or {}).get("media", 0.0))


def seed_outcome(
    strategy: Strategy,
    plan: list[WeekStats],
    base: list[WeekStats],
    campaign_weeks: int,
    population_size: int,
    ref_price: float,
    market: dict[str, Any],
) -> SeedOutcome:
    if len(plan) != len(base):
        raise ValueError("Plan and baseline must cover the same weeks")
    plan_rows = [week_row(w + 1, s, market) for w, s in enumerate(plan)]
    base_rows = [week_row(w + 1, s, market) for w, s in enumerate(base)]
    media = media_cost(strategy)
    media_per_week = media / max(campaign_weeks, 1)
    weekly = []
    for week, (p_row, b_row) in enumerate(zip(plan_rows, base_rows, strict=True), start=1):
        spend = media_per_week if week <= campaign_weeks else 0.0
        weekly.append(p_row.profit - b_row.profit - spend)

    incr_revenue = sum(r.sales for r in plan_rows) - sum(r.sales for r in base_rows)
    discount_cost = sum(
        s.discounted_units_us * ref_price * float(strategy.discount_pct) / 100 for s in plan
    )
    cost = discount_cost + media
    iroas = incr_revenue / cost if cost > 0 else None

    discounted = sum(s.discounted_units_us for s in plan)
    leakage = None
    if discounted > 0:
        campaign = range(min(campaign_weeks, len(plan)))
        exposed_share = max((plan[w].exposed for w in campaign), default=0) / max(
            population_size, 1
        )
        anyway = sum(base[w].units_us for w in campaign) * exposed_share
        leakage = float(min(1.0, anyway / discounted))

    new_buyers = sum(s.new_buyers_us for s in plan) - sum(s.new_buyers_us for s in base)
    return SeedOutcome(
        incr_profit=float(sum(weekly)),
        incr_revenue=float(incr_revenue),
        campaign_cost=float(cost),
        iroas=iroas,
        leakage=leakage,
        new_buyers=float(new_buyers),
        weekly_incr_profit=tuple(weekly),
        plan_rows=tuple(plan_rows),
    )


def median_rows(outcomes: list[SeedOutcome]) -> list[WeekRow]:
    """Median plan WeekRow per week across seeds (what the weekly chart shows)."""
    weeks = len(outcomes[0].plan_rows)
    rows = []
    for index in range(weeks):
        column = [o.plan_rows[index] for o in outcomes]
        rows.append(WeekRow(
            week=index + 1,
            sales=round(float(np.median([r.sales for r in column])), 2),
            profit=round(float(np.median([r.profit for r in column])), 2),
            buyers=int(round(np.median([r.buyers for r in column]))),
            from_rival=int(round(np.median([r.from_rival for r in column]))),
            waiting=int(round(np.median([r.waiting for r in column]))),
        ))
    return rows


def _rival_attacking(rivals: list[RivalScenario], week: int) -> bool:
    for row in rivals:
        start = 1 + int(row.delay_weeks)
        end = start + max(int(row.duration_weeks), 1) - 1
        if row.reaction in {"match", "undercut"} and row.depth > 0 and start <= week <= end:
            return True
    return False


def breakpoint(
    outcomes: list[SeedOutcome], campaign_weeks: int, rivals: list[RivalScenario]
) -> Breakpoint:
    p_loss = float(np.mean([o.incr_profit < 0 for o in outcomes])) if outcomes else 0.0
    cumulative = np.median([np.cumsum(o.weekly_incr_profit) for o in outcomes], axis=0)
    if cumulative[-1] >= 0:
        return Breakpoint(week=None, cause=None, share=round(p_loss, 4))
    week = len(cumulative)
    while week > 1 and cumulative[week - 2] < 0:
        week -= 1
    if _rival_attacking(rivals, week):
        cause = "rival_reaction"
    elif week > campaign_weeks:
        cause = "post_campaign_dip"
    else:
        cause = "discount_cost"
    return Breakpoint(week=week, cause=cause, share=round(p_loss, 4))


def sim_result(
    strategy: Strategy,
    scenario: str,
    outcomes: list[SeedOutcome],
    campaign_weeks: int,
    rivals: list[RivalScenario],
    target_profit: float = 0.0,
) -> SimResult:
    if not outcomes:
        raise ValueError("A SimResult needs at least one seed")
    profits = [o.incr_profit for o in outcomes]
    return SimResult(
        strategy_id=strategy.id,
        scenario=scenario,
        weekly=median_rows(outcomes),
        kpi=KPI(
            incr_profit=pct(profits),
            iroas=pct([o.iroas for o in outcomes if o.iroas is not None]),
            leakage=pct([o.leakage for o in outcomes if o.leakage is not None]),
            new_buyers=pct([o.new_buyers for o in outcomes]),
            p_hit_target=round(float(np.mean([p >= target_profit for p in profits])), 4),
            p_loss=round(float(np.mean([p < 0 for p in profits])), 4),
        ),
        breakpoint=breakpoint(outcomes, campaign_weeks, rivals),
    )


def scenario_weights(rivals_by_scenario: dict[str, list[RivalScenario]]) -> dict[str, float]:
    """Normalized S1-S3 weights from the top rival's likelihoods. S4 is unweighted."""
    raw = {}
    for scenario in ("S1", "S2", "S3"):
        rows = rivals_by_scenario.get(scenario) or []
        if rows:
            top = max(rows, key=lambda r: r.rival_score)
            raw[scenario] = max(float(top.likelihood), 0.0)
    total = sum(raw.values())
    if total <= 0:
        return {s: 1 / len(raw) for s in raw} if raw else {}
    return {s: w / total for s, w in raw.items()}


def summarize(
    results: list[SimResult],
    weights_by_strategy: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    """Per strategy: likelihood-weighted EV of median profit, worst case, rank."""
    by_strategy: dict[str, dict[str, SimResult]] = {}
    for result in results:
        by_strategy.setdefault(result.strategy_id, {})[result.scenario] = result
    rows = []
    for strategy_id, scenarios in by_strategy.items():
        weights = weights_by_strategy.get(strategy_id, {})
        weighted = [
            weights[s] * scenarios[s].kpi.incr_profit.p50 for s in weights if s in scenarios
        ]
        ev = float(sum(weighted)) if weighted else scenarios[min(scenarios)].kpi.incr_profit.p50
        worst = min(scenarios.values(), key=lambda r: r.kpi.incr_profit.p50)
        rows.append({
            "strategy_id": strategy_id,
            "ev_incr_profit": round(ev, 2),
            "worst_scenario": worst.scenario,
            "worst_incr_profit": worst.kpi.incr_profit.p50,
            "p10_s1": scenarios["S1"].kpi.incr_profit.p10 if "S1" in scenarios else None,
            "max_p_loss": max(r.kpi.p_loss for r in scenarios.values()),
            "weights": weights,
        })
    rows.sort(key=lambda r: (-r["ev_incr_profit"], -(r["p10_s1"] or 0), r["strategy_id"]))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows

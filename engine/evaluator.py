"""Deterministic, paired campaign accounting. Money is THB and returns are ratios."""

import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import fields
from fractions import Fraction
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel

from contracts.campaign_spec import SmartGoal
from contracts.models import (
    KPI,
    Breakpoint,
    Pct,
    RivalScenario,
    Scenario,
    SimResult,
    Strategy,
    WeekRow,
)
from engine.shopper import WeekStats, offer_from_strategy, week_row

Runs = Mapping[int, Sequence[WeekStats]]
_STATS_FIELDS = tuple(f.name for f in fields(WeekStats) if f.name != "extra")


class EvaluatedKPI(KPI):
    """iROI is the primary return measure. iROAS measures revenue per baht spent."""

    iroi: Pct
    incr_revenue: Pct
    discount_cost: Pct
    media_cost: Pct
    campaign_cost: Pct


class EvaluatedPlan(SimResult):
    """Engine result. Direct model_dump() retains these fields for report builders."""

    kpi: EvaluatedKPI
    media_cost_label: Literal["data", "default"]
    is_baseline: bool
    is_user_plan: bool
    post_campaign_profit_p50: float  # median of paired incremental post-campaign totals


def _paired_runs(a: Runs, b: Runs) -> tuple[list[Sequence[WeekStats]], list[Sequence[WeekStats]]]:
    if not a or a.keys() != b.keys():
        raise ValueError("Paired runs require the same nonempty set of seeds")
    horizon = len(next(iter(a.values())))
    if not horizon:
        raise ValueError("At least one week is required")
    first, second = [], []
    for seed in sorted(a):
        for source, ordered in ((a, first), (b, second)):
            run = source[seed]
            if len(run) != horizon:
                raise ValueError(f"All runs must cover the same weeks (seed {seed})")
            for week, stats in enumerate(run, 1):
                values = [getattr(stats, name) for name in _STATS_FIELDS]
                if not np.isfinite(values).all() or min(values) < 0:
                    raise ValueError(f"Nonfinite or negative WeekStats at seed {seed}, week {week}")
                if stats.discounted_units_us > stats.units_us:
                    raise ValueError(
                        f"Discounted units exceed total units at seed {seed}, week {week}"
                    )
            ordered.append(run)
    return first, second


def _column(runs, name: str) -> np.ndarray:
    """Seed × week matrix from either shopper statistics or contract rows."""
    return np.array([[getattr(week, name) for week in run] for run in runs], dtype=float)


def _sales(run: Sequence[WeekStats]) -> Fraction:
    """Preserve the shopper's revenue precision across weeks and inclusive thresholds."""
    return sum((Fraction(str(week.revenue_us)) for week in run), start=Fraction(0))


def _pct(samples: np.ndarray) -> Pct:
    p10, p50, p90 = np.percentile(samples, (10, 50, 90))
    if not np.isfinite([p10, p50, p90]).all():
        raise ValueError("KPI must be finite")
    return Pct(p10=p10, p50=p50, p90=p90)


def _ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    # The contract and baseline replay use zero when there is no spend or redemption.
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)


def _hit_probability(samples: dict[str, np.ndarray], target: float | SmartGoal | None) -> float:
    if target is None:
        return 0.0
    metric, minimum = "incr_profit", target
    if isinstance(target, SmartGoal):
        aliases = {
            "incremental_profit": "incr_profit",
            "incremental_revenue": "incr_revenue",
            "incremental_roi": "iroi",
            "incremental_roas": "iroas",
            "incremental_new_buyers": "new_buyers",
            "subsidy_leakage": "leakage",
        }
        metric = aliases.get(target.metric.casefold(), target.metric.casefold())
        minimum = target.target
        unit = target.unit.casefold()
        if metric in {"leakage", "iroi", "iroas"}:
            if unit in {"%", "pct", "percent"}:
                minimum /= 100
            elif unit not in {"ratio", "fraction", "x"}:
                raise ValueError(f"{metric} targets require a ratio or percentage")
        elif metric == "new_buyers" and unit not in {"buyers", "people", "count"}:
            raise ValueError("new_buyers targets require a buyer count")
        elif metric in samples and metric != "new_buyers" and unit not in {"thb", "฿"}:
            raise ValueError(f"{metric} targets require THB")
    if metric not in samples or not math.isfinite(minimum):
        raise ValueError("Target must name a measured KPI and have a finite threshold")
    return float(np.mean(samples[metric] >= minimum))


def _median_rows(plan_rows: list[list[WeekRow]]) -> list[WeekRow]:
    columns = {
        name: np.median(_column(plan_rows, name), axis=0)
        for name in ("sales", "profit", "buyers", "from_rival", "waiting")
    }
    if not all(np.isfinite(values).all() for values in columns.values()):
        raise ValueError("Weekly medians must be finite")
    # WeekRow count fields are integers. Round a half-count median to the next whole count.
    return [
        WeekRow(
            week=i + 1,
            sales=columns["sales"][i],
            profit=columns["profit"][i],
            **{
                name: math.floor(columns[name][i] + 0.5)
                for name in ("buyers", "from_rival", "waiting")
            },
        )
        for i in range(len(plan_rows[0]))
    ]


def evaluate_plan(
    plan_runs: Runs,
    baseline_runs: Runs,
    strategy: Strategy,
    scenario: Scenario | RivalScenario,
    market: dict[str, Any],
    budget_thb: float,
    target: float | SmartGoal | None = None,
) -> EvaluatedPlan:
    """Evaluate one strategy/scenario against its matching no-campaign seed runs.

    Use the same market campaign-week count as shopper.simulate. Without a window,
    the supplied weeks are the campaign. Flash offers use the shopper's shorter window.
    The plan's weekly rows show gross profit. For the net cumulative curve, spread the
    one-time media allocation evenly across offer weeks. Missing media share costs ฿0.
    A numeric target is incremental profit in THB. A SmartGoal names a full-horizon KPI.

    Leakage is a proxy until per-shopper counterfactuals are available. Sum each offer
    week's min(discounted plan units, baseline units), then divide by all discounted
    units in those weeks. Percentiles summarize paired seed metrics, including ratios.
    """
    plan, base = _paired_runs(plan_runs, baseline_runs)
    horizon = len(plan[0])
    campaign_weeks = market.get("weeks", {}).get("campaign", horizon)
    if not isinstance(campaign_weeks, int) or campaign_weeks < 0:
        raise ValueError("Campaign weeks must be a nonnegative integer")
    active = np.array(
        [offer_from_strategy(strategy, w, campaign_weeks).active for w in range(1, horizon + 1)]
    )
    price = float(market["ref_price"])
    share = strategy.budget_split.get("media", 0.0)
    if not 0 <= share <= 1 or not 0 <= strategy.discount_pct <= 100:
        raise ValueError("Media share must be in [0, 1] and discount_pct in [0, 100]")
    cost_keys = ("unit_cost", "platform_fee_pct", "payment_fee_pct", "shipping_cost_per_order")
    economics = [price, budget_thb, *(float(market[key]) for key in cost_keys)]
    if not np.isfinite(economics).all() or min(economics) < 0:
        raise ValueError("Market costs, price and budget must be finite and nonnegative")
    media = (
        Fraction(0) if strategy.is_baseline else Fraction(str(share)) * Fraction(str(budget_thb))
    )
    if media and not active.any():
        raise ValueError("A nonzero media budget needs an active campaign week")

    plan_rows = [[week_row(w, stats, market) for w, stats in enumerate(run, 1)] for run in plan]
    base_profit = np.array(
        [[week_row(w, stats, market).profit for w, stats in enumerate(run, 1)] for run in base]
    )
    # week_row already rounds gross profit to satang. Accumulate whole satang so a
    # break-even plan cannot acquire a negative probability event from float subtraction.
    gross_delta_satang = np.rint(100 * _column(plan_rows, "profit")) - np.rint(100 * base_profit)
    net_curve = np.cumsum(gross_delta_satang, axis=1)
    if media:
        active_weeks = int(active.sum())
        net_curve = (
            net_curve * active_weeks - float(100 * media) * np.cumsum(active)
        ) / active_weeks
    net_curve /= 100
    profit = net_curve[:, -1]
    revenue = np.array([float(_sales(p) - _sales(b)) for p, b in zip(plan, base, strict=True)])
    discounted = _column(plan, "discounted_units_us")
    if np.any(discounted[:, ~active] > 0):
        raise ValueError("Discounted plan units outside offer weeks: check the simulation window")
    discounted = discounted[:, active]
    redeemed = discounted.sum(axis=1)
    discount_cost = redeemed * price * (strategy.discount_pct / 100)
    cost = discount_cost + float(media)
    overlap = np.minimum(discounted, _column(base, "units_us")[:, active]).sum(axis=1)
    samples = {
        "incr_profit": profit,
        "incr_revenue": revenue,
        "discount_cost": discount_cost,
        "media_cost": np.full(len(plan), float(media)),
        "campaign_cost": cost,
        "iroi": _ratio(profit, cost),
        "iroas": _ratio(revenue, cost),
        "leakage": _ratio(overlap, redeemed),
        "new_buyers": (_column(plan, "new_buyers_us") - _column(base, "new_buyers_us")).sum(axis=1),
    }
    median_curve = np.median(net_curve, axis=0)
    post_p50 = float(np.median(gross_delta_satang[:, ~active].sum(axis=1))) / 100
    if not np.isfinite(median_curve).all() or not math.isfinite(post_p50):
        raise ValueError("Incremental profit curve must be finite")
    downturns = np.flatnonzero(np.diff(median_curve, prepend=0) < 0)
    week = int(downturns[0] + 1) if median_curve[-1] < 0 and downturns.size else None
    return EvaluatedPlan(
        strategy_id=strategy.id,
        scenario=scenario.scenario if isinstance(scenario, RivalScenario) else scenario,
        weekly=_median_rows(plan_rows),
        kpi=EvaluatedKPI(
            **{name: _pct(values) for name, values in samples.items()},
            p_loss=float(np.mean(profit < 0)),
            p_hit_target=_hit_probability(samples, target),
        ),
        breakpoint=Breakpoint(week=week, cause=None, share=None),
        media_cost_label="data" if "media" in strategy.budget_split else "default",
        is_baseline=strategy.is_baseline,
        is_user_plan=strategy.is_user_plan,
        post_campaign_profit_p50=post_p50,
    )


def _group_results(results: Sequence[SimResult]) -> dict[str, dict[str, SimResult]]:
    plans: dict[str, dict[str, SimResult]] = {}
    for result in results:
        scenarios = plans.setdefault(result.strategy_id, {})
        if result.scenario in scenarios:
            raise ValueError(f"Duplicate scenario {result.scenario} for {result.strategy_id}")
        scenarios[result.scenario] = result
    return plans


def explain_breakpoints(results: Sequence[EvaluatedPlan]) -> list[EvaluatedPlan]:
    """Return results in input order with causes and clipped shares for detected breaks.

    Without a matching S1 the comparative share is unavailable. The paired post-window
    evidence and discount-cost fallback can still explain the plan's own downturn.
    """
    plans = _group_results(results)
    explained = []
    for result in results:
        if result.breakpoint.week is None:
            explained.append(result)
            continue
        calm = plans[result.strategy_id].get("S1")
        if result.scenario in ("S2", "S3") and calm is not None and calm.breakpoint.week is None:
            cause = "rival undercut"
        elif result.post_campaign_profit_p50 < 0:
            cause = "pull-forward / post-campaign dip"
        else:
            cause = "discount cost"
        profit = result.kpi.incr_profit.p50
        share = None
        if calm is not None and profit < 0:
            share = float(np.clip((calm.kpi.incr_profit.p50 - profit) / abs(profit), 0, 1))
        breakpoint = Breakpoint(week=result.breakpoint.week, cause=cause, share=share)
        explained.append(result.model_copy(update={"breakpoint": breakpoint}))
    return explained


def aa_lift(baseline_a: Runs, baseline_b: Runs) -> float:
    """Pooled sales lift of B versus A in percentage points. Warn if |lift| exceeds 2."""
    a, b = _paired_runs(baseline_a, baseline_b)
    sales_a = sum(map(_sales, a))
    sales_b = sum(map(_sales, b))
    if sales_a == 0:
        if sales_b:
            raise ValueError("A/A sales lift is undefined with zero reference sales")
        return 0.0
    lift = 100 * (sales_b - sales_a) / sales_a
    if abs(lift) > 2:
        warnings.warn(f"A/A sales lift {float(lift):+.2f}% exceeds 2% in magnitude", stacklevel=2)
    return float(lift)


class RankedPlan(BaseModel):
    strategy_id: str
    filtered: bool
    score: float
    reason: str
    worst_p10: float
    max_regret: float
    max_p_loss: float
    is_baseline: bool
    is_user_plan: bool


def rank_plans(
    results: Sequence[SimResult],
    likelihoods: Mapping[str, float] | None = None,
    max_loss_prob: float = 0.2,
) -> list[RankedPlan]:
    """Rank complete S1–S3 comparisons. S4 is an optional stress test.

    The loss limit applies to each scenario. Missing scenario weights are zero.
    Regret is the gap to the best P50 of any submitted plan in the same scenario.
    Filtered plans remain visible. The baseline is a comparator, listed last when
    there are campaigns to rank. A baseline-only comparison returns an empty list.
    """
    scenarios = ("S1", "S2", "S3")
    if not 0 <= max_loss_prob <= 1:
        raise ValueError("max_loss_prob must be a probability between 0 and 1")

    weights = None
    if likelihoods is not None:
        values = [likelihoods.get(s, 0.0) for s in scenarios]
        if not np.isfinite(values).all() or min(values) < 0 or max(values) == 0:
            raise ValueError("S1–S3 likelihoods must be finite, nonnegative and not all zero")
        # Keep equal weighted profits tied regardless of scenario order.
        shares = [Fraction(str(value)) for value in values]
        total = sum(shares)
        weights = [share / total for share in shares]

    grouped = _group_results(results)
    if not grouped:
        return []
    for plan_id, by_scenario in grouped.items():
        missing = [s for s in scenarios if s not in by_scenario]
        if missing:
            raise ValueError(f"{plan_id} is missing ranking scenarios {', '.join(missing)}")

    best_medians = {
        s: max(plan[s].kpi.incr_profit.p50 for plan in grouped.values()) for s in scenarios
    }
    ranking = []
    for plan_id, by_scenario in grouped.items():
        rows = [by_scenario[s] for s in scenarios]
        medians = [r.kpi.incr_profit.p50 for r in rows]
        lower = [r.kpi.incr_profit.p10 for r in rows]
        risks = [r.kpi.p_loss for r in rows]
        if not np.isfinite([medians, lower]).all() or not all(0 <= p <= 1 for p in risks):
            raise ValueError(f"{plan_id} needs finite profit values and valid loss probabilities")
        worst_p10 = min(lower)
        score = worst_p10
        if weights is not None:
            score = float(sum(w * Fraction(str(m)) for w, m in zip(weights, medians, strict=True)))
        max_regret = max(best_medians[r.scenario] - r.kpi.incr_profit.p50 for r in rows)
        if not np.isfinite([score, max_regret]).all():
            raise ValueError(f"Ranking score or regret for {plan_id} exceeds the numeric range")
        risk = max(risks)
        filtered = risk > max_loss_prob
        baseline = getattr(rows[0], "is_baseline", plan_id == "baseline")
        if baseline:
            reason = "No-campaign comparator."
        elif filtered:
            scenario = scenarios[risks.index(risk)]
            reason = f"{scenario} loss risk {risk:.2%} exceeds the {max_loss_prob:.2%} limit."
        elif weights is None:
            reason = f"Meets the loss limit. Worst-scenario P10 profit is ฿{score:,.2f}."
        else:
            reason = f"Meets the loss limit. Likelihood-weighted P50 profit is ฿{score:,.2f}."
        ranking.append(
            RankedPlan(
                strategy_id=plan_id,
                filtered=filtered,
                score=score,
                reason=reason,
                worst_p10=worst_p10,
                max_regret=max_regret,
                max_p_loss=risk,
                is_baseline=baseline,
                is_user_plan=getattr(rows[0], "is_user_plan", plan_id == "user-plan"),
            )
        )
    if all(plan.is_baseline for plan in ranking):
        return []
    return sorted(
        ranking,
        key=lambda plan: (
            plan.is_baseline,
            plan.filtered,
            -plan.score,
            -plan.worst_p10,
            plan.max_regret,
            plan.strategy_id,
        ),
    )

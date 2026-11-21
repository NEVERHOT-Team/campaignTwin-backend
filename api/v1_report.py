"""Additional v1 report fields for the plans actually evaluated by the arena."""

from typing import Any

from agents.strategist import unit_contribution_thb
from contracts.models import SimulationPlan, Strategy, WeekRow
from data.loaders import load_market
from engine import evaluator, shopper
from engine.arena import ArenaResult


def _baseline_weekly(
    plan: SimulationPlan, outcome: ArenaResult, strategy: Strategy, base_seed: int,
) -> list[dict]:
    for result in outcome.results:
        if result.strategy_id == strategy.id:
            return [row.model_dump() for row in result.weekly]
    captured = getattr(outcome, "baseline_weekly", None)
    if captured is not None:
        return [WeekRow.model_validate(row).model_dump() for row in captured]

    # The current ArenaResult discards its paired control. Replay only that control
    # with the same population, seeds and horizon, once for all scenarios.
    population = shopper.build_population(plan.n_agents, seed=base_seed)
    market = load_market()
    controls = []
    for seed in outcome.seeds:
        weeks = shopper.simulate(population, strategy, outcome.weeks, outcome.campaign_weeks, seed)
        controls.append(evaluator.seed_outcome(
            strategy, weeks, weeks, outcome.campaign_weeks, population.size,
            population.ref_price, market,
        ))
    return [row.model_dump() for row in evaluator.median_rows(controls)]


def _scenario_kpis(result):
    kpi = result.kpi
    return {
        "incremental_profit": kpi.incr_profit.model_dump(),
        "iroi": kpi.iroi.model_dump(),
        "iroas": kpi.iroas.model_dump(),
        "leakage": kpi.leakage.model_dump() if kpi.leakage is not None else None,
        "leakage_label": kpi.leakage_label,
        "new_buyers": kpi.new_buyers.model_dump(),
        "p_loss": kpi.p_loss, "p_hit_target": kpi.p_hit_target,
    }


def _ranking(outcome: ArenaResult, baseline_id: str):
    ranking = []
    for item in outcome.summary:
        plan_id = item["strategy_id"]
        if plan_id == baseline_id:
            continue
        evaluated = {r.scenario for r in outcome.results if r.strategy_id == plan_id}
        if evaluated.intersection(item["weights"]):
            basis = "Likelihood-weighted P50 incremental profit"
        else:
            # evaluator.summarize falls back to the first scenario in lexical order.
            basis = f"{min(evaluated)} P50 incremental profit (no scenario weights)"
        profit = item["ev_incr_profit"]
        ranking.append({
            "plan_id": plan_id, "rank": item["rank"], "weighted_profit": profit,
            "rank_reason": f"{basis} of ฿{profit:,.2f}. Ties use S1 P10 profit, then plan ID.",
        })
    ranking.append({
        "plan_id": baseline_id, "rank": None, "weighted_profit": 0.0,
        "rank_reason": "Paired no-campaign comparator, excluded from campaign ranking.",
    })
    return ranking


def _risks(user_scenarios, strategy: Strategy):
    risks = []
    if strategy.below_cost:
        loss = -unit_contribution_thb(strategy.discount_pct)
        risks.append(f"User plan sells below cost by ฿{loss:,.2f} per unit")
    if user_scenarios and all(kpi["incremental_profit"]["p50"] < 0
                              for kpi in user_scenarios.values()):
        risks.append("The user plan loses money at P50 in every evaluated scenario.")
    if "S1" in user_scenarios and "S3" in user_scenarios:
        diff = (user_scenarios["S1"]["incremental_profit"]["p50"]
                - user_scenarios["S3"]["incremental_profit"]["p50"])
        if diff > 0:
            risks.append(f"User-plan P50 incremental profit is ฿{diff:,.2f} lower in S3 than S1.")
    for scenario, kpi in user_scenarios.items():
        leakage = kpi["leakage"]
        if leakage is not None and leakage["p50"] > 0:
            risks.append(f"User-plan P50 subsidy leakage in {scenario} is {leakage['p50']:.2%}.")
    return risks


def arena_report_fields(
    plan: SimulationPlan, outcome: ArenaResult, *, base_seed: int = 0, profit_target: float = 0.0,
) -> dict[str, Any]:
    """Project actual evaluated rows, including only the S4 policies the arena ran."""
    by_plan: dict[str, dict] = {}
    for row in outcome.results:
        by_plan.setdefault(row.strategy_id, {})[row.scenario] = row
    scenarios = list(dict.fromkeys(row.scenario for row in outcome.results))
    baseline = next(strategy for strategy in plan.strategies if strategy.is_baseline)
    baseline_weeks = _baseline_weekly(plan, outcome, baseline, base_seed)
    zero = {"p10": 0.0, "p50": 0.0, "p90": 0.0}
    baseline_kpis = {
        "incremental_profit": zero, "iroi": None, "iroas": None, "leakage": None,
        "leakage_label": "unavailable", "new_buyers": zero, "p_loss": 0.0,
        "p_hit_target": float(profit_target <= 0),
    }
    plans, weekly, breakpoints, risks = [], {}, {}, []
    for strategy in plan.strategies:
        sid = strategy.id
        if strategy.is_baseline:
            role = "baseline"
            kpis = dict.fromkeys(scenarios, baseline_kpis)
            weekly[sid] = dict.fromkeys(scenarios, baseline_weeks)
            # The comparator's incremental curve is exactly zero in every seed.
            breakpoints[sid] = dict.fromkeys(scenarios, {
                "week": None, "cause": "never_profitable", "peak_week": None,
            })
        else:
            role = "user" if strategy.is_user_plan else "ai"
            rows = by_plan.get(sid, {})
            kpis = {s: _scenario_kpis(row) for s, row in rows.items()}
            weekly[sid] = {s: [week.model_dump() for week in row.weekly] for s, row in rows.items()}
            breakpoints[sid] = {
                s: {"week": row.breakpoint.week, "cause": row.breakpoint.cause,
                    "peak_week": row.peak_week}
                for s, row in rows.items()
            }
            if strategy.is_user_plan:
                risks = _risks(kpis, strategy)
        plans.append({
            "plan_id": sid, "name": strategy.name, "role": role,
            "below_cost": strategy.below_cost,
            "discount_pct": strategy.discount_pct, "mechanic": strategy.mechanic, "scenarios": kpis,
        })

    # The arena emits one trace per plan/scenario from the middle seed. A notice
    # precedes an actual response, and one event may name several rivals acting together.
    events = {s: {"rival_reactions": 0, "own_rule_firings": 0, "rival_notices": 0}
              for s in scenarios}
    for trace in outcome.events.values():
        for scenario, actions in trace.items():
            for event in actions:
                if event["side"] == "us":
                    events[scenario]["own_rule_firings"] += 1
                elif event["side"] == "rival":
                    key = "rival_notices" if event["action"] == "noticed" else "rival_reactions"
                    events[scenario][key] += 1
    return {
        "plans": plans, "ranking": _ranking(outcome, baseline.id),
        "weekly": weekly, "breakpoints": breakpoints,
        "risks": risks, "events": events,
    }

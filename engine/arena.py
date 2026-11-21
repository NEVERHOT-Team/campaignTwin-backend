"""AI-vs-AI arena: every strategy x scenario x seed against a paired no-campaign baseline.

Inputs are frozen before the weekly loop (AGENTS.md rule 8):
- strategies: from agents.strategist.propose (baseline included or added here);
- rival policies: S1-S3 per strategy from data.rival.get_rival_scenarios(spec, strategy),
  S4 from agents.rival_policy.red_team; depth is in percentage points;
- message appeal: SSR results if supplied, otherwise data/processed/appeal_fallback.csv.

Then engine.shopper simulates each run as a weekly closed loop: at the start of every
week our plan's contingency rules (engine.rules) react to last week's observations and
each rival (engine.rival) reacts to our posted offer, so both sides adapt during the
campaign. engine.evaluator turns paired runs into SimResult rows. The baseline never
faces a rival reaction (rivals react to our campaign) and has no rules, and it shares
every seed with the plans, so differences come only from the offer and its consequences.
ArenaResult.events records who did what each week in the middle seed.

Run `python -m engine.arena` for the demo brief with fixture context and offline fallbacks.
"""

import json
import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from contracts.campaign_spec import CampaignSpec
from contracts.models import ContextPack, RivalScenario, SimResult, SimulationPlan, Strategy
from data.loaders import load_appeal_fallback, load_market
from engine import evaluator, shopper

logger = logging.getLogger(__name__)
SCENARIOS = ("S1", "S2", "S3", "S4")
PERCENT_MECHANICS = {"percent_off", "voucher", "flash_sale", "coins_cashback"}

ProgressFn = Callable[[int, int, SimResult], None]
EvidenceFn = Callable[[CampaignSpec, Strategy | None], list[RivalScenario]]
RedTeamFn = Callable[[CampaignSpec, list[Strategy], list[RivalScenario]], list[RivalScenario]]
Appeal = dict[tuple[str, str], float]


@dataclass
class ArenaResult:
    results: list[SimResult]
    summary: list[dict[str, Any]]
    rivals: dict[str, dict[str, list[RivalScenario]]]  # strategy_id -> scenario -> rows
    campaign_weeks: int
    weeks: int
    seeds: list[int]
    appeal_source: str
    # strategy_id -> scenario -> weekly decisions in the middle seed: our rules firing and
    # rival reactions ({"week", "side": "us"|"rival", "action", "detail"}).
    events: dict[str, dict[str, list[dict[str, Any]]]] = field(default_factory=dict)
    adaptive: bool = True


# ---------------------------------------------------------------- inputs


def schedule(spec: CampaignSpec, max_weeks: int | None = None) -> tuple[int, int]:
    """(campaign weeks, total weeks including the post window)."""
    post = int(spec.timing.post_window_weeks)
    start, end = spec.timing.start_date, spec.timing.end_date
    days = (end - start).days if start and end else 28
    campaign = max(1, math.ceil(days / 7))
    if max_weeks:
        campaign = min(campaign, max(1, max_weeks - post))
    return campaign, campaign + post


URGENCY_WORDS = ("flash", "48", "hour", "only", "ชั่วโมง", "ด่วน", "วันเดียว")
FAMILY_WORDS = ("family", "member", "loyal", "ครอบครัว", "สมาชิก")


def message_style(strategy: Strategy) -> str:
    text = f"{strategy.name} {strategy.message}".casefold()
    # Family first: "members-only" is a loyalty message, not urgency.
    if any(word in text for word in FAMILY_WORDS):
        return "family"
    if any(word in text for word in URGENCY_WORDS):
        return "urgency"
    if "premium" in text or "พรีเมียม" in text:
        return "premium"
    if strategy.discount_pct > 0 and strategy.mechanic in PERCENT_MECHANICS:
        return "price"
    return "value"


def fallback_appeal(population: shopper.Population, strategies: list[Strategy]) -> Appeal:
    """Appeal per (persona, strategy) from Gan's archetype x mechanic x style table."""
    table = load_appeal_fallback()
    lookup = {
        (row.archetype, row.mechanic, row.message_style): float(row.appeal)
        for row in table.itertuples()
    }
    mechanics = {key[1] for key in lookup}
    appeal: Appeal = {}
    for strategy in strategies:
        if strategy.is_baseline:
            continue
        mechanic = strategy.mechanic if strategy.mechanic in mechanics else "percent_off"
        style = message_style(strategy)
        for persona_id, archetype in zip(
            population.persona_ids, population.persona_segments, strict=True
        ):
            appeal[(persona_id, strategy.id)] = lookup.get((archetype, mechanic, style), 0.0)
    return appeal


def _default_evidence(spec: CampaignSpec, strategy: Strategy | None) -> list[RivalScenario]:
    from data.rival import get_rival_scenarios

    return get_rival_scenarios(spec, strategy)


def _default_red_team(
    spec: CampaignSpec, strategies: list[Strategy], rivals: list[RivalScenario]
) -> list[RivalScenario]:
    from agents.rival_policy import red_team

    return red_team(spec, strategies, rivals)


def _evidence(
    spec: CampaignSpec, strategy: Strategy | None, context: ContextPack, evidence_fn: EvidenceFn
) -> list[RivalScenario]:
    try:
        return list(evidence_fn(spec, strategy))
    except Exception as error:  # fixture rivals keep the run going
        logger.warning("arena rival_evidence_failed error_type=%s", type(error).__name__)
        return [row for row in context.rivals if row.scenario in {"S1", "S2", "S3"}]


def _baseline(strategies: list[Strategy]) -> Strategy:
    for strategy in strategies:
        if strategy.is_baseline:
            return strategy
    return Strategy(
        id="baseline", name="No campaign", discount_pct=0, mechanic="none",
        target_segments=[], message="", channel_mix={}, timing={"budget_thb": 0},
        cap_per_user=None, budget_split={}, contingency_rules=[],
        rationale="No-campaign twin added by the arena.", is_baseline=True,
    )


def _target_profit(spec: CampaignSpec) -> float:
    for goal in spec.objective.smart_goals:
        if "profit" in goal.metric.casefold() and goal.unit.strip().upper() in {"THB", "฿"}:
            return float(goal.target)
    return 0.0


# ---------------------------------------------------------------- run


def run(
    plan: SimulationPlan,
    appeal: Appeal | None = None,
    *,
    base_seed: int = 0,
    params: shopper.ShopperParams | None = None,
    evidence_fn: EvidenceFn | None = None,
    red_team_fn: RedTeamFn | None = None,
    progress: ProgressFn | None = None,
    adaptive: bool = True,
) -> ArenaResult:
    """adaptive=True: weekly closed loop (our contingency rules and reactive rivals).
    adaptive=False: the earlier scheduled-rival, fixed-plan arena, kept for comparison."""
    spec, context = plan.spec, plan.context
    evidence_fn = evidence_fn or _default_evidence
    red_team_fn = red_team_fn or _default_red_team
    campaign_weeks, weeks = schedule(spec, plan.weeks)
    scenarios = [s for s in dict.fromkeys(plan.scenarios) if s in SCENARIOS]
    population = shopper.build_population(plan.n_agents, seed=base_seed, params=params)
    market = load_market()
    seeds = [base_seed * 1000 + index for index in range(max(1, plan.seeds))]

    base_strategy = _baseline(plan.strategies)
    candidates = [s for s in plan.strategies if not s.is_baseline]
    if appeal is None:
        appeal, appeal_source = fallback_appeal(population, candidates), "appeal_fallback"
    else:
        appeal_source = "ssr"

    rivals: dict[str, dict[str, list[RivalScenario]]] = {}
    for strategy in candidates:
        rows = _evidence(spec, strategy, context, evidence_fn)
        rivals[strategy.id] = {
            s: [row for row in rows if row.scenario == s] for s in ("S1", "S2", "S3")
        }
    if "S4" in scenarios and candidates:
        brief_rivals = _evidence(spec, None, context, evidence_fn)
        try:
            for row in red_team_fn(spec, candidates, brief_rivals):
                strategy_id = getattr(row, "strategy_id", None)
                if strategy_id in rivals:
                    rivals[strategy_id]["S4"] = [row]
        except Exception as error:
            logger.warning("arena red_team_failed error_type=%s", type(error).__name__)

    baselines = {
        seed: shopper.simulate(population, base_strategy, weeks, campaign_weeks, seed, None, None,
                               params)
        for seed in seeds
    }

    jobs = [(s, sc) for s in candidates for sc in scenarios if rivals[s.id].get(sc) is not None]
    results: list[SimResult] = []
    events: dict[str, dict[str, list[dict[str, Any]]]] = {}
    target = _target_profit(spec)
    trace_seed = seeds[len(seeds) // 2]
    for done, (strategy, scenario) in enumerate(jobs, start=1):
        rows = rivals[strategy.id][scenario]
        runs = {
            seed: shopper.simulate(
                population, strategy, weeks, campaign_weeks, seed, rows, appeal, params,
                adaptive=adaptive, reference=baselines[seed], market=market,
            )
            for seed in seeds
        }
        outcomes = [
            evaluator.seed_outcome(
                strategy, runs[seed], baselines[seed], campaign_weeks, population.size,
                population.ref_price, market,
            )
            for seed in seeds
        ]
        trace = runs[trace_seed]
        rival_weeks = None
        if adaptive:
            rival_weeks = {
                week for week, stats in enumerate(trace, 1)
                if stats.extra.get("rival_discount_pct", 0) > 0
                or stats.extra.get("rival_non_price")
            }
            events.setdefault(strategy.id, {})[scenario] = [
                event for stats in trace for event in stats.extra.get("events", [])
            ]
        result = evaluator.sim_result(strategy, scenario, outcomes, campaign_weeks, rows, target,
                                      rival_weeks=rival_weeks)
        results.append(result)
        if progress is not None:
            progress(done, len(jobs), result)

    weights = {sid: evaluator.scenario_weights(by_s) for sid, by_s in rivals.items()}
    return ArenaResult(
        results=results,
        summary=evaluator.summarize(results, weights),
        rivals=rivals,
        campaign_weeks=campaign_weeks,
        weeks=weeks,
        seeds=seeds,
        appeal_source=appeal_source,
        events=events,
        adaptive=adaptive,
    )


# ---------------------------------------------------------------- demo


def demo(seeds: int = 5, n_agents: int = 3000) -> ArenaResult:
    """Demo brief + fixture context + strategist templates + offline rival/appeal paths."""
    from agents.strategist import propose
    from data.loaders import load_demo_brief
    from mocks import load_mock

    spec = CampaignSpec.model_validate(load_demo_brief()["user_plan_spec"])
    context = ContextPack.model_validate(load_mock("context_pack"))
    plan = SimulationPlan(
        spec=spec, context=context, strategies=propose(spec, context),
        scenarios=list(SCENARIOS), seeds=seeds, n_agents=n_agents,
    )
    return run(plan)


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    outcome = demo()
    print(json.dumps(
        {
            "campaign_weeks": outcome.campaign_weeks,
            "weeks": outcome.weeks,
            "appeal_source": outcome.appeal_source,
            "summary": outcome.summary,
            "results": [
                {
                    "strategy": r.strategy_id, "scenario": r.scenario,
                    "incr_profit_p50": r.kpi.incr_profit.p50, "p_loss": r.kpi.p_loss,
                    "breakpoint": r.breakpoint.model_dump(),
                    "events": outcome.events.get(r.strategy_id, {}).get(r.scenario, []),
                }
                for r in outcome.results
            ],
        },
        ensure_ascii=False, indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

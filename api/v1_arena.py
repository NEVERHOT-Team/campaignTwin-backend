"""Evaluate an approved v1 review with the AI-vs-AI arena and publish a v1 RunReport.

review (frozen at approval) -> CampaignSpec + ContextPack -> strategist.propose ->
engine.arena.run -> RunReport with the shape in docs/campaigntwin-api-v1.openapi.yaml:

- `scenario_results`: the marketer's original plan, one entry per evaluated scenario;
- `plan_results`: role `original` (the brief), `conservative` (the strategist's top-ranked
  plan whose mechanic the contract can express) and `baseline` (no campaign);
- `decision`: the approved gates on the original plan in S1, like the synthetic policy;
- `commentary`: ranking, worst case and breakpoint in plain text.
Unsupported metrics are reported as unavailable instead of being invented.
"""

import logging
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from agents.strategist import propose
from contracts.campaign_spec import CampaignSpec
from contracts.models import ContextPack, SimResult, SimulationPlan, Strategy
from data.loaders import load_brands, load_market, load_personas
from engine import arena
from engine.evaluator import media_cost, plan_budget

logger = logging.getLogger(__name__)
SEEDS = 8
N_AGENTS = 3000
REPRESENTABLE = {"percent_off", "voucher", "fixed_off"}
TIERS = {"low": "budget", "mid": "mid", "high": "premium"}
Progress = Callable[[float], None]


# ---------------------------------------------------------------- review -> engine inputs


def offer_pct(review: dict[str, Any]) -> float:
    discount = review["resolved_offer"]["discount"]
    value = float(discount["value"])
    if discount["unit"] == "pct":
        return value
    price = float(review["costs"]["regular_price"]["value"])
    return 100 * value / price


def spec_from_review(review: dict[str, Any]) -> CampaignSpec:
    facts = review["facts"]
    discount = review["resolved_offer"]["discount"]
    costs = review["costs"]
    brands = load_brands()
    ours = brands.loc[brands["is_ours"]].iloc[0]
    price = float(costs["regular_price"]["value"])
    cost = float(costs["cogs"]["value"])
    fees = sum(
        float(costs[key]["value"]) for key in ("channel_commission", "payment_fee") if key in costs
    )
    mechanic = {
        "type": discount["kind"] if discount["kind"] in {"percent_off", "voucher"} else "voucher",
        "value": round(offer_pct(review), 4), "value_unit": "pct",
        "min_spend_thb": None, "cap_per_user": discount.get("cap_per_user"),
        "eligibility": discount.get("eligibility", "all"), "duration_days": None,
    }
    evaluation = review["evaluation"]
    raw = {
        "spec_version": "0.1",
        "campaign_id": f"{review['review_id']}-r{review['revision']}",
        "name": f"{ours['display_name']} {facts['start_date']}",
        "status": "ready",
        "brand": {
            "name": str(ours["display_name"]), "category": review["category"],
            "price_tier": TIERS.get(str(ours["price_tier"]), "mid"),
        },
        "objective": {"primary": facts.get("objective", "incremental_profit"),
                      "secondary": [], "smart_goals": []},
        "audience": {"description": facts.get("audience_hint") or "all buyers",
                     "include_hints": list(review.get("target_segment_ids") or []),
                     "exclude_hints": [], "geo": ["TH"]},
        "offer": {"mechanics": [mechanic], "message": None},
        "budget": {"total_thb": float(facts["budget_thb"]), "split": None,
                   "max_discount_pct": None},
        "channels": [review["resolved_product"]["channel"]],
        "timing": {"start_date": facts["start_date"], "end_date": facts["end_date"],
                   "event": None, "post_window_weeks": 2},
        "unit_economics": {"avg_price_thb": price, "margin_pct": 100 * (price - cost) / price,
                           "fee_pct": fees},
        "constraints": {"margin_floor_pct": None,
                        "max_loss_prob": float(evaluation["max_loss_probability"]),
                        "must_avoid": [], "must_include": []},
        "strategist_bounds": None,
        "competitor_hints": [],
    }
    meta = {"source": "user", "confidence": 1.0, "evidence": None}
    raw["field_meta"] = {
        key: dict(meta) for key in (
            "campaign_id", "name", "brand", "objective", "audience", "offer", "budget",
            "channels", "timing", "unit_economics", "constraints",
        )
    }
    return CampaignSpec.model_validate(raw)


def context_from_review(review: dict[str, Any], spec: CampaignSpec) -> ContextPack:
    market = {row["name"]: row for row in load_market()["segments"]}
    promo: dict[str, list[float]] = {}
    for persona in load_personas():
        promo.setdefault(persona["segment"], []).append(float(persona.get("promo_share", 0.5)))
    segments = []
    for row in review["segments"]:
        name = row["segment_id"]
        shares = promo.get(name) or [0.5]
        segments.append({
            **row,
            "archetype": market.get(name, {}).get("archetype", name),
            "weekly_purchase_prob": row["weekly_purchase_probability"],
            "price_elasticity": float(market.get(name, {}).get("price_elasticity", -1.5)),
            "promo_response": sum(shares) / len(shares),
        })
    try:
        from data.rival import get_rival_scenarios

        rivals = get_rival_scenarios(spec)
    except Exception as error:  # the arena falls back per strategy as well
        logger.warning("v1 arena rival_context_failed error_type=%s", type(error).__name__)
        rivals = []
    return ContextPack(
        category=review["category"], segments=segments, rivals=rivals, assumptions=[], gaps=[]
    )


# ---------------------------------------------------------------- arena results -> v1 shapes


def _percentiles(pct) -> dict[str, float]:
    return {"p10": float(pct.p10), "p50": float(pct.p50), "p90": float(pct.p90)}


def _available(metric, unit, value):
    return {"metric": metric, "unit": unit, "status": "available", "value": value,
            "unavailable_reason": None}


def _unavailable(metric, unit, reason="undefined_denominator"):
    return {"metric": metric, "unit": unit, "status": "unavailable", "value": None,
            "unavailable_reason": reason}


def plan_metrics(result: SimResult, strategy: Strategy) -> list[dict[str, Any]]:
    kpi = result.kpi
    spends = strategy.discount_pct > 0 or media_cost(strategy) > 0
    leakage = {key: min(max(value, 0.0), 1.0) for key, value in _percentiles(kpi.leakage).items()}
    return [
        _available("incremental_profit", "THB", _percentiles(kpi.incr_profit)),
        _available("incremental_roas", "ratio", _percentiles(kpi.iroas))
        if spends else _unavailable("incremental_roas", "ratio"),
        _available("subsidy_leakage", "fraction", leakage)
        if strategy.discount_pct > 0 else _unavailable("subsidy_leakage", "fraction"),
        _available("incremental_new_buyers", "buyers", _percentiles(kpi.new_buyers)),
        _available("probability_hit_target", "fraction", float(kpi.p_hit_target)),
        _available("probability_loss", "fraction", float(kpi.p_loss)),
    ]


def baseline_metrics(target: float) -> list[dict[str, Any]]:
    zero = {"p10": 0.0, "p50": 0.0, "p90": 0.0}
    return [
        _available("incremental_profit", "THB", dict(zero)),
        _unavailable("incremental_roas", "ratio"),
        _unavailable("subsidy_leakage", "fraction"),
        _available("incremental_new_buyers", "buyers", dict(zero)),
        _available("probability_hit_target", "fraction", 1.0 if target <= 0 else 0.0),
        _available("probability_loss", "fraction", 0.0),
    ]


def _resolved_offer(strategy: Strategy, review: dict[str, Any]) -> dict[str, Any]:
    price = float(review["costs"]["regular_price"]["value"])
    pct = round(float(strategy.discount_pct), 2)
    if strategy.mechanic == "fixed_off":
        discount = {"kind": "fixed_off", "value": round(price * pct / 100, 2), "unit": "THB"}
    else:
        kind = "voucher" if strategy.mechanic == "voucher" else "percent_off"
        discount = {"kind": kind, "value": pct, "unit": "pct"}
    discount["eligibility"] = "all"
    if strategy.cap_per_user:
        discount["cap_per_user"] = int(strategy.cap_per_user)
    return {
        "offer_id": f"arena-{strategy.id}", "discount": discount, "funder": "seller",
        "stackable": False, "min_spend_basis": None, "max_redemptions_per_order": 1,
        "eligible_order_scope": "One pack per purchase",
    }


def _changes(original: Strategy, plan: Strategy) -> list[dict[str, Any]]:
    reason = (plan.rationale.split(" [")[0] or plan.name)[:300]
    changes = []
    if round(original.discount_pct, 2) != round(plan.discount_pct, 2):
        changes.append({"field": "/offer/discount/value", "before": float(original.discount_pct),
                        "after": float(plan.discount_pct), "reason": reason})
    if round(plan_budget(original), 2) != round(plan_budget(plan), 2):
        changes.append({"field": "/subsidy_budget", "before": plan_budget(original),
                        "after": plan_budget(plan), "reason": reason})
    return changes


def plan_input(
    run_id: str, approval_id: str, review: dict[str, Any], role: str,
    strategy: Strategy | None, original: Strategy, seeds: list[int],
) -> dict[str, Any]:
    facts = review["facts"]
    if role == "original":
        offer, budget, targets = review["resolved_offer"], float(facts["budget_thb"]), list(
            review.get("target_segment_ids") or [])
    elif role == "baseline":
        offer, budget, targets = None, 0.0, []
    else:
        offer, budget = _resolved_offer(strategy, review), plan_budget(strategy)
        targets = list(strategy.target_segments)
    return {
        "plan_id": f"{run_id}/{role}", "role": role,
        "parent_plan_id": f"{run_id}/original" if role == "conservative" else None,
        "paired_baseline_id": f"{run_id}/baseline",
        "changes": _changes(original, strategy) if role == "conservative" else [],
        "approval_id": approval_id, "review_id": review["review_id"],
        "revision": review["revision"], "data_versions": review["data_versions"],
        "product": review["resolved_product"], "offer": offer, "currency": "THB",
        "subsidy_budget": max(0.0, float(budget)), "costs": review["costs"],
        "campaign_window": {"start_date": facts["start_date"], "end_date": facts["end_date"]},
        "segments": review["segments"], "target_segment_ids": targets,
        "evaluation": review["evaluation"], "seeds": list(seeds), "contingency_rules": [],
    }


def _decision(metrics: list[dict[str, Any]], evaluation: dict[str, Any]) -> tuple[str, str, str]:
    by_name = {m["metric"]: m for m in metrics}
    target = (evaluation.get("goal_target") or {}).get("minimum", 0)
    leakage = by_name["subsidy_leakage"]
    if leakage["status"] != "available":
        return ("insufficient_evidence", "missing_decision_evidence",
                "Subsidy leakage could not be evaluated for the original plan.")
    if by_name["probability_loss"]["value"] > evaluation["max_loss_probability"]:
        return ("hold", "guardrail_failed", "The original exceeded its approved loss risk in S1.")
    if by_name["incremental_profit"]["value"]["p50"] < target:
        return ("hold", "target_missed", "Median S1 incremental profit fell below the target.")
    return ("proceed", "passed", "The original met its S1 median profit and loss-risk gates.")


def _commentary(outcome: arena.ArenaResult, original: Strategy, best: Strategy | None) -> str:
    rows = outcome.summary[:3]
    ranking = "; ".join(
        f"#{r['rank']} {r['strategy_id']} EV {r['ev_incr_profit']:,.0f} THB" for r in rows
    )
    parts = [f"Top plans by likelihood-weighted incremental profit: {ranking}."]
    worst = [r for r in outcome.results if r.strategy_id == original.id]
    if worst:
        low = min(worst, key=lambda r: r.kpi.incr_profit.p50)
        text = (f"The original plan is weakest in {low.scenario} "
                f"(median {low.kpi.incr_profit.p50:,.0f} THB")
        if low.breakpoint.week:
            text += f", loss from week {low.breakpoint.week} due to {low.breakpoint.cause}"
        parts.append(text + ").")
    if best is not None:
        parts.append(f"Suggested alternative: {best.name} ({best.discount_pct:g}% "
                     f"{best.mechanic}).")
    parts.append(f"Message appeal source: {outcome.appeal_source}.")
    return " ".join(parts)


# ---------------------------------------------------------------- entry point


def evaluate(
    run_id: str, approval_id: str, review: dict[str, Any], progress: Progress | None = None
) -> dict[str, Any]:
    spec = spec_from_review(review)
    context = context_from_review(review, spec)
    evaluation = review["evaluation"]
    strategies = propose(spec, context)
    plan = SimulationPlan(
        spec=spec, context=context, strategies=strategies,
        scenarios=list(evaluation["scenarios"]), seeds=SEEDS, n_agents=N_AGENTS,
    )

    def report_progress(done: int, total: int, _result) -> None:
        if progress is not None and total:
            progress(min(0.99, done / total))

    outcome = arena.run(plan, progress=report_progress)
    by_key = {(r.strategy_id, r.scenario): r for r in outcome.results}
    order = list(dict.fromkeys(evaluation["scenarios"]))

    original = next(s for s in strategies if s.is_user_plan)
    ranked = [row["strategy_id"] for row in outcome.summary]
    candidates = {s.id: s for s in strategies if not s.is_baseline and not s.is_user_plan
                  and s.mechanic in REPRESENTABLE and s.discount_pct > 0}
    best = next((candidates[sid] for sid in ranked if sid in candidates), None)
    target = float((evaluation.get("goal_target") or {}).get("minimum", 0))

    def scenario_results(strategy: Strategy) -> list[dict[str, Any]]:
        return [
            {"scenario": s, "metrics": plan_metrics(by_key[(strategy.id, s)], strategy)}
            for s in order if (strategy.id, s) in by_key
        ]

    original_results = scenario_results(original)
    if not original_results:
        raise ValueError("The arena produced no results for the original plan.")
    plan_results = [{
        "role": "original",
        "input": plan_input(run_id, approval_id, review, "original", original, original,
                            outcome.seeds),
        "discount_pct": round(offer_pct(review), 4), "scenario_results": original_results,
    }]
    if best is not None:
        plan_results.append({
            "role": "conservative",
            "input": plan_input(run_id, approval_id, review, "conservative", best, original,
                                outcome.seeds),
            "discount_pct": round(float(best.discount_pct), 4),
            "scenario_results": scenario_results(best),
        })
    plan_results.append({
        "role": "baseline",
        "input": plan_input(run_id, approval_id, review, "baseline", None, original,
                            outcome.seeds),
        "discount_pct": 0,
        "scenario_results": [{"scenario": s, "metrics": baseline_metrics(target)} for s in order],
    })

    s1 = next((r for r in original_results if r["scenario"] == "S1"), original_results[0])
    status, reason, explanation = _decision(s1["metrics"], evaluation)
    return {
        "run_id": run_id, "approval_id": approval_id, "review_id": review["review_id"],
        "revision": review["revision"], "data_versions": review["data_versions"],
        "comparator": "paired_no_campaign",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "measurement_window": evaluation["measurement_window"],
        "scenario_results": original_results,
        "plan_results": plan_results,
        "decision": {"status": status, "reason_code": reason, "explanation": explanation},
        "commentary": _commentary(outcome, original, best),
    }

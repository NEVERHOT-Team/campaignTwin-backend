"""Paired, synthetic S1 rehearsal. The approved review supplies all market inputs."""

from datetime import date, datetime, timezone
from decimal import ROUND_DOWN, Decimal
from fractions import Fraction

import numpy as np

POLICY = "synthetic-tissue-v1"
SEEDS = range(8)


def _metric(name, unit, values, reason=None):
    value = None
    if values is not None:
        if isinstance(values, list):
            value = dict(zip(("p10", "p50", "p90"), np.percentile(values, [10, 50, 90]).tolist()))
        else:
            value = float(values)
    return {
        "metric": name, "unit": unit, "status": "unavailable" if value is None else "available",
        "value": value, "unavailable_reason": reason,
    }


def _screen(daily_probability, days, active_days, price, cost, discount_pct, budget, target):
    profit, roas, leakage = [], [], []
    subsidy = price * discount_pct / 100
    redemption_limit = (
        Fraction(str(budget)) // (Fraction(str(price)) * Fraction(str(discount_pct)) / 100)
        if discount_pct else 0
    )
    boosted = np.minimum(1, daily_probability * (1 + discount_pct / 100))
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        redemptions = overlapping = extra_units = 0
        # Every plan replays the same daily draws across the same evaluation population.
        for day in range(days):
            draws = rng.random(len(daily_probability))
            if day >= active_days or subsidy == 0:
                continue
            possible = np.flatnonzero(draws < boosted)
            remaining = max(0, redemption_limit - redemptions)
            # Randomize redemption priority independently of segment and purchase draw.
            priority = np.random.default_rng([seed, day, 1]).permutation(possible)
            redeemed = priority[:remaining]
            overlap = np.count_nonzero(draws[redeemed] < daily_probability[redeemed])
            redemptions += len(redeemed)
            overlapping += overlap
            extra_units += len(redeemed) - overlap
        spend = redemptions * subsidy
        profit.append(extra_units * (price - cost) - spend)
        if spend:
            roas.append((extra_units * price - spend) / spend)
            leakage.append(overlapping / redemptions)

    def ratio(name, unit, values):
        if len(values) == len(SEEDS):
            return _metric(name, unit, values)
        return _metric(name, unit, None, "undefined_denominator")

    metrics = [
        _metric("incremental_profit", "THB", profit),
        ratio("incremental_roas", "ratio", roas),
        ratio("subsidy_leakage", "fraction", leakage),
        _metric("incremental_new_buyers", "buyers", None, "measurement_unsupported"),
        _metric("probability_hit_target", "fraction", np.mean(np.array(profit) >= target)),
        _metric("probability_loss", "fraction", np.mean(np.array(profit) < 0)),
    ]
    return [{"scenario": "S1", "metrics": metrics}]


def _plan_input(run_id, approval_id, review, role):
    """Materialize the exact effective terms before evaluating this plan."""
    facts = review["facts"]
    original_offer = review["resolved_offer"]
    original_value = original_offer["discount"]["value"]
    baseline = role == "baseline"
    changes = []
    offer = None if baseline else original_offer
    if role == "conservative":
        half = original_value / 2
        kind = original_offer["discount"]["kind"]
        if kind == "voucher":
            half = float((Decimal(str(original_value)) / 2).quantize(
                Decimal("0.01"), rounding=ROUND_DOWN
            ))
        offer = {
            **original_offer, "offer_id": f"synthetic-{kind}-{half}",
            "discount": {**original_offer["discount"], "value": half},
        }
        changes = [{
            "field": "/offer/discount/value", "before": original_value, "after": half,
            "reason": "Reduce subsidy per pack within the approved discount bound.",
        }]
    return {
        "plan_id": f"{run_id}/{role}", "role": role,
        "parent_plan_id": f"{run_id}/original" if role == "conservative" else None,
        "paired_baseline_id": f"{run_id}/baseline", "changes": changes,
        "approval_id": approval_id, "review_id": review["review_id"],
        "revision": review["revision"],
        "data_versions": review["data_versions"], "product": review["resolved_product"],
        "offer": offer, "currency": "THB", "subsidy_budget": 0 if baseline else facts["budget_thb"],
        "costs": review["costs"], "campaign_window": {
            "start_date": facts["start_date"], "end_date": facts["end_date"],
        },
        "segments": review["segments"], "target_segment_ids": review["target_segment_ids"],
        "evaluation": review["evaluation"], "seeds": list(SEEDS), "contingency_rules": [],
    }


def evaluate(run_id, approval_id, review):
    facts = review["facts"]
    evaluation = review["evaluation"]
    discount = review["resolved_offer"]["discount"]
    price = review["costs"]["regular_price"]["value"]
    cost = review["costs"]["cogs"]["value"]
    if (
        review["data_versions"]["evaluation_policy"] != POLICY
        or review["resolved_product"]["product_id"] != "synthetic-tissue"
        or facts["channel"] != "shopee" or review["resolved_offer"]["funder"] != "seller"
        or evaluation["comparator"] != "paired_no_campaign"
        or evaluation["scenarios"] != ["S1"]
        or facts["objective"] != "incremental_profit"
        or (discount["kind"], discount["unit"]) not in {("percent_off", "pct"), ("voucher", "THB")}
        or discount.get("eligibility", "all") != "all"
        or discount.get("min_spend_thb", 0) or "cap_per_user" in discount
        or not 0 < discount["value"] <= 30
        or price != 100 or cost != 60
        or any(term["value"] != 0 for name, term in review["costs"].items()
               if name not in {"regular_price", "cogs"})
    ):
        raise ValueError("Inputs fall outside the synthetic tissue policy.")
    if discount["kind"] == "voucher":
        amount = Decimal(str(discount["value"]))
        if amount < Decimal("0.02") or amount != amount.quantize(Decimal("0.01")):
            raise ValueError("Unsupported voucher amount.")
    segments = review["segments"]
    if review["target_segment_ids"] and set(review["target_segment_ids"]) != {
        seg["segment_id"] for seg in segments
    }:
        raise ValueError("The synthetic policy requires broad eligibility.")
    counts = [seg["households"] for seg in segments]
    if not counts or min(counts) < 50 or sum(counts) > 100000:
        raise ValueError("Unsupported synthetic population.")
    rates = np.array([seg["weekly_purchase_probability"] for seg in segments])
    if not np.all((0 <= rates) & (rates <= 1)):
        raise ValueError("Unsupported segment probability.")
    daily = np.repeat(1 - (1 - rates) ** (1 / 7), counts)
    window = evaluation["measurement_window"]
    start = date.fromisoformat(facts["start_date"])
    active_days = (date.fromisoformat(facts["end_date"]) - start).days
    days = (date.fromisoformat(window["end_date"]) - start).days
    if window["start_date"] != facts["start_date"] or not 1 <= active_days <= 42:
        raise ValueError("Unsupported campaign window.")
    if days != active_days + 14:
        raise ValueError("Unsupported evaluation window.")
    target = evaluation["goal_target"]["minimum"]
    plan_results = []
    for role in ("original", "conservative", "baseline"):
        plan = _plan_input(run_id, approval_id, review, role)
        pct = 0
        if plan["offer"]:
            discount = plan["offer"]["discount"]
            pct = discount["value"]
            if discount["unit"] == "THB":
                pct = float(Decimal(str(pct)) * 100 / Decimal(str(price)))
        results = _screen(
            daily, days, active_days, price, cost, pct, plan["subsidy_budget"], target
        )
        plan_results.append({
            "role": role, "input": plan, "discount_pct": pct, "scenario_results": results,
        })
    original = plan_results[0]["scenario_results"]
    metrics = {metric["metric"]: metric["value"] for metric in original[0]["metrics"]}
    if metrics["subsidy_leakage"] is None:
        decision = (
            "insufficient_evidence", "missing_decision_evidence",
            "At least one paired seed had no subsidized purchases to evaluate.",
        )
    elif metrics["probability_loss"] > evaluation["max_loss_probability"]:
        decision = ("hold", "guardrail_failed", "The original exceeded its approved loss risk.")
    elif metrics["incremental_profit"]["p50"] < target:
        decision = ("hold", "target_missed", "Median incremental profit fell below target.")
    else:
        decision = ("proceed", "passed", "The original met its median profit and loss-risk gates.")
    return {
        "run_id": run_id, "approval_id": approval_id, "review_id": review["review_id"],
        "revision": review["revision"], "data_versions": review["data_versions"],
        "comparator": "paired_no_campaign", "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "measurement_window": window, "scenario_results": original, "plan_results": plan_results,
        "decision": dict(zip(("status", "reason_code", "explanation"), decision)),
    }

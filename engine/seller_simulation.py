"""Versioned, paired single-pack campaign rehearsal for the planner API."""

from datetime import date, datetime, timezone
from decimal import ROUND_DOWN, Decimal

import numpy as np

from engine.seller_ledger import (
    Settlement,
    metric,
    simulate,
    subsidy_per_pack,
    valid_discount_amount,
)

POLICY = "synthetic-voucher-v2"
SEEDS = range(8)


def _plan_input(run_id, approval_id, review, role, proposal):
    """Materialize the exact effective terms before evaluating this plan."""
    facts = review["facts"]
    original_offer = review["resolved_offer"]
    original_value = original_offer["discount"]["value"]
    baseline = role == "baseline"
    changes = []
    offer = None if baseline else original_offer
    if role == "conservative":
        value = original_value / 2
        kind = original_offer["discount"]["kind"]
        if kind == "voucher":
            value = float(
                (Decimal(str(original_value)) / 2).quantize(Decimal("0.01"), rounding=ROUND_DOWN)
            )
        if proposal and proposal["value"] is not None:
            value = proposal["value"]
        offer = {
            **original_offer,
            "offer_id": f"{original_offer['offer_id']}-alternative-{value}",
            "discount": {**original_offer["discount"], "value": value},
        }
        changes = [
            {
                "field": "/offer/discount/value",
                "before": original_value,
                "after": value,
                "reason": "Reduce subsidy per pack within the approved discount bound.",
            }
        ]
    costs = review["costs"]
    if baseline:
        costs = {
            **costs,
            "paid_media": {
                **costs["paid_media"],
                "value": 0,
                "source": "default",
                "basis": "No campaign media in the paired baseline",
                "evidence_ref": POLICY,
            },
        }
    return {
        "plan_id": f"{run_id}/{role}",
        "role": role,
        "parent_plan_id": f"{run_id}/original" if role == "conservative" else None,
        "paired_baseline_id": f"{run_id}/baseline",
        "changes": changes,
        "approval_id": approval_id,
        "review_id": review["review_id"],
        "revision": review["revision"],
        "data_versions": review["data_versions"],
        "product": review["resolved_product"],
        "offer": offer,
        "currency": "THB",
        "total_budget_thb": 0 if baseline else facts["budget_thb"],
        "subsidy_budget": 0
        if baseline
        else float(
            Decimal(str(facts["budget_thb"])) - Decimal(str(review["costs"]["paid_media"]["value"]))
        ),
        "costs": costs,
        "campaign_window": {
            "start_date": facts["start_date"],
            "end_date": facts["end_date"],
        },
        "segments": review["segments"],
        "target_segment_ids": review["target_segment_ids"],
        "evaluation": review["evaluation"],
        "seeds": list(SEEDS),
        "contingency_rules": [] if baseline else review["evaluation"]["contingency_rules"],
    }


def _gates(scenario, plan):
    values = {m["metric"]: m["value"] for m in scenario["metrics"]}
    terms = plan["evaluation"]
    cost = Settlement.from_plan(plan)
    offer = plan["offer"]["discount"]
    discount = subsidy_per_pack(offer, cost.price)
    unit_profit = cost.ledger(1, 1, discount, Decimal(0))["contribution_profit"]

    def gate(metric_id, observed, condition, threshold, passed, unit, statistic):
        return {
            "scenario": scenario["scenario"],
            "metric": metric_id,
            "observed": observed,
            "statistic": statistic,
            "unit": unit,
            "condition": condition,
            "threshold": threshold,
            "status": "unavailable" if passed is None else "pass" if passed else "fail",
            "unavailable_reason": "undefined_denominator" if observed is None else None,
        }

    leakage = values["subsidy_leakage"]
    target = terms["goal_target"]["minimum"]
    risk = terms["max_loss_probability"]
    profit = values["incremental_profit"]["p50"]
    loss = values["probability_loss"]
    spend = max(r["campaign_spend"] for r in scenario["seed_ledgers"])
    checks = [
        gate(
            "subsidy_leakage",
            leakage and leakage["p50"],
            "available",
            None,
            True if leakage is not None else None,
            "fraction",
            "p50",
        ),
        gate("incremental_profit", profit, "at_least", target, profit >= target, "THB", "p50"),
        gate("probability_loss", loss, "at_most", risk, loss <= risk, "fraction", "value"),
        gate(
            "campaign_spend",
            spend,
            "at_most",
            plan["total_budget_thb"],
            spend <= plan["total_budget_thb"],
            "THB",
            "max",
        ),
    ]
    floor = terms["min_unit_contribution_thb"]
    if floor is not None:
        checks.append(
            gate(
                "unit_contribution",
                float(unit_profit),
                "at_least",
                floor,
                unit_profit >= Decimal(str(floor)),
                "THB_per_pack",
                "value",
            )
        )
    return checks


def _complete_pairing(plans):
    baseline = next((row for row in plans if row["role"] == "baseline"), None)
    if not baseline:
        return False
    common = (
        "approval_id",
        "review_id",
        "revision",
        "product",
        "currency",
        "segments",
        "target_segment_ids",
        "campaign_window",
        "evaluation",
        "data_versions",
        "seeds",
        "paired_baseline_id",
    )
    reference = baseline["input"]
    if (reference["plan_id"] != reference["paired_baseline_id"] or reference["currency"] != "THB"
            or reference["offer"] is not None or reference["total_budget_thb"] != 0
            or reference["contingency_rules"]):
        return False
    roles = [row["role"] for row in plans]
    if roles not in (["original", "baseline"], ["original", "conservative", "baseline"]):
        return False
    identifiers = [row["input"]["plan_id"] for row in plans]
    if len(set(identifiers)) != len(identifiers):
        return False
    scenarios = reference["evaluation"]["scenarios"]
    seeds = reference["seeds"]
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or [s["scenario"] for s in baseline["scenario_results"]] != scenarios
    ):
        return False
    for row in plans:
        plan = row["input"]
        if (plan["role"] != row["role"] or
                {k: v for k, v in plan["costs"].items() if k != "paid_media"} !=
                {k: v for k, v in reference["costs"].items() if k != "paid_media"}):
            return False
        if any(plan[key] != reference[key] for key in common):
            return False
        if [s["scenario"] for s in row["scenario_results"]] != scenarios:
            return False
        for result, paired in zip(
            row["scenario_results"], baseline["scenario_results"], strict=True
        ):
            observations = result["seed_ledgers"]
            base = paired["seed_ledgers"]
            if (result["paired_seed_count"] != len(seeds)
                    or [r["seed"] for r in observations] != seeds
                    or [r["seed"] for r in base] != seeds):
                return False
            if any(
                r["baseline_units"] != b["units"]
                or r["baseline_contribution"] != b["contribution_profit"]
                for r, b in zip(observations, base, strict=True)
            ):
                return False
    return True


def compare_plans(plans):
    if not _complete_pairing(plans):
        return {
            "status": "insufficient_evidence",
            "selected_plan_id": None,
            "assessments": [],
            "reason": "incomplete_pairing",
        }
    original = plans[0]
    assessments = []
    eligible = []
    for row in plans:
        plan = row["input"]
        if plan["role"] == "baseline":
            continue
        results = row["scenario_results"]
        checks = [g for s in results for g in _gates(s, plan)]
        statuses = {g["status"] for g in checks}
        state = (
            "unscorable"
            if "unavailable" in statuses
            else "ineligible"
            if "fail" in statuses
            else "eligible"
        )
        downside = []
        medians = []
        differences = []
        for scenario, reference in zip(results, original["scenario_results"], strict=True):
            paired = scenario["seed_ledgers"]
            original_rows = reference["seed_ledgers"]
            profit = next(
                m["value"] for m in scenario["metrics"] if m["metric"] == "incremental_profit"
            )
            downside.append(profit["p10"])
            medians.append(profit["p50"])
            diff = [
                r["incremental_profit"] - o["incremental_profit"]
                for r, o in zip(paired, original_rows, strict=True)
            ]
            differences.append(
                {
                    "scenario": scenario["scenario"],
                    "metric": metric("profit_vs_original", "THB", diff),
                }
            )
        item = {
            "plan_id": plan["plan_id"],
            "role": plan["role"],
            "eligibility": state,
            "worst_p10_profit_thb": min(downside),
            "checks": checks,
            "versus_original": differences,
        }
        assessments.append(item)
        if state == "eligible":
            eligible.append((item, medians))
    recommendation = {"status": "hold", "selected_plan_id": None, "assessments": assessments}
    if any(a["eligibility"] == "unscorable" for a in assessments):
        recommendation["status"] = "insufficient_evidence"
    elif eligible:
        best_medians = [max(column) for column in zip(*(v for _, v in eligible), strict=True)]

        def rank(entry):
            item, medians = entry
            regret = max(best - value for best, value in zip(best_medians, medians, strict=True))
            return item["worst_p10_profit_thb"], -regret, item["role"] == "original"

        winner, _ = max(eligible, key=rank)
        recommendation.update(
            status="keep" if winner["role"] == "original" else "revise",
            selected_plan_id=winner["plan_id"],
        )
    return recommendation


def _commercial_evidence(review):
    """Count every effective cost once, using the same sources pinned for the run."""
    catalog = review.get("catalog_evidence") or {}
    sources = {row["evidence_ref"]: row for row in catalog.get("sources", [])}
    planner_refs = {f"/commercial_inputs/{key}" for key in review.get("commercial_inputs", {})}
    classes = dict.fromkeys(
        ("business_input", "observed_local_market", "transferred_behavior",
         "assumption", "synthetic_fixture", "unavailable"), 0
    )
    labels = {"planner_provided": 0, "catalog_data": 0, "assumed_or_synthetic": 0}
    for cost in review["costs"].values():
        ref = cost["evidence_ref"]
        if ref in planner_refs:
            group, evidence_class = "planner_provided", "business_input"
        else:
            group = "catalog_data" if cost["source"] == "data" else "assumed_or_synthetic"
            fallback = "assumption" if catalog else "synthetic_fixture"
            evidence_class = sources[ref].get("evidence_class") if ref in sources else fallback
        labels[group] += 1
        classes[evidence_class if evidence_class in classes else "unavailable"] += 1
    return {
        "commercial_terms": {"total": len(review["costs"]), **labels},
        "commercial_evidence_classes": classes,
    }


def evaluate(run_id, approval_id, review, *, proposal=None):
    facts, terms = review["facts"], review["evaluation"]
    if (
        review["data_versions"]["evaluation_policy"] != POLICY
        or terms["comparator"] != "paired_no_campaign"
        or facts["channel"] != "shopee"
        or terms["scenarios"] not in (["S1"], ["S1", "S2"])
        or facts["objective"] != "incremental_profit"
    ):
        raise ValueError("Unsupported approved comparison")
    segments = review["segments"]
    counts = [s["households"] for s in segments]
    rates = np.array([s["weekly_purchase_probability"] for s in segments])
    if (
        not counts
        or min(counts) < 50
        or sum(counts) > 100000
        or not np.all((0 <= rates) & (rates <= 1))
    ):
        raise ValueError("Unsupported cohort")
    daily = np.repeat(1 - (1 - rates) ** (1 / 7), counts)
    start = date.fromisoformat(facts["start_date"])
    active_days = (date.fromisoformat(facts["end_date"]) - start).days
    window = terms["measurement_window"]
    days = (date.fromisoformat(window["end_date"]) - start).days
    if (
        window["start_date"] != facts["start_date"]
        or not 1 <= active_days <= 42
        or days < active_days
    ):
        raise ValueError("Unmatched campaign windows")

    plans = []
    shortfall = None
    price = Decimal(str(review["costs"]["regular_price"]["value"]))
    for role in ("original", "conservative", "baseline"):
        if role == "conservative" and terms["alternative_policy"] == "fixed_offer":
            shortfall = "discount_change_not_permitted"
            continue
        plan = _plan_input(run_id, approval_id, review, role, proposal)
        if role == "conservative":
            candidate = plan["offer"]["discount"]
            subsidy = subsidy_per_pack(candidate, price)
            original_subsidy = subsidy_per_pack(review["resolved_offer"]["discount"], price)
            if subsidy == 0 or subsidy == original_subsidy:
                shortfall = "no_distinct_positive_subsidy_at_satang_precision"
                continue
            if not valid_discount_amount(candidate, price) or subsidy > original_subsidy:
                shortfall = "alternative_outside_listing_bounds"
                continue
        pct = Decimal(0)
        if plan["offer"]:
            discount = plan["offer"]["discount"]
            pct = Decimal(str(discount["value"]))
            if discount["unit"] == "THB":
                pct = pct * 100 / Decimal(str(plan["costs"]["regular_price"]["value"]))
        results = [
            simulate(plan, scenario, daily, active_days, days) for scenario in terms["scenarios"]
        ]
        plans.append(
            {"role": role, "input": plan, "discount_pct": float(pct), "scenario_results": results}
        )
    ranking = compare_plans(plans)
    original = next(iter(ranking["assessments"]), {"eligibility": "unscorable", "checks": []})
    ranking["strategy"] = proposal or {
        "source": "default",
        "reason": terms["alternative_policy"],
        "value": None,
    }
    ranking["alternative_shortfall"] = shortfall
    ranking["confidence"] = {
        "label": "unscorable" if ranking["status"] == "insufficient_evidence" else "low",
        "rubric": "voucher-evidence-v1",
        "reasons": [
            "Price response and carryover are simulation assumptions.",
            "No aligned seller-specific held-out profit validation.",
        ],
    }
    if ranking["status"] == "insufficient_evidence":
        ranking["confidence"]["reasons"].append("Incomplete decision evidence across paired seeds.")

    failures = {g["metric"] for g in original["checks"] if g["status"] == "fail"}
    if ranking.get("reason") == "incomplete_pairing":
        decision = (
            "insufficient_evidence",
            "missing_decision_evidence",
            "The scenario, population or seed ledgers could not be paired.",
        )
    elif original["eligibility"] == "unscorable":
        decision = (
            "insufficient_evidence",
            "missing_decision_evidence",
            "At least one paired seed had no subsidized purchases to evaluate.",
        )
    elif failures - {"incremental_profit"}:
        decision = ("hold", "guardrail_failed", "The original breached an approved guardrail.")
    elif failures:
        decision = (
            "hold",
            "target_missed",
            "The original missed the approved median profit target.",
        )
    else:
        decision = ("proceed", "passed", "The original met the approved goal and guardrails.")
    if ranking["status"] == "revise":
        decision = (
            *decision[:2],
            decision[2] + " The smaller discount is the highest-ranked "
            "eligible plan under the approved comparison.",
        )
    coverage = {
        **_commercial_evidence(review),
        "scenario_basis": [
            {
                "scenario": scenario,
                "type": "passive" if scenario == "S1" else "assumed_stress",
                "active_demand_reduction_pct": 0
                if scenario == "S1"
                else terms["behavior"]["rival_demand_reduction_pct"],
                "likelihood": None,
                "reason": "No calibrated rival likelihood for this configuration.",
            }
            for scenario in terms["scenarios"]
        ],
        "customer_voice": {
            "status": "unavailable",
            "reason": "No persona reactions were scored for these plans.",
        },
        "source": "paired_simulation",
        "measurement_scope": "one_kept_pack_per_order",
        "historical_comparison_url": f"/api/v2/runs/{run_id}/historical-comparison",
    }
    return {
        "evidence": coverage,
        "run_id": run_id,
        "approval_id": approval_id,
        "review_id": review["review_id"],
        "revision": review["revision"],
        "data_versions": review["data_versions"],
        "comparator": "paired_no_campaign",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "measurement_window": window,
        "scenario_results": plans[0]["scenario_results"],
        "plan_results": plans,
        "recommendation": ranking,
        "decision": dict(zip(("status", "reason_code", "explanation"), decision)),
    }

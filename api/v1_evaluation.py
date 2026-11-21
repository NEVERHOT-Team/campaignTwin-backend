"""Present the approved policy checks alongside their persisted simulation evidence."""

import operator

from engine.synthetic import POLICY


def _check(metrics, name, statistic, unit, condition, threshold=None):
    metric = metrics.get(name)
    observed = None
    reason = "missing_evidence"
    if metric is not None:
        observed = metric["value"]
        if statistic == "p50" and observed is not None:
            observed = observed["p50"]
        reason = metric["unavailable_reason"]
    if observed is None:
        status = "unavailable"
    elif condition == "available":
        status = "pass"
    else:
        compare = operator.ge if condition == "at_least" else operator.le
        status = "pass" if compare(observed, threshold) else "fail"
    return {
        "metric": name, "statistic": statistic, "unit": unit, "observed": observed,
        "condition": condition, "threshold": threshold, "status": status,
        "unavailable_reason": reason,
    }


def business_evaluation(report):
    plans = report.get("plan_results", [])
    comparisons = [{
        "role": row["role"], "plan_id": row["input"]["plan_id"],
        "discount_pct": row["discount_pct"], "scenario_results": row["scenario_results"],
    } for row in plans]
    original = next((row for row in plans if row["role"] == "original"), None)
    if original is None:
        comparisons.insert(0, {
            "role": "original", "plan_id": None, "discount_pct": None,
            "scenario_results": report["scenario_results"],
        })
    policy = report["data_versions"]["evaluation_policy"]
    checks = []
    reason = None
    if original is None:
        reason = "This report has no stored original plan input with approved thresholds."
    elif policy != POLICY:
        reason = "Decision checks are unavailable for this evaluation policy."
    else:
        evaluation = original["input"]["evaluation"]
        target = evaluation["goal_target"]
        supports_profit = target and (target["metric"], target["unit"]) == (
            "incremental_profit", "THB"
        )
        if not supports_profit:
            reason = "The stored evaluation terms have no incremental profit target in THB."
        for scenario in original["scenario_results"]:
            metrics = {item["metric"]: item for item in scenario["metrics"]}
            gates = [_check(metrics, "subsidy_leakage", "p50", "fraction", "available")]
            if supports_profit:
                gates.append(_check(metrics, "incremental_profit", "p50", "THB", "at_least",
                                    target["minimum"]))
            gates.append(_check(metrics, "probability_loss", "value", "fraction", "at_most",
                                evaluation["max_loss_probability"]))
            for check in gates:
                checks.append({"scenario": scenario["scenario"], **check})
    return {
        **{key: report[key] for key in ("run_id", "approval_id", "review_id", "revision",
                                        "comparator", "measurement_window", "decision")},
        "evaluation_policy": policy, "checks": checks, "comparisons": comparisons,
        "checks_unavailable_reason": reason,
    }

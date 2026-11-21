"""Project the saved v2 Arena report into the seller's business comparison."""


def business_evaluation(report):
    ranking = report["recommendation"]
    original = next((a for a in ranking["assessments"] if a["role"] == "original"), None)
    return {
        **{
            key: report[key]
            for key in (
                "run_id",
                "approval_id",
                "review_id",
                "revision",
                "comparator",
                "measurement_window",
                "decision",
                "recommendation",
                "evidence",
            )
        },
        "evaluation_policy": report["data_versions"]["evaluation_policy"],
        "checks": original["checks"] if original else [],
        "checks_unavailable_reason": None if original else "Incomplete paired ledgers.",
        "comparisons": [
            {
                "role": row["role"],
                "plan_id": row["input"]["plan_id"],
                "discount_pct": row["discount_pct"],
                "scenario_results": row["scenario_results"],
            }
            for row in report["plan_results"]
        ],
    }

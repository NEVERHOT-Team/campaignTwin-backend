from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

import db
from agents import rival_policy
from api import v1_arena
from api.v1_contract import contract, validator
from data.loaders import load_market
from engine import evaluator, shopper
from main import app


@pytest.fixture(scope="module")
def tissue_run(tmp_path_factory):
    reviews = "/api/v1/campaign-reviews"
    body = contract()["paths"][reviews]["post"]["requestBody"]["content"]["application/json"]
    fields = body["examples"]["arenaRehearsal"]["value"]["fields"]
    calls = []
    run_arena = v1_arena.arena.run

    controls = []
    simulate = shopper.simulate

    def capture_weekly(population, strategy, weeks, campaign_weeks, seed, *args, **kwargs):
        stats = simulate(population, strategy, weeks, campaign_weeks, seed, *args, **kwargs)
        if strategy.is_baseline:
            controls.append(evaluator.seed_outcome(
                strategy, stats, stats, campaign_weeks, population.size, population.ref_price,
                load_market(),
            ))
        return stats

    def capture(plan, *args, **kwargs):
        # Observe exclusively the paired controls inside the original arena call.
        with pytest.MonkeyPatch.context() as observer:
            observer.setattr(shopper, "simulate", capture_weekly)
            result = run_arena(plan, *args, **kwargs)
        calls.append((plan, result))
        return result

    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("DB_MODE", "json")
        patch.setenv("OPENAI_API_KEY", "")
        patch.setattr(db, "RUNS_DIR", tmp_path_factory.mktemp("v1-report"))
        patch.setattr(rival_policy, "call_llm", lambda *args, **kwargs: None)
        patch.setattr(v1_arena, "SEEDS", 3)
        patch.setattr(v1_arena, "N_AGENTS", 600)
        patch.setattr(v1_arena.arena, "run", capture)
        with TestClient(app) as client:
            review = client.post(reviews, json={"fields": fields}).json()
            assert review["status"] == "ready_for_approval", review["blockers"]
            approval = client.post(
                f"{reviews}/{review['review_id']}/approvals",
                json={"revision": review["revision"], "confirmed": True},
            )
            assert approval.status_code == 201
            run = client.post("/api/v1/runs", json={
                "approval_id": approval.json()["approval_id"], "submission_id": str(uuid4()),
            })
            assert run.status_code == 202
            status = client.get(run.json()["status_url"]).json()
            assert status["status"] == "done", status
            report = client.get(status["report_url"]).json()
            validator("RunReport").validate(report)
    assert len(calls) == 1
    simulation, outcome = calls[0]
    assert len(controls) == len(outcome.seeds)
    expected_baseline = [row.model_dump() for row in evaluator.median_rows(controls)]
    assert all(rows == expected_baseline for rows in report["weekly"]["baseline"].values())
    return review, report, simulation, outcome


def test_arena_report_preserves_every_plan_scenario_and_legacy_values(tissue_run):
    review, report, simulation, outcome = tissue_run
    assert report["evaluation_engine"] == "arena"
    assert review["costs"]["regular_price"]["source"] == "data"
    assert review["costs"]["cogs"]["source"] == "default"
    sources = report["field_sources"]
    assert sources["unit_economics.avg_price_thb"] == "data"
    assert sources["constraints.margin_floor_pct"] == "default"
    assert sources["unit_economics.margin_pct"] == "derived"
    assert sources["timing.event"] == "user"
    assert simulation.spec.timing.event == "11.11"
    assert simulation.spec.channels == ["shopee", "line"]
    plans = {plan["plan_id"]: plan for plan in report["plans"]}
    assert set(plans) == {s.id for s in simulation.strategies}
    assert {"baseline", "user-plan"} <= set(plans)
    assert sum(key.startswith("g0-") for key in plans) == 6
    assert plans["user-plan"]["role"] == "user"
    assert plans["user-plan"]["discount_pct"] == 40
    for strategy in simulation.strategies:
        assert plans[strategy.id]["mechanic"] == strategy.mechanic
        assert plans[strategy.id]["discount_pct"] == strategy.discount_pct
    actual = {(pid, sc) for pid, plan in plans.items() if pid != "baseline"
              for sc in plan["scenarios"]}
    assert actual == {(row.strategy_id, row.scenario) for row in outcome.results}
    assert any(sc == "S4" for _, sc in actual)
    for row in outcome.results:
        sid, sc = row.strategy_id, row.scenario
        metrics = plans[sid]["scenarios"][sc]
        for name, value in (
            ("incremental_profit", row.kpi.incr_profit), ("iroi", row.kpi.iroi),
            ("iroas", row.kpi.iroas), ("new_buyers", row.kpi.new_buyers),
            ("leakage", row.kpi.leakage),
        ):
            assert metrics[name] == (value.model_dump() if value is not None else None)
        assert metrics["leakage_label"] == row.kpi.leakage_label
        assert metrics["p_loss"] == row.kpi.p_loss
        assert metrics["p_hit_target"] == row.kpi.p_hit_target
        assert report["weekly"][sid][sc] == [week.model_dump() for week in row.weekly]
        assert report["breakpoints"][sid][sc] == {
            "week": row.breakpoint.week, "cause": row.breakpoint.cause, "peak_week": row.peak_week,
        }

    assert any(bp["cause"] == "never_profitable" and bp["week"] is None
               for bp in report["breakpoints"]["user-plan"].values())
    baseline = report["weekly"]["baseline"]
    assert set(baseline) == {sc for _, sc in actual}
    assert all(weeks == baseline["S1"] for weeks in baseline.values())
    assert [w["week"] for w in baseline["S1"]] == list(range(1, outcome.weeks + 1))
    assert any(w["sales"] > 0 for w in baseline["S1"])

    original = next(p for p in report["plan_results"] if p["role"] == "original")
    assert original["scenario_results"] == report["scenario_results"]
    for scenario in original["scenario_results"]:
        legacy = {m["metric"]: m["value"] for m in scenario["metrics"]}
        assert legacy["incremental_profit"] == plans["user-plan"]["scenarios"][
            scenario["scenario"]]["incremental_profit"]


def test_arena_ranking_events_and_competitive_outlook(tissue_run):
    _, report, _, outcome = tissue_run
    ranking = report["ranking"]
    assert [entry["plan_id"] for entry in ranking] == [
        row["strategy_id"] for row in outcome.summary
    ] + ["baseline"]
    for entry, summary in zip(ranking[:-1], outcome.summary, strict=True):
        assert entry["rank"] == summary["rank"]
        assert entry["weighted_profit"] == summary["ev_incr_profit"]
        assert entry["rank_reason"]
    assert ranking[-1]["rank"] is None
    for sc, counts in report["events"].items():
        trace = [e for plan in outcome.events.values() for e in plan.get(sc, [])]
        assert counts["rival_reactions"] == sum(
            e["side"] == "rival" and e["action"] != "noticed" for e in trace
        )
        assert counts["own_rule_firings"] == sum(e["side"] == "us" for e in trace)
    outlook = report["competitive_outlook"]
    assert outlook["user_plan"]["plan_id"] == "user-plan"
    assert outlook["top_ranked_plan"]["plan_id"] == ranking[0]["plan_id"]
    for entry in outlook.values():
        assert entry["prior_label"] == "data+default"
        assert {r["display_name"] for r in entry["rivals"]} == {"Scott", "Zilk", "House brand"}
        assert any(c["source_url"].startswith("https://") for c in entry["similar_campaigns"])
    assert report["customer_voice"] == {
        "status": "unavailable", "reason": "rule-based appeal table (label default)",
    }


def test_invalid_structured_event_rejected_before_approval():
    with TestClient(app) as client:
        response = client.post("/api/v1/campaign-reviews", json={"fields": {"event": "surprise"}})
    assert response.status_code == 422
    assert any(issue["field"] == "/fields/event" for issue in response.json()["issues"])

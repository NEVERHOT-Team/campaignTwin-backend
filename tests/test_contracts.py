import json
from pathlib import Path

import pytest

from contracts.models import BriefParseResponse, ContextPack, DemoReplay, Report, RunStatus
from mocks import load_mock

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("name", "model"),
    [
        ("campaign_spec", BriefParseResponse),
        ("context_pack", ContextPack),
        ("run_status", RunStatus),
        ("report", Report),
    ],
)
def test_mock_matches_its_contract(name, model):
    model.model_validate(load_mock(name))


def test_demo_replay_projects_the_completed_v1_tissue_run():
    saved = json.loads((ROOT / "demo" / "replay.json").read_text(encoding="utf-8"))
    replay = DemoReplay.model_validate(saved)
    status, report = saved["v1"]["status"], saved["v1"]["report"]
    assert replay.status.run_id == report["run_id"] == status["run_id"]
    assert replay.status.status == status["status"] == "done"
    original = next(plan for plan in report["plan_results"] if plan["role"] == "original")
    assert original["input"]["product"]["choice_id"] == "cellox-tissue-shopee"
    assert report["evaluation_engine"] == "arena"

    plans = {p["plan_id"]: p["scenarios"] for p in report["plans"]}
    expected = {(sid, scenario) for sid, rows in plans.items() for scenario in rows}
    scorecard = {(r["strategy_id"], r["scenario"]): r for r in replay.report.scorecard}
    assert set(scorecard) == expected
    for (sid, scenario), card in scorecard.items():
        assert card["incr_profit"] == plans[sid][scenario]["incremental_profit"]
        crossing = report["breakpoints"][sid][scenario]
        assert (card["breakpoint"]["week"], card["breakpoint"]["cause"]) == (
            crossing["week"], crossing["cause"]
        )
    assert replay.report.weekly == [
        {"strategy_id": sid, "scenario": scenario, **week}
        for sid, scenarios in plans.items() for scenario in scenarios
        for week in report["weekly"][sid][scenario]
    ]
    assert replay.report.recommendation["strategy_id"] == report["ranking"][0]["plan_id"]
    assert replay.report.risks == report["risks"]

    partial = {(p.strategy_id, p.scenario): p for p in replay.status.partial_results}
    required = ("iroas", "leakage", "p_hit_target")
    assert set(partial) == {
        (sid, scenario) for sid, scenario in expected
        if all(plans[sid][scenario][key] is not None for key in required)
    }
    for (sid, scenario), result in partial.items():
        kpi = plans[sid][scenario]
        assert [row.model_dump() for row in result.weekly] == report["weekly"][sid][scenario]
        assert result.kpi.model_dump() == {
            "incr_profit": kpi["incremental_profit"], "iroas": kpi["iroas"],
            "leakage": kpi["leakage"], "new_buyers": kpi["new_buyers"],
            "p_hit_target": kpi["p_hit_target"], "p_loss": kpi["p_loss"],
        }

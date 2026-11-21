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


def test_demo_replay_matches_mocks_and_demo_story():
    replay = DemoReplay.model_validate_json((ROOT / "demo" / "replay.json").read_text())
    assert replay.status == RunStatus.model_validate(load_mock("run_status"))
    assert replay.report == Report.model_validate(load_mock("report"))
    results = {
        (result.scenario, result.strategy_id): result for result in replay.status.partial_results
    }
    deep = "deep-1111"
    targeted = "targeted-loyalty"
    assert len(results) == 9
    assert results["S1", deep].kpi.incr_profit.p50 > results["S1", targeted].kpi.incr_profit.p50
    assert results["S3", deep].breakpoint.model_dump() == {
        "week": 3,
        "cause": "rival undercut",
        "share": 0.68,
    }
    assert results["S3", deep].kpi.incr_profit.p50 < 0
    assert replay.report.recommendation["strategy_id"] == targeted
    assert len(replay.report.scorecard) == 9
    assert all(
        results[scenario, targeted].kpi.incr_profit.p10 > 0 for scenario in ("S1", "S2", "S3")
    )

    for (scenario, _), result in results.items():
        baseline = results[scenario, "baseline"]
        assert [row.week for row in result.weekly] == list(range(1, 7))
        profit = result.kpi.incr_profit
        paired_weeks = zip(result.weekly, baseline.weekly, strict=True)
        assert profit.p50 == sum(week.profit - base.profit for week, base in paired_weeks)
        assert profit.p10 <= profit.p50 <= profit.p90

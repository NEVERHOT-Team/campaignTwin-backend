import pytest

from engine import backtest, shopper


@pytest.fixture(scope="module")
def result():
    return backtest.run_backtest(seeds=(0, 1), n_agents=2000)


def test_backtest_reports_lift_and_label(result):
    assert result.campaign_id == "18"
    assert result.weeks == 8
    assert result.discount_pct == pytest.approx(22.2469, abs=1e-3)
    assert len(result.lift_by_seed) == 2
    assert result.evaluation["label"] in {"Calibrated", "Partially calibrated", "Uncalibrated"}
    assert result.evaluation["simulated"] == pytest.approx(result.simulated_lift_pct, abs=1e-3)
    assert result.simulated_lift_pct > 0  # a discount raises category sales


def test_backtest_is_deterministic(result):
    again = backtest.run_backtest(seeds=(0, 1), n_agents=2000)
    assert again.to_dict() == result.to_dict()


def test_coupon_split_uses_treated_share_not_redemption_rate():
    record = dict(backtest.load_backtest(), shelf_share=0.5, treated_share=0.4)
    out = backtest.run_backtest(seeds=(0,), n_agents=1500, record=record)
    assert out.shelf_share == 0.5 and out.treated_share == 0.4
    assert out.labels["shelf_share"] == "data" and out.labels["treated_share"] == "data"
    assert out.simulated_redemption_rate is not None
    assert out.observed_redemption_rate == pytest.approx(0.0070609)


def test_sweep_reports_one_row_per_override():
    rows = backtest.sweep([{"beta_price": 3.0}, {"beta_price": 5.0}], seeds=(0,), n_agents=1000)
    assert [row["params"] for row in rows] == [{"beta_price": 3.0}, {"beta_price": 5.0}]
    assert all("simulated_lift_pct" in row for row in rows)
    assert shopper.default_params() is shopper.default_params()  # cached

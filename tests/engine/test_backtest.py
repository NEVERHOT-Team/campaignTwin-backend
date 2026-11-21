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


def test_backtest_uses_measured_shelf_and_coupon_depths(result):
    record = backtest.load_backtest()
    shelf = record["mechanics"]["all_tissue_purchases"]["shelf_depth_fraction"]
    coupon = record["mechanics"]["matched_campaign_redemptions"]["coupon_depth_fraction"]
    assert result.shelf_discount_pct == pytest.approx(100 * shelf, abs=1e-3)
    assert result.coupon_discount_pct == pytest.approx(100 * coupon, abs=1e-3)
    assert result.treated_share == pytest.approx(record["treated_share"])
    assert result.labels["shelf_pct"] == "data" and result.labels["treated_share"] == "data"
    # The observed redemption rate is a check, never an input.
    assert result.observed_redemption_rate == pytest.approx(record["observed_redemption_rate"])
    assert result.simulated_redemption_rate is not None


def test_backtest_is_deterministic(result):
    again = backtest.run_backtest(seeds=(0, 1), n_agents=2000)
    assert again.to_dict() == result.to_dict()


def test_fallback_split_without_measured_depths():
    record = {k: v for k, v in backtest.load_backtest().items() if k != "mechanics"}
    inputs = backtest.campaign_inputs(dict(record, shelf_share=0.5))
    assert inputs.shelf_pct == pytest.approx(inputs.coupon_pct)
    assert inputs.labels["shelf_pct"] == "data"
    bare = backtest.campaign_inputs({k: v for k, v in record.items() if k != "shelf_share"})
    assert bare.coupon_pct == 0 and bare.labels["shelf_pct"] == "default"


def test_sweep_reports_one_row_per_override():
    rows = backtest.sweep([{"beta_price": 3.0}, {"beta_price": 5.0}], seeds=(0,), n_agents=1000)
    assert [row["params"] for row in rows] == [{"beta_price": 3.0}, {"beta_price": 5.0}]
    assert all("simulated_lift_pct" in row for row in rows)
    assert shopper.default_params() is shopper.default_params()  # cached

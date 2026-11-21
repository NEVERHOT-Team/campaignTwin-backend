import pandas as pd
import pytest

from data import loaders
from data.etl.backtest import (
    CONTROL_PRODUCT_TYPES,
    _control_changes,
    _markdown_split,
    _score,
    build_backtest,
)
from data.etl.build_tissue import _category_purchases, _elasticity
from data.etl.config import DEFAULTS


@pytest.fixture
def control_inputs():
    products = pd.DataFrame({"product_id": range(4), "product_type": CONTROL_PRODUCT_TYPES})
    rows = [
        (product_id, week, baseline if week < 45 else campaign)
        for product_id, (baseline, campaign) in enumerate(
            [(100, 110), (10, 12), (20, 14), (40, 40)]
        )
        for week in range(37, 53)
    ]
    # A purchase and its return cancel. Large sales just outside the windows have no effect.
    rows.extend([(3, 52, 5), (3, 52, -5), (3, 36, 999), (3, 53, 999)])
    return products, pd.DataFrame(rows, columns=["product_id", "week", "sales_value"])


def test_campaign_coupon_on_unsold_control_product_disqualifies_category(control_inputs):
    products, tx = control_inputs
    products.loc[len(products)] = [77, CONTROL_PRODUCT_TYPES[0]]
    with pytest.raises(ValueError, match="Campaign 18 coupons overlap controls"):
        _control_changes(products, tx, pd.Series([77]))


def test_markdown_split_keeps_store_coupon_match_conditional_on_redemption():
    lines = pd.DataFrame({
        "sales_value": [5.0, 12.0],
        "retail_disc": [-1.0, 0.0],
        "coupon_match_disc": [-2.0, 0.0],
        "coupon_disc": [-3.0, 0.0],
    })
    split = _markdown_split(lines)
    assert split["regular_shelf_value_usd"] == 20
    assert split["shelf_depth_fraction"] == 1 / 20
    assert split["coupon_depth_fraction"] == (2 + 3) / 20
    assert split["combined_depth_fraction"] == 6 / 20


def test_category_sales_include_nonrecipients_and_returns_and_deduplicate_redemptions(
    control_inputs,
):
    rows = [
        {
            "week": w,
            "sales_value": 10 if w < 45 else 20,
            "coupon_disc": -1 if w == 37 else 0,
            "quantity": 1,
            "household_id": 99,
            "product_id": 100,
            "transaction_timestamp": pd.Timestamp("2017-09-04") + pd.Timedelta(weeks=w - 37),
        }
        for w in range(37, 53)
    ]
    offer = {**rows[8], "coupon_disc": -1, "sales_value": 3, "household_id": 7}
    refund = {**rows[8], "quantity": -1, "sales_value": -6}
    tx = pd.DataFrame([*rows, offer, offer, refund]).assign(retail_disc=0, coupon_match_disc=0)
    coupons = pd.DataFrame({"campaign_id": [13, 18, 18], "product_id": 100, "coupon_upc": 123})
    redemptions = pd.DataFrame(
        {
            "campaign_id": [13, 18, 18],
            "household_id": [99, 7, 7],
            "coupon_upc": 123,
            "redemption_date": [
                rows[0]["transaction_timestamp"],
                offer["transaction_timestamp"],
                offer["transaction_timestamp"],
            ],
        }
    )
    controls = _control_changes(*control_inputs, pd.Series([100]))
    assert [control["change_pct"] for control in controls] == pytest.approx([10, 20, -30, 0])
    backtest = _score(tx, coupons, redemptions, pd.Index([7, 8]), pd.Index(range(1, 45)), controls)
    assert backtest["treated_households"] == 2
    assert backtest["treated_share"] == 0  # neither assignee bought tissue before week 45
    assert backtest["mechanics"]["all_tissue_purchases"]["purchase_lines"] == 10
    assert backtest["mechanics"]["matched_campaign_redemptions"]["purchase_lines"] == 2
    assert backtest["observed_redemption_rate"] == 0.5
    assert backtest["actual"] == {
        "lift_pct_raw": 100.0,
        "control_change_pct": 0.0,
        "control_change_pct_median": 5.0,
        "lift_pct": 100.0,
        "lift_pct_median": 95.0,
        "lift_range_pct": [95.0, 100.0],
        "incremental_sales": 80 * DEFAULTS["usd_thb"],
        "redemption_rate": 0.5,
    }
    assert backtest["discount_depth"] == pytest.approx(1 / 3, abs=1e-6)
    assert "redemptions were 1 versus 1" in backtest["description"]
    with pytest.raises(ValueError, match="strictly before"):
        _score(tx, coupons, redemptions, pd.Index([7, 8]), pd.Index([44, 45]), controls)


def test_published_actuals_and_training_window():
    backtest = loaders.load_backtest()
    assert (backtest["start_week"], backtest["end_week"]) == (45, 52)
    assert backtest["calibration_weeks"] == list(range(1, 45))
    actual = backtest["actual"]
    assert actual["lift_pct_raw"] == round(100 * (5669.07 / 5221.50 - 1), 2) == 8.57
    controls = backtest["seasonality_control"]["categories"]
    assert {item["product_type"] for item in controls} == set(CONTROL_PRODUCT_TYPES)
    changes = [
        100 * (item["campaign_sales_usd"] / item["baseline_sales_usd"] - 1) for item in controls
    ]
    assert [item["change_pct"] for item in controls] == pytest.approx(changes, abs=1e-6)
    control_mean = sum(changes) / len(changes)
    assert actual["control_change_pct"] == round(control_mean, 2) == 6.86
    assert actual["lift_pct"] == round(100 * (5669.07 / 5221.50 - 1) - control_mean, 2) == 1.71
    control_median = sum(sorted(changes)[1:3]) / 2
    assert actual["control_change_pct_median"] == round(control_median, 2) == 2.33
    assert (
        actual["lift_pct_median"]
        == round(100 * (5669.07 / 5221.50 - 1) - control_median, 2)
        == 6.24
    )
    assert actual["lift_range_pct"] == [1.71, 6.24]
    assert backtest["actual"]["incremental_sales"] == round(
        (5669.07 - 5221.50) * DEFAULTS["usd_thb"], 2
    )
    assert backtest["actual"]["redemption_rate"] == pytest.approx(8 / 1133)


def test_published_mechanics_use_separate_assignment_and_redemption_denominators():
    case = loaders.load_backtest()
    mechanics = case["mechanics"]
    assert mechanics["treated_share_basis"] == {
        "category_buying_households": 1417,
        "assigned_category_buying_households": 885,
        "weeks": [1, 44],
    }
    assert case["treated_share"] == pytest.approx(885 / 1417)
    assert case["shelf_reach"] == 1.0
    assert mechanics["redemption_rate_basis"] == {
        "assigned_households": 1133,
        "assigned_households_with_matched_redemption": 8,
    }
    assert case["observed_redemption_rate"] == case["actual"]["redemption_rate"]
    assert case["observed_redemption_rate_note"] == "use to validate, not as reach input"
    broad = mechanics["all_tissue_purchases"]
    assert broad["purchase_lines"] == 1185
    assert broad["shelf_depth_fraction"] == pytest.approx(1055.31 / 6731.58)
    matched = mechanics["matched_campaign_redemptions"]
    assert matched["purchase_lines"] == 8
    assert matched["shelf_depth_fraction"] == pytest.approx(14 / 71.92)
    assert matched["coupon_depth_fraction"] == pytest.approx(8 / 71.92)
    assert matched["combined_depth_fraction"] == pytest.approx(22 / 71.92)
    assert case["discount_depth"] == 0.222469


@pytest.mark.skipif(
    not (loaders.CALIB / "household_behavior.parquet").exists(), reason="local calibration required"
)
def test_built_panel_contains_only_the_declared_training_weeks():
    panel = loaders.load_household_behavior()
    weeks = loaders.load_backtest()["calibration_weeks"]
    assert set(panel["week"]) == set(weeks)
    assert len(panel) == loaders.load_segments()["households"].sum() * len(weeks)


@pytest.mark.skipif(
    not (loaders.RAW / "transactions.parquet").exists(), reason="local raw data required"
)
def test_raw_holdout_independent_of_fitting():
    lines, visits, weeks = _category_purchases()
    assert set(lines["week"]) == set(visits.index.get_level_values("week")) == set(range(1, 45))
    assert build_backtest(weeks) == loaders.load_backtest()
    observed = loaders.load_market()["source_notes"]["pooled_price_elasticity_observed"]["value"]
    assert observed == pytest.approx(_elasticity(lines, visits), abs=1e-6)
    schedule = loaders.load_raw_table(
        "campaign_descriptions", filters=[("campaign_id", "==", 18)]
    ).iloc[0]
    assert (schedule.start_date, schedule.end_date) == (
        pd.Timestamp("2017-10-30"),
        pd.Timestamp("2017-12-24"),
    )

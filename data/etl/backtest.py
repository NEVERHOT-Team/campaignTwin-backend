"""Score the fixed toilet-tissue campaign against the preceding eight weeks."""

import pandas as pd

from data.etl.config import BACKTEST_CAMPAIGN_ID, BACKTEST_END_WEEK, BACKTEST_START_WEEK, DEFAULTS
from data.loaders import load_category_products, load_raw_table

BASELINE_WEEKS = 8
EVENT_KEY = ["campaign_id", "coupon_upc", "household_id", "redemption_date"]


def _score(
    transactions: pd.DataFrame,
    coupons: pd.DataFrame,
    redemptions: pd.DataFrame,
    recipients: pd.Index,
    calibration_weeks: pd.Index,
) -> dict:
    before = range(BACKTEST_START_WEEK - BASELINE_WEEKS, BACKTEST_START_WEEK)
    during = range(BACKTEST_START_WEEK, BACKTEST_END_WEEK + 1)
    if len(calibration_weeks) == 0 or calibration_weeks.max() >= BACKTEST_START_WEEK:
        raise ValueError("Backtest requires calibration weeks strictly before campaign 18")
    if not set(before).union(during).issubset(transactions["week"].unique()):
        raise ValueError("Incomplete toilet-tissue sales coverage for the backtest")
    sales = transactions.groupby("week")["sales_value"].sum()
    baseline_total = float(sales.loc[list(before)].sum())
    expected_sales = baseline_total * len(during) / len(before)
    actual_sales = float(sales.loc[list(during)].sum())
    if expected_sales <= 0 or recipients.empty:
        raise ValueError("Backtest requires positive baseline sales and assigned households")

    discounted = (
        transactions.loc[
            transactions["quantity"].gt(0)
            & transactions[["coupon_disc", "coupon_match_disc"]].ne(0).any(axis=1)
        ]
        .reset_index(drop=True)
        .rename_axis("line_id")
        .reset_index()
    )
    discounted["redemption_date"] = discounted["transaction_timestamp"].dt.normalize()
    matches = (
        redemptions[EVENT_KEY]
        .drop_duplicates()
        .merge(
            coupons[["campaign_id", "coupon_upc", "product_id"]].drop_duplicates(),
            on=["campaign_id", "coupon_upc"],
        )
        .merge(discounted, on=["household_id", "product_id", "redemption_date"])
        .sort_values("transaction_timestamp", kind="stable")
    )
    events = matches.drop_duplicates(EVENT_KEY)
    pre_count = int(events["week"].isin(before).sum())
    campaign_count = int(events["week"].isin(during).sum())
    change = campaign_count - pre_count
    redemption_change = f"{change:+d}"
    if pre_count:
        redemption_change += f", {100 * change / pre_count:+.1f}%"
    offer = matches.loc[
        matches["campaign_id"].eq(BACKTEST_CAMPAIGN_ID)
        & matches["week"].isin(during)
        & matches["household_id"].isin(recipients)
    ]
    offer_lines = offer.drop_duplicates("line_id")
    if offer_lines.empty:
        raise ValueError(
            "No campaign 18 coupon redemptions matched eligible toilet-tissue purchases"
        )
    shelf = (
        offer_lines["sales_value"]
        + offer_lines["retail_disc"].abs()
        + offer_lines["coupon_match_disc"].abs()
    )
    paid = (offer_lines["sales_value"] - offer_lines["coupon_disc"].abs()).clip(lower=0)
    depth = (1 - paid / shelf.where(shelf.gt(0))).median()
    if pd.isna(depth) or not 0 <= depth <= 1:
        raise ValueError("Invalid observed campaign discount")

    return {
        "campaign_id": str(BACKTEST_CAMPAIGN_ID),
        "description": (
            f"Complete Journey toilet-tissue coupon campaign {BACKTEST_CAMPAIGN_ID}. "
            f"All-household category sales were USD {actual_sales:.2f} in weeks "
            f"{during.start}–{during.stop - 1} versus USD {baseline_total:.2f} in weeks "
            f"{before.start}–{before.stop - 1}. Matched category redemptions were "
            f"{campaign_count} versus {pre_count} ({redemption_change})."
        ),
        "start_week": BACKTEST_START_WEEK,
        "end_week": BACKTEST_END_WEEK,
        "discount_depth": round(float(depth), 6),
        "target_segments": [],  # The campaign assignments contain no archetype targeting rule.
        "treated_households": len(recipients),
        "actual": {
            "lift_pct": round(100 * (actual_sales - expected_sales) / expected_sales, 6),
            "redemption_rate": round(offer["household_id"].nunique() / len(recipients), 8),
            "incremental_sales": round((actual_sales - expected_sales) * DEFAULTS["usd_thb"], 2),
        },
        "calibration_weeks": calibration_weeks.tolist(),
        "source": "data",
    }


def build_backtest(calibration_weeks: pd.Index) -> dict:
    products = load_category_products()["product_id"].tolist()
    transactions = load_raw_table(
        "transactions",
        filters=[
            ("product_id", "in", products),
            ("week", ">=", BACKTEST_START_WEEK - BASELINE_WEEKS),
            ("week", "<=", BACKTEST_END_WEEK),
        ],
    )
    coupons = load_raw_table("coupons", filters=[("product_id", "in", products)])
    redemptions = load_raw_table(
        "coupon_redemptions",
        filters=[("coupon_upc", "in", coupons["coupon_upc"].unique().tolist())],
    )
    assigned = load_raw_table("campaigns", filters=[("campaign_id", "==", BACKTEST_CAMPAIGN_ID)])
    return _score(
        transactions,
        coupons,
        redemptions,
        pd.Index(assigned["household_id"].unique()),
        calibration_weeks,
    )

"""Score the fixed toilet-tissue campaign against the preceding eight weeks."""

from statistics import mean, median

import pandas as pd

from data.etl.config import BACKTEST_CAMPAIGN_ID, BACKTEST_END_WEEK, BACKTEST_START_WEEK, DEFAULTS
from data.loaders import load_category_products, load_raw_table

BASELINE_WEEKS = 8
BASELINE = range(BACKTEST_START_WEEK - BASELINE_WEEKS, BACKTEST_START_WEEK)
CAMPAIGN = range(BACKTEST_START_WEEK, BACKTEST_END_WEEK + 1)
CONTROL_PRODUCT_TYPES = (
    "FACIAL TISSUE & PAPER HANDKE",
    "PAPER NAPKINS",
    "CLEANSING WIPES",
    "BASIC PAPER PLATES",
)
EVENT_KEY = ["campaign_id", "coupon_upc", "household_id", "redemption_date"]


def _period_sales(lines: pd.DataFrame, category: str) -> tuple[float, float, float]:
    sales = lines.groupby("week")["sales_value"].sum()
    if not set(BASELINE).union(CAMPAIGN).issubset(sales.index):
        raise ValueError(f"Incomplete sales coverage for {category}")
    baseline = float(sales.loc[list(BASELINE)].sum())
    during = float(sales.loc[list(CAMPAIGN)].sum())
    if baseline <= 0:
        raise ValueError(f"Backtest requires positive baseline sales for {category}")
    return baseline, during, 100 * (during - baseline) / baseline


def _control_changes(
    products: pd.DataFrame, transactions: pd.DataFrame, campaign_coupon_products: pd.Series
) -> list[dict]:
    # Screen the whole category catalog, including products with no recorded sales.
    offered = products.loc[products["product_id"].isin(campaign_coupon_products), "product_type"]
    if not offered.empty:
        raise ValueError(
            f"Campaign {BACKTEST_CAMPAIGN_ID} coupons overlap controls: {sorted(set(offered))}"
        )
    sales = transactions[["product_id", "week", "sales_value"]].merge(
        products, on="product_id", validate="many_to_one"
    )
    controls = []
    for category in CONTROL_PRODUCT_TYPES:
        baseline, during, change = _period_sales(
            sales.loc[sales["product_type"].eq(category)], category
        )
        controls.append(
            {
                "product_type": category,
                "baseline_sales_usd": round(baseline, 2),
                "campaign_sales_usd": round(during, 2),
                "change_pct": change,
            }
        )
    return controls


def _score(
    transactions: pd.DataFrame,
    coupons: pd.DataFrame,
    redemptions: pd.DataFrame,
    recipients: pd.Index,
    calibration_weeks: pd.Index,
    controls: list[dict],
) -> dict:
    if len(calibration_weeks) == 0 or calibration_weeks.max() >= BACKTEST_START_WEEK:
        raise ValueError("Backtest requires calibration weeks strictly before campaign 18")
    baseline_total, actual_sales, raw_change = _period_sales(transactions, "toilet tissue")
    changes = [item["change_pct"] for item in controls]
    control_change = mean(changes)
    control_median = median(changes)
    lift = round(raw_change - control_change, 2)
    lift_median = round(raw_change - control_median, 2)
    if recipients.empty:
        raise ValueError("Backtest requires assigned households")

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
    pre_count = int(events["week"].isin(BASELINE).sum())
    campaign_count = int(events["week"].isin(CAMPAIGN).sum())
    change = campaign_count - pre_count
    redemption_change = f"{change:+d}"
    if pre_count:
        redemption_change += f", {100 * change / pre_count:+.1f}%"
    offer = matches.loc[
        matches["campaign_id"].eq(BACKTEST_CAMPAIGN_ID)
        & matches["week"].isin(CAMPAIGN)
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
            f"{CAMPAIGN.start}–{CAMPAIGN.stop - 1} versus USD {baseline_total:.2f} in weeks "
            f"{BASELINE.start}–{BASELINE.stop - 1}. Matched category redemptions were "
            f"{campaign_count} versus {pre_count} ({redemption_change})."
        ),
        "start_week": BACKTEST_START_WEEK,
        "end_week": BACKTEST_END_WEEK,
        "discount_depth": round(float(depth), 6),
        "target_segments": [],  # The campaign assignments contain no archetype targeting rule.
        "treated_households": len(recipients),
        "actual": {
            "lift_pct": lift,
            "lift_pct_raw": round(raw_change, 2),
            "control_change_pct": round(control_change, 2),
            "control_change_pct_median": round(control_median, 2),
            "lift_pct_median": lift_median,
            "lift_range_pct": sorted([lift, lift_median]),
            "redemption_rate": round(offer["household_id"].nunique() / len(recipients), 8),
            "incremental_sales": round((actual_sales - baseline_total) * DEFAULTS["usd_thb"], 2),
        },
        "seasonality_control": {
            "method": (
                "Difference-in-differences in percentage points. Subtract the unweighted mean "
                "control-category sales change from the toilet-tissue sales change. Each change "
                "compares total retailer sales including returns in the same eight-week windows. "
                "The reported range spans the mean-adjusted and median-adjusted lifts."
            ),
            "selection": (
                "Household paper and hygiene product types in GROCERY with zero "
                "campaign-18 coupon-eligible products."
            ),
            "categories": [
                {**item, "change_pct": round(item["change_pct"], 6)} for item in controls
            ],
        },
        "calibration_weeks": calibration_weeks.tolist(),
        "source": "data",
    }


def build_backtest(calibration_weeks: pd.Index) -> dict:
    products = load_category_products()["product_id"].tolist()
    control_products = load_raw_table(
        "products",
        columns=["product_id", "product_type"],
        filters=[
            ("department", "==", "GROCERY"),
            ("product_type", "in", list(CONTROL_PRODUCT_TYPES)),
        ],
    )
    transactions = load_raw_table(
        "transactions",
        filters=[
            ("product_id", "in", products + control_products["product_id"].tolist()),
            ("week", ">=", BASELINE.start),
            ("week", "<=", BACKTEST_END_WEEK),
        ],
    )
    coupons = load_raw_table("coupons", filters=[("product_id", "in", products)])
    redemptions = load_raw_table(
        "coupon_redemptions",
        filters=[("coupon_upc", "in", coupons["coupon_upc"].unique().tolist())],
    )
    assigned = load_raw_table("campaigns", filters=[("campaign_id", "==", BACKTEST_CAMPAIGN_ID)])
    campaign_coupons = load_raw_table(
        "coupons", columns=["product_id"], filters=[("campaign_id", "==", BACKTEST_CAMPAIGN_ID)]
    )
    return _score(
        transactions.loc[transactions["product_id"].isin(products)],
        coupons,
        redemptions,
        pd.Index(assigned["household_id"].unique()),
        calibration_weeks,
        _control_changes(control_products, transactions, campaign_coupons["product_id"]),
    )

"""Score the fixed toilet-tissue campaign against the preceding eight weeks."""

from statistics import mean, median

import pandas as pd

from data.etl.config import (
    BACKTEST_CAMPAIGN_ID,
    BACKTEST_END_WEEK,
    BACKTEST_START_WEEK,
    DEFAULTS,
    PRODUCT_PATH,
)
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


def _period_sales(
    sales: pd.Series, category: str, pre: range = BASELINE, active: range = CAMPAIGN
) -> tuple[float, float, float]:
    if not set(pre).union(active).issubset(sales.dropna().index):
        raise ValueError(f"Incomplete sales coverage for {category}")
    baseline = float(sales.loc[list(pre)].sum())
    during = float(sales.loc[list(active)].sum())
    if baseline <= 0:
        raise ValueError(f"Backtest requires positive baseline sales for {category}")
    return baseline, during, 100 * ((during / len(active)) / (baseline / len(pre)) - 1)


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
            sales.loc[sales["product_type"].eq(category)].groupby("week")["sales_value"].sum(),
            category,
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


def _historical_benchmark(
    weekly: pd.DataFrame,
    schedule: pd.DataFrame,
    offered: pd.DataFrame,
    week_one: pd.Timestamp,
    first_full_week: int,
) -> tuple[float, dict]:
    """Score other coupon campaigns using only complete pre-holdout weeks."""
    included = []
    excluded = []
    lifts = []
    tissue = PRODUCT_PATH[-1]
    weekly = weekly.reindex(columns=[tissue, *CONTROL_PRODUCT_TYPES])
    tissue_offers = offered.loc[offered["product_type"].eq(tissue), "campaign_id"]
    candidates = schedule.loc[schedule["campaign_id"].isin(tissue_offers)].sort_values("start_date")
    for campaign in candidates.itertuples():
        start = (campaign.start_date - week_one).days
        end = (campaign.end_date - week_one).days
        launch_week = start // 7 + 1
        if campaign.campaign_id == BACKTEST_CAMPAIGN_ID or end // 7 + 1 >= BACKTEST_START_WEEK:
            continue
        pre = range(launch_week - BASELINE_WEEKS, launch_week)
        # Round the start forward to Monday and the end back to Sunday.
        active = range((start + 6) // 7 + 1, (end + 1) // 7 + 1)
        promoted = set(offered.loc[offered["campaign_id"].eq(campaign.campaign_id), "product_type"])
        controls = [name for name in CONTROL_PRODUCT_TYPES if name not in promoted]
        if pre.start < first_full_week:
            reason = "incomplete baseline"
        elif not active:
            reason = "no full campaign week"
        elif not controls:
            reason = "no coupon-free control"
        else:
            reason = None
        if reason:
            excluded.append({"campaign_id": str(campaign.campaign_id), "reason": reason})
            continue
        try:
            _, _, raw = _period_sales(weekly[tissue], tissue, pre, active)
            changes = {name: _period_sales(weekly[name], name, pre, active)[2] for name in controls}
        except ValueError as error:
            excluded.append({"campaign_id": str(campaign.campaign_id), "reason": str(error)})
            continue
        lift = raw - mean(changes.values())
        lifts.append(lift)
        included.append(
            {
                "campaign_id": str(campaign.campaign_id),
                "baseline_weeks": [pre.start, pre.stop - 1],
                "campaign_weeks": [active.start, active.stop - 1],
                "tissue_change_pct": round(raw, 6),
                "control_changes_pct": {name: round(change, 6) for name, change in changes.items()},
                "lift_pct": round(lift, 6),
            }
        )
    naive = round(mean(lifts), 2) if lifts else 0.0
    return naive, {
        "method": (
            "Equal-weight mean of other toilet-tissue coupon campaign lifts wholly in weeks "
            "1–44, or zero if none qualify. For each, compare mean weekly net retailer sales in "
            "complete active weeks against the eight complete weeks before launch. Subtract "
            "the unweighted mean percentage change of the holdout controls without coupons "
            "in that campaign. Round only the final benchmark to two decimal places."
        ),
        "source": "data" if included else "default",
        "campaigns": included,
        "excluded": excluded,
    }


def _markdown_split(lines: pd.DataFrame) -> dict:
    """Value-weighted shopper markdowns against the same regular shelf-price basket."""
    shelf_cut = float(lines["retail_disc"].abs().sum())
    match = float(lines["coupon_match_disc"].abs().sum())
    manufacturer = float(lines["coupon_disc"].abs().sum())
    regular = float(lines["sales_value"].sum()) + shelf_cut + match
    if regular <= 0:
        raise ValueError("Campaign markdowns require positive aggregate shelf value")
    coupon_cut = manufacturer + match
    return {
        "purchase_lines": len(lines),
        "regular_shelf_value_usd": round(regular, 2),
        "shelf_markdown_usd": round(shelf_cut, 2),
        "manufacturer_coupon_usd": round(manufacturer, 2),
        "retailer_coupon_match_usd": round(match, 2),
        "shelf_depth_fraction": round(shelf_cut / regular, 8),
        "coupon_depth_fraction": round(coupon_cut / regular, 8),
        "combined_depth_fraction": round((shelf_cut + coupon_cut) / regular, 8),
    }


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
    baseline_total, actual_sales, raw_change = _period_sales(
        transactions.groupby("week")["sales_value"].sum(), "toilet tissue"
    )
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

    purchases = transactions.loc[transactions["quantity"].gt(0)]
    buyers = pd.Index(purchases.loc[
        purchases["week"].isin(calibration_weeks), "household_id"
    ].unique())
    if buyers.empty:
        raise ValueError("Campaign assignment share requires pre-launch tissue buyers")
    assigned_buyers = len(buyers.intersection(recipients))
    campaign_lines = purchases.loc[purchases["week"].isin(CAMPAIGN)]
    redeeming_households = offer["household_id"].nunique()
    redemption_rate = round(redeeming_households / len(recipients), 8)

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
        "mechanics": {
            "all_tissue_purchases": _markdown_split(campaign_lines),
            "matched_campaign_redemptions": _markdown_split(offer_lines),
            "depth_basis": (
                "Ratios of summed USD amounts on positive-quantity tissue purchase lines in "
                "weeks 45–52. Regular shelf value = sales_value + abs(retail_disc) + "
                "abs(coupon_match_disc). Shelf markdown = abs(retail_disc). Coupon markdown = "
                "abs(coupon_disc) paid by the manufacturer + abs(coupon_match_disc) paid by the "
                "retailer only upon coupon use. The two depths add on the same shelf denominator. "
                "Retailer sales_value includes manufacturer reimbursement."
            ),
            "scope": (
                "All tissue purchases include regular-price lines and overlapping promotions. "
                "Only matched lines join a campaign 18 redemption by an assigned household to "
                "a coupon-discounted eligible SKU purchased that day, counting each transaction "
                "line once. Matched depths describe realized redemptions. The existing "
                "discount_depth remains the median combined line depth on those purchases."
            ),
            "treated_share_basis": {
                "category_buying_households": len(buyers),
                "assigned_category_buying_households": assigned_buyers,
                "weeks": [int(min(calibration_weeks)), int(max(calibration_weeks))],
            },
            "redemption_rate_basis": {
                "assigned_households": len(recipients),
                "assigned_households_with_matched_redemption": redeeming_households,
            },
            "shelf_reach_basis": (
                "Default availability to all shoppers at a qualifying shelf. Actual store "
                "visits and shelf impressions are unobserved."
            ),
        },
        "treated_share": round(assigned_buyers / len(buyers), 8),
        "shelf_reach": 1.0,
        "observed_redemption_rate": redemption_rate,
        "observed_redemption_rate_note": "use to validate, not as reach input",
        "actual": {
            "lift_pct": lift,
            "lift_pct_raw": round(raw_change, 2),
            "control_change_pct": round(control_change, 2),
            "control_change_pct_median": round(control_median, 2),
            "lift_pct_median": lift_median,
            "lift_range_pct": sorted([lift, lift_median]),
            "redemption_rate": redemption_rate,
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
            ("week", ">=", 1),
            ("week", "<=", BACKTEST_END_WEEK),
        ],
    )
    coupons = load_raw_table("coupons", filters=[("product_id", "in", products)])
    redemptions = load_raw_table(
        "coupon_redemptions",
        filters=[("coupon_upc", "in", coupons["coupon_upc"].unique().tolist())],
    )
    assigned = load_raw_table("campaigns", filters=[("campaign_id", "==", BACKTEST_CAMPAIGN_ID)])
    catalog = pd.concat(
        [
            control_products,
            pd.DataFrame(
                {
                    "product_id": products,
                    "product_type": PRODUCT_PATH[-1],
                }
            ),
        ],
        ignore_index=True,
    )
    offered = load_raw_table(
        "coupons",
        columns=["campaign_id", "product_id"],
        filters=[("product_id", "in", catalog["product_id"].tolist())],
    ).merge(catalog, on="product_id", validate="many_to_one")
    campaign_coupons = offered.loc[offered["campaign_id"].eq(BACKTEST_CAMPAIGN_ID)]
    result = _score(
        transactions.loc[transactions["product_id"].isin(products)],
        coupons,
        redemptions,
        pd.Index(assigned["household_id"].unique()),
        calibration_weeks,
        _control_changes(control_products, transactions, campaign_coupons["product_id"]),
    )
    history = transactions.loc[transactions["week"].lt(BACKTEST_START_WEEK)]
    first_week = int(history["week"].min())
    first_day = (
        history.loc[history["week"].eq(first_week), "transaction_timestamp"].min().normalize()
    )
    first_monday = first_day.to_period("W-SUN").start_time
    week_one = first_monday - pd.Timedelta(weeks=first_week - 1)
    weekly = (
        history.merge(catalog, on="product_id", validate="many_to_one")
        .groupby(["week", "product_type"])["sales_value"]
        .sum()
        .unstack()
    )
    naive, method = _historical_benchmark(
        weekly,
        load_raw_table("campaign_descriptions"),
        offered,
        week_one,
        first_week + int(first_day != first_monday),
    )
    result["actual"]["naive_lift_pct"] = naive
    result["naive_benchmark"] = method
    return result

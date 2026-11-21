"""Rank Complete Journey sub-categories for the CampaignTwin demo."""

import pandas as pd

from data.loaders import load_raw_table

CATEGORY = ["department", "product_category", "product_type"]
DISCOUNTS = ["retail_disc", "coupon_disc", "coupon_match_disc"]
COUPON_KEYS = ["coupon_upc", "campaign_id"]
PURCHASE_KEYS = ["household_id", "product_id", "redemption_date"]
MIN_MANUFACTURER_SHARE = 0.05


def _count_redemptions(
    purchases: pd.DataFrame, coupons: pd.DataFrame, redemptions: pd.DataFrame
) -> tuple[pd.Series, int]:
    events = redemptions.drop_duplicates().reset_index(drop=True).reset_index(names="event_id")
    eligible = events.merge(coupons.drop_duplicates(), on=COUPON_KEYS)
    coupon_purchase = purchases[["coupon_disc", "coupon_match_disc"]].abs().gt(0).any(axis=1)
    discounted = purchases.loc[
        coupon_purchase, ["household_id", "product_id", "transaction_timestamp", *CATEGORY]
    ].copy()
    discounted["redemption_date"] = discounted.pop("transaction_timestamp").dt.normalize()
    matched = eligible.merge(discounted, on=PURCHASE_KEYS)
    by_category = matched.groupby(CATEGORY)["event_id"].nunique()
    return by_category, matched["event_id"].nunique()


def rank_categories(
    products: pd.DataFrame,
    transactions: pd.DataFrame,
    coupons: pd.DataFrame,
    redemptions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    products = products.dropna(subset=CATEGORY)
    products = products.loc[products["product_type"].str.strip().ne("")]
    purchases = transactions.loc[transactions["quantity"].gt(0)].merge(
        products, on="product_id", validate="many_to_one"
    )
    on_promo = purchases[DISCOUNTS].abs().gt(0).any(axis=1)
    purchases["promo_sales"] = purchases["sales_value"].where(on_promo, 0)
    stats = purchases.groupby(CATEGORY).agg(
        sales=("sales_value", "sum"),
        promo_sales=("promo_sales", "sum"),
        buying_households=("household_id", "nunique"),
    )
    stats = stats.loc[stats["sales"].gt(0)]
    stats["promo_sales_share"] = stats["promo_sales"] / stats["sales"]

    maker_sales = purchases.groupby([*CATEGORY, "manufacturer_id"])["sales_value"].sum()
    maker_shares = (
        maker_sales.rename("sales")
        .to_frame()
        .join(stats["sales"].rename("category_sales"), on=CATEGORY, how="inner")
    )
    maker_shares["share"] = maker_shares["sales"] / maker_shares["category_sales"]
    material = maker_shares.loc[maker_shares["share"].ge(MIN_MANUFACTURER_SHARE)]
    stats["material_manufacturers"] = material.groupby(level=CATEGORY).size()

    weekly = purchases[[*CATEGORY, "household_id", "week"]].drop_duplicates().sort_values("week")
    weekly["repeat_weeks"] = weekly.groupby([*CATEGORY, "household_id"])["week"].diff()
    stats["median_repeat_weeks"] = weekly.groupby(CATEGORY)["repeat_weeks"].median()
    stats["coupon_redemptions"], matched = _count_redemptions(purchases, coupons, redemptions)
    stats["coupon_redemptions"] = stats["coupon_redemptions"].fillna(0).astype(int)

    candidates = stats.loc[
        stats["material_manufacturers"].ge(3) & stats["median_repeat_weeks"].between(1, 6)
    ]
    ranked = candidates.sort_index().sort_values(
        ["coupon_redemptions", "promo_sales_share", "material_manufacturers"],
        ascending=False,
    )
    return ranked, material, matched


def main() -> None:
    products = load_raw_table("products", columns=["product_id", "manufacturer_id", *CATEGORY])
    transactions = load_raw_table(
        "transactions",
        columns=[
            "product_id",
            "household_id",
            "quantity",
            "sales_value",
            "week",
            "transaction_timestamp",
            *DISCOUNTS,
        ],
    )
    coupons = load_raw_table("coupons")
    redemptions = load_raw_table("coupon_redemptions")
    ranked, material, matched = rank_categories(products, transactions, coupons, redemptions)

    print(
        f"{len(ranked):,} sub-categories have ≥3 manufacturers at "
        f"≥{MIN_MANUFACTURER_SHARE:.0%} of paid sales and a median repeat gap of 1–6 weeks."
    )
    print(
        f"Coupon evidence: {matched:,}/{len(redemptions.drop_duplicates()):,} redemption events "
        "matched a same-day coupon-discounted eligible purchase in a known sub-category. "
        "Each event counts once per matched sub-category."
    )
    print("Promo share is the share of paid sales on lines with a retail or coupon discount.")
    print("Ranking uses coupon redemptions, then promo share, then material manufacturer count.")

    for index, (category, row) in enumerate(ranked.head(5).iterrows(), start=1):
        print(f"\n{index}. {' / '.join(category)}")
        print(
            f"   Buying households: {row.buying_households:,.0f} | "
            f"Material manufacturers: {row.material_manufacturers:.0f} | "
            f"Coupon redemptions: {row.coupon_redemptions:,.0f} | "
            f"Median repeat: {row.median_repeat_weeks:g} weeks | "
            f"Promo sales: {row.promo_sales_share:.1%}"
        )
        print(f"   {'Manufacturer ID':<18} {'Paid sales (USD)':>18} {'Category share':>16}")
        makers = material.loc[category].sort_values("share", ascending=False)
        for maker_id, maker in makers.iterrows():
            print(f"   {maker_id:<18} {maker.sales:>18,.2f} {maker.share:>15.1%}")
        remaining_share = max(0, 1 - makers["share"].sum())
        remaining_sales = max(0, row.sales - makers["sales"].sum())
        print(f"   {'All other':<18} {remaining_sales:>18,.2f} {remaining_share:>15.1%}")


if __name__ == "__main__":
    main()

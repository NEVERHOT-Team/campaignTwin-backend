"""Build the brand catalog and category reaction prior from Complete Journey."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from data.etl.build_tissue import _category_purchases, _household_features, _purchase_weeks
from data.etl.config import CATEGORY, CHANNELS, DEFAULTS, MANUFACTURERS, PRODUCT_PATH
from data.loaders import PROCESSED, load_raw_table

BRANDS = (751, 764, 595, 69, 1011, 1315)
DEFAULT_PROBABILITY = {"none": 0.45, "match": 0.30, "undercut": 0.15, "non_price": 0.10}
EVENT_COLUMNS = [
    "brand",
    "week",
    "trigger_depth",
    "rival",
    "reaction_type",
    "reaction_week",
    "reaction_depth",
    "lag_weeks",
    "duration_weeks",
]


def brand_catalog(lines: pd.DataFrame, households: pd.DataFrame) -> pd.DataFrame:
    lines = lines.join(households["archetype"], on="household_id")
    by_brand = lines.groupby("manufacturer_id")
    sales = by_brand["sales_value"].sum()
    buyers = lines.groupby(["manufacturer_id", "archetype"])["household_id"].nunique()
    prices = (by_brand["paid_total"].sum() / by_brand["quantity"].sum()).loc[list(BRANDS)]
    tiers = pd.qcut(prices.rank(method="first"), 3, labels=["low", "mid", "high"])
    rows = []
    for brand in BRANDS:
        promoted = lines.loc[lines["manufacturer_id"].eq(brand) & lines["discount_pct"].gt(0)]
        targets = buyers.loc[brand].sort_values(ascending=False, kind="stable").head(2)
        rows.append(
            {
                "brand": brand,
                "is_ours": MANUFACTURERS.get(brand) == "our_brand",
                "category": CATEGORY,
                "sub_category": PRODUCT_PATH[-1],
                "price_tier": tiers.loc[brand],
                "avg_price": round(prices.loc[brand], 4),
                "target_segments": "|".join(targets.index),
                "channels": "|".join(CHANNELS),
                "geo": "TH",
                "market_share": round(sales.loc[brand] / sales.sum(), 8),
                "typical_discount_pct": round(promoted["discount_pct"].median(), 4)
                if len(promoted)
                else 0,
                "margin_floor_pct": DEFAULTS["margin_floor_pct"],
            }
        )
    return pd.DataFrame(rows)


def weekly_signals(lines: pd.DataFrame, weeks: pd.Index) -> pd.DataFrame:
    keys = ["brand", "week"]
    products = load_raw_table("products")
    criteria = (
        products["department"].eq(PRODUCT_PATH[0])
        & products["product_category"].eq(PRODUCT_PATH[1])
        & products["product_type"].eq(PRODUCT_PATH[2])
        & products["manufacturer_id"].isin(BRANDS)
    )
    catalog = products.loc[criteria].set_index("product_id")["manufacturer_id"]
    promo = load_raw_table(
        "promotions",
        columns=["product_id", "week", "mailer_location", "display_location"],
        filters=[("product_id", "in", catalog.index.tolist())],
    )
    promo["brand"] = promo["product_id"].map(catalog)
    featured = promo[["mailer_location", "display_location"]].fillna("0").ne("0").any(axis=1)
    placements = promo.loc[featured].groupby(keys).size()
    lines = lines.loc[lines["manufacturer_id"].isin(BRANDS)].rename(
        columns={"manufacturer_id": "brand"}
    )
    unit_price = lines["paid_total"] / lines["quantity"]
    paid = lines.assign(unit_price=unit_price).groupby(keys)["unit_price"].median()
    grid = pd.MultiIndex.from_product([BRANDS, weeks], names=keys)
    weekly = paid.reindex(grid).to_frame()
    # The preceding eight calendar weeks provide a local typical paid price per brand.
    base = weekly.groupby(level="brand")["unit_price"].transform(
        lambda prices: prices.shift().rolling(8, min_periods=1).median()
    )
    weekly["depth"] = (1 - weekly["unit_price"] / base.where(base.gt(0))).clip(0, 1).round(8)
    weekly["featured"] = placements.reindex(grid, fill_value=0).gt(0)
    return weekly


def reaction_events(weekly: pd.DataFrame) -> pd.DataFrame:
    episodes = {}
    triggers = []
    for brand, rows in weekly.sort_index().groupby(level="brand"):
        rows = rows.droplevel("brand")
        cut = rows["depth"].ge(0.05)
        deep = rows["depth"].ge(0.10)
        featured = rows["featured"]
        for week in rows.index[deep | featured]:
            depth = rows.at[week, "depth"]
            triggers.append((brand, week, float(depth) if pd.notna(depth) else 0.0))

        actions = []
        for active, price_move in ((cut, True), (featured & ~cut, False)):
            # Measure a response from the matched week through the end of its run.
            active_rows = rows.loc[active]
            for _, run in active_rows.groupby((~active).cumsum()[active]):
                for position, (week, depth) in enumerate(run["depth"].items()):
                    actions.append(
                        (week, depth if price_move else 0.0, price_move, len(run) - position)
                    )
        episodes[brand] = actions

    outcomes = []
    for brand, week, depth in triggers:
        for rival, actions in episodes.items():
            if brand == rival:
                continue
            candidates = []
            for response_week, rival_depth, price_move, duration in actions:
                lag = response_week - week
                if abs(lag) > 3:
                    continue
                difference = round(rival_depth - depth, 8)
                if not price_move:
                    reaction = "non_price"
                elif difference > 0.05:
                    reaction = "undercut"
                elif difference >= -0.05:
                    reaction = "match"
                else:
                    continue
                candidates.append(
                    (
                        abs(lag),
                        lag < 0,
                        not price_move,
                        response_week,
                        reaction,
                        rival_depth,
                        lag,
                        duration,
                    )
                )
            if candidates:
                _, _, _, response_week, reaction, rival_depth, lag, duration = min(candidates)
            else:
                response_week, reaction, rival_depth, lag, duration = None, "none", 0.0, None, 0
            outcomes.append(
                (brand, week, depth, rival, reaction, response_week, rival_depth, lag, duration)
            )
    return (
        pd.DataFrame(outcomes, columns=EVENT_COLUMNS)
        .astype({"reaction_week": "Int64", "lag_weeks": "Int64"})
        .assign(source="data")
    )


def base_rate(events: pd.DataFrame) -> dict:
    count = len(events[["brand", "week"]].drop_duplicates())
    rates = events["reaction_type"].value_counts(normalize=True)
    weight = min(count / 20, 1)
    probabilities = {
        label: weight * rates.get(label, 0) + (1 - weight) * default
        for label, default in DEFAULT_PROBABILITY.items()
    }
    responses = events.loc[events["reaction_type"].ne("none")]
    priced = responses.loc[responses["reaction_type"].isin(["match", "undercut"]), "reaction_depth"]
    delay = responses["lag_weeks"].abs()
    return {
        **probabilities,
        "n_events": count,
        "n_comparisons": len(events),
        "depth_p50": float(priced.median()) if len(priced) else 0.25,
        "depth_p90": float(priced.quantile(0.9)) if len(priced) else 0.44,
        "delay_median": float(delay.median()) if len(delay) else 1,
        "delay_min": int(delay.min()) if len(delay) else 0,
        "duration_weeks": float(responses["duration_weeks"].median()) if len(responses) else 2,
        "source": "data" if count >= 20 else "default",
        "depth_source": "data" if len(priced) else "default",
        "timing_source": "data" if len(responses) else "default",
    }


def main() -> None:
    lines, visits, weeks = _category_purchases()
    households, _, _ = _household_features(lines, _purchase_weeks(lines), visits, weeks)
    brands = brand_catalog(lines, households)
    source = dict.fromkeys(brands.columns, "data")
    source.update(dict.fromkeys(["is_ours", "channels", "geo", "margin_floor_pct"], "default"))
    events = reaction_events(weekly_signals(lines, weeks))
    prior = base_rate(events)
    with TemporaryDirectory(dir=PROCESSED) as stage:
        root = Path(stage)
        brands.to_csv(root / "brands.csv", index=False)
        events.to_csv(root / "reaction_events.csv", index=False)
        for name, document in (
            (
                "brands_sources.json",
                {"fields": source, "usd_thb": {"value": DEFAULTS["usd_thb"], "source": "default"}},
            ),
            ("reaction_base_rate.json", {CATEGORY: prior}),
        ):
            (root / name).write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
        for path in root.iterdir():
            path.replace(PROCESSED / path.name)
    print(f"{len(brands)} brands; {prior['n_events']} triggers; {len(events)} rival comparisons")


if __name__ == "__main__":
    main()

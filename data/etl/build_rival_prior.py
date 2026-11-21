"""Build the brand catalog and category reaction prior from Complete Journey."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import NamedTuple

import numpy as np
import pandas as pd

from data.etl.build_tissue import _category_purchases, _household_features, _purchase_weeks
from data.etl.config import (
    CATEGORY,
    CHANNELS,
    DEFAULTS,
    MANUFACTURERS,
    PRODUCT_PATH,
    ROLE_PRICE_TIERS,
    SEED,
)
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
    "washout_clear",
    "background_week",
    "background_washout_clear",
    "background_reaction_type",
    "background_reaction_week",
]


def _price_tiers(lines: pd.DataFrame, sizes: pd.Series) -> pd.DataFrame:
    # Area is comparable across packs without assuming the size of a sheet or roll.
    area = sizes.astype("string").str.extract(
        r"^\s*(\d+(?:\.\d+)?)\s+S(?:Q)?\s+FT\s*$", expand=False
    )
    area = pd.to_numeric(area, errors="coerce")
    amounts = lines.loc[lines["manufacturer_id"].isin(BRANDS)].copy()
    amounts["area"] = amounts["product_id"].map(area) * amounts["quantity"]
    all_sales = amounts.groupby("manufacturer_id")["sales_value"].sum()
    all_units = amounts.groupby("manufacturer_id")["quantity"].sum()
    measured = amounts.loc[amounts["area"].gt(0)].groupby("manufacturer_id")
    price = (measured["paid_total"].sum() / measured["area"].sum()).sort_values(kind="stable")
    unique_prices = all_sales.loc[price.index].groupby(price).sum().sort_index()
    percentiles = (unique_prices.cumsum() - unique_prices / 2) / unique_prices.sum()
    bands = pd.cut(
        percentiles, [0, 1 / 3, 2 / 3, 1], labels=["low", "mid", "high"], include_lowest=True
    )
    table = pd.DataFrame(index=pd.Index(BRANDS, name="brand"))
    table["paid_thb_per_sq_ft"] = price
    table["unit_coverage"] = (
        measured["quantity"].sum().reindex(table.index, fill_value=0) / all_units
    )
    table["price_tier"] = [
        bands.loc[price.at[brand]]
        if brand in price.index
        else ROLE_PRICE_TIERS.get(MANUFACTURERS.get(brand), "mid")
        for brand in table.index
    ]
    table["source"] = np.where(table["paid_thb_per_sq_ft"].notna(), "data", "default")
    return table


def brand_catalog(
    lines: pd.DataFrame, households: pd.DataFrame, tiers: pd.DataFrame
) -> pd.DataFrame:
    lines = lines.join(households["archetype"], on="household_id")
    by_brand = lines.groupby("manufacturer_id")
    sales = by_brand["sales_value"].sum()
    buyers = lines.groupby(["manufacturer_id", "archetype"])["household_id"].nunique()
    category_buyers = lines.groupby("archetype")["household_id"].nunique()
    category_share = category_buyers / category_buyers.sum()
    prices = (by_brand["paid_total"].sum() / by_brand["quantity"].sum()).loc[list(BRANDS)]
    rows = []
    for brand in BRANDS:
        promoted = lines.loc[lines["manufacturer_id"].eq(brand) & lines["discount_pct"].gt(0)]
        brand_buyers = buyers.loc[brand]
        lift = (brand_buyers / brand_buyers.sum()) / category_share
        targets = lift.dropna().sort_values(ascending=False, kind="stable").head(2)
        rows.append(
            {
                "brand": brand,
                "is_ours": MANUFACTURERS.get(brand) == "our_brand",
                "category": CATEGORY,
                "sub_category": PRODUCT_PATH[-1],
                "price_tier": tiers.at[brand, "price_tier"],
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
    weekly["depth"] = (
        ((1 - weekly["unit_price"] / base.where(base.gt(0))) * 100).clip(0, 100).round(6)
    )
    weekly["featured"] = placements.reindex(grid, fill_value=0).gt(0)
    return weekly


def _response_signals(rows: pd.DataFrame) -> pd.DataFrame:
    depth = rows["depth"]
    cut = depth.ge(5)
    featured = rows["featured"]
    quiet = depth.lt(5) & ~featured
    non_price = featured & depth.eq(0)
    signals = pd.DataFrame(index=rows.index)
    signals["cut_start"] = cut & depth.shift().lt(5)
    signals["placement_start"] = non_price & ~featured.shift(fill_value=True)
    signals["washout_clear"] = quiet.shift(1, fill_value=False) & quiet.shift(2, fill_value=False)
    signals["depth"] = depth
    # At an onset the run length is the observed duration of the new action.
    signals["cut_duration"] = cut.groupby((~cut).cumsum()).transform("sum")
    signals["placement_duration"] = non_price.groupby((~non_price).cumsum()).transform("sum")
    return signals


class Response(NamedTuple):
    reaction: str
    week: int | None
    depth: float
    lag: int | None
    duration: int
    washout_clear: bool


def _response(actions: pd.DataFrame, week: int, trigger_depth: float) -> Response:
    clear = bool(actions.at[week, "washout_clear"])
    if clear:
        for lag in (1, 2, 3):
            action = actions.loc[week + lag]
            if action.cut_start:
                difference = round(action.depth - trigger_depth, 6)
                if difference >= -5:
                    label = "undercut" if difference > 5 else "match"
                    return Response(
                        label, week + lag, action.depth, lag, int(action.cut_duration), clear
                    )
            if action.placement_start:
                return Response(
                    "non_price", week + lag, 0.0, lag, int(action.placement_duration), clear
                )
    return Response("none", None, 0.0, None, 0, clear)


def reaction_events(weekly: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    brands = {
        brand: rows.droplevel("brand") for brand, rows in weekly.sort_index().groupby("brand")
    }
    signals = {brand: _response_signals(rows) for brand, rows in brands.items()}
    outcomes = []
    for brand, rows in brands.items():
        # Both arms need the full two-week lookback and three-week follow-up.
        anchors = rows.loc[rows.index.min() + 2 : rows.index.max() - 3]
        trigger = anchors["depth"].ge(10) | anchors["featured"]
        controls = anchors.index[~trigger & anchors["depth"].notna()]
        for week, raw_depth in anchors.loc[trigger, "depth"].items():
            depth = float(raw_depth) if pd.notna(raw_depth) else 0.0
            control = int(rng.choice(controls)) if len(controls) else None
            for rival, actions in signals.items():
                if brand == rival:
                    continue
                event_outcome = _response(actions, week, depth)
                # Reuse the event depth to classify a placebo cut on the same scale.
                placebo = _response(actions, control, depth) if control is not None else None
                outcomes.append(
                    (
                        brand,
                        week,
                        depth,
                        rival,
                        *event_outcome,
                        control,
                        placebo.washout_clear if placebo else None,
                        placebo.reaction if placebo else None,
                        placebo.week if placebo else None,
                    )
                )
    return (
        pd.DataFrame(outcomes, columns=EVENT_COLUMNS)
        .astype(
            dict.fromkeys(
                ["reaction_week", "lag_weeks", "background_week", "background_reaction_week"],
                "Int64",
            )
        )
        .assign(source="data")
    )


def _rates(labels: pd.Series) -> dict[str, float]:
    frequencies = labels.value_counts(normalize=True)
    return {label: float(frequencies.get(label, 0)) for label in DEFAULT_PROBABILITY}


def base_rate(events: pd.DataFrame) -> dict:
    # Compare the same focal brands, rivals and trigger depths in the two arms.
    matched = events.loc[events["background_week"].notna()]
    event_rates = _rates(matched["reaction_type"])
    background = _rates(matched["background_reaction_type"])
    lift = {r: max(0.0, event_rates[r] - background[r]) for r in ("match", "undercut", "non_price")}
    adjusted = {"none": max(0.0, 1 - sum(lift.values())), **lift}
    # Only distinct matched triggers with a rival at risk inform the evidence threshold.
    sampled_weeks = matched[["brand", "week"]].drop_duplicates()
    at_risk = matched.loc[matched["washout_clear"].eq(True)]
    count = len(at_risk[["brand", "week"]].drop_duplicates())
    blend = count < 20 or adjusted["none"] < 0.2
    probabilities = {
        label: (value + DEFAULT_PROBABILITY[label]) / 2 if blend else value
        for label, value in adjusted.items()
    }
    responses = matched.loc[matched["reaction_type"].ne("none")]
    priced = responses.loc[responses["reaction_type"].isin(["match", "undercut"]), "reaction_depth"]
    delay = responses["lag_weeks"]
    return {
        **probabilities,
        "event_rates": event_rates,
        "background_rates": background,
        "lift": lift,
        "n_events": count,
        "n_trigger_weeks": len(events[["brand", "week"]].drop_duplicates()),
        "n_matched_trigger_weeks": len(sampled_weeks),
        "n_background_weeks": len(matched[["brand", "background_week"]].drop_duplicates()),
        "n_comparisons": len(matched),
        "background_seed": SEED,
        "depth_p50": float(priced.median()) if len(priced) else 25.0,
        "depth_p90": float(priced.quantile(0.9)) if len(priced) else 44.0,
        "delay_median": float(delay.median()) if len(delay) else 1,
        "delay_min": int(delay.min()) if len(delay) else 1,
        "duration_weeks": float(responses["duration_weeks"].median()) if len(responses) else 2,
        "source": "data+default" if blend else "data",
        "depth_source": "data" if len(priced) else "default",
        "timing_source": "data" if len(responses) else "default",
    }


def main() -> None:
    lines, visits, weeks = _category_purchases()
    households, _, _ = _household_features(lines, _purchase_weeks(lines), visits, weeks)
    sizes = load_raw_table("products", columns=["product_id", "package_size"]).set_index(
        "product_id"
    )["package_size"]
    tiers = _price_tiers(lines, sizes)
    brands = brand_catalog(lines, households, tiers)
    source = dict.fromkeys(brands.columns, "data")
    source.update(dict.fromkeys(["is_ours", "channels", "geo", "margin_floor_pct"], "default"))
    source["price_tier"] = (
        tiers["source"].iloc[0] if tiers["source"].nunique() == 1 else tiers["source"].to_dict()
    )
    tier_evidence = json.loads(tiers.to_json(orient="index"))
    events = reaction_events(weekly_signals(lines, weeks))
    prior = base_rate(events)
    with TemporaryDirectory(dir=PROCESSED) as stage:
        root = Path(stage)
        brands.to_csv(root / "brands.csv", index=False)
        events.to_csv(root / "reaction_events.csv", index=False)
        for name, document in (
            (
                "brands_sources.json",
                {
                    "fields": source,
                    "price_tiers": tier_evidence,
                    "usd_thb": {"value": DEFAULTS["usd_thb"], "source": "default"},
                },
            ),
            ("reaction_base_rate.json", {CATEGORY: prior}),
        ):
            (root / name).write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
        for path in root.iterdir():
            path.replace(PROCESSED / path.name)
    print(
        f"{len(brands)} brands, {prior['n_events']} informative triggers, "
        f"{prior['n_comparisons']} matched comparisons"
    )


if __name__ == "__main__":
    main()

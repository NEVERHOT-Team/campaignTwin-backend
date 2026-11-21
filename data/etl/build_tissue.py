"""Build CampaignTwin's toilet tissue inputs from the local Complete Journey tables."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from data.etl.backtest import build_backtest
from data.etl.config import (
    BACKTEST_START_WEEK,
    CATEGORY,
    CHANNELS,
    CURRENCY,
    DEFAULTS,
    DISCOUNT_CAP_PCT,
    ELASTICITY_PRIOR_LEAST_PROMO,
    ELASTICITY_PRIOR_MOST_PROMO,
    INCOME_ELASTICITY_FACTORS,
    INCOME_ELASTICITY_NOTE,
    MANUFACTURERS,
    MIN_SEGMENT_HOUSEHOLDS,
    PANEL_WEEKS,
    SEED,
)
from data.etl.personas import build_personas, sample_households
from data.etl.selection_bias import calibrate_selection
from data.etl.ssr_personas import build_ssr_personas
from data.loaders import CALIB, DATA_DIR, PROCESSED, load_category_products, load_raw_table

PROMOTION_KEYS = ["product_id", "store_id", "week"]


def _category_purchases() -> tuple[pd.DataFrame, pd.Series, pd.Index]:
    products = load_category_products()[["product_id", "manufacturer_id", "brand"]].rename(
        columns={"brand": "label_type"}
    )

    transactions = load_raw_table("transactions", filters=[("week", "<", BACKTEST_START_WEEK)])
    visits = transactions.groupby(["store_id", "week"])["basket_id"].nunique()
    weeks = pd.Index(range(transactions["week"].min(), transactions["week"].max() + 1), name="week")
    purchases = transactions.loc[transactions["quantity"].gt(0)].merge(
        products, on="product_id", validate="many_to_one"
    )
    missing_makers = set(MANUFACTURERS).difference(purchases["manufacturer_id"])
    if missing_makers:
        raise ValueError(
            f"No toilet tissue purchases for configured manufacturer IDs {missing_makers}"
        )

    # Transaction sales include manufacturer reimbursements. Re-add retailer discounts
    # to recover shelf value and subtract the manufacturer coupon for shopper-paid value.
    coupon = purchases["coupon_disc"].abs()
    retail = purchases["retail_disc"].abs()
    match = purchases["coupon_match_disc"].abs()
    purchases["paid_total"] = (purchases["sales_value"] - coupon).clip(lower=0) * DEFAULTS[
        "usd_thb"
    ]
    purchases["shelf_total"] = (purchases["sales_value"] + retail + match) * DEFAULTS["usd_thb"]
    purchases["discount_pct"] = (
        100 * (purchases["shelf_total"] - purchases["paid_total"]) / purchases["shelf_total"]
    ).where(purchases["shelf_total"].gt(0))
    purchases["coupon_used"] = coupon.gt(0) | match.gt(0)
    purchases["on_promo"] = retail.gt(0) | purchases["coupon_used"]
    purchases["private_units"] = purchases["quantity"].where(
        purchases["label_type"].eq("Private"), 0
    )

    promotions = load_raw_table(
        "promotions",
        columns=[*PROMOTION_KEYS, "mailer_location", "display_location"],
        filters=[
            ("product_id", "in", products["product_id"].tolist()),
            ("week", "<", BACKTEST_START_WEEK),
        ],
    )
    promotions["in_mailer"] = promotions["mailer_location"].fillna("0").ne("0")
    promotions["on_display"] = promotions["display_location"].fillna("0").ne("0")
    promotion_flags = promotions.groupby(PROMOTION_KEYS)[["in_mailer", "on_display"]].any()
    purchases = purchases.join(promotion_flags, on=PROMOTION_KEYS, validate="many_to_one")
    for flag in ("in_mailer", "on_display"):
        purchases[flag] = purchases[flag].eq(True)
    return purchases, visits, weeks


def _purchase_weeks(purchases: pd.DataFrame) -> pd.DataFrame:
    keys = ["household_id", "week"]
    weekly = purchases.groupby(keys).agg(
        units=("quantity", "sum"),
        paid_total=("paid_total", "sum"),
        shelf_total=("shelf_total", "sum"),
        on_promo=("on_promo", "any"),
        coupon_used=("coupon_used", "any"),
        in_mailer=("in_mailer", "any"),
        on_display=("on_display", "any"),
    )
    brand_units = purchases.groupby([*keys, "manufacturer_id"], as_index=False)["quantity"].sum()
    winners = brand_units.sort_values(
        ["quantity", "manufacturer_id"], ascending=[False, True]
    ).drop_duplicates(keys)
    weekly["brand"] = winners.set_index(keys)["manufacturer_id"].astype("Int64")
    return weekly


def _classify(
    households: pd.DataFrame,
) -> tuple[np.ndarray, dict[int, str], dict[int, str], np.ndarray]:
    features = households[["recency", "frequency", "monetary", "coupon_rate", "promo_share"]].copy()
    features["frequency"] = np.log1p(features["frequency"])
    features["monetary"] = np.log1p(features["monetary"])
    scaled = StandardScaler().fit_transform(features)
    with threadpool_limits(limits=1):
        model = KMeans(n_clusters=5, random_state=SEED, n_init=20).fit(scaled)

    labels = model.labels_
    remaining = set(range(5))
    names = {}
    centers = model.cluster_centers_
    rules = {
        "deal_hunter": centers[:, 3] + centers[:, 4],
        "frequent_replenisher": centers[:, 1] - centers[:, 0],
        "high_spender": centers[:, 2],
        "occasional": centers[:, 0] - centers[:, 1],
        "steady_shopper": centers[:, 1],
    }
    for name, score in rules.items():
        cluster = max(remaining, key=lambda index: (score[index], -index))
        names[cluster] = name
        remaining.remove(cluster)

    counts = np.bincount(labels, minlength=5)
    eligible = np.flatnonzero(counts >= MIN_SEGMENT_HOUSEHOLDS)
    if not len(eligible):
        raise ValueError("There are not enough households to produce a segment of 50")
    segments = {}
    for label in names:
        distances = np.linalg.norm(centers[eligible] - centers[label], axis=1)
        segments[label] = names[eligible[distances.argmin()]]
    distances = np.linalg.norm(scaled - centers[labels], axis=1)
    return labels, names, segments, distances


def _elasticity(purchases: pd.DataFrame, visits: pd.Series) -> float:
    demand = purchases.groupby(PROMOTION_KEYS).agg(
        units=("quantity", "sum"), revenue=("paid_total", "sum")
    )
    demand = demand.loc[demand["revenue"].gt(0)].join(
        visits.rename("visits"), on=["store_id", "week"]
    )
    log_demand = pd.DataFrame(
        {
            "x": np.log(demand["revenue"] / demand["units"]),
            "y": np.log(demand["units"] / demand["visits"]),
        }
    )
    within = log_demand - log_demand.groupby(level=["product_id", "store_id"]).transform("mean")
    varying_price = within.loc[within["x"].abs().gt(1e-9)]
    if len(varying_price) < 20:
        return float("nan")
    return float(
        varying_price["x"].dot(varying_price["y"]) / varying_price["x"].dot(varying_price["x"])
    )


def _design_archetypes(centroids: pd.DataFrame) -> dict[str, str]:
    # Assign one cluster per role. Sorted cluster labels break ties reproducibly.
    scores = {
        "deal_hunter": centroids["promo_share"] + centroids["coupon_rate"],
        "loyal": centroids["brand_loyalty"] - centroids["promo_share"],
        "waiter": centroids["promo_share"].rank() + centroids["purchase_cycle_weeks"].rank(),
        "explorer": -centroids["brand_loyalty"],
    }
    remaining = centroids.index
    names = {}
    for name, score in scores.items():
        cluster = score.loc[remaining].idxmax()
        names[cluster] = name
        remaining = remaining.drop(cluster)
    names[remaining.item()] = "indifferent"
    return names


def _household_features(
    purchases: pd.DataFrame, weekly: pd.DataFrame, visits: pd.Series, weeks: pd.Index
) -> tuple[pd.DataFrame, dict[str, str], float]:
    by_household = purchases.groupby("household_id")
    households = by_household.agg(
        monetary=("paid_total", "sum"),
        units=("quantity", "sum"),
        baskets=("basket_id", "nunique"),
        private_units=("private_units", "sum"),
        last_week=("week", "max"),
    )
    purchase_weeks = weekly.reset_index()
    purchase_weeks["gap"] = purchase_weeks.groupby("household_id")["week"].diff()
    activity = purchase_weeks.groupby("household_id").agg(
        frequency=("week", "size"),
        median_gap=("gap", "median"),
        promo_share=("on_promo", "mean"),
        coupon_rate=("coupon_used", "mean"),
    )
    households = households.join(activity)
    households["recency"] = weeks.max() - households["last_week"]
    households["weekly_buy_rate"] = households["frequency"] / len(weeks)
    households["purchase_cycle_weeks"] = households["median_gap"].fillna(
        len(weeks) / households["frequency"]
    )
    households["avg_units"] = households["units"] / households["frequency"]
    households["avg_price_paid"] = households["monetary"] / households["units"]
    households["avg_basket_thb"] = households["monetary"] / households["baskets"]
    households["private_label_share"] = households["private_units"] / households["units"]
    top_brand_units = purchases.groupby(["household_id", "manufacturer_id"])["quantity"].sum()
    households["brand_loyalty"] = (
        top_brand_units.groupby(level="household_id").max() / households["units"]
    )

    labels, cluster_names, segments, distance = _classify(households)
    cluster_ids = pd.Series(labels, index=households.index)
    households["cluster_label"] = cluster_ids.map(cluster_names)
    centroids = households.groupby("cluster_label")[
        ["promo_share", "coupon_rate", "brand_loyalty", "purchase_cycle_weeks"]
    ].mean()
    design_names = _design_archetypes(centroids)
    households["archetype"] = households["cluster_label"].map(design_names)
    households["segment"] = cluster_ids.map(segments).map(design_names)
    households["distance_to_center"] = distance

    promo_score = centroids["promo_share"] + centroids["coupon_rate"]
    ranked = promo_score.sort_values(kind="stable").index
    priors = pd.Series(
        np.linspace(ELASTICITY_PRIOR_LEAST_PROMO, ELASTICITY_PRIOR_MOST_PROMO, len(ranked)).round(
            6
        ),
        index=ranked,
    )
    households["price_elasticity"] = households["cluster_label"].map(priors)
    pooled = _elasticity(purchases, visits)
    if not np.isfinite(pooled):
        raise ValueError("Category data did not support a pooled observed price elasticity")
    by_cluster = purchases.join(households["archetype"], on="household_id").groupby("archetype")
    estimates = {name: _elasticity(group, visits) for name, group in by_cluster}
    estimates = {name: fit if np.isfinite(fit) else pooled for name, fit in estimates.items()}
    households["price_elasticity_observed"] = households["archetype"].map(estimates)
    demographics = load_raw_table("demographics").set_index("household_id")
    return (
        households.join(demographics, validate="one_to_one"),
        {
            design_names[name]: design_names[segments[label]]
            for label, name in cluster_names.items()
        },
        pooled,
    )


def _segments(households: pd.DataFrame, personas: list[dict]) -> pd.DataFrame:
    grouped = households.groupby("segment").agg(
        households=("segment", "size"),
        weekly_purchase_prob=("weekly_buy_rate", "mean"),
        avg_basket_thb=("avg_basket_thb", "mean"),
        price_elasticity_observed=("price_elasticity_observed", "mean"),
        coupon_rate=("coupon_rate", "mean"),
        promo_response=("promo_share", "mean"),
        private_label_share=("private_label_share", "mean"),
    )
    grouped["share"] = grouped["households"] / len(households)
    grouped["source"] = "data"
    grouped["archetype"] = grouped.index
    profiles = pd.DataFrame(personas)
    weighted_priors = profiles["weight"] * profiles["price_elasticity"]
    segment_weights = profiles.groupby("segment")["weight"].sum()
    grouped["price_elasticity"] = (
        weighted_priors.groupby(profiles["segment"]).sum() / segment_weights
    ).round(6)
    representative = households.loc[households["archetype"].eq(households["segment"])]
    grouped["cluster_label"] = representative.groupby("segment")["cluster_label"].first()
    grouped["price_elasticity_source"] = "default"
    grouped["price_elasticity_observed_source"] = "data"
    grouped["name"] = grouped.index
    return grouped.rename_axis("segment_id").reset_index()[
        [
            "segment_id",
            "name",
            "archetype",
            "cluster_label",
            "households",
            "share",
            "weekly_purchase_prob",
            "avg_basket_thb",
            "price_elasticity",
            "price_elasticity_source",
            "price_elasticity_observed",
            "price_elasticity_observed_source",
            "coupon_rate",
            "promo_response",
            "private_label_share",
            "source",
        ]
    ]


def _market_inputs(
    purchases: pd.DataFrame,
    segments: pd.DataFrame,
    archetype_segments: dict[str, str],
    elasticity: float,
) -> tuple[dict, dict, dict]:
    source = dict.fromkeys(DEFAULTS, "default")
    company_id = next(maker_id for maker_id, role in MANUFACTURERS.items() if role == "our_brand")
    brand_sales = purchases.loc[
        purchases["manufacturer_id"].eq(company_id) & purchases["shelf_total"].gt(0)
    ]
    ref_price = round(float((brand_sales["shelf_total"] / brand_sales["quantity"]).median()), 2)
    market = {
        "category": CATEGORY,
        "currency": CURRENCY,
        "ref_price": ref_price,
        "unit_cost": round(ref_price * (1 - DEFAULTS["reference_margin_pct"] / 100), 2),
        "platform_fee_pct": DEFAULTS["platform_fee_pct"],
        "payment_fee_pct": DEFAULTS["payment_fee_pct"],
        "shipping_cost_per_order": DEFAULTS["shipping_cost_per_order"],
        "margin_floor_pct": DEFAULTS["margin_floor_pct"],
        "segments": segments[
            [
                "name",
                "archetype",
                "cluster_label",
                "share",
                "weekly_purchase_prob",
                "households",
                "price_elasticity",
                "price_elasticity_observed",
            ]
        ]
        .rename(columns={"weekly_purchase_prob": "weekly_buy_rate"})
        .to_dict("records"),
        "weeks": PANEL_WEEKS,
        "source_notes": {
            "fields": {
                "category": "data",
                "currency": "default",
                "ref_price": "data",
                "unit_cost": "default",
                "platform_fee_pct": "default",
                "payment_fee_pct": "default",
                "shipping_cost_per_order": "default",
                "margin_floor_pct": "default",
                "segments": "data",
                "segments.price_elasticity": "default",
                "segments.price_elasticity_observed": "data",
                "weeks": "default",
            },
            "usd_thb": {"value": DEFAULTS["usd_thb"], "source": "default"},
            "manufacturers": {
                str(maker_id): {"role": role, "source": "default"}
                for maker_id, role in MANUFACTURERS.items()
            },
            "archetype_segments": {"values": archetype_segments, "source": "data"},
            "income_elasticity_factors": {
                "value": INCOME_ELASTICITY_FACTORS,
                "source": "default",
                "note": INCOME_ELASTICITY_NOTE,
                "aggregation": "Segment priors are weighted means of adjusted personas.",
            },
            "pooled_price_elasticity": {
                "value": round((ELASTICITY_PRIOR_LEAST_PROMO + ELASTICITY_PRIOR_MOST_PROMO) / 2, 6),
                "source": "default",
                "note": "observational slope biased toward 0 by promo endogeneity",
                "scope": "Archetype base prior midpoint before income adjustment.",
            },
            "pooled_price_elasticity_observed": {"value": round(elasticity, 6), "source": "data"},
            "persona_fields": {
                "demographics": "data",
                "imputed_demographics": "default",
                "weight": "data",
                "channels": "default",
            },
            "dataset": "Complete Journey as packaged by completejourney_py (simulated)",
        },
    }
    economics = {
        "category": CATEGORY,
        "currency": CURRENCY,
        "usd_thb": DEFAULTS["usd_thb"],
        "price_thb": market["ref_price"],
        "margin_pct": DEFAULTS["reference_margin_pct"],
        "shopee_commission_pct": market["platform_fee_pct"],
        "shopee_txn_fee_pct": market["payment_fee_pct"],
        "media_cost_per_reach_thb": DEFAULTS["media_cost_per_reach_thb"],
        "shipping_cost_per_order": market["shipping_cost_per_order"],
        "unit_cost": market["unit_cost"],
        "margin_floor_pct": market["margin_floor_pct"],
        "collected_at": None,
        "source": {
            "category": "data",
            "currency": "default",
            "usd_thb": source["usd_thb"],
            "price_thb": "data",
            "margin_pct": source["reference_margin_pct"],
            "shopee_commission_pct": source["platform_fee_pct"],
            "shopee_txn_fee_pct": source["payment_fee_pct"],
            "media_cost_per_reach_thb": source["media_cost_per_reach_thb"],
            "shipping_cost_per_order": source["shipping_cost_per_order"],
            "unit_cost": source["reference_margin_pct"],
            "margin_floor_pct": source["margin_floor_pct"],
            "collected_at": "default",
        },
    }
    discounts = purchases.loc[purchases["discount_pct"].gt(0), "discount_pct"]
    quantiles = discounts.quantile([0, 0.5, 0.9, 1]).round(6)
    calendar_dates = {
        "11.11": ["2026-11-11"],
        "12.12": ["2026-12-12"],
        "payday": ["2026-11-25", "2026-12-25"],
    }
    knobs = {
        "discount_pct": {
            **dict(zip(("min", "p50", "p90", "raw_max"), quantiles, strict=True)),
            "max": DISCOUNT_CAP_PCT,
        },
        "duration_days": {"min": 1, "max": 28},
        "mechanics": ["percent_off", "voucher", "bundle", "free_shipping", "flash_sale"],
        "voucher_eligibility": ["all", "members", "new_only"],
        "cap_per_user": {"min": 1, "max": 4},
        "calendar": list(calendar_dates),
        "calendar_dates": calendar_dates,
        "channels": list(CHANNELS),
    }
    knobs["source"] = dict.fromkeys(knobs, "default") | {
        "discount_pct": "data",
        "discount_pct.raw_max": "data",
        "discount_pct.max": "default",
    }
    knobs["source_notes"] = {
        "discount_pct.max": "The 50% planning cap excludes fully discounted historical purchases.",
        "voucher_eligibility": "A new-buyer voucher uses type voucher and eligibility new_only.",
        "calendar_dates": "November and December 2026 demo dates assume payday on the 25th.",
    }
    return market, economics, knobs


def _calibration(households: pd.DataFrame, weekly: pd.DataFrame, weeks: pd.Index) -> pd.DataFrame:
    grid = pd.MultiIndex.from_product([households.index, weeks], names=["household_id", "week"])
    panel = weekly.reindex(grid).reset_index()
    # Mixed-brand weeks lose the runner-up's units in the weekly winner column.
    # Carry the purchase-line loyalty measure for exact household-level evaluation.
    panel = panel.join(households[["archetype", "segment", "brand_loyalty"]], on="household_id")
    local_keys = pd.Series(
        [f"tt-hh-{row:04d}" for row in range(1, len(households) + 1)],
        index=households.index,
    )
    panel["household_key"] = panel["household_id"].map(local_keys)
    panel["bought"] = panel["units"].notna()
    panel["price_paid"] = (panel["paid_total"] / panel["units"]).round(6)
    panel["shelf_price"] = (panel["shelf_total"] / panel["units"]).round(6)
    panel["discount_pct"] = (100 * (1 - panel["paid_total"] / panel["shelf_total"])).round(6)
    panel["units"] = panel["units"].fillna(0).astype(int)
    for column in ("on_promo", "coupon_used", "in_mailer", "on_display"):
        panel[column] = panel[column].eq(True)
    return panel[
        [
            "household_key",
            "archetype",
            "segment",
            "week",
            "bought",
            "units",
            "price_paid",
            "shelf_price",
            "on_promo",
            "discount_pct",
            "coupon_used",
            "in_mailer",
            "on_display",
            "brand",
            "brand_loyalty",
        ]
    ]


def _write_artifacts(
    json_records: dict[str, dict | list], segments: pd.DataFrame, panel: pd.DataFrame, bias: dict
) -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    CALIB.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=DATA_DIR) as temp:
        staging = Path(temp)
        for name, record in json_records.items():
            text = json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False)
            (staging / name).write_text(text + "\n", encoding="utf-8")
        segments.to_parquet(staging / "segments.parquet", index=False)
        panel.to_parquet(staging / "household_behavior.parquet", index=False)
        audit = staging / "selection_bias.json"
        audit.write_text(json.dumps(bias, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        for name in [*json_records, "segments.parquet"]:
            (staging / name).replace(PROCESSED / name)
        (staging / "household_behavior.parquet").replace(CALIB / "household_behavior.parquet")
        audit.replace(CALIB / audit.name)


def main() -> None:
    purchases, visits, weeks = _category_purchases()
    purchase_weeks = _purchase_weeks(purchases)
    households, archetype_segments, elasticity = _household_features(
        purchases, purchase_weeks, visits, weeks
    )
    sample, bias = calibrate_selection(households, sample_households(households))
    personas = build_personas(households, sample=sample)
    segments = _segments(households, personas)
    market, economics, knobs = _market_inputs(purchases, segments, archetype_segments, elasticity)
    panel = _calibration(households, purchase_weeks, weeks)
    _write_artifacts(
        {
            "personas.json": personas,
            "ssr_personas.json": build_ssr_personas(personas),
            "market.json": market,
            "knob_ranges.json": knobs,
            "unit_economics.json": economics,
            "backtest_campaign.json": build_backtest(weeks),
        },
        segments,
        panel,
        bias,
    )
    print(f"{CATEGORY}: {len(personas)} personas and {len(segments)} segments")
    print(f"Local calibration: {len(panel):,} household-weeks from {len(households):,} households")


if __name__ == "__main__":
    main()

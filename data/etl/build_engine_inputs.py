"""Publish the aggregate appeal fallback and assumptions used by the engine."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from data import loaders
from data.etl import config
from data.etl.appeal import build_appeal_fallback
from data.etl.build_rival_prior import (
    BLEND_DEFAULT_WEIGHT,
    BLEND_MIN_EVENTS,
    BLEND_MIN_NONE,
    DEFAULT_PROBABILITY,
)
from data.etl.personas import DEFAULT_TRAITS, TRAITS


def build_assumptions() -> list[dict]:
    market = loaders.load_market()
    economics = loaders.load_unit_econ()
    personas = loaders.load_personas()
    prior = loaders.load_rival_prior()
    rows = []

    def add(name, value, unit, source, *used_by, label="default"):
        rows.append(
            dict(
                name=name, value=value, unit=unit, label=label, source=source, used_by=list(used_by)
            )
        )

    settings = "data/etl/config.py"
    finance = {
        "usd_thb": ("THB/USD", "market.source_notes.usd_thb", "unit_economics.usd_thb"),
        "reference_margin_pct": ("percent", "unit_economics.margin_pct", "market.unit_cost"),
        "platform_fee_pct": (
            "percent",
            "market.platform_fee_pct",
            "unit_economics.shopee_commission_pct",
        ),
        "payment_fee_pct": (
            "percent",
            "market.payment_fee_pct",
            "unit_economics.shopee_txn_fee_pct",
        ),
        "margin_floor_pct": (
            "percent",
            "market.margin_floor_pct",
            "unit_economics.margin_floor_pct",
            "brands.margin_floor_pct",
        ),
        "shipping_cost_per_order": (
            "THB/order",
            "market.shipping_cost_per_order",
            "unit_economics.shipping_cost_per_order",
        ),
        "media_cost_per_reach_thb": ("THB/reach", "unit_economics.media_cost_per_reach_thb"),
    }
    for name, value in config.DEFAULTS.items():
        unit, *uses = finance[name]
        add(name, value, unit, f"{settings} DEFAULTS[{name}]", *uses)
    add(
        "currency",
        config.CURRENCY,
        "ISO 4217",
        settings,
        "market.currency",
        "unit_economics.currency",
    )
    add(
        "unit_cost",
        market["unit_cost"],
        "THB/purchased unit",
        "data/etl/build_tissue.py: ref_price × (1 − reference_margin_pct / 100), rounded to satang",
        "market.unit_cost",
        "unit_economics.unit_cost",
    )
    add(
        "collected_at",
        economics["collected_at"],
        "date",
        "data/etl/build_tissue.py: no seller collection date for placeholder fees",
        "unit_economics.collected_at",
    )
    add("panel_weeks", config.PANEL_WEEKS, "weeks", settings, "market.weeks")
    add(
        "channels",
        list(config.CHANNELS),
        "channel names",
        settings,
        "personas.channels",
        "market.source_notes.persona_fields.channels",
        "brands.channels",
        "knob_ranges.channels",
    )
    add(
        "manufacturers",
        {str(k): v for k, v in config.MANUFACTURERS.items()},
        "roles by ID",
        settings,
        "market.source_notes.manufacturers",
        "rival candidate selection",
    )
    add(
        "role_price_tiers",
        config.ROLE_PRICE_TIERS,
        "price tier by role",
        settings,
        "brands.price_tier when a comparable pack area is missing",
    )
    add(
        "seed",
        config.SEED,
        "random seed",
        settings,
        "KMeans",
        "persona sampling",
        "rival background-week sampling",
    )
    add(
        "discount_cap_pct",
        config.DISCOUNT_CAP_PCT,
        "percent",
        settings,
        "knob_ranges.discount_pct.max",
    )
    add(
        "min_segment_households",
        config.MIN_SEGMENT_HOUSEHOLDS,
        "households",
        settings,
        "released market and persona segments",
    )
    add(
        "category",
        config.CATEGORY,
        "category identifier",
        settings,
        "market.category",
        "unit_economics.category",
        label="data",
    )
    add(
        "product_path",
        list(config.PRODUCT_PATH),
        "catalog hierarchy",
        settings,
        "load_category_products",
        label="data",
    )
    add(
        "backtest_campaign_id",
        config.BACKTEST_CAMPAIGN_ID,
        "campaign ID",
        settings,
        "backtest_campaign.campaign_id",
        label="data",
    )
    add(
        "backtest_start_week",
        config.BACKTEST_START_WEEK,
        "retail week",
        settings,
        "backtest_campaign.start_week",
        "calibration cutoff",
        label="data",
    )
    add(
        "backtest_end_week",
        config.BACKTEST_END_WEEK,
        "retail week",
        settings,
        "backtest_campaign.end_week",
        label="data",
    )
    rationale = market["source_notes"]["pooled_price_elasticity"]["note"]
    elasticity_source = f"{settings} and data/etl/build_tissue.py: {rationale}"
    add(
        "elasticity_prior_least_promo",
        config.ELASTICITY_PRIOR_LEAST_PROMO,
        "elasticity",
        elasticity_source,
        "least promotion-driven cluster",
    )
    add(
        "elasticity_prior_most_promo",
        config.ELASTICITY_PRIOR_MOST_PROMO,
        "elasticity",
        elasticity_source,
        "most promotion-driven cluster",
    )
    add(
        "price_elasticity_by_archetype",
        {s["archetype"]: s["price_elasticity"] for s in market["segments"]},
        "elasticity",
        elasticity_source,
        "market.segments.price_elasticity",
        "personas.price_elasticity",
        "segments.price_elasticity",
    )
    add(
        "pooled_price_elasticity",
        market["source_notes"]["pooled_price_elasticity"]["value"],
        "elasticity",
        elasticity_source,
        "market.source_notes.pooled_price_elasticity",
    )

    rival_source = "data/etl/build_rival_prior.py: base_rate"
    add(
        "rival_default_probabilities",
        DEFAULT_PROBABILITY,
        "probability by reaction",
        rival_source,
        "reaction_base_rate",
    )
    add(
        "rival_blend_min_events",
        BLEND_MIN_EVENTS,
        "distinct at-risk events",
        rival_source,
        "reaction_base_rate",
    )
    add("rival_blend_min_none", BLEND_MIN_NONE, "probability", rival_source, "reaction_base_rate")
    add(
        "rival_blend_default_weight",
        BLEND_DEFAULT_WEIGHT,
        "fraction",
        f"{rival_source}: applied if event count or background-adjusted P(none) is below threshold",
        "reaction_base_rate",
    )
    add(
        "rival_blend_applied_weight",
        BLEND_DEFAULT_WEIGHT if prior["source"] == "data+default" else 0.0,
        "fraction",
        f"{rival_source}: published prior has {prior['n_events']} at-risk events",
        "load_rival_prior.p",
    )

    clusters = {p["archetype"]: p["cluster_label"] for p in personas}
    add(
        "twin_trait_defaults",
        {
            name: dict(zip(TRAITS, DEFAULT_TRAITS[cluster], strict=True))
            for name, cluster in sorted(clusters.items())
        },
        "0–1 by trait and archetype",
        "data/etl/personas.py DEFAULT_TRAITS by original cluster when Twin-2K-500 is absent",
        "personas.traits",
        "appeal_fallback",
    )
    add(
        "persona_trait_sources",
        sorted({p["source"]["traits"] for p in personas}),
        "source labels",
        "personas.source.traits; optional Twin proxies use QID32, QID250 and QID84",
        "personas.traits",
        "appeal_fallback",
        label="data",
    )
    add(
        "persona_backstories",
        "Short archetype templates",
        "text",
        "data/etl/personas.py BACKSTORIES, reserved for later LLM enrichment",
        "personas.backstory",
        label="llm_estimate",
    )
    add(
        "simulated_data_notice",
        "completejourney_py supplies simulated grocery shopping data for educational purposes.",
        "dataset provenance",
        "https://pypi.org/project/completejourney-py/#data-source-important-notice",
        "market",
        "unit_economics",
        "personas",
        "reaction_base_rate",
        "backtest_campaign",
        label="data",
    )
    return rows


def main() -> None:
    appeal = build_appeal_fallback(loaders.load_personas())
    assumptions = build_assumptions()
    with TemporaryDirectory(dir=loaders.PROCESSED) as directory:
        stage = Path(directory)
        appeal.to_csv(stage / "appeal_fallback.csv", index=False)
        (stage / "assumptions.json").write_text(
            json.dumps(assumptions, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        for name in ("appeal_fallback.csv", "assumptions.json"):
            (stage / name).replace(loaders.PROCESSED / name)


if __name__ == "__main__":
    main()

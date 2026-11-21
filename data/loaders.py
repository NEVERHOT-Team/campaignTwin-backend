"""Read local source tables and validate the engine inputs."""

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
PROCESSED = DATA_DIR / "processed"
RAW = DATA_DIR / "raw"
CALIB = DATA_DIR / "calib"
RAW_TABLES = (
    "transactions",
    "products",
    "demographics",
    "campaigns",
    "campaign_descriptions",
    "coupons",
    "coupon_redemptions",
    "promotions",
)


def load_raw_table(
    name: str,
    *,
    columns: list[str] | None = None,
    filters: list[tuple[str, str, Any]] | None = None,
) -> pd.DataFrame:
    if name not in RAW_TABLES:
        raise ValueError(f"Unknown Complete Journey table: {name}")
    path = RAW / f"{name}.parquet"
    try:
        return pd.read_parquet(path, columns=columns, filters=filters)
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"{path} is missing. Run python -m data.etl.download_raw first."
        ) from error


def load_twin_responses(columns: list[str]) -> pd.DataFrame | None:
    filename = "wave1_3_response.csv"
    response_directory = "question_catalog_and_human_response_csv"
    candidates = (
        RAW / filename,
        RAW / response_directory / filename,
        RAW / "Twin-2K-500" / response_directory / filename,
    )
    for path in candidates:
        if path.is_file():
            return pd.read_csv(path, usecols=columns)
    return None


def _require_keys(record, expected: str, location: str) -> None:
    missing = set(expected.split()).difference(record)
    if missing:
        keys = ", ".join(sorted(missing))
        raise ValueError(f"{location} missing required keys or columns: {keys}")


def _read_json(filename: str) -> Any:
    return json.loads((PROCESSED / filename).read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_segments() -> pd.DataFrame:
    filename = "segments.parquet"
    frame = pd.read_parquet(PROCESSED / filename)
    _require_keys(
        frame.columns,
        "segment_id name archetype households share weekly_purchase_prob avg_basket_thb "
        "cluster_label price_elasticity price_elasticity_observed price_elasticity_source "
        "price_elasticity_observed_source coupon_rate promo_response private_label_share source",
        filename,
    )
    return frame


@lru_cache(maxsize=1)
def load_unit_econ() -> dict[str, Any]:
    filename = "unit_economics.json"
    record = _read_json(filename)
    fields = (
        "category currency usd_thb price_thb margin_pct shopee_commission_pct "
        "shopee_txn_fee_pct media_cost_per_reach_thb collected_at"
    )
    _require_keys(record, f"{fields} source", filename)
    _require_keys(record["source"], fields, f"{filename}.source")
    return record


@lru_cache(maxsize=1)
def load_personas() -> list[dict[str, Any]]:
    filename = "personas.json"
    records = _read_json(filename)
    for index, record in enumerate(records):
        location = f"{filename}[{index}]"
        _require_keys(
            record,
            "persona_id segment archetype cluster_label weight age_band income_band household_size "
            "kids homeowner weekly_buy_rate purchase_cycle_weeks avg_units avg_price_paid "
            "price_elasticity price_elasticity_observed promo_share coupon_rate brand_loyalty "
            "traits channels backstory source",
            location,
        )
        _require_keys(
            record["traits"], "impulsiveness risk_aversion time_preference", f"{location}.traits"
        )
        _require_keys(
            record["source"],
            "behavior traits backstory price_elasticity price_elasticity_observed",
            f"{location}.source",
        )
    return records


@lru_cache(maxsize=1)
def load_market() -> dict[str, Any]:
    filename = "market.json"
    market = _read_json(filename)
    _require_keys(
        market,
        "category currency ref_price unit_cost platform_fee_pct payment_fee_pct "
        "shipping_cost_per_order margin_floor_pct segments weeks source_notes",
        filename,
    )
    _require_keys(market["weeks"], "campaign post", f"{filename}.weeks")
    for index, segment in enumerate(market["segments"]):
        _require_keys(
            segment,
            "name archetype cluster_label share weekly_buy_rate households "
            "price_elasticity price_elasticity_observed",
            f"{filename}.segments[{index}]",
        )
    return market


@lru_cache(maxsize=1)
def load_knobs() -> dict[str, Any]:
    filename = "knob_ranges.json"
    knobs = _read_json(filename)
    _require_keys(
        knobs,
        "discount_pct duration_days mechanics cap_per_user calendar channels source",
        filename,
    )
    _require_keys(knobs["discount_pct"], "min p50 p90 max raw_max", f"{filename}.discount_pct")
    return knobs


def load_household_behavior() -> pd.DataFrame:
    """Load local-only household data for calibration runs."""
    path = CALIB / "household_behavior.parquet"
    try:
        panel = pd.read_parquet(path)
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"{path} is missing. Run python -m data.etl.build_tissue after downloading raw data."
        ) from error
    _require_keys(
        panel.columns,
        "household_key archetype segment week bought units price_paid shelf_price on_promo "
        "discount_pct coupon_used in_mailer on_display brand",
        path.name,
    )
    return panel


@lru_cache(maxsize=1)
def load_library() -> pd.DataFrame:
    filename = "campaign_library.csv"
    frame = pd.read_csv(PROCESSED / filename, keep_default_na=False)
    _require_keys(
        frame.columns,
        "campaign_id brand platform category price_tier start_date end_date season mechanic "
        "discount_depth voucher_type min_spend target_segment channel_mix reaction_to "
        "reaction_type reaction_delay_days reported_outcome description source_url confidence",
        filename,
    )
    return frame


@lru_cache(maxsize=1)
def load_rival_prior() -> dict[str, Any]:
    filename = "rival_prior.json"
    record = _read_json(filename)
    _require_keys(
        record,
        "p depth_p50 depth_p90 delay_weeks_p50 delay_weeks_min duration_weeks n_events source",
        filename,
    )
    _require_keys(record["p"], "none match undercut non_price", f"{filename}.p")
    return record


@lru_cache(maxsize=1)
def load_anchors() -> dict[str, list[list[str]]]:
    filename = "ssr_anchors.json"
    record = _read_json(filename)
    _require_keys(record, "th en", filename)
    return record


@lru_cache(maxsize=1)
def load_backtest() -> dict[str, Any]:
    filename = "backtest_campaign.json"
    record = _read_json(filename)
    _require_keys(
        record,
        "campaign_id description start_week end_week discount_depth target_segments "
        "treated_households actual calibration_weeks source",
        filename,
    )
    _require_keys(
        record["actual"], "lift_pct redemption_rate incremental_sales", f"{filename}.actual"
    )
    return record

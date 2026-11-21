"""Validate and load the checked-in synthetic data."""

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

PROCESSED = Path(__file__).resolve().parent / "processed"


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
        "price_elasticity coupon_rate promo_response private_label_share source",
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
            "persona_id segment_id archetype demo behavior psycho "
            "story channels weight edge labels",
            location,
        )
        _require_keys(
            record["demo"],
            "age_band income_band hh_size kids homeowner gender urban",
            f"{location}.demo",
        )
        _require_keys(
            record["behavior"],
            "purchase_every_days aov_thb elasticity coupon_rate promo_sensitivity",
            f"{location}.behavior",
        )
        _require_keys(
            record["psycho"],
            "deal_proneness impulsiveness brand_consciousness risk_aversion digital",
            f"{location}.psycho",
        )
        _require_keys(record["labels"], "demo behavior psycho story", f"{location}.labels")
    return records


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

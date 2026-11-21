"""Load registered public sources, local calibration data and validated engine inputs."""

import json
from functools import lru_cache
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import pandas as pd

from contracts.campaign_spec import CampaignSpec
from data.etl.appeal import APPEAL_KEYS, ARCHETYPES, MECHANICS, MESSAGE_STYLES
from data.etl.config import BACKTEST_START_WEEK, CATEGORY, PRODUCT_PATH
from data.sources import ASSETS, CJ_TABLES

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


def load_category_products() -> pd.DataFrame:
    fields = ("department", "product_category", "product_type")
    return load_raw_table(
        "products",
        filters=[(field, "==", value) for field, value in zip(fields, PRODUCT_PATH, strict=True)],
    )


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
    if panel["week"].ge(BACKTEST_START_WEEK).any():
        raise ValueError(
            f"{path.name} contains post-cutoff weeks. Run python -m data.etl.build_tissue."
        )
    return panel


@lru_cache(maxsize=1)
def load_library() -> pd.DataFrame:
    filename = "campaign_library.csv"
    frame = pd.read_csv(
        PROCESSED / filename,
        keep_default_na=False,
        dtype={"campaign_id": str, "reaction_to_campaign_id": str, "brand": str, "season": str},
    )
    _require_keys(
        frame.columns,
        "campaign_id brand platform category price_tier start_date end_date season mechanic "
        "discount_depth voucher_type min_spend target_segment channel_mix reaction_to_campaign_id "
        "reaction_type reaction_delay_days reported_outcome description source_url confidence",
        filename,
    )
    if frame["campaign_id"].eq("").any() or frame["campaign_id"].duplicated().any():
        raise ValueError("campaign_library.csv requires a unique nonempty campaign_id per row")
    if not frame["reaction_type"].isin(["", "none", "match", "undercut", "non_price"]).all():
        raise ValueError("campaign_library.csv contains an unknown reaction_type")
    return frame


@lru_cache(maxsize=1)
def load_brands() -> pd.DataFrame:
    filename = "brands.csv"
    frame = pd.read_csv(PROCESSED / filename, dtype={"brand": str}, keep_default_na=False)
    _require_keys(
        frame.columns,
        "brand is_ours category sub_category price_tier avg_price target_segments "
        "channels geo market_share typical_discount_pct margin_floor_pct",
        filename,
    )
    if frame["is_ours"].dtype != bool or frame["is_ours"].sum() != 1:
        raise ValueError("brands.csv requires exactly one is_ours brand")
    return frame


@lru_cache(maxsize=1)
def load_rival_prior() -> dict[str, Any]:
    filename = "reaction_base_rate.json"
    record = _read_json(filename)[CATEGORY]
    _require_keys(
        record,
        "none match undercut non_price depth_p50 depth_p90 delay_median n_events source "
        "event_rates background_rates lift n_trigger_weeks n_matched_trigger_weeks "
        "n_background_weeks n_comparisons",
        filename,
    )
    return {
        "p": {name: record[name] for name in ("none", "match", "undercut", "non_price")},
        "event_rates": record["event_rates"],
        "background_rates": record["background_rates"],
        "lift": record["lift"],
        "n_trigger_weeks": record["n_trigger_weeks"],
        "n_matched_trigger_weeks": record["n_matched_trigger_weeks"],
        "n_background_weeks": record["n_background_weeks"],
        "n_comparisons": record["n_comparisons"],
        "depth_p50": record["depth_p50"],
        "depth_p90": record["depth_p90"],
        "delay_weeks_p50": record["delay_median"],
        "delay_weeks_min": record.get("delay_min", record["delay_median"]),
        "duration_weeks": record.get("duration_weeks", 2),
        "n_events": record["n_events"],
        "source": record["source"],
    }


@lru_cache(maxsize=1)
def load_anchors() -> dict[str, list[dict[str, Any]]]:
    filename = "ssr_anchors.json"
    record = _read_json(filename)
    _require_keys(record, "sets test_responses", filename)
    for group in record["sets"]:
        _require_keys(group, "lang anchors", filename)
        if len(group["anchors"]) != 5 or not all(
            isinstance(a, str) and a.strip() for a in group["anchors"]
        ):
            raise ValueError(f"{filename} requires five nonempty anchors per set")
    for response in record["test_responses"]:
        _require_keys(response, "text expected", filename)
        if type(response["expected"]) is not int or not 1 <= response["expected"] <= 5:
            raise ValueError(f"{filename} expected scores must be integers from 1 to 5")
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
        record["actual"],
        "lift_pct lift_range_pct naive_lift_pct redemption_rate incremental_sales",
        f"{filename}.actual",
    )
    return record


@lru_cache(maxsize=1)
def load_appeal_fallback() -> pd.DataFrame:
    filename = "appeal_fallback.csv"
    frame = pd.read_csv(PROCESSED / filename)
    _require_keys(frame.columns, "archetype mechanic message_style appeal label", filename)
    keys = pd.MultiIndex.from_frame(frame[list(APPEAL_KEYS)])
    expected = pd.MultiIndex.from_product([ARCHETYPES, MECHANICS, MESSAGE_STYLES])
    if not keys.is_unique or len(keys) != len(expected) or not expected.isin(keys).all():
        raise ValueError(f"{filename} requires one row for every archetype, mechanic and style")
    if not frame["label"].eq("default").all() or not frame["appeal"].between(-1, 1).all():
        raise ValueError(f"{filename} requires default appeal scores between -1 and 1")
    return frame


@lru_cache(maxsize=1)
def load_assumptions() -> list[dict[str, Any]]:
    filename = "assumptions.json"
    rows = _read_json(filename)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{filename} requires a nonempty list of assumptions")
    seen = set()
    for row in rows:
        _require_keys(row, "name value unit label source used_by", filename)
        if not row["name"] or row["name"] in seen:
            raise ValueError(f"{filename} requires unique assumption names")
        if row["label"] not in {"data", "default", "llm_estimate"}:
            raise ValueError(f"{filename} contains an unknown assumption label")
        consumers = row["used_by"]
        if not row["unit"] or not row["source"] or not isinstance(consumers, list) or not consumers:
            raise ValueError(f"{filename} requires a unit, source and list of consumers")
        seen.add(row["name"])
    return rows


@lru_cache(maxsize=1)
def load_demo_brief() -> dict[str, Any]:
    """Load the stage brief and check its expected parser output against CampaignSpec."""
    filename = "demo_brief.json"
    record = _read_json(filename)
    _require_keys(record, "brief_text user_plan_spec approved_plan expected_story", filename)
    spec = CampaignSpec.model_validate(record["user_plan_spec"])
    for path, meta in spec.field_meta.items():
        if meta.source == "extracted" and meta.evidence not in record["brief_text"]:
            raise ValueError(f"{filename}: evidence for {path} must appear in brief_text")
    return record


def source_asset_path(key: str) -> Path:
    """Resolve only registered files inside the ignored raw-data directory."""
    asset = ASSETS[key]
    return RAW / asset.dataset / asset.filename


def source_asset_info(key: str) -> dict[str, Any] | None:
    path = source_asset_path(key)
    if not path.is_file():
        return None
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def load_source_manifest() -> dict[str, Any]:
    path = PROCESSED / "source_manifest.json"
    if not path.is_file():
        return {"version": 1, "assets": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def save_source_manifest(manifest: dict[str, Any]) -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    destination = PROCESSED / "source_manifest.json"
    temporary = destination.with_suffix(".json.partial")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)


def _source_cache_path(table: str) -> Path:
    # Validate before constructing a path: table is never a user-supplied filename.
    CJ_TABLES[table]
    return RAW / "completejourney" / "normalized" / f"{table}.parquet"


def _read_completejourney_r(table: str, *, offset: int = 0, limit: int = 0) -> pd.DataFrame:
    import pyreadr

    path = source_asset_path(f"completejourney/{CJ_TABLES[table]}")
    objects = pyreadr.read_r(path, row_offset=offset, row_limit=limit)
    if len(objects) != 1:
        raise ValueError(f"Expected one data frame in {table}")
    frame = next(iter(objects.values()))
    if not isinstance(frame, pd.DataFrame):
        raise ValueError(f"Expected tabular data in {table}")
    return frame


def load_completejourney(table: str, *, limit: int | None = None) -> pd.DataFrame:
    """Household-level source data: local ETL/evaluation only, never an LLM input."""
    path = source_asset_path(f"completejourney/{CJ_TABLES[table]}")
    cached = _source_cache_path(table)
    if cached.is_file() and cached.stat().st_mtime_ns >= path.stat().st_mtime_ns:
        if limit is not None:
            import pyarrow.parquet as pq

            if limit < 1:
                raise ValueError("limit must be positive")
            batch = next(pq.ParquetFile(cached).iter_batches(batch_size=limit))
            return batch.to_pandas()
        return pd.read_parquet(cached)
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    if table == "promotions" and limit is None:
        raise ValueError("Large promotions table: pass limit= or normalize it explicitly first")
    return _read_completejourney_r(table, limit=limit or 0)


def normalize_completejourney(table: str, *, chunk_size: int = 1_000_000) -> dict[str, Any]:
    """Convert source R data to local Parquet with bounded per-chunk memory."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    source = source_asset_path(f"completejourney/{CJ_TABLES[table]}")
    destination = _source_cache_path(table)
    if destination.is_file() and destination.stat().st_mtime_ns >= source.stat().st_mtime_ns:
        metadata = pq.ParquetFile(destination)
        return {"rows": metadata.metadata.num_rows, "columns": metadata.schema_arrow.names}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".parquet.partial")
    writer = None
    rows = 0
    columns = []
    try:
        while True:
            frame = _read_completejourney_r(table, offset=rows, limit=chunk_size)
            if frame.empty:
                break
            batch = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(temporary, batch.schema, compression="zstd")
                columns = list(frame.columns)
            writer.write_table(batch)
            rows += len(frame)
            if len(frame) < chunk_size:
                break
        if writer is None:
            raise ValueError(f"Empty source table: {table}")
        writer.close()
        writer = None
        temporary.replace(destination)
    finally:
        if writer is not None:
            writer.close()
        temporary.unlink(missing_ok=True)
    return {"rows": rows, "columns": columns}


def load_twin_responses(
    wave: str | list[str] = "1_3", *, columns: list[str] | None = None
) -> pd.DataFrame | None:
    """Read a registered survey wave, or optional local columns for persona fitting."""
    if isinstance(wave, list):
        if columns is not None:
            raise TypeError("Pass columns only once")
        columns, wave = wave, "1_3"
    filename = {"1_3": "wave1_3_response.csv", "4": "wave4_response.csv"}[wave]
    registered = source_asset_path(f"twin2k/{filename}")
    if columns is None:
        return pd.read_csv(registered, low_memory=False)
    directory = "question_catalog_and_human_response_csv"
    for path in (
        RAW / filename,
        RAW / directory / filename,
        RAW / "Twin-2K-500" / directory / filename,
        registered,
    ):
        if path.is_file():
            return pd.read_csv(path, usecols=columns, low_memory=False)
    return None


def load_twin_catalog() -> Any:
    return json.loads(source_asset_path("twin2k/question_catalog.json").read_text(encoding="utf-8"))


def load_hillstrom() -> pd.DataFrame:
    """Independent randomized email experiment; reserved for evaluation."""
    frame = pd.read_csv(source_asset_path("hillstrom/hillstrom.csv.gz"))
    _require_keys(frame.columns, "segment visit conversion spend", "Hillstrom")
    return frame


def list_breakfast_files() -> list[str]:
    with ZipFile(source_asset_path("breakfast/breakfast_at_the_frat.zip")) as archive:
        return [info.filename for info in archive.infolist() if not info.is_dir()]


def list_breakfast_sheets(member: str) -> list[dict[str, Any]]:
    import openpyxl

    with ZipFile(source_asset_path("breakfast/breakfast_at_the_frat.zip")) as archive:
        content = BytesIO(archive.read(member))
    workbook = openpyxl.load_workbook(content, read_only=True, data_only=True)
    try:
        return [
            {"sheet": sheet.title, "worksheet_rows": sheet.max_row, "columns": sheet.max_column}
            for sheet in workbook.worksheets
        ]
    finally:
        workbook.close()


def load_breakfast_table(
    member: str,
    *,
    sheet_name: str | int = 0,
    nrows: int | None = None,
) -> pd.DataFrame:
    """Read an archive member in memory; never extract untrusted ZIP paths."""
    with ZipFile(source_asset_path("breakfast/breakfast_at_the_frat.zip")) as archive:
        if member not in archive.namelist():
            raise KeyError(member)
        suffix = Path(member).suffix.lower()
        content = BytesIO(archive.read(member))
        if suffix == ".csv":
            return pd.read_csv(content, low_memory=False, nrows=nrows)
        if suffix == ".xlsx":
            return pd.read_excel(content, sheet_name=sheet_name, engine="openpyxl", nrows=nrows)
        raise ValueError("Only CSV and XLSX tables are supported")

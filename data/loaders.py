"""Load synthetic engine fixtures and isolated public-source ETL data."""

import json
from functools import lru_cache
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import pandas as pd

from data.sources import ASSETS, CJ_TABLES

PROCESSED = Path(__file__).resolve().parent / "processed"
RAW = Path(__file__).resolve().parent / "raw"


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


def load_twin_responses(wave: str = "1_3") -> pd.DataFrame:
    """US survey respondents, not a Thai population or an LLM prompt payload."""
    filename = {"1_3": "wave1_3_response.csv", "4": "wave4_response.csv"}[wave]
    return pd.read_csv(source_asset_path(f"twin2k/{filename}"), low_memory=False)


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

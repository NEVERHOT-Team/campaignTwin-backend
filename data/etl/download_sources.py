"""Download public sources, verify bytes, and prepare local ETL tables.

Run: python -m data.etl.download_sources
This never sends source data to a model or replaces the engine's demo fixtures.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from hashlib import md5, sha256

import httpx

from data import loaders
from data.sources import ASSETS, CJ_TABLES, SourceAsset


def download_asset(
    asset: SourceAsset,
    previous: dict,
    client: httpx.Client,
) -> dict:
    existing = loaders.source_asset_info(asset.key)
    if (
        existing
        and previous.get("url") == asset.url
        and existing["sha256"] == previous.get("sha256")
    ):
        return {**previous, "status": "downloaded"}
    destination = loaders.source_asset_path(asset.key)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    digest = sha256()
    published_digest = md5(usedforsecurity=False)
    size = 0
    first = b""
    try:
        with client.stream("GET", asset.url) as response:
            response.raise_for_status()
            with partial.open("wb") as handle:
                for chunk in response.iter_bytes(1024 * 1024):
                    if not first:
                        first = chunk[:256].lstrip().lower()
                    size += len(chunk)
                    digest.update(chunk)
                    published_digest.update(chunk)
                    handle.write(chunk)
        if not size or first.startswith((b"<!doctype html", b"<html")):
            raise ValueError(f"Expected source data, received empty/HTML response: {asset.key}")
        if asset.md5 and published_digest.hexdigest() != asset.md5:
            raise ValueError(f"Source publisher checksum mismatch: {asset.key}")
        if (
            previous.get("url") == asset.url
            and previous.get("sha256")
            and digest.hexdigest() != previous["sha256"]
        ):
            raise ValueError(f"Source changed from the recorded SHA256: {asset.key}")
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)
    return {
        "status": "downloaded",
        "source": "data",
        "url": asset.url,
        "source_page": asset.source_page,
        "license": asset.license,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "bytes": size,
        "sha256": digest.hexdigest(),
        "local_path": f"data/raw/{asset.key}",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--normalize-promotions", action="store_true")
    parser.add_argument(
        "--datasets", nargs="+", default=["completejourney", "twin2k", "hillstrom", "breakfast"]
    )
    args = parser.parse_args()
    known = {asset.dataset for asset in ASSETS.values()}
    if set(args.datasets) - known:
        parser.error(f"datasets must be selected from {sorted(known)}")
    manifest = loaders.load_source_manifest()
    manifest["notes"] = [
        "Raw and normalized household/respondent data stay in ignored data/raw/.",
        "Existing processed engine inputs remain synthetic fixtures (source=default).",
        "US grocery and survey data have not been reweighted to Thailand.",
        "Downloading data is not model calibration, causal validation, or a back-test.",
    ]
    failed = False
    selected = [asset for asset in ASSETS.values() if asset.dataset in args.datasets]
    with httpx.Client(follow_redirects=True, timeout=60) as client:
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {
                pool.submit(
                    download_asset, asset, manifest["assets"].get(asset.key, {}), client
                ): asset
                for asset in selected
            }
            for future in as_completed(futures):
                asset = futures[future]
                try:
                    info = future.result()
                    manifest["assets"][asset.key] = info
                    print(f"downloaded {asset.key}: {info['bytes']:,} bytes", flush=True)
                except Exception as error:
                    failed = True
                    manifest["assets"][asset.key] = {
                        **manifest["assets"].get(asset.key, {}),
                        "status": "failed",
                        "url": asset.url,
                        "error_type": type(error).__name__,
                    }
                    print(f"FAILED {asset.key}: {type(error).__name__}", flush=True)
                loaders.save_source_manifest(manifest)
    if not args.download_only and "completejourney" in args.datasets:
        for table, filename in CJ_TABLES.items():
            info = manifest["assets"].get(f"completejourney/{filename}", {})
            if info.get("status") != "downloaded":
                continue
            if table == "promotions" and not args.normalize_promotions:
                info["normalization"] = "deferred_large_table_use_normalize_promotions_flag"
                print("promotions downloaded in full; bulk normalization is opt-in", flush=True)
                loaders.save_source_manifest(manifest)
                continue
            try:
                info["table"] = loaders.normalize_completejourney(table)
                info.pop("normalization_error", None)
                print(f"normalized {table}: {info['table']['rows']:,} rows", flush=True)
            except Exception as error:
                failed = True
                info["normalization_error"] = type(error).__name__
                print(f"FAILED normalization {table}: {type(error).__name__}", flush=True)
            loaders.save_source_manifest(manifest)
    if not args.download_only and "twin2k" in args.datasets:
        for wave in ("1_3", "4"):
            key = f"twin2k/wave{wave}_response.csv"
            if manifest["assets"].get(key, {}).get("status") == "downloaded":
                frame = loaders.load_twin_responses(wave)
                manifest["assets"][key]["table"] = {
                    "rows": len(frame),
                    "column_count": len(frame.columns),
                }
                print(f"validated Twin wave {wave}: {len(frame):,} respondents", flush=True)
        if manifest["assets"].get("twin2k/question_catalog.json", {}).get("status") == "downloaded":
            loaders.load_twin_catalog()
    if not args.download_only and "hillstrom" in args.datasets:
        key = "hillstrom/hillstrom.csv.gz"
        if manifest["assets"].get(key, {}).get("status") == "downloaded":
            frame = loaders.load_hillstrom()
            manifest["assets"][key]["table"] = {
                "rows": len(frame),
                "columns": list(frame.columns),
                "treatment_counts": frame["segment"].value_counts().to_dict(),
                "publisher_checksum_verified": True,
            }
            print(f"validated Hillstrom: {len(frame):,} customers", flush=True)
    if not args.download_only and "breakfast" in args.datasets:
        key = "breakfast/breakfast_at_the_frat.zip"
        if manifest["assets"].get(key, {}).get("status") == "downloaded":
            manifest["assets"][key]["archive_members"] = loaders.list_breakfast_files()
            print("validated Breakfast ZIP directory", flush=True)
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    loaders.save_source_manifest(manifest)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

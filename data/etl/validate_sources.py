"""Inspect downloaded sources without exposing respondent/household records."""

import json
from datetime import datetime, timezone

from data import loaders


def main() -> int:
    manifest = loaders.load_source_manifest()
    failures = [key for key, info in manifest["assets"].items() if info["status"] != "downloaded"]
    if failures:
        raise ValueError(f"Download these assets before validation: {failures}")
    for key, info in manifest["assets"].items():
        if loaders.source_asset_info(key)["sha256"] != info["sha256"]:
            raise ValueError(f"Downloaded source checksum mismatch: {key}")
    workbook = next(name for name in loaders.list_breakfast_files() if name.endswith(".xlsx"))
    worksheets = []
    for sheet in loaders.list_breakfast_sheets(workbook):
        sample = loaders.load_breakfast_table(workbook, sheet_name=sheet["sheet"], nrows=3)
        worksheets.append(
            {
                **sheet,
                "readable": True,
                "sample_rows_read": len(sample),
                "header": [str(column) for column in sample.columns],
            }
        )
    manifest["assets"]["breakfast/breakfast_at_the_frat.zip"]["worksheets"] = worksheets
    loaders.save_source_manifest(manifest)
    transactions = loaders.load_completejourney("transactions")
    products = loaders.load_completejourney("products")
    demographics = loaders.load_completejourney("demographics")
    personas = loaders.load_personas()
    library = loaders.load_library()
    missing_products = ~transactions["product_id"].isin(products["product_id"])
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "data",
        "downloaded_assets": len(manifest["assets"]),
        "all_downloads_complete": bool(manifest["assets"]) and not failures,
        "download_bytes": sum(item["bytes"] for item in manifest["assets"].values()),
        "completejourney": {
            "transactions": len(transactions),
            "households": int(transactions["household_id"].nunique()),
            "products": len(products),
            "households_with_demographics": len(demographics),
            "transaction_product_ids_missing_from_products": int(missing_products.sum()),
            "unique_product_ids_missing_from_products": int(
                transactions.loc[missing_products, "product_id"].nunique()
            ),
            "week_min": int(transactions["week"].min()),
            "week_max": int(transactions["week"].max()),
            "source_currency": "USD",
            "promotions": (
                "full RDS downloaded; bulk conversion deferred unless explicitly requested"
            ),
            "brand_field_values": sorted(products["brand"].dropna().astype(str).unique().tolist()),
        },
        "twin2k": {
            "wave1_3_respondents": len(loaders.load_twin_responses("1_3")),
            "wave4_respondents": len(loaders.load_twin_responses("4")),
            "population": "US; no Thai reweighting performed",
        },
        "hillstrom": {
            "rows": len(loaders.load_hillstrom()),
            "publisher_checksum_verified": True,
            "distribution": "scikit-uplift mirror of Kevin Hillstrom dataset",
        },
        "breakfast": {"archive_readable": True, "worksheets": worksheets},
        "engine_persona_profiles": len(personas),
        "engine_campaign_library_records": len(library),
        "engine_inputs_remain_synthetic": (
            all(set(persona["labels"].values()) <= {"default"} for persona in personas)
            and set(library["confidence"]) == {"default"}
        ),
        "calibration_performed": False,
        "limitations": [
            "Household/respondent records stay in ignored data/raw and are never LLM inputs.",
            "Unmatched product metadata must be handled explicitly before category fitting.",
            "National/Private labels and manufacturer IDs are not named Thai competitors.",
            "No Thai campaign library, Thai reweighting or verified platform fees added.",
            "No model calibration or historical back-test has run.",
            "Worksheet dimensions include headers/formatting, not verified data-row counts.",
        ],
    }
    path = loaders.PROCESSED / "source_quality.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "downloaded_assets": report["downloaded_assets"],
                "all_downloads_complete": report["all_downloads_complete"],
                "source_quality_report": "data/processed/source_quality.json",
                "unmatched_transaction_rows": int(missing_products.sum()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

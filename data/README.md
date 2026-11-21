# Backend fixture data

`processed/` contains synthetic segments, personas, unit economics, rival examples,
purchase-intent anchors, and a held-out campaign example. Their source labels
are `default`. The campaign-library brands and outcomes are fictional, so their
`source_url` cells are empty.

Import the cached readers in `data/loaders.py` for all fixture reads. Treat the
returned tables and dictionaries as read-only. Keep household-level inputs in
the ignored `raw/` directory and out of model prompts.

Fields ending in `_pct` use percentage points. The `discount_depth` and rival
`depth` fields, probabilities, and shares use fractions between 0 and 1.

## Downloaded public sources

The source downloader uses `data/sources.py` and records actual URLs, retrieval
times, byte counts, SHA256 checksums, attribution, and table metadata in
`processed/source_manifest.json`. See `processed/source_quality.json` for the
observed counts and limitations from the latest validation.

```powershell
.venv/Scripts/python.exe -m pip install -e ".[data]"
.venv/Scripts/python.exe -m data.etl.download_sources
.venv/Scripts/python.exe -m data.etl.validate_sources
```

Downloads are resumable by asset: reruns skip files matching their recorded
checksum and repair corrupted local files. Unexpected remote changes to a
recorded source checksum are rejected. Hillstrom is additionally checked against
the checksum published by scikit-uplift. Download errors preserve existing files.

| Source | Local preparation | Attribution |
|---|---|---|
| Complete Journey | Eight original R assets; seven normalized Parquet tables | [completejourney](https://bradleyboehmke.github.io/completejourney/), Brad Boehmke / Steven Mortimer, original study 84.51° |
| Twin-2K-500 | Numeric responses for waves 1–3 and wave 4, question catalog and documentation | [LLM-Digital-Twin](https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500), CC-BY-4.0 |
| Hillstrom | Compressed CSV with 64,000 customers | Kevin Hillstrom / MineThatData; [scikit-uplift mirror](https://www.uplift-modeling.com/en/latest/api/datasets/fetch_hillstrom.html) because the author's download host was unavailable |
| Breakfast at the Frat | Original ZIP containing workbook and user guide | [dunnhumby Source Files](https://www.dunnhumby.com/source-files/) |

Originals and normalized household/respondent tables stay in ignored `raw/`.
The full promotions RDS is downloaded but bulk conversion is opt-in because it
contains about 20.9 million rows and R parsing is expensive on a small laptop.
Run the downloader with `--datasets completejourney --normalize-promotions`
to explicitly convert it. `load_completejourney` requires
an explicit `limit` for unnormalized promotions to prevent accidental bulk reads.
Even a limited R read may scan the original compressed object; it is an ETL
operation, not something to perform inside a simulation loop.

Read all source data through the loaders:

```python
from data.loaders import (
    list_breakfast_files,
    load_breakfast_table,
    load_completejourney,
    load_hillstrom,
    load_twin_catalog,
    load_twin_responses,
)

transactions = load_completejourney("transactions")
demographics = load_completejourney("demographics")
products = load_completejourney("products")
survey = load_twin_responses("1_3")
repeat_survey = load_twin_responses("4")
questions = load_twin_catalog()
randomized_evaluation = load_hillstrom()
workbook = next(name for name in list_breakfast_files() if name.endswith(".xlsx"))
weekly_sample = load_breakfast_table(
    workbook, sheet_name="dh Transaction Data", nrows=1000,
)
```

The existing `load_personas()`, `load_segments()`, `load_library()`, and
`load_unit_econ()` still return the original synthetic engine fixtures. Downloading
source files does not fit segments, create a data-grounded persona bank, estimate
rival likelihood, or validate simulation lift. Those are separate ETL/model steps.

Important source limitations:

- Complete Journey prices are USD. Thai prices, incomes, channels, costs, and fees
  have not been verified or inferred by this downloader.
- Transactions include product IDs without lookup metadata; the quality report
  records their count. Handle these explicitly before fitting a category model.
- The product `brand` field distinguishes National/Private labels. Manufacturer
  IDs are available, but are not named Thai competitors.
- Twin survey respondents are a separate US sample. Never join their participant
  IDs to grocery household IDs or describe the result as observed joint data.
- Keep a historical campaign held out before calibration. Hillstrom is an
  independent evaluation experiment, not a grocery campaign holdout.
- Before creating personas, publish only aggregates with at least 50 households;
  retain provenance for observed behavior versus assumed psychographics/context.
- Worksheet dimensions in the report include headers/formatting, so they are not
  claimed as verified data-row counts. The ZIP is never extracted to disk.

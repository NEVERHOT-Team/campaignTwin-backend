"""Extract Complete Journey's Parquet tables from the installed Python package."""

from importlib.metadata import version
from pathlib import Path
from tempfile import NamedTemporaryFile

from completejourney_py import get_data

from data.loaders import RAW, RAW_TABLES

TWIN_2K_URL = "https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500"
BREAKFAST_URL = "https://www.dunnhumby.com/source-files/"


def main() -> None:
    print(f"Manual download — Twin-2K-500: {TWIN_2K_URL}")
    print(f"Manual download — Breakfast at the Frat: {BREAKFAST_URL}")
    print(f"Extracting completejourney_py {version('completejourney_py')} to {RAW}", flush=True)
    RAW.mkdir(parents=True, exist_ok=True)

    for name in RAW_TABLES:
        frame = get_data(name)[name]
        if frame.empty:
            raise ValueError(f"Complete Journey returned an empty table: {name}")

        path = RAW / f"{name}.parquet"
        with NamedTemporaryFile(dir=RAW, prefix=f".{name}-", suffix=".parquet", delete=False) as f:
            temporary_path = Path(f.name)
        try:
            frame.to_parquet(temporary_path, index=False)
            temporary_path.replace(path)
        finally:
            temporary_path.unlink(missing_ok=True)
        print(f"{path.name:29} {len(frame):>10,} rows", flush=True)


if __name__ == "__main__":
    main()

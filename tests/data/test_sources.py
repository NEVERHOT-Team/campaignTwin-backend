from hashlib import sha256
from io import BytesIO
from zipfile import ZipFile

import httpx
import pandas as pd
import pytest

from data import loaders
from data.etl.download_sources import download_asset
from data.sources import ASSETS


@pytest.fixture
def isolated_sources(monkeypatch, tmp_path):
    monkeypatch.setattr(loaders, "RAW", tmp_path / "raw")
    monkeypatch.setattr(loaders, "PROCESSED", tmp_path / "processed")


def test_download_records_provenance_skips_verified_bytes_and_repairs_corruption(isolated_sources):
    asset = ASSETS["twin2k/wave4_response.csv"]
    content = b"pid,answer\n1,3\n"
    requests = []

    def handler(request):
        requests.append(request.url)
        return httpx.Response(200, content=content)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        info = download_asset(asset, {}, client)
        assert info["sha256"] == sha256(content).hexdigest()
        assert info["source"] == "data"
        assert info["license"] == "CC-BY-4.0"
        assert download_asset(asset, info, client) == info
        assert len(requests) == 1
        loaders.source_asset_path(asset.key).write_bytes(b"corrupted")
        repaired = download_asset(asset, info, client)
        assert repaired["sha256"] == info["sha256"]
        assert len(requests) == 2


@pytest.mark.parametrize(
    "status,content", [(200, b"<html>error</html>"), (200, b""), (503, b"busy")]
)
def test_failed_download_preserves_existing_source(isolated_sources, status, content):
    asset = ASSETS["twin2k/wave4_response.csv"]
    destination = loaders.source_asset_path(asset.key)
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"existing source")
    transport = httpx.MockTransport(lambda _: httpx.Response(status, content=content))
    with httpx.Client(transport=transport) as client:
        with pytest.raises((ValueError, httpx.HTTPStatusError)):
            download_asset(asset, {}, client)
    assert destination.read_bytes() == b"existing source"
    assert not destination.with_suffix(".csv.partial").exists()


def test_published_hillstrom_checksum_is_enforced(isolated_sources):
    asset = ASSETS["hillstrom/hillstrom.csv.gz"]
    transport = httpx.MockTransport(lambda _: httpx.Response(200, content=b"wrong archive"))
    with httpx.Client(transport=transport) as client:
        with pytest.raises(ValueError, match="publisher checksum mismatch"):
            download_asset(asset, {}, client)
    assert not loaders.source_asset_path(asset.key).exists()


def test_changed_remote_source_does_not_silently_replace_recorded_data(isolated_sources):
    asset = ASSETS["twin2k/wave4_response.csv"]
    previous = {"url": asset.url, "sha256": sha256(b"original source").hexdigest()}
    transport = httpx.MockTransport(lambda _: httpx.Response(200, content=b"changed source"))
    with httpx.Client(transport=transport) as client:
        with pytest.raises(ValueError, match="Source changed from the recorded SHA256"):
            download_asset(asset, previous, client)
    assert not loaders.source_asset_path(asset.key).exists()


def test_chunked_conversion_roundtrips_without_duplicate_or_missing_rows(
    isolated_sources, monkeypatch
):
    source = loaders.source_asset_path("completejourney/transactions.rds")
    source.parent.mkdir(parents=True)
    source.write_bytes(b"mock R table")
    expected = pd.DataFrame({"week": [1, 2, 3, 4, 5], "quantity": [2, 1, 4, 1, 3]})
    calls = []

    def read_chunk(table, *, offset=0, limit=0):
        calls.append(offset)
        return expected.iloc[offset : offset + limit].copy()

    monkeypatch.setattr(loaders, "_read_completejourney_r", read_chunk)
    info = loaders.normalize_completejourney("transactions", chunk_size=2)
    assert info == {"rows": 5, "columns": ["week", "quantity"]}
    assert calls == [0, 2, 4]
    pd.testing.assert_frame_equal(loaders.load_completejourney("transactions"), expected)
    pd.testing.assert_frame_equal(
        loaders.load_completejourney("transactions", limit=2), expected[:2]
    )
    assert loaders.normalize_completejourney("transactions") == info
    assert calls == [0, 2, 4]


def test_source_paths_are_allowlisted_and_large_promotions_need_explicit_loading(isolated_sources):
    with pytest.raises(KeyError):
        loaders.source_asset_path("../../.env")
    with pytest.raises(KeyError):
        loaders.load_completejourney("../../.env")
    source = loaders.source_asset_path("completejourney/promotions.rds")
    source.parent.mkdir(parents=True)
    source.write_bytes(b"mock R table")
    with pytest.raises(ValueError, match="Large promotions table"):
        loaders.load_completejourney("promotions")
    with pytest.raises(ValueError, match="limit must be positive"):
        loaders.load_completejourney("promotions", limit=0)


def test_zip_members_are_read_without_extracting_paths(isolated_sources):
    archive = BytesIO()
    with ZipFile(archive, "w") as handle:
        handle.writestr("../outside.csv", "week,units\n1,3\n")
    path = loaders.source_asset_path("breakfast/breakfast_at_the_frat.zip")
    path.parent.mkdir(parents=True)
    path.write_bytes(archive.getvalue())
    assert loaders.list_breakfast_files() == ["../outside.csv"]
    frame = loaders.load_breakfast_table("../outside.csv")
    assert frame.to_dict("records") == [{"week": 1, "units": 3}]
    assert not (loaders.RAW / "outside.csv").exists()

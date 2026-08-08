from __future__ import annotations

from dataclasses import replace
import sys
from pathlib import Path
import json
from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lunar_policy_training.polar_data.split import (  # noqa: E402
    SplitError,
    build_split_catalog,
    jaxa_sites,
    nasa_windows,
    nasa_window_from_world_bounds,
    nasa_windows_from_dem,
    build_split_manifest,
    main,
)
from lunar_policy_training.polar_data.source_lock import PolarSourceLock, write_aggregate_source_lock  # noqa: E402
import lunar_policy_training.polar_data.split as split_module  # noqa: E402


def test_jaxa_sites_are_holdout_only() -> None:
    catalog = build_split_catalog(nasa_windows(), jaxa_sites(), seed=4080)

    assert {row.split for row in catalog if row.source == "JAXA_LUPEX"} == {"holdout"}
    assert sum(row.split == "train" for row in catalog) == 192
    assert sum(row.split == "validation" for row in catalog) == 48
    assert sum(row.split == "test" for row in catalog) == 48
    assert {row.window_id for row in catalog if row.source == "JAXA_LUPEX"} == {
        "CR1", "GR1", "GR2", "LP1", "MP1", "MP2"
    }


def test_split_catalog_is_deterministic_redacted_and_spatially_separated() -> None:
    first = build_split_catalog(nasa_windows(), jaxa_sites(), seed=4080)
    second = build_split_catalog(nasa_windows(), jaxa_sites(), seed=4080)

    assert first == second
    assert len({row.split_sha256 for row in first}) == 1
    assert all(row.width_m == row.height_m == 1024 for row in first if row.source == "NASA_LOLA")
    assert all("/" not in row.window_id and "\\" not in row.window_id for row in first)
    assert all("kai" not in row.to_json().lower() for row in first)
    nasa_rows = [row for row in first if row.source == "NASA_LOLA"]
    for index, left in enumerate(nasa_rows):
        for right in nasa_rows[index + 1 :]:
            if left.split != right.split:
                assert max(abs(left.center_x_m - right.center_x_m), abs(left.center_y_m - right.center_y_m)) >= 2000


def test_split_catalog_rejects_overlapping_or_too_close_nasa_windows() -> None:
    windows = list(nasa_windows())
    windows[1] = windows[0]

    with pytest.raises(SplitError, match="overlap|2000"):
        build_split_catalog(windows, jaxa_sites(), seed=4080)


def test_split_cli_requires_source_lock_and_output() -> None:
    with pytest.raises(SystemExit, match="2"):
        main([])


def test_split_cli_binds_rows_to_verified_dem_pixel_bounds(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    raw = tmp_path / "raw"
    raw.mkdir()
    dem = raw / "ldem_87s_5mpp.tif"
    with rasterio.open(
        dem, "w", driver="GTiff", height=192, width=384, count=1, dtype="float32",
        crs="EPSG:3031", transform=from_origin(0.0, 50_000.0, 256.0, 256.0), nodata=-9999.0,
    ) as dataset:
        dataset.write(np.zeros((1, 192, 384), dtype="float32"))
    count = raw / "ldec_87s_5mpp.tif"
    with rasterio.open(
        count, "w", driver="GTiff", height=192, width=384, count=1, dtype="uint8",
        crs="EPSG:3031", transform=from_origin(0.0, 50_000.0, 256.0, 256.0),
    ) as dataset:
        dataset.write(np.ones((1, 192, 384), dtype="uint8"))
    archive = raw / "DataS1.zip"
    _write_jaxa_archive(archive)
    dem_lock = PolarSourceLock.from_file(
        "NASA_LOLA_87S_DEM", dem, citation="NASA PGDA product 81", license="NASA reproduction guidance",
        final_url="https://pgda.gsfc.nasa.gov/data/LOLA_5mpp/87S/ldem_87s_5mpp.tif",
    )
    count_lock = PolarSourceLock.from_file(
        "NASA_LOLA_87S_COUNT", count, citation="NASA PGDA product 81", license="NASA reproduction guidance",
        final_url="https://pgda.gsfc.nasa.gov/data/LOLA_5mpp/87S/ldec_87s_5mpp.tif",
    )
    jaxa_lock = PolarSourceLock.from_file(
        "JAXA_LUPEX_DATA_S1", archive, citation="https://doi.org/10.5281/zenodo.17153447",
        license="cc-by-4.0", final_url="https://zenodo.org/api/files/DataS1.zip",
    )
    lock_dir = tmp_path / "locks"
    lock_dir.mkdir()
    aggregate = lock_dir / "polar_source_lock_v1.json"
    write_aggregate_source_lock(aggregate, raw, (dem_lock, count_lock, jaxa_lock), repository_root=repository)
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    output = split_dir / "polar_split_v1.json"

    assert main(["--source-lock", str(aggregate), "--output", str(output), "--seed", "4080", "--repository-root", str(repository)]) == 0
    document = json.loads(output.read_text(encoding="utf-8"))
    nasa_rows = [row for row in document["rows"] if row["source"] == "NASA_LOLA"]
    assert len(nasa_rows) == 288
    pixel_bounds = nasa_rows[0]["source_pixel_bounds"]
    assert pixel_bounds[2] - pixel_bounds[0] == 4
    assert nasa_rows[0]["world_bounds_m"][2] - nasa_rows[0]["world_bounds_m"][0] == 1024
    assert len(nasa_rows[0]["window_sha256"]) == 64
    jaxa_rows = [row for row in document["rows"] if row["source"] == "JAXA_LUPEX"]
    assert all(row["archive_sha256"] == jaxa_lock.sha256 for row in jaxa_rows)
    assert all(row["archive_member_sha256s"] for row in jaxa_rows)
    assert str(raw) not in output.read_text(encoding="utf-8")


def _archive_raster_bytes() -> bytes:
    with MemoryFile() as memory:
        with memory.open(
            driver="GTiff", height=2, width=2, count=1, dtype="float32", crs="EPSG:3031",
            transform=from_origin(0.0, 10.0, 1.0, 1.0), nodata=-9999.0,
        ) as dataset:
            dataset.write(np.ones((1, 2, 2), dtype="float32"))
        return memory.read()


def _write_jaxa_archive(path: Path) -> None:
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        for site_id in ("CR1", "GR1", "GR2", "LP1", "MP1", "MP2"):
            archive.writestr(f"{site_id}/{site_id}_DTM.tif", _archive_raster_bytes())


def test_5m_window_keeps_exact_1024m_world_and_fractional_pixel_bounds() -> None:
    window = nasa_window_from_world_bounds(
        from_origin(0.0, 1024.0, 5.0, 5.0), (0.0, 0.0, 1024.0, 1024.0), "d" * 64, "c" * 64
    )
    assert window.source_pixel_bounds == (0.0, 0.0, 204.8, 204.8)
    assert window.world_bounds_m == (0.0, 0.0, 1024.0, 1024.0)
    assert window.read_pixel_envelope == (0, 0, 205, 205)


def test_split_manifest_requires_count_and_jaxa_locks(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    raw = tmp_path / "raw"
    raw.mkdir()
    dem = raw / "dem.tif"
    with rasterio.open(dem, "w", driver="GTiff", height=192, width=384, count=1, dtype="float32", crs="EPSG:3031", transform=from_origin(0, 50_000, 256, 256), nodata=-9999) as dataset:
        dataset.write(np.zeros((1, 192, 384), dtype="float32"))
    dem_lock = PolarSourceLock.from_file("NASA_LOLA_87S_DEM", dem, citation="NASA", license="NASA", final_url="https://example.invalid/dem")
    lock_dir = tmp_path / "locks"
    lock_dir.mkdir()
    aggregate = lock_dir / "lock.json"
    output_dir = tmp_path / "splits"
    output_dir.mkdir()
    write_aggregate_source_lock(aggregate, raw, (dem_lock,), repository_root=repository)
    with pytest.raises(SplitError, match="three|required"):
        build_split_manifest(aggregate, output_dir / "split.json", seed=4080, repository_root=repository)


@pytest.mark.parametrize("missing_id", ["NASA_LOLA_87S_COUNT", "JAXA_LUPEX_DATA_S1"])
def test_split_manifest_rejects_each_required_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing_id: str
) -> None:
    required = ("NASA_LOLA_87S_DEM", "NASA_LOLA_87S_COUNT", "JAXA_LUPEX_DATA_S1")
    locks = tuple(
        type("Lock", (), {"source_id": source_id, "artifact_kind": "raster"})()
        for source_id in required if source_id != missing_id
    )
    monkeypatch.setattr(split_module, "load_aggregate_source_lock", lambda *_args, **_kwargs: (tmp_path, locks))
    with pytest.raises(SplitError, match="all three"):
        build_split_manifest(tmp_path / "lock.json", tmp_path / "split.json", seed=4080, repository_root=tmp_path)


def test_count_hash_changes_split_identity_and_window_identity() -> None:
    baseline = build_split_catalog(nasa_windows(), jaxa_sites(), seed=4080)
    count_changed = build_split_catalog(
        tuple(replace(window, count_sha256="f" * 64) for window in nasa_windows()),
        jaxa_sites(), seed=4080,
    )
    assert baseline[0].split_sha256 != count_changed[0].split_sha256
    assert baseline[0].window_sha256 != count_changed[0].window_sha256


def test_count_quality_stratification_changes_selected_candidate_set(tmp_path: Path) -> None:
    dem = tmp_path / "dem.tif"
    count_flat = tmp_path / "count-flat.tif"
    count_banded = tmp_path / "count-banded.tif"
    transform = from_origin(0.0, 50_000.0, 256.0, 256.0)
    zeros = np.zeros((1, 208, 384), dtype="float32")
    flat = np.ones((1, 208, 384), dtype="float32")
    banded = flat.copy()
    banded[:, :112, :] = 0.0
    for path, values in ((dem, zeros), (count_flat, flat), (count_banded, banded)):
        with rasterio.open(path, "w", driver="GTiff", height=208, width=384, count=1, dtype="float32", crs="EPSG:3031", transform=transform, nodata=-9999) as dataset:
            dataset.write(values)
    flat_selection = nasa_windows_from_dem(dem, "d" * 64, count_flat, "a" * 64)
    banded_selection = nasa_windows_from_dem(dem, "d" * 64, count_banded, "b" * 64)
    assert len(flat_selection) == len(banded_selection) == 288
    assert {window.count_quality for window in flat_selection} == {1.0}
    assert {window.count_quality for window in banded_selection} == {0.0, 1.0}
    assert {window.count_sha256 for window in flat_selection} != {window.count_sha256 for window in banded_selection}


def test_split_manifest_rejects_dem_count_alignment_drift(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    raw = tmp_path / "raw"
    raw.mkdir()
    for name, offset in (("dem.tif", 0.0), ("count.tif", 1.0)):
        with rasterio.open(raw / name, "w", driver="GTiff", height=192, width=384, count=1, dtype="float32", crs="EPSG:3031", transform=from_origin(offset, 50_000, 256, 256), nodata=-9999) as dataset:
            dataset.write(np.ones((1, 192, 384), dtype="float32"))
    archive = raw / "DataS1.zip"
    _write_jaxa_archive(archive)
    locks = (
        PolarSourceLock.from_file("NASA_LOLA_87S_DEM", raw / "dem.tif", citation="NASA", license="NASA", final_url="https://example.invalid/dem"),
        PolarSourceLock.from_file("NASA_LOLA_87S_COUNT", raw / "count.tif", citation="NASA", license="NASA", final_url="https://example.invalid/count"),
        PolarSourceLock.from_file("JAXA_LUPEX_DATA_S1", archive, citation="Zenodo", license="cc-by-4.0", final_url="https://example.invalid/jaxa"),
    )
    lock_dir = tmp_path / "locks"
    lock_dir.mkdir()
    aggregate = lock_dir / "lock.json"
    write_aggregate_source_lock(aggregate, raw, locks, repository_root=repository)
    output_dir = tmp_path / "splits"
    output_dir.mkdir()
    with pytest.raises(SplitError, match="alignment"):
        build_split_manifest(aggregate, output_dir / "split.json", seed=4080, repository_root=repository)

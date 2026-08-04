from __future__ import annotations

import sys
from pathlib import Path
import json

import numpy as np
import rasterio
from rasterio.transform import from_origin

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lunar_policy_training.polar_data.split import (  # noqa: E402
    SplitError,
    build_split_catalog,
    jaxa_sites,
    nasa_windows,
    main,
)
from lunar_policy_training.polar_data.source_lock import PolarSourceLock, write_aggregate_source_lock  # noqa: E402


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
    lock = PolarSourceLock.from_file(
        "NASA_LOLA_87S_DEM", dem, citation="NASA PGDA product 81", license="NASA reproduction guidance",
        final_url="https://pgda.gsfc.nasa.gov/data/LOLA_5mpp/87S/ldem_87s_5mpp.tif",
    )
    lock_dir = tmp_path / "locks"
    lock_dir.mkdir()
    aggregate = lock_dir / "polar_source_lock_v1.json"
    write_aggregate_source_lock(aggregate, raw, (lock,), repository_root=repository)
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    output = split_dir / "polar_split_v1.json"

    assert main(["--source-lock", str(aggregate), "--output", str(output), "--seed", "4080", "--repository-root", str(repository)]) == 0
    document = json.loads(output.read_text(encoding="utf-8"))
    nasa_rows = [row for row in document["rows"] if row["source"] == "NASA_LOLA"]
    assert len(nasa_rows) == 288
    assert nasa_rows[0]["source_pixel_bounds"] == [0, 0, 4, 4]
    assert len(nasa_rows[0]["window_sha256"]) == 64
    assert str(raw) not in output.read_text(encoding="utf-8")

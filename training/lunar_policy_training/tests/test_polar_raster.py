from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import lunar_policy_training.polar_data.raster as raster_module  # noqa: E402
from lunar_policy_training.polar_data.raster import (  # noqa: E402
    GLOBAL_GEOMETRY,
    LOCAL_GEOMETRY,
    GridGeometry,
    MapCanvas,
    load_polar_window,
    RasterError,
    resample_average,
    resample_bilinear,
    resample_nearest,
    validate_roi_bounds,
)


def test_fixed_axis_aligned_global_and_local_geometries() -> None:
    local_tile_geometry = getattr(raster_module, "LOCAL_TILE_GEOMETRY", None)
    assert local_tile_geometry is not None, "formal 0.2 m tile geometry is missing"
    assert GLOBAL_GEOMETRY == GridGeometry(size_m=1024.0, resolution_m=4.0, cells=256)
    assert local_tile_geometry == GridGeometry(
        size_m=64.0,
        resolution_m=0.2,
        cells=320,
    )
    assert LOCAL_GEOMETRY == GridGeometry(size_m=6.4, resolution_m=0.2, cells=32)
    assert GLOBAL_GEOMETRY.axis_aligned
    assert local_tile_geometry.axis_aligned
    assert LOCAL_GEOMETRY.axis_aligned


def test_semantic_resampling_uses_bilinear_nearest_and_area_average() -> None:
    elevation = np.arange(16, dtype=np.float32).reshape(4, 4)
    mask = np.array([[0, 0, 1, 1], [0, 1, 1, 1], [1, 1, 0, 0], [1, 1, 0, 0]], dtype=bool)
    ratio = np.array([[0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 1.0, 1.0], [1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]], dtype=np.float32)

    np.testing.assert_allclose(resample_bilinear(elevation, (2, 2)), [[2.5, 4.5], [10.5, 12.5]])
    np.testing.assert_array_equal(resample_nearest(mask, (2, 2)), [[True, True], [True, False]])
    np.testing.assert_allclose(resample_average(ratio, (2, 2)), [[0.25, 1.0], [1.0, 0.0]])


def test_resampling_rejects_nodata_as_observed_and_invalid_geometry() -> None:
    with pytest.raises(RasterError, match="NoData"):
        resample_bilinear(np.array([[0.0, np.nan], [2.0, 3.0]], dtype=np.float32), (1, 1), observed_mask=np.ones((2, 2), dtype=bool))
    with pytest.raises(RasterError, match="integer"):
        GridGeometry(size_m=3.0, resolution_m=1.0, cells=2)
    with pytest.raises(RasterError, match="outside"):
        validate_roi_bounds((-1.0, 0.0, 1.0, 1.0), GLOBAL_GEOMETRY)


def test_canvas_is_absolute_north_up_and_centered_on_roi_bbox() -> None:
    canvas = MapCanvas.from_roi_bounds("a" * 64, (1_000.0, 2_000.0, 1_024.0, 2_040.0))
    assert canvas.bounds_m == (500.0, 1_508.0, 1_524.0, 2_532.0)
    assert canvas.world_to_grid(502.0, 2_530.0) == (0, 0)
    assert canvas.world_to_grid(502.0, 1_510.0) == (255, 0)
    assert canvas.grid_center_world(0, 0) == (502.0, 2_530.0)


def test_loaded_window_reads_real_north_up_raster_and_keeps_nodata_padding_unknown(tmp_path: pathlib.Path) -> None:
    source = tmp_path / "dem.tif"
    values = np.arange(256 * 256, dtype=np.float32).reshape(256, 256)
    values[0, 0] = -9999.0
    with rasterio.open(source, "w", driver="GTiff", height=256, width=256, count=1, dtype="float32", transform=from_origin(1_000.0, 3_000.0, 4.0, 4.0), nodata=-9999.0) as dataset:
        dataset.write(values, 1)
    canvas = MapCanvas.from_roi_bounds("b" * 64, (1_500.0, 2_480.0, 1_524.0, 2_520.0))
    loaded = load_polar_window(source, canvas)
    assert loaded.canvas is canvas
    assert not loaded.observed_mask[0, 0]
    assert loaded.observed_mask[4, 4]
    assert loaded.elevation_m[4, 4] == values[1, 4]


def test_canvas_rejects_non_lowercase_hex_sha() -> None:
    with pytest.raises(RasterError, match="lowercase"):
        MapCanvas.from_roi_bounds("A" * 64, (0.0, 0.0, 1.0, 1.0))

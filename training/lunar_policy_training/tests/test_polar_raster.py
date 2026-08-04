from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from lunar_policy_training.polar_data.raster import (  # noqa: E402
    GLOBAL_GEOMETRY,
    LOCAL_GEOMETRY,
    GridGeometry,
    RasterError,
    resample_average,
    resample_bilinear,
    resample_nearest,
    validate_roi_bounds,
)


def test_fixed_axis_aligned_global_and_local_geometries() -> None:
    assert GLOBAL_GEOMETRY == GridGeometry(size_m=1024.0, resolution_m=4.0, cells=256)
    assert LOCAL_GEOMETRY == GridGeometry(size_m=8.0, resolution_m=0.25, cells=32)
    assert GLOBAL_GEOMETRY.axis_aligned
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

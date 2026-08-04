"""Axis-aligned raster geometry and semantic resampling for polar windows."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import from_bounds


class RasterError(ValueError):
    """A polar raster does not meet the fixed map contract."""


@dataclass(frozen=True)
class GridGeometry:
    """A square, map-axis-aligned grid expressed in metres."""

    size_m: float
    resolution_m: float
    cells: int

    def __post_init__(self) -> None:
        if self.size_m <= 0 or self.resolution_m <= 0 or self.cells <= 0:
            raise RasterError("grid geometry values must be positive")
        if not math.isclose(self.size_m / self.resolution_m, self.cells, rel_tol=0.0, abs_tol=1e-9):
            raise RasterError("grid size / resolution must be an integer cell count")

    @property
    def axis_aligned(self) -> bool:
        return True


GLOBAL_GEOMETRY = GridGeometry(size_m=1024.0, resolution_m=4.0, cells=256)
LOCAL_GEOMETRY = GridGeometry(size_m=8.0, resolution_m=0.25, cells=32)


def _require_sha(value: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise RasterError("window_sha256 must be exactly 64 lowercase hexadecimal characters")


@dataclass(frozen=True)
class MapCanvas:
    """The fixed map-axis canvas positioned around one mission ROI bbox."""

    window_sha256: str
    bounds_m: tuple[float, float, float, float]
    geometry: GridGeometry = GLOBAL_GEOMETRY

    def __post_init__(self) -> None:
        _require_sha(self.window_sha256)
        left, bottom, right, top = self.bounds_m
        if not math.isclose(right - left, self.geometry.size_m) or not math.isclose(top - bottom, self.geometry.size_m):
            raise RasterError("canvas bounds must match fixed geometry")

    @classmethod
    def from_roi_bounds(cls, window_sha256: str, roi_bounds_m: tuple[float, float, float, float]) -> "MapCanvas":
        left, bottom, right, top = roi_bounds_m
        if left >= right or bottom >= top:
            raise RasterError("ROI bounds are invalid")
        center_x, center_y = (left + right) / 2.0, (bottom + top) / 2.0
        half = GLOBAL_GEOMETRY.size_m / 2.0
        return cls(window_sha256, (center_x - half, center_y - half, center_x + half, center_y + half))

    @property
    def identity(self) -> str:
        return sha256(repr((self.window_sha256, self.bounds_m, self.geometry)).encode("utf-8")).hexdigest()

    def world_to_grid(self, x_m: float, y_m: float) -> tuple[int, int]:
        left, bottom, right, top = self.bounds_m
        if not (left <= x_m < right and bottom < y_m <= top):
            raise RasterError("world point lies outside canvas")
        return int(math.floor((top - y_m) / self.geometry.resolution_m)), int(math.floor((x_m - left) / self.geometry.resolution_m))

    def grid_center_world(self, row: int, column: int) -> tuple[float, float]:
        if not (0 <= row < self.geometry.cells and 0 <= column < self.geometry.cells):
            raise RasterError("grid index lies outside canvas")
        left, _, _, top = self.bounds_m
        return left + (column + 0.5) * self.geometry.resolution_m, top - (row + 0.5) * self.geometry.resolution_m


@dataclass(frozen=True)
class LoadedPolarWindow:
    """A source-bound global DEM view; padding and NoData are unobserved."""

    canvas: MapCanvas
    elevation_m: np.ndarray
    observed_mask: np.ndarray
    ratio: np.ndarray | None = None


def _read(dataset: rasterio.io.DatasetReader, canvas: MapCanvas, resampling: Resampling, fill: float) -> np.ndarray:
    left, bottom, right, top = canvas.bounds_m
    return dataset.read(1, window=from_bounds(left, bottom, right, top, dataset.transform), out_shape=(canvas.geometry.cells, canvas.geometry.cells), boundless=True, fill_value=fill, resampling=resampling).astype(np.float32)


def load_polar_window(elevation_path: str | Path, canvas: MapCanvas, *, valid_mask_path: str | Path | None = None, ratio_path: str | Path | None = None) -> LoadedPolarWindow:
    """Read actual north-up source data into the absolute fixed canvas."""
    with rasterio.open(elevation_path) as elevation_source:
        if elevation_source.transform.b != 0.0 or elevation_source.transform.d != 0.0 or elevation_source.transform.e >= 0.0:
            raise RasterError("source raster must use north-up transform")
        elevation = _read(elevation_source, canvas, Resampling.bilinear, np.nan)
        nodata = elevation_source.nodata
    observed = np.isfinite(elevation)
    if nodata is not None:
        observed &= ~np.isclose(elevation, nodata)
    if valid_mask_path is not None:
        with rasterio.open(valid_mask_path) as mask_source:
            observed &= _read(mask_source, canvas, Resampling.nearest, 0.0).astype(bool)
    ratio = None
    if ratio_path is not None:
        with rasterio.open(ratio_path) as ratio_source:
            ratio = _read(ratio_source, canvas, Resampling.average, 0.0)
            if not np.isfinite(ratio).all() or ((ratio < 0.0) | (ratio > 1.0)).any():
                raise RasterError("ratio source must produce finite [0,1] values")
    return LoadedPolarWindow(canvas, np.where(observed, elevation, np.nan).astype(np.float32), observed, ratio)


@dataclass(frozen=True)
class WorldTruth:
    """Complete DEM truth retained by scene generation, never the network builder."""

    window_sha256: str
    elevation_m: np.ndarray

    def __post_init__(self) -> None:
        elevation = np.asarray(self.elevation_m, dtype=np.float32)
        if elevation.shape != (GLOBAL_GEOMETRY.cells, GLOBAL_GEOMETRY.cells) or not np.isfinite(elevation).all():
            raise RasterError("WorldTruth elevation must be finite [256,256]")
        _require_sha(self.window_sha256)
        object.__setattr__(self, "elevation_m", elevation)


def validate_roi_bounds(bounds_m: tuple[float, float, float, float], geometry: GridGeometry = GLOBAL_GEOMETRY) -> None:
    """Reject a requested map ROI that is outside its fixed square canvas."""
    left, bottom, right, top = bounds_m
    if left < 0.0 or bottom < 0.0 or right > geometry.size_m or top > geometry.size_m or left >= right or bottom >= top:
        raise RasterError("ROI lies outside fixed map canvas")


def _validate_input(values: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] < 1:
        raise RasterError("raster must be a non-empty two-dimensional array")
    if len(output_shape) != 2 or min(output_shape) < 1:
        raise RasterError("output shape must be two positive dimensions")
    return array


def resample_bilinear(values: np.ndarray, output_shape: tuple[int, int], *, observed_mask: np.ndarray | None = None) -> np.ndarray:
    """Read continuous elevation at output-cell centres with bilinear sampling.

    NoData is never permitted in a cell labelled observed.  Unknown samples are
    represented as zero here; their observed mask remains authoritative.
    """
    array = _validate_input(values, output_shape).astype(np.float32, copy=False)
    if observed_mask is not None:
        mask = np.asarray(observed_mask, dtype=bool)
        if mask.shape != array.shape:
            raise RasterError("observed mask shape must match raster")
        if np.isnan(array[mask]).any():
            raise RasterError("NoData cannot be marked observed")
    finite = np.where(np.isfinite(array), array, 0.0)
    source_rows, source_columns = array.shape
    output_rows, output_columns = output_shape
    y = (np.arange(output_rows, dtype=np.float64) + 0.5) * source_rows / output_rows - 0.5
    x = (np.arange(output_columns, dtype=np.float64) + 0.5) * source_columns / output_columns - 0.5
    y = np.clip(y, 0.0, source_rows - 1.0)
    x = np.clip(x, 0.0, source_columns - 1.0)
    y0 = np.floor(y).astype(np.intp)
    x0 = np.floor(x).astype(np.intp)
    y1 = np.minimum(y0 + 1, source_rows - 1)
    x1 = np.minimum(x0 + 1, source_columns - 1)
    wy = (y - y0).astype(np.float32)[:, None]
    wx = (x - x0).astype(np.float32)[None, :]
    top = finite[y0[:, None], x0[None, :]] * (1.0 - wx) + finite[y0[:, None], x1[None, :]] * wx
    bottom = finite[y1[:, None], x0[None, :]] * (1.0 - wx) + finite[y1[:, None], x1[None, :]] * wx
    return (top * (1.0 - wy) + bottom * wy).astype(np.float32)


def resample_nearest(values: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    """Read categorical masks and labels by containing source-cell lookup."""
    array = _validate_input(values, output_shape)
    source_rows, source_columns = array.shape
    output_rows, output_columns = output_shape
    rows = np.minimum(((np.arange(output_rows) + 0.5) * source_rows / output_rows).astype(np.intp), source_rows - 1)
    columns = np.minimum(((np.arange(output_columns) + 0.5) * source_columns / output_columns).astype(np.intp), source_columns - 1)
    return array[rows[:, None], columns[None, :]].copy()


def resample_average(values: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    """Area-average a ratio raster, including non-integral scale changes."""
    array = _validate_input(values, output_shape).astype(np.float32, copy=False)
    if not np.isfinite(array).all():
        raise RasterError("ratio raster must be finite")
    source_rows, source_columns = array.shape
    output_rows, output_columns = output_shape
    result = np.empty(output_shape, dtype=np.float32)
    for target_row in range(output_rows):
        y0, y1 = target_row * source_rows / output_rows, (target_row + 1) * source_rows / output_rows
        row_start, row_end = math.floor(y0), math.ceil(y1)
        for target_column in range(output_columns):
            x0, x1 = target_column * source_columns / output_columns, (target_column + 1) * source_columns / output_columns
            total = 0.0
            for source_row in range(row_start, row_end):
                row_overlap = max(0.0, min(y1, source_row + 1.0) - max(y0, source_row))
                if row_overlap == 0.0:
                    continue
                for source_column in range(math.floor(x0), math.ceil(x1)):
                    column_overlap = max(0.0, min(x1, source_column + 1.0) - max(x0, source_column))
                    if column_overlap:
                        total += float(array[source_row, source_column]) * row_overlap * column_overlap
            result[target_row, target_column] = total / ((y1 - y0) * (x1 - x0))
    return result


__all__ = [
    "GLOBAL_GEOMETRY",
    "LOCAL_GEOMETRY",
    "GridGeometry",
    "LoadedPolarWindow",
    "MapCanvas",
    "RasterError",
    "WorldTruth",
    "load_polar_window",
    "resample_average",
    "resample_bilinear",
    "resample_nearest",
    "validate_roi_bounds",
]

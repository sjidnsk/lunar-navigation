"""Explicit native visibility boundary plus a test-only Python reference."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol

import numpy as np

from lunar_planner_training_bridge import VisibilityKernel


@dataclass(frozen=True, slots=True)
class SensorGeometry:
    range_m: float
    fov_rad: float

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.range_m)
            or self.range_m <= 0.0
            or not math.isfinite(self.fov_rad)
            or not 0.0 < self.fov_rad <= 2.0 * math.pi
        ):
            raise ValueError("sensor range/FOV are invalid")

    @property
    def anchor_spacing_m(self) -> float:
        return max(4.0, self.range_m / 8.0)

    @property
    def standoff_m(self) -> float:
        return max(4.0, self.range_m / 16.0)

    @property
    def is_full_circle(self) -> bool:
        return self.fov_rad == 2.0 * math.pi


class VisibilityEstimator(Protocol):
    sensor: SensorGeometry
    resolution_m: float

    def estimate_candidate_gains(
        self,
        observed_mask: np.ndarray,
        obstacle_ratio: np.ndarray,
        roi_ratio: np.ndarray,
        priority_weight: np.ndarray,
        candidate_cells: np.ndarray,
    ) -> np.ndarray: ...

    def reveal_from_pose(
        self,
        truth_obstacle_ratio: np.ndarray,
        pose_cell: tuple[int, int],
    ) -> np.ndarray: ...


def _require_full_circle(sensor: SensorGeometry) -> None:
    if not sensor.is_full_circle:
        raise ValueError("native visibility requires the approved 360-degree FOV")


def _require_resolution(value: float) -> float:
    resolution = float(value)
    if not math.isfinite(resolution) or resolution <= 0.0:
        raise ValueError("visibility resolution must be finite and positive")
    return resolution


def _exact_grid(
    name: str,
    values: np.ndarray,
    dtype: np.dtype,
    shape: tuple[int, int] | None = None,
) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype != dtype:
        raise TypeError(f"{name} dtype must be {dtype.name}")
    if array.ndim != 2 or min(array.shape) <= 0:
        raise ValueError(f"{name} must be a non-empty 2D grid")
    if not array.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} shape mismatch")
    if np.issubdtype(dtype, np.floating) and not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def _candidate_array(values: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    candidates = np.asarray(values)
    if candidates.dtype != np.dtype(np.int32):
        raise TypeError("candidate cells dtype must be int32")
    if (
        candidates.ndim != 2
        or candidates.shape[1:] != (2,)
        or not candidates.flags.c_contiguous
    ):
        raise ValueError("candidate cells must be C-contiguous [N,2]")
    if candidates.size and (
        (candidates[:, 0] < 0).any()
        or (candidates[:, 1] < 0).any()
        or (candidates[:, 0] >= shape[0]).any()
        or (candidates[:, 1] >= shape[1]).any()
    ):
        raise ValueError("candidate cell is outside the grid")
    return candidates


def _validated_inputs(
    observed_mask: np.ndarray,
    obstacle_ratio: np.ndarray,
    roi_ratio: np.ndarray,
    priority_weight: np.ndarray,
    candidate_cells: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    observed = _exact_grid(
        "observed mask", observed_mask, np.dtype(np.bool_)
    )
    shape = observed.shape
    obstacle = _exact_grid(
        "obstacle ratio", obstacle_ratio, np.dtype(np.float32), shape
    )
    roi = _exact_grid("ROI ratio", roi_ratio, np.dtype(np.float32), shape)
    priority = _exact_grid(
        "priority weight", priority_weight, np.dtype(np.float32), shape
    )
    if (
        (obstacle < 0.0).any()
        or (roi < 0.0).any()
        or (priority < 0.0).any()
    ):
        raise ValueError("visibility ratios and weights must be non-negative")
    return observed, obstacle, roi, priority, _candidate_array(
        candidate_cells, shape
    )


class NativeVisibilityEstimator:
    """One Release C++ kernel bound to one map resolution and sensor geometry."""

    def __init__(self, sensor: SensorGeometry, *, resolution_m: float) -> None:
        if not isinstance(sensor, SensorGeometry):
            raise TypeError("sensor must use SensorGeometry")
        _require_full_circle(sensor)
        self.sensor = sensor
        self.resolution_m = _require_resolution(resolution_m)
        self._kernel = VisibilityKernel(self.resolution_m, sensor.range_m)

    def estimate_candidate_gains(
        self,
        observed_mask: np.ndarray,
        obstacle_ratio: np.ndarray,
        roi_ratio: np.ndarray,
        priority_weight: np.ndarray,
        candidate_cells: np.ndarray,
    ) -> np.ndarray:
        observed, obstacle, roi, priority, candidates = _validated_inputs(
            observed_mask,
            obstacle_ratio,
            roi_ratio,
            priority_weight,
            candidate_cells,
        )
        gains = self._kernel.estimate_candidate_gains(
            observed, obstacle, roi, priority, candidates
        )
        if (
            gains.shape != (candidates.shape[0], 2)
            or gains.dtype != np.dtype(np.float32)
            or not gains.flags.c_contiguous
            or not np.isfinite(gains).all()
        ):
            raise RuntimeError("native candidate visibility result is invalid")
        return gains

    def reveal_from_pose(
        self,
        truth_obstacle_ratio: np.ndarray,
        pose_cell: tuple[int, int],
    ) -> np.ndarray:
        truth = _exact_grid(
            "truth obstacle ratio",
            truth_obstacle_ratio,
            np.dtype(np.float32),
        )
        if (truth < 0.0).any():
            raise ValueError("truth obstacle ratio must be non-negative")
        row, column = _pose_cell(pose_cell, truth.shape)
        visible = self._kernel.reveal_from_pose(truth, row, column)
        if (
            visible.shape != truth.shape
            or visible.dtype != np.dtype(np.bool_)
            or not visible.flags.c_contiguous
        ):
            raise RuntimeError("native reveal result is invalid")
        return visible


def _pose_cell(
    value: tuple[int, int], shape: tuple[int, int]
) -> tuple[int, int]:
    if (
        not isinstance(value, tuple)
        or len(value) != 2
        or any(type(item) is not int for item in value)
    ):
        raise TypeError("pose cell must be a pair of integers")
    row, column = value
    if not (0 <= row < shape[0] and 0 <= column < shape[1]):
        raise ValueError("pose cell is outside the grid")
    return row, column


def _ray_cells(
    start: tuple[int, int], end: tuple[int, int]
) -> list[tuple[int, int]]:
    row, column = start
    end_row, end_column = end
    delta_column = abs(end_column - column)
    delta_row = abs(end_row - row)
    step_column = 1 if column < end_column else -1
    step_row = 1 if row < end_row else -1
    error = delta_column - delta_row
    result: list[tuple[int, int]] = []
    while True:
        result.append((row, column))
        if (row, column) == (end_row, end_column):
            return result
        doubled = 2 * error
        if doubled > -delta_row:
            error -= delta_row
            column += step_column
        if doubled < delta_column:
            error += delta_column
            row += step_row


class SlowVisibilityReference:
    """Readable Bresenham oracle; construction requires explicit test intent."""

    def __init__(
        self,
        sensor: SensorGeometry,
        *,
        resolution_m: float,
        test_only: bool = False,
    ) -> None:
        if not test_only:
            raise ValueError("slow visibility is test-only")
        if not isinstance(sensor, SensorGeometry):
            raise TypeError("sensor must use SensorGeometry")
        _require_full_circle(sensor)
        self.sensor = sensor
        self.resolution_m = _require_resolution(resolution_m)
        radius = math.floor(sensor.range_m / self.resolution_m)
        self._offsets = tuple(
            (row, column)
            for row in range(-radius, radius + 1)
            for column in range(-radius, radius + 1)
            if (row != 0 or column != 0)
            and math.hypot(row, column) * self.resolution_m
            <= sensor.range_m
        )

    def estimate_candidate_gains(
        self,
        observed_mask: np.ndarray,
        obstacle_ratio: np.ndarray,
        roi_ratio: np.ndarray,
        priority_weight: np.ndarray,
        candidate_cells: np.ndarray,
    ) -> np.ndarray:
        observed, obstacle, roi, priority, candidates = _validated_inputs(
            observed_mask,
            obstacle_ratio,
            roi_ratio,
            priority_weight,
            candidate_cells,
        )
        gains = np.zeros((candidates.shape[0], 2), dtype=np.float32)
        rows, columns = observed.shape
        for candidate_index, (candidate_row, candidate_column) in enumerate(
            candidates.tolist()
        ):
            roi_gain = 0.0
            priority_gain = 0.0
            for offset_row, offset_column in self._offsets:
                endpoint = (
                    candidate_row + offset_row,
                    candidate_column + offset_column,
                )
                if not (0 <= endpoint[0] < rows and 0 <= endpoint[1] < columns):
                    continue
                if observed[endpoint]:
                    continue
                path = _ray_cells(
                    (candidate_row, candidate_column), endpoint
                )
                if all(
                    observed[cell] and obstacle[cell] == 0.0
                    for cell in path[:-1]
                ):
                    roi_gain += float(roi[endpoint])
                    priority_gain += float(priority[endpoint])
            gains[candidate_index] = roi_gain, priority_gain
        return gains

    def reveal_from_pose(
        self,
        truth_obstacle_ratio: np.ndarray,
        pose_cell: tuple[int, int],
    ) -> np.ndarray:
        truth = _exact_grid(
            "truth obstacle ratio",
            truth_obstacle_ratio,
            np.dtype(np.float32),
        )
        if (truth < 0.0).any():
            raise ValueError("truth obstacle ratio must be non-negative")
        pose = _pose_cell(pose_cell, truth.shape)
        visible = np.zeros(truth.shape, dtype=np.bool_)
        visible[pose] = True
        rows, columns = truth.shape
        for offset_row, offset_column in self._offsets:
            endpoint = pose[0] + offset_row, pose[1] + offset_column
            if not (0 <= endpoint[0] < rows and 0 <= endpoint[1] < columns):
                continue
            for cell in _ray_cells(pose, endpoint):
                visible[cell] = True
                if truth[cell] > 0.0:
                    break
        return visible


__all__ = [
    "NativeVisibilityEstimator",
    "SensorGeometry",
    "SlowVisibilityReference",
    "VisibilityEstimator",
]

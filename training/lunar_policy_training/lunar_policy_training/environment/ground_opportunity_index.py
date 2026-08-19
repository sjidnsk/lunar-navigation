"""Observed-only, bounded-work opportunity indexing for ground platforms.

The normal frontier path must never promote an empty candidate batch into a
whole-task scan.  This module records the exact observed-only witnesses needed
for that fallback as deterministic, resumable slices instead.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
import math

import numpy as np


_DETAIL_CELL_COARSE_EQUIVALENT = float(np.float32((0.2 / 4.0) ** 2))


def ground_potential_gain_mask(
    unknown_roi: np.ndarray,
    *,
    sensor_range_m: float,
    resolution_m: float,
) -> np.ndarray:
    """Return the exact square prefilter used before detail visibility.

    This uses only the observed task mask and sensor geometry.  It has no
    access to terrain truth, platform capability, planner state, or coverage
    denominator.
    """
    unknown = np.asarray(unknown_roi)
    if (
        unknown.dtype != np.dtype(np.bool_)
        or unknown.ndim != 2
        or min(unknown.shape) <= 0
        or not unknown.flags.c_contiguous
    ):
        raise ValueError("unknown ROI mask is invalid")
    if (
        not math.isfinite(sensor_range_m)
        or sensor_range_m <= 0.0
        or not math.isfinite(resolution_m)
        or resolution_m <= 0.0
    ):
        raise ValueError("ground gain prefilter geometry is invalid")
    if not unknown.any():
        return np.zeros_like(unknown)
    rows, columns = unknown.shape
    radius = math.ceil(sensor_range_m / resolution_m)
    if radius >= max(rows - 1, columns - 1):
        return np.ones_like(unknown)
    padded = np.pad(unknown.astype(np.int32), ((1, 0), (1, 0)))
    integral = padded.cumsum(axis=0, dtype=np.int64).cumsum(
        axis=1, dtype=np.int64
    )
    row_low = np.maximum(np.arange(rows) - radius, 0)
    row_high = np.minimum(np.arange(rows) + radius + 1, rows)
    column_low = np.maximum(np.arange(columns) - radius, 0)
    column_high = np.minimum(np.arange(columns) + radius + 1, columns)
    sums = (
        integral[row_high[:, None], column_high[None, :]]
        - integral[row_low[:, None], column_high[None, :]]
        - integral[row_high[:, None], column_low[None, :]]
        + integral[row_low[:, None], column_low[None, :]]
    )
    return np.ascontiguousarray(sums > 0, dtype=np.bool_)


def _mask_sha256(mask: np.ndarray) -> str:
    return sha256(np.ascontiguousarray(mask, dtype=np.bool_).tobytes()).hexdigest()


def _validate_mask(name: str, value: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    mask = np.asarray(value)
    if (
        mask.dtype != np.dtype(np.bool_)
        or mask.shape != shape
        or not mask.flags.c_contiguous
    ):
        raise ValueError(f"{name} mask is invalid")
    return mask


def _validate_generation(value: int, *, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _tuple_cells(cells: np.ndarray) -> tuple[tuple[int, int], ...]:
    array = np.ascontiguousarray(cells, dtype=np.int32).reshape((-1, 2))
    result = tuple((int(row), int(column)) for row, column in array)
    if result != tuple(sorted(result)) or len(result) != len(set(result)):
        raise ValueError("opportunity pose cells must be unique and sorted")
    return result


@dataclass(frozen=True, slots=True)
class GroundOpportunityIndexState:
    """Serializable episode state for an observed-only ground index."""

    observation_generation: int
    physical_snapshot_id: str
    indexed_generation: int
    positive_pose_cells: tuple[tuple[int, int], ...]
    positive_gain_pairs: tuple[tuple[float, float], ...]
    reachable_mask_sha256: str
    observed_mask_sha256: str
    pending_pose_cells: tuple[tuple[int, int], ...]
    next_pending_pose_offset: int

    def __post_init__(self) -> None:
        generation = _validate_generation(
            self.observation_generation, name="observation generation"
        )
        if (
            type(self.indexed_generation) is not int
            or not -1 <= self.indexed_generation <= generation
        ):
            raise ValueError("indexed generation is invalid")
        if (
            not isinstance(self.physical_snapshot_id, str)
            or len(self.physical_snapshot_id) != 64
            or not isinstance(self.reachable_mask_sha256, str)
            or len(self.reachable_mask_sha256) != 64
            or not isinstance(self.observed_mask_sha256, str)
            or len(self.observed_mask_sha256) != 64
        ):
            raise ValueError("ground opportunity index identity is invalid")
        if (
            len(self.positive_pose_cells) != len(self.positive_gain_pairs)
            or tuple(sorted(self.positive_pose_cells)) != self.positive_pose_cells
            or len(set(self.positive_pose_cells)) != len(self.positive_pose_cells)
            or tuple(sorted(self.pending_pose_cells)) != self.pending_pose_cells
            or len(set(self.pending_pose_cells)) != len(self.pending_pose_cells)
        ):
            raise ValueError("ground opportunity index cells are invalid")
        if any(
            len(pair) != 2
            or not all(math.isfinite(float(value)) and float(value) >= 0.0 for value in pair)
            for pair in self.positive_gain_pairs
        ):
            raise ValueError("ground opportunity gains are invalid")
        if (
            type(self.next_pending_pose_offset) is not int
            or not 0 <= self.next_pending_pose_offset <= len(self.pending_pose_cells)
        ):
            raise ValueError("ground opportunity pending offset is invalid")
        if self.next_pending_pose_offset == len(self.pending_pose_cells):
            if self.indexed_generation != generation:
                raise ValueError("completed ground opportunity index has stale generation")
        elif self.indexed_generation >= generation:
            raise ValueError("pending ground opportunity index cannot be current")

    @property
    def pending_pose_count(self) -> int:
        return len(self.pending_pose_cells) - self.next_pending_pose_offset

    def is_current(
        self, *, observation_generation: int, physical_snapshot_id: str
    ) -> bool:
        return (
            self.indexed_generation == observation_generation
            and self.observation_generation == observation_generation
            and self.physical_snapshot_id == physical_snapshot_id
            and self.pending_pose_count == 0
        )


@dataclass(frozen=True, slots=True)
class GroundOpportunityIndexUpdate:
    """One bounded update slice and its resulting serializable state."""

    state: GroundOpportunityIndexState
    endpoint_certified_pose_count: int
    exact_gain_evaluated_pose_count: int
    pending_pose_count: int

    def __post_init__(self) -> None:
        counts = (
            self.endpoint_certified_pose_count,
            self.exact_gain_evaluated_pose_count,
            self.pending_pose_count,
        )
        if (
            any(type(value) is not int or value < 0 for value in counts)
            or self.exact_gain_evaluated_pose_count
            != self.endpoint_certified_pose_count
            or self.pending_pose_count != self.state.pending_pose_count
        ):
            raise ValueError("ground opportunity index update is invalid")


class GroundOpportunityIndex:
    """Evaluate eligible observed-only poses in deterministic bounded slices."""

    def __init__(
        self,
        *,
        task_roi_ratio: np.ndarray,
        sensor_range_m: float,
        resolution_m: float,
        positive_gain_threshold: float = _DETAIL_CELL_COARSE_EQUIVALENT,
    ) -> None:
        roi = np.asarray(task_roi_ratio)
        if (
            roi.dtype != np.dtype(np.float32)
            or roi.ndim != 2
            or min(roi.shape) <= 0
            or not roi.flags.c_contiguous
            or not np.isfinite(roi).all()
            or (roi < 0.0).any()
            or (roi > 1.0).any()
        ):
            raise ValueError("ground opportunity task ROI is invalid")
        if (
            not math.isfinite(sensor_range_m)
            or sensor_range_m <= 0.0
            or not math.isfinite(resolution_m)
            or resolution_m <= 0.0
            or not math.isfinite(positive_gain_threshold)
            or positive_gain_threshold < 0.0
        ):
            raise ValueError("ground opportunity index geometry is invalid")
        self._task_roi_ratio = roi
        self._sensor_range_m = float(sensor_range_m)
        self._resolution_m = float(resolution_m)
        self._positive_gain_threshold = float(positive_gain_threshold)

    @property
    def shape(self) -> tuple[int, int]:
        return self._task_roi_ratio.shape

    def update(
        self,
        *,
        previous: GroundOpportunityIndexState | None,
        observed_mask: np.ndarray,
        reachable_pose_mask: np.ndarray,
        observation_positions_m: np.ndarray,
        dirty_observed_mask: np.ndarray,
        reachable_delta_mask: np.ndarray,
        endpoint_feasibility: Callable[[np.ndarray], np.ndarray],
        estimate_gains: Callable[[np.ndarray, np.ndarray], np.ndarray],
        observation_generation: int,
        physical_snapshot_id: str,
        max_pose_count: int,
    ) -> GroundOpportunityIndexUpdate:
        """Advance one deterministic pose slice without truth-dependent input."""
        generation = _validate_generation(
            observation_generation, name="observation generation"
        )
        if (
            not isinstance(physical_snapshot_id, str)
            or len(physical_snapshot_id) != 64
            or type(max_pose_count) is not int
            or max_pose_count <= 0
            or not callable(endpoint_feasibility)
            or not callable(estimate_gains)
        ):
            raise ValueError("ground opportunity index update arguments are invalid")
        observed = _validate_mask("observed", observed_mask, self.shape)
        reachable = _validate_mask("reachable", reachable_pose_mask, self.shape)
        dirty = _validate_mask("dirty observed", dirty_observed_mask, self.shape)
        reachable_delta = _validate_mask(
            "reachable delta", reachable_delta_mask, self.shape
        )
        reachable_count = int(reachable.sum(dtype=np.int64))
        positions = np.asarray(observation_positions_m)
        if (
            positions.dtype != np.dtype(np.float64)
            or positions.shape != (reachable_count, 3)
            or not positions.flags.c_contiguous
            or not np.isfinite(positions).all()
        ):
            raise ValueError("ground opportunity observation positions are invalid")

        observed_sha256 = _mask_sha256(observed)
        reachable_sha256 = _mask_sha256(reachable)
        residual = np.ascontiguousarray(
            (self._task_roi_ratio > 0.0) & ~observed, dtype=np.bool_
        )
        potential = ground_potential_gain_mask(
            residual,
            sensor_range_m=self._sensor_range_m,
            resolution_m=self._resolution_m,
        )
        eligible = np.ascontiguousarray(reachable & potential, dtype=np.bool_)

        continuation = (
            previous is not None
            and previous.observation_generation == generation
            and previous.physical_snapshot_id == physical_snapshot_id
            and previous.observed_mask_sha256 == observed_sha256
            and previous.reachable_mask_sha256 == reachable_sha256
        )
        if continuation:
            assert previous is not None
            work_cells = previous.pending_pose_cells
            offset = previous.next_pending_pose_offset
            positive = dict(
                zip(
                    previous.positive_pose_cells,
                    previous.positive_gain_pairs,
                    strict=True,
                )
            )
        else:
            offset = 0
            if previous is None or previous.pending_pose_count:
                positive = {}
                work_cells = _tuple_cells(np.argwhere(eligible))
            else:
                positive = {
                    cell: gain
                    for cell, gain in zip(
                        previous.positive_pose_cells,
                        previous.positive_gain_pairs,
                        strict=True,
                    )
                    if eligible[cell]
                }
                dirty_affected = ground_potential_gain_mask(
                    dirty,
                    sensor_range_m=self._sensor_range_m,
                    resolution_m=self._resolution_m,
                )
                affected = np.ascontiguousarray(
                    eligible & (dirty_affected | reachable_delta), dtype=np.bool_
                )
                work_cells = _tuple_cells(np.argwhere(affected))
                for cell in work_cells:
                    positive.pop(cell, None)

        slice_cells = work_cells[offset : offset + max_pose_count]
        offset += len(slice_cells)
        cell_array = np.ascontiguousarray(slice_cells, dtype=np.int32).reshape((-1, 2))
        endpoint_positions = self._positions_for_cells(
            reachable=reachable,
            observation_positions_m=positions,
            pose_cells=cell_array,
        )
        if len(cell_array):
            endpoint_mask = np.asarray(endpoint_feasibility(endpoint_positions))
            if (
                endpoint_mask.dtype != np.dtype(np.bool_)
                or endpoint_mask.shape != (len(cell_array),)
                or not endpoint_mask.flags.c_contiguous
            ):
                raise ValueError("ground opportunity endpoint feasibility is invalid")
        else:
            endpoint_mask = np.zeros(0, dtype=np.bool_)
        endpoint_cells = np.ascontiguousarray(cell_array[endpoint_mask], dtype=np.int32)
        endpoint_positions = np.ascontiguousarray(
            endpoint_positions[endpoint_mask], dtype=np.float64
        )
        if len(endpoint_cells):
            gains = np.asarray(estimate_gains(endpoint_cells, endpoint_positions))
            if (
                gains.dtype != np.dtype(np.float32)
                or gains.shape != (len(endpoint_cells), 2)
                or not gains.flags.c_contiguous
                or not np.isfinite(gains).all()
                or (gains < 0.0).any()
            ):
                raise ValueError("ground opportunity gains are invalid")
            for (row, column), gain in zip(endpoint_cells, gains, strict=True):
                if float(gain[0]) >= self._positive_gain_threshold:
                    positive[(int(row), int(column))] = (
                        float(gain[0]),
                        float(gain[1]),
                    )

        ordered_positive = tuple(sorted(positive.items()))
        complete = offset == len(work_cells)
        state = GroundOpportunityIndexState(
            observation_generation=generation,
            physical_snapshot_id=physical_snapshot_id,
            indexed_generation=generation if complete else generation - 1,
            positive_pose_cells=tuple(cell for cell, _ in ordered_positive),
            positive_gain_pairs=tuple(gain for _, gain in ordered_positive),
            reachable_mask_sha256=reachable_sha256,
            observed_mask_sha256=observed_sha256,
            pending_pose_cells=work_cells,
            next_pending_pose_offset=offset,
        )
        return GroundOpportunityIndexUpdate(
            state=state,
            endpoint_certified_pose_count=len(endpoint_cells),
            exact_gain_evaluated_pose_count=len(endpoint_cells),
            pending_pose_count=state.pending_pose_count,
        )

    def _positions_for_cells(
        self,
        *,
        reachable: np.ndarray,
        observation_positions_m: np.ndarray,
        pose_cells: np.ndarray,
    ) -> np.ndarray:
        if not len(pose_cells):
            return np.empty((0, 3), dtype=np.float64)
        rows, columns = self.shape
        reachable_cells = np.argwhere(reachable)
        reachable_codes = np.ascontiguousarray(
            reachable_cells[:, 0].astype(np.int64) * columns
            + reachable_cells[:, 1],
            dtype=np.int64,
        )
        pose_codes = np.ascontiguousarray(
            pose_cells[:, 0].astype(np.int64) * columns
            + pose_cells[:, 1],
            dtype=np.int64,
        )
        del rows
        indices = np.searchsorted(reachable_codes, pose_codes)
        if (
            (indices >= len(reachable_codes)).any()
            or not np.array_equal(reachable_codes[indices], pose_codes)
        ):
            raise ValueError("ground opportunity positions are not reachable")
        return np.ascontiguousarray(observation_positions_m[indices], dtype=np.float64)

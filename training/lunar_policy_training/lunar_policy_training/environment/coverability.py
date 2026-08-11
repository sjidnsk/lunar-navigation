"""Exact platform/start-specific exploration coverability contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
import json
import math
from typing import Callable, Iterator, Mapping

import numpy as np

from ..training_semantics import (
    FORMAL_MINIMUM_MISSION_COVERABLE_RATIO,
    FORMAL_SUCCESS_COVERAGE_RATIO,
)


_PLATFORM_TYPES = frozenset(("WHEELED", "LEGGED", "HOPPER"))
PHYSICAL_PROJECTION_SCHEMA = "lunar-physical-coverability-projection/v1"
PHYSICAL_GRID_AXIS_CONVENTION = (
    "north-up-row-major-row-decreases-y-column-increases-x"
)
_LINEAR_QUANTIZATION_PER_M = 1_000_000
_BYTE_POPCOUNT = np.asarray(
    [value.bit_count() for value in range(256)], dtype=np.uint8
)


class CoverabilityError(ValueError):
    """A coverability artifact is ambiguous, drifted, or internally inconsistent."""


class IneligibleReason(str, Enum):
    """The only ordinary task-ineligibility outcomes, in evaluation order."""

    UNSAFE_START = "UNSAFE_START"
    ZERO_MISSION_TARGET = "ZERO_MISSION_TARGET"
    MISSION_COVERABLE_BELOW_95 = "MISSION_COVERABLE_BELOW_95"
    INITIAL_ALREADY_SUCCESS = "INITIAL_ALREADY_SUCCESS"
    NO_INITIAL_CANDIDATE = "NO_INITIAL_CANDIDATE"


def _require_bool_mask(value: np.ndarray, name: str) -> np.ndarray:
    if (
        not isinstance(value, np.ndarray)
        or value.dtype != np.dtype(np.bool_)
        or value.ndim != 2
        or not value.flags.c_contiguous
        or any(dimension <= 0 for dimension in value.shape)
    ):
        raise CoverabilityError(
            f"{name} must be a non-empty C-contiguous boolean matrix"
        )
    return value


def _require_sha(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CoverabilityError(f"{name} must be a lowercase SHA-256 digest")
    return value


def mask_sha256(mask: np.ndarray) -> str:
    """Hash the semantic row-major boolean cells rather than packed padding."""
    checked = _require_bool_mask(mask, "mask")
    return sha256(checked.view(np.uint8).tobytes(order="C")).hexdigest()


def canonical_physical_positions_um(positions_m: np.ndarray) -> np.ndarray:
    """Return canonical row-major little-endian int64 micrometre positions."""
    positions = positions_m
    if (
        not isinstance(positions, np.ndarray)
        or positions.dtype != np.dtype(np.float64)
        or positions.ndim != 2
        or positions.shape[1:] != (3,)
        or not positions.flags.c_contiguous
        or not np.isfinite(positions).all()
    ):
        raise CoverabilityError(
            "physical observation positions must be row-major float64 [N,3]"
        )
    scaled = positions * _LINEAR_QUANTIZATION_PER_M
    int64_limit = float(np.iinfo(np.int64).max)
    if not np.isfinite(scaled).all() or (np.abs(scaled) > int64_limit).any():
        raise CoverabilityError("physical projection coordinate is out of range")
    return np.ascontiguousarray(np.rint(scaled), dtype="<i8")


def physical_projection_sha256(
    *,
    platform_type: str,
    physical_reachability_algorithm_id: str,
    physical_evidence_algorithm_id: str,
    physical_observation_pose_mask: np.ndarray,
    physical_observation_positions_m: np.ndarray,
    physical_grid_resolution_m: float,
    physical_grid_origin_m: tuple[float, float],
    physical_grid_world_bounds_m: tuple[float, float, float, float],
    physical_grid_axis_convention: str,
    capability_content_sha256: str,
    start_identity_sha256: str,
) -> str:
    """Hash only canonical physical authority, never planner primitives."""
    if platform_type not in _PLATFORM_TYPES:
        raise CoverabilityError("physical projection platform is invalid")
    for name, value in (
        (
            "physical reachability algorithm ID",
            physical_reachability_algorithm_id,
        ),
        ("physical evidence algorithm ID", physical_evidence_algorithm_id),
    ):
        if not isinstance(value, str) or not value:
            raise CoverabilityError(f"{name} is missing")
    mask = _require_bool_mask(
        physical_observation_pose_mask, "physical observation pose mask"
    )
    positions = physical_observation_positions_m
    if (
        not isinstance(positions, np.ndarray)
        or positions.dtype != np.dtype(np.float64)
        or positions.ndim != 2
        or positions.shape[1:] != (3,)
        or len(positions) != int(mask.sum(dtype=np.int64))
        or not positions.flags.c_contiguous
        or not np.isfinite(positions).all()
    ):
        raise CoverabilityError(
            "physical observation positions must be row-major float64 [N,3]"
        )
    if (
        not isinstance(physical_grid_resolution_m, float)
        or not math.isfinite(physical_grid_resolution_m)
        or physical_grid_resolution_m <= 0.0
        or physical_grid_axis_convention != PHYSICAL_GRID_AXIS_CONVENTION
    ):
        raise CoverabilityError("physical grid resolution or axis is invalid")
    geometry_values = (
        physical_grid_origin_m,
        physical_grid_world_bounds_m,
    )
    if (
        not isinstance(physical_grid_origin_m, tuple)
        or len(physical_grid_origin_m) != 2
        or not isinstance(physical_grid_world_bounds_m, tuple)
        or len(physical_grid_world_bounds_m) != 4
        or any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            for values in geometry_values
            for value in values
        )
    ):
        raise CoverabilityError("physical grid origin or bounds are invalid")
    left, bottom, right, top = tuple(
        float(value) for value in physical_grid_world_bounds_m
    )
    origin_x, origin_y = tuple(float(value) for value in physical_grid_origin_m)
    if (
        left >= right
        or bottom >= top
        or not math.isclose(origin_x, left, rel_tol=0.0, abs_tol=5.0e-7)
        or not math.isclose(origin_y, top, rel_tol=0.0, abs_tol=5.0e-7)
        or not math.isclose(
            right - left,
            mask.shape[1] * physical_grid_resolution_m,
            rel_tol=0.0,
            abs_tol=5.0e-7,
        )
        or not math.isclose(
            top - bottom,
            mask.shape[0] * physical_grid_resolution_m,
            rel_tol=0.0,
            abs_tol=5.0e-7,
        )
    ):
        raise CoverabilityError("physical grid geometry is inconsistent")

    def quantized(value: float) -> int:
        scaled = float(value) * _LINEAR_QUANTIZATION_PER_M
        if not math.isfinite(scaled) or abs(scaled) > np.iinfo(np.int64).max:
            raise CoverabilityError("physical projection coordinate is out of range")
        return int(round(scaled))

    quantized_positions = canonical_physical_positions_um(positions)
    capability_sha256 = _require_sha(
        capability_content_sha256, "capability content hash"
    )
    start_sha256 = _require_sha(start_identity_sha256, "start identity hash")
    body = {
        "platform_type": platform_type,
        "physical_reachability_algorithm_id": (
            physical_reachability_algorithm_id
        ),
        "physical_evidence_algorithm_id": physical_evidence_algorithm_id,
        "physical_grid_geometry": {
            "shape": list(mask.shape),
            "linear_quantization_um": 1,
            "resolution_um": quantized(physical_grid_resolution_m),
            "origin_um": [quantized(origin_x), quantized(origin_y)],
            "world_bounds_um": [
                quantized(left),
                quantized(bottom),
                quantized(right),
                quantized(top),
            ],
            "axis_convention": physical_grid_axis_convention,
        },
        "physical_observation_pose_mask_sha256": mask_sha256(mask),
        "physical_observation_positions": {
            "count": len(positions),
            "component_order": "x-y-z",
            "quantization_um": 1,
            "sha256": sha256(
                quantized_positions.tobytes(order="C")
            ).hexdigest(),
        },
        "capability_content_sha256": capability_sha256,
        "start_identity_sha256": start_sha256,
    }
    return sha256(
        json.dumps(
            body,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def build_mission_target_detail_mask(
    *,
    inside_mission_roi: np.ndarray,
    detail_valid: np.ndarray,
    forbidden_ratio: np.ndarray,
    physical_obstacle_ratio: np.ndarray,
    intrinsic_terrain_feasible: np.ndarray,
) -> np.ndarray:
    """Classify exact detail cells that may contribute to mission success."""
    roi = _require_bool_mask(inside_mission_roi, "detail mission ROI")
    valid = _require_bool_mask(detail_valid, "detail validity")
    intrinsic = _require_bool_mask(
        intrinsic_terrain_feasible, "intrinsic terrain feasibility"
    )
    for name, values in (
        ("detail forbidden ratio", forbidden_ratio),
        ("detail physical obstacle ratio", physical_obstacle_ratio),
    ):
        if (
            not isinstance(values, np.ndarray)
            or values.dtype != np.dtype(np.float32)
            or values.ndim != 2
            or not values.flags.c_contiguous
            or values.shape != roi.shape
            or not np.isfinite(values).all()
            or (values < 0.0).any()
        ):
            raise CoverabilityError(
                f"{name} must be a same-shape finite non-negative float32 matrix"
            )
    if valid.shape != roi.shape or intrinsic.shape != roi.shape:
        raise CoverabilityError("detail target inputs must have one exact shape")
    return np.ascontiguousarray(
        roi
        & valid
        & (forbidden_ratio == 0.0)
        & (physical_obstacle_ratio == 0.0)
        & intrinsic,
        dtype=np.bool_,
    )


def build_coverable_detail_mask(
    *,
    mission_target_detail_mask: np.ndarray,
    truth_obstacle_ratio: np.ndarray,
    reachable_pose_mask: np.ndarray,
    observation_pose_cells: np.ndarray | None = None,
    reveal_from_pose: Callable[[np.ndarray, tuple[int, int]], np.ndarray],
) -> np.ndarray:
    """Union detail visibility from exact graph poses or legacy coarse centers."""
    target = _require_bool_mask(
        mission_target_detail_mask, "mission target detail mask"
    )
    reachable = _require_bool_mask(reachable_pose_mask, "reachable pose mask")
    truth = truth_obstacle_ratio
    if (
        not isinstance(truth, np.ndarray)
        or truth.dtype != np.dtype(np.float32)
        or truth.shape != target.shape
        or not truth.flags.c_contiguous
        or not np.isfinite(truth).all()
        or (truth < 0.0).any()
    ):
        raise CoverabilityError(
            "truth obstacle ratio must be a same-shape finite non-negative "
            "float32 matrix"
        )
    if not callable(reveal_from_pose):
        raise CoverabilityError("detail reveal function is required")
    if (
        target.shape[0] % reachable.shape[0] != 0
        or target.shape[1] % reachable.shape[1] != 0
    ):
        raise CoverabilityError(
            "detail geometry must divide evenly into the reachable grid"
        )
    row_scale = target.shape[0] // reachable.shape[0]
    column_scale = target.shape[1] // reachable.shape[1]
    if observation_pose_cells is None:
        pose_cells = np.asarray(
            [
                (
                    int(row) * row_scale + row_scale // 2,
                    int(column) * column_scale + column_scale // 2,
                )
                for row, column in np.argwhere(reachable)
            ],
            dtype=np.int32,
        ).reshape((-1, 2))
    else:
        pose_cells = observation_pose_cells
        if (
            not isinstance(pose_cells, np.ndarray)
            or pose_cells.dtype != np.dtype(np.int32)
            or pose_cells.ndim != 2
            or pose_cells.shape[1:] != (2,)
            or not pose_cells.flags.c_contiguous
            or (
                pose_cells.size
                and (
                    (pose_cells < 0).any()
                    or (pose_cells[:, 0] >= target.shape[0]).any()
                    or (pose_cells[:, 1] >= target.shape[1]).any()
                )
            )
        ):
            raise CoverabilityError(
                "observation pose cells must be C-contiguous int32 [N,2]"
            )
    coverable = np.zeros(target.shape, dtype=np.bool_)
    for row, column in pose_cells:
        pose_cell = int(row), int(column)
        visible = reveal_from_pose(truth, pose_cell)
        if (
            not isinstance(visible, np.ndarray)
            or visible.dtype != np.dtype(np.bool_)
            or visible.shape != target.shape
            or not visible.flags.c_contiguous
        ):
            raise CoverabilityError(
                "detail reveal must return a same-shape C-contiguous boolean matrix"
            )
        coverable |= visible
    coverable &= target
    return np.ascontiguousarray(coverable, dtype=np.bool_)


def pack_detail_mask(mask: np.ndarray) -> np.ndarray:
    """Pack a two-dimensional boolean mask in row-major, MSB-first order."""
    checked = _require_bool_mask(mask, "detail mask")
    return np.ascontiguousarray(
        np.packbits(checked.reshape(-1), bitorder="big"),
        dtype=np.uint8,
    )


def _detail_shape(value: object) -> tuple[int, int]:
    if (
        not isinstance(value, tuple)
        or len(value) != 2
        or any(type(dimension) is not int or dimension <= 0 for dimension in value)
    ):
        raise CoverabilityError("detail mask shape must contain two positive integers")
    return value


def unpack_detail_mask(
    packed: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    """Unpack exact semantic bits and reject any non-zero padding bits."""
    checked_shape = _detail_shape(shape)
    cell_count = math.prod(checked_shape)
    byte_count = (cell_count + 7) // 8
    if (
        not isinstance(packed, np.ndarray)
        or packed.dtype != np.dtype(np.uint8)
        or packed.ndim != 1
        or not packed.flags.c_contiguous
        or packed.size != byte_count
    ):
        raise CoverabilityError("packed detail mask size or dtype is invalid")
    padding = byte_count * 8 - cell_count
    if padding and int(packed[-1]) & ((1 << padding) - 1):
        raise CoverabilityError("packed detail mask has non-zero padding bits")
    unpacked = np.unpackbits(packed, bitorder="big", count=cell_count)
    return np.ascontiguousarray(
        unpacked.reshape(checked_shape).astype(np.bool_, copy=False)
    )


def _packed_rows(packed: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    checked_shape = _detail_shape(shape)
    if checked_shape[1] % 8:
        raise CoverabilityError(
            "streamed detail mask width must be byte aligned"
        )
    expected = checked_shape[0] * (checked_shape[1] // 8)
    if (
        not isinstance(packed, np.ndarray)
        or packed.dtype != np.dtype(np.uint8)
        or packed.ndim != 1
        or not packed.flags.c_contiguous
        or packed.size != expected
    ):
        raise CoverabilityError("streamed packed detail mask is invalid")
    return packed.reshape(checked_shape[0], checked_shape[1] // 8)


def _write_aligned_tile(
    packed: np.ndarray,
    shape: tuple[int, int],
    *,
    start_row: int,
    start_column: int,
    mask: np.ndarray,
) -> None:
    checked = _require_bool_mask(mask, "streamed detail tile")
    if start_column % 8 or checked.shape[1] % 8:
        raise CoverabilityError("streamed detail tile must be byte aligned")
    rows = _packed_rows(packed, shape)
    end_row = start_row + checked.shape[0]
    end_column = start_column + checked.shape[1]
    if (
        start_row < 0
        or start_column < 0
        or end_row > shape[0]
        or end_column > shape[1]
    ):
        raise CoverabilityError("streamed detail tile lies outside the mask")
    rows[
        start_row:end_row,
        start_column // 8 : end_column // 8,
    ] = np.packbits(checked, axis=1, bitorder="big")


def _read_packed_window(
    packed: np.ndarray,
    shape: tuple[int, int],
    *,
    start_row: int,
    start_column: int,
    cells: int,
) -> np.ndarray:
    rows = _packed_rows(packed, shape)
    if (
        type(cells) is not int
        or cells < 1
        or start_row < 0
        or start_column < 0
        or start_row + cells > shape[0]
        or start_column + cells > shape[1]
    ):
        raise CoverabilityError("packed detail window lies outside the mask")
    byte_start = start_column // 8
    bit_offset = start_column % 8
    byte_end = (start_column + cells + 7) // 8
    unpacked = np.unpackbits(
        rows[start_row : start_row + cells, byte_start:byte_end],
        axis=1,
        bitorder="big",
    )
    return np.ascontiguousarray(
        unpacked[:, bit_offset : bit_offset + cells].astype(
            np.bool_, copy=False
        )
    )


def read_packed_detail_window(
    packed: np.ndarray,
    shape: tuple[int, int],
    *,
    start_row: int,
    start_column: int,
    cells: int,
) -> np.ndarray:
    """Read one square semantic window without unpacking the full detail map."""
    return _read_packed_window(
        packed,
        shape,
        start_row=start_row,
        start_column=start_column,
        cells=cells,
    )


def _or_packed_window(
    packed: np.ndarray,
    shape: tuple[int, int],
    *,
    start_row: int,
    start_column: int,
    mask: np.ndarray,
) -> None:
    checked = _require_bool_mask(mask, "streamed coverable window")
    cells = checked.shape[0]
    if checked.shape[1] != cells:
        raise CoverabilityError("streamed coverable window must be square")
    rows = _packed_rows(packed, shape)
    if (
        start_row < 0
        or start_column < 0
        or start_row + cells > shape[0]
        or start_column + cells > shape[1]
    ):
        raise CoverabilityError("streamed coverable window lies outside mask")
    byte_start = start_column // 8
    bit_offset = start_column % 8
    byte_end = (start_column + cells + 7) // 8
    storage_width = (byte_end - byte_start) * 8
    aligned = np.zeros((cells, storage_width), dtype=np.bool_)
    aligned[:, bit_offset : bit_offset + cells] = checked
    rows[
        start_row : start_row + cells,
        byte_start:byte_end,
    ] |= np.packbits(aligned, axis=1, bitorder="big")


def _streamed_mask_sha256(
    packed: np.ndarray, shape: tuple[int, int]
) -> str:
    rows = _packed_rows(packed, shape)
    digest = sha256()
    for row in rows:
        semantic = np.unpackbits(row, bitorder="big", count=shape[1])
        digest.update(semantic.tobytes(order="C"))
    return digest.hexdigest()


def _streamed_coarse_ratio(
    packed: np.ndarray,
    shape: tuple[int, int],
    coarse_shape: tuple[int, int],
) -> tuple[np.ndarray, int]:
    if (
        shape[0] % coarse_shape[0]
        or shape[1] % coarse_shape[1]
    ):
        raise CoverabilityError("detail mask does not divide into coarse cells")
    row_scale = shape[0] // coarse_shape[0]
    column_scale = shape[1] // coarse_shape[1]
    ratio = np.zeros(coarse_shape, dtype=np.float32)
    count = 0
    rows = _packed_rows(packed, shape)
    for coarse_row in range(coarse_shape[0]):
        start_row = coarse_row * row_scale
        detail = np.unpackbits(
            rows[start_row : start_row + row_scale],
            axis=1,
            bitorder="big",
            count=shape[1],
        ).astype(np.bool_, copy=False)
        grouped = detail.reshape(
            row_scale, coarse_shape[1], column_scale
        )
        counts = grouped.sum(axis=(0, 2), dtype=np.int64)
        ratio[coarse_row] = counts.astype(np.float64) / (
            row_scale * column_scale
        )
        count += int(counts.sum(dtype=np.int64))
    return np.ascontiguousarray(ratio), count


def _spatially_prioritized_sources(
    source_mask: np.ndarray,
    *,
    row_scale: int,
    column_scale: int,
    window_cells: int,
) -> Iterator[tuple[int, int]]:
    """Yield one central source per sensor-sized block before dense fallback."""
    sources = np.argwhere(source_mask)
    row_stride = max(1, window_cells // (2 * row_scale))
    column_stride = max(1, window_cells // (2 * column_scale))
    representatives: dict[
        tuple[int, int], tuple[tuple[int, int, int], tuple[int, int]]
    ] = {}
    for row_value, column_value in sources:
        row = int(row_value)
        column = int(column_value)
        bucket = (row // row_stride, column // column_stride)
        row_start = bucket[0] * row_stride
        column_start = bucket[1] * column_stride
        row_end = min(row_start + row_stride, source_mask.shape[0])
        column_end = min(
            column_start + column_stride, source_mask.shape[1]
        )
        distance = (
            (2 * row - (row_start + row_end - 1)) ** 2
            + (2 * column - (column_start + column_end - 1)) ** 2
        )
        candidate = ((distance, row, column), (row, column))
        current = representatives.get(bucket)
        if current is None or candidate[0] < current[0]:
            representatives[bucket] = candidate
    selected = {
        coordinate for _, coordinate in representatives.values()
    }
    for bucket in sorted(representatives):
        yield representatives[bucket][1]
    for row_value, column_value in sources:
        coordinate = (int(row_value), int(column_value))
        if coordinate not in selected:
            yield coordinate


def _prioritize_exact_detail_sources(
    sources: tuple[tuple[int, int], ...], *, window_cells: int
) -> Iterator[tuple[int, int]]:
    """Prefer one exact graph pose per half-sensor block, then all remainder."""
    stride = max(1, window_cells // 2)
    representatives: dict[
        tuple[int, int], tuple[tuple[int, int, int], tuple[int, int]]
    ] = {}
    for row, column in sources:
        bucket = row // stride, column // stride
        center_row = bucket[0] * stride + stride // 2
        center_column = bucket[1] * stride + stride // 2
        candidate = (
            (
                (row - center_row) ** 2 + (column - center_column) ** 2,
                row,
                column,
            ),
            (row, column),
        )
        current = representatives.get(bucket)
        if current is None or candidate[0] < current[0]:
            representatives[bucket] = candidate
    selected = {coordinate for _, coordinate in representatives.values()}
    for bucket in sorted(representatives):
        yield representatives[bucket][1]
    for coordinate in sources:
        if coordinate not in selected:
            yield coordinate


@dataclass(frozen=True, slots=True)
class StreamedDetailCoverability:
    """Bounded-memory exact target and coverable detail-mask build result."""

    detail_shape: tuple[int, int]
    mission_target_detail_bits: np.ndarray
    coverable_detail_bits: np.ndarray
    coverable_ratio: np.ndarray
    mission_target_detail_cell_count: int
    coverable_detail_cell_count: int
    mission_target_mask_sha256: str
    coverable_mask_sha256: str

    def __post_init__(self) -> None:
        shape = _detail_shape(self.detail_shape)
        _packed_rows(self.mission_target_detail_bits, shape)
        _packed_rows(self.coverable_detail_bits, shape)
        if (
            self.coverable_ratio.dtype != np.dtype(np.float32)
            or self.coverable_ratio.ndim != 2
            or not self.coverable_ratio.flags.c_contiguous
            or any(value <= 0 for value in self.coverable_ratio.shape)
            or not np.isfinite(self.coverable_ratio).all()
            or ((self.coverable_ratio < 0.0) | (self.coverable_ratio > 1.0)).any()
        ):
            raise CoverabilityError("streamed coverable ratio is invalid")
        if np.bitwise_and(
            self.coverable_detail_bits,
            np.bitwise_not(self.mission_target_detail_bits),
        ).any():
            raise CoverabilityError("coverable mask exceeds mission target mask")
        ratio, count = _streamed_coarse_ratio(
            self.coverable_detail_bits,
            shape,
            self.coverable_ratio.shape,
        )
        target_count = int(
            _BYTE_POPCOUNT[self.mission_target_detail_bits].sum(
                dtype=np.int64
            )
        )
        if (
            not np.array_equal(self.coverable_ratio, ratio)
            or self.coverable_detail_cell_count != count
            or self.mission_target_detail_cell_count != target_count
            or count > target_count
        ):
            raise CoverabilityError("streamed detail counts or ratio differ")
        if _require_sha(
            self.mission_target_mask_sha256,
            "streamed mission target hash",
        ) != _streamed_mask_sha256(
            self.mission_target_detail_bits, shape
        ):
            raise CoverabilityError("streamed mission target hash differs")
        if _require_sha(
            self.coverable_mask_sha256,
            "streamed coverable hash",
        ) != _streamed_mask_sha256(
            self.coverable_detail_bits, shape
        ):
            raise CoverabilityError("streamed coverable hash differs")


def build_streamed_detail_coverability(
    *,
    tile_provider: object,
    inside_mission_roi: np.ndarray,
    reachable_pose_mask: np.ndarray,
    observation_positions_m: np.ndarray | None = None,
    intrinsic_terrain_feasible: Callable[[object], np.ndarray],
    reveal_from_pose: Callable[[np.ndarray, tuple[int, int]], np.ndarray],
) -> StreamedDetailCoverability:
    """Build exact masks while retaining only packed output and one truth window."""
    roi = _require_bool_mask(inside_mission_roi, "coarse mission ROI")
    reachable = _require_bool_mask(reachable_pose_mask, "reachable pose mask")
    if roi.shape != reachable.shape:
        raise CoverabilityError("mission ROI and reachable masks must align")
    if observation_positions_m is not None and (
        not isinstance(observation_positions_m, np.ndarray)
        or observation_positions_m.dtype != np.dtype(np.float64)
        or observation_positions_m.ndim != 2
        or observation_positions_m.shape[1:] != (3,)
        or len(observation_positions_m)
        != int(reachable.sum(dtype=np.int64))
        or not observation_positions_m.flags.c_contiguous
        or not np.isfinite(observation_positions_m).all()
    ):
        raise CoverabilityError(
            "observation positions must be finite C-contiguous float64 [N,3]"
        )
    if not callable(intrinsic_terrain_feasible) or not callable(
        reveal_from_pose
    ):
        raise CoverabilityError("intrinsic and reveal callbacks are required")
    try:
        total = int(tile_provider.detail_cells_per_axis)
        tile_cells = int(tile_provider.tile_geometry.cells)
        tile_indices = tile_provider.iter_tile_indices()
    except (AttributeError, TypeError, ValueError) as error:
        raise CoverabilityError("detail tile provider is invalid") from error
    shape = (total, total)
    if (
        total <= 0
        or tile_cells <= 0
        or total % 8
        or tile_cells % 8
        or total % roi.shape[0]
        or total % roi.shape[1]
    ):
        raise CoverabilityError("detail and coarse geometries are incompatible")
    packed_size = math.prod(shape) // 8
    target_bits = np.zeros(packed_size, dtype=np.uint8)
    coverable_bits = np.zeros(packed_size, dtype=np.uint8)
    row_scale = total // roi.shape[0]
    column_scale = total // roi.shape[1]

    for tile_row, tile_column in tile_indices:
        window = tile_provider.tile_with_halo(
            tile_row, tile_column, halo_cells=1
        )
        intrinsic = intrinsic_terrain_feasible(window.projected)
        if (
            not isinstance(intrinsic, np.ndarray)
            or intrinsic.dtype != np.dtype(np.bool_)
            or intrinsic.shape != window.projected.valid_mask.shape
            or not intrinsic.flags.c_contiguous
        ):
            raise CoverabilityError(
                "intrinsic callback returned an invalid detail mask"
            )
        crop = (window.tile_rows, window.tile_columns)
        start_row = tile_row * tile_cells
        start_column = tile_column * tile_cells
        coarse_rows = (
            start_row + np.arange(tile_cells, dtype=np.int64)
        ) // row_scale
        coarse_columns = (
            start_column + np.arange(tile_cells, dtype=np.int64)
        ) // column_scale
        detail_roi = np.ascontiguousarray(
            roi[np.ix_(coarse_rows, coarse_columns)], dtype=np.bool_
        )
        target = build_mission_target_detail_mask(
            inside_mission_roi=detail_roi,
            detail_valid=np.ascontiguousarray(
                window.projected.valid_mask[crop], dtype=np.bool_
            ),
            forbidden_ratio=np.ascontiguousarray(
                window.projected.forbidden_ratio[crop], dtype=np.float32
            ),
            physical_obstacle_ratio=np.ascontiguousarray(
                window.projected.physical_obstacle_ratio[crop],
                dtype=np.float32,
            ),
            intrinsic_terrain_feasible=np.ascontiguousarray(
                intrinsic[crop], dtype=np.bool_
            ),
        )
        _write_aligned_tile(
            target_bits,
            shape,
            start_row=start_row,
            start_column=start_column,
            mask=target,
        )

    if observation_positions_m is None:
        coarse_sources = _spatially_prioritized_sources(
            reachable & roi,
            row_scale=row_scale,
            column_scale=column_scale,
            window_cells=tile_cells,
        )
        pose_sources: Iterator[tuple[int, int]] = (
            (
                coarse_row * row_scale + row_scale // 2,
                coarse_column * column_scale + column_scale // 2,
            )
            for coarse_row, coarse_column in coarse_sources
        )
    else:
        try:
            exact_sources = tuple(
                sorted(
                    {
                        tile_provider.world_to_detail(
                            float(position[0]), float(position[1])
                        )
                        for position in observation_positions_m
                    }
                )
            )
        except (AttributeError, TypeError, ValueError) as error:
            raise CoverabilityError(
                "physical observation position lies outside detail geometry"
            ) from error
        pose_sources = _prioritize_exact_detail_sources(
            exact_sources, window_cells=tile_cells
        )
    for pose_row, pose_column in pose_sources:
        start_row = pose_row - tile_cells // 2
        start_column = pose_column - tile_cells // 2
        if (
            start_row < 0
            or start_column < 0
            or start_row + tile_cells > total
            or start_column + tile_cells > total
        ):
            continue
        target = _read_packed_window(
            target_bits,
            shape,
            start_row=start_row,
            start_column=start_column,
            cells=tile_cells,
        )
        already_coverable = _read_packed_window(
            coverable_bits,
            shape,
            start_row=start_row,
            start_column=start_column,
            cells=tile_cells,
        )
        remaining = target & ~already_coverable
        if not remaining.any():
            continue
        truth_obstacle_ratio = tile_provider.read_visibility_obstacle_window(
            start_row, start_column, cells=tile_cells
        )
        visible = reveal_from_pose(
            truth_obstacle_ratio,
            (pose_row - start_row, pose_column - start_column),
        )
        if (
            not isinstance(visible, np.ndarray)
            or visible.dtype != np.dtype(np.bool_)
            or visible.shape != (tile_cells, tile_cells)
            or not visible.flags.c_contiguous
        ):
            raise CoverabilityError(
                "detail reveal returned an invalid visibility mask"
            )
        _or_packed_window(
            coverable_bits,
            shape,
            start_row=start_row,
            start_column=start_column,
            mask=np.ascontiguousarray(visible & remaining),
        )

    ratio, coverable_count = _streamed_coarse_ratio(
        coverable_bits, shape, reachable.shape
    )
    target_count = int(_BYTE_POPCOUNT[target_bits].sum(dtype=np.int64))
    return StreamedDetailCoverability(
        detail_shape=shape,
        mission_target_detail_bits=target_bits,
        coverable_detail_bits=coverable_bits,
        coverable_ratio=ratio,
        mission_target_detail_cell_count=target_count,
        coverable_detail_cell_count=coverable_count,
        mission_target_mask_sha256=_streamed_mask_sha256(target_bits, shape),
        coverable_mask_sha256=_streamed_mask_sha256(coverable_bits, shape),
    )


def classify_ineligibility(
    *,
    qualified_start_cell: tuple[int, int] | None,
    mission_target_detail_cell_count: int,
    coverable_detail_cell_count: int,
    initial_coverable_fraction: float,
    initial_candidate_count: int,
) -> IneligibleReason | None:
    """Return the first ordinary eligibility failure in the frozen order."""
    for name, count in (
        ("mission target cell count", mission_target_detail_cell_count),
        ("coverable detail cell count", coverable_detail_cell_count),
        ("initial candidate count", initial_candidate_count),
    ):
        if type(count) is not int or count < 0:
            raise CoverabilityError(f"{name} must be a non-negative integer")
    if coverable_detail_cell_count > mission_target_detail_cell_count:
        raise CoverabilityError("coverable detail count exceeds mission target count")
    if (
        not isinstance(initial_coverable_fraction, float)
        or not math.isfinite(initial_coverable_fraction)
        or not 0.0 <= initial_coverable_fraction <= 1.0
    ):
        raise CoverabilityError("initial coverable fraction must be finite in [0,1]")
    if qualified_start_cell is not None and (
        not isinstance(qualified_start_cell, tuple)
        or len(qualified_start_cell) != 2
        or any(type(index) is not int or index < 0 for index in qualified_start_cell)
    ):
        raise CoverabilityError("qualified start cell is invalid")

    if qualified_start_cell is None:
        return IneligibleReason.UNSAFE_START
    if mission_target_detail_cell_count == 0:
        return IneligibleReason.ZERO_MISSION_TARGET
    if (
        coverable_detail_cell_count / mission_target_detail_cell_count
        < FORMAL_MINIMUM_MISSION_COVERABLE_RATIO
    ):
        return IneligibleReason.MISSION_COVERABLE_BELOW_95
    if initial_coverable_fraction >= FORMAL_SUCCESS_COVERAGE_RATIO:
        return IneligibleReason.INITIAL_ALREADY_SUCCESS
    if initial_candidate_count == 0:
        return IneligibleReason.NO_INITIAL_CANDIDATE
    return None


@dataclass(frozen=True, slots=True)
class QualifiedStartState:
    """Canonical platform state anchoring one truth reachability graph."""

    position_m: tuple[float, float, float]
    yaw_rad: float
    motion_mode: int
    body_z_m: tuple[float, float]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.position_m, tuple)
            or len(self.position_m) != 3
            or not isinstance(self.body_z_m, tuple)
            or len(self.body_z_m) != 2
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in (*self.position_m, *self.body_z_m, self.yaw_rad)
            )
            or type(self.motion_mode) is not int
            or self.motion_mode < 0
            or self.body_z_m[0] > self.body_z_m[1]
        ):
            raise CoverabilityError("qualified start state is invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            "position_m": [float(value) for value in self.position_m],
            "yaw_rad": float(self.yaw_rad),
            "motion_mode": self.motion_mode,
            "body_z_m": [float(value) for value in self.body_z_m],
        }

    @classmethod
    def from_dict(cls, value: object) -> "QualifiedStartState":
        fields = {"position_m", "yaw_rad", "motion_mode", "body_z_m"}
        if not isinstance(value, Mapping) or set(value) != fields:
            raise CoverabilityError("qualified start state fields are invalid")
        position = value["position_m"]
        body_z = value["body_z_m"]
        if not isinstance(position, list) or not isinstance(body_z, list):
            raise CoverabilityError("qualified start state vectors are invalid")
        return cls(
            position_m=tuple(position),
            yaw_rad=value["yaw_rad"],
            motion_mode=value["motion_mode"],
            body_z_m=tuple(body_z),
        )


@dataclass(frozen=True, slots=True)
class PlatformCoverability:
    """Validated exact cache payload for one scene/platform/fixed start."""

    platform_type: str
    qualified_start_cell: tuple[int, int] | None
    physical_observation_pose_mask: np.ndarray
    physical_projection_schema: str
    physical_reachability_algorithm_id: str
    physical_safe_pose_count: int
    physically_reachable_pose_count: int
    physical_projection_sha256: str
    mission_target_detail_mask_sha256: str
    coverable_detail_shape: tuple[int, int]
    coverable_detail_bits: np.ndarray
    coverable_ratio: np.ndarray
    mission_target_detail_cell_count: int
    coverable_detail_cell_count: int
    mission_coverable_fraction: float
    initial_coverable_fraction: float
    initial_candidate_count: int
    coverable_detail_mask_sha256: str
    sensor_visibility_algorithm_id: str
    capability_content_sha256: str
    start_identity_sha256: str
    exact: bool
    eligible: bool
    ineligible_reason: IneligibleReason | None

    def __post_init__(self) -> None:
        if self.platform_type not in _PLATFORM_TYPES:
            raise CoverabilityError("platform type is invalid")
        physical = _require_bool_mask(
            self.physical_observation_pose_mask,
            "physical observation pose mask",
        )
        shape = _detail_shape(self.coverable_detail_shape)
        detail = unpack_detail_mask(self.coverable_detail_bits, shape)
        if shape[0] % physical.shape[0] or shape[1] % physical.shape[1]:
            raise CoverabilityError(
                "detail mask shape must divide evenly into the physical grid"
            )
        ratio = self.coverable_ratio
        if (
            not isinstance(ratio, np.ndarray)
            or ratio.dtype != np.dtype(np.float32)
            or ratio.shape != physical.shape
            or not ratio.flags.c_contiguous
            or not np.isfinite(ratio).all()
            or ((ratio < 0.0) | (ratio > 1.0)).any()
        ):
            raise CoverabilityError("coverable ratio matrix is invalid")
        rows_per_cell = shape[0] // physical.shape[0]
        columns_per_cell = shape[1] // physical.shape[1]
        expected_ratio = detail.reshape(
            physical.shape[0],
            rows_per_cell,
            physical.shape[1],
            columns_per_cell,
        ).mean(axis=(1, 3), dtype=np.float64).astype(np.float32)
        if not np.array_equal(ratio, expected_ratio):
            raise CoverabilityError("coverable ratio differs from exact detail bits")

        if type(self.exact) is not bool or not self.exact:
            raise CoverabilityError("coverability payload must be exact")
        if type(self.eligible) is not bool:
            raise CoverabilityError("coverability eligibility must be boolean")
        for name, value in (
            ("physical projection schema", self.physical_projection_schema),
            (
                "physical reachability algorithm ID",
                self.physical_reachability_algorithm_id,
            ),
            (
                "sensor visibility algorithm ID",
                self.sensor_visibility_algorithm_id,
            ),
        ):
            if not isinstance(value, str) or not value:
                raise CoverabilityError(f"{name} is missing")
        if self.physical_projection_schema != PHYSICAL_PROJECTION_SCHEMA:
            raise CoverabilityError("physical projection schema is unsupported")
        for name, value in (
            (
                "mission target detail mask hash",
                self.mission_target_detail_mask_sha256,
            ),
            ("capability content hash", self.capability_content_sha256),
            ("start identity hash", self.start_identity_sha256),
        ):
            _require_sha(value, name)
        reachable_count = int(physical.sum(dtype=np.int64))
        if (
            type(self.physically_reachable_pose_count) is not int
            or self.physically_reachable_pose_count != reachable_count
        ):
            raise CoverabilityError(
                "physically reachable pose count differs from physical mask"
            )
        if (
            type(self.physical_safe_pose_count) is not int
            or self.physical_safe_pose_count < reachable_count
        ):
            raise CoverabilityError(
                "physical safe pose count is below reachable pose count"
            )
        _require_sha(self.physical_projection_sha256, "physical projection hash")
        if mask_sha256(detail) != _require_sha(
            self.coverable_detail_mask_sha256,
            "coverable mask hash",
        ):
            raise CoverabilityError("coverable mask hash differs")

        if type(self.coverable_detail_cell_count) is not int or (
            self.coverable_detail_cell_count != int(detail.sum(dtype=np.int64))
        ):
            raise CoverabilityError("coverable detail cell count differs from mask")
        expected_reason = classify_ineligibility(
            qualified_start_cell=self.qualified_start_cell,
            mission_target_detail_cell_count=self.mission_target_detail_cell_count,
            coverable_detail_cell_count=self.coverable_detail_cell_count,
            initial_coverable_fraction=self.initial_coverable_fraction,
            initial_candidate_count=self.initial_candidate_count,
        )
        expected_fraction = (
            self.coverable_detail_cell_count
            / self.mission_target_detail_cell_count
            if self.mission_target_detail_cell_count
            else 0.0
        )
        if (
            not isinstance(self.mission_coverable_fraction, float)
            or not math.isclose(
                self.mission_coverable_fraction,
                expected_fraction,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        ):
            raise CoverabilityError("mission coverable fraction differs from counts")
        if self.eligible is not (expected_reason is None):
            raise CoverabilityError("coverability eligibility disagrees with gates")
        if self.ineligible_reason is not expected_reason:
            raise CoverabilityError("coverability ineligible reason is invalid")
        if self.qualified_start_cell is None:
            if reachable_count:
                raise CoverabilityError(
                    "physical observation poses require a qualified start"
                )
        else:
            row, column = self.qualified_start_cell
            if (
                type(row) is not int
                or type(column) is not int
                or row < 0
                or column < 0
                or row >= physical.shape[0]
                or column >= physical.shape[1]
                or not bool(physical[row, column])
            ):
                raise CoverabilityError(
                    "qualified start is not physically reachable"
                )

    @property
    def coverable_detail_mask(self) -> np.ndarray:
        """Return a detached exact boolean mask for runtime accounting."""
        return unpack_detail_mask(
            self.coverable_detail_bits.copy(), self.coverable_detail_shape
        )


__all__ = [
    "build_coverable_detail_mask",
    "build_mission_target_detail_mask",
    "build_streamed_detail_coverability",
    "canonical_physical_positions_um",
    "CoverabilityError",
    "IneligibleReason",
    "PlatformCoverability",
    "PHYSICAL_GRID_AXIS_CONVENTION",
    "PHYSICAL_PROJECTION_SCHEMA",
    "QualifiedStartState",
    "StreamedDetailCoverability",
    "classify_ineligibility",
    "mask_sha256",
    "pack_detail_mask",
    "physical_projection_sha256",
    "read_packed_detail_window",
    "unpack_detail_mask",
]

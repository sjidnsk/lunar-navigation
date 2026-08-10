"""Exact platform/start-specific exploration coverability contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
import math
from typing import Callable, Iterator

import numpy as np

from ..training_semantics import FORMAL_SUCCESS_COVERAGE_RATIO


_PLATFORM_TYPES = frozenset(("WHEELED", "LEGGED", "HOPPER"))
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
    reveal_from_pose: Callable[[np.ndarray, tuple[int, int]], np.ndarray],
) -> np.ndarray:
    """Union exact detail visibility from every reachable coarse pose center."""
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
    coverable = np.zeros(target.shape, dtype=np.bool_)
    for row, column in np.argwhere(reachable):
        pose_cell = (
            int(row) * row_scale + row_scale // 2,
            int(column) * column_scale + column_scale // 2,
        )
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
    intrinsic_terrain_feasible: Callable[[object], np.ndarray],
    reveal_from_pose: Callable[[np.ndarray, tuple[int, int]], np.ndarray],
) -> StreamedDetailCoverability:
    """Build exact masks while retaining only packed output and one truth window."""
    roi = _require_bool_mask(inside_mission_roi, "coarse mission ROI")
    reachable = _require_bool_mask(reachable_pose_mask, "reachable pose mask")
    if roi.shape != reachable.shape:
        raise CoverabilityError("mission ROI and reachable masks must align")
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

    observation_sources = reachable & roi
    for coarse_row, coarse_column in _spatially_prioritized_sources(
        observation_sources,
        row_scale=row_scale,
        column_scale=column_scale,
        window_cells=tile_cells,
    ):
        pose_row = coarse_row * row_scale + row_scale // 2
        pose_column = (
            coarse_column * column_scale + column_scale // 2
        )
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
        < FORMAL_SUCCESS_COVERAGE_RATIO
    ):
        return IneligibleReason.MISSION_COVERABLE_BELOW_95
    if initial_coverable_fraction >= FORMAL_SUCCESS_COVERAGE_RATIO:
        return IneligibleReason.INITIAL_ALREADY_SUCCESS
    if initial_candidate_count == 0:
        return IneligibleReason.NO_INITIAL_CANDIDATE
    return None


@dataclass(frozen=True, slots=True)
class PlatformCoverability:
    """Validated exact cache payload for one scene/platform/fixed start."""

    platform_type: str
    qualified_start_cell: tuple[int, int] | None
    reachable_pose_mask: np.ndarray
    coverable_detail_shape: tuple[int, int]
    coverable_detail_bits: np.ndarray
    coverable_ratio: np.ndarray
    mission_target_detail_cell_count: int
    coverable_detail_cell_count: int
    mission_coverable_fraction: float
    initial_coverable_fraction: float
    initial_candidate_count: int
    reachability_algorithm_id: str
    visibility_algorithm_id: str
    reachable_mask_sha256: str
    coverable_mask_sha256: str
    exact: bool
    eligible: bool
    ineligible_reason: IneligibleReason | None

    def __post_init__(self) -> None:
        if self.platform_type not in _PLATFORM_TYPES:
            raise CoverabilityError("platform type is invalid")
        reachable = _require_bool_mask(
            self.reachable_pose_mask, "reachable pose mask"
        )
        shape = _detail_shape(self.coverable_detail_shape)
        detail = unpack_detail_mask(self.coverable_detail_bits, shape)
        if shape[0] % reachable.shape[0] or shape[1] % reachable.shape[1]:
            raise CoverabilityError(
                "detail mask shape must divide evenly into the reachable grid"
            )
        ratio = self.coverable_ratio
        if (
            not isinstance(ratio, np.ndarray)
            or ratio.dtype != np.dtype(np.float32)
            or ratio.shape != reachable.shape
            or not ratio.flags.c_contiguous
            or not np.isfinite(ratio).all()
            or ((ratio < 0.0) | (ratio > 1.0)).any()
        ):
            raise CoverabilityError("coverable ratio matrix is invalid")
        rows_per_cell = shape[0] // reachable.shape[0]
        columns_per_cell = shape[1] // reachable.shape[1]
        expected_ratio = detail.reshape(
            reachable.shape[0],
            rows_per_cell,
            reachable.shape[1],
            columns_per_cell,
        ).mean(axis=(1, 3), dtype=np.float64).astype(np.float32)
        if not np.array_equal(ratio, expected_ratio):
            raise CoverabilityError("coverable ratio differs from exact detail bits")

        if type(self.exact) is not bool or not self.exact:
            raise CoverabilityError("coverability payload must be exact")
        if type(self.eligible) is not bool:
            raise CoverabilityError("coverability eligibility must be boolean")
        for name, value in (
            ("reachability algorithm ID", self.reachability_algorithm_id),
            ("visibility algorithm ID", self.visibility_algorithm_id),
        ):
            if not isinstance(value, str) or not value:
                raise CoverabilityError(f"{name} is missing")
        if mask_sha256(reachable) != _require_sha(
            self.reachable_mask_sha256, "reachable mask hash"
        ):
            raise CoverabilityError("reachable mask hash differs")
        if mask_sha256(detail) != _require_sha(
            self.coverable_mask_sha256, "coverable mask hash"
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
        if self.qualified_start_cell is not None:
            row, column = self.qualified_start_cell
            if (
                row >= reachable.shape[0]
                or column >= reachable.shape[1]
                or not bool(reachable[row, column])
            ):
                raise CoverabilityError("qualified start is not reachable")

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
    "CoverabilityError",
    "IneligibleReason",
    "PlatformCoverability",
    "StreamedDetailCoverability",
    "classify_ineligibility",
    "mask_sha256",
    "pack_detail_mask",
    "read_packed_detail_window",
    "unpack_detail_mask",
]

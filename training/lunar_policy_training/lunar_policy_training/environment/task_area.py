"""Deterministic bounded task views over immutable formal scene evidence."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import numpy as np

from ..capability_freeze import (
    FrozenCapabilityBundle,
    FrozenHopperCapability,
    FrozenLeggedCapability,
    FrozenWheeledCapability,
)
from ..config import TaskAreaConfig
from ..polar_data.raster import GLOBAL_GEOMETRY, LOCAL_TILE_GEOMETRY
from ..reward_contract import TaskScaleBucket
from .coverability import mask_sha256


DETAIL_PER_GLOBAL = 20
_PLATFORMS = frozenset(("WHEELED", "LEGGED", "HOPPER"))
_SCALE_BUCKET_ORDER = tuple(TaskScaleBucket)
SCALE_BUCKET_SPAN_CELLS = {
    TaskScaleBucket.M100_200: (25, 49),
    TaskScaleBucket.M200_300: (50, 74),
    TaskScaleBucket.M300_400: (75, 99),
    TaskScaleBucket.M400_500: (100, 125),
}
_TASK_EVIDENCE_HALO_ALGORITHM_ID = "formal-task-evidence-halo/v1"
_TASK_GEOMETRY_SCHEMA = "formal-task-geometry/v1"
_NATIVE_STENCIL_DETAIL_CELLS = 2


@dataclass(frozen=True, slots=True)
class TaskEvidenceHaloContract:
    coarse_cells: int
    detail_cells: int
    sensor_radius_m: float
    platform_support_radius_m: float
    native_stencil_radius_m: float
    algorithm_id: str
    capability_bundle_sha256: str

    def __post_init__(self) -> None:
        if type(self.coarse_cells) is not int or self.coarse_cells < 0:
            raise ValueError("task evidence halo coarse width is invalid")
        if (
            type(self.detail_cells) is not int
            or self.detail_cells != self.coarse_cells * DETAIL_PER_GLOBAL
        ):
            raise ValueError("task evidence halo detail width is not coarse-aligned")
        for value in (
            self.sensor_radius_m,
            self.platform_support_radius_m,
            self.native_stencil_radius_m,
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("task evidence halo radius is invalid")
        if self.algorithm_id != _TASK_EVIDENCE_HALO_ALGORITHM_ID:
            raise ValueError("task evidence halo algorithm is unsupported")
        _validate_sha256(
            self.capability_bundle_sha256,
            "task evidence halo capability bundle",
        )


@dataclass(frozen=True, slots=True)
class FrozenTaskGeometry:
    episode_seed: str
    scale_bucket: TaskScaleBucket
    span_cells: int
    coarse_bounds_half_open: tuple[int, int, int, int]
    detail_bounds_half_open: tuple[int, int, int, int]
    halo_coarse_bounds_half_open: tuple[int, int, int, int]
    local_start_cell: tuple[int, int]
    geometry_sha256: str

    def __post_init__(self) -> None:
        _validate_sha256(self.episode_seed, "task geometry episode seed")
        _validate_sha256(self.geometry_sha256, "task geometry identity")


def derive_task_evidence_halo(
    capability_bundle: FrozenCapabilityBundle,
) -> TaskEvidenceHaloContract:
    """Derive one shared evidence-only halo from frozen platform authority."""
    if not isinstance(capability_bundle, FrozenCapabilityBundle):
        raise TypeError("task evidence halo requires FrozenCapabilityBundle")
    if not capability_bundle.formal_eligible:
        raise ValueError("task evidence halo requires a formal capability bundle")
    sensor_radius_m = max(
        platform.observation_capability.sensor_range_m
        for platform in capability_bundle.platforms
    )
    support_radii: list[float] = []
    for platform in capability_bundle.platforms:
        typed = platform.typed_capability
        if isinstance(typed, FrozenWheeledCapability):
            body_radius = max(
                math.hypot(vertex.x, vertex.y)
                for vertex in typed.footprint_xy_m
            )
            support_radii.append(body_radius + typed.minimum_clearance_m)
        elif isinstance(typed, FrozenLeggedCapability):
            body_radius = math.hypot(
                typed.body_extent_m.x / 2.0,
                typed.body_extent_m.y / 2.0,
            )
            support_radii.append(body_radius + typed.maximum_gap_width_m)
        elif isinstance(typed, FrozenHopperCapability):
            support_radii.append(
                max(
                    typed.landing_support_radius_m
                    + typed.landing_lateral_margin_m,
                    typed.flight_collision_radius_m + typed.flight_map_margin_m,
                )
            )
        else:  # pragma: no cover - closed frozen union
            raise TypeError("task evidence halo capability type is unsupported")
    platform_support_radius_m = max(support_radii)
    native_stencil_radius_m = (
        _NATIVE_STENCIL_DETAIL_CELLS * LOCAL_TILE_GEOMETRY.resolution_m
    )
    required_radius_m = max(
        sensor_radius_m,
        platform_support_radius_m,
        native_stencil_radius_m,
    )
    coarse_cells = math.ceil(required_radius_m / GLOBAL_GEOMETRY.resolution_m)
    return TaskEvidenceHaloContract(
        coarse_cells=coarse_cells,
        detail_cells=coarse_cells * DETAIL_PER_GLOBAL,
        sensor_radius_m=sensor_radius_m,
        platform_support_radius_m=platform_support_radius_m,
        native_stencil_radius_m=native_stencil_radius_m,
        algorithm_id=_TASK_EVIDENCE_HALO_ALGORITHM_ID,
        capability_bundle_sha256=capability_bundle.bundle_sha256,
    )


def freeze_formal_task_geometry(
    *,
    start_cell: tuple[int, int],
    config: TaskAreaConfig,
    episode_seed: str,
    scale_bucket: TaskScaleBucket,
    halo: TaskEvidenceHaloContract,
) -> FrozenTaskGeometry:
    """Freeze replayable task-local bounds before any derived computation."""
    if not isinstance(scale_bucket, TaskScaleBucket):
        raise TypeError("task geometry scale bucket is invalid")
    if not isinstance(halo, TaskEvidenceHaloContract):
        raise TypeError("task geometry halo contract is invalid")
    span_cells = sample_task_area_span_cells(
        config,
        episode_seed,
        scale_bucket=scale_bucket,
    )
    coarse_bounds = task_area_bounds(start_cell, span_cells)
    row0, row1, column0, column1 = coarse_bounds
    detail_bounds = tuple(value * DETAIL_PER_GLOBAL for value in coarse_bounds)
    halo_bounds = (
        max(0, row0 - halo.coarse_cells),
        min(GLOBAL_GEOMETRY.cells, row1 + halo.coarse_cells),
        max(0, column0 - halo.coarse_cells),
        min(GLOBAL_GEOMETRY.cells, column1 + halo.coarse_cells),
    )
    local_start = start_cell[0] - row0, start_cell[1] - column0
    identity = {
        "schema": _TASK_GEOMETRY_SCHEMA,
        "episode_seed": episode_seed,
        "scale_bucket": scale_bucket.value,
        "span_cells": span_cells,
        "coarse_bounds_half_open": list(coarse_bounds),
        "detail_bounds_half_open": list(detail_bounds),
        "halo_coarse_bounds_half_open": list(halo_bounds),
        "local_start_cell": list(local_start),
        "halo": {
            "coarse_cells": halo.coarse_cells,
            "detail_cells": halo.detail_cells,
            "sensor_radius_m": halo.sensor_radius_m,
            "platform_support_radius_m": halo.platform_support_radius_m,
            "native_stencil_radius_m": halo.native_stencil_radius_m,
            "algorithm_id": halo.algorithm_id,
            "capability_bundle_sha256": halo.capability_bundle_sha256,
        },
    }
    geometry_sha256 = hashlib.sha256(
        json.dumps(
            identity,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    return FrozenTaskGeometry(
        episode_seed=episode_seed,
        scale_bucket=scale_bucket,
        span_cells=span_cells,
        coarse_bounds_half_open=coarse_bounds,
        detail_bounds_half_open=detail_bounds,
        halo_coarse_bounds_half_open=halo_bounds,
        local_start_cell=local_start,
        geometry_sha256=geometry_sha256,
    )


@dataclass(frozen=True, slots=True)
class WorkerStratum:
    worker_index: int
    platform_type: str
    platform_worker_index: int
    platform_worker_count: int
    scale_bucket: TaskScaleBucket


@dataclass(frozen=True, slots=True)
class TaskPriorityMask:
    coarse_mask: np.ndarray
    side_cells: int
    priority_seed: str
    coarse_mask_sha256: str


def task_roi_diagonal_m(
    roi_ratio: np.ndarray,
    *,
    resolution_m: float = float(GLOBAL_GEOMETRY.resolution_m),
) -> float:
    """Return the physical diagonal of the active half-open ROI envelope."""
    roi = np.asarray(roi_ratio)
    if (
        roi.ndim != 2
        or min(roi.shape) <= 0
        or not np.issubdtype(roi.dtype, np.number)
        or not np.isfinite(roi).all()
        or (roi < 0.0).any()
        or not math.isfinite(resolution_m)
        or resolution_m <= 0.0
    ):
        raise ValueError("task ROI geometry is invalid")
    rows, columns = np.nonzero(roi > 0.0)
    if not len(rows):
        raise ValueError("task ROI is empty")
    height_m = (int(rows.max()) - int(rows.min()) + 1) * resolution_m
    width_m = (int(columns.max()) - int(columns.min()) + 1) * resolution_m
    diagonal = math.hypot(width_m, height_m)
    if not math.isfinite(diagonal) or diagonal <= 0.0:
        raise ValueError("task ROI diagonal is invalid")
    return diagonal


def formal_worker_strata(stage: object) -> tuple[WorkerStratum, ...]:
    """Return the frozen platform/bucket mapping for one curriculum stage."""
    stage_value = getattr(stage, "value", stage)
    if stage_value == "GROUND_R1":
        allocations = (("WHEELED", 12), ("LEGGED", 12))
    elif stage_value in {"GROUND_R2_HOPPER_R1", "THREE_PLATFORM_R2"}:
        allocations = (("WHEELED", 8), ("LEGGED", 8), ("HOPPER", 8))
    else:
        raise ValueError("formal training stage is invalid")
    strata: list[WorkerStratum] = []
    worker_index = 0
    for platform_type, platform_worker_count in allocations:
        per_bucket = platform_worker_count // len(_SCALE_BUCKET_ORDER)
        lane = 0
        for scale_bucket in _SCALE_BUCKET_ORDER:
            for _ in range(per_bucket):
                strata.append(
                    WorkerStratum(
                        worker_index=worker_index,
                        platform_type=platform_type,
                        platform_worker_index=lane,
                        platform_worker_count=platform_worker_count,
                        scale_bucket=scale_bucket,
                    )
                )
                worker_index += 1
                lane += 1
    return tuple(strata)


def scale_bucket_for_worker(
    platform_worker_index: int, platform_worker_count: int
) -> TaskScaleBucket:
    """Resolve one platform-local lane to its immutable scale bucket."""
    if (
        type(platform_worker_index) is not int
        or type(platform_worker_count) is not int
        or platform_worker_count <= 0
        or not 0 <= platform_worker_index < platform_worker_count
    ):
        raise ValueError("formal platform worker lane is invalid")
    bucket_index = min(
        len(_SCALE_BUCKET_ORDER) - 1,
        platform_worker_index * len(_SCALE_BUCKET_ORDER)
        // platform_worker_count,
    )
    return _SCALE_BUCKET_ORDER[bucket_index]


def sample_task_area_span_cells(
    config: TaskAreaConfig,
    episode_seed: str,
    *,
    scale_bucket: TaskScaleBucket | None = None,
) -> int:
    """Sample one unbiased 4 m-aligned square side from a replayable seed."""
    if not isinstance(config, TaskAreaConfig):
        raise TypeError("task area sampling requires TaskAreaConfig")
    if (
        not isinstance(episode_seed, str)
        or len(episode_seed) != 64
        or any(character not in "0123456789abcdef" for character in episode_seed)
    ):
        raise ValueError("task area episode seed must be a lowercase SHA-256")
    resolution_m = float(GLOBAL_GEOMETRY.resolution_m)
    minimum = round(config.minimum_size_m / resolution_m)
    maximum = round(config.maximum_size_m / resolution_m)
    if (
        minimum * resolution_m != config.minimum_size_m
        or maximum * resolution_m != config.maximum_size_m
        or not 0 < minimum <= maximum <= GLOBAL_GEOMETRY.cells
    ):
        raise ValueError("task area range is not aligned to the global grid")
    if scale_bucket is not None:
        if not isinstance(scale_bucket, TaskScaleBucket):
            raise ValueError("task area scale bucket is invalid")
        bucket_minimum, bucket_maximum = SCALE_BUCKET_SPAN_CELLS[scale_bucket]
        minimum = max(minimum, bucket_minimum)
        maximum = min(maximum, bucket_maximum)
        if minimum > maximum:
            raise ValueError("task area scale bucket does not intersect config")
    count = maximum - minimum + 1
    modulus = 1 << 256
    acceptance_limit = modulus - modulus % count
    counter = 0
    while True:
        if scale_bucket is None:
            payload = (
                f"{config.sampling_algorithm}\0{episode_seed}\0{counter}"
            ).encode("utf-8")
        else:
            payload = (
                f"{config.sampling_algorithm}\0{episode_seed}\0"
                f"{scale_bucket.value}\0{counter}"
            ).encode("utf-8")
        value = int.from_bytes(hashlib.sha256(payload).digest(), "big")
        if value < acceptance_limit:
            return minimum + value % count
        counter += 1


def build_task_priority_mask(
    roi: np.ndarray,
    *,
    span_cells: int,
    priority_seed: str,
) -> TaskPriorityMask:
    """Place the task-owned binary priority square without coverability input."""
    values = np.ascontiguousarray(np.asarray(roi, dtype=np.bool_))
    if values.ndim != 2 or min(values.shape) <= 0 or not bool(values.any()):
        raise ValueError("task priority ROI is invalid")
    if type(span_cells) is not int or span_cells <= 0:
        raise ValueError("task priority span is invalid")
    _validate_sha256(priority_seed, "task priority seed")
    side_cells = max(1, math.ceil(span_cells / 4))
    rows, columns = np.nonzero(values)
    row_min, row_max = int(rows.min()), int(rows.max()) + 1
    column_min, column_max = int(columns.min()), int(columns.max()) + 1
    row_positions = row_max - row_min - side_cells + 1
    column_positions = column_max - column_min - side_cells + 1
    if row_positions <= 0 or column_positions <= 0:
        raise ValueError("task priority square does not fit the ROI envelope")
    placement_count = row_positions * column_positions
    placement = _uniform_index(
        placement_count,
        namespace="task-priority-placement/v1",
        seed=priority_seed,
    )
    row0 = row_min + placement // column_positions
    column0 = column_min + placement % column_positions
    mask = np.zeros(values.shape, dtype=np.bool_)
    mask[
        row0 : row0 + side_cells,
        column0 : column0 + side_cells,
    ] = True
    mask &= values
    mask = np.ascontiguousarray(mask)
    digest = mask_sha256(mask)
    mask.setflags(write=False)
    return TaskPriorityMask(
        coarse_mask=mask,
        side_cells=side_cells,
        priority_seed=priority_seed,
        coarse_mask_sha256=digest,
    )


def _uniform_index(count: int, *, namespace: str, seed: str) -> int:
    if type(count) is not int or count <= 0:
        raise ValueError("uniform sample count is invalid")
    modulus = 1 << 256
    acceptance_limit = modulus - modulus % count
    counter = 0
    while True:
        value = int.from_bytes(
            hashlib.sha256(
                f"{namespace}\0{seed}\0{counter}".encode("utf-8")
            ).digest(),
            "big",
        )
        if value < acceptance_limit:
            return value % count
        counter += 1


def _validate_sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _bucket_for_span(span_cells: int) -> TaskScaleBucket:
    for bucket, (lower, upper) in SCALE_BUCKET_SPAN_CELLS.items():
        if lower <= span_cells <= upper:
            return bucket
    raise ValueError("task area span is outside Reward V4 scale buckets")


def task_area_bounds(
    start_cell: tuple[int, int], span_cells: int
) -> tuple[int, int, int, int]:
    """Center an intact square where possible and translate it at map edges."""
    if (
        not isinstance(start_cell, tuple)
        or len(start_cell) != 2
        or any(type(value) is not int for value in start_cell)
    ):
        raise TypeError("task area start cell must be an integer pair")
    if type(span_cells) is not int or not 0 < span_cells <= GLOBAL_GEOMETRY.cells:
        raise ValueError("task area span is invalid")

    def axis_bounds(center: int) -> tuple[int, int]:
        if not 0 <= center < GLOBAL_GEOMETRY.cells:
            raise ValueError("task area start leaves the global grid")
        lower = center - span_cells // 2
        lower = min(max(lower, 0), GLOBAL_GEOMETRY.cells - span_cells)
        return lower, lower + span_cells

    row0, row1 = axis_bounds(start_cell[0])
    column0, column1 = axis_bounds(start_cell[1])
    return row0, row1, column0, column1


__all__ = [
    "SCALE_BUCKET_SPAN_CELLS",
    "TaskPriorityMask",
    "WorkerStratum",
    "build_task_priority_mask",
    "formal_worker_strata",
    "sample_task_area_span_cells",
    "scale_bucket_for_worker",
    "task_roi_diagonal_m",
    "task_area_bounds",
]

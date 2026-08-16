"""Task-scoped common evidence and platform coverability builders."""

from __future__ import annotations

import math
from types import MappingProxyType, SimpleNamespace
from typing import Mapping

import numpy as np

from ..capability_freeze import (
    FrozenCapabilityBundle,
    FrozenPlatformCapability,
)
from ..environment.candidate_builder import (
    _frontier_chains,
    _map_frontier_chain_candidates,
    _points,
)
from ..environment.coverability import (
    PHYSICAL_GRID_AXIS_CONVENTION,
    canonical_physical_positions_um,
    classify_ineligibility,
    mask_sha256,
    physical_projection_sha256,
)
from ..environment.platform_reachability import (
    PHYSICAL_PROJECTION_SCHEMA,
    PlatformReachabilityError,
    _validate_ground_reachability_projection,
)
from ..environment.task_area import (
    DETAIL_PER_GLOBAL,
    FrozenTaskGeometry,
    derive_task_evidence_halo,
)
from ..environment.visibility import NativeVisibilityEstimator, SensorGeometry
from .formal_cache import (
    FormalSceneIndex,
    TaskSourceCrop,
    _physical_capability_content_sha256,
    _projection_request,
    load_task_source_crop,
)
from .raster import GridGeometry, MapCanvas
from .task_cache import (
    PlatformTaskPayload,
    TaskCommonArtifact,
    TaskCommonPayload,
)


class TaskCoverabilityError(ValueError):
    """Task geometry, source evidence, or platform derivation is invalid."""


_COMMON_SOURCE_ARRAYS = frozenset(
    (
        "elevation_m",
        "valid_mask",
        "physical_obstacle_ratio",
        "physical_obstacle_height_m",
        "forbidden_ratio",
        "detail_elevation_m",
        "detail_valid_mask",
        "detail_physical_obstacle_ratio",
        "detail_physical_obstacle_height_m",
        "detail_forbidden_ratio",
    )
)
_GROUND_PLATFORMS = frozenset(("WHEELED", "LEGGED"))
_GROUND_PHYSICAL_EVIDENCE_ALGORITHM_ID = (
    "cpp-safe-traversability-projection/v1"
)
_GROUND_SENSOR_ALGORITHM_ID = "sensor-30m-360/v1"
_GROUND_VISIBILITY_ALGORITHM_ID = "two-dimensional-detail-los/v1"
_HOPPER_REACHABILITY_ALGORITHM_ID = (
    "truth-hopper-task-safe-hop-observation-closure/v1"
)
_HOPPER_SENSOR_ALGORITHM_ID = (
    "hopper-unobstructed-trajectory-capsule-closure/v1"
)
_STATIC_PROJECTION_STAGE = "PLATFORM_STATIC_PROJECTION_COMPLETE"
_ZERO_DENOMINATOR_REASON = "ZERO_TASK_COVERABLE_DENOMINATOR"
_DETAIL_STATIC_TILE_CELLS = 320
_DETAIL_STATIC_HALO_CELLS = 1


def _expected_halo_bounds(
    geometry: FrozenTaskGeometry,
    halo_cells: int,
) -> tuple[int, int, int, int]:
    row0, row1, column0, column1 = geometry.coarse_bounds_half_open
    return (
        max(0, row0 - halo_cells),
        min(256, row1 + halo_cells),
        max(0, column0 - halo_cells),
        min(256, column1 + halo_cells),
    )


def _readonly(values: np.ndarray, dtype: object) -> np.ndarray:
    output = np.ascontiguousarray(np.asarray(values, dtype=dtype))
    output.setflags(write=False)
    return output


def _validate_source_arrays(
    crop: TaskSourceCrop,
    geometry: FrozenTaskGeometry,
) -> dict[str, np.ndarray]:
    if not isinstance(crop.arrays, Mapping) or set(crop.arrays) != set(
        _COMMON_SOURCE_ARRAYS
    ):
        raise TaskCoverabilityError("task source array inventory differs")
    halo = geometry.halo_coarse_bounds_half_open
    coarse_shape = halo[1] - halo[0], halo[3] - halo[2]
    detail_shape = tuple(value * DETAIL_PER_GLOBAL for value in coarse_shape)
    output: dict[str, np.ndarray] = {}
    for name in sorted(_COMMON_SOURCE_ARRAYS):
        values = np.asarray(crop.arrays[name])
        expected_shape = detail_shape if name.startswith("detail_") else coarse_shape
        expected_dtype = np.bool_ if name.endswith("valid_mask") else np.float32
        if values.shape != expected_shape or values.dtype != np.dtype(
            expected_dtype
        ):
            raise TaskCoverabilityError(
                f"task source array {name} geometry or dtype differs"
            )
        if expected_dtype is np.float32:
            if name.endswith("elevation_m"):
                valid_name = (
                    "detail_valid_mask"
                    if name.startswith("detail_")
                    else "valid_mask"
                )
                valid = np.asarray(crop.arrays[valid_name], dtype=np.bool_)
                if not np.isfinite(values[valid]).all():
                    raise TaskCoverabilityError(
                        f"task source array {name} valid values are non-finite"
                    )
            elif not np.isfinite(values).all() or (values < 0.0).any():
                raise TaskCoverabilityError(
                    f"task source array {name} values are invalid"
                )
        output[name] = _readonly(values, expected_dtype)
    return output


def build_task_common(
    *,
    scene_index: FormalSceneIndex,
    scene_id: str,
    geometry: FrozenTaskGeometry,
    capability_bundle: FrozenCapabilityBundle,
) -> TaskCommonPayload:
    """Crop one shared task view before any platform-derived computation."""
    if not isinstance(geometry, FrozenTaskGeometry):
        raise TypeError("task common builder requires FrozenTaskGeometry")
    if not isinstance(capability_bundle, FrozenCapabilityBundle):
        raise TypeError("task common builder requires FrozenCapabilityBundle")
    halo_contract = derive_task_evidence_halo(capability_bundle)
    expected_halo = _expected_halo_bounds(
        geometry,
        halo_contract.coarse_cells,
    )
    if geometry.halo_coarse_bounds_half_open != expected_halo:
        raise TaskCoverabilityError(
            "task geometry halo differs from frozen capability authority"
        )
    crop = load_task_source_crop(
        scene_index=scene_index,
        scene_id=scene_id,
        geometry=geometry,
    )
    if not isinstance(crop, TaskSourceCrop):
        raise TaskCoverabilityError("task source crop type differs")
    source_identity = scene_index.identity.source_lock_file_sha256
    if (
        crop.scene_id != scene_id
        or crop.source_identity_sha256 != source_identity
        or crop.coarse_bounds_half_open != expected_halo
        or crop.detail_bounds_half_open
        != tuple(value * DETAIL_PER_GLOBAL for value in expected_halo)
    ):
        raise TaskCoverabilityError("task source crop identity or bounds differ")
    arrays = _validate_source_arrays(crop, geometry)
    halo_row0, _, halo_column0, _ = expected_halo
    row0, row1, column0, column1 = geometry.coarse_bounds_half_open
    local_rows = slice(row0 - halo_row0, row1 - halo_row0)
    local_columns = slice(
        column0 - halo_column0,
        column1 - halo_column0,
    )
    detail_rows = slice(
        (row0 - halo_row0) * DETAIL_PER_GLOBAL,
        (row1 - halo_row0) * DETAIL_PER_GLOBAL,
    )
    detail_columns = slice(
        (column0 - halo_column0) * DETAIL_PER_GLOBAL,
        (column1 - halo_column0) * DETAIL_PER_GLOBAL,
    )

    authorization = np.zeros(arrays["valid_mask"].shape, dtype=np.bool_)
    authorization[local_rows, local_columns] = True
    detail_authorization = np.zeros(
        arrays["detail_valid_mask"].shape,
        dtype=np.bool_,
    )
    detail_authorization[detail_rows, detail_columns] = True
    local_valid = arrays["valid_mask"][local_rows, local_columns]
    local_forbidden = arrays["forbidden_ratio"][local_rows, local_columns]
    local_obstacle = arrays["physical_obstacle_ratio"][
        local_rows,
        local_columns,
    ]
    mission_roi = np.ascontiguousarray(
        local_valid & (local_forbidden == 0.0) & (local_obstacle == 0.0),
        dtype=np.bool_,
    )
    detail_valid = arrays["detail_valid_mask"][detail_rows, detail_columns]
    detail_forbidden = arrays["detail_forbidden_ratio"][
        detail_rows,
        detail_columns,
    ]
    detail_obstacle = arrays["detail_physical_obstacle_ratio"][
        detail_rows,
        detail_columns,
    ]
    mission_target = np.ascontiguousarray(
        detail_valid
        & (detail_forbidden == 0.0)
        & (detail_obstacle == 0.0),
        dtype=np.bool_,
    )
    arrays.update(
        {
            "task_authorization_mask": _readonly(authorization, np.bool_),
            "evidence_only_mask": _readonly(~authorization, np.bool_),
            "detail_task_authorization_mask": _readonly(
                detail_authorization,
                np.bool_,
            ),
            "mission_roi_mask": _readonly(mission_roi, np.bool_),
            "mission_target_detail_mask": _readonly(
                mission_target,
                np.bool_,
            ),
        }
    )
    return TaskCommonPayload(
        geometry=geometry,
        local_world_bounds_m=crop.local_world_bounds_m,
        local_halo_world_bounds_m=crop.local_halo_world_bounds_m,
        arrays=MappingProxyType(arrays),
        source_identity_sha256=source_identity,
    )


def _local_slices(
    geometry: FrozenTaskGeometry,
) -> tuple[slice, slice, slice, slice]:
    halo_row0, _, halo_column0, _ = (
        geometry.halo_coarse_bounds_half_open
    )
    row0, row1, column0, column1 = geometry.coarse_bounds_half_open
    coarse_rows = slice(row0 - halo_row0, row1 - halo_row0)
    coarse_columns = slice(
        column0 - halo_column0,
        column1 - halo_column0,
    )
    detail_rows = slice(
        coarse_rows.start * DETAIL_PER_GLOBAL,
        coarse_rows.stop * DETAIL_PER_GLOBAL,
    )
    detail_columns = slice(
        coarse_columns.start * DETAIL_PER_GLOBAL,
        coarse_columns.stop * DETAIL_PER_GLOBAL,
    )
    return coarse_rows, coarse_columns, detail_rows, detail_columns


def _task_projected_view(
    common: TaskCommonArtifact,
    *,
    detail: bool,
) -> object:
    geometry = common.geometry
    coarse_rows, coarse_columns, detail_rows, detail_columns = _local_slices(
        geometry
    )
    if detail:
        rows, columns = detail_rows, detail_columns
        prefix = "detail_"
        cells = geometry.span_cells * DETAIL_PER_GLOBAL
        resolution_m = 4.0 / DETAIL_PER_GLOBAL
        valid = np.ascontiguousarray(
            common.arrays["detail_valid_mask"][rows, columns],
            dtype=np.bool_,
        )
    else:
        rows, columns = coarse_rows, coarse_columns
        prefix = ""
        cells = geometry.span_cells
        resolution_m = 4.0
        valid = np.ascontiguousarray(
            common.arrays["valid_mask"][rows, columns]
            & common.arrays["mission_roi_mask"],
            dtype=np.bool_,
        )
    canvas = MapCanvas(
        common.key.scene_id,
        common.local_world_bounds_m,
        GridGeometry(cells * resolution_m, resolution_m, cells),
    )
    return SimpleNamespace(
        canvas=canvas,
        elevation_m=np.ascontiguousarray(
            common.arrays[f"{prefix}elevation_m"][rows, columns],
            dtype=np.float32,
        ),
        valid_mask=valid,
        physical_obstacle_ratio=np.ascontiguousarray(
            common.arrays[f"{prefix}physical_obstacle_ratio"][rows, columns],
            dtype=np.float32,
        ),
        physical_obstacle_height_m=np.ascontiguousarray(
            common.arrays[
                f"{prefix}physical_obstacle_height_m"
            ][rows, columns],
            dtype=np.float32,
        ),
        forbidden_ratio=np.ascontiguousarray(
            common.arrays[f"{prefix}forbidden_ratio"][rows, columns],
            dtype=np.float32,
        ),
    )


def _projection_mask(
    projection: object,
    name: str,
    *,
    platform_type: str,
    cells: int,
) -> np.ndarray:
    values = getattr(projection, name, None)
    if getattr(projection, "platform_type", None) != platform_type:
        raise TaskCoverabilityError("task static projection platform differs")
    if (
        not isinstance(values, np.ndarray)
        or values.dtype not in (np.dtype(np.uint8), np.dtype(np.bool_))
        or values.shape != (cells, cells)
        or not values.flags.c_contiguous
        or (values.dtype == np.dtype(np.uint8) and (values > 1).any())
    ):
        raise TaskCoverabilityError(
            f"task static projection {name} geometry differs"
        )
    return np.ascontiguousarray(np.flipud(values).astype(np.bool_))


def _padded_detail_tile(
    projected: object,
    *,
    start_row: int,
    start_column: int,
) -> tuple[object, slice, slice, int, int]:
    total = projected.canvas.geometry.cells
    core_rows = min(_DETAIL_STATIC_TILE_CELLS, total - start_row)
    core_columns = min(_DETAIL_STATIC_TILE_CELLS, total - start_column)
    cells = _DETAIL_STATIC_TILE_CELLS + 2 * _DETAIL_STATIC_HALO_CELLS
    logical_row0 = start_row - _DETAIL_STATIC_HALO_CELLS
    logical_column0 = start_column - _DETAIL_STATIC_HALO_CELLS
    source_row0 = max(0, logical_row0)
    source_column0 = max(0, logical_column0)
    source_row1 = min(total, logical_row0 + cells)
    source_column1 = min(total, logical_column0 + cells)
    target_row0 = source_row0 - logical_row0
    target_column0 = source_column0 - logical_column0
    target_rows = slice(target_row0, target_row0 + source_row1 - source_row0)
    target_columns = slice(
        target_column0,
        target_column0 + source_column1 - source_column0,
    )
    source_rows = slice(source_row0, source_row1)
    source_columns = slice(source_column0, source_column1)

    arrays: dict[str, np.ndarray] = {}
    for name in (
        "elevation_m",
        "physical_obstacle_ratio",
        "physical_obstacle_height_m",
        "forbidden_ratio",
    ):
        values = np.zeros((cells, cells), dtype=np.float32)
        values[target_rows, target_columns] = getattr(projected, name)[
            source_rows,
            source_columns,
        ]
        arrays[name] = values
    valid = np.zeros((cells, cells), dtype=np.bool_)
    valid[target_rows, target_columns] = projected.valid_mask[
        source_rows,
        source_columns,
    ]
    resolution_m = projected.canvas.geometry.resolution_m
    left, _, _, top = projected.canvas.bounds_m
    window_left = left + logical_column0 * resolution_m
    window_top = top - logical_row0 * resolution_m
    canvas = MapCanvas(
        projected.canvas.window_sha256,
        (
            window_left,
            window_top - cells * resolution_m,
            window_left + cells * resolution_m,
            window_top,
        ),
        GridGeometry(cells * resolution_m, resolution_m, cells),
    )
    return (
        SimpleNamespace(canvas=canvas, valid_mask=valid, **arrays),
        slice(
            _DETAIL_STATIC_HALO_CELLS,
            _DETAIL_STATIC_HALO_CELLS + core_rows,
        ),
        slice(
            _DETAIL_STATIC_HALO_CELLS,
            _DETAIL_STATIC_HALO_CELLS + core_columns,
        ),
        core_rows,
        core_columns,
    )


def _detail_intrinsic_projection(
    *,
    common: TaskCommonArtifact,
    platform: FrozenPlatformCapability,
    bridge: object,
    projected: object,
) -> tuple[np.ndarray, int]:
    total = projected.canvas.geometry.cells
    intrinsic = np.zeros((total, total), dtype=np.bool_)
    scene = SimpleNamespace(scene_id=common.key.scene_id)
    calls = 0
    for start_row in range(0, total, _DETAIL_STATIC_TILE_CELLS):
        for start_column in range(0, total, _DETAIL_STATIC_TILE_CELLS):
            window, crop_rows, crop_columns, core_rows, core_columns = (
                _padded_detail_tile(
                    projected,
                    start_row=start_row,
                    start_column=start_column,
                )
            )
            if not bool(window.valid_mask.any()):
                continue
            output = bridge.project_traversability(
                _projection_request(platform, scene, window)
            )
            calls += 1
            projected_intrinsic = _projection_mask(
                output,
                "intrinsic_feasible",
                platform_type=platform.platform_type,
                cells=window.canvas.geometry.cells,
            )
            intrinsic[
                start_row : start_row + core_rows,
                start_column : start_column + core_columns,
            ] = projected_intrinsic[crop_rows, crop_columns]
    return np.ascontiguousarray(intrinsic), calls


def _qualified_local_start(
    *,
    common: TaskCommonArtifact,
    platform: FrozenPlatformCapability,
    qualified_start: Mapping[str, object],
) -> tuple[tuple[int, int], str]:
    if not isinstance(qualified_start, Mapping):
        raise TypeError("task coverability start must be a mapping")
    if qualified_start.get("qualified") is not True:
        raise TaskCoverabilityError("task coverability start is not qualified")
    raw_cell = qualified_start.get("qualified_start_cell")
    if (
        not isinstance(raw_cell, (tuple, list))
        or len(raw_cell) != 2
        or any(type(value) is not int for value in raw_cell)
    ):
        raise TaskCoverabilityError("task coverability start cell is invalid")
    row0, row1, column0, column1 = (
        common.geometry.coarse_bounds_half_open
    )
    global_cell = int(raw_cell[0]), int(raw_cell[1])
    if not (
        row0 <= global_cell[0] < row1
        and column0 <= global_cell[1] < column1
    ):
        raise TaskCoverabilityError("qualified start leaves the task ROI")
    local_cell = global_cell[0] - row0, global_cell[1] - column0
    if local_cell != common.geometry.local_start_cell:
        raise TaskCoverabilityError("qualified start local identity differs")
    if qualified_start.get(
        "capability_content_sha256"
    ) != _physical_capability_content_sha256(platform):
        raise TaskCoverabilityError("qualified start capability identity differs")
    start_identity = qualified_start.get("start_identity_sha256")
    if (
        not isinstance(start_identity, str)
        or len(start_identity) != 64
        or any(value not in "0123456789abcdef" for value in start_identity)
    ):
        raise TaskCoverabilityError("qualified start identity is invalid")
    return local_cell, start_identity


def _visibility_sources(
    pose_cells: tuple[tuple[int, int], ...],
    *,
    start_cell: tuple[int, int],
    window_cells: int,
) -> tuple[tuple[int, int], ...]:
    stride = max(1, window_cells // 2)
    representatives: dict[
        tuple[int, int], tuple[tuple[int, int, int], tuple[int, int]]
    ] = {}
    for row, column in pose_cells:
        bucket = row // stride, column // stride
        center = (
            bucket[0] * stride + stride // 2,
            bucket[1] * stride + stride // 2,
        )
        candidate = (
            (
                (row - center[0]) ** 2 + (column - center[1]) ** 2,
                row,
                column,
            ),
            (row, column),
        )
        current = representatives.get(bucket)
        if current is None or candidate[0] < current[0]:
            representatives[bucket] = candidate
    ordered = [start_cell]
    selected = {start_cell}
    for bucket in sorted(representatives):
        coordinate = representatives[bucket][1]
        if coordinate not in selected:
            ordered.append(coordinate)
            selected.add(coordinate)
    for coordinate in pose_cells:
        if coordinate not in selected:
            ordered.append(coordinate)
            selected.add(coordinate)
    return tuple(ordered)


def _reveal_task_window(
    *,
    truth: np.ndarray,
    pose_cell: tuple[int, int],
    estimator: object,
    radius_cells: int,
) -> tuple[slice, slice, np.ndarray]:
    cells = 2 * radius_cells + 1
    start_row = pose_cell[0] - radius_cells
    start_column = pose_cell[1] - radius_cells
    source_row0 = max(0, start_row)
    source_column0 = max(0, start_column)
    source_row1 = min(truth.shape[0], start_row + cells)
    source_column1 = min(truth.shape[1], start_column + cells)
    target_row0 = source_row0 - start_row
    target_column0 = source_column0 - start_column
    window = np.ones((cells, cells), dtype=np.float32)
    window[
        target_row0 : target_row0 + source_row1 - source_row0,
        target_column0 : target_column0 + source_column1 - source_column0,
    ] = truth[source_row0:source_row1, source_column0:source_column1]
    visible = estimator.reveal_from_pose(
        window,
        (radius_cells, radius_cells),
    )
    if (
        not isinstance(visible, np.ndarray)
        or visible.dtype != np.dtype(np.bool_)
        or visible.shape != window.shape
        or not visible.flags.c_contiguous
    ):
        raise TaskCoverabilityError("task visibility result geometry differs")
    return (
        slice(source_row0, source_row1),
        slice(source_column0, source_column1),
        np.ascontiguousarray(
            visible[
                target_row0 : target_row0 + source_row1 - source_row0,
                target_column0 : target_column0
                + source_column1
                - source_column0,
            ]
        ),
    )


def _initial_candidate_count(
    *,
    mission_roi: np.ndarray,
    reachable: np.ndarray,
    start_cell: tuple[int, int],
    initial_observed_detail: np.ndarray,
    mission_target: np.ndarray,
    truth_obstacle_ratio: np.ndarray,
    estimator: object,
    sensor: SensorGeometry,
) -> int:
    span = reachable.shape[0]
    observed = initial_observed_detail.reshape(
        span,
        DETAIL_PER_GLOBAL,
        span,
        DETAIL_PER_GLOBAL,
    ).any(axis=(1, 3))
    unknown_roi = mission_roi & ~observed
    adjacent_unknown = np.zeros_like(observed)
    adjacent_unknown[1:] |= unknown_roi[:-1]
    adjacent_unknown[:-1] |= unknown_roi[1:]
    adjacent_unknown[:, 1:] |= unknown_roi[:, :-1]
    adjacent_unknown[:, :-1] |= unknown_roi[:, 1:]
    boundary = observed & mission_roi & adjacent_unknown
    segments = _frontier_chains(_points(boundary), span)
    standoff_cells = max(
        1,
        round(sensor.standoff_m / 4.0),
    )
    mapped: set[tuple[int, int]] = set()
    for segment in segments:
        mapped.update(
            candidate.pose_cell
            for candidate in _map_frontier_chain_candidates(
                segment,
                robot=start_cell,
                observed_mask=np.ascontiguousarray(observed),
                physical_pose_mask=reachable,
                standoff_cells=standoff_cells,
            )
        )
    if not mapped:
        return 0
    detail_candidates = np.ascontiguousarray(
        [
            (
                row * DETAIL_PER_GLOBAL + DETAIL_PER_GLOBAL // 2,
                column * DETAIL_PER_GLOBAL + DETAIL_PER_GLOBAL // 2,
            )
            for row, column in sorted(mapped)
        ],
        dtype=np.int32,
    ).reshape((-1, 2))
    remaining = np.ascontiguousarray(
        mission_target & ~initial_observed_detail,
        dtype=np.float32,
    )
    gains = estimator.estimate_candidate_gains(
        np.ascontiguousarray(initial_observed_detail, dtype=np.bool_),
        truth_obstacle_ratio,
        remaining,
        remaining,
        detail_candidates,
    )
    if (
        not isinstance(gains, np.ndarray)
        or gains.dtype != np.dtype(np.float32)
        or gains.shape != (len(detail_candidates), 2)
        or not gains.flags.c_contiguous
        or not np.isfinite(gains).all()
        or (gains < 0.0).any()
    ):
        raise TaskCoverabilityError("initial candidate gain result differs")
    return int((gains[:, 0] >= 1.0).sum(dtype=np.int64))


def _canonical_position_m(values: object) -> tuple[float, float, float]:
    array = np.ascontiguousarray(
        np.asarray(values, dtype=np.float64).reshape((1, 3))
    )
    if not np.isfinite(array).all():
        raise TaskCoverabilityError("hopper task position is non-finite")
    micrometres = canonical_physical_positions_um(array)[0]
    return tuple(float(value) / 1_000_000.0 for value in micrometres)


def _hopper_capsule_patch(
    *,
    detail: object,
    positions_m: tuple[tuple[float, float, float], ...],
    radius_m: float,
) -> tuple[slice, slice, np.ndarray]:
    if not positions_m:
        raise TaskCoverabilityError("hopper task trajectory is empty")
    xy = np.ascontiguousarray(
        [(position[0], position[1]) for position in positions_m],
        dtype=np.float64,
    )
    resolution_m = float(detail.canvas.geometry.resolution_m)
    total = int(detail.canvas.geometry.cells)
    left, _bottom, _right, top = detail.canvas.bounds_m
    first_column = max(
        0,
        int(math.ceil((float(xy[:, 0].min()) - radius_m - left)
                      / resolution_m - 0.5)),
    )
    last_column = min(
        total - 1,
        int(math.floor((float(xy[:, 0].max()) + radius_m - left)
                       / resolution_m - 0.5)),
    )
    first_row = max(
        0,
        int(math.ceil((top - (float(xy[:, 1].max()) + radius_m))
                      / resolution_m - 0.5)),
    )
    last_row = min(
        total - 1,
        int(math.floor((top - (float(xy[:, 1].min()) - radius_m))
                       / resolution_m - 0.5)),
    )
    if first_row > last_row or first_column > last_column:
        raise TaskCoverabilityError("hopper task trajectory leaves the ROI")
    rows = np.arange(first_row, last_row + 1, dtype=np.int64)
    columns = np.arange(first_column, last_column + 1, dtype=np.int64)
    y = top - (rows.astype(np.float64) + 0.5) * resolution_m
    x = left + (columns.astype(np.float64) + 0.5) * resolution_m
    yy = y[:, None]
    xx = x[None, :]
    minimum_distance_squared = np.full(
        (len(rows), len(columns)), np.inf, dtype=np.float64
    )
    segments = tuple(zip(xy[:-1], xy[1:], strict=True))
    if not segments:
        segments = ((xy[0], xy[0]),)
    for start, end in segments:
        dx = float(end[0] - start[0])
        dy = float(end[1] - start[1])
        length_squared = dx * dx + dy * dy
        if length_squared == 0.0:
            candidate = (xx - start[0]) ** 2 + (yy - start[1]) ** 2
        else:
            alpha = np.clip(
                ((xx - start[0]) * dx + (yy - start[1]) * dy)
                / length_squared,
                0.0,
                1.0,
            )
            candidate = (
                (xx - (start[0] + alpha * dx)) ** 2
                + (yy - (start[1] + alpha * dy)) ** 2
            )
        np.minimum(minimum_distance_squared, candidate, out=minimum_distance_squared)
    return (
        slice(first_row, last_row + 1),
        slice(first_column, last_column + 1),
        np.ascontiguousarray(
            minimum_distance_squared <= radius_m * radius_m,
            dtype=np.bool_,
        ),
    )


def _observed_coarse_task_view(
    *,
    common: TaskCommonArtifact,
    coarse_truth: object,
    detail_truth: object,
    observed_detail: np.ndarray,
) -> object:
    span = common.geometry.span_cells
    shape = (span, DETAIL_PER_GLOBAL, span, DETAIL_PER_GLOBAL)
    known = observed_detail.reshape(shape)
    valid = np.ascontiguousarray(
        known.all(axis=(1, 3))
        & np.asarray(common.arrays["mission_roi_mask"], dtype=np.bool_),
        dtype=np.bool_,
    )

    def aggregate(values: np.ndarray, *, maximum: bool) -> np.ndarray:
        grouped = np.asarray(values).reshape(shape)
        if maximum:
            result = grouped.max(axis=(1, 3), initial=0.0)
        else:
            result = grouped.mean(axis=(1, 3), dtype=np.float64)
        return np.ascontiguousarray(
            np.where(valid, result, 0.0), dtype=np.float32
        )

    return SimpleNamespace(
        canvas=coarse_truth.canvas,
        elevation_m=aggregate(detail_truth.elevation_m, maximum=False),
        valid_mask=valid,
        physical_obstacle_ratio=aggregate(
            detail_truth.physical_obstacle_ratio, maximum=True
        ),
        physical_obstacle_height_m=aggregate(
            detail_truth.physical_obstacle_height_m, maximum=True
        ),
        forbidden_ratio=aggregate(detail_truth.forbidden_ratio, maximum=True),
    )


def _observed_local_detail_view(
    *,
    common: TaskCommonArtifact,
    detail_truth: object,
    observed_detail: np.ndarray,
    source_position_m: tuple[float, float, float],
) -> object:
    cells = _DETAIL_STATIC_TILE_CELLS
    resolution_m = float(detail_truth.canvas.geometry.resolution_m)
    pose_row, pose_column = detail_truth.canvas.world_to_grid(
        source_position_m[0], source_position_m[1]
    )
    logical_row0 = pose_row - cells // 2
    logical_column0 = pose_column - cells // 2
    total = int(detail_truth.canvas.geometry.cells)
    source_row0 = max(0, logical_row0)
    source_column0 = max(0, logical_column0)
    source_row1 = min(total, logical_row0 + cells)
    source_column1 = min(total, logical_column0 + cells)
    target_row0 = source_row0 - logical_row0
    target_column0 = source_column0 - logical_column0
    source_rows = slice(source_row0, source_row1)
    source_columns = slice(source_column0, source_column1)
    target_rows = slice(target_row0, target_row0 + source_row1 - source_row0)
    target_columns = slice(
        target_column0, target_column0 + source_column1 - source_column0
    )
    valid = np.zeros((cells, cells), dtype=np.bool_)
    valid[target_rows, target_columns] = observed_detail[
        source_rows, source_columns
    ]

    def local_values(name: str) -> np.ndarray:
        output = np.zeros((cells, cells), dtype=np.float32)
        source = np.asarray(getattr(detail_truth, name), dtype=np.float32)
        output[target_rows, target_columns] = source[source_rows, source_columns]
        output[~valid] = 0.0
        return output

    left, _bottom, _right, top = detail_truth.canvas.bounds_m
    window_left = left + logical_column0 * resolution_m
    window_top = top - logical_row0 * resolution_m
    canvas = MapCanvas(
        common.key.scene_id,
        (
            window_left,
            window_top - cells * resolution_m,
            window_left + cells * resolution_m,
            window_top,
        ),
        GridGeometry(cells * resolution_m, resolution_m, cells),
    )
    return SimpleNamespace(
        canvas=canvas,
        elevation_m=local_values("elevation_m"),
        valid_mask=np.ascontiguousarray(valid),
        physical_obstacle_ratio=local_values("physical_obstacle_ratio"),
        physical_obstacle_height_m=local_values(
            "physical_obstacle_height_m"
        ),
        forbidden_ratio=local_values("forbidden_ratio"),
    )


def _observed_local_detail_bounds(
    *,
    detail_truth: object,
    source_position_m: tuple[float, float, float],
) -> tuple[float, float, float, float]:
    cells = _DETAIL_STATIC_TILE_CELLS
    resolution_m = float(detail_truth.canvas.geometry.resolution_m)
    pose_row, pose_column = detail_truth.canvas.world_to_grid(
        source_position_m[0], source_position_m[1]
    )
    logical_row0 = pose_row - cells // 2
    logical_column0 = pose_column - cells // 2
    left, _bottom, _right, top = detail_truth.canvas.bounds_m
    window_left = left + logical_column0 * resolution_m
    window_top = top - logical_row0 * resolution_m
    return (
        window_left,
        window_top - cells * resolution_m,
        window_left + cells * resolution_m,
        window_top,
    )


def _hopper_disposition_name(value: object) -> str:
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return name
    text = str(value)
    return text.rsplit(".", 1)[-1]


def _hopper_edge_fits_task_roi(
    *,
    detail: object,
    detail_roi: np.ndarray,
    source_position_m: tuple[float, float, float],
    target_position_m: tuple[float, float, float],
    landing_boundary_m: np.ndarray,
    platform: FrozenPlatformCapability,
) -> bool:
    typed = platform.typed_capability
    landing_radius = max(
        math.hypot(
            float(point[0]) - target_position_m[0],
            float(point[1]) - target_position_m[1],
        )
        for point in landing_boundary_m
    )
    radius_m = (
        float(getattr(typed, "flight_collision_radius_m"))
        + float(getattr(typed, "flight_map_margin_m"))
        + landing_radius
    )
    if not math.isfinite(radius_m) or radius_m <= 0.0:
        raise TaskCoverabilityError("hopper task flight tube radius is invalid")
    left, bottom, right, top = detail.canvas.bounds_m
    if (
        min(source_position_m[0], target_position_m[0]) - radius_m < left
        or max(source_position_m[0], target_position_m[0]) + radius_m > right
        or min(source_position_m[1], target_position_m[1]) - radius_m < bottom
        or max(source_position_m[1], target_position_m[1]) + radius_m > top
    ):
        return False
    resolution_m = float(detail.canvas.geometry.resolution_m)
    rows, columns, touched = _hopper_capsule_patch(
        detail=detail,
        positions_m=(source_position_m, target_position_m),
        radius_m=radius_m + math.sqrt(2.0) * resolution_m / 2.0,
    )
    return not bool((touched & ~detail_roi[rows, columns]).any())


def _build_hopper_task_coverability(
    *,
    common: TaskCommonArtifact,
    platform: FrozenPlatformCapability,
    qualified_start: Mapping[str, object],
    bridge: object,
    incremental: bool,
    round_callback: object | None = None,
    metrics_output: dict[str, int] | None = None,
    resume_state: Mapping[str, object] | None = None,
) -> PlatformTaskPayload:
    """Build a simple exact HOPPER closure on the task-induced domain."""
    import lunar_planner_training_bridge as bridge_api

    if not isinstance(common, TaskCommonArtifact):
        raise TypeError("hopper task builder requires TaskCommonArtifact")
    if not isinstance(platform, FrozenPlatformCapability):
        raise TypeError("hopper task builder requires FrozenPlatformCapability")
    if platform.platform_type != "HOPPER":
        raise TaskCoverabilityError("hopper task builder platform is invalid")
    for method in (
        "project_traversability",
        "project_hopper_landing_evidence",
        "project_hopper_incremental_edges",
    ):
        if not callable(getattr(bridge, method, None)):
            raise TypeError(f"hopper task builder requires bridge {method}")
    start_cell, start_identity = _qualified_local_start(
        common=common,
        platform=platform,
        qualified_start=qualified_start,
    )
    observation = platform.observation_capability
    if observation.sensor_range_m != 30.0 or not math.isclose(
        observation.sensor_fov_rad, 2.0 * math.pi, rel_tol=0.0, abs_tol=1e-9
    ):
        raise TaskCoverabilityError("hopper task sensor authority differs")

    scene = SimpleNamespace(scene_id=common.key.scene_id)
    coarse_truth = _task_projected_view(common, detail=False)
    detail_truth = _task_projected_view(common, detail=True)
    if not bool(coarse_truth.valid_mask[start_cell]):
        raise TaskCoverabilityError("qualified Hopper start is outside task ROI")
    detail_intrinsic, detail_static_calls = _detail_intrinsic_projection(
        common=common,
        platform=platform,
        bridge=bridge,
        projected=detail_truth,
    )
    mission_target = np.ascontiguousarray(
        np.asarray(common.arrays["mission_target_detail_mask"], dtype=np.bool_)
        & detail_intrinsic,
        dtype=np.bool_,
    )
    start_xy = coarse_truth.canvas.grid_center_world(*start_cell)
    start_position = _canonical_position_m(
        (*start_xy, float(coarse_truth.elevation_m[start_cell]))
    )
    if coarse_truth.canvas.world_to_grid(*start_position[:2]) != start_cell:
        raise TaskCoverabilityError("qualified Hopper start position drifted")

    def initial_observation(
        position: tuple[float, float, float],
    ) -> np.ndarray:
        observed = np.zeros(mission_target.shape, dtype=np.bool_)
        rows, columns, visible = _hopper_capsule_patch(
            detail=detail_truth,
            positions_m=(position,),
            radius_m=float(observation.sensor_range_m),
        )
        observed[rows, columns] |= (
            visible & detail_truth.valid_mask[rows, columns]
        )
        return observed

    # The lightweight scene index freezes a coarse safe start.  Resolve its
    # exact 0.2 m landing elevation once before the closure so the source pose,
    # edge projector and persisted physical position share one authority.
    observed_detail = initial_observation(start_position)
    provisional_coarse = _observed_coarse_task_view(
        common=common,
        coarse_truth=coarse_truth,
        detail_truth=detail_truth,
        observed_detail=observed_detail,
    )
    provisional_local = _observed_local_detail_view(
        common=common,
        detail_truth=detail_truth,
        observed_detail=observed_detail,
        source_position_m=start_position,
    )
    start_request = _projection_request(
        platform,
        scene,
        provisional_coarse,
        start_cell=start_cell,
        local_projected=provisional_local,
        exact_start_position_m=start_position,
    )
    start_request.request_id = (
        f"task-cache/{common.key.scene_id}/hopper-start-qualification"
    )
    start_request.config.global_map.base_resolution_m = 0.2
    start_request.config.global_map.maximum_level = 5
    start_request.config.global_map.target_axis_cells = 256
    start_landing = bridge.project_hopper_landing_evidence(
        start_request,
        np.ascontiguousarray(
            np.asarray((start_position,), dtype=np.float64)
        ),
    )
    start_certified = getattr(start_landing, "certified", None)
    start_aim = getattr(start_landing, "aim_positions_m", None)
    if (
        not isinstance(start_certified, np.ndarray)
        or start_certified.dtype != np.dtype(np.bool_)
        or start_certified.shape != (1,)
        or not isinstance(start_aim, np.ndarray)
        or start_aim.dtype != np.dtype(np.float64)
        or start_aim.shape != (1, 3)
        or not np.isfinite(start_aim).all()
        or not bool(start_certified[0])
    ):
        raise TaskCoverabilityError(
            "hopper task exact start landing evidence differs"
        )
    start_position = _canonical_position_m(start_aim[0])
    if coarse_truth.canvas.world_to_grid(*start_position[:2]) != start_cell:
        raise TaskCoverabilityError(
            "hopper task exact start landing leaves its cell"
        )
    observed_detail = initial_observation(start_position)
    initial_observed_coverable = np.ascontiguousarray(
        observed_detail & mission_target, dtype=np.bool_
    )

    reached_positions: dict[
        tuple[int, int], tuple[float, float, float]
    ] = {start_cell: start_position}
    processed_certified_edges: set[
        tuple[tuple[int, int], tuple[int, int]]
    ] = set()
    edge_states: dict[
        tuple[tuple[int, int], tuple[int, int]],
        tuple[str, str, tuple[int, ...]],
    ] = {}
    safe_union = np.zeros((common.geometry.span_cells,) * 2, dtype=np.bool_)
    landing_algorithm_id: str | None = None
    edge_algorithm_id: str | None = None
    round_index = 0
    first_round_candidate_count = 0
    trajectory_edge_patch_count = 0
    landing_evidence_call_count = 0
    edge_projection_call_count = 0
    native_landing_evidence_calls = 1
    native_edge_projection_calls = 0
    outside_roi_support_count = 0
    mission_roi = np.asarray(common.arrays["mission_roi_mask"], dtype=np.bool_)
    detail_roi = np.ascontiguousarray(
        np.repeat(
            np.repeat(mission_roi, DETAIL_PER_GLOBAL, axis=0),
            DETAIL_PER_GLOBAL,
            axis=1,
        ),
        dtype=np.bool_,
    )
    span = common.geometry.span_cells
    local_bounds = tuple(float(value) for value in common.local_world_bounds_m)
    pending_sources: set[tuple[int, int]] = {start_cell}
    source_process_count = 0
    global_map_assembly_count = 0

    if resume_state is not None:
        if not incremental or not isinstance(resume_state, Mapping):
            raise TaskCoverabilityError("hopper closure resume state is invalid")
        try:
            resumed_observed = np.ascontiguousarray(
                np.asarray(resume_state["observed_detail"], dtype=np.bool_)
            )
            resumed_safe = np.ascontiguousarray(
                np.asarray(resume_state["safe_union"], dtype=np.bool_)
            )
            resumed_reached = dict(resume_state["reached_positions"])
            resumed_processed = set(
                resume_state["processed_certified_edges"]
            )
            resumed_edges = dict(resume_state["edge_states"])
            resumed_pending = set(resume_state["pending_sources"])
            resumed_round = int(resume_state["completed_round"])
            resumed_first_count = int(
                resume_state["first_round_candidate_count"]
            )
            resumed_patch_count = int(
                resume_state["trajectory_edge_patch_count"]
            )
            resumed_logical_calls = int(
                resume_state["logical_projection_call_count"]
            )
            resumed_support_count = int(
                resume_state["outside_roi_support_count"]
            )
            resumed_landing_algorithm = resume_state[
                "landing_algorithm_id"
            ]
            resumed_edge_algorithm = resume_state["edge_algorithm_id"]
        except (KeyError, TypeError, ValueError) as error:
            raise TaskCoverabilityError(
                "hopper closure resume state is incomplete"
            ) from error
        if (
            resumed_observed.shape != observed_detail.shape
            or resumed_safe.shape != safe_union.shape
            or resumed_round < 0
            or resumed_first_count < 0
            or resumed_patch_count < 0
            or resumed_logical_calls < 0
            or resumed_support_count < 0
            or not isinstance(resumed_landing_algorithm, str)
            or not resumed_landing_algorithm
            or not isinstance(resumed_edge_algorithm, str)
            or not resumed_edge_algorithm
            or not resumed_pending.issubset(resumed_reached)
            or start_cell not in resumed_reached
            or resumed_reached[start_cell] != start_position
            or bool(initial_observed_coverable.any())
            and bool((initial_observed_coverable & ~resumed_observed).any())
        ):
            raise TaskCoverabilityError(
                "hopper closure resume state authority differs"
            )
        observed_detail = resumed_observed
        safe_union = resumed_safe
        reached_positions = resumed_reached
        processed_certified_edges = resumed_processed
        edge_states = resumed_edges
        pending_sources = resumed_pending
        round_index = resumed_round
        first_round_candidate_count = resumed_first_count
        trajectory_edge_patch_count = resumed_patch_count
        landing_evidence_call_count = resumed_logical_calls
        edge_projection_call_count = resumed_logical_calls
        outside_roi_support_count = resumed_support_count
        landing_algorithm_id = resumed_landing_algorithm
        edge_algorithm_id = resumed_edge_algorithm

    while True:
        global_map_assembly_count += 1
        coarse_observed = _observed_coarse_task_view(
            common=common,
            coarse_truth=coarse_truth,
            detail_truth=detail_truth,
            observed_detail=observed_detail,
        )
        observed_safe = np.ascontiguousarray(
            coarse_observed.valid_mask
            & (coarse_observed.physical_obstacle_ratio == 0.0)
            & (coarse_observed.forbidden_ratio == 0.0)
            & mission_roi,
            dtype=np.bool_,
        )
        proposed: dict[
            tuple[tuple[int, int], tuple[int, int]],
            tuple[tuple[float, float, float], float],
        ] = {}
        generation = round_index + 1
        logical_source_count = len(reached_positions)
        landing_evidence_call_count += logical_source_count
        edge_projection_call_count += logical_source_count
        sources_this_round = (
            tuple(sorted(pending_sources))
            if incremental
            else tuple(sorted(reached_positions))
        )
        pending_sources.clear()
        if incremental and not sources_this_round:
            break
        for source_cell in sources_this_round:
            source_process_count += 1
            source_position = reached_positions[source_cell]
            if (
                not bool(mission_roi[source_cell])
                or coarse_truth.canvas.world_to_grid(*source_position[:2])
                != source_cell
            ):
                raise TaskCoverabilityError(
                    "hopper task landed source leaves the ROI"
                )
            local_observed = _observed_local_detail_view(
                common=common,
                detail_truth=detail_truth,
                observed_detail=observed_detail,
                source_position_m=source_position,
            )
            request = _projection_request(
                platform,
                scene,
                coarse_observed,
                start_cell=source_cell,
                local_projected=local_observed,
                exact_start_position_m=source_position,
            )
            request.request_id = (
                f"task-cache/{common.key.scene_id}/hopper-reference/"
                f"{generation}/{source_cell[0]}-{source_cell[1]}"
            )
            request.global_map_generation = generation
            request.local_map_generation = generation
            request.state_time.nanoseconds_since_epoch = (
                1_000_000_000 + generation * 1_000_000
            )
            request.world.map_from_odom.stamp.nanoseconds_since_epoch = (
                request.state_time.nanoseconds_since_epoch
            )
            request.config.global_map.base_resolution_m = 0.2
            request.config.global_map.maximum_level = 5
            request.config.global_map.target_axis_cells = 256

            local_left, local_bottom, local_right, local_top = (
                local_observed.canvas.bounds_m
            )
            candidate_cells = []
            for raw_cell in np.argwhere(observed_safe):
                cell = tuple(int(value) for value in raw_cell)
                target_x, target_y = coarse_truth.canvas.grid_center_world(
                    *cell
                )
                if (
                    local_left <= target_x < local_right
                    and local_bottom <= target_y < local_top
                ):
                    candidate_cells.append(cell)
            if source_cell not in candidate_cells:
                candidate_cells.append(source_cell)
                candidate_cells.sort()
            targets = np.ascontiguousarray(
                [
                    source_position
                    if cell == source_cell
                    else _canonical_position_m(
                        (
                            *coarse_truth.canvas.grid_center_world(*cell),
                            float(coarse_observed.elevation_m[cell]),
                        )
                    )
                    for cell in candidate_cells
                ],
                dtype=np.float64,
            ).reshape((-1, 3))
            landing = bridge.project_hopper_landing_evidence(request, targets)
            native_landing_evidence_calls += 1
            certified = getattr(landing, "certified", None)
            aim = getattr(landing, "aim_positions_m", None)
            boundary = getattr(landing, "boundary_m", None)
            area = getattr(landing, "area_m2", None)
            algorithm_id = getattr(landing, "algorithm_id", None)
            count = len(candidate_cells)
            if (
                not isinstance(certified, np.ndarray)
                or certified.dtype != np.dtype(np.bool_)
                or certified.shape != (count,)
                or not isinstance(aim, np.ndarray)
                or aim.dtype != np.dtype(np.float64)
                or aim.shape != (count, 3)
                or not np.isfinite(aim).all()
                or not isinstance(boundary, np.ndarray)
                or boundary.dtype != np.dtype(np.float64)
                or boundary.shape != (count, 4, 3)
                or not np.isfinite(boundary).all()
                or not isinstance(area, np.ndarray)
                or area.dtype != np.dtype(np.float64)
                or area.shape != (count,)
                or not np.isfinite(area).all()
                or (area < 0.0).any()
                or not isinstance(algorithm_id, str)
                or not algorithm_id
            ):
                raise TaskCoverabilityError(
                    "hopper task landing evidence geometry differs"
                )
            if landing_algorithm_id is None:
                landing_algorithm_id = algorithm_id
            elif landing_algorithm_id != algorithm_id:
                raise TaskCoverabilityError(
                    "hopper task landing evidence algorithm drifted"
                )
            north_certified = np.zeros((span, span), dtype=np.bool_)
            north_aim = np.zeros((span, span, 3), dtype=np.float64)
            north_boundary = np.zeros((span, span, 4, 3), dtype=np.float64)
            north_area = np.zeros((span, span), dtype=np.float64)
            for index, cell in enumerate(candidate_cells):
                if not bool(certified[index]):
                    continue
                position = _canonical_position_m(aim[index])
                if coarse_truth.canvas.world_to_grid(*position[:2]) != cell:
                    raise TaskCoverabilityError(
                        "hopper task certified landing leaves its cell"
                    )
                support_inside = True
                for point in boundary[index]:
                    if not (
                        local_bounds[0] <= float(point[0]) <= local_bounds[2]
                        and local_bounds[1] <= float(point[1]) <= local_bounds[3]
                    ):
                        support_inside = False
                        break
                    try:
                        support_cell = coarse_truth.canvas.world_to_grid(
                            float(point[0]), float(point[1])
                        )
                    except ValueError:
                        support_inside = False
                        break
                    if not bool(mission_roi[support_cell]):
                        support_inside = False
                        break
                if not support_inside:
                    outside_roi_support_count += 1
                    continue
                north_certified[cell] = True
                north_aim[cell] = position
                north_boundary[cell] = boundary[index]
                north_area[cell] = area[index]
            if not bool(north_certified[source_cell]) or tuple(
                north_aim[source_cell]
            ) != source_position:
                raise TaskCoverabilityError(
                    "hopper task source landing evidence drifted"
                )
            safe_union |= north_certified
            evidence = bridge_api.HopperLandingEvidenceGrid(
                np.ascontiguousarray(np.flipud(north_certified)),
                np.ascontiguousarray(np.flipud(north_aim)),
                np.ascontiguousarray(np.flipud(north_boundary)),
                np.ascontiguousarray(np.flipud(north_area)),
                landing_algorithm_id,
            )
            authorized_target = np.ascontiguousarray(
                observed_safe & north_certified, dtype=np.bool_
            )
            authorized_target[source_cell] = False
            if incremental:
                for (old_source, old_target), old_state in edge_states.items():
                    if old_source == source_cell and (
                        old_state[0] == "STABLE_PHYSICAL_REJECTION"
                        or (old_source, old_target)
                        in processed_certified_edges
                    ):
                        authorized_target[old_target] = False
            for target_row, target_column in np.argwhere(authorized_target):
                target_cell = int(target_row), int(target_column)
                if _hopper_edge_fits_task_roi(
                    detail=detail_truth,
                    detail_roi=detail_roi,
                    source_position_m=source_position,
                    target_position_m=tuple(north_aim[target_cell]),
                    landing_boundary_m=north_boundary[target_cell],
                    platform=platform,
                ):
                    continue
                authorized_target[target_cell] = False
                edge_states[(source_cell, target_cell)] = (
                    "STABLE_PHYSICAL_REJECTION",
                    "HOPPER_TASK_FLIGHT_TUBE_LEAVES_ROI",
                    (),
                )
            task_target = np.ascontiguousarray(
                np.flipud(authorized_target), dtype=np.bool_
            )
            projected_edges = bridge.project_hopper_incremental_edges(
                request, evidence, task_target
            )
            native_edge_projection_calls += 1
            current_edge_algorithm = getattr(
                projected_edges, "algorithm_id", None
            )
            if (
                not isinstance(current_edge_algorithm, str)
                or not current_edge_algorithm
            ):
                raise TaskCoverabilityError(
                    "hopper task edge algorithm is invalid"
                )
            if edge_algorithm_id is None:
                edge_algorithm_id = current_edge_algorithm
            elif edge_algorithm_id != current_edge_algorithm:
                raise TaskCoverabilityError(
                    "hopper task edge algorithm drifted"
                )
            edges = getattr(projected_edges, "edges", None)
            if not isinstance(edges, tuple):
                raise TaskCoverabilityError("hopper task edge records differ")
            for edge in edges:
                target_index = getattr(edge, "target_index", None)
                if (
                    type(target_index) is not int
                    or not 0 <= target_index < span * span
                ):
                    raise TaskCoverabilityError(
                        "hopper task edge target index is invalid"
                    )
                raw_row, target_column = divmod(target_index, span)
                target_cell = span - 1 - raw_row, target_column
                if not bool(task_target[raw_row, target_column]):
                    raise TaskCoverabilityError(
                        "hopper task edge escapes its target mask"
                    )
                disposition = _hopper_disposition_name(edge.disposition)
                if disposition not in {
                    "CERTIFIED",
                    "STABLE_PHYSICAL_REJECTION",
                    "WAITING_EVIDENCE",
                }:
                    raise TaskCoverabilityError(
                        "hopper task edge disposition is invalid"
                    )
                reason_code = str(getattr(edge, "reason_code", ""))
                edge_key = source_cell, target_cell
                raw_dependencies = getattr(
                    edge, "dependency_tile_indices", None
                )
                if (
                    not isinstance(raw_dependencies, np.ndarray)
                    or raw_dependencies.dtype != np.dtype(np.int64)
                    or raw_dependencies.ndim != 1
                    or (raw_dependencies < 0).any()
                    or (raw_dependencies >= span * span).any()
                ):
                    raise TaskCoverabilityError(
                        "hopper task edge dependencies are invalid"
                    )
                dependencies = tuple(
                    sorted(int(value) for value in raw_dependencies)
                )
                edge_states[edge_key] = (
                    disposition,
                    reason_code,
                    dependencies,
                )
                if disposition != "CERTIFIED":
                    continue
                target_position = _canonical_position_m(
                    getattr(edge, "exact_target_position_m")
                )
                if (
                    not bool(mission_roi[target_cell])
                    or coarse_truth.canvas.world_to_grid(
                        *target_position[:2]
                    )
                    != target_cell
                ):
                    raise TaskCoverabilityError(
                        "hopper task certified edge leaves the ROI"
                    )
                if edge_key in processed_certified_edges:
                    old_position = reached_positions.get(target_cell)
                    if old_position is not None and old_position != target_position:
                        raise TaskCoverabilityError(
                            "hopper task certified landing drifted"
                        )
                    continue
                flight_time_s = float(edge.nominal_flight_time_s)
                if not math.isfinite(flight_time_s) or flight_time_s <= 0.0:
                    raise TaskCoverabilityError(
                        "hopper task certified flight time is invalid"
                    )
                proposed[edge_key] = target_position, flight_time_s
        if round_index == 0:
            first_round_candidate_count = len(proposed)
        if not proposed:
            break
        before_observed_mask = observed_detail.copy()
        before_observed = int(before_observed_mask.sum(dtype=np.int64))
        before_reached = len(reached_positions)
        newly_reached: set[tuple[int, int]] = set()
        round_visible = np.zeros(observed_detail.shape, dtype=np.bool_)
        for edge_key in sorted(proposed):
            source_cell, target_cell = edge_key
            target_position, _flight_time_s = proposed[edge_key]
            old_target = reached_positions.get(target_cell)
            if old_target is not None and old_target != target_position:
                raise TaskCoverabilityError(
                    "hopper task target position is ambiguous"
                )
            if old_target is None:
                newly_reached.add(target_cell)
            reached_positions[target_cell] = target_position
            processed_certified_edges.add(edge_key)
            rows, columns, visible = _hopper_capsule_patch(
                detail=detail_truth,
                positions_m=(reached_positions[source_cell], target_position),
                radius_m=float(observation.sensor_range_m),
            )
            round_visible[rows, columns] |= visible
            trajectory_edge_patch_count += 1
        observed_delta = np.ascontiguousarray(
            round_visible & detail_truth.valid_mask & ~before_observed_mask,
            dtype=np.bool_,
        )
        observed_detail |= observed_delta
        round_index += 1
        if incremental:
            pending_sources.update(newly_reached)
            if observed_delta.any():
                next_coarse = _observed_coarse_task_view(
                    common=common,
                    coarse_truth=coarse_truth,
                    detail_truth=detail_truth,
                    observed_detail=observed_detail,
                )
                newly_known = np.argwhere(
                    next_coarse.valid_mask & ~coarse_observed.valid_mask
                )
                if len(newly_known):
                    local_radius_m = (
                        _DETAIL_STATIC_TILE_CELLS
                        * float(detail_truth.canvas.geometry.resolution_m)
                        / 2.0
                    )
                    coarse_resolution_m = float(
                        coarse_truth.canvas.geometry.resolution_m
                    )
                    envelope_cells = (
                        int(math.ceil(local_radius_m / coarse_resolution_m))
                        + 1
                    )
                    affected_sources: set[tuple[int, int]] = set()
                    for row, column in newly_known:
                        for source_row in range(
                            max(0, int(row) - envelope_cells),
                            min(span, int(row) + envelope_cells + 1),
                        ):
                            for source_column in range(
                                max(0, int(column) - envelope_cells),
                                min(
                                    span,
                                    int(column) + envelope_cells + 1,
                                ),
                            ):
                                source_cell = source_row, source_column
                                if source_cell in reached_positions:
                                    affected_sources.add(source_cell)
                    newly_known_positions = tuple(
                        coarse_truth.canvas.grid_center_world(
                            int(row), int(column)
                        )
                        for row, column in newly_known
                    )
                    for source_cell in sorted(affected_sources):
                        left, bottom, right, top = (
                            _observed_local_detail_bounds(
                                detail_truth=detail_truth,
                                source_position_m=reached_positions[source_cell],
                            )
                        )
                        if any(
                            left <= x_m < right and bottom <= y_m < top
                            for x_m, y_m in newly_known_positions
                        ):
                            pending_sources.add(source_cell)
                new_native_indices = {
                    (span - 1 - int(row)) * span + int(column)
                    for row, column in newly_known
                }
                if new_native_indices:
                    for (source_cell, _target_cell), state in edge_states.items():
                        disposition, _reason, dependencies = state
                        if (
                            disposition == "WAITING_EVIDENCE"
                            and new_native_indices.intersection(dependencies)
                        ):
                            pending_sources.add(source_cell)
        if callable(round_callback):
            round_callback(
                {
                    "completed_round": round_index,
                    "reached_positions": dict(reached_positions),
                    "observed_detail": observed_detail.copy(),
                    "pending_sources": tuple(sorted(pending_sources)),
                    "edge_states": dict(edge_states),
                    "safe_union": safe_union.copy(),
                    "processed_certified_edges": tuple(
                        sorted(processed_certified_edges)
                    ),
                    "landing_algorithm_id": landing_algorithm_id,
                    "edge_algorithm_id": edge_algorithm_id,
                    "first_round_candidate_count": (
                        first_round_candidate_count
                    ),
                    "trajectory_edge_patch_count": (
                        trajectory_edge_patch_count
                    ),
                    "logical_projection_call_count": (
                        edge_projection_call_count
                    ),
                    "outside_roi_support_count": (
                        outside_roi_support_count
                    ),
                }
            )
        if (
            len(reached_positions) == before_reached
            and int(observed_detail.sum(dtype=np.int64)) == before_observed
        ):
            break

    if landing_algorithm_id is None or edge_algorithm_id is None:
        raise TaskCoverabilityError("hopper task closure produced no evidence")
    physical_mask = np.zeros((span, span), dtype=np.bool_)
    for cell in reached_positions:
        physical_mask[cell] = True
    ordered_positions = np.ascontiguousarray(
        [reached_positions[tuple(cell)] for cell in np.argwhere(physical_mask)],
        dtype=np.float64,
    ).reshape((-1, 3))
    positions_um = canonical_physical_positions_um(ordered_positions)
    canonical_positions_m = np.ascontiguousarray(
        positions_um.astype(np.float64) / 1_000_000.0, dtype=np.float64
    )
    physical_sha256 = physical_projection_sha256(
        platform_type="HOPPER",
        physical_reachability_algorithm_id=_HOPPER_REACHABILITY_ALGORITHM_ID,
        physical_evidence_algorithm_id=edge_algorithm_id,
        physical_observation_pose_mask=physical_mask,
        physical_observation_positions_m=canonical_positions_m,
        physical_grid_resolution_m=4.0,
        physical_grid_origin_m=(
            float(common.local_world_bounds_m[0]),
            float(common.local_world_bounds_m[3]),
        ),
        physical_grid_world_bounds_m=tuple(
            float(value) for value in common.local_world_bounds_m
        ),
        physical_grid_axis_convention=PHYSICAL_GRID_AXIS_CONVENTION,
        capability_content_sha256=platform.content_sha256,
        start_identity_sha256=start_identity,
    )
    coverable = np.ascontiguousarray(
        observed_detail & mission_target, dtype=np.bool_
    )
    initial_coverable = np.ascontiguousarray(
        initial_observed_coverable & coverable, dtype=np.bool_
    )
    coverable_count = int(coverable.sum(dtype=np.int64))
    target_count = int(mission_target.sum(dtype=np.int64))
    initial_count = int(initial_coverable.sum(dtype=np.int64))
    initial_fraction = (
        float(initial_count / coverable_count) if coverable_count else 0.0
    )
    reason = classify_ineligibility(
        qualified_start_cell=start_cell,
        mission_target_detail_cell_count=target_count,
        coverable_detail_cell_count=coverable_count,
        initial_coverable_fraction=initial_fraction,
        initial_candidate_count=first_round_candidate_count,
    )
    zero_denominator = coverable_count == 0
    resample_reason = (
        _ZERO_DENOMINATOR_REASON
        if zero_denominator
        else None if reason is None else reason.value
    )
    coverable_ratio = np.ascontiguousarray(
        coverable.reshape(
            span,
            DETAIL_PER_GLOBAL,
            span,
            DETAIL_PER_GLOBAL,
        ).mean(axis=(1, 3), dtype=np.float64),
        dtype=np.float32,
    )
    disposition_counts = {
        name: sum(
            1 for disposition, _reason, _dependencies in edge_states.values()
            if disposition == name
        )
        for name in (
            "CERTIFIED",
            "STABLE_PHYSICAL_REJECTION",
            "WAITING_EVIDENCE",
        )
    }
    diagnostics = MappingProxyType(
        {
            "schema": "lunar-hopper-task-coverability-diagnostics/v1",
            "platform_type": "HOPPER",
            "common_artifact_sha256": common.artifact_sha256,
            "qualified_start_global_cell": list(
                qualified_start["qualified_start_cell"]
            ),
            "qualified_start_local_cell": list(start_cell),
            "start_identity_sha256": start_identity,
            "platform_capability_sha256": platform.content_sha256,
            "static_projection_stage": _STATIC_PROJECTION_STAGE,
            "static_projection_call_count": detail_static_calls,
            "landing_evidence_call_count": landing_evidence_call_count,
            "edge_projection_call_count": edge_projection_call_count,
            "physical_projection_schema": PHYSICAL_PROJECTION_SCHEMA,
            "physical_reachability_algorithm_id": (
                _HOPPER_REACHABILITY_ALGORITHM_ID
            ),
            "physical_evidence_algorithm_id": edge_algorithm_id,
            "landing_evidence_algorithm_id": landing_algorithm_id,
            "physical_safe_pose_count": int(safe_union.sum(dtype=np.int64)),
            "physically_reachable_pose_count": int(
                physical_mask.sum(dtype=np.int64)
            ),
            "physical_projection_sha256": physical_sha256,
            "sensor_algorithm_id": _HOPPER_SENSOR_ALGORITHM_ID,
            "initial_observation_commit_count": 1,
            "trajectory_round_commit_count": round_index,
            "trajectory_edge_patch_count": trajectory_edge_patch_count,
            "edge_certified_count": disposition_counts["CERTIFIED"],
            "edge_stable_physical_rejection_count": disposition_counts[
                "STABLE_PHYSICAL_REJECTION"
            ],
            "edge_waiting_evidence_count": disposition_counts[
                "WAITING_EVIDENCE"
            ],
            "edge_diagnostics": [
                {
                    "source_index": source[0] * span + source[1],
                    "target_index": target[0] * span + target[1],
                    "disposition": edge_states[(source, target)][0],
                    "reason_code": edge_states[(source, target)][1],
                    "dependency_tile_indices": list(
                        edge_states[(source, target)][2]
                    ),
                }
                for source, target in sorted(edge_states)
            ],
            "mission_target_detail_mask_sha256": mask_sha256(mission_target),
            "task_coverable_detail_mask_sha256": mask_sha256(coverable),
            "mission_target_detail_cell_count": target_count,
            "coverable_detail_cell_count": coverable_count,
            "mission_coverable_fraction": (
                float(coverable_count / target_count) if target_count else 0.0
            ),
            "initial_coverable_fraction": initial_fraction,
            "initial_candidate_count": first_round_candidate_count,
            "zero_denominator": zero_denominator,
            "eligible": reason is None,
            "ineligible_reason": None if reason is None else reason.value,
            "resample_required": reason is not None,
            "resample_reason": resample_reason,
            "roi_authorization_enforced": True,
            "outside_roi_endpoint_count": 0,
            "outside_roi_support_count": outside_roi_support_count,
            "outside_roi_trajectory_cell_count": 0,
            "halo_state_count": 0,
        }
    )
    arrays = MappingProxyType(
        {
            "platform_intrinsic_detail_mask": _readonly(
                detail_intrinsic, np.bool_
            ),
            "physical_safe_pose_mask": _readonly(safe_union, np.bool_),
            "physical_observation_pose_mask": _readonly(
                physical_mask, np.bool_
            ),
            "physical_observation_positions_um": _readonly(
                positions_um, np.int64
            ),
            "mission_target_detail_mask": _readonly(
                mission_target, np.bool_
            ),
            "task_coverable_detail_mask": _readonly(coverable, np.bool_),
            "initial_observed_coverable_detail_mask": _readonly(
                initial_coverable, np.bool_
            ),
            "coverable_ratio": _readonly(coverable_ratio, np.float32),
        }
    )
    if metrics_output is not None:
        metrics_output.update(
            {
                "native_landing_evidence_calls": (
                    native_landing_evidence_calls
                ),
                "native_edge_projection_calls": native_edge_projection_calls,
                "source_process_count": source_process_count,
                "global_map_assembly_count": global_map_assembly_count,
            }
        )
    return PlatformTaskPayload(
        platform_type="HOPPER",
        common_artifact_sha256=common.artifact_sha256,
        arrays=arrays,
        diagnostics=diagnostics,
    )


def build_hopper_task_coverability_reference(
    *,
    common: TaskCommonArtifact,
    platform: FrozenPlatformCapability,
    qualified_start: Mapping[str, object],
    bridge: object,
) -> PlatformTaskPayload:
    """Build the deliberately naive task-local HOPPER reference closure."""
    return _build_hopper_task_coverability(
        common=common,
        platform=platform,
        qualified_start=qualified_start,
        bridge=bridge,
        incremental=False,
    )


def build_ground_task_coverability(
    *,
    common: TaskCommonArtifact,
    platform: FrozenPlatformCapability,
    qualified_start: Mapping[str, object],
    bridge: object,
) -> PlatformTaskPayload:
    """Freeze one exact ground denominator on the task-induced subgraph."""
    if not isinstance(common, TaskCommonArtifact):
        raise TypeError("ground task builder requires TaskCommonArtifact")
    if not isinstance(platform, FrozenPlatformCapability):
        raise TypeError("ground task builder requires FrozenPlatformCapability")
    if platform.platform_type not in _GROUND_PLATFORMS:
        raise TaskCoverabilityError("ground task builder platform is invalid")
    if not callable(getattr(bridge, "project_traversability", None)) or not callable(
        getattr(bridge, "project_reachability", None)
    ):
        raise TypeError("ground task builder requires projection bridge")
    start_cell, start_identity = _qualified_local_start(
        common=common,
        platform=platform,
        qualified_start=qualified_start,
    )
    observation = platform.observation_capability
    sensor = SensorGeometry(
        observation.sensor_range_m,
        observation.sensor_fov_rad,
    )
    if sensor.range_m != 30.0 or not sensor.is_full_circle:
        raise TaskCoverabilityError("ground task sensor authority differs")

    scene = SimpleNamespace(scene_id=common.key.scene_id)
    coarse = _task_projected_view(common, detail=False)
    if not bool(coarse.valid_mask[start_cell]):
        raise TaskCoverabilityError("qualified start is invalid inside task ROI")
    coarse_request = _projection_request(
        platform,
        scene,
        coarse,
        start_cell=start_cell,
    )
    coarse_static = bridge.project_traversability(coarse_request)
    safe = _projection_mask(
        coarse_static,
        "hard_feasible",
        platform_type=platform.platform_type,
        cells=common.geometry.span_cells,
    ) & coarse.valid_mask

    detail = _task_projected_view(common, detail=True)
    detail_intrinsic, detail_static_calls = _detail_intrinsic_projection(
        common=common,
        platform=platform,
        bridge=bridge,
        projected=detail,
    )
    static_projection_call_count = 1 + detail_static_calls

    reachability_output = bridge.project_reachability(coarse_request, 30.0)
    try:
        (
            reachable,
            reachability_algorithm_id,
            minimum_cost,
            parent_index,
            start_index,
            _search_elapsed_s,
        ) = _validate_ground_reachability_projection(
            reachability_output,
            platform_type=platform.platform_type,
            cells=common.geometry.span_cells,
        )
    except PlatformReachabilityError as error:
        raise TaskCoverabilityError(
            "ground task reachability tree is invalid"
        ) from error
    if not bool(reachable[start_cell]):
        raise TaskCoverabilityError("ground task start is unreachable")
    if np.logical_and(reachable, ~safe).any():
        raise TaskCoverabilityError(
            "ground task reachability exceeds ROI-authorized safe states"
        )

    pose_rows, pose_columns = np.nonzero(reachable)
    positions_m = np.ascontiguousarray(
        [
            (
                *coarse.canvas.grid_center_world(int(row), int(column)),
                float(coarse.elevation_m[row, column]),
            )
            for row, column in zip(pose_rows, pose_columns, strict=True)
        ],
        dtype=np.float64,
    ).reshape((-1, 3))
    positions_um = canonical_physical_positions_um(positions_m)
    canonical_positions_m = np.ascontiguousarray(
        positions_um.astype(np.float64) / 1_000_000.0,
        dtype=np.float64,
    )
    physical_sha256 = physical_projection_sha256(
        platform_type=platform.platform_type,
        physical_reachability_algorithm_id=reachability_algorithm_id,
        physical_evidence_algorithm_id=(
            _GROUND_PHYSICAL_EVIDENCE_ALGORITHM_ID
        ),
        physical_observation_pose_mask=reachable,
        physical_observation_positions_m=canonical_positions_m,
        physical_grid_resolution_m=4.0,
        physical_grid_origin_m=(
            float(common.local_world_bounds_m[0]),
            float(common.local_world_bounds_m[3]),
        ),
        physical_grid_world_bounds_m=tuple(
            float(value) for value in common.local_world_bounds_m
        ),
        physical_grid_axis_convention=PHYSICAL_GRID_AXIS_CONVENTION,
        capability_content_sha256=platform.content_sha256,
        start_identity_sha256=start_identity,
    )

    mission_target = np.ascontiguousarray(
        common.arrays["mission_target_detail_mask"] & detail_intrinsic,
        dtype=np.bool_,
    )
    truth_obstacle_ratio = np.ascontiguousarray(
        detail.physical_obstacle_ratio,
        dtype=np.float32,
    )
    estimator = NativeVisibilityEstimator(sensor, resolution_m=0.2)
    radius_cells = math.ceil(sensor.range_m / 0.2)
    detail_pose_cells = tuple(
        (
            int(row) * DETAIL_PER_GLOBAL + DETAIL_PER_GLOBAL // 2,
            int(column) * DETAIL_PER_GLOBAL + DETAIL_PER_GLOBAL // 2,
        )
        for row, column in zip(pose_rows, pose_columns, strict=True)
    )
    start_detail_cell = (
        start_cell[0] * DETAIL_PER_GLOBAL + DETAIL_PER_GLOBAL // 2,
        start_cell[1] * DETAIL_PER_GLOBAL + DETAIL_PER_GLOBAL // 2,
    )
    ordered_sources = _visibility_sources(
        detail_pose_cells,
        start_cell=start_detail_cell,
        window_cells=2 * radius_cells + 1,
    )
    coverable = np.zeros(mission_target.shape, dtype=np.bool_)
    initial_observed = np.zeros(mission_target.shape, dtype=np.bool_)
    visibility_window_count = 0
    visibility_skipped_covered_count = 0
    for pose_cell in ordered_sources:
        row_slice = slice(
            max(0, pose_cell[0] - radius_cells),
            min(mission_target.shape[0], pose_cell[0] + radius_cells + 1),
        )
        column_slice = slice(
            max(0, pose_cell[1] - radius_cells),
            min(mission_target.shape[1], pose_cell[1] + radius_cells + 1),
        )
        if not bool(
            (mission_target[row_slice, column_slice]
            & ~coverable[row_slice, column_slice]).any()
        ) and pose_cell != start_detail_cell:
            visibility_skipped_covered_count += 1
            continue
        rows, columns, visible = _reveal_task_window(
            truth=truth_obstacle_ratio,
            pose_cell=pose_cell,
            estimator=estimator,
            radius_cells=radius_cells,
        )
        visibility_window_count += 1
        if pose_cell == start_detail_cell:
            initial_observed[rows, columns] |= visible
        coverable[rows, columns] |= visible & mission_target[rows, columns]
    coverable &= mission_target
    initial_coverable = initial_observed & coverable
    coverable_count = int(coverable.sum(dtype=np.int64))
    target_count = int(mission_target.sum(dtype=np.int64))
    initial_count = int(initial_coverable.sum(dtype=np.int64))
    initial_fraction = (
        float(initial_count / coverable_count) if coverable_count else 0.0
    )
    initial_candidate_count = _initial_candidate_count(
        mission_roi=np.ascontiguousarray(
            common.arrays["mission_roi_mask"], dtype=np.bool_
        ),
        reachable=reachable,
        start_cell=start_cell,
        initial_observed_detail=initial_observed,
        mission_target=mission_target,
        truth_obstacle_ratio=truth_obstacle_ratio,
        estimator=estimator,
        sensor=sensor,
    )
    reason = classify_ineligibility(
        qualified_start_cell=start_cell,
        mission_target_detail_cell_count=target_count,
        coverable_detail_cell_count=coverable_count,
        initial_coverable_fraction=initial_fraction,
        initial_candidate_count=initial_candidate_count,
    )
    zero_denominator = coverable_count == 0
    resample_reason = (
        _ZERO_DENOMINATOR_REASON
        if zero_denominator
        else None if reason is None else reason.value
    )
    coverable_ratio = np.ascontiguousarray(
        coverable.reshape(
            common.geometry.span_cells,
            DETAIL_PER_GLOBAL,
            common.geometry.span_cells,
            DETAIL_PER_GLOBAL,
        ).mean(axis=(1, 3), dtype=np.float64),
        dtype=np.float32,
    )
    visibility_algorithm_id = str(
        getattr(estimator, "algorithm_id", _GROUND_VISIBILITY_ALGORITHM_ID)
    )
    diagnostics = MappingProxyType(
        {
            "schema": "lunar-ground-task-coverability-diagnostics/v1",
            "platform_type": platform.platform_type,
            "common_artifact_sha256": common.artifact_sha256,
            "qualified_start_global_cell": list(
                qualified_start["qualified_start_cell"]
            ),
            "qualified_start_local_cell": list(start_cell),
            "start_identity_sha256": start_identity,
            "platform_capability_sha256": platform.content_sha256,
            "static_projection_stage": _STATIC_PROJECTION_STAGE,
            "static_projection_call_count": static_projection_call_count,
            "reachability_tree_call_count": 1,
            "global_plan_call_count": 0,
            "physical_projection_schema": PHYSICAL_PROJECTION_SCHEMA,
            "physical_reachability_algorithm_id": (
                reachability_algorithm_id
            ),
            "physical_evidence_algorithm_id": (
                _GROUND_PHYSICAL_EVIDENCE_ALGORITHM_ID
            ),
            "physical_safe_pose_count": int(safe.sum(dtype=np.int64)),
            "physically_reachable_pose_count": int(
                reachable.sum(dtype=np.int64)
            ),
            "physical_projection_sha256": physical_sha256,
            "sensor_algorithm_id": _GROUND_SENSOR_ALGORITHM_ID,
            "visibility_algorithm_id": visibility_algorithm_id,
            "visibility_source_count": len(ordered_sources),
            "visibility_window_count": visibility_window_count,
            "visibility_skipped_covered_count": (
                visibility_skipped_covered_count
            ),
            "mission_target_detail_mask_sha256": mask_sha256(
                mission_target
            ),
            "task_coverable_detail_mask_sha256": mask_sha256(coverable),
            "mission_target_detail_cell_count": target_count,
            "coverable_detail_cell_count": coverable_count,
            "mission_coverable_fraction": (
                float(coverable_count / target_count) if target_count else 0.0
            ),
            "initial_coverable_fraction": initial_fraction,
            "initial_candidate_count": initial_candidate_count,
            "zero_denominator": zero_denominator,
            "eligible": reason is None,
            "ineligible_reason": None if reason is None else reason.value,
            "resample_required": reason is not None,
            "resample_reason": resample_reason,
            "roi_authorization_enforced": True,
            "halo_state_count": 0,
        }
    )
    arrays = {
        "platform_intrinsic_detail_mask": _readonly(
            detail_intrinsic, np.bool_
        ),
        "physical_safe_pose_mask": _readonly(safe, np.bool_),
        "physical_observation_pose_mask": _readonly(reachable, np.bool_),
        "physical_observation_positions_um": _readonly(
            positions_um, np.int64
        ),
        "minimum_global_cost": _readonly(minimum_cost, np.float64),
        "global_parent_index": _readonly(parent_index, np.int64),
        "mission_target_detail_mask": _readonly(
            mission_target, np.bool_
        ),
        "task_coverable_detail_mask": _readonly(coverable, np.bool_),
        "initial_observed_coverable_detail_mask": _readonly(
            initial_coverable, np.bool_
        ),
        "coverable_ratio": _readonly(coverable_ratio, np.float32),
    }
    diagnostics = MappingProxyType(
        {**dict(diagnostics), "global_start_index": start_index}
    )
    return PlatformTaskPayload(
        platform_type=platform.platform_type,
        common_artifact_sha256=common.artifact_sha256,
        arrays=MappingProxyType(arrays),
        diagnostics=diagnostics,
    )


__all__ = [
    "TaskCoverabilityError",
    "TaskSourceCrop",
    "build_ground_task_coverability",
    "build_hopper_task_coverability_reference",
    "build_task_common",
]

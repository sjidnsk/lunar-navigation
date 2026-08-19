"""Single authoritative builder for cache-backed formal C++ v3 workers."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from collections import OrderedDict
from typing import ClassVar, Mapping, Sequence

import numpy as np
import torch

import lunar_planner_training_bridge as bridge_api

from ..config import TaskAreaConfig
from ..capability_freeze import (
    FrozenCapabilityBundle,
    FrozenCapabilityEnvironmentFactory,
    FrozenPlatformCapability,
    ScenarioIdentity,
)
from ..polar_data.formal_cache import (
    FormalCache,
    FormalSceneIndexRecord,
    _physical_capability_content_sha256,
    load_formal_cache,
)
from ..polar_data.hazards import (
    VectorHazardScene,
)
from ..polar_data.multires_scene import (
    GENERATOR_SHA256,
    LOCAL_DETAIL_PROVENANCE,
    MultiResolutionScene,
    ProjectedScene,
    SceneTileProvider,
)
from ..polar_data.raster import (
    GLOBAL_GEOMETRY,
    LOCAL_GEOMETRY,
    LOCAL_TILE_GEOMETRY,
    GridGeometry,
    MapCanvas,
)
from ..polar_data.task_cache import (
    PlatformTaskArtifact,
    PlatformTaskKey,
    TaskCacheStore,
    TaskCommonArtifact,
    TaskCommonKey,
)
from ..polar_data.task_cache_scheduler import (
    TaskBuildPriority,
    TaskCacheCoordinatorError,
)
from ..policy.action_semantics import apply_goal_theta
from ..policy.observation import ObservationIdentity, PolicyBatch
from ..reward_contract import TaskScaleBucket
from ..training_semantics import (
    FORMAL_SENSOR_FOV_RAD,
    FORMAL_SENSOR_RANGE_M,
    training_semantics_sha256,
)
from .candidate_builder import (
    CandidateBatch,
    CandidateBuilderV2,
    CandidateDecisionSnapshot,
    CandidateDiagnostics,
    PhysicalCandidateUniverse,
    _RawFrontierCandidate,
    _sample_frontier_chain,
    _select_narrow_frontier_strip_positions,
)
from .coverability import mask_sha256, pack_detail_mask, unpack_detail_mask
from .formal_episode_state import (
    FormalPathSampleState,
    FormalPoseState,
    FormalRevealState,
    FormalWorkerState,
    policy_batch_sha256,
)
from .formal_start_qualification import (
    build_formal_mission_roi,
)
from .task_area import (
    FrozenTaskGeometry,
    build_task_priority_mask,
    derive_task_evidence_halo,
    freeze_formal_task_geometry,
    scale_bucket_for_worker,
)
from .macro_step import ExecutionEvents, PolicyAction
from .multires_observation import (
    DetailObservedWindow,
    HopperTrajectoryPoint,
    MultiresSensorObservationState,
)
from .observation_boundary import (
    BoundaryObservationResult,
    ObservationBoundaryController,
    SensorBoundaryEvidence,
    SensorPathSample,
)
from .observation_builder import (
    LocalObservation,
    MissionRaster,
    ObservationBuilderV2,
    PlatformProjection,
    Pose2,
)
from .parallel_pool import ParallelEnvironmentWorker
from .platform_reachability import (
    PhysicalReachabilityResult,
    PlatformCandidateReachability,
    ground_point_goal_feasibility,
)
from .v3_environment import (
    CommittedHopExecutionFeedback,
    EnvironmentInvariantError,
    PreparedPlanRequest,
    ReferenceExecutionResult,
    _audit_candidate_boundary,
    create_v3_environment,
)
from .visibility import NativeVisibilityEstimator, SensorGeometry


_PLATFORMS = ("WHEELED", "LEGGED", "HOPPER")
_FORMAL_MACRO_STEP_TIME_SCALE_S = 2.0
_MAX_GROUND_SENSOR_SAMPLE_SPACING_M = 1.0
_GROUND_TASK_REACHABILITY_ALGORITHM_ID = "cpp-ground-global-cost-tree/v1"
_GROUND_TASK_EVIDENCE_ALGORITHM_ID = "cpp-safe-traversability-projection/v1"
_GROUND_TASK_SENSOR_ALGORITHM_ID = "sensor-30m-360/v1"
_GROUND_TASK_VISIBILITY_ALGORITHM_ID = "two-dimensional-detail-los/v1"
_HOPPER_TASK_REACHABILITY_ALGORITHM_ID = (
    "truth-hopper-task-safe-hop-observation-closure/v1"
)
_HOPPER_TASK_EVIDENCE_ALGORITHM_ID = (
    "cpp-hopper-incremental-edge-evidence/v1"
)
_HOPPER_TASK_SENSOR_ALGORITHM_ID = (
    "hopper-unobstructed-trajectory-capsule-closure/v1"
)


class _GroundStartQualificationProbeError(EnvironmentInvariantError):
    """One planner probe failed before it could classify the start cell."""


def _ground_start_resample_reason(
    output: object,
    platform_type: str,
) -> str | None:
    expected_reason = {
        "WHEELED": "WHEEL_START_NOT_SAFE",
        "LEGGED": "LEGGED_START_NOT_SAFE",
    }.get(platform_type)
    if expected_reason is None:
        raise ValueError("ground start qualification platform is invalid")
    start_reasons = {
        "WHEEL_START_NOT_SAFE",
        "LEGGED_START_NOT_SAFE",
    }
    reported_reasons = {
        str(output.reason_code),
        *(str(reason) for reason in output.diagnostics.warning_codes),
    }
    observed_start_reasons = reported_reasons & start_reasons
    if observed_start_reasons:
        if observed_start_reasons != {expected_reason}:
            raise EnvironmentInvariantError(
                "GROUND_START_QUALIFICATION_PLATFORM_MISMATCH"
            )
        return expected_reason
    if output.outcome in {
        bridge_api.PlanningOutcome.INVALID_REQUEST,
        bridge_api.PlanningOutcome.STALE_INPUT,
        bridge_api.PlanningOutcome.NUMERICAL_FAILURE,
        bridge_api.PlanningOutcome.RESOURCE_EXHAUSTED,
        bridge_api.PlanningOutcome.ACTIVE_REFERENCE_INVALIDATED,
        bridge_api.PlanningOutcome.CANCELED,
    }:
        warnings = ",".join(
            str(reason) for reason in output.diagnostics.warning_codes
        )
        raise _GroundStartQualificationProbeError(
            "GROUND_START_QUALIFICATION_FAILED "
            f"outcome={output.outcome} "
            f"reason_code={output.reason_code} "
            f"warning_codes={warnings or '-'}"
        )
    return None


def _formal_scheduled_entries(
    entries: Sequence[Mapping[str, object]],
    *,
    scenario_schedule_id: str,
) -> tuple[Mapping[str, object], ...]:
    """Return one stable seeded permutation of a split inventory."""
    if not isinstance(scenario_schedule_id, str) or not scenario_schedule_id:
        raise ValueError("formal scenario schedule identity is missing")
    materialized = tuple(entries)
    scene_ids: list[str] = []
    for entry in materialized:
        if not isinstance(entry, Mapping):
            raise ValueError("formal scheduled scene entry is invalid")
        scene_id = entry.get("scene_id")
        if not isinstance(scene_id, str) or len(scene_id) != 64:
            raise ValueError("formal scheduled scene ID is invalid")
        scene_ids.append(scene_id)
    if len(set(scene_ids)) != len(scene_ids):
        raise ValueError("formal scheduled scene IDs are not unique")

    def schedule_key(entry: Mapping[str, object]) -> tuple[str, str]:
        scene_id = str(entry["scene_id"])
        digest = hashlib.sha256(
            f"{scenario_schedule_id}\0scene\0{scene_id}".encode("utf-8")
        ).hexdigest()
        return digest, scene_id

    return tuple(sorted(materialized, key=schedule_key))


def _formal_schedule_index(
    worker_index: int,
    episode_cursor: int,
    scene_count: int,
    *,
    platform_worker_index: int | None = None,
    platform_worker_count: int | None = None,
) -> int:
    """Map a fixed platform lane and episode ordinal onto the scene inventory."""
    if type(worker_index) is not int or worker_index < 0:
        raise ValueError("formal worker index must be non-negative")
    if type(episode_cursor) is not int or episode_cursor < 0:
        raise ValueError("formal episode cursor must be non-negative")
    if type(scene_count) is not int or scene_count <= 0:
        raise ValueError("formal scene count must be positive")
    if platform_worker_index is None:
        platform_worker_index = worker_index
    if platform_worker_count is None:
        platform_worker_count = platform_worker_index + 1
    if (
        type(platform_worker_index) is not int
        or type(platform_worker_count) is not int
        or platform_worker_index < 0
        or platform_worker_count <= platform_worker_index
    ):
        raise ValueError("formal platform worker lane is invalid")
    return (
        episode_cursor * platform_worker_count + platform_worker_index
    ) % scene_count


def _formal_episode_seed(kind: str, identity: ScenarioIdentity) -> str:
    if kind not in {"start", "episode"}:
        raise ValueError("formal episode seed kind is invalid")
    if kind == "episode":
        # Matching platform-local lanes traverse the same physical scenes.
        payload = (
            f"{identity.scenario_schedule_id}\0{kind}\0"
            f"{identity.platform_worker_index}\0"
            f"{identity.platform_worker_count}\0{identity.episode_cursor}"
        )
    else:
        # Starts remain platform-specific because feasibility projections differ.
        payload = (
            f"{identity.scenario_schedule_id}\0{kind}\0"
            f"{identity.platform_type}\0{identity.worker_index}\0"
            f"{identity.platform_worker_index}\0"
            f"{identity.platform_worker_count}\0{identity.episode_cursor}"
        )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _task_halo_contract_sha256(halo: object) -> str:
    return _canonical_json_sha256(
        {
            "coarse_cells": halo.coarse_cells,
            "detail_cells": halo.detail_cells,
            "sensor_radius_m": halo.sensor_radius_m,
            "platform_support_radius_m": halo.platform_support_radius_m,
            "native_stencil_radius_m": halo.native_stencil_radius_m,
            "algorithm_id": halo.algorithm_id,
            "capability_bundle_sha256": halo.capability_bundle_sha256,
        }
    )


def _task_platform_key(
    *,
    common_key: TaskCommonKey,
    platform: FrozenPlatformCapability,
    qualified_start: Mapping[str, object],
) -> PlatformTaskKey:
    if platform.platform_type in {"WHEELED", "LEGGED"}:
        identities = (
            _GROUND_TASK_REACHABILITY_ALGORITHM_ID,
            _GROUND_TASK_EVIDENCE_ALGORITHM_ID,
            _GROUND_TASK_SENSOR_ALGORITHM_ID,
            _GROUND_TASK_VISIBILITY_ALGORITHM_ID,
        )
    elif platform.platform_type == "HOPPER":
        identities = (
            _HOPPER_TASK_REACHABILITY_ALGORITHM_ID,
            _HOPPER_TASK_EVIDENCE_ALGORITHM_ID,
            _HOPPER_TASK_SENSOR_ALGORITHM_ID,
            _HOPPER_TASK_SENSOR_ALGORITHM_ID,
        )
    else:  # pragma: no cover - the frozen capability union is closed
        raise ValueError("formal task platform is invalid")
    return PlatformTaskKey(
        common_key_sha256=common_key.sha256(),
        platform_type=platform.platform_type,
        qualified_start_identity_sha256=str(
            qualified_start["start_identity_sha256"]
        ),
        platform_capability_sha256=platform.content_sha256,
        reachability_algorithm_id=identities[0],
        physical_evidence_algorithm_id=identities[1],
        sensor_algorithm_id=identities[2],
        visibility_algorithm_id=identities[3],
        training_semantics_sha256=training_semantics_sha256(),
    )


class _TaskLocalMultiResolutionScene(MultiResolutionScene):
    """Exact task-sized view backed only by verified common artifact arrays."""

    def __init__(
        self,
        common: TaskCommonArtifact,
        *,
        window_sha256: str,
        scene_seed: str,
    ) -> None:
        geometry = common.geometry
        span = geometry.span_cells
        canvas = MapCanvas(
            window_sha256,
            common.local_world_bounds_m,
            GridGeometry(span * 4.0, 4.0, span),
        )
        halo_row0, _, halo_column0, _ = (
            geometry.halo_coarse_bounds_half_open
        )
        row0, row1, column0, column1 = geometry.coarse_bounds_half_open
        coarse_rows = slice(row0 - halo_row0, row1 - halo_row0)
        coarse_columns = slice(
            column0 - halo_column0,
            column1 - halo_column0,
        )
        detail_rows = slice(coarse_rows.start * 20, coarse_rows.stop * 20)
        detail_columns = slice(
            coarse_columns.start * 20,
            coarse_columns.stop * 20,
        )
        empty_hazards = VectorHazardScene(
            seed=scene_seed,
            canvas=canvas,
            rocks=(),
            craters=(),
            no_go_polygons=(),
        )
        coarse = {
            name: np.ascontiguousarray(
                common.arrays[name][coarse_rows, coarse_columns]
            )
            for name in (
                "elevation_m",
                "valid_mask",
                "physical_obstacle_ratio",
                "physical_obstacle_height_m",
                "forbidden_ratio",
            )
        }
        detail = {
            name.removeprefix("detail_"): np.ascontiguousarray(
                common.arrays[name][detail_rows, detail_columns]
            )
            for name in (
                "detail_elevation_m",
                "detail_valid_mask",
                "detail_physical_obstacle_ratio",
                "detail_physical_obstacle_height_m",
                "detail_forbidden_ratio",
            )
        }
        super().__init__(
            canvas,
            np.where(coarse["valid_mask"], coarse["elevation_m"], 0.0).astype(
                np.float32
            ),
            coarse["valid_mask"],
            empty_hazards,
            scenario_id=common.key.scene_id,
        )
        object.__setattr__(self, "_task_coarse", coarse)
        object.__setattr__(self, "_task_detail", detail)

    def _project_indices(
        self,
        *,
        start_row: int,
        start_column: int,
        cells: int,
        detail: bool,
        canvas: MapCanvas,
        padded: bool,
    ) -> ProjectedScene:
        source = self._task_detail if detail else self._task_coarse
        total = source["valid_mask"].shape[0]
        if min(start_row, start_column) < 0 or cells <= 0:
            raise ValueError("task projection indices are invalid")
        available_rows = max(0, min(cells, total - start_row))
        available_columns = max(0, min(cells, total - start_column))
        if not padded and (
            available_rows != cells or available_columns != cells
        ):
            raise ValueError("projection target lies outside task scene")
        shape = (cells, cells)
        elevation = np.full(shape, np.nan, dtype=np.float32)
        valid = np.zeros(shape, dtype=np.bool_)
        obstacle = np.zeros(shape, dtype=np.float32)
        obstacle_height = np.zeros(shape, dtype=np.float32)
        forbidden = np.zeros(shape, dtype=np.float32)
        if available_rows and available_columns:
            source_slice = (
                slice(start_row, start_row + available_rows),
                slice(start_column, start_column + available_columns),
            )
            target_slice = (
                slice(0, available_rows),
                slice(0, available_columns),
            )
            for target, name in (
                (elevation, "elevation_m"),
                (valid, "valid_mask"),
                (obstacle, "physical_obstacle_ratio"),
                (obstacle_height, "physical_obstacle_height_m"),
                (forbidden, "forbidden_ratio"),
            ):
                target[target_slice] = source[name][source_slice]
        return ProjectedScene(
            vector_sha256=self.hazards.vector_sha256,
            canvas=canvas,
            crater_elevation_delta_m=np.zeros(shape, dtype=np.float32),
            physical_obstacle_ratio=obstacle,
            physical_obstacle_height_m=obstacle_height,
            forbidden_ratio=forbidden,
            local_detail_provenance=LOCAL_DETAIL_PROVENANCE,
            elevation_m=elevation,
            valid_mask=valid,
        )

    def project(self, target_canvas: MapCanvas) -> ProjectedScene:
        if target_canvas.window_sha256 != self.base_canvas.window_sha256:
            raise ValueError("projection target window identity differs")
        resolution = target_canvas.geometry.resolution_m
        if math.isclose(resolution, 4.0, rel_tol=0.0, abs_tol=1.0e-9):
            detail = False
        elif math.isclose(resolution, 0.2, rel_tol=0.0, abs_tol=1.0e-9):
            detail = True
        else:
            raise ValueError("task scene projection resolution is unsupported")
        left, _, _, top = self.base_canvas.bounds_m
        start_column = round((target_canvas.bounds_m[0] - left) / resolution)
        start_row = round((top - target_canvas.bounds_m[3]) / resolution)
        return self._project_indices(
            start_row=start_row,
            start_column=start_column,
            cells=target_canvas.geometry.cells,
            detail=detail,
            canvas=target_canvas,
            padded=False,
        )


class _TaskLocalSceneTileProvider(SceneTileProvider):
    """Fixed 64 m tiles over an exact, potentially partial final task tile."""

    def __init__(
        self,
        scene: _TaskLocalMultiResolutionScene,
        *,
        capacity: int = 8,
    ) -> None:
        if capacity < 1:
            raise ValueError("tile cache capacity must be positive")
        self.scene = scene
        self.tile_geometry = LOCAL_TILE_GEOMETRY
        self.capacity = capacity
        self.tiles_per_axis = math.ceil(
            self.detail_cells_per_axis / self.tile_geometry.cells
        )
        self._cache = OrderedDict()

    @property
    def detail_cells_per_axis(self) -> int:
        return self.scene.base_canvas.geometry.cells * 20

    def tile(self, tile_row: int, tile_column: int) -> ProjectedScene:
        if not (
            0 <= tile_row < self.tiles_per_axis
            and 0 <= tile_column < self.tiles_per_axis
        ):
            raise ValueError("tile index lies outside the task scene")
        key = (self.scene.scene_id, tile_row, tile_column, GENERATOR_SHA256)
        cached = self._cache.pop(key, None)
        if cached is not None:
            self._cache[key] = cached
            return cached
        cells = self.tile_geometry.cells
        start_row = tile_row * cells
        start_column = tile_column * cells
        left, _, _, top = self.scene.base_canvas.bounds_m
        tile_left = left + start_column * self.tile_geometry.resolution_m
        tile_top = top - start_row * self.tile_geometry.resolution_m
        canvas = MapCanvas(
            self.scene.base_canvas.window_sha256,
            (
                tile_left,
                tile_top - self.tile_geometry.size_m,
                tile_left + self.tile_geometry.size_m,
                tile_top,
            ),
            self.tile_geometry,
        )
        projected = self.scene._project_indices(
            start_row=start_row,
            start_column=start_column,
            cells=cells,
            detail=True,
            canvas=canvas,
            padded=True,
        )
        self._cache[key] = projected
        while len(self._cache) > self.capacity:
            self._cache.popitem(last=False)
        return projected

    def read_visibility_obstacle_window(
        self,
        start_row: int,
        start_column: int,
        *,
        cells: int,
    ) -> np.ndarray:
        return self.read_window(
            start_row,
            start_column,
            cells=cells,
        ).physical_obstacle_ratio


@dataclass(frozen=True, slots=True)
class _LoadedScene:
    scene: MultiResolutionScene
    arrays: Mapping[str, np.ndarray]
    entry: Mapping[str, object]
    scenario: Mapping[str, object]
    task_geometry: FrozenTaskGeometry | None = None
    task_common_key_sha256: str | None = None
    task_common_artifact_sha256: str | None = None
    platform_task_key_sha256: str | None = None
    platform_task_artifact_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class _MapSnapshot:
    revision: int
    candidates: CandidateBatch
    global_map: object
    planner_global_map: object
    local_map: object
    world: object
    projection: PlatformProjection
    physical_reachability: PhysicalReachabilityResult
    candidate_universe: PhysicalCandidateUniverse
    planning_physical_snapshot_id: str
    ground_start_resample_reason: str | None

    @property
    def candidate_universe_sha256(self) -> str:
        return self.candidate_universe.universe_sha256


@dataclass(frozen=True, slots=True)
class _GroundOption:
    target_x_m: float
    target_y_m: float
    target_z_m: float
    tolerance_m: float
    theta_rad: float
    goal_id: str
    candidate_id: str


@dataclass(frozen=True, slots=True)
class FormalEnvironmentWorker(ParallelEnvironmentWorker):
    episode: "FormalEpisode"

    def snapshot_episode_state(self) -> dict[str, object]:
        return self.episode.snapshot_state(self.environment).to_dict()

    def current_candidate_diagnostics(self) -> CandidateDiagnostics:
        getter = getattr(self.environment, "current_candidate_diagnostics", None)
        if not callable(getter):
            raise ValueError("formal environment diagnostics are unavailable")
        diagnostics = getter()
        if not isinstance(diagnostics, CandidateDiagnostics):
            raise ValueError("formal environment diagnostics are invalid")
        return diagnostics


@dataclass(frozen=True, slots=True)
class FormalEnvironmentAssembly:
    factory: FrozenCapabilityEnvironmentFactory
    observation_template: PolicyBatch
    cache_manifest_sha256: str
    scenario_schedule_id: str


def _task_local_loaded_scene(
    *,
    record: FormalSceneIndexRecord,
    common: TaskCommonArtifact,
    platform: PlatformTaskArtifact,
) -> _LoadedScene:
    """Translate verified task artifacts into the existing episode boundary."""
    if platform.common_artifact_sha256 != common.artifact_sha256:
        raise ValueError("formal platform task common artifact identity differs")
    if platform.key.common_key_sha256 != common.key.sha256():
        raise ValueError("formal platform task common key identity differs")
    if platform.platform_type != platform.key.platform_type:
        raise ValueError("formal platform task type differs")
    expected_diagnostics = {
        "start_identity_sha256": platform.key.qualified_start_identity_sha256,
        "platform_capability_sha256": platform.key.platform_capability_sha256,
        "physical_reachability_algorithm_id": (
            platform.key.reachability_algorithm_id
        ),
        "physical_evidence_algorithm_id": (
            platform.key.physical_evidence_algorithm_id
        ),
        "sensor_algorithm_id": platform.key.sensor_algorithm_id,
    }
    if any(
        platform.diagnostics.get(name) != expected
        for name, expected in expected_diagnostics.items()
    ):
        raise ValueError("formal platform task diagnostics differ from key")
    if (
        platform.platform_type in {"WHEELED", "LEGGED"}
        and platform.diagnostics.get("visibility_algorithm_id")
        != platform.key.visibility_algorithm_id
    ):
        raise ValueError("formal ground task visibility identity differs")
    geometry = common.geometry
    span = geometry.span_cells
    detail_cells = span * 20
    coarse_shape = (span, span)
    detail_shape = (detail_cells, detail_cells)
    mission_roi = np.ascontiguousarray(
        common.arrays["mission_roi_mask"], dtype=np.bool_
    )
    coverable = np.ascontiguousarray(
        platform.arrays["task_coverable_detail_mask"], dtype=np.bool_
    )
    physical_mask = np.ascontiguousarray(
        platform.arrays["physical_observation_pose_mask"], dtype=np.bool_
    )
    safe_mask = np.ascontiguousarray(
        platform.arrays["physical_safe_pose_mask"], dtype=np.bool_
    )
    if (
        mission_roi.shape != coarse_shape
        or physical_mask.shape != coarse_shape
        or safe_mask.shape != coarse_shape
        or coverable.shape != detail_shape
    ):
        raise ValueError("formal task artifact local geometry differs")
    if platform.diagnostics.get("resample_required") is True:
        raise ValueError("formal task artifact requires deterministic resampling")
    if not bool(coverable.any()):
        raise ValueError("formal task artifact has a zero denominator")
    priority_seed = hashlib.sha256(
        f"task-priority/v1\0{geometry.episode_seed}".encode("utf-8")
    ).hexdigest()
    priority = build_task_priority_mask(
        mission_roi,
        span_cells=span,
        priority_seed=priority_seed,
    )
    priority_detail = np.repeat(
        np.repeat(priority.coarse_mask, 20, axis=0),
        20,
        axis=1,
    )
    priority_coverable = np.ascontiguousarray(
        priority_detail & coverable,
        dtype=np.bool_,
    )
    prefix = platform.platform_type.lower()
    halo_row0, _, halo_column0, _ = geometry.halo_coarse_bounds_half_open
    row0, row1, column0, column1 = geometry.coarse_bounds_half_open
    local_rows = slice(row0 - halo_row0, row1 - halo_row0)
    local_columns = slice(
        column0 - halo_column0,
        column1 - halo_column0,
    )
    arrays: dict[str, np.ndarray] = {
        name: np.ascontiguousarray(
            common.arrays[name][local_rows, local_columns]
        )
        for name in (
            "elevation_m",
            "valid_mask",
            "physical_obstacle_ratio",
            "physical_obstacle_height_m",
            "forbidden_ratio",
        )
    }
    arrays.update(
        {
            "scoped_mission_roi": mission_roi,
            "task_priority_coarse_mask": np.ascontiguousarray(
                priority.coarse_mask, dtype=np.bool_
            ),
            "task_priority_detail_bits": pack_detail_mask(
                priority_coverable
            ),
            f"{prefix}_hard_feasible": safe_mask.astype(np.uint8),
            # The task artifact already carries the exact binary safe authority.
            # Runtime clearance remains a ranking feature and must not expand it.
            f"{prefix}_clearance_margin_norm": safe_mask.astype(np.float32),
            f"{prefix}_physical_observation_pose_bits": pack_detail_mask(
                physical_mask
            ),
            f"{prefix}_physical_observation_positions_um": (
                np.ascontiguousarray(
                    platform.arrays["physical_observation_positions_um"],
                    dtype=np.int64,
                )
            ),
            f"{prefix}_coverable_detail_bits": pack_detail_mask(coverable),
            f"{prefix}_coverable_ratio": np.ascontiguousarray(
                platform.arrays["coverable_ratio"], dtype=np.float32
            ),
        }
    )
    scene = _TaskLocalMultiResolutionScene(
        common,
        window_sha256=record.window_sha256,
        scene_seed=record.scene_seed,
    )
    coverable_count = int(coverable.sum(dtype=np.int64))
    coverable_sha = mask_sha256(coverable)
    priority_count = int(priority_coverable.sum(dtype=np.int64))
    entry = {
        **record.to_dict(),
        "task_area": {
            "sampling_algorithm": "deterministic-uniform-square/v1",
            "episode_seed": geometry.episode_seed,
            "span_cells": span,
            "size_m": span * 4.0,
            "coarse_bounds_half_open": list(
                geometry.coarse_bounds_half_open
            ),
            "detail_bounds_half_open": list(
                geometry.detail_bounds_half_open
            ),
            "halo_coarse_bounds_half_open": list(
                geometry.halo_coarse_bounds_half_open
            ),
            "coverable_detail_cell_count": coverable_count,
            "coverable_detail_mask_sha256": coverable_sha,
            "scale_bucket": geometry.scale_bucket.value,
            "priority_seed": priority_seed,
            "priority_side_cells": priority.side_cells,
            "priority_coarse_mask_sha256": priority.coarse_mask_sha256,
            "priority_detail_mask_sha256": mask_sha256(priority_coverable),
            "priority_coverable_detail_cell_count": priority_count,
        },
        "platform_coverability": {
            platform.platform_type: {
                "eligible": True,
                "qualified_start_cell": list(geometry.local_start_cell),
                "coverable_detail_shape": [detail_cells, detail_cells],
                "coverable_detail_cell_count": coverable_count,
                "coverable_detail_mask_sha256": coverable_sha,
            }
        },
    }
    return _LoadedScene(
        scene=scene,
        arrays=arrays,
        entry=entry,
        scenario={
            "scene_id": record.scene_id,
            "scene_seed": record.scene_seed,
            "world_bounds_m": list(record.world_bounds_m),
        },
        task_geometry=geometry,
        task_common_key_sha256=common.key.sha256(),
        task_common_artifact_sha256=common.artifact_sha256,
        platform_task_key_sha256=platform.key.sha256(),
        platform_task_artifact_sha256=platform.artifact_sha256,
    )


def _vec3(x: float, y: float, z: float) -> object:
    value = bridge_api.Vec3()
    value.x = float(x)
    value.y = float(y)
    value.z = float(z)
    return value


def _pose3(pose: Pose2) -> object:
    value = bridge_api.Pose3()
    value.position_m = _vec3(pose.x_m, pose.y_m, pose.elevation_m)
    value.orientation.w = math.cos(pose.yaw_rad / 2.0)
    value.orientation.z = math.sin(pose.yaw_rad / 2.0)
    return value


def _layer(values: np.ndarray, dtype: object) -> object:
    south_up = np.ascontiguousarray(
        np.flipud(np.asarray(values)).astype(dtype, copy=False).reshape(-1)
    )
    return bridge_api.GridLayer(south_up)


def _grid_map(
    *,
    canvas: MapCanvas,
    frame_id: str,
    elevation_m: np.ndarray,
    valid_mask: np.ndarray,
    physical_obstacle_ratio: np.ndarray,
    physical_obstacle_height_m: np.ndarray,
    forbidden_ratio: np.ndarray,
    observation_age_s: np.ndarray,
    observation_quality: np.ndarray,
    observation_count: np.ndarray,
    stamp_ns: int,
) -> object:
    valid = np.ascontiguousarray(valid_mask, dtype=np.bool_)
    obstacle = (np.asarray(physical_obstacle_ratio) > 0.0) & valid
    forbidden = (np.asarray(forbidden_ratio) > 0.0) & valid
    zeros = np.zeros(valid.shape, dtype=np.float32)
    grid = bridge_api.GridMap()
    grid.frame_id = frame_id
    grid.stamp.nanoseconds_since_epoch = stamp_ns
    grid.width = canvas.geometry.cells
    grid.height = canvas.geometry.cells
    grid.resolution_m = canvas.geometry.resolution_m
    grid.origin_m = _vec3(canvas.bounds_m[0], canvas.bounds_m[1], 0.0)
    grid.layers = {
        "elevation": _layer(np.where(valid, elevation_m, 0.0), np.float32),
        "valid_mask": _layer(valid, np.uint8),
        "obstacle": _layer(obstacle, np.uint8),
        "obstacle_height": _layer(
            np.where(valid, physical_obstacle_height_m, 0.0), np.float32
        ),
        "observation_age_s": _layer(
            np.where(valid, observation_age_s, 0.0), np.float32
        ),
        "observation_quality": _layer(
            np.where(valid, observation_quality, 0.0), np.float32
        ),
        "elevation_variance": _layer(zeros, np.float32),
        "obstacle_variance": _layer(zeros, np.float32),
        "observation_count": _layer(
            np.where(valid, observation_count, 0), np.uint32
        ),
        "forbidden": _layer(forbidden, np.uint8),
    }
    return grid


_PLANNER_GLOBAL_SCALE_FACTORS = (1, 2, 4, 8, 16, 20)
_PLANNER_GLOBAL_TARGET_AXIS_CELLS = 256


def _planner_global_canvas(source: MapCanvas) -> MapCanvas:
    """Choose the C++ hierarchical level for one exact task-local extent."""
    size_m = source.geometry.size_m
    for factor in _PLANNER_GLOBAL_SCALE_FACTORS:
        resolution_m = LOCAL_GEOMETRY.resolution_m * factor
        cells = int(
            math.ceil(
                size_m / resolution_m
                - 1.0e-9 * max(1.0, abs(size_m / resolution_m))
            )
        )
        if cells <= _PLANNER_GLOBAL_TARGET_AXIS_CELLS:
            padded_size_m = cells * resolution_m
            left, _, _, top = source.bounds_m
            bounds = (
                left,
                top - padded_size_m,
                left + padded_size_m,
                top,
            )
            identity = hashlib.sha256(
                (
                    f"formal-planner-global-level/v1\0{source.identity}\0"
                    f"{resolution_m:.17g}\0{cells}"
                ).encode("utf-8")
            ).hexdigest()
            return MapCanvas(
                identity,
                bounds,
                GridGeometry(
                    size_m=padded_size_m,
                    resolution_m=resolution_m,
                    cells=cells,
                ),
            )
    raise ValueError("formal task extent exceeds the planner global hierarchy")


@dataclass(frozen=True, slots=True)
class _PlannerGridProjection:
    source_identity: str
    target_identity: str
    target_cells: int
    valid_rows: np.ndarray
    valid_columns: np.ndarray
    source_rows: np.ndarray
    source_columns: np.ndarray
    source_flat_sorted: np.ndarray
    target_flat_sorted: np.ndarray


def _planner_grid_projection(
    source: MapCanvas,
    target: MapCanvas,
) -> _PlannerGridProjection:
    """Build immutable geometry indices shared by all projected layers."""
    target_cells = target.geometry.cells
    left, bottom, right, top = source.bounds_m
    target_left, _, _, target_top = target.bounds_m
    resolution_m = target.geometry.resolution_m
    target_columns_m = (
        target_left
        + (np.arange(target_cells, dtype=np.float64) + 0.5) * resolution_m
    )
    target_rows_m = (
        target_top
        - (np.arange(target_cells, dtype=np.float64) + 0.5) * resolution_m
    )
    valid_columns = np.ascontiguousarray(
        (target_columns_m >= left) & (target_columns_m < right)
    )
    valid_rows = np.ascontiguousarray(
        (target_rows_m > bottom) & (target_rows_m <= top)
    )
    source_columns = np.ascontiguousarray(
        np.floor(
            (target_columns_m[valid_columns] - left)
            / source.geometry.resolution_m
        ).astype(np.intp)
    )
    source_rows = np.ascontiguousarray(
        np.floor(
            (top - target_rows_m[valid_rows])
            / source.geometry.resolution_m
        ).astype(np.intp)
    )
    source_flat = np.ascontiguousarray(
        (
            source_rows[:, None] * source.geometry.cells
            + source_columns[None, :]
        ).reshape(-1),
        dtype=np.intp,
    )
    target_flat = np.ascontiguousarray(
        (
            np.flatnonzero(valid_rows)[:, None] * target_cells
            + np.flatnonzero(valid_columns)[None, :]
        ).reshape(-1),
        dtype=np.intp,
    )
    order = np.argsort(source_flat, kind="stable")
    source_flat_sorted = np.ascontiguousarray(source_flat[order], dtype=np.intp)
    target_flat_sorted = np.ascontiguousarray(target_flat[order], dtype=np.intp)
    for values in (
        valid_rows,
        valid_columns,
        source_rows,
        source_columns,
        source_flat_sorted,
        target_flat_sorted,
    ):
        values.flags.writeable = False
    return _PlannerGridProjection(
        source_identity=source.identity,
        target_identity=target.identity,
        target_cells=target_cells,
        valid_rows=valid_rows,
        valid_columns=valid_columns,
        source_rows=source_rows,
        source_columns=source_columns,
        source_flat_sorted=source_flat_sorted,
        target_flat_sorted=target_flat_sorted,
    )


def _planner_patch_indices_for_source_cells(
    projection: _PlannerGridProjection,
    source_flat_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Map source cells to all exact north-up planner cells that sample them."""
    if not isinstance(projection, _PlannerGridProjection):
        raise TypeError("planner projection is invalid")
    source = np.asarray(source_flat_indices)
    if (
        source.dtype != np.dtype(np.intp)
        or source.ndim != 1
        or not source.flags.c_contiguous
        or source.size == 0
    ):
        raise ValueError("planner source patch indices are invalid")
    if (source < 0).any() or not np.array_equal(source, np.unique(source)):
        raise ValueError("planner source patch indices must be unique and sorted")
    starts = np.searchsorted(projection.source_flat_sorted, source, side="left")
    stops = np.searchsorted(projection.source_flat_sorted, source, side="right")
    target_chunks = [
        projection.target_flat_sorted[start:stop]
        for start, stop in zip(starts.tolist(), stops.tolist(), strict=True)
        if start != stop
    ]
    source_chunks = [
        np.full(stop - start, value, dtype=np.intp)
        for value, start, stop in zip(
            source.tolist(), starts.tolist(), stops.tolist(), strict=True
        )
        if start != stop
    ]
    if not target_chunks:
        return (
            np.empty((0,), dtype=np.intp),
            np.empty((0,), dtype=np.intp),
        )
    target = np.concatenate(target_chunks)
    source_for_target = np.concatenate(source_chunks)
    order = np.argsort(target, kind="stable")
    return (
        np.ascontiguousarray(target[order], dtype=np.intp),
        np.ascontiguousarray(source_for_target[order], dtype=np.intp),
    )


def _candidate_boundary_evidence_sha256(sensor_state: object) -> str:
    """Materialize deferred evidence before forming a candidate identity."""
    sensor_state.materialize_all_observation_ages()
    return sensor_state.physical_evidence_sha256()


def _resolve_rebuilt_planning_failure(
    *,
    candidate_universe: PhysicalCandidateUniverse,
    failure_snapshot_id: str | None,
    planner_failed_candidate_ids: set[str],
    pending_failure: tuple[
        str,
        bridge_api.CandidateDisposition,
        str,
        bool,
    ]
    | None,
) -> tuple[str, set[str]]:
    """Apply a planner failure only when its physical candidate boundary survives.

    A rolling ground option can collect new evidence while retaining its locked
    target.  If the subsequent local request fails, that request belongs to the
    pre-rebuild candidate boundary.  Its suppression must not be carried into
    the new physical snapshot: the fresh universe is the authority from there.
    All other snapshot changes remain fail-closed.
    """
    rebuilt_snapshot_id = candidate_universe.physical_snapshot_id
    rebuilt_failed_ids = set(planner_failed_candidate_ids)
    if failure_snapshot_id != rebuilt_snapshot_id:
        failure_snapshot_id = rebuilt_snapshot_id
        rebuilt_failed_ids.clear()
    if pending_failure is None:
        return failure_snapshot_id, rebuilt_failed_ids

    (
        candidate_id,
        disposition,
        pending_snapshot_id,
        allow_locked_target_exit,
    ) = pending_failure
    if pending_snapshot_id != rebuilt_snapshot_id:
        if allow_locked_target_exit:
            return failure_snapshot_id, rebuilt_failed_ids
        raise ValueError("planning failure physical snapshot changed during rebuild")
    if (
        disposition
        == bridge_api.CandidateDisposition.SUPPRESS_FOR_CURRENT_PHYSICAL_SNAPSHOT
    ):
        current_ids = {
            candidate.candidate_id for candidate in candidate_universe.candidates
        }
        if candidate_id in current_ids:
            rebuilt_failed_ids.add(candidate_id)
        elif not allow_locked_target_exit:
            raise ValueError("planning failure candidate changed during rebuild")
    return failure_snapshot_id, rebuilt_failed_ids


def _project_task_grid_to_planner_level(
    values: np.ndarray,
    *,
    source: MapCanvas,
    target: MapCanvas,
    fill_value: object,
    projection: _PlannerGridProjection | None = None,
) -> np.ndarray:
    """Nearest containing-cell projection with any edge padding kept invalid."""
    array = np.asarray(values)
    source_cells = source.geometry.cells
    if array.shape != (source_cells, source_cells):
        raise ValueError("formal planner source grid geometry differs")
    fixed = (
        _planner_grid_projection(source, target)
        if projection is None
        else projection
    )
    if (
        not isinstance(fixed, _PlannerGridProjection)
        or fixed.source_identity != source.identity
        or fixed.target_identity != target.identity
        or fixed.target_cells != target.geometry.cells
    ):
        raise ValueError("formal planner projection geometry differs")
    output = np.full(
        (fixed.target_cells, fixed.target_cells),
        fill_value,
        dtype=array.dtype,
    )
    output[np.ix_(fixed.valid_rows, fixed.valid_columns)] = array[
        np.ix_(fixed.source_rows, fixed.source_columns)
    ]
    return np.ascontiguousarray(output)


def _yaw_from_pose(value: object) -> float:
    orientation = value.orientation
    return math.atan2(
        2.0 * (orientation.w * orientation.z + orientation.x * orientation.y),
        1.0 - 2.0 * (orientation.y * orientation.y + orientation.z * orientation.z),
    )


class FormalEpisode:
    """One cache-bound observed-only episode for one platform."""

    def __init__(
        self,
        *,
        worker_index: int,
        platform_type: str,
        capability: FrozenPlatformCapability,
        scenario_identity: ScenarioIdentity,
        loaded: _LoadedScene,
        start_cell: tuple[int, int],
        visited_candidate_filter_enabled: bool = True,
        detail_candidate_gain_enabled: bool = True,
        platform_candidate_reachability_enabled: bool = True,
    ) -> None:
        if (
            type(visited_candidate_filter_enabled) is not bool
            or type(detail_candidate_gain_enabled) is not bool
            or type(platform_candidate_reachability_enabled) is not bool
        ):
            raise TypeError("formal candidate compatibility flags must be boolean")
        self.worker_index = worker_index
        self.platform_type = platform_type
        self.capability = capability
        self.scenario_identity = scenario_identity
        self.episode_cursor = scenario_identity.episode_cursor
        self.loaded = loaded
        self.scene_id = loaded.scene.scene_id
        self.scene_seed = str(loaded.scenario.get("scene_seed"))
        self.start_seed = _formal_episode_seed("start", scenario_identity)
        self.episode_seed = _formal_episode_seed("episode", scenario_identity)
        if (
            loaded.task_geometry is None
            or loaded.task_common_key_sha256 is None
            or loaded.task_common_artifact_sha256 is None
            or loaded.platform_task_key_sha256 is None
            or loaded.platform_task_artifact_sha256 is None
        ):
            raise ValueError("formal task cache identity is missing")
        self.task_geometry = loaded.task_geometry
        self.task_geometry_sha256 = loaded.task_geometry.geometry_sha256
        self.task_common_key_sha256 = loaded.task_common_key_sha256
        self.task_common_artifact_sha256 = (
            loaded.task_common_artifact_sha256
        )
        self.platform_task_key_sha256 = loaded.platform_task_key_sha256
        self.platform_task_artifact_sha256 = (
            loaded.platform_task_artifact_sha256
        )
        task_area_payload = loaded.entry.get("task_area")
        if not isinstance(task_area_payload, Mapping):
            raise ValueError("formal Reward V4 task identity is missing")
        try:
            self.scale_bucket = TaskScaleBucket(
                str(task_area_payload["scale_bucket"])
            )
            self.task_span_cells = int(task_area_payload["span_cells"])
            self.priority_seed = str(task_area_payload["priority_seed"])
            self.priority_coarse_mask_sha256 = str(
                task_area_payload["priority_coarse_mask_sha256"]
            )
            self.priority_detail_mask_sha256 = str(
                task_area_payload["priority_detail_mask_sha256"]
            )
            self.priority_coverable_detail_cell_count = int(
                task_area_payload["priority_coverable_detail_cell_count"]
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("formal Reward V4 task identity is invalid") from error
        self._static_hard = loaded.arrays[
            f"{platform_type.lower()}_hard_feasible"
        ].astype(bool)
        self._static_clearance = loaded.arrays[
            f"{platform_type.lower()}_clearance_margin_norm"
        ]
        self.start_cell = start_cell
        prefix = platform_type.lower()
        coarse_cells = loaded.scene.base_canvas.geometry.cells
        cached_physical_mask = unpack_detail_mask(
            loaded.arrays[f"{prefix}_physical_observation_pose_bits"].copy(),
            (coarse_cells, coarse_cells),
        )
        if not bool(cached_physical_mask[start_cell]):
            raise ValueError("formal start is absent from cached physical authority")
        terrain_elevation_m = float(loaded.arrays["elevation_m"][start_cell])
        start_x_m, start_y_m = loaded.scene.base_canvas.grid_center_world(
            *start_cell
        )
        pose_elevation_m = terrain_elevation_m
        if platform_type == "HOPPER":
            rows, columns = np.nonzero(cached_physical_mask)
            matching = np.flatnonzero(
                (rows == start_cell[0]) & (columns == start_cell[1])
            )
            positions_um = loaded.arrays[
                "hopper_physical_observation_positions_um"
            ]
            if len(matching) != 1 or len(positions_um) != len(rows):
                raise ValueError("formal Hopper start authority is ambiguous")
            exact_start = np.asarray(
                positions_um[int(matching[0])], dtype=np.float64
            ) / 1_000_000.0
            if loaded.scene.base_canvas.world_to_grid(
                float(exact_start[0]), float(exact_start[1])
            ) != start_cell:
                raise ValueError("formal Hopper start leaves its cached cell")
            start_x_m = float(exact_start[0])
            start_y_m = float(exact_start[1])
            pose_elevation_m = float(exact_start[2])
        self.current_pose = Pose2(
            float(start_x_m),
            float(start_y_m),
            0.0,
            "map",
            pose_elevation_m,
        )
        self._current_legged_body_z_m = (
            terrain_elevation_m
            + (
                float(capability.typed_capability.body_height_m.lower)
                + float(capability.typed_capability.body_height_m.upper)
            )
            / 2.0
            if platform_type == "LEGGED"
            else pose_elevation_m
        )
        self.start_is_safe = bool(self._static_hard[start_cell])
        self._mission_roi = self._build_mission_roi()
        priority_coarse = np.ascontiguousarray(
            loaded.arrays.get("task_priority_coarse_mask"),
            dtype=np.bool_,
        )
        if priority_coarse.shape != self._mission_roi.shape:
            raise ValueError("formal task priority geometry is invalid")
        if mask_sha256(priority_coarse) != self.priority_coarse_mask_sha256:
            raise ValueError("formal task priority coarse identity differs")
        self.mission = MissionRaster(
            loaded.scene.base_canvas,
            priority=priority_coarse.astype(np.float32),
            roi_ratio=self._mission_roi.astype(np.float32),
        )
        coverability = loaded.entry["platform_coverability"][platform_type]
        detail_shape_raw = coverability["coverable_detail_shape"]
        if not isinstance(detail_shape_raw, list) or len(detail_shape_raw) != 2:
            raise ValueError("formal coverability detail geometry is invalid")
        self.sensor_state = MultiresSensorObservationState(
            scene=loaded.scene,
            tile_provider=_TaskLocalSceneTileProvider(loaded.scene, capacity=8),
            mission_roi_ratio=self.mission.roi_ratio,
            mission_priority=self.mission.priority,
            coverable_detail_shape=(
                int(detail_shape_raw[0]), int(detail_shape_raw[1])
            ),
            coverable_detail_bits=loaded.arrays[
                f"{prefix}_coverable_detail_bits"
            ],
            coverable_detail_cell_count=int(
                coverability["coverable_detail_cell_count"]
            ),
            coverable_mask_sha256=str(
                coverability["coverable_detail_mask_sha256"]
            ),
            priority_detail_shape=(
                int(detail_shape_raw[0]), int(detail_shape_raw[1])
            ),
            priority_detail_bits=loaded.arrays["task_priority_detail_bits"],
            priority_detail_cell_count=(
                self.priority_coverable_detail_cell_count
            ),
            priority_detail_mask_sha256=self.priority_detail_mask_sha256,
        )
        self._planner_global_canvas = _planner_global_canvas(
            self.sensor_state.observed.canvas
        )
        self._planner_grid_projection = _planner_grid_projection(
            self.sensor_state.observed.canvas,
            self._planner_global_canvas,
        )
        self._persistent_planner_global_map: object | None = None
        self._candidate_builder = CandidateBuilderV2(self.sensor_state)
        self._legacy_candidate_builder = CandidateBuilderV2(
            NativeVisibilityEstimator(
                SensorGeometry(FORMAL_SENSOR_RANGE_M, FORMAL_SENSOR_FOV_RAD),
                resolution_m=loaded.scene.base_canvas.geometry.resolution_m,
            )
        )
        self._detail_candidate_gain_enabled = detail_candidate_gain_enabled
        self._platform_candidate_reachability_enabled = (
            platform_candidate_reachability_enabled
        )
        self._current_candidate_gain_resolution_m: float | None = None
        self._observation_builder = ObservationBuilderV2()
        self._bridge = bridge_api.PlannerBridge()
        self._snapshot: _MapSnapshot | None = None
        self._revision = 0
        self._pending_hop_landing: Pose2 | None = None
        self._pending_hop_midpoint: Pose2 | None = None
        self._pending_hop_elapsed_s = 0.0
        self._hopper_feedback_phase = 0
        self._active_ground_option: _GroundOption | None = None
        self._defer_candidate_rebuild = False
        self._planner_failure_snapshot_id: str | None = None
        self._planner_failed_candidate_ids: set[str] = set()
        self._pending_planning_failure: (
            tuple[str, bridge_api.CandidateDisposition, str, bool] | None
        ) = None
        self.last_hop_available_delta_v_mps = 0.0
        self._reveal_history: list[FormalRevealState] = []
        self._replay_event_kinds: list[str] = []
        self._visited_candidate_cells = {start_cell}
        self._navigation_stack = [self.current_pose]
        self._visited_candidate_filter_enabled = visited_candidate_filter_enabled
        self.controller = ObservationBoundaryController(
            platform_type=platform_type,
            sensor_state=self.sensor_state,
            policy_observation_builder=self.build_policy_observation,
            episode_id=(
                f"{self.scene_id}/{platform_type.lower()}/{worker_index}"
                f"/episode-{self.episode_cursor}"
            ),
            mission_revision=1,
            initial_state_time_ns=1_000_000_000,
        )
        initial = self.controller.reset(self.current_pose)
        self.initial_observation = initial.next_observation

    @staticmethod
    def _pose_state(pose: Pose2) -> FormalPoseState:
        return FormalPoseState(
            x_m=float(pose.x_m),
            y_m=float(pose.y_m),
            yaw_rad=float(pose.yaw_rad),
            elevation_m=float(pose.elevation_m),
            frame_id=pose.frame_id,
        )

    @staticmethod
    def _pose_from_state(pose: FormalPoseState) -> Pose2:
        return Pose2(
            pose.x_m,
            pose.y_m,
            pose.yaw_rad,
            pose.frame_id,
            pose.elevation_m,
        )

    def _record_reveal(
        self,
        evidence: SensorBoundaryEvidence,
        execution_state: str,
        *,
        defer_candidate_rebuild: bool | None = None,
    ) -> None:
        if defer_candidate_rebuild is None:
            option = self._active_ground_option
            defer_candidate_rebuild = option is not None and math.hypot(
                evidence.pose_map.x_m - option.target_x_m,
                evidence.pose_map.y_m - option.target_y_m,
            ) > option.tolerance_m + 1.0e-6
        elif type(defer_candidate_rebuild) is not bool:
            raise ValueError("formal reveal candidate rebuild flag is invalid")
        self._defer_candidate_rebuild = defer_candidate_rebuild
        canvas = self.loaded.scene.base_canvas
        cell = canvas.world_to_grid(
            evidence.pose_map.x_m,
            evidence.pose_map.y_m,
        )
        if (
            len(self._navigation_stack) >= 2
            and cell
            == canvas.world_to_grid(
                self._navigation_stack[-2].x_m,
                self._navigation_stack[-2].y_m,
            )
        ):
            self._navigation_stack.pop()
        elif cell != canvas.world_to_grid(
            self._navigation_stack[-1].x_m,
            self._navigation_stack[-1].y_m,
        ):
            self._navigation_stack.append(evidence.pose_map)
        visited_poses = [evidence.pose_map]
        visited_poses.extend(sample.pose_map for sample in evidence.path_samples)
        self._visited_candidate_cells.update(
            canvas.world_to_grid(pose.x_m, pose.y_m) for pose in visited_poses
        )
        self._reveal_history.append(
            FormalRevealState(
                pose=self._pose_state(evidence.pose_map),
                elapsed_s=float(evidence.elapsed_s),
                path_samples=tuple(
                    FormalPathSampleState(
                        pose=self._pose_state(sample.pose_map),
                        elapsed_s=float(sample.elapsed_s),
                    )
                    for sample in evidence.path_samples
                ),
                execution_state=execution_state,
                legged_body_z_m=float(self._current_legged_body_z_m),
                defer_candidate_rebuild=defer_candidate_rebuild,
            )
        )
        self._replay_event_kinds.append("REVEAL")

    def replay_state(self, state: FormalWorkerState) -> None:
        """Replay physical evidence and verify every persisted boundary identity."""
        if not isinstance(state, FormalWorkerState):
            raise ValueError("formal replay state is invalid")
        expected_static = (
            self.scenario_identity.scenario_schedule_id,
            self.platform_type,
            self.worker_index,
            self.scenario_identity.platform_worker_index,
            self.scenario_identity.platform_worker_count,
            self.episode_cursor,
            self.scene_id,
            self.scene_seed,
            self.start_seed,
            self.episode_seed,
            self.task_geometry_sha256,
            self.task_common_key_sha256,
            self.task_common_artifact_sha256,
            self.platform_task_key_sha256,
            self.platform_task_artifact_sha256,
            self.task_geometry.coarse_bounds_half_open,
            self.task_geometry.detail_bounds_half_open,
            self.task_geometry.halo_coarse_bounds_half_open,
            self.sensor_state.coverable_mask_sha256,
            self.sensor_state.coverable_detail_cell_count,
            self.scale_bucket.value,
            self.task_span_cells,
            self.priority_seed,
            self.priority_coarse_mask_sha256,
            self.priority_detail_mask_sha256,
            self.priority_coverable_detail_cell_count,
            self.start_cell,
        )
        persisted_static = (
            state.scenario_schedule_id,
            state.platform_type,
            state.worker_index,
            state.platform_worker_index,
            state.platform_worker_count,
            state.episode_cursor,
            state.scene_id,
            state.scene_seed,
            state.start_seed,
            state.episode_seed,
            state.task_geometry_sha256,
            state.task_common_key_sha256,
            state.task_common_artifact_sha256,
            state.platform_task_key_sha256,
            state.platform_task_artifact_sha256,
            state.task_coarse_bounds_half_open,
            state.task_detail_bounds_half_open,
            state.task_halo_bounds_half_open,
            state.coverability_mask_sha256,
            state.coverable_detail_cell_count,
            state.scale_bucket,
            state.task_span_cells,
            state.priority_seed,
            state.priority_coarse_mask_sha256,
            state.priority_detail_mask_sha256,
            state.priority_coverable_detail_cell_count,
            state.start_cell,
        )
        if persisted_static != expected_static:
            raise ValueError("formal replay worker identity differs")
        if self._active_ground_option is not None:
            raise ValueError("formal replay cannot contain an active ground option")

        reveal_index = 0
        last_reveal_index = max(
            (
                index
                for index, kind in enumerate(state.replay_event_kinds)
                if kind == "REVEAL"
            ),
            default=-1,
        )
        active_failure_ids = iter(state.planner_failed_candidate_ids)
        for event_index, event_kind in enumerate(state.replay_event_kinds):
            if event_kind != "REVEAL":
                if (
                    event_kind == "PLANNING_FAILURE_SUPPRESS"
                    and event_index > last_reveal_index
                ):
                    candidate_id = next(active_failure_ids)
                    self.refresh_after_planning_failure(
                        candidate_id,
                        bridge_api.CandidateDisposition.SUPPRESS_FOR_CURRENT_PHYSICAL_SNAPSHOT,
                        self._snapshot.planning_physical_snapshot_id,
                    )
                    continue
                self._defer_candidate_rebuild = False
                execution_state = (
                    self.controller.current_observation.observation_identities[0]
                    .execution_state
                )
                self.controller.rebuild_without_sensor_update(
                    pose_map=self.current_pose,
                    execution_state=execution_state,
                )
                self._replay_event_kinds.append(event_kind)
                continue
            reveal = state.reveal_history[reveal_index]
            reveal_index += 1
            pose = self._pose_from_state(reveal.pose)
            evidence = SensorBoundaryEvidence(
                pose,
                reveal.elapsed_s,
                path_samples=tuple(
                    SensorPathSample(
                        self._pose_from_state(sample.pose),
                        sample.elapsed_s,
                    )
                    for sample in reveal.path_samples
                ),
            )
            self.current_pose = pose
            self._current_legged_body_z_m = reveal.legged_body_z_m
            self._record_reveal(
                evidence,
                reveal.execution_state,
                defer_candidate_rebuild=reveal.defer_candidate_rebuild,
            )
            if self.platform_type == "HOPPER" and evidence.path_samples:
                self.controller.replay_hopper_trajectory(evidence)
            else:
                self.controller.after_execution(
                    platform_type=self.platform_type,
                    execution_state=reveal.execution_state,
                    evidence=evidence,
                )

        snapshot = self._snapshot
        if snapshot is None:
            raise ValueError("formal replay produced no physical snapshot")
        if snapshot.candidate_universe.physical_snapshot_id != state.physical_snapshot_id:
            raise ValueError("formal replay physical snapshot differs")
        if (
            self.sensor_state.observed_coverable_detail_cell_count
            != state.observed_coverable_detail_cell_count
        ):
            raise ValueError("formal replay observed coverage count differs")
        if (
            self.sensor_state.observed_coverable_mask_sha256()
            != state.observed_coverable_mask_sha256
        ):
            raise ValueError("formal replay observed coverage mask differs")

        self.last_hop_available_delta_v_mps = (
            state.last_hop_available_delta_v_mps
        )
        observation = self.controller.current_observation
        snapshot = self._snapshot
        if snapshot is None:
            raise ValueError("formal replay produced no final physical snapshot")
        identity = observation.observation_identities[0]
        candidate_ids = tuple(str(item) for item in snapshot.candidates.candidate_ids)
        candidate_mask = tuple(bool(item) for item in snapshot.candidates.mask)
        decision_snapshot = snapshot.candidate_universe.decision_snapshot
        if decision_snapshot is None:
            raise ValueError("formal replay candidate decision snapshot is missing")
        terminal_reason = _audit_candidate_boundary(
            decision_snapshot, self.platform_type
        )
        replayed_decision_snapshot = replace(
            decision_snapshot,
            global_search_elapsed_s=(
                state.candidate_decision_snapshot.global_search_elapsed_s
            ),
            candidate_refresh_elapsed_s=(
                state.candidate_decision_snapshot.candidate_refresh_elapsed_s
            ),
            scan_elapsed_s=state.candidate_decision_snapshot.scan_elapsed_s,
            sort_elapsed_s=state.candidate_decision_snapshot.sort_elapsed_s,
            top64_elapsed_s=state.candidate_decision_snapshot.top64_elapsed_s,
            reserve_elapsed_s=(
                state.candidate_decision_snapshot.reserve_elapsed_s
            ),
        )
        replayed = (
            self._revision,
            self.sensor_state.evidence_generation,
            self.sensor_state.physical_evidence_sha256(),
            snapshot.candidate_universe.physical_snapshot_id,
            snapshot.candidate_universe_sha256,
            tuple(sorted(self._planner_failed_candidate_ids)),
            self._pose_state(self.current_pose),
            self._current_legged_body_z_m,
            identity.state_time_ns,
            identity,
            policy_batch_sha256(observation),
            candidate_ids,
            candidate_mask,
            replayed_decision_snapshot,
            None if terminal_reason is None else terminal_reason.value,
            self._defer_candidate_rebuild,
            self._current_candidate_gain_resolution_m,
        )
        persisted = (
            state.observation_revision,
            state.physical_evidence_generation,
            state.physical_evidence_sha256,
            state.physical_snapshot_id,
            state.physical_candidate_universe_sha256,
            state.planner_failed_candidate_ids,
            state.current_pose,
            state.legged_body_z_m,
            state.state_time_ns,
            state.observation_identity,
            state.policy_batch_sha256,
            state.candidate_ids,
            state.candidate_mask,
            state.candidate_decision_snapshot,
            state.terminal_reason,
            state.defer_candidate_rebuild,
            state.candidate_gain_resolution_m,
        )
        if replayed != persisted:
            names = (
                "observation_revision",
                "physical_evidence_generation",
                "physical_evidence_sha256",
                "physical_snapshot_id",
                "physical_candidate_universe_sha256",
                "planner_failed_candidate_ids",
                "current_pose",
                "legged_body_z_m",
                "state_time_ns",
                "observation_identity",
                "policy_batch_sha256",
                "candidate_ids",
                "candidate_mask",
                "candidate_decision_snapshot",
                "terminal_reason",
                "defer_candidate_rebuild",
                "candidate_gain_resolution_m",
            )
            mismatches = tuple(
                name
                for name, replayed_value, persisted_value in zip(
                    names, replayed, persisted, strict=True
                )
                if replayed_value != persisted_value
            )
            raise ValueError(
                f"formal replay physical identity differs: {mismatches}"
            )
        self._snapshot = replace(
            snapshot,
            candidate_universe=replace(
                snapshot.candidate_universe,
                decision_snapshot=replayed_decision_snapshot,
            ),
        )
        self.initial_observation = observation

    def snapshot_state(self, environment: object) -> FormalWorkerState:
        """Capture one strict physical decision-boundary replay state."""
        if self._active_ground_option is not None:
            raise ValueError("formal snapshot cannot contain an active ground option")
        stable_state = getattr(environment, "snapshot_stable_state", None)
        if not callable(stable_state):
            raise ValueError("formal snapshot environment is invalid")
        stable_payload = stable_state()
        execution_state = stable_payload.get("execution_state")
        cumulative_executed_path_m = stable_payload.get(
            "cumulative_executed_path_m"
        )
        observation = getattr(environment, "current_observation", None)
        if not isinstance(observation, PolicyBatch):
            raise ValueError("formal snapshot observation is invalid")
        controller_observation = self.controller.current_observation
        if (
            execution_state
            != controller_observation.observation_identities[0].execution_state
            or observation.observation_identities
            != controller_observation.observation_identities
            or policy_batch_sha256(observation)
            != policy_batch_sha256(controller_observation)
        ):
            raise ValueError("formal snapshot environment boundary differs")
        snapshot = self._snapshot
        if snapshot is None:
            raise ValueError("formal snapshot has no physical candidate state")
        if self._current_candidate_gain_resolution_m is None:
            raise ValueError("formal snapshot candidate gain identity is missing")
        failed_ids = tuple(sorted(self._planner_failed_candidate_ids))
        if failed_ids and self._planner_failure_snapshot_id != (
            snapshot.candidate_universe.physical_snapshot_id
        ):
            raise ValueError("formal snapshot planner failure identity differs")
        decision_snapshot = snapshot.candidate_universe.decision_snapshot
        if decision_snapshot is None:
            raise ValueError("formal candidate decision snapshot is missing")
        terminal_reason = _audit_candidate_boundary(
            decision_snapshot, self.platform_type
        )
        persisted_decision_snapshot = replace(
            decision_snapshot,
            global_search_elapsed_s=0.0,
            candidate_refresh_elapsed_s=0.0,
            scan_elapsed_s=0.0,
            sort_elapsed_s=0.0,
            top64_elapsed_s=0.0,
            reserve_elapsed_s=0.0,
        )
        identity = observation.observation_identities[0]
        return FormalWorkerState(
            scenario_schedule_id=self.scenario_identity.scenario_schedule_id,
            platform_type=self.platform_type,
            worker_index=self.worker_index,
            platform_worker_index=self.scenario_identity.platform_worker_index,
            platform_worker_count=self.scenario_identity.platform_worker_count,
            episode_cursor=self.episode_cursor,
            scene_id=self.scene_id,
            scene_seed=self.scene_seed,
            start_seed=self.start_seed,
            episode_seed=self.episode_seed,
            task_geometry_sha256=self.task_geometry_sha256,
            task_common_key_sha256=self.task_common_key_sha256,
            task_common_artifact_sha256=self.task_common_artifact_sha256,
            platform_task_key_sha256=self.platform_task_key_sha256,
            platform_task_artifact_sha256=(
                self.platform_task_artifact_sha256
            ),
            task_coarse_bounds_half_open=(
                self.task_geometry.coarse_bounds_half_open
            ),
            task_detail_bounds_half_open=(
                self.task_geometry.detail_bounds_half_open
            ),
            task_halo_bounds_half_open=(
                self.task_geometry.halo_coarse_bounds_half_open
            ),
            coverability_mask_sha256=str(
                self.sensor_state.coverable_mask_sha256
            ),
            observed_coverable_mask_sha256=(
                self.sensor_state.observed_coverable_mask_sha256()
            ),
            coverable_detail_cell_count=int(
                self.sensor_state.coverable_detail_cell_count
            ),
            observed_coverable_detail_cell_count=int(
                self.sensor_state.observed_coverable_detail_cell_count
            ),
            scale_bucket=self.scale_bucket.value,
            task_span_cells=self.task_span_cells,
            priority_seed=self.priority_seed,
            priority_coarse_mask_sha256=(
                self.priority_coarse_mask_sha256
            ),
            priority_detail_mask_sha256=self.priority_detail_mask_sha256,
            priority_coverable_detail_cell_count=(
                self.priority_coverable_detail_cell_count
            ),
            start_cell=self.start_cell,
            current_pose=self._pose_state(self.current_pose),
            legged_body_z_m=float(self._current_legged_body_z_m),
            execution_state=str(execution_state),
            cumulative_executed_path_m=float(cumulative_executed_path_m),
            observation_revision=self._revision,
            physical_snapshot_id=(
                snapshot.candidate_universe.physical_snapshot_id
            ),
            physical_evidence_generation=self.sensor_state.evidence_generation,
            physical_evidence_sha256=(
                self.sensor_state.physical_evidence_sha256()
            ),
            physical_candidate_universe_sha256=(
                snapshot.candidate_universe_sha256
            ),
            planner_failed_candidate_ids=failed_ids,
            state_time_ns=identity.state_time_ns,
            reveal_history=tuple(self._reveal_history),
            replay_event_kinds=tuple(self._replay_event_kinds),
            observation_identity=identity,
            policy_batch_sha256=policy_batch_sha256(observation),
            candidate_ids=tuple(
                str(item) for item in snapshot.candidates.candidate_ids
            ),
            candidate_mask=tuple(
                bool(item) for item in snapshot.candidates.mask
            ),
            candidate_decision_snapshot=persisted_decision_snapshot,
            terminal_reason=(
                None if terminal_reason is None else terminal_reason.value
            ),
            defer_candidate_rebuild=self._defer_candidate_rebuild,
            last_hop_available_delta_v_mps=(
                self.last_hop_available_delta_v_mps
            ),
            candidate_gain_resolution_m=(
                self._current_candidate_gain_resolution_m
            ),
        )

    def current_candidate_diagnostics(self) -> CandidateDiagnostics:
        snapshot = self._snapshot
        if snapshot is None:
            return CandidateDiagnostics()
        return snapshot.candidates.diagnostics

    def current_candidate_decision_snapshot(self) -> CandidateDecisionSnapshot:
        snapshot = self._snapshot
        if snapshot is None or snapshot.candidate_universe.decision_snapshot is None:
            raise RuntimeError("formal candidate decision snapshot is unavailable")
        return snapshot.candidate_universe.decision_snapshot

    def remaining_coverable_detail_cell_count(self) -> int:
        remaining = self.sensor_state.remaining_coverable_detail_cell_count
        if remaining is None:
            raise RuntimeError("formal coverability truth diagnostic is unavailable")
        return remaining

    def _build_mission_roi(self) -> np.ndarray:
        scoped = self.loaded.arrays.get("scoped_mission_roi")
        if scoped is not None:
            roi = np.ascontiguousarray(scoped, dtype=np.bool_)
            cells = self.loaded.scene.base_canvas.geometry.cells
            if roi.shape != (cells, cells):
                raise ValueError("formal scoped mission ROI geometry is invalid")
            if not bool(roi[self.start_cell]):
                raise ValueError("formal scoped mission ROI excludes the start")
            return roi
        return build_formal_mission_roi(self.loaded.arrays)

    def _observed_maps(
        self,
    ) -> tuple[object, object, object, DetailObservedWindow]:
        observed = self.sensor_state.observed
        stamp_ns = 1_000_000_000 + self._revision * 1_000_000
        global_map = _grid_map(
            canvas=observed.canvas,
            frame_id="map",
            elevation_m=observed.elevation_m,
            valid_mask=observed.valid_mask,
            physical_obstacle_ratio=observed.physical_obstacle_ratio,
            physical_obstacle_height_m=(
                self.sensor_state.coarse_obstacle_height_m
            ),
            forbidden_ratio=self.sensor_state.coarse_forbidden_ratio,
            observation_age_s=observed.observation_age_s,
            observation_quality=observed.observation_quality,
            observation_count=observed.observation_count,
            stamp_ns=stamp_ns,
        )
        planner_canvas = self._planner_global_canvas

        def planner_level(values: np.ndarray, fill_value: object) -> np.ndarray:
            return _project_task_grid_to_planner_level(
                values,
                source=observed.canvas,
                target=planner_canvas,
                fill_value=fill_value,
                projection=self._planner_grid_projection,
            )

        planner_global_map = _grid_map(
            canvas=planner_canvas,
            frame_id="map",
            elevation_m=planner_level(observed.elevation_m, np.float32(0.0)),
            valid_mask=planner_level(observed.valid_mask, False),
            physical_obstacle_ratio=planner_level(
                observed.physical_obstacle_ratio, np.float32(0.0)
            ),
            physical_obstacle_height_m=planner_level(
                self.sensor_state.coarse_obstacle_height_m, np.float32(0.0)
            ),
            forbidden_ratio=planner_level(
                self.sensor_state.coarse_forbidden_ratio, np.float32(0.0)
            ),
            observation_age_s=planner_level(
                observed.observation_age_s, np.float32(0.0)
            ),
            observation_quality=planner_level(
                observed.observation_quality, np.float32(0.0)
            ),
            observation_count=planner_level(observed.observation_count, 0),
            stamp_ns=stamp_ns,
        )
        detail = self.sensor_state.planning_observation(self.current_pose)
        local_map = _grid_map(
            canvas=detail.canvas,
            frame_id="odom",
            elevation_m=detail.elevation_m,
            valid_mask=detail.valid_mask,
            physical_obstacle_ratio=detail.physical_obstacle_ratio,
            physical_obstacle_height_m=detail.physical_obstacle_height_m,
            forbidden_ratio=detail.forbidden_ratio,
            observation_age_s=detail.observation_age_s,
            observation_quality=detail.observation_quality,
            observation_count=detail.observation_count,
            stamp_ns=stamp_ns,
        )
        self._persistent_planner_global_map = planner_global_map
        self.sensor_state.consume_dirty_coarse_cell_ids()
        return global_map, planner_global_map, local_map, detail

    def _rolling_planner_and_local_map(self) -> tuple[object, object]:
        """Patch only committed coarse evidence before the next local request."""
        if self._persistent_planner_global_map is None:
            _, planner_global_map, local_map, _ = self._observed_maps()
            return planner_global_map, local_map
        stamp_ns = 1_000_000_000 + self._revision * 1_000_000
        planner_global_map = self._persistent_planner_global_map
        self._patch_persistent_planner_global_map(
            planner_global_map, stamp_ns=stamp_ns
        )
        detail = self.sensor_state.planning_observation(self.current_pose)
        local_map = _grid_map(
            canvas=detail.canvas,
            frame_id="odom",
            elevation_m=detail.elevation_m,
            valid_mask=detail.valid_mask,
            physical_obstacle_ratio=detail.physical_obstacle_ratio,
            physical_obstacle_height_m=detail.physical_obstacle_height_m,
            forbidden_ratio=detail.forbidden_ratio,
            observation_age_s=detail.observation_age_s,
            observation_quality=detail.observation_quality,
            observation_count=detail.observation_count,
            stamp_ns=stamp_ns,
        )
        return planner_global_map, local_map

    def _patch_persistent_planner_global_map(
        self,
        planner_global_map: object,
        *,
        stamp_ns: int,
    ) -> None:
        dirty = self.sensor_state.consume_dirty_coarse_cell_ids()
        planner_global_map.stamp.nanoseconds_since_epoch = stamp_ns
        if not dirty:
            return
        source_flat = np.ascontiguousarray(dirty, dtype=np.intp)
        target_north_flat, source_for_target = (
            _planner_patch_indices_for_source_cells(
                self._planner_grid_projection, source_flat
            )
        )
        if target_north_flat.size == 0:
            return
        target_cells = self._planner_global_canvas.geometry.cells
        target_rows = target_north_flat // target_cells
        target_columns = target_north_flat % target_cells
        target_south_flat = (
            (target_cells - 1 - target_rows) * target_cells + target_columns
        )
        order = np.argsort(target_south_flat, kind="stable")
        patch_indices = np.ascontiguousarray(
            target_south_flat[order], dtype=np.uint32
        )
        source_for_target = source_for_target[order]
        observed = self.sensor_state.observed
        source_cells = observed.canvas.geometry.cells
        source_rows = source_for_target // source_cells
        source_columns = source_for_target % source_cells
        valid = observed.valid_mask[source_rows, source_columns]

        def sampled(values: np.ndarray, fill_value: object) -> np.ndarray:
            return np.ascontiguousarray(
                np.where(
                    valid,
                    values[source_rows, source_columns],
                    fill_value,
                )
            )

        layer_values = {
            "elevation": sampled(observed.elevation_m, np.float32(0.0)).astype(
                np.float32, copy=False
            ),
            "valid_mask": np.ascontiguousarray(valid, dtype=np.uint8),
            "obstacle": np.ascontiguousarray(
                (observed.physical_obstacle_ratio[source_rows, source_columns] > 0.0)
                & valid,
                dtype=np.uint8,
            ),
            "obstacle_height": sampled(
                self.sensor_state.coarse_obstacle_height_m, np.float32(0.0)
            ).astype(np.float32, copy=False),
            "forbidden": np.ascontiguousarray(
                (self.sensor_state.coarse_forbidden_ratio[
                    source_rows, source_columns
                ] > 0.0)
                & valid,
                dtype=np.uint8,
            ),
            "observation_age_s": sampled(
                observed.observation_age_s, np.float32(0.0)
            ).astype(np.float32, copy=False),
            "observation_quality": sampled(
                observed.observation_quality, np.float32(0.0)
            ).astype(np.float32, copy=False),
            "observation_count": sampled(
                observed.observation_count, np.uint32(0)
            ).astype(np.uint32, copy=False),
        }
        for name, values in layer_values.items():
            planner_global_map.patch_layer_flat_indices(
                name, patch_indices, np.ascontiguousarray(values)
            )

    def _base_request(self, global_map: object, local_map: object) -> object:
        request = bridge_api.TrainingPlanRequest()
        request.mission_id = f"formal/{self.scene_id}"
        request.mission_revision = 1
        request.platform_id = self.capability.platform_id
        request.capability_version = self.capability.capability_version
        request.global_map_generation = self._revision
        request.local_map_generation = self._revision
        request.map_from_odom_generation = 1
        request.state_time.nanoseconds_since_epoch = (
            1_000_000_000 + self._revision * 1_000_000
        )
        pose = _pose3(self.current_pose)
        if self.platform_type == "WHEELED":
            state = bridge_api.WheeledState()
            state.pose = pose
        elif self.platform_type == "LEGGED":
            state = bridge_api.LeggedState()
            state.body_pose = _pose3(
                Pose2(
                    self.current_pose.x_m,
                    self.current_pose.y_m,
                    self.current_pose.yaw_rad,
                    self.current_pose.frame_id,
                    self._current_legged_body_z_m,
                )
            )
        else:
            state = bridge_api.HopperState()
            state.pose = pose
        request.current_state = state
        request.capability = self.capability.to_bridge_capability()
        request.world.global_map = global_map
        request.world.local_map = local_map
        request.world.map_from_odom.parent_frame = "map"
        request.world.map_from_odom.child_frame = "odom"
        request.world.map_from_odom.stamp.nanoseconds_since_epoch = (
            request.state_time.nanoseconds_since_epoch
        )
        request.config.global_map.base_resolution_m = LOCAL_GEOMETRY.resolution_m
        request.config.global_map.maximum_level = 5
        request.config.global_map.target_axis_cells = 256
        request.config.wheel.xy_resolution_m = LOCAL_GEOMETRY.resolution_m
        request.config.wheel.yaw_bin_count = 64
        request.config.legged.xy_resolution_m = LOCAL_GEOMETRY.resolution_m
        request.config.legged.yaw_bin_count = 64
        request.config.local_frontier.additional_corridor_margin_m = 2.0
        return request

    def _ground_endpoint_feasibility(
        self,
        target_positions_m: np.ndarray,
        *,
        planner_global_map: object,
    ) -> np.ndarray:
        """Batch exact ground endpoints through observed-only 0.2 m windows."""
        targets = np.asarray(target_positions_m)
        if (
            self.platform_type not in {"WHEELED", "LEGGED"}
            or targets.dtype != np.dtype(np.float64)
            or targets.ndim != 2
            or targets.shape[1:] != (3,)
            or not targets.flags.c_contiguous
            or not np.isfinite(targets).all()
        ):
            raise ValueError("ground endpoint authority inputs are invalid")
        output = np.zeros(len(targets), dtype=np.bool_)
        if not len(targets):
            return output
        bucket_size_m = 48.0
        grouped: dict[tuple[int, int], list[int]] = {}
        for index, position in enumerate(targets):
            key = (
                math.floor(float(position[0]) / bucket_size_m),
                math.floor(float(position[1]) / bucket_size_m),
            )
            grouped.setdefault(key, []).append(index)
        for group_index, key in enumerate(sorted(grouped)):
            indices = grouped[key]
            group_targets = np.ascontiguousarray(
                targets[indices], dtype=np.float64
            )
            center_x_m = float(
                (
                    group_targets[:, 0].min()
                    + group_targets[:, 0].max()
                )
                / 2.0
            )
            center_y_m = float(
                (
                    group_targets[:, 1].min()
                    + group_targets[:, 1].max()
                )
                / 2.0
            )
            center_z_m = float(group_targets[:, 2].mean(dtype=np.float64))
            center_pose = Pose2(
                center_x_m,
                center_y_m,
                self.current_pose.yaw_rad,
                "map",
                center_z_m,
            )
            detail = self.sensor_state.planning_observation(center_pose)
            stamp_ns = (
                1_000_000_000
                + self._revision * 1_000_000
                + group_index
            )
            local_map = _grid_map(
                canvas=detail.canvas,
                frame_id="odom",
                elevation_m=detail.elevation_m,
                valid_mask=detail.valid_mask,
                physical_obstacle_ratio=detail.physical_obstacle_ratio,
                physical_obstacle_height_m=(
                    detail.physical_obstacle_height_m
                ),
                forbidden_ratio=detail.forbidden_ratio,
                observation_age_s=detail.observation_age_s,
                observation_quality=detail.observation_quality,
                observation_count=detail.observation_count,
                stamp_ns=stamp_ns,
            )
            request = self._base_request(planner_global_map, local_map)
            request.request_id = (
                f"formal-endpoint/{self.scene_id}/{self._revision}/"
                f"{group_index}"
            )
            point = bridge_api.PointGoal()
            point.position_m = _vec3(center_x_m, center_y_m, center_z_m)
            point.tolerance_m = 0.2
            request.goal.goal_id = f"endpoint/{group_index}"
            request.goal.target = point
            apply_goal_theta(
                request.goal,
                self.platform_type,
                self.current_pose.yaw_rad,
            )
            projection = self._bridge.project_traversability(request)
            raw_hard = np.asarray(
                getattr(projection, "hard_feasible", None)
            )
            cells = detail.canvas.geometry.cells
            if (
                raw_hard.dtype != np.dtype(np.uint8)
                or raw_hard.shape != (cells, cells)
                or not raw_hard.flags.c_contiguous
                or (raw_hard.size and (raw_hard > 1).any())
            ):
                raise RuntimeError(
                    "ground endpoint traversability geometry differs"
                )
            hard = np.ascontiguousarray(
                np.flipud(raw_hard).astype(np.bool_)
            )
            covered, feasible = ground_point_goal_feasibility(
                hard,
                canvas=detail.canvas,
                target_positions_m=group_targets,
                tolerance_m=0.2,
            )
            output[np.asarray(indices, dtype=np.intp)] = covered & feasible
        return np.ascontiguousarray(output)

    def _ground_detail_frontier_candidates(
        self,
        segments: list[list[tuple[int, int]]],
        world: ObservedWorld,
        *,
        planner_global_map: object,
    ) -> tuple[tuple[int, _RawFrontierCandidate], ...]:
        """Resolve three fixed coarse anchors into observed-only 0.2 m poses.

        The detail window and the C++ traversability projection are both bound
        to the current observation revision.  They deliberately receive no
        task coverability or scene-truth layer.
        """
        detail_factor = int(round(
            GLOBAL_GEOMETRY.resolution_m / self.sensor_state.resolution_m
        ))
        if detail_factor <= 0 or (
            detail_factor * self.sensor_state.resolution_m
            != GLOBAL_GEOMETRY.resolution_m
        ):
            raise RuntimeError("ground detail candidate resolution differs")
        anchors: list[tuple[int, int, tuple[int, int], float, float]] = []
        for segment_id, segment in enumerate(segments):
            for sample_rank, frontier_cell in enumerate(
                _sample_frontier_chain(segment)
            ):
                x_m, y_m = world.canvas.grid_center_world(*frontier_cell)
                anchors.append((
                    segment_id,
                    sample_rank,
                    frontier_cell,
                    x_m,
                    y_m,
                ))
        groups: dict[tuple[int, int], list[tuple[int, int, tuple[int, int], float, float]]] = {}
        # A 32 m bucket plus 4 m strip fits inside one 64 m observed window.
        for anchor in anchors:
            groups.setdefault(
                (int(math.floor(anchor[3] / 32.0)), int(math.floor(anchor[4] / 32.0))),
                [],
            ).append(anchor)
        output: list[tuple[int, _RawFrontierCandidate]] = []
        for group_index, anchors_in_group in enumerate(groups.values()):
            mean_x = float(np.mean([item[3] for item in anchors_in_group]))
            mean_y = float(np.mean([item[4] for item in anchors_in_group]))
            # Align the 64 m detail window to the 4 m global lattice.  The
            # selector can then derive its local coarse frontier neighbours
            # without any resampling or half-cell shift.
            left, bottom, right, top = world.canvas.bounds_m
            center_x = left + 32.0 + 4.0 * round((mean_x - left - 32.0) / 4.0)
            center_y = top - 32.0 - 4.0 * round((top - 32.0 - mean_y) / 4.0)
            # Alignment rounding can otherwise select the exterior canvas
            # boundary for a frontier in the outermost 4 m cell.
            center_x = min(max(center_x, left + 32.0), right - 32.0)
            center_y = min(max(center_y, bottom + 32.0), top - 32.0)
            center_row, center_column = world.canvas.world_to_grid(center_x, center_y)
            center_pose = Pose2(
                center_x,
                center_y,
                self.current_pose.yaw_rad,
                "map",
                float(world.elevation_m[center_row, center_column]),
            )
            detail = self.sensor_state.planning_observation(center_pose)
            local_map = _grid_map(
                canvas=detail.canvas,
                frame_id="odom",
                elevation_m=detail.elevation_m,
                valid_mask=detail.valid_mask,
                physical_obstacle_ratio=detail.physical_obstacle_ratio,
                physical_obstacle_height_m=detail.physical_obstacle_height_m,
                forbidden_ratio=detail.forbidden_ratio,
                observation_age_s=detail.observation_age_s,
                observation_quality=detail.observation_quality,
                observation_count=detail.observation_count,
                stamp_ns=(2_000_000_000 + self._revision * 1_000_000 + group_index),
            )
            request = self._base_request(planner_global_map, local_map)
            request.request_id = (
                f"formal-frontier-strip/{self.scene_id}/{self._revision}/"
                f"{group_index}"
            )
            point = bridge_api.PointGoal()
            point.position_m = _vec3(
                center_pose.x_m, center_pose.y_m, center_pose.elevation_m
            )
            point.tolerance_m = 0.2
            request.goal.goal_id = f"frontier-strip/{group_index}"
            request.goal.target = point
            apply_goal_theta(request.goal, self.platform_type, center_pose.yaw_rad)
            projection = self._bridge.project_traversability(request)
            cells = detail.canvas.geometry.cells
            hard = np.asarray(getattr(projection, "hard_feasible", None))
            clearance = np.asarray(getattr(projection, "clearance_m", None))
            if (
                hard.dtype != np.dtype(np.uint8)
                or clearance.dtype != np.dtype(np.float32)
                or hard.shape != (cells, cells)
                or clearance.shape != (cells, cells)
                or not hard.flags.c_contiguous
                or not clearance.flags.c_contiguous
                or (hard.size and (hard > 1).any())
                or not np.isfinite(clearance).all()
                or (clearance < 0.0).any()
            ):
                raise RuntimeError(
                    "ground detail candidate traversability geometry differs"
                )
            physical_safe = np.ascontiguousarray(np.flipud(hard).astype(np.bool_))
            clearance_m = np.ascontiguousarray(
                np.flipud(clearance).astype(np.float32)
            )
            local_coarse_cells = cells // detail_factor
            coarse_observed = np.zeros(
                (local_coarse_cells, local_coarse_cells), dtype=np.bool_
            )
            for local_row in range(local_coarse_cells):
                for local_column in range(local_coarse_cells):
                    x_m, y_m = detail.canvas.grid_center_world(
                        local_row * detail_factor + detail_factor // 2,
                        local_column * detail_factor + detail_factor // 2,
                    )
                    try:
                        world_row, world_column = world.canvas.world_to_grid(x_m, y_m)
                    except ValueError:
                        continue
                    coarse_observed[local_row, local_column] = world.observed_mask[
                        world_row, world_column
                    ]
            for segment_id, sample_rank, frontier_cell, x_m, y_m in anchors_in_group:
                try:
                    detail_row, detail_column = detail.canvas.world_to_grid(x_m, y_m)
                except ValueError:
                    continue
                selection = _select_narrow_frontier_strip_positions(
                    [(
                        detail_row // detail_factor,
                        detail_column // detail_factor,
                    )],
                    coarse_observed_mask=np.ascontiguousarray(coarse_observed),
                    observed_detail_mask=np.ascontiguousarray(detail.valid_mask),
                    physical_safe_detail_mask=physical_safe,
                    clearance_detail=clearance_m,
                    detail_cells_per_coarse=detail_factor,
                    # C++ hard feasibility already applies the platform body
                    # footprint and safety margins; this one-cell offset only
                    # prevents selecting the unknown frontier boundary itself.
                    minimum_standoff_detail_cells=1,
                    maximum_standoff_detail_cells=20,
                    lateral_half_width_detail_cells=1,
                )
                if not selection.pose_cells:
                    continue
                pose_row, pose_column = selection.pose_cells[0]
                target_x, target_y = detail.canvas.grid_center_world(
                    pose_row, pose_column
                )
                try:
                    pose_cell = world.canvas.world_to_grid(target_x, target_y)
                except ValueError:
                    continue
                output.append((
                    segment_id,
                    _RawFrontierCandidate(
                        sample_rank=sample_rank,
                        frontier_cell=frontier_cell,
                        pose_cell=pose_cell,
                        target_pose=Pose2(
                            target_x,
                            target_y,
                            elevation_m=float(detail.elevation_m[pose_row, pose_column]),
                        ),
                    ),
                ))
        return tuple(output)

    def build_policy_observation(
        self, observed, pose: Pose2
    ) -> PolicyBatch:
        if pose != self.current_pose:
            raise ValueError("policy observation pose differs from episode state")
        self._revision += 1
        snapshot = self._snapshot
        pending_failure = self._pending_planning_failure
        if (
            snapshot is not None
            and pending_failure is not None
            and pending_failure[2]
            == snapshot.candidate_universe.physical_snapshot_id
            and snapshot.planning_physical_snapshot_id
            == snapshot.candidate_universe.physical_snapshot_id
        ):
            return self._reselect_current_candidate_snapshot(pose)
        if self._defer_candidate_rebuild:
            snapshot = self._snapshot
            if snapshot is None:
                raise RuntimeError("ground continuation has no prior map snapshot")
            planner_global_map, local_map = self._rolling_planner_and_local_map()
            return self._build_ground_continuation_observation(
                pose,
                global_map=snapshot.global_map,
                planner_global_map=planner_global_map,
                local_map=local_map,
            )
        candidate_evidence_sha256 = _candidate_boundary_evidence_sha256(
            self.sensor_state
        )
        global_map, planner_global_map, local_map, _ = self._observed_maps()
        local = self.sensor_state.local_observation(pose)
        world = observed.to_observed_world(local=local)
        projection_request = self._base_request(
            (
                global_map
                if self.platform_type == "HOPPER"
                else planner_global_map
            ),
            local_map,
        )
        projection_request.request_id = (
            f"formal-projection/{self.scene_id}/{self._revision}"
        )
        point = bridge_api.PointGoal()
        point.position_m = _vec3(pose.x_m, pose.y_m, pose.elevation_m)
        point.tolerance_m = 0.0 if self.platform_type == "HOPPER" else 0.2
        projection_request.goal.goal_id = "projection"
        projection_request.goal.target = point
        apply_goal_theta(projection_request.goal, self.platform_type, pose.yaw_rad)
        local_cpp = self._bridge.project_traversability(projection_request)
        ground_start_resample_reason = self._qualify_ground_start(
            planner_global_map=planner_global_map,
            local_map=local_map,
            pose=pose,
        )
        local_known = np.ascontiguousarray(np.flipud(local_cpp.known).astype(bool))
        local_hard = np.ascontiguousarray(
            np.flipud(local_cpp.hard_feasible).astype(bool)
        )
        center = (local_known & local_hard)[144:176, 144:176].astype(np.float32)
        observed_safe = np.ascontiguousarray(
            world.observed_mask
            & (world.physical_obstacle_layer.values == 0.0)
        )
        physical_capability_sha256 = _physical_capability_content_sha256(
            self.capability
        )
        projection = PlatformProjection(
            canvas=world.canvas,
            traversable_ratio=observed_safe.astype(np.float32),
            local_traversable_ratio=center,
            clearance_margin_norm=self._static_clearance,
            source=f"cpp_v3/{physical_capability_sha256}",
        )
        candidate_builder = (
            self._candidate_builder
            if self._detail_candidate_gain_enabled
            else self._legacy_candidate_builder
        )
        self._current_candidate_gain_resolution_m = (
            self.sensor_state.resolution_m
            if self._detail_candidate_gain_enabled
            else GLOBAL_GEOMETRY.resolution_m
        )
        safe_cells = np.ascontiguousarray(
            np.column_stack(np.nonzero(observed_safe)), dtype=np.int32
        ).reshape((-1, 2))
        physical_reachability = PlatformCandidateReachability(
            platform_type=self.platform_type,
            canvas=world.canvas,
            pose_map=pose,
            observed_elevation_m=world.elevation_m,
            bridge=self._bridge,
            request=projection_request,
            maximum_edge_distance_m=(
                None if self.platform_type == "HOPPER" else 30.0
            ),
            local_traversability_projection=(
                None if self.platform_type == "HOPPER" else local_cpp
            ),
            planner_global_canvas=(
                None
                if self.platform_type == "HOPPER"
                else _planner_global_canvas(world.canvas)
            ),
        ).project_physical(safe_cells)
        backtrack_pose = (
            self._navigation_stack[-2]
            if len(self._navigation_stack) >= 2
            else None
        )
        candidate_universe = candidate_builder.build_physical_universe(
            world,
            self.mission,
            pose,
            projection,
            physical_reachability=physical_reachability,
            platform_type=self.platform_type,
            platform_id=self.capability.platform_id,
            capability_content_sha256=physical_capability_sha256,
            mission_revision=1,
            evidence_generation=self.sensor_state.evidence_generation,
            physical_evidence_sha256=candidate_evidence_sha256,
            physical_reachability_algorithm_id=(
                physical_reachability.physical_reachability_algorithm_id
            ),
            goal_tolerance_mm=(0 if self.platform_type == "HOPPER" else 200),
            excluded_cells=(
                self._visited_candidate_cells
                if self.platform_type == "HOPPER"
                and self._visited_candidate_filter_enabled
                else ()
            ),
            ground_endpoint_feasibility=(
                None
                if self.platform_type == "HOPPER"
                else lambda positions: self._ground_endpoint_feasibility(
                    positions,
                    planner_global_map=planner_global_map,
                )
            ),
            ground_detail_candidate_provider=(
                None
                if self.platform_type == "HOPPER"
                else lambda segments, candidate_world: (
                    self._ground_detail_frontier_candidates(
                        segments,
                        candidate_world,
                        planner_global_map=planner_global_map,
                    )
                )
            ),
        )
        failure_snapshot_id, failed_candidate_ids = (
            _resolve_rebuilt_planning_failure(
                candidate_universe=candidate_universe,
                failure_snapshot_id=self._planner_failure_snapshot_id,
                planner_failed_candidate_ids=self._planner_failed_candidate_ids,
                pending_failure=self._pending_planning_failure,
            )
        )
        candidate_result = candidate_builder.select_available(
            candidate_universe,
            canvas_id=world.canvas.identity,
            failure_snapshot_id=failure_snapshot_id,
            planner_failed_candidate_ids=failed_candidate_ids,
            excluded_cells=(
                self._visited_candidate_cells
                if self._visited_candidate_filter_enabled
                else ()
            ),
            backtrack_pose=backtrack_pose,
        )
        self._planner_failure_snapshot_id = failure_snapshot_id
        self._planner_failed_candidate_ids = failed_candidate_ids
        self._pending_planning_failure = None
        candidates = candidate_result.batch
        arrays = self._observation_builder.build(
            world,
            self.mission,
            pose,
            projection,
            candidates,
            self.platform_type,
        )
        self._snapshot = _MapSnapshot(
            self._revision,
            candidates,
            global_map,
            planner_global_map,
            local_map,
            world,
            projection,
            physical_reachability,
            candidate_result.universe,
            candidate_universe.physical_snapshot_id,
            ground_start_resample_reason,
        )
        return PolicyBatch(
            **{
                name: torch.from_numpy(value.copy())
                for name, value in arrays.items()
            }
        )

    def _reselect_current_candidate_snapshot(self, pose: Pose2) -> PolicyBatch:
        """Promote reserve entries without rerunning global search or gain."""
        snapshot = self._snapshot
        pending = self._pending_planning_failure
        if snapshot is None or pending is None:
            raise RuntimeError("candidate reselection requires a pending failure")
        candidate_id, disposition, pending_snapshot_id, _ = pending
        universe = snapshot.candidate_universe
        if (
            pending_snapshot_id != universe.physical_snapshot_id
            or snapshot.planning_physical_snapshot_id
            != universe.physical_snapshot_id
        ):
            raise ValueError("candidate reselection snapshot differs")
        failed_candidate_ids = set(self._planner_failed_candidate_ids)
        if self._planner_failure_snapshot_id != universe.physical_snapshot_id:
            failed_candidate_ids.clear()
        if disposition == (
            bridge_api.CandidateDisposition.SUPPRESS_FOR_CURRENT_PHYSICAL_SNAPSHOT
        ):
            failed_candidate_ids.add(candidate_id)
        backtrack_pose = (
            self._navigation_stack[-2]
            if len(self._navigation_stack) >= 2
            else None
        )
        candidate_result = self._candidate_builder.select_available(
            universe,
            canvas_id=snapshot.candidates.canvas_id,
            failure_snapshot_id=universe.physical_snapshot_id,
            planner_failed_candidate_ids=failed_candidate_ids,
            excluded_cells=(
                self._visited_candidate_cells
                if self._visited_candidate_filter_enabled
                else ()
            ),
            backtrack_pose=backtrack_pose,
        )
        candidates = candidate_result.batch
        arrays = self._observation_builder.build(
            snapshot.world,
            self.mission,
            pose,
            snapshot.projection,
            candidates,
            self.platform_type,
        )
        self._planner_failure_snapshot_id = universe.physical_snapshot_id
        self._planner_failed_candidate_ids = failed_candidate_ids
        self._pending_planning_failure = None
        self._snapshot = _MapSnapshot(
            self._revision,
            candidates,
            snapshot.global_map,
            snapshot.planner_global_map,
            snapshot.local_map,
            snapshot.world,
            snapshot.projection,
            snapshot.physical_reachability,
            candidate_result.universe,
            snapshot.planning_physical_snapshot_id,
            snapshot.ground_start_resample_reason,
        )
        return PolicyBatch(
            **{
                name: torch.from_numpy(value.copy())
                for name, value in arrays.items()
            }
        )

    def _build_ground_continuation_observation(
        self,
        pose: Pose2,
        *,
        global_map: object,
        planner_global_map: object,
        local_map: object,
    ) -> PolicyBatch:
        """Refresh planning maps without rebuilding unused policy candidates."""
        snapshot = self._snapshot
        if snapshot is None:
            raise RuntimeError("ground continuation has no prior map snapshot")
        # The continuation pose is the certified endpoint of the reference that
        # just executed.  Replanning from that pose back to itself duplicates
        # the next rolling request and turns a valid one-pose connector into
        # LOCAL_DETAIL_ROUTE_RESULT_INVALID.  Initial decision boundaries still
        # use _qualify_ground_start; the next real rolling plan remains the
        # fail-closed authority for this endpoint.
        ground_start_resample_reason = None
        self._snapshot = _MapSnapshot(
            self._revision,
            snapshot.candidates,
            global_map,
            planner_global_map,
            local_map,
            snapshot.world,
            snapshot.projection,
            snapshot.physical_reachability,
            snapshot.candidate_universe,
            CandidateBuilderV2._physical_snapshot_id(
                platform_type=self.platform_type,
                platform_id=self.capability.platform_id,
                capability_content_sha256=(
                    _physical_capability_content_sha256(self.capability)
                ),
                mission_revision=1,
                pose_map=pose,
                evidence_generation=self.sensor_state.evidence_generation,
                physical_evidence_sha256=(
                    self.sensor_state.physical_evidence_identity_sha256()
                ),
            ),
            ground_start_resample_reason,
        )
        previous = self.controller.current_observation
        pose_features = previous.pose_features.clone()
        canvas = self.loaded.scene.base_canvas
        pose_features[0, 0] = (
            pose.x_m - canvas.bounds_m[0]
        ) / canvas.geometry.size_m
        pose_features[0, 1] = (
            canvas.bounds_m[3] - pose.y_m
        ) / canvas.geometry.size_m
        pose_features[0, 2] = math.sin(pose.yaw_rad)
        pose_features[0, 3] = math.cos(pose.yaw_rad)
        return PolicyBatch(
            prior_channels=previous.prior_channels.clone(),
            coverage_summary=previous.coverage_summary.clone(),
            local_crop=previous.local_crop.clone(),
            frontier_features=previous.frontier_features.clone(),
            pose_features=pose_features,
            candidate_mask=previous.candidate_mask.clone(),
            platform_context=previous.platform_context.clone(),
        )

    def _qualify_ground_start(
        self,
        *,
        planner_global_map: object,
        local_map: object,
        pose: Pose2,
    ) -> str | None:
        if self.platform_type == "HOPPER":
            return None
        request_id = (
            f"formal-start-qualification/{self.scene_id}/{self._revision}"
        )
        for attempt in range(2):
            request = self._base_request(planner_global_map, local_map)
            request.request_id = (
                request_id if attempt == 0 else f"{request_id}/retry-{attempt}"
            )
            goal = bridge_api.PointGoal()
            goal.position_m = _vec3(
                pose.x_m,
                pose.y_m,
                pose.elevation_m,
            )
            goal.tolerance_m = 0.0
            request.goal.goal_id = "current-pose-start-qualification"
            request.goal.target = goal
            apply_goal_theta(request.goal, self.platform_type, pose.yaw_rad)
            output = self._bridge.plan(request)
            try:
                return _ground_start_resample_reason(output, self.platform_type)
            except _GroundStartQualificationProbeError:
                if attempt == 1:
                    raise
        raise AssertionError("ground start qualification retry loop is incomplete")

    def ground_start_qualification_reason(self) -> str | None:
        snapshot = self._snapshot
        if snapshot is None:
            raise RuntimeError("ground start qualification has no map snapshot")
        return snapshot.ground_start_resample_reason

    def build_request(
        self, action: PolicyAction, expected_identity: ObservationIdentity
    ) -> PreparedPlanRequest:
        if not isinstance(expected_identity, ObservationIdentity):
            raise ValueError("formal request requires an observation identity")
        snapshot = self._snapshot
        if snapshot is None:
            raise RuntimeError("formal request has no observation snapshot")
        if not 0 <= action.frontier_index < snapshot.candidates.count:
            raise ValueError("formal action candidate is invalid")
        canvas = self.loaded.scene.base_canvas
        target_position = snapshot.candidates.target_positions_m[
            action.frontier_index
        ]
        target_x = float(target_position[0])
        target_y = float(target_position[1])
        canvas.world_to_grid(target_x, target_y)
        target_z = float(target_position[2])
        target_yaw = (
            float(snapshot.candidates.target_yaw_rad[action.frontier_index])
            if self.platform_type == "HOPPER"
            else float(action.theta_rad)
        )
        request = self._base_request(
            snapshot.planner_global_map,
            snapshot.local_map,
        )
        # The observation boundary owns the authoritative state clock.  Cache
        # map objects are materialized while the observation is being built,
        # before that controller attaches its identity, so bind every request
        # timestamp atomically to the selected observation here.
        state_time_ns = expected_identity.state_time_ns
        request.state_time.nanoseconds_since_epoch = state_time_ns
        request.world.global_map.stamp.nanoseconds_since_epoch = state_time_ns
        request.world.local_map.stamp.nanoseconds_since_epoch = state_time_ns
        request.world.map_from_odom.stamp.nanoseconds_since_epoch = state_time_ns
        request.request_id = (
            f"formal/{self.scene_id}/{expected_identity.map_snapshot_id}/"
            f"{action.frontier_index}"
        )
        goal = bridge_api.PointGoal()
        goal.position_m = _vec3(target_x, target_y, target_z)
        goal.tolerance_m = 0.0 if self.platform_type == "HOPPER" else 0.2
        request.goal.goal_id = f"frontier/{action.frontier_index}"
        request.goal.target = goal
        apply_goal_theta(request.goal, self.platform_type, target_yaw)
        return PreparedPlanRequest(
            request=request,
            identity=expected_identity,
            candidate_id=str(
                snapshot.candidates.candidate_ids[action.frontier_index]
            ),
            physical_snapshot_id=(
                snapshot.planning_physical_snapshot_id
            ),
        )

    def begin_ground_option(
        self, action: PolicyAction, expected_identity: ObservationIdentity
    ) -> PreparedPlanRequest:
        if self.platform_type not in {"WHEELED", "LEGGED"}:
            raise ValueError("ground option requires a ground platform")
        if self._active_ground_option is not None:
            raise ValueError("formal episode already has an active ground option")
        prepared = self.build_request(action, expected_identity)
        target = prepared.request.goal.target
        if not isinstance(target, bridge_api.PointGoal):
            raise ValueError("formal ground option requires a point goal")
        position = target.position_m
        target_yaw = float(action.theta_rad)
        values = (
            position.x,
            position.y,
            position.z,
            target.tolerance_m,
            target_yaw,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("formal ground option contains non-finite values")
        goal_id = (
            f"ground-option/{expected_identity.map_snapshot_id}/"
            f"{action.frontier_index}"
        )
        self._active_ground_option = _GroundOption(
            target_x_m=float(position.x),
            target_y_m=float(position.y),
            target_z_m=float(position.z),
            tolerance_m=float(target.tolerance_m),
            theta_rad=target_yaw,
            goal_id=goal_id,
            candidate_id=prepared.candidate_id,
        )
        prepared.request.goal.goal_id = goal_id
        return prepared

    def continue_ground_option(
        self, expected_identity: ObservationIdentity
    ) -> PreparedPlanRequest:
        if not isinstance(expected_identity, ObservationIdentity):
            raise ValueError("formal ground continuation requires an identity")
        option = self._active_ground_option
        if option is None:
            raise RuntimeError("formal episode has no active ground option")
        snapshot = self._snapshot
        if snapshot is None:
            raise RuntimeError("formal ground continuation has no map snapshot")
        request = self._base_request(
            snapshot.planner_global_map,
            snapshot.local_map,
        )
        state_time_ns = expected_identity.state_time_ns
        request.state_time.nanoseconds_since_epoch = state_time_ns
        request.world.global_map.stamp.nanoseconds_since_epoch = state_time_ns
        request.world.local_map.stamp.nanoseconds_since_epoch = state_time_ns
        request.world.map_from_odom.stamp.nanoseconds_since_epoch = state_time_ns
        request.request_id = (
            f"formal/{self.scene_id}/{expected_identity.map_snapshot_id}/"
            f"{option.goal_id}"
        )
        goal = bridge_api.PointGoal()
        goal.position_m = _vec3(
            option.target_x_m,
            option.target_y_m,
            option.target_z_m,
        )
        goal.tolerance_m = option.tolerance_m
        request.goal.goal_id = option.goal_id
        request.goal.target = goal
        apply_goal_theta(request.goal, self.platform_type, option.theta_rad)
        return PreparedPlanRequest(
            request=request,
            identity=expected_identity,
            candidate_id=option.candidate_id,
            physical_snapshot_id=(
                snapshot.planning_physical_snapshot_id
            ),
        )

    def ground_option_distance_m(self) -> float:
        option = self._active_ground_option
        if option is None:
            raise RuntimeError("formal episode has no active ground option")
        distance_m = math.hypot(
            self.current_pose.x_m - option.target_x_m,
            self.current_pose.y_m - option.target_y_m,
        )
        if not math.isfinite(distance_m):
            raise ValueError("formal ground option distance is non-finite")
        return distance_m

    def clear_ground_option(self) -> None:
        self._active_ground_option = None

    def refresh_after_planning_failure(
        self,
        candidate_id: str,
        disposition: bridge_api.CandidateDisposition,
        physical_snapshot_id: str,
    ) -> BoundaryObservationResult:
        """Rebuild availability after a planner response without new evidence."""
        if disposition not in {
            bridge_api.CandidateDisposition.KEEP,
            bridge_api.CandidateDisposition.SUPPRESS_FOR_CURRENT_PHYSICAL_SNAPSHOT,
        }:
            raise ValueError("planner candidate disposition is invalid")
        snapshot = self._snapshot
        if snapshot is None:
            raise RuntimeError("planning failure refresh has no map snapshot")
        if physical_snapshot_id != snapshot.planning_physical_snapshot_id:
            raise ValueError("planning failure physical snapshot is stale")
        universe_ids = {
            candidate.candidate_id
            for candidate in snapshot.candidate_universe.candidates
        }
        if candidate_id not in universe_ids:
            raise ValueError(
                "planning failure candidate is outside the physical universe"
            )
        option = self._active_ground_option
        allow_locked_target_exit = False
        if option is not None:
            if self.platform_type not in {"WHEELED", "LEGGED"}:
                raise ValueError("active ground option has a non-ground platform")
            if candidate_id != option.candidate_id:
                raise ValueError(
                    "planning failure candidate differs from the locked ground target"
                )
            # Candidate rebuilding is deliberately deferred while a locked
            # ground option rolls across multiple sensor boundaries.  The
            # final rebuild may legitimately remove that stable target after
            # the accumulated evidence makes it zero-gain or unreachable.
            allow_locked_target_exit = self._defer_candidate_rebuild
        self._active_ground_option = None
        self._defer_candidate_rebuild = False
        self._pending_planning_failure = (
            candidate_id,
            disposition,
            physical_snapshot_id,
            allow_locked_target_exit,
        )
        execution_state = (
            self.controller.current_observation.observation_identities[0]
            .execution_state
        )
        rebuilt = self.controller.rebuild_without_sensor_update(
            pose_map=self.current_pose,
            execution_state=execution_state,
        )
        event_kind = (
            "PLANNING_FAILURE_SUPPRESS"
            if (
                disposition
                == bridge_api.CandidateDisposition.SUPPRESS_FOR_CURRENT_PHYSICAL_SNAPSHOT
                and candidate_id in self._planner_failed_candidate_ids
            )
            else "PLANNING_FAILURE_REBUILD"
        )
        self._replay_event_kinds.append(event_kind)
        return rebuilt

    def execute_reference(self, reference: object) -> ReferenceExecutionResult:
        if not isinstance(reference, bridge_api.MotionReference):
            return self._execution_failure()
        if reference.platform_type != self.platform_type or not reference.plan_id:
            return self._execution_failure()
        if self.platform_type == "HOPPER":
            return self._commit_hop(reference)
        data = reference.data
        if not isinstance(data, bridge_api.TrajectoryReference):
            return self._execution_failure()
        points = tuple(data.points)
        if len(points) < 2:
            return self._execution_failure()
        start = points[0].pose.position_m
        if math.dist(
            (start.x, start.y),
            (self.current_pose.x_m, self.current_pose.y_m),
        ) > 0.5:
            return self._execution_failure()
        try:
            evidence = self._ground_sensor_evidence(points)
        except (OverflowError, TypeError, ValueError):
            return self._execution_failure()
        end = points[-1].pose
        if self.platform_type == "LEGGED":
            self._current_legged_body_z_m = float(end.position_m.z)
        self.current_pose = evidence.pose_map
        elapsed_s = evidence.elapsed_s
        self._record_reveal(evidence, "DECISION_BOUNDARY")
        return ReferenceExecutionResult(
            next_observation=self.controller.current_observation,
            mission_observed_delta=0.0,
            priority_observed_delta=0.0,
            normalized_execution_cost_contribution=0.0,
            normalized_execution_time_contribution=(
                elapsed_s / _FORMAL_MACRO_STEP_TIME_SCALE_S
            ),
            executed_without_new_coverage=False,
            success_first_crossing=False,
            episode_ended_without_success=False,
            hard_safety_violation=False,
            terminated=False,
            execution_state="DECISION_BOUNDARY",
            execution_events=ExecutionEvents(
                reference_samples_consumed=len(points),
                selected_action_observed_safe=True,
            ),
            sensor_boundary_evidence=evidence,
        )

    def _ground_sensor_evidence(
        self, points: tuple[object, ...]
    ) -> SensorBoundaryEvidence:
        timestamps_s = tuple(
            float(point.time_from_start.total_seconds()) for point in points
        )
        positions = tuple(
            (
                float(point.pose.position_m.x),
                float(point.pose.position_m.y),
                float(point.pose.position_m.z),
            )
            for point in points
        )
        yaws = tuple(float(_yaw_from_pose(point.pose)) for point in points)
        if (
            not all(math.isfinite(value) for value in timestamps_s)
            or not all(
                math.isfinite(value)
                for position in positions
                for value in position
            )
            or not all(math.isfinite(value) for value in yaws)
            or timestamps_s[0] < 0.0
            or any(
                right < left
                for left, right in zip(timestamps_s, timestamps_s[1:])
            )
        ):
            raise ValueError("ground reference path is invalid")
        timestamps_ns = tuple(
            int(round(value * 1_000_000_000.0)) for value in timestamps_s
        )
        if timestamps_ns[-1] > (1 << 63) - 1:
            raise OverflowError("ground reference elapsed time is out of range")

        def interpolated_pose(index: int, alpha: float) -> Pose2:
            left = positions[index]
            right = positions[index + 1]
            if alpha >= 1.0:
                x_m, y_m, body_or_terrain_z_m = right
                yaw_rad = yaws[index + 1]
            else:
                x_m = left[0] + alpha * (right[0] - left[0])
                y_m = left[1] + alpha * (right[1] - left[1])
                body_or_terrain_z_m = left[2] + alpha * (
                    right[2] - left[2]
                )
                yaw_delta = math.atan2(
                    math.sin(yaws[index + 1] - yaws[index]),
                    math.cos(yaws[index + 1] - yaws[index]),
                )
                yaw_rad = yaws[index] + alpha * yaw_delta
                yaw_rad = math.atan2(math.sin(yaw_rad), math.cos(yaw_rad))
            terrain_z_m = (
                self._terrain_elevation_m(x_m, y_m)
                if self.platform_type == "LEGGED"
                else body_or_terrain_z_m
            )
            return Pose2(x_m, y_m, yaw_rad, "map", terrain_z_m)

        samples: list[SensorPathSample] = []
        previous_time_ns = 0
        first_distance_m = math.dist(
            (self.current_pose.x_m, self.current_pose.y_m),
            positions[0][:2],
        )
        if first_distance_m > 1.0e-12:
            first_pose = Pose2(
                positions[0][0],
                positions[0][1],
                yaws[0],
                "map",
                (
                    self._terrain_elevation_m(*positions[0][:2])
                    if self.platform_type == "LEGGED"
                    else positions[0][2]
                ),
            )
            samples.append(
                SensorPathSample(
                    first_pose,
                    timestamps_ns[0] / 1_000_000_000.0,
                )
            )
            previous_time_ns = timestamps_ns[0]
        for index, (left, right) in enumerate(
            zip(positions, positions[1:])
        ):
            distance_m = math.dist(left[:2], right[:2])
            steps = max(
                1,
                int(
                    math.ceil(
                        distance_m / _MAX_GROUND_SENSOR_SAMPLE_SPACING_M
                    )
                ),
            )
            segment_ns = timestamps_ns[index + 1] - timestamps_ns[index]
            for step in range(1, steps + 1):
                alpha = step / steps
                sample_time_ns = timestamps_ns[index] + int(
                    round(alpha * segment_ns)
                )
                if sample_time_ns < previous_time_ns:
                    raise ValueError("ground reference sample time regressed")
                samples.append(
                    SensorPathSample(
                        interpolated_pose(index, alpha),
                        (sample_time_ns - previous_time_ns)
                        / 1_000_000_000.0,
                    )
                )
                previous_time_ns = sample_time_ns
        if not samples or previous_time_ns != timestamps_ns[-1]:
            raise ValueError("ground reference endpoint sample is missing")
        return SensorBoundaryEvidence(
            samples[-1].pose_map,
            timestamps_ns[-1] / 1_000_000_000.0,
            path_samples=tuple(samples),
        )

    def _terrain_elevation_m(self, x_m: float, y_m: float) -> float:
        row, column = self.sensor_state.tile_provider.world_to_detail(x_m, y_m)
        cells = self.sensor_state.tile_provider.tile_geometry.cells
        tile_row, local_row = divmod(row, cells)
        tile_column, local_column = divmod(column, cells)
        tile = self.sensor_state.tile_provider.tile(tile_row, tile_column)
        if not tile.valid_mask[local_row, local_column]:
            raise ValueError("certified endpoint has no terrain elevation")
        return float(tile.elevation_m[local_row, local_column])

    def _commit_hop(self, reference: object) -> ReferenceExecutionResult:
        data = reference.data
        if not isinstance(data, bridge_api.HopReference):
            return self._execution_failure(hopper=True)
        segments = tuple(data.segments)
        if len(segments) != 1:
            return self._execution_failure(hopper=True)
        segment = segments[0]
        landing = segment.nominal_landing_point_m
        values = (
            landing.x,
            landing.y,
            landing.z,
            segment.available_delta_v_mps,
            segment.required_delta_v_mps,
            segment.flight_time.total_seconds(),
        )
        if not all(math.isfinite(float(value)) for value in values) or values[-1] <= 0.0:
            return self._execution_failure(hopper=True)
        self._pending_hop_landing = Pose2(
            landing.x, landing.y, 0.0, "map", landing.z
        )
        self._pending_hop_elapsed_s = float(
            segment.flight_time.total_seconds()
        )
        start = self.current_pose
        duration = self._pending_hop_elapsed_s
        velocity = segment.launch_velocity_mps
        gravity = tuple(
            2.0 * (end - begin - speed * duration) / (duration * duration)
            for begin, end, speed in (
                (start.x_m, landing.x, velocity.x),
                (start.y_m, landing.y, velocity.y),
                (start.elevation_m, landing.z, velocity.z),
            )
        )
        midpoint_time = 0.5 * duration
        self._pending_hop_midpoint = Pose2(
            start.x_m
            + velocity.x * midpoint_time
            + 0.5 * gravity[0] * midpoint_time * midpoint_time,
            start.y_m
            + velocity.y * midpoint_time
            + 0.5 * gravity[1] * midpoint_time * midpoint_time,
            0.0,
            "map",
            start.elevation_m
            + velocity.z * midpoint_time
            + 0.5 * gravity[2] * midpoint_time * midpoint_time,
        )
        self.last_hop_available_delta_v_mps = float(
            segment.available_delta_v_mps
        )
        self._hopper_feedback_phase = 0
        return ReferenceExecutionResult(
            next_observation=self.controller.current_observation,
            mission_observed_delta=0.0,
            priority_observed_delta=0.0,
            normalized_execution_cost_contribution=0.0,
            normalized_execution_time_contribution=(
                self._pending_hop_elapsed_s
                / _FORMAL_MACRO_STEP_TIME_SCALE_S
            ),
            executed_without_new_coverage=True,
            success_first_crossing=False,
            episode_ended_without_success=False,
            hard_safety_violation=False,
            terminated=False,
            execution_state="JUMP_COMMITTED",
            execution_events=ExecutionEvents(
                reference_samples_consumed=1,
                selected_action_observed_safe=True,
                hopper_commitment_states=("JUMP_COMMITTED",),
            ),
        )

    def committed_hop_feedback(self) -> CommittedHopExecutionFeedback:
        landing = self._pending_hop_landing
        if landing is None:
            return self._hopper_feedback_failure()
        if self._hopper_feedback_phase == 0:
            self._hopper_feedback_phase = 1
            midpoint = self._pending_hop_midpoint
            if midpoint is None:
                return self._hopper_feedback_failure()
            return CommittedHopExecutionFeedback(
                execution_state="IN_FLIGHT",
                next_observation=self.controller.current_observation,
                mission_observed_delta=0.0,
                priority_observed_delta=0.0,
                normalized_execution_cost_contribution=0.0,
                normalized_execution_time_contribution=0.0,
                executed_without_new_coverage=True,
                success_first_crossing=False,
                episode_ended_without_success=False,
                hard_safety_violation=False,
                terminated=False,
                execution_events=ExecutionEvents(
                    hopper_commitment_states=("IN_FLIGHT",)
                ),
                trajectory_points=(
                    HopperTrajectoryPoint(
                        midpoint, 0.5 * self._pending_hop_elapsed_s
                    ),
                ),
            )
        self._pending_hop_landing = None
        midpoint = self._pending_hop_midpoint
        self._pending_hop_midpoint = None
        elapsed_s = self._pending_hop_elapsed_s
        self._pending_hop_elapsed_s = 0.0
        self._hopper_feedback_phase = 0
        self.current_pose = landing
        evidence = SensorBoundaryEvidence(landing, elapsed_s)
        if midpoint is None:
            return self._hopper_feedback_failure()
        self._record_reveal(
            SensorBoundaryEvidence(
                landing,
                elapsed_s,
                path_samples=(
                    SensorPathSample(midpoint, 0.5 * elapsed_s),
                    SensorPathSample(landing, 0.5 * elapsed_s),
                ),
            ),
            "LANDED_HOLD",
        )
        return CommittedHopExecutionFeedback(
            execution_state="LANDED_HOLD",
            next_observation=self.controller.current_observation,
            mission_observed_delta=0.0,
            priority_observed_delta=0.0,
            normalized_execution_cost_contribution=0.0,
            normalized_execution_time_contribution=0.0,
            executed_without_new_coverage=False,
            success_first_crossing=False,
            episode_ended_without_success=False,
            hard_safety_violation=False,
            terminated=False,
            execution_events=ExecutionEvents(
                hopper_commitment_states=("LANDED_HOLD",)
            ),
            sensor_boundary_evidence=evidence,
            trajectory_points=(HopperTrajectoryPoint(landing, elapsed_s),),
        )

    def _execution_failure(self, *, hopper: bool = False) -> ReferenceExecutionResult:
        return ReferenceExecutionResult(
            next_observation=self.controller.current_observation,
            mission_observed_delta=0.0,
            priority_observed_delta=0.0,
            normalized_execution_cost_contribution=0.0,
            normalized_execution_time_contribution=0.0,
            executed_without_new_coverage=True,
            success_first_crossing=False,
            episode_ended_without_success=True,
            hard_safety_violation=True,
            terminated=True,
            execution_state="GROUND_HOLD" if hopper else "DECISION_BOUNDARY",
            execution_events=ExecutionEvents(
                safety_violation_count=1,
                hopper_commitment_violation_count=1 if hopper else 0,
                execution_failure_count=1,
            ),
        )

    def _hopper_feedback_failure(self) -> CommittedHopExecutionFeedback:
        return CommittedHopExecutionFeedback(
            execution_state="LANDED_HOLD",
            next_observation=self.controller.current_observation,
            mission_observed_delta=0.0,
            priority_observed_delta=0.0,
            normalized_execution_cost_contribution=0.0,
            normalized_execution_time_contribution=0.0,
            executed_without_new_coverage=True,
            success_first_crossing=False,
            episode_ended_without_success=True,
            hard_safety_violation=True,
            terminated=True,
            execution_events=ExecutionEvents(
                hopper_commitment_violation_count=1,
                execution_failure_count=1,
            ),
        )


@dataclass(frozen=True, slots=True)
class FormalWorkerBuilder:
    cache_manifest_path: str
    split: str
    allow_preflight: bool
    scenario_schedule_id: str
    task_area: TaskAreaConfig
    platform_scenario_schedule_ids: Mapping[str, str] | None = None
    paired_evaluation: bool = False
    task_cache_client: object | None = None
    source_commit: str | None = None
    task_halo: object | None = None
    sensor_closed_loop: ClassVar[bool] = True

    def task_inventory_count(self, platform_type: str) -> int:
        if platform_type not in _PLATFORMS:
            raise ValueError("formal task inventory platform is invalid")
        cache = load_formal_cache(
            Path(self.cache_manifest_path),
            require_full=not self.allow_preflight,
        )
        if self.paired_evaluation:
            common_ids = set(
                cache.manifest["exact_common_evaluation"]["splits"][
                    self.split
                ]["scene_ids"]
            )
            count = sum(
                entry["scene_id"] in common_ids
                for entry in cache.manifest["scenes"]
            )
        else:
            count = sum(
                entry["split"] == self.split
                and entry["platform_starts"][platform_type]["qualified"]
                for entry in cache.manifest["scenes"]
            )
        if type(count) is not int or count <= 0:
            raise ValueError("formal task inventory is empty")
        return count

    def __call__(
        self,
        worker_index: int,
        platform_type: str,
        capability: FrozenPlatformCapability,
        scenario_identity: ScenarioIdentity,
    ) -> FormalEnvironmentWorker:
        loaded = self._load_scheduled_scene(
            worker_index,
            scenario_identity,
            capability,
        )
        if loaded.task_geometry is None:
            raise ValueError("formal task geometry is unavailable")
        start_cell = loaded.task_geometry.local_start_cell
        episode = FormalEpisode(
            worker_index=worker_index,
            platform_type=platform_type,
            capability=capability,
            scenario_identity=scenario_identity,
            loaded=loaded,
            start_cell=start_cell,
        )
        return self._make_worker(episode)

    def restore(
        self,
        worker_index: int,
        platform_type: str,
        capability: FrozenPlatformCapability,
        scenario_identity: ScenarioIdentity,
        state: object,
    ) -> FormalEnvironmentWorker:
        parsed = FormalWorkerState.from_dict(state)
        if (
            parsed.worker_index != worker_index
            or parsed.platform_type != platform_type
            or parsed.episode_cursor != scenario_identity.episode_cursor
            or parsed.platform_worker_index
            != scenario_identity.platform_worker_index
            or parsed.platform_worker_count
            != scenario_identity.platform_worker_count
            or parsed.scenario_schedule_id
            != scenario_identity.scenario_schedule_id
        ):
            raise ValueError("formal restore worker identity differs")
        loaded = self._restore_task_scene(
            scenario_identity=scenario_identity,
            capability=capability,
            state=parsed,
        )
        episode = FormalEpisode(
            worker_index=worker_index,
            platform_type=platform_type,
            capability=capability,
            scenario_identity=scenario_identity,
            loaded=loaded,
            start_cell=parsed.start_cell,
        )
        episode.replay_state(parsed)
        worker = self._make_worker(episode)
        worker.environment.restore_stable_state(
            execution_state=parsed.execution_state,
            cumulative_executed_path_m=(
                parsed.cumulative_executed_path_m
            ),
        )
        return worker

    def _load_scheduled_scene(
        self,
        worker_index: int,
        scenario_identity: ScenarioIdentity,
        capability: FrozenPlatformCapability,
    ) -> _LoadedScene:
        cache = load_formal_cache(
            Path(self.cache_manifest_path), require_full=not self.allow_preflight
        )
        if scenario_identity.scenario_schedule_id != self.scenario_schedule_id:
            raise ValueError("formal worker scenario schedule identity differs")
        if self.paired_evaluation:
            common_ids = set(
                cache.manifest["exact_common_evaluation"]["splits"][self.split][
                    "scene_ids"
                ]
            )
            eligible_entries = [
                entry
                for entry in cache.manifest["scenes"]
                if entry["scene_id"] in common_ids
            ]
            schedule_id = str(
                cache.manifest["exact_common_evaluation"]["splits"][self.split][
                    "scenario_schedule_id"
                ]
            )
        else:
            platform_type = scenario_identity.platform_type
            eligible_entries = [
                entry
                for entry in cache.manifest["scenes"]
                if entry["split"] == self.split
                and entry["platform_starts"][platform_type]["qualified"]
            ]
            schedule_id = (
                self.platform_scenario_schedule_ids[platform_type]
                if self.platform_scenario_schedule_ids is not None
                else self.scenario_schedule_id
            )
        if not eligible_entries:
            raise ValueError(
                "formal cache has no platform-eligible scenes for the requested split"
            )
        entries = _formal_scheduled_entries(
            eligible_entries,
            scenario_schedule_id=schedule_id,
        )
        base_index = _formal_schedule_index(
            worker_index,
            scenario_identity.episode_cursor,
            len(entries),
            platform_worker_index=scenario_identity.platform_worker_index,
            platform_worker_count=scenario_identity.platform_worker_count,
        )
        base_episode_seed = _formal_episode_seed(
            "episode", scenario_identity
        )
        scale_bucket = scale_bucket_for_worker(
            scenario_identity.platform_worker_index,
            scenario_identity.platform_worker_count,
        )
        rejected: list[dict[str, object]] = []
        for attempt in range(len(entries)):
            entry = entries[(base_index + attempt) % len(entries)]
            record = cache.record(str(entry["scene_id"]))
            qualified_start = record.platform_starts[capability.platform_type]
            raw_start = qualified_start["qualified_start_cell"]
            if (
                qualified_start.get("qualified") is not True
                or not isinstance(raw_start, (list, tuple))
                or len(raw_start) != 2
            ):
                continue
            geometry_seed = (
                base_episode_seed
                if attempt == 0
                else hashlib.sha256(
                    (
                        f"{base_episode_seed}\0task-resample/v1\0{attempt}"
                    ).encode("utf-8")
                ).hexdigest()
            )
            geometry = freeze_formal_task_geometry(
                start_cell=(int(raw_start[0]), int(raw_start[1])),
                config=self.task_area,
                episode_seed=geometry_seed,
                scale_bucket=scale_bucket,
                halo=self._resolved_task_halo(),
            )
            common_key, platform_key = self._keys(
                cache=cache,
                record=record,
                geometry=geometry,
                capability=capability,
                qualified_start=qualified_start,
                scenario_identity_sha256=base_episode_seed,
            )
            common, platform = self._resolve_task_artifacts(
                cache=cache,
                common_key=common_key,
                platform_key=platform_key,
                geometry=geometry,
                record=record,
                capability=capability,
            )
            if (
                platform.diagnostics.get("resample_required") is True
                or int(
                    platform.diagnostics.get(
                        "coverable_detail_cell_count", 0
                    )
                )
                <= 0
            ):
                rejected.append(
                    {
                        "attempt": attempt,
                        "platform_task_key_sha256": platform_key.sha256(),
                        "reason": str(
                            platform.diagnostics.get(
                                "resample_reason", "ZERO_DENOMINATOR"
                            )
                        ),
                    }
                )
                continue
            loaded = _task_local_loaded_scene(
                record=record,
                common=common,
                platform=platform,
            )
            if rejected:
                loaded = replace(
                    loaded,
                    entry={
                        **dict(loaded.entry),
                        "task_resample_audit": rejected,
                    },
                )
            return loaded
        raise ValueError("formal task schedule has no nonzero denominator")

    def _restore_task_scene(
        self,
        *,
        scenario_identity: ScenarioIdentity,
        capability: FrozenPlatformCapability,
        state: FormalWorkerState,
    ) -> _LoadedScene:
        cache = load_formal_cache(
            Path(self.cache_manifest_path), require_full=not self.allow_preflight
        )
        record = cache.record(state.scene_id)
        qualified_start = record.platform_starts[capability.platform_type]
        coarse = state.task_coarse_bounds_half_open
        detail = state.task_detail_bounds_half_open
        global_start = qualified_start.get("qualified_start_cell")
        if (
            not isinstance(global_start, (list, tuple))
            or len(global_start) != 2
        ):
            raise ValueError("formal restore qualified start is invalid")
        local_start = (
            int(global_start[0]) - coarse[0],
            int(global_start[1]) - coarse[2],
        )
        if local_start != state.start_cell:
            raise ValueError("formal restore local start identity differs")
        geometry = FrozenTaskGeometry(
            episode_seed=state.episode_seed,
            scale_bucket=TaskScaleBucket(state.scale_bucket),
            span_cells=state.task_span_cells,
            coarse_bounds_half_open=coarse,
            detail_bounds_half_open=detail,
            halo_coarse_bounds_half_open=state.task_halo_bounds_half_open,
            local_start_cell=local_start,
            geometry_sha256=state.task_geometry_sha256,
        )
        common_key, platform_key = self._keys(
            cache=cache,
            record=record,
            geometry=geometry,
            capability=capability,
            qualified_start=qualified_start,
            scenario_identity_sha256=state.episode_seed,
        )
        if (
            common_key.sha256() != state.task_common_key_sha256
            or platform_key.sha256() != state.platform_task_key_sha256
        ):
            raise ValueError("formal restore task cache key differs")
        common, platform = self._resolve_task_artifacts(
            cache=cache,
            common_key=common_key,
            platform_key=platform_key,
            geometry=geometry,
            record=record,
            capability=capability,
        )
        if (
            common.artifact_sha256 != state.task_common_artifact_sha256
            or platform.artifact_sha256
            != state.platform_task_artifact_sha256
            or common.geometry.geometry_sha256
            != state.task_geometry_sha256
            or common.geometry.coarse_bounds_half_open != coarse
            or common.geometry.detail_bounds_half_open != detail
            or common.geometry.halo_coarse_bounds_half_open
            != state.task_halo_bounds_half_open
        ):
            raise ValueError("formal restore task artifact identity differs")
        return _task_local_loaded_scene(
            record=record,
            common=common,
            platform=platform,
        )

    def _keys(
        self,
        *,
        cache: FormalCache,
        record: FormalSceneIndexRecord,
        geometry: FrozenTaskGeometry,
        capability: FrozenPlatformCapability,
        qualified_start: Mapping[str, object],
        scenario_identity_sha256: str,
    ) -> tuple[TaskCommonKey, PlatformTaskKey]:
        halo = self._resolved_task_halo()
        source_commit = self.source_commit or cache.identity.v3_source_commit
        common_key = TaskCommonKey(
            scene_id=record.scene_id,
            scenario_identity_sha256=scenario_identity_sha256,
            source_identity_sha256=cache.identity.source_lock_file_sha256,
            coarse_bounds_half_open=geometry.coarse_bounds_half_open,
            detail_bounds_half_open=geometry.detail_bounds_half_open,
            task_span_cells=geometry.span_cells,
            scale_bucket=geometry.scale_bucket.value,
            geometry_sha256=geometry.geometry_sha256,
            halo_contract_sha256=_task_halo_contract_sha256(halo),
            capability_bundle_sha256=halo.capability_bundle_sha256,
            generator_sha256=GENERATOR_SHA256,
            source_commit=source_commit,
        )
        return common_key, _task_platform_key(
            common_key=common_key,
            platform=capability,
            qualified_start=qualified_start,
        )

    def _resolved_task_halo(self) -> object:
        halo = self.task_halo
        required = (
            "coarse_cells",
            "detail_cells",
            "sensor_radius_m",
            "platform_support_radius_m",
            "native_stencil_radius_m",
            "algorithm_id",
            "capability_bundle_sha256",
        )
        if halo is None or any(not hasattr(halo, name) for name in required):
            raise ValueError("formal task evidence halo is missing")
        return halo

    def _resolve_task_artifacts(
        self,
        *,
        cache: FormalCache,
        common_key: TaskCommonKey,
        platform_key: PlatformTaskKey,
        geometry: FrozenTaskGeometry,
        record: FormalSceneIndexRecord,
        capability: FrozenPlatformCapability,
    ) -> tuple[TaskCommonArtifact, PlatformTaskArtifact]:
        store = TaskCacheStore(cache.root)
        if self.task_cache_client is None:
            platform = store.load_platform(platform_key)
            if platform is None:
                raise TaskCacheCoordinatorError(
                    "formal task artifact is missing and no coordinator is attached"
                )
        else:
            contextual_getter = getattr(
                self.task_cache_client, "get_task_artifacts", None
            )
            if callable(contextual_getter):
                common, platform = contextual_getter(
                    cache=cache,
                    common_key=common_key,
                    platform_key=platform_key,
                    geometry=geometry,
                    record=record,
                    capability=capability,
                )
                if not isinstance(common, TaskCommonArtifact) or not isinstance(
                    platform, PlatformTaskArtifact
                ):
                    raise TypeError("formal contextual task cache result is invalid")
                if (
                    common.root.parent.parent.parent.parent != cache.root
                    or platform.root.parent.parent.parent.parent != cache.root
                ):
                    raise ValueError("formal task cache root differs")
                if (
                    platform.key != platform_key
                    or common.key != common_key
                    or platform.common_artifact_sha256
                    != common.artifact_sha256
                ):
                    raise ValueError("formal task artifact reference differs")
                return common, platform
            getter = getattr(self.task_cache_client, "get", None)
            if not callable(getter):
                raise TypeError("formal task cache client is invalid")
            reference = getter(
                platform_key,
                TaskBuildPriority.CURRENT_CURRICULUM,
            )
            platform = reference.load()
            if reference.cache_root != cache.root:
                raise ValueError("formal task cache root differs")
        common = store.load_common(common_key)
        if common is None:
            raise TaskCacheCoordinatorError(
                "formal task common artifact is missing after platform build"
            )
        if (
            platform.key != platform_key
            or common.key != common_key
            or platform.common_artifact_sha256 != common.artifact_sha256
        ):
            raise ValueError("formal task artifact reference differs")
        return common, platform

    @staticmethod
    def _make_worker(episode: FormalEpisode) -> FormalEnvironmentWorker:
        platform_type = episode.platform_type
        environment = create_v3_environment(
            platform_type=platform_type,
            request_builder=(
                episode.build_request
                if platform_type == "HOPPER"
                else episode.begin_ground_option
            ),
            initial_observation=episode.initial_observation,
            observation_boundary_controller=episode.controller,
            require_sensor_closed_loop=True,
            planning_failure_refresher=(
                episode.refresh_after_planning_failure
            ),
            reference_executor=episode.execute_reference,
            committed_hop_executor=(
                episode.committed_hop_feedback
                if platform_type == "HOPPER"
                else None
            ),
            ground_option_continuation_builder=(
                episode.continue_ground_option
                if platform_type != "HOPPER"
                else None
            ),
            ground_option_distance_provider=(
                episode.ground_option_distance_m
                if platform_type != "HOPPER"
                else None
            ),
            ground_option_clearer=(
                episode.clear_ground_option
                if platform_type != "HOPPER"
                else None
            ),
            candidate_diagnostics_provider=(
                episode.current_candidate_diagnostics
            ),
            candidate_decision_snapshot_provider=(
                episode.current_candidate_decision_snapshot
            ),
            remaining_coverable_detail_cell_count_provider=(
                episode.remaining_coverable_detail_cell_count
            ),
            ground_start_qualification_provider=(
                episode.ground_start_qualification_reason
                if platform_type != "HOPPER"
                else None
            ),
            plan_cost_scale=100.0,
            planner_elapsed_scale_s=_FORMAL_MACRO_STEP_TIME_SCALE_S,
            include_planner_wall_time_in_reward=False,
            task_scale_m=(
                episode.task_span_cells
                * episode.loaded.scene.base_canvas.geometry.resolution_m
            ),
        )
        return FormalEnvironmentWorker(
            environment=environment,
            initial_observation=episode.initial_observation,
            episode=episode,
        )



@dataclass(frozen=True, slots=True)
class FormalEnvironmentBuilder:
    cache_manifest_path: Path
    capability_bundle: FrozenCapabilityBundle
    task_area: TaskAreaConfig
    split: str = "train"
    allow_preflight: bool = False
    task_cache_client: object | None = None
    source_commit: str | None = None

    def build(self) -> FormalEnvironmentAssembly:
        cache = load_formal_cache(
            self.cache_manifest_path, require_full=not self.allow_preflight
        )
        if not self.capability_bundle.formal_eligible:
            raise ValueError("formal environment requires a formal capability bundle")
        if cache.identity.capability_sha256 != self.capability_bundle.bundle_sha256:
            raise ValueError("cache capability identity differs from current bundle")
        if self.split not in {"train", "validation", "test", "holdout"}:
            raise ValueError("formal environment split is invalid")
        platform_schedule_ids = {
            platform: str(
                cache.manifest["platform_eligibility"][platform][
                    "scenario_schedule_ids"
                ][self.split]
            )
            for platform in _PLATFORMS
        }
        schedule_digest = hashlib.sha256(
            json.dumps(
                {
                    "platform_scenario_schedule_ids": platform_schedule_ids,
                    "task_area": {
                        "minimum_size_m": self.task_area.minimum_size_m,
                        "maximum_size_m": self.task_area.maximum_size_m,
                        "sampling_algorithm": self.task_area.sampling_algorithm,
                    },
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        schedule_id = f"lunar-formal-combined-schedule/v1:{self.split}:{schedule_digest}"
        worker_builder = FormalWorkerBuilder(
            cache_manifest_path=str(self.cache_manifest_path),
            split=self.split,
            allow_preflight=self.allow_preflight,
            scenario_schedule_id=schedule_id,
            task_area=self.task_area,
            platform_scenario_schedule_ids=platform_schedule_ids,
            paired_evaluation=self.split != "train",
            task_cache_client=self.task_cache_client,
            source_commit=(
                self.source_commit or cache.identity.v3_source_commit
            ),
            task_halo=derive_task_evidence_halo(self.capability_bundle),
        )
        factory = FrozenCapabilityEnvironmentFactory(
            bundle=self.capability_bundle,
            scenario_schedule_id=schedule_id,
            builder=worker_builder,
        )
        template = factory(0, "WHEELED").initial_observation
        return FormalEnvironmentAssembly(
            factory=factory,
            observation_template=template,
            cache_manifest_sha256=str(cache.manifest["cache_manifest_sha256"]),
            scenario_schedule_id=schedule_id,
        )


__all__ = [
    "FormalEnvironmentAssembly",
    "FormalEnvironmentBuilder",
    "FormalEnvironmentWorker",
    "FormalEpisode",
    "FormalWorkerBuilder",
]

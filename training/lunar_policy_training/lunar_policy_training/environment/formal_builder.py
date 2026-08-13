"""Single authoritative builder for cache-backed formal C++ v3 workers."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
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
    FormalCacheError,
    _physical_capability_content_sha256,
    load_formal_cache,
)
from ..polar_data.hazards import (
    CraterBowl,
    NoGoPolygon,
    RockCircle,
    VectorHazardScene,
)
from ..polar_data.multires_scene import MultiResolutionScene, SceneTileProvider
from ..polar_data.raster import (
    GLOBAL_GEOMETRY,
    LOCAL_GEOMETRY,
    MapCanvas,
)
from ..policy.action_semantics import apply_goal_theta
from ..policy.observation import ObservationIdentity, PolicyBatch
from ..training_semantics import FORMAL_SENSOR_FOV_RAD, FORMAL_SENSOR_RANGE_M
from .candidate_builder import (
    CandidateBatch,
    CandidateBuilderV2,
    CandidateDiagnostics,
    PhysicalCandidateUniverse,
)
from .coverability import unpack_detail_mask
from .formal_episode_state import (
    FormalPathSampleState,
    FormalPoseState,
    FormalRevealState,
    FormalWorkerState,
    policy_batch_sha256,
)
from .frontier_oracle import (
    FrontierOpportunityOracle,
    FrontierOracleResult,
)
from .formal_start_qualification import (
    build_formal_mission_roi,
    formal_safe_start_cells,
)
from .task_area import scope_formal_task_area
from .macro_step import ExecutionEvents, PolicyAction
from .multires_observation import DetailObservedWindow, MultiresSensorObservationState
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


@dataclass(frozen=True, slots=True)
class _LoadedScene:
    scene: MultiResolutionScene
    arrays: Mapping[str, np.ndarray]
    entry: Mapping[str, object]
    scenario: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class _MapSnapshot:
    revision: int
    candidates: CandidateBatch
    global_map: object
    local_map: object
    world: object
    projection: PlatformProjection
    physical_reachability: PhysicalReachabilityResult
    candidate_universe: PhysicalCandidateUniverse
    planning_physical_snapshot_id: str
    frontier_oracle: FrontierOracleResult

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


def _scenario_inventory(cache: FormalCache) -> dict[str, Mapping[str, object]]:
    path = cache.root / "scenario-manifest.json"
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FormalCacheError("cache scenario manifest cannot be decoded") from error
    scenarios = document.get("scenarios") if isinstance(document, Mapping) else None
    if not isinstance(scenarios, list) or any(
        not isinstance(item, Mapping) for item in scenarios
    ):
        raise FormalCacheError("cache scenario manifest inventory is invalid")
    inventory = {str(item.get("scene_id")): item for item in scenarios}
    if len(inventory) != len(scenarios):
        raise FormalCacheError("cache scenario manifest has duplicate scene IDs")
    return inventory


def _load_multires_scene(cache: FormalCache, scene_id: str) -> _LoadedScene:
    entry = next(
        (item for item in cache.manifest["scenes"] if item["scene_id"] == scene_id),
        None,
    )
    if entry is None:
        raise FormalCacheError("scheduled scene is absent from cache")
    scenario = _scenario_inventory(cache).get(scene_id)
    if scenario is None:
        raise FormalCacheError("scheduled scene is absent from scenario manifest")
    arrays = cache.load_scene(scene_id)
    window_sha = entry.get("window_sha256")
    bounds = entry.get("world_bounds_m")
    if not isinstance(window_sha, str) or not isinstance(bounds, list) or len(bounds) != 4:
        raise FormalCacheError("cached scene canvas identity is invalid")
    canvas = MapCanvas(window_sha, tuple(float(value) for value in bounds))
    vector = VectorHazardScene(
        seed=str(scenario.get("scene_seed")),
        canvas=canvas,
        rocks=tuple(RockCircle(*row) for row in arrays["rocks"].tolist()),
        craters=tuple(CraterBowl(*row) for row in arrays["craters"].tolist()),
        no_go_polygons=tuple(
            NoGoPolygon(tuple(tuple(point) for point in polygon))
            for polygon in arrays["no_go_vertices"].tolist()
        ),
    )
    overlay = MultiResolutionScene(
        canvas,
        np.where(arrays["valid_mask"], arrays["elevation_m"], 0.0).astype(
            np.float32
        ),
        arrays["valid_mask"],
        vector,
        scenario_id=scene_id,
    ).project(canvas)
    base_elevation = (
        arrays["elevation_m"] - overlay.crater_elevation_delta_m
    ).astype(np.float32)
    scene = MultiResolutionScene(
        canvas,
        np.where(arrays["valid_mask"], base_elevation, 0.0).astype(np.float32),
        arrays["valid_mask"],
        vector,
        scenario_id=scene_id,
    )
    regenerated = scene.project(canvas)
    for name in (
        "physical_obstacle_ratio",
        "physical_obstacle_height_m",
        "forbidden_ratio",
    ):
        if not np.array_equal(getattr(regenerated, name), arrays[name]):
            raise FormalCacheError(f"cached {name} differs from vector scene")
    if not np.allclose(
        regenerated.elevation_m[arrays["valid_mask"]],
        arrays["elevation_m"][arrays["valid_mask"]],
        rtol=0.0,
        atol=2.0e-5,
    ):
        raise FormalCacheError("cached elevation differs from reconstructed scene")
    return _LoadedScene(scene, arrays, entry, scenario)


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
        self._static_hard = loaded.arrays[
            f"{platform_type.lower()}_hard_feasible"
        ].astype(bool)
        self._static_clearance = loaded.arrays[
            f"{platform_type.lower()}_clearance_margin_norm"
        ]
        self.start_cell = start_cell
        prefix = platform_type.lower()
        cached_physical_mask = unpack_detail_mask(
            loaded.arrays[f"{prefix}_physical_observation_pose_bits"].copy(),
            (GLOBAL_GEOMETRY.cells, GLOBAL_GEOMETRY.cells),
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
        self.mission = MissionRaster(
            loaded.scene.base_canvas,
            priority=self._mission_roi.astype(np.float32),
            roi_ratio=self._mission_roi.astype(np.float32),
        )
        coverability = loaded.entry["platform_coverability"][platform_type]
        detail_shape_raw = coverability["coverable_detail_shape"]
        if not isinstance(detail_shape_raw, list) or len(detail_shape_raw) != 2:
            raise ValueError("formal coverability detail geometry is invalid")
        self.sensor_state = MultiresSensorObservationState(
            scene=loaded.scene,
            tile_provider=SceneTileProvider(loaded.scene, capacity=8),
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
        )
        self._candidate_builder = CandidateBuilderV2(self.sensor_state)
        self._frontier_oracle = FrontierOpportunityOracle(self.sensor_state)
        self._legacy_candidate_builder = CandidateBuilderV2(
            NativeVisibilityEstimator(
                SensorGeometry(FORMAL_SENSOR_RANGE_M, FORMAL_SENSOR_FOV_RAD),
                resolution_m=GLOBAL_GEOMETRY.resolution_m,
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
            self.sensor_state.coverable_mask_sha256,
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
            state.coverability_mask_sha256,
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
        terminal_reason = _audit_candidate_boundary(
            snapshot.candidates.diagnostics,
            snapshot.frontier_oracle,
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
            snapshot.frontier_oracle.oracle_opportunity_count,
            snapshot.frontier_oracle.oracle_opportunity_set_sha256,
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
            state.oracle_opportunity_count,
            state.oracle_opportunity_set_sha256,
            state.terminal_reason,
            state.defer_candidate_rebuild,
            state.candidate_gain_resolution_m,
        )
        if replayed != persisted:
            raise ValueError("formal replay physical identity differs")
        self.initial_observation = observation

    def snapshot_state(self, environment: object) -> FormalWorkerState:
        """Capture one strict physical decision-boundary replay state."""
        if self._active_ground_option is not None:
            raise ValueError("formal snapshot cannot contain an active ground option")
        stable_state = getattr(environment, "snapshot_stable_state", None)
        if not callable(stable_state):
            raise ValueError("formal snapshot environment is invalid")
        execution_state = stable_state().get("execution_state")
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
        terminal_reason = _audit_candidate_boundary(
            snapshot.candidates.diagnostics,
            snapshot.frontier_oracle,
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
            coverability_mask_sha256=str(
                self.sensor_state.coverable_mask_sha256
            ),
            start_cell=self.start_cell,
            current_pose=self._pose_state(self.current_pose),
            legged_body_z_m=float(self._current_legged_body_z_m),
            execution_state=str(execution_state),
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
            oracle_opportunity_count=(
                snapshot.frontier_oracle.oracle_opportunity_count
            ),
            oracle_opportunity_set_sha256=(
                snapshot.frontier_oracle.oracle_opportunity_set_sha256
            ),
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

    def frontier_oracle_result(self) -> FrontierOracleResult:
        snapshot = self._snapshot
        if snapshot is None:
            raise RuntimeError("formal frontier oracle has no observed snapshot")
        return snapshot.frontier_oracle

    def remaining_coverable_detail_cell_count(self) -> int:
        remaining = self.sensor_state.remaining_coverable_detail_cell_count
        if remaining is None:
            raise RuntimeError("formal coverability truth diagnostic is unavailable")
        return remaining

    def _build_mission_roi(self) -> np.ndarray:
        scoped = self.loaded.arrays.get("scoped_mission_roi")
        if scoped is not None:
            roi = np.ascontiguousarray(scoped, dtype=np.bool_)
            if roi.shape != (
                GLOBAL_GEOMETRY.cells,
                GLOBAL_GEOMETRY.cells,
            ):
                raise ValueError("formal scoped mission ROI geometry is invalid")
            if not bool(roi[self.start_cell]):
                raise ValueError("formal scoped mission ROI excludes the start")
            return roi
        return build_formal_mission_roi(self.loaded.arrays)

    def _observed_maps(self) -> tuple[object, object, DetailObservedWindow]:
        observed = self.sensor_state.observed
        stamp_ns = 1_000_000_000 + self._revision * 1_000_000
        global_map = _grid_map(
            canvas=observed.canvas,
            frame_id="map",
            elevation_m=observed.elevation_m,
            valid_mask=observed.valid_mask,
            physical_obstacle_ratio=observed.physical_obstacle_ratio,
            physical_obstacle_height_m=self.sensor_state.coarse_obstacle_height_m,
            forbidden_ratio=self.sensor_state.coarse_forbidden_ratio,
            observation_age_s=observed.observation_age_s,
            observation_quality=observed.observation_quality,
            observation_count=observed.observation_count,
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
        return global_map, local_map, detail

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

    def build_policy_observation(
        self, observed, pose: Pose2
    ) -> PolicyBatch:
        if pose != self.current_pose:
            raise ValueError("policy observation pose differs from episode state")
        self._revision += 1
        global_map, local_map, _ = self._observed_maps()
        if self._defer_candidate_rebuild:
            return self._build_ground_continuation_observation(
                pose,
                global_map=global_map,
                local_map=local_map,
            )
        local = self.sensor_state.local_observation(pose)
        world = observed.to_observed_world(local=local)
        projection_request = self._base_request(global_map, local_map)
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
            maximum_edge_distance_m=30.0,
            local_traversability_projection=(
                None if self.platform_type == "HOPPER" else local_cpp
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
            physical_evidence_sha256=(
                self.sensor_state.physical_evidence_sha256()
            ),
            physical_reachability_algorithm_id=(
                physical_reachability.physical_reachability_algorithm_id
            ),
            goal_tolerance_mm=(0 if self.platform_type == "HOPPER" else 200),
        )
        failure_snapshot_id = self._planner_failure_snapshot_id
        failed_candidate_ids = set(self._planner_failed_candidate_ids)
        if failure_snapshot_id != candidate_universe.physical_snapshot_id:
            failure_snapshot_id = candidate_universe.physical_snapshot_id
            failed_candidate_ids.clear()
        pending_failure = self._pending_planning_failure
        if pending_failure is not None:
            (
                candidate_id,
                disposition,
                pending_snapshot_id,
                allow_locked_target_exit,
            ) = pending_failure
            if pending_snapshot_id != candidate_universe.physical_snapshot_id:
                raise ValueError(
                    "planning failure physical snapshot changed during rebuild"
                )
            if disposition == (
                bridge_api.CandidateDisposition.SUPPRESS_FOR_CURRENT_PHYSICAL_SNAPSHOT
            ):
                current_ids = {
                    candidate.candidate_id
                    for candidate in candidate_universe.candidates
                }
                if candidate_id in current_ids:
                    failed_candidate_ids.add(candidate_id)
                elif not allow_locked_target_exit:
                    raise ValueError(
                        "planning failure candidate changed during rebuild"
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
        frontier_oracle = self._frontier_oracle.evaluate_physical(
            world,
            self.mission,
            pose_map=pose,
            physical_reachability=physical_reachability,
        )
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
            local_map,
            world,
            projection,
            physical_reachability,
            candidate_result.universe,
            candidate_universe.physical_snapshot_id,
            frontier_oracle,
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
        local_map: object,
    ) -> PolicyBatch:
        """Refresh planning maps without rebuilding unused policy candidates."""
        snapshot = self._snapshot
        if snapshot is None:
            raise RuntimeError("ground continuation has no prior map snapshot")
        self._snapshot = _MapSnapshot(
            self._revision,
            snapshot.candidates,
            global_map,
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
                    self.sensor_state.physical_evidence_sha256()
                ),
            ),
            snapshot.frontier_oracle,
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
        request = self._base_request(snapshot.global_map, snapshot.local_map)
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
        request = self._base_request(snapshot.global_map, snapshot.local_map)
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
        before_failed_ids = set(self._planner_failed_candidate_ids)
        rebuilt = self.controller.rebuild_without_sensor_update(
            pose_map=self.current_pose,
            execution_state=execution_state,
        )
        event_kind = (
            "PLANNING_FAILURE_SUPPRESS"
            if self._planner_failed_candidate_ids != before_failed_ids
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
            )
        self._pending_hop_landing = None
        elapsed_s = self._pending_hop_elapsed_s
        self._pending_hop_elapsed_s = 0.0
        self._hopper_feedback_phase = 0
        self.current_pose = landing
        evidence = SensorBoundaryEvidence(landing, elapsed_s)
        self._record_reveal(evidence, "LANDED_HOLD")
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
    sensor_closed_loop: ClassVar[bool] = True

    def __call__(
        self,
        worker_index: int,
        platform_type: str,
        capability: FrozenPlatformCapability,
        scenario_identity: ScenarioIdentity,
    ) -> FormalEnvironmentWorker:
        loaded = self._load_scheduled_scene(worker_index, scenario_identity)
        safe = self._safe_start_cells(loaded, platform_type)
        payload = loaded.entry["platform_coverability"][platform_type]
        start_raw = payload["qualified_start_cell"]
        if (
            payload.get("eligible") is not True
            or not isinstance(start_raw, list)
            or len(start_raw) != 2
        ):
            raise ValueError("formal scheduled scene has no qualified platform start")
        start_cell = (int(start_raw[0]), int(start_raw[1]))
        if start_cell not in safe:
            raise ValueError("formal cached qualified start is not platform-safe")
        loaded, _ = scope_formal_task_area(
            loaded,
            platform_type=platform_type,
            start_cell=start_cell,
            config=self.task_area,
            episode_seed=_formal_episode_seed("episode", scenario_identity),
        )
        episode = FormalEpisode(
            worker_index=worker_index,
            platform_type=platform_type,
            capability=capability,
            scenario_identity=scenario_identity,
            loaded=loaded,
            start_cell=start_cell,
        )
        if not bool(episode.initial_observation.candidate_mask.any()):
            raise ValueError(
                "formal cached start qualification differs from the observed-only candidate"
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
        loaded = self._load_scheduled_scene(worker_index, scenario_identity)
        loaded, _ = scope_formal_task_area(
            loaded,
            platform_type=platform_type,
            start_cell=parsed.start_cell,
            config=self.task_area,
            episode_seed=_formal_episode_seed("episode", scenario_identity),
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
            execution_state=parsed.execution_state
        )
        return worker

    def _load_scheduled_scene(
        self,
        worker_index: int,
        scenario_identity: ScenarioIdentity,
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
                and entry["platform_coverability"][platform_type]["eligible"]
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
        entry = entries[base_index]
        return _load_multires_scene(cache, str(entry["scene_id"]))

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
            frontier_oracle=episode.frontier_oracle_result,
            remaining_coverable_detail_cell_count_provider=(
                episode.remaining_coverable_detail_cell_count
            ),
            plan_cost_scale=100.0,
            planner_elapsed_scale_s=_FORMAL_MACRO_STEP_TIME_SCALE_S,
            include_planner_wall_time_in_reward=False,
        )
        return FormalEnvironmentWorker(
            environment=environment,
            initial_observation=episode.initial_observation,
            episode=episode,
        )

    @staticmethod
    def _safe_start_cells(
        loaded: _LoadedScene, platform_type: str
    ) -> tuple[tuple[int, int], ...]:
        return formal_safe_start_cells(loaded.arrays, platform_type)


@dataclass(frozen=True, slots=True)
class FormalEnvironmentBuilder:
    cache_manifest_path: Path
    capability_bundle: FrozenCapabilityBundle
    task_area: TaskAreaConfig
    split: str = "train"
    allow_preflight: bool = False

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

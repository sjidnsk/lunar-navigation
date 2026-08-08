"""Single authoritative builder for cache-backed formal C++ v3 workers."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import ClassVar, Mapping

import numpy as np
import torch

import lunar_planner_training_bridge as bridge_api

from ..capability_freeze import (
    FrozenCapabilityBundle,
    FrozenCapabilityEnvironmentFactory,
    FrozenLeggedCapability,
    FrozenPlatformCapability,
    ScenarioIdentity,
)
from ..polar_data.formal_cache import FormalCache, FormalCacheError, load_formal_cache
from ..polar_data.hazards import (
    CraterBowl,
    NoGoPolygon,
    RockCircle,
    VectorHazardScene,
)
from ..polar_data.multires_scene import MultiResolutionScene, SceneTileProvider
from ..polar_data.raster import GLOBAL_GEOMETRY, LOCAL_GEOMETRY, MapCanvas
from ..policy.action_semantics import apply_goal_theta
from ..policy.observation import ObservationIdentity, PolicyBatch
from ..training_semantics import FORMAL_SENSOR_FOV_RAD, FORMAL_SENSOR_RANGE_M
from .candidate_builder import CandidateBatch, CandidateBuilderV2
from .formal_episode_state import (
    FormalPoseState,
    FormalRevealState,
    FormalWorkerState,
    policy_batch_sha256,
)
from .formal_start_qualification import (
    build_formal_mission_roi,
    formal_safe_start_cells,
)
from .macro_step import ExecutionEvents, PolicyAction
from .multires_observation import DetailObservedWindow, MultiresSensorObservationState
from .observation_boundary import (
    ObservationBoundaryController,
    SensorBoundaryEvidence,
)
from .observation_builder import (
    LocalObservation,
    MissionRaster,
    ObservationBuilderV2,
    PlatformProjection,
    Pose2,
)
from .parallel_pool import ParallelEnvironmentWorker
from .v3_environment import (
    CommittedHopExecutionFeedback,
    PreparedPlanRequest,
    ReferenceExecutionResult,
    create_v3_environment,
)
from .visibility import NativeVisibilityEstimator, SensorGeometry


_PLATFORMS = ("WHEELED", "LEGGED", "HOPPER")
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


@dataclass(frozen=True, slots=True)
class FormalEnvironmentWorker(ParallelEnvironmentWorker):
    episode: "FormalEpisode"

    def snapshot_episode_state(self) -> dict[str, object]:
        return self.episode.snapshot_state(self.environment).to_dict()


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
    ) -> None:
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
        x_m, y_m = loaded.scene.base_canvas.grid_center_world(*start_cell)
        elevation_m = float(loaded.arrays["elevation_m"][start_cell])
        self.current_pose = Pose2(x_m, y_m, 0.0, "map", elevation_m)
        self._current_legged_body_z_m = elevation_m
        if isinstance(capability.typed_capability, FrozenLeggedCapability):
            interval = capability.typed_capability.body_height_m
            self._current_legged_body_z_m += 0.5 * (
                interval.lower + interval.upper
            )
        self.start_is_safe = bool(self._static_hard[start_cell])
        self._mission_roi = self._build_mission_roi()
        self.mission = MissionRaster(
            loaded.scene.base_canvas,
            priority=self._mission_roi.astype(np.float32),
            roi_ratio=self._mission_roi.astype(np.float32),
        )
        self.sensor_state = MultiresSensorObservationState(
            scene=loaded.scene,
            tile_provider=SceneTileProvider(loaded.scene, capacity=8),
            mission_roi_ratio=self.mission.roi_ratio,
            mission_priority=self.mission.priority,
        )
        self._global_visibility = NativeVisibilityEstimator(
            SensorGeometry(FORMAL_SENSOR_RANGE_M, FORMAL_SENSOR_FOV_RAD),
            resolution_m=GLOBAL_GEOMETRY.resolution_m,
        )
        self._candidate_builder = CandidateBuilderV2(self._global_visibility)
        self._observation_builder = ObservationBuilderV2()
        self._bridge = bridge_api.PlannerBridge()
        self._snapshot: _MapSnapshot | None = None
        self._revision = 0
        self._pending_hop_landing: Pose2 | None = None
        self._hopper_feedback_phase = 0
        self.last_hop_available_delta_v_mps = 0.0
        self._reveal_history: list[FormalRevealState] = []
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
    ) -> None:
        self._reveal_history.append(
            FormalRevealState(
                pose=self._pose_state(evidence.pose_map),
                elapsed_s=float(evidence.elapsed_s),
                execution_state=execution_state,
                legged_body_z_m=float(self._current_legged_body_z_m),
            )
        )

    def replay_state(self, state: FormalWorkerState) -> None:
        """Rebuild dynamic sensor state from frozen truth and reveal history."""
        expected = (
            state.scenario_schedule_id == self.scenario_identity.scenario_schedule_id
            and state.platform_type == self.platform_type
            and state.worker_index == self.worker_index
            and state.platform_worker_index
            == self.scenario_identity.platform_worker_index
            and state.platform_worker_count
            == self.scenario_identity.platform_worker_count
            and state.episode_cursor == self.episode_cursor
            and state.scene_id == self.scene_id
            and state.scene_seed == self.scene_seed
            and state.start_seed == self.start_seed
            and state.episode_seed == self.episode_seed
            and state.start_cell == self.start_cell
        )
        if not expected:
            raise ValueError("formal replay identity differs from frozen episode")
        for reveal in state.reveal_history:
            self.current_pose = self._pose_from_state(reveal.pose)
            self._current_legged_body_z_m = reveal.legged_body_z_m
            evidence = SensorBoundaryEvidence(
                self.current_pose, reveal.elapsed_s
            )
            boundary = self.controller.after_execution(
                platform_type=self.platform_type,
                execution_state=reveal.execution_state,
                evidence=evidence,
            )
            if not boundary.updated:
                raise ValueError("formal replay did not produce an observation")
            self._record_reveal(evidence, reveal.execution_state)
        self.initial_observation = self.controller.current_observation
        self.last_hop_available_delta_v_mps = (
            state.last_hop_available_delta_v_mps
        )
        if (
            self._revision != state.observation_revision
            or self.current_pose != self._pose_from_state(state.current_pose)
            or not math.isclose(
                self._current_legged_body_z_m,
                state.legged_body_z_m,
                rel_tol=0.0,
                abs_tol=0.0,
            )
        ):
            raise ValueError("formal replay dynamic state differs")

    def snapshot_state(self, environment: object) -> FormalWorkerState:
        observation = environment.current_observation
        identities = observation.observation_identities
        if identities is None or len(identities) != 1:
            raise ValueError("formal snapshot observation identity is missing")
        dynamic = environment.snapshot_stable_state()
        identity = identities[0]
        return FormalWorkerState.from_dict(
            {
                "scenario_schedule_id": self.scenario_identity.scenario_schedule_id,
                "platform_type": self.platform_type,
                "worker_index": self.worker_index,
                "platform_worker_index": self.scenario_identity.platform_worker_index,
                "platform_worker_count": self.scenario_identity.platform_worker_count,
                "episode_cursor": self.episode_cursor,
                "scene_id": self.scene_id,
                "scene_seed": self.scene_seed,
                "start_seed": self.start_seed,
                "episode_seed": self.episode_seed,
                "start_cell": list(self.start_cell),
                "current_pose": self._pose_state(self.current_pose).to_dict(),
                "legged_body_z_m": float(self._current_legged_body_z_m),
                "execution_state": dynamic["execution_state"],
                "observation_revision": self._revision,
                "state_time_ns": identity.state_time_ns,
                "reveal_history": [
                    reveal.to_dict() for reveal in self._reveal_history
                ],
                "observation_identity": {
                    "episode_id": identity.episode_id,
                    "mission_revision": identity.mission_revision,
                    "map_snapshot_id": identity.map_snapshot_id,
                    "robot_state_id": identity.robot_state_id,
                    "state_time_ns": identity.state_time_ns,
                    "execution_state": identity.execution_state,
                    "candidate_set_id": identity.candidate_set_id,
                },
                "policy_batch_sha256": policy_batch_sha256(observation),
                "rejected_candidate_indices": dynamic[
                    "rejected_candidate_indices"
                ],
                "last_hop_available_delta_v_mps": float(
                    self.last_hop_available_delta_v_mps
                ),
            }
        )

    def _build_mission_roi(self) -> np.ndarray:
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
        return request

    def build_policy_observation(
        self, observed, pose: Pose2
    ) -> PolicyBatch:
        if pose != self.current_pose:
            raise ValueError("policy observation pose differs from episode state")
        self._revision += 1
        local = self.sensor_state.local_observation(pose)
        world = observed.to_observed_world(local=local)
        global_map, local_map, _ = self._observed_maps()
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
        projection = PlatformProjection(
            canvas=world.canvas,
            traversable_ratio=(
                self._static_hard
                & self._mission_roi
            ).astype(np.float32),
            local_traversable_ratio=center,
            clearance_margin_norm=self._static_clearance,
            source=f"cpp_v3/{self.capability.content_sha256}",
        )
        candidates = self._candidate_builder.build(
            world, self.mission, pose, projection
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
            self._revision, candidates, global_map, local_map
        )
        return PolicyBatch(
            **{
                name: torch.from_numpy(value.copy())
                for name, value in arrays.items()
            }
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
        feature = snapshot.candidates.features[action.frontier_index]
        canvas = self.loaded.scene.base_canvas
        target_x = canvas.bounds_m[0] + float(feature[0]) * canvas.geometry.size_m
        target_y = canvas.bounds_m[3] - float(feature[1]) * canvas.geometry.size_m
        target_row, target_column = canvas.world_to_grid(target_x, target_y)
        target_z = float(self.loaded.arrays["elevation_m"][target_row, target_column])
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
        apply_goal_theta(request.goal, self.platform_type, action.theta_rad)
        return PreparedPlanRequest(request=request, identity=expected_identity)

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
        end = points[-1].pose
        terrain_z = end.position_m.z
        if self.platform_type == "LEGGED":
            self._current_legged_body_z_m = end.position_m.z
            terrain_z = self._terrain_elevation_m(
                end.position_m.x, end.position_m.y
            )
        next_pose = Pose2(
            end.position_m.x,
            end.position_m.y,
            _yaw_from_pose(end),
            "map",
            terrain_z,
        )
        self.current_pose = next_pose
        evidence = SensorBoundaryEvidence(
            next_pose,
            points[-1].time_from_start.total_seconds(),
        )
        self._record_reveal(evidence, "DECISION_BOUNDARY")
        return ReferenceExecutionResult(
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
            execution_state="DECISION_BOUNDARY",
            execution_events=ExecutionEvents(
                reference_samples_consumed=len(points),
                selected_action_observed_safe=True,
            ),
            sensor_boundary_evidence=evidence,
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
        self.last_hop_available_delta_v_mps = float(
            segment.available_delta_v_mps
        )
        self._hopper_feedback_phase = 0
        return ReferenceExecutionResult(
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
        self._hopper_feedback_phase = 0
        self.current_pose = landing
        evidence = SensorBoundaryEvidence(landing, 1.0)
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
    sensor_closed_loop: ClassVar[bool] = True

    def __call__(
        self,
        worker_index: int,
        platform_type: str,
        capability: FrozenPlatformCapability,
        scenario_identity: ScenarioIdentity,
    ) -> FormalEnvironmentWorker:
        loaded = self._load_scheduled_scene(worker_index, scenario_identity)
        safe = list(self._safe_start_cells(loaded, platform_type))
        start_seed = _formal_episode_seed("start", scenario_identity)
        if safe:
            offset = int(start_seed[:16], 16) % len(safe)
            safe = safe[offset:] + safe[:offset]
        qualification = loaded.entry["start_qualification"]
        fallback_raw = qualification["platform_start_cells"][platform_type]
        if not isinstance(fallback_raw, list) or len(fallback_raw) != 2:
            raise ValueError("formal scheduled scene has no qualified platform start")
        fallback = (int(fallback_raw[0]), int(fallback_raw[1]))
        if fallback not in safe:
            raise ValueError("formal cached qualified start is not platform-safe")
        trial_cells = [safe[0]]
        if fallback != trial_cells[0]:
            trial_cells.append(fallback)
        for start_cell in trial_cells:
            episode = FormalEpisode(
                worker_index=worker_index,
                platform_type=platform_type,
                capability=capability,
                scenario_identity=scenario_identity,
                loaded=loaded,
                start_cell=start_cell,
            )
            if bool(episode.initial_observation.candidate_mask.any()):
                return self._make_worker(episode)
        raise ValueError(
            "formal cached start qualification differs from the observed-only candidate"
        )

    def restore(
        self,
        worker_index: int,
        platform_type: str,
        capability: FrozenPlatformCapability,
        scenario_identity: ScenarioIdentity,
        state: object,
    ) -> FormalEnvironmentWorker:
        restored = FormalWorkerState.from_dict(state)
        loaded = self._load_scheduled_scene(worker_index, scenario_identity)
        if restored.scene_id != loaded.scene.scene_id:
            raise ValueError("formal restored scene differs from schedule")
        safe = self._safe_start_cells(loaded, platform_type)
        if restored.start_cell not in safe:
            raise ValueError("formal restored start is not platform-safe")
        episode = FormalEpisode(
            worker_index=worker_index,
            platform_type=platform_type,
            capability=capability,
            scenario_identity=scenario_identity,
            loaded=loaded,
            start_cell=restored.start_cell,
        )
        episode.replay_state(restored)
        worker = self._make_worker(episode)
        worker.environment.restore_stable_state(
            execution_state=restored.execution_state,
            rejected_candidate_indices=restored.rejected_candidate_indices,
        )
        observation = worker.environment.current_observation
        if (
            observation.observation_identities != (restored.observation_identity,)
            or policy_batch_sha256(observation)
            != restored.policy_batch_sha256
        ):
            raise ValueError("formal restored observation digest differs")
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
        entries = [
            entry
            for entry in cache.manifest["scenes"]
            if entry["split"] == self.split
            and entry["start_qualification"]["common_eligible"]
        ]
        if not entries:
            raise ValueError(
                "formal cache has no common start-eligible scenes for the requested split"
            )
        base_index = _formal_schedule_index(
            worker_index,
            scenario_identity.episode_cursor,
            len(entries),
            platform_worker_index=scenario_identity.platform_worker_index,
            platform_worker_count=scenario_identity.platform_worker_count,
        )
        episode_seed = _formal_episode_seed("episode", scenario_identity)
        entry = entries[(base_index + int(episode_seed[:16], 16)) % len(entries)]
        return _load_multires_scene(cache, str(entry["scene_id"]))

    @staticmethod
    def _make_worker(episode: FormalEpisode) -> FormalEnvironmentWorker:
        platform_type = episode.platform_type
        environment = create_v3_environment(
            platform_type=platform_type,
            request_builder=episode.build_request,
            initial_observation=episode.initial_observation,
            observation_boundary_controller=episode.controller,
            require_sensor_closed_loop=True,
            reference_executor=episode.execute_reference,
            committed_hop_executor=(
                episode.committed_hop_feedback
                if platform_type == "HOPPER"
                else None
            ),
            plan_cost_scale=100.0,
            planner_elapsed_scale_s=2.0,
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
        schedule_id = (
            f"{cache.manifest['cache_manifest_sha256']}/{self.split}/v5"
        )
        worker_builder = FormalWorkerBuilder(
            str(self.cache_manifest_path),
            self.split,
            self.allow_preflight,
            schedule_id,
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

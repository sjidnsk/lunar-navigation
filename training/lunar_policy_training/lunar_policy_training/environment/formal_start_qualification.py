"""Exact, planner-free qualification of actionable formal episode starts."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np

from ..capability_freeze import FrozenPlatformCapability, PLATFORMS
from ..polar_data.multires_scene import MultiResolutionScene, SceneTileProvider
from ..polar_data.raster import GLOBAL_GEOMETRY, LOCAL_GEOMETRY
from ..training_semantics import FORMAL_SENSOR_FOV_RAD, FORMAL_SENSOR_RANGE_M
from .candidate_builder import CandidateBuilderV2
from .coverability import canonical_physical_positions_um
from .multires_observation import MultiresSensorObservationState
from .observation_builder import MissionRaster, PlatformProjection, Pose2
from .platform_reachability import (
    PhysicalReachabilityResult,
    PlatformCandidateReachability,
)
from .primitive_reachability import native_start_failure_is_ineligible


FORMAL_BOUNDARY_MARGIN_CELLS = math.ceil(
    (FORMAL_SENSOR_RANGE_M + LOCAL_GEOMETRY.size_m / 2.0)
    / GLOBAL_GEOMETRY.resolution_m
)


@dataclass(frozen=True, slots=True)
class FormalStartQualification:
    cell: tuple[int, int]
    initial_candidate_count: int
    exact_start_position_m: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        if (
            len(self.cell) != 2
            or any(type(value) is not int or value < 0 for value in self.cell)
            or type(self.initial_candidate_count) is not int
            or self.initial_candidate_count <= 0
        ):
            raise ValueError("formal start qualification is invalid")
        position = self.exact_start_position_m
        if position is not None and (
            not isinstance(position, tuple)
            or len(position) != 3
            or any(type(value) is not float for value in position)
            or _canonical_position_m(position) != position
        ):
            raise ValueError("formal exact start position is invalid")


def _canonical_position_m(
    position_m: tuple[float, float, float] | np.ndarray,
) -> tuple[float, float, float]:
    values = np.ascontiguousarray(
        np.asarray(position_m, dtype=np.float64).reshape((1, 3))
    )
    canonical_um = canonical_physical_positions_um(values)[0]
    return tuple(float(value) / 1_000_000.0 for value in canonical_um)


def _physical_position_at_cell(
    physical: PhysicalReachabilityResult,
    cell: tuple[int, int],
) -> tuple[float, float, float] | None:
    envelope = physical.hopper_single_hop_envelope
    if physical.platform_type == "HOPPER" and envelope is not None:
        mask = envelope.certified_mask
        positions = envelope.certified_positions_m
    else:
        mask = physical.physical_observation_pose_mask
        positions = physical.observation_positions_m
    rows, columns = np.nonzero(mask)
    matching = np.flatnonzero((rows == cell[0]) & (columns == cell[1]))
    if len(matching) == 0:
        return None
    if len(matching) != 1 or len(positions) != len(rows):
        raise RuntimeError("physical start position authority is ambiguous")
    return tuple(
        float(value)
        for value in positions[int(matching[0])]
    )


def build_formal_mission_roi(arrays: Mapping[str, np.ndarray]) -> np.ndarray:
    """Return the exact finite mission ROI shared by qualification and episodes."""
    roi = np.asarray(arrays["valid_mask"], dtype=np.bool_).copy()
    if roi.shape != (GLOBAL_GEOMETRY.cells, GLOBAL_GEOMETRY.cells):
        raise ValueError("formal mission ROI source has invalid geometry")
    margin = FORMAL_BOUNDARY_MARGIN_CELLS
    roi[:margin] = False
    roi[-margin:] = False
    roi[:, :margin] = False
    roi[:, -margin:] = False
    forbidden = np.asarray(arrays["forbidden_ratio"], dtype=np.float32)
    if forbidden.shape != roi.shape:
        raise ValueError("formal forbidden source has invalid geometry")
    roi &= forbidden == 0.0
    return np.ascontiguousarray(roi)


def formal_safe_start_cells(
    arrays: Mapping[str, np.ndarray],
    platform_type: str,
) -> tuple[tuple[int, int], ...]:
    """Enumerate all physically safe starts without an arbitrary search cap."""
    if platform_type not in PLATFORMS:
        raise ValueError("formal start platform is invalid")
    prefix = platform_type.lower()
    hard = np.asarray(arrays[f"{prefix}_hard_feasible"], dtype=np.bool_)
    clearance = np.asarray(
        arrays[f"{prefix}_clearance_margin_norm"], dtype=np.float32
    )
    valid = np.asarray(arrays["valid_mask"], dtype=np.bool_)
    shape = (GLOBAL_GEOMETRY.cells, GLOBAL_GEOMETRY.cells)
    if hard.shape != shape or clearance.shape != shape or valid.shape != shape:
        raise ValueError("formal start projection has invalid geometry")
    mask = hard & valid & (clearance > 0.0)
    margin = FORMAL_BOUNDARY_MARGIN_CELLS
    mask[:margin] = False
    mask[-margin:] = False
    mask[:, :margin] = False
    mask[:, -margin:] = False
    rows, columns = np.nonzero(mask)
    center = (GLOBAL_GEOMETRY.cells - 1) / 2.0
    return tuple(
        sorted(
            zip(rows.tolist(), columns.tolist(), strict=True),
            key=lambda value: (
                (value[0] - center) ** 2 + (value[1] - center) ** 2,
                value[0],
                value[1],
            ),
        )
    )


def _qualify_at_pose(
    *,
    scene: MultiResolutionScene,
    arrays: Mapping[str, np.ndarray],
    capability: FrozenPlatformCapability,
    mission: MissionRaster,
    start_cell: tuple[int, int],
    pose: Pose2,
    exact_hopper_start_position_m: tuple[float, float, float] | None,
) -> tuple[PhysicalReachabilityResult, int]:
    platform_type = capability.platform_type
    sensor_state = MultiresSensorObservationState(
        scene=scene,
        tile_provider=SceneTileProvider(scene, capacity=8),
        mission_roi_ratio=mission.roi_ratio,
        mission_priority=mission.priority,
    )
    sensor_state.observe_world(pose, elapsed_s=0.0)
    local = sensor_state.local_observation(pose)
    world = sensor_state.observed.to_observed_world(local=local)
    from ..polar_data.formal_cache import (
        _physical_capability_content_sha256,
        _projection_request,
    )
    from .formal_builder import _grid_map
    import lunar_planner_training_bridge as bridge_api

    stamp_ns = 1_001_000_000
    observed = sensor_state.observed
    global_map = _grid_map(
        canvas=observed.canvas,
        frame_id="map",
        elevation_m=observed.elevation_m,
        valid_mask=observed.valid_mask,
        physical_obstacle_ratio=observed.physical_obstacle_ratio,
        physical_obstacle_height_m=sensor_state.coarse_obstacle_height_m,
        forbidden_ratio=sensor_state.coarse_forbidden_ratio,
        observation_age_s=observed.observation_age_s,
        observation_quality=observed.observation_quality,
        observation_count=observed.observation_count,
        stamp_ns=stamp_ns,
    )
    planning_detail = sensor_state.planning_observation(pose)
    local_map = _grid_map(
        canvas=planning_detail.canvas,
        frame_id="odom",
        elevation_m=planning_detail.elevation_m,
        valid_mask=planning_detail.valid_mask,
        physical_obstacle_ratio=planning_detail.physical_obstacle_ratio,
        physical_obstacle_height_m=planning_detail.physical_obstacle_height_m,
        forbidden_ratio=planning_detail.forbidden_ratio,
        observation_age_s=planning_detail.observation_age_s,
        observation_quality=planning_detail.observation_quality,
        observation_count=planning_detail.observation_count,
        stamp_ns=stamp_ns,
    )
    request = _projection_request(
        capability,
        scene,
        scene.project(scene.base_canvas),
        start_cell=start_cell,
        exact_start_position_m=exact_hopper_start_position_m,
    )
    request.request_id = (
        f"formal-observed/{platform_type.lower()}/"
        f"{scene.scene_id}/qualification/1"
    )
    request.global_map_generation = 1
    request.local_map_generation = 1
    request.state_time.nanoseconds_since_epoch = stamp_ns
    request.world.global_map = global_map
    request.world.local_map = local_map
    request.world.map_from_odom.stamp.nanoseconds_since_epoch = stamp_ns
    request.capability = capability.to_bridge_capability()
    request.config.global_map.base_resolution_m = LOCAL_GEOMETRY.resolution_m
    request.config.global_map.maximum_level = 5
    request.config.global_map.target_axis_cells = 256
    request.config.wheel.xy_resolution_m = LOCAL_GEOMETRY.resolution_m
    request.config.wheel.yaw_bin_count = 64
    request.config.legged.xy_resolution_m = LOCAL_GEOMETRY.resolution_m
    request.config.legged.yaw_bin_count = 64
    request.config.local_frontier.additional_corridor_margin_m = 2.0
    bridge = bridge_api.PlannerBridge()
    local_cpp = bridge.project_traversability(request)
    local_known = np.ascontiguousarray(
        np.flipud(local_cpp.known).astype(np.bool_)
    )
    local_hard = np.ascontiguousarray(
        np.flipud(local_cpp.hard_feasible).astype(np.bool_)
    )
    local_traversable = np.ascontiguousarray(
        (local_known & local_hard)[144:176, 144:176], dtype=np.float32
    )
    physical_capability_sha256 = _physical_capability_content_sha256(capability)
    observed_safe = np.ascontiguousarray(
        world.observed_mask & (world.physical_obstacle_layer.values == 0.0)
    )
    projection = PlatformProjection(
        canvas=world.canvas,
        traversable_ratio=observed_safe.astype(np.float32),
        local_traversable_ratio=local_traversable,
        clearance_margin_norm=np.ascontiguousarray(
            arrays[f"{platform_type.lower()}_clearance_margin_norm"],
            dtype=np.float32,
        ),
        source=f"cpp_v3/{physical_capability_sha256}",
    )
    safe_cells = np.ascontiguousarray(
        np.column_stack(np.nonzero(observed_safe)), dtype=np.int32
    ).reshape((-1, 2))
    physical = PlatformCandidateReachability(
        platform_type=platform_type,
        canvas=world.canvas,
        pose_map=pose,
        observed_elevation_m=world.elevation_m,
        bridge=bridge,
        request=request,
        maximum_edge_distance_m=(None if platform_type == "HOPPER" else 30.0),
        local_traversability_projection=(
            None if platform_type == "HOPPER" else local_cpp
        ),
    ).project_physical(safe_cells)
    universe = CandidateBuilderV2(sensor_state).build_physical_universe(
        world,
        mission,
        pose,
        projection,
        physical_reachability=physical,
        platform_type=platform_type,
        platform_id=capability.platform_id,
        capability_content_sha256=physical_capability_sha256,
        mission_revision=1,
        evidence_generation=sensor_state.evidence_generation,
        physical_evidence_sha256=sensor_state.physical_evidence_sha256(),
        physical_reachability_algorithm_id=(
            physical.physical_reachability_algorithm_id
        ),
        goal_tolerance_mm=(0 if platform_type == "HOPPER" else 200),
    )
    return physical, universe.diagnostics.physical_candidate_universe_count


def qualify_initial_start(
    *,
    scene: MultiResolutionScene,
    arrays: Mapping[str, np.ndarray],
    capability: FrozenPlatformCapability,
) -> FormalStartQualification | None:
    """Find one exact start whose initial 30 m reveal yields an action candidate."""
    if not isinstance(scene, MultiResolutionScene):
        raise TypeError("formal start qualification requires a multires scene")
    platform_type = capability.platform_type
    if platform_type not in PLATFORMS:
        raise ValueError("formal start capability platform is invalid")
    mission_roi = build_formal_mission_roi(arrays)
    if not bool(mission_roi.any()):
        return None
    mission = MissionRaster(
        scene.base_canvas,
        priority=mission_roi.astype(np.float32),
        roi_ratio=mission_roi.astype(np.float32),
    )
    elevation = np.asarray(arrays["elevation_m"], dtype=np.float32)
    for start_cell in formal_safe_start_cells(arrays, platform_type):
        x_m, y_m = scene.base_canvas.grid_center_world(*start_cell)
        pose = Pose2(
            x_m, y_m, 0.0, "map", float(elevation[start_cell])
        )
        exact_start: tuple[float, float, float] | None = None
        try:
            physical, candidate_count = _qualify_at_pose(
                scene=scene,
                arrays=arrays,
                capability=capability,
                mission=mission,
                start_cell=start_cell,
                pose=pose,
                exact_hopper_start_position_m=None,
            )
            if platform_type == "HOPPER":
                provisional = _physical_position_at_cell(physical, start_cell)
                if provisional is None:
                    continue
                exact_start = _canonical_position_m(provisional)
                pose = Pose2(
                    exact_start[0],
                    exact_start[1],
                    0.0,
                    "map",
                    exact_start[2],
                )
                physical, candidate_count = _qualify_at_pose(
                    scene=scene,
                    arrays=arrays,
                    capability=capability,
                    mission=mission,
                    start_cell=start_cell,
                    pose=pose,
                    exact_hopper_start_position_m=exact_start,
                )
                confirmed = _physical_position_at_cell(physical, start_cell)
                if (
                    confirmed is None
                    or _canonical_position_m(confirmed) != exact_start
                ):
                    continue
        except RuntimeError as error:
            if native_start_failure_is_ineligible(platform_type, error):
                continue
            raise
        if candidate_count > 0:
            return FormalStartQualification(
                start_cell,
                candidate_count,
                exact_start_position_m=exact_start,
            )
    return None


def qualify_initial_start_cell(
    *,
    scene: MultiResolutionScene,
    arrays: Mapping[str, np.ndarray],
    capability: FrozenPlatformCapability,
) -> tuple[int, int] | None:
    """Compatibility wrapper returning only the deterministic fixed cell."""
    result = qualify_initial_start(
        scene=scene, arrays=arrays, capability=capability
    )
    return None if result is None else result.cell


__all__ = [
    "FORMAL_BOUNDARY_MARGIN_CELLS",
    "FormalStartQualification",
    "build_formal_mission_roi",
    "formal_safe_start_cells",
    "qualify_initial_start",
    "qualify_initial_start_cell",
]

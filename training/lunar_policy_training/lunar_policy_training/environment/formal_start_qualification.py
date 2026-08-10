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
from .multires_observation import MultiresSensorObservationState
from .observation_builder import MissionRaster, Pose2
from .primitive_reachability import ObservedPrimitiveReachability


FORMAL_BOUNDARY_MARGIN_CELLS = math.ceil(
    (FORMAL_SENSOR_RANGE_M + LOCAL_GEOMETRY.size_m / 2.0)
    / GLOBAL_GEOMETRY.resolution_m
)


@dataclass(frozen=True, slots=True)
class FormalStartQualification:
    cell: tuple[int, int]
    initial_candidate_count: int

    def __post_init__(self) -> None:
        if (
            len(self.cell) != 2
            or any(type(value) is not int or value < 0 for value in self.cell)
            or type(self.initial_candidate_count) is not int
            or self.initial_candidate_count <= 0
        ):
            raise ValueError("formal start qualification is invalid")


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


def qualify_initial_start(
    *,
    scene: MultiResolutionScene,
    arrays: Mapping[str, np.ndarray],
    capability: FrozenPlatformCapability,
) -> FormalStartQualification | None:
    """Find one exact start whose initial 30 m reveal yields an action candidate.

    Qualification intentionally stops before policy tensor construction and final C++
    path planning. It runs the same observed-only primitive graph and graph-first
    candidate enumeration used by an episode's first decision boundary.
    """
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
            x_m,
            y_m,
            0.0,
            "map",
            float(elevation[start_cell]),
        )
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
            _global_composed_capability,
            _projection_request,
        )
        from .formal_builder import _aggregate_primitive_detail, _grid_map
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
        primitive_detail = (
            planning_detail
            if platform_type == "HOPPER"
            else _aggregate_primitive_detail(planning_detail)
        )
        primitive_local_map = _grid_map(
            canvas=primitive_detail.canvas,
            frame_id="odom",
            elevation_m=primitive_detail.elevation_m,
            valid_mask=primitive_detail.valid_mask,
            physical_obstacle_ratio=primitive_detail.physical_obstacle_ratio,
            physical_obstacle_height_m=(
                primitive_detail.physical_obstacle_height_m
            ),
            forbidden_ratio=primitive_detail.forbidden_ratio,
            observation_age_s=primitive_detail.observation_age_s,
            observation_quality=primitive_detail.observation_quality,
            observation_count=primitive_detail.observation_count,
            stamp_ns=stamp_ns,
        )
        request = _projection_request(
            capability,
            scene,
            scene.project(scene.base_canvas),
            start_cell=start_cell,
        )
        request.request_id = (
            f"formal-observed/{platform_type.lower()}/"
            f"{scene.scene_id}/qualification/1"
        )
        request.global_map_generation = 1
        request.local_map_generation = 1
        request.state_time.nanoseconds_since_epoch = stamp_ns
        request.world.global_map = global_map
        request.world.local_map = primitive_local_map
        request.world.map_from_odom.stamp.nanoseconds_since_epoch = stamp_ns
        request.capability = _global_composed_capability(
            capability,
            resolution_m=2.0,
            bridge_api=bridge_api,
        )
        request.config.wheel.xy_resolution_m = 2.0
        request.config.wheel.yaw_bin_count = 32
        request.config.legged.xy_resolution_m = 2.0
        request.config.legged.yaw_bin_count = 32
        primitive_graph = ObservedPrimitiveReachability(platform_type).update(
            request,
            observation_revision=1,
        )
        candidates = CandidateBuilderV2(sensor_state).build_from_primitive_graph(
            world,
            mission,
            pose,
            primitive_graph,
            platform_type=platform_type,
        )
        if candidates.count > 0:
            return FormalStartQualification(start_cell, candidates.count)
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

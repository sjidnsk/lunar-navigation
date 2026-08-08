"""Exact, planner-free qualification of actionable formal episode starts."""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np

from ..capability_freeze import FrozenPlatformCapability, PLATFORMS
from ..polar_data.multires_scene import MultiResolutionScene, SceneTileProvider
from ..polar_data.raster import GLOBAL_GEOMETRY, LOCAL_GEOMETRY
from ..training_semantics import FORMAL_SENSOR_FOV_RAD, FORMAL_SENSOR_RANGE_M
from .candidate_builder import CandidateBuilderV2
from .multires_observation import MultiresSensorObservationState
from .observation_builder import MissionRaster, PlatformProjection, Pose2
from .visibility import NativeVisibilityEstimator, SensorGeometry


FORMAL_BOUNDARY_MARGIN_CELLS = math.ceil(
    (FORMAL_SENSOR_RANGE_M + LOCAL_GEOMETRY.size_m / 2.0)
    / GLOBAL_GEOMETRY.resolution_m
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


def qualify_initial_start_cell(
    *,
    scene: MultiResolutionScene,
    arrays: Mapping[str, np.ndarray],
    capability: FrozenPlatformCapability,
) -> tuple[int, int] | None:
    """Find one exact start whose initial 30 m reveal yields an action candidate.

    Qualification intentionally stops before policy tensor construction and C++ path
    planning. Candidate feasibility depends only on the observed map, mission ROI, and
    cached C++ traversability projection, so this is equivalent to the episode's
    initial candidate mask while remaining cheap enough for cache materialization.
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
    candidate_builder = CandidateBuilderV2(
        NativeVisibilityEstimator(
            SensorGeometry(FORMAL_SENSOR_RANGE_M, FORMAL_SENSOR_FOV_RAD),
            resolution_m=GLOBAL_GEOMETRY.resolution_m,
        )
    )
    prefix = platform_type.lower()
    projection = PlatformProjection(
        canvas=scene.base_canvas,
        traversable_ratio=(
            np.asarray(arrays[f"{prefix}_hard_feasible"], dtype=np.bool_)
            & mission_roi
        ).astype(np.float32),
        local_traversable_ratio=np.zeros(
            (LOCAL_GEOMETRY.cells, LOCAL_GEOMETRY.cells), dtype=np.float32
        ),
        clearance_margin_norm=np.asarray(
            arrays[f"{prefix}_clearance_margin_norm"], dtype=np.float32
        ),
        source=f"cpp_v3/{capability.content_sha256}",
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
        candidates = candidate_builder.build(world, mission, pose, projection)
        if candidates.count > 0:
            return start_cell
    return None


__all__ = [
    "FORMAL_BOUNDARY_MARGIN_CELLS",
    "build_formal_mission_roi",
    "formal_safe_start_cells",
    "qualify_initial_start_cell",
]

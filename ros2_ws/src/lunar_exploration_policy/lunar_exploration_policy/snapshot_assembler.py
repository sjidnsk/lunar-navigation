"""把规范 ROS 栅格和任务状态组装成 fed9 旧观测快照。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol

import numpy as np

from lunar_policy_training.environment.candidate_builder import CandidateBuilderV2
from lunar_policy_training.environment.observation_builder import (
    LocalObservation,
    MissionRaster,
    ObservedWorld,
    PlatformProjection,
    Pose2,
)
from lunar_policy_training.polar_data.hazards import CanvasRatioLayer
from lunar_policy_training.polar_data.raster import MapCanvas

from .grid_map_runtime import DecodedGridMap
from .identity import DecisionIdentity
from .observation_runtime import Fed9ObservationRuntime, ObservationSnapshot


class SnapshotAssemblyError(ValueError):
    """输入消息虽可解码，但不能形成旧策略要求的同一快照。"""


@dataclass(frozen=True, slots=True)
class MissionDefinition:
    mission_id: str
    revision: int
    roi_bounds_m: tuple[float, float, float, float]


class TraversabilityProjector(Protocol):
    def project(
        self,
        world: ObservedWorld,
        platform_type: str,
        *,
        global_map: DecodedGridMap,
        local_map: DecodedGridMap,
    ) -> PlatformProjection: ...


def _north_up(grid: DecodedGridMap, name: str) -> np.ndarray:
    return np.ascontiguousarray(np.flipud(grid.layers[name]))


def _center_crop(values: np.ndarray, cells: int = 32) -> np.ndarray:
    if values.shape[0] < cells or values.shape[1] < cells:
        raise SnapshotAssemblyError("local map is smaller than 32x32")
    row = (values.shape[0] - cells) // 2
    column = (values.shape[1] - cells) // 2
    return np.ascontiguousarray(values[row : row + cells, column : column + cells])


def _mission_ratio(
    canvas: MapCanvas, bounds: tuple[float, float, float, float]
) -> np.ndarray:
    left, bottom, right, top = bounds
    canvas_left, canvas_bottom, canvas_right, canvas_top = canvas.bounds_m
    if (
        not all(math.isfinite(value) for value in bounds)
        or left >= right
        or bottom >= top
        or left < canvas_left
        or bottom < canvas_bottom
        or right > canvas_right
        or top > canvas_top
    ):
        raise SnapshotAssemblyError("mission ROI is outside the global canvas")
    resolution = canvas.geometry.resolution_m
    columns = np.arange(canvas.geometry.cells, dtype=np.float64)
    cell_left = canvas_left + columns * resolution
    overlap_x = np.clip(
        np.minimum(cell_left + resolution, right) - np.maximum(cell_left, left),
        0.0,
        resolution,
    )
    rows = np.arange(canvas.geometry.cells, dtype=np.float64)
    cell_top = canvas_top - rows * resolution
    overlap_y = np.clip(
        np.minimum(cell_top, top) - np.maximum(cell_top - resolution, bottom),
        0.0,
        resolution,
    )
    return np.ascontiguousarray(
        np.outer(overlap_y, overlap_x) / (resolution * resolution),
        dtype=np.float32,
    )


class SnapshotAssembler:
    """固定 4 m 全局、0.2 m 局部输入的旧语义组装器。"""

    def __init__(
        self,
        candidate_builder: CandidateBuilderV2,
        projector: TraversabilityProjector,
    ) -> None:
        self._runtime = Fed9ObservationRuntime(candidate_builder)
        if not hasattr(projector, "project"):
            raise TypeError("projector must provide project()")
        self._projector = projector

    def build(
        self,
        *,
        global_map: DecodedGridMap,
        local_map: DecodedGridMap,
        pose_map: Pose2,
        robot_state_id: str,
        state_time_ns: int,
        mission: MissionDefinition,
        platform_type: str,
    ) -> ObservationSnapshot:
        if (
            global_map.frame_id != "map"
            or global_map.width != 256
            or global_map.height != 256
            or not math.isclose(global_map.resolution_m, 4.0)
        ):
            raise SnapshotAssemblyError("global map must be map/256x256/4m")
        if local_map.frame_id != "odom" or not math.isclose(local_map.resolution_m, 0.2):
            raise SnapshotAssemblyError("local map must be odom/0.2m")
        if pose_map.frame_id != "map":
            raise SnapshotAssemblyError("pose must already be resolved in map")
        bounds = (
            global_map.origin_xy_m[0],
            global_map.origin_xy_m[1],
            global_map.origin_xy_m[0] + global_map.width * global_map.resolution_m,
            global_map.origin_xy_m[1] + global_map.height * global_map.resolution_m,
        )
        canvas = MapCanvas(global_map.content_id, bounds)
        observed = _north_up(global_map, "valid_mask").astype(np.bool_)
        elevation = _north_up(global_map, "elevation").astype(np.float32)
        obstacle = np.maximum(
            _north_up(global_map, "obstacle"),
            _north_up(global_map, "forbidden"),
        ).astype(np.float32)
        half = 3.2
        local = LocalObservation(
            canvas.identity,
            (pose_map.x_m - half, pose_map.y_m - half, pose_map.x_m + half, pose_map.y_m + half),
            _center_crop(_north_up(local_map, "elevation")).astype(np.float32),
            _center_crop(_north_up(local_map, "valid_mask")).astype(np.bool_),
            _center_crop(
                np.maximum(
                    _north_up(local_map, "obstacle"),
                    _north_up(local_map, "forbidden"),
                )
            ).astype(np.float32),
        )
        world = ObservedWorld(
            canvas,
            elevation,
            observed,
            CanvasRatioLayer(canvas, obstacle),
            local,
        )
        roi = _mission_ratio(canvas, mission.roi_bounds_m)
        mission_raster = MissionRaster(canvas, roi.copy(), roi)
        projection = self._projector.project(
            world,
            platform_type,
            global_map=global_map,
            local_map=local_map,
        )
        identity = DecisionIdentity(
            mission_revision=mission.revision,
            map_snapshot_id=f"{global_map.content_id}:{local_map.content_id}",
            robot_state_id=robot_state_id,
            state_time_ns=state_time_ns,
            execution_state="DECISION_BOUNDARY",
        )
        return self._runtime.build(
            world,
            mission_raster,
            pose_map,
            projection,
            platform_type,
            identity,
        )


__all__ = [
    "MissionDefinition",
    "SnapshotAssembler",
    "SnapshotAssemblyError",
]

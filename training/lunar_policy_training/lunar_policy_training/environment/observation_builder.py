"""Observed-only construction of the active V3 policy observation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

import numpy as np

from lunar_model_contract import ObservationContractV3, validate_observation_inputs
from lunar_model_contract.observation import PLATFORM_CONTEXTS

from ..polar_data.hazards import CanvasRatioLayer
from ..polar_data.raster import GLOBAL_GEOMETRY, LOCAL_GEOMETRY, MapCanvas

if TYPE_CHECKING:
    from .candidate_builder import CandidateBatch


class TransformUnavailable(ValueError):
    """The required map/odom transform was not supplied."""


def _grid(name: str, value: np.ndarray, cells: int, *, finite: bool = True) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (cells, cells):
        raise ValueError(f"{name} must have shape [{cells},{cells}]")
    if finite and not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def _ratio(name: str, value: np.ndarray, cells: int) -> np.ndarray:
    array = _grid(name, value, cells)
    if ((array < 0.0) | (array > 1.0)).any():
        raise ValueError(f"{name} must be in [0,1]")
    return array


@dataclass(frozen=True)
class Pose2:
    x_m: float
    y_m: float
    yaw_rad: float = 0.0
    frame_id: str = "map"
    elevation_m: float = 0.0


def resolve_map_pose(pose: Pose2, map_from_odom: Pose2 | None) -> Pose2:
    if pose.frame_id == "map":
        return pose
    if pose.frame_id != "odom" or map_from_odom is None or map_from_odom.frame_id != "map":
        raise TransformUnavailable("map/odom transform is unavailable")
    cosine, sine = math.cos(map_from_odom.yaw_rad), math.sin(map_from_odom.yaw_rad)
    return Pose2(map_from_odom.x_m + cosine * pose.x_m - sine * pose.y_m, map_from_odom.y_m + sine * pose.x_m + cosine * pose.y_m, map_from_odom.yaw_rad + pose.yaw_rad, "map", pose.elevation_m)


@dataclass(frozen=True)
class ElevationReference:
    """Frozen training/mission global elevation reference in metres."""

    global_reference_m: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.global_reference_m):
            raise ValueError("global elevation reference must be finite")


@dataclass(frozen=True)
class LocalObservation:
    canvas_id: str
    bounds_m: tuple[float, float, float, float]
    elevation_m: np.ndarray
    observed_mask: np.ndarray
    physical_obstacle_ratio: np.ndarray

    def __post_init__(self) -> None:
        left, bottom, right, top = self.bounds_m
        if not math.isclose(right - left, LOCAL_GEOMETRY.size_m) or not math.isclose(top - bottom, LOCAL_GEOMETRY.size_m):
            raise ValueError(
                "local bounds must be a map-axis "
                f"{LOCAL_GEOMETRY.size_m:g}m square"
            )
        elevation = _grid("local_elevation_m", self.elevation_m, LOCAL_GEOMETRY.cells, finite=False)
        observed = np.asarray(self.observed_mask, dtype=bool)
        if observed.shape != elevation.shape or not np.isfinite(elevation[observed]).all():
            raise ValueError("local NoData cannot be marked observed")
        object.__setattr__(self, "elevation_m", elevation)
        object.__setattr__(self, "observed_mask", observed)
        object.__setattr__(self, "physical_obstacle_ratio", _ratio("local_physical_obstacle_ratio", self.physical_obstacle_ratio, LOCAL_GEOMETRY.cells))


@dataclass(frozen=True)
class ObservedWorld:
    canvas: MapCanvas
    elevation_m: np.ndarray
    observed_mask: np.ndarray
    physical_obstacle_layer: CanvasRatioLayer
    local: LocalObservation

    def __post_init__(self) -> None:
        elevation = _grid("elevation_m", self.elevation_m, GLOBAL_GEOMETRY.cells, finite=False)
        observed = np.asarray(self.observed_mask, dtype=bool)
        if observed.shape != elevation.shape or not np.isfinite(elevation[observed]).all():
            raise ValueError("NoData cannot be marked observed")
        if self.local.canvas_id != self.canvas.identity:
            raise ValueError("local input canvas identity does not match observed world")
        if not isinstance(self.physical_obstacle_layer, CanvasRatioLayer):
            raise ValueError("physical obstacle input must be a canvas-bound ratio layer")
        if self.physical_obstacle_layer.canvas.identity != self.canvas.identity:
            raise ValueError("physical obstacle layer canvas identity does not match observed world")
        if self.physical_obstacle_layer.values.shape != elevation.shape:
            raise ValueError("physical obstacle layer geometry does not match observed world")
        object.__setattr__(self, "elevation_m", elevation)
        object.__setattr__(self, "observed_mask", observed)


@dataclass(frozen=True)
class MissionRaster:
    canvas: MapCanvas
    priority: np.ndarray
    roi_ratio: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "priority", _ratio("mission priority", self.priority, GLOBAL_GEOMETRY.cells))
        object.__setattr__(self, "roi_ratio", _ratio("mission ROI", self.roi_ratio, GLOBAL_GEOMETRY.cells))


@dataclass(frozen=True)
class PlatformProjection:
    canvas: MapCanvas
    traversable_ratio: np.ndarray
    local_traversable_ratio: np.ndarray
    clearance_margin_norm: np.ndarray
    source: str

    def __post_init__(self) -> None:
        if self.source != "test_only/proxy" and not self.source.startswith("cpp_v3/"):
            raise ValueError("projection source must be test_only/proxy or cpp_v3/")
        object.__setattr__(self, "traversable_ratio", _ratio("traversable_ratio", self.traversable_ratio, GLOBAL_GEOMETRY.cells))
        object.__setattr__(self, "local_traversable_ratio", _ratio("local_traversable_ratio", self.local_traversable_ratio, LOCAL_GEOMETRY.cells))
        object.__setattr__(self, "clearance_margin_norm", _ratio("clearance_margin_norm", self.clearance_margin_norm, GLOBAL_GEOMETRY.cells))


class ObservationBuilderV2:
    def __init__(self, elevation_reference: ElevationReference = ElevationReference(0.0)) -> None:
        self._reference = elevation_reference

    def build(self, world: ObservedWorld, mission: MissionRaster, pose_map: Pose2, projection: PlatformProjection, candidates: CandidateBatch, platform_type: str) -> dict[str, np.ndarray]:
        if not isinstance(world, ObservedWorld):
            raise ValueError("network builder requires ObservedWorld, never WorldTruth")
        if pose_map.frame_id != "map":
            raise TransformUnavailable("pose_map must be in map frame")
        if world.canvas != mission.canvas or world.canvas != projection.canvas:
            raise ValueError("observed world, mission and projection must share canvas identity")
        if candidates.count and candidates.canvas_id != world.canvas.identity:
            raise ValueError("candidate batch canvas identity does not match observation canvas")
        left, bottom, right, top = world.local.bounds_m
        if not (math.isclose((left + right) / 2.0, pose_map.x_m) and math.isclose((bottom + top) / 2.0, pose_map.y_m)):
            raise ValueError("local map-axis bounds must be centered on resolved robot pose")
        if platform_type not in PLATFORM_CONTEXTS or candidates.features.shape != (64, 12) or candidates.mask.shape != (64,):
            raise ValueError("invalid platform or candidate batch")
        observed = world.observed_mask
        elevation = np.where(observed, world.elevation_m - self._reference.global_reference_m, 0.0).astype(np.float32)
        local = world.local
        prior = np.stack((elevation, mission.priority, np.where(observed, world.physical_obstacle_layer.values, 0.0), np.where(observed, projection.traversable_ratio, 0.0)), axis=0)[None].astype(np.float32)
        coverage = np.stack((observed.astype(np.float32), mission.roi_ratio, mission.priority * mission.roi_ratio * (~observed)), axis=0)[None].astype(np.float32)
        local_crop = np.stack((np.where(local.observed_mask, local.elevation_m - pose_map.elevation_m, 0.0), local.observed_mask.astype(np.float32), np.where(local.observed_mask, local.physical_obstacle_ratio, 0.0), np.where(local.observed_mask, projection.local_traversable_ratio, 0.0)), axis=0)[None].astype(np.float32)
        total = float(mission.roi_ratio.sum())
        observed_ratio = float((observed * mission.roi_ratio).sum() / total) if total else 0.0
        canvas = world.canvas
        pose_features = np.asarray([[(pose_map.x_m - canvas.bounds_m[0]) / canvas.geometry.size_m, (canvas.bounds_m[3] - pose_map.y_m) / canvas.geometry.size_m, math.sin(pose_map.yaw_rad), math.cos(pose_map.yaw_rad), observed_ratio]], dtype=np.float32)
        result = {"prior_channels": prior, "coverage_summary": coverage, "local_crop": local_crop, "frontier_features": candidates.features[None], "pose_features": pose_features, "candidate_mask": candidates.mask[None], "platform_context": np.asarray([PLATFORM_CONTEXTS[platform_type]], dtype=np.float32)}
        for name in ("coverage_summary",):
            if ((result[name] < 0.0) | (result[name] > 1.0)).any():
                raise ValueError(f"{name} must be in [0,1]")
        validate_observation_inputs(result)
        return result


__all__ = ["ElevationReference", "LocalObservation", "MissionRaster", "ObservationBuilderV2", "ObservedWorld", "PlatformProjection", "Pose2", "TransformUnavailable", "resolve_map_pose"]

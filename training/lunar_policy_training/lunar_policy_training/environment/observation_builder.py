"""Observed-only construction of the frozen V2 policy observation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

import numpy as np

from lunar_model_contract import ObservationContractV2, validate_observation_inputs
from lunar_model_contract.observation import PLATFORM_CONTEXTS

from ..polar_data.raster import GLOBAL_GEOMETRY, LOCAL_GEOMETRY

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


@dataclass(frozen=True)
class Pose2:
    x_m: float
    y_m: float
    yaw_rad: float = 0.0
    frame_id: str = "map"


def resolve_map_pose(pose: Pose2, map_from_odom: Pose2 | None) -> Pose2:
    """Resolve an odom pose only when a map/odom transform is explicitly present."""
    if pose.frame_id == "map":
        return pose
    if pose.frame_id != "odom" or map_from_odom is None or map_from_odom.frame_id != "map":
        raise TransformUnavailable("map/odom transform is unavailable")
    cosine, sine = math.cos(map_from_odom.yaw_rad), math.sin(map_from_odom.yaw_rad)
    return Pose2(
        map_from_odom.x_m + cosine * pose.x_m - sine * pose.y_m,
        map_from_odom.y_m + sine * pose.x_m + cosine * pose.y_m,
        map_from_odom.yaw_rad + pose.yaw_rad,
        "map",
    )


@dataclass(frozen=True)
class ObservedWorld:
    """Only sensor-observed world state; it intentionally has no full DEM field."""

    elevation_m: np.ndarray
    observed_mask: np.ndarray
    physical_obstacle_ratio: np.ndarray
    local_elevation_m: np.ndarray | None = None
    local_observed_mask: np.ndarray | None = None
    local_physical_obstacle_ratio: np.ndarray | None = None

    def __post_init__(self) -> None:
        elevation = _grid("elevation_m", self.elevation_m, GLOBAL_GEOMETRY.cells, finite=False)
        observed = np.asarray(self.observed_mask, dtype=bool)
        if observed.shape != elevation.shape:
            raise ValueError("observed_mask must have global raster shape")
        if not np.isfinite(elevation[observed]).all():
            raise ValueError("NoData cannot be marked observed")
        obstacle = _grid("physical_obstacle_ratio", self.physical_obstacle_ratio, GLOBAL_GEOMETRY.cells)
        if ((obstacle < 0.0) | (obstacle > 1.0)).any():
            raise ValueError("physical_obstacle_ratio must be a ratio")
        object.__setattr__(self, "elevation_m", elevation)
        object.__setattr__(self, "observed_mask", observed)
        object.__setattr__(self, "physical_obstacle_ratio", obstacle)
        local_values = (self.local_elevation_m, self.local_observed_mask, self.local_physical_obstacle_ratio)
        if any(value is not None for value in local_values) and any(value is None for value in local_values):
            raise ValueError("local observed fields must be supplied together")
        if self.local_elevation_m is not None:
            local_elevation = _grid("local_elevation_m", self.local_elevation_m, LOCAL_GEOMETRY.cells, finite=False)
            local_observed = np.asarray(self.local_observed_mask, dtype=bool)
            if local_observed.shape != local_elevation.shape or not np.isfinite(local_elevation[local_observed]).all():
                raise ValueError("local NoData cannot be marked observed")
            local_obstacle = _grid("local_physical_obstacle_ratio", self.local_physical_obstacle_ratio, LOCAL_GEOMETRY.cells)
            object.__setattr__(self, "local_elevation_m", local_elevation)
            object.__setattr__(self, "local_observed_mask", local_observed)
            object.__setattr__(self, "local_physical_obstacle_ratio", local_obstacle)


@dataclass(frozen=True)
class MissionRaster:
    priority: np.ndarray
    roi_ratio: np.ndarray
    remaining_decision_budget_ratio: float

    def __post_init__(self) -> None:
        priority = _grid("mission priority", self.priority, GLOBAL_GEOMETRY.cells)
        roi = _grid("mission ROI", self.roi_ratio, GLOBAL_GEOMETRY.cells)
        if (priority < 0.0).any() or (roi < 0.0).any() or (roi > 1.0).any():
            raise ValueError("mission priority must be non-negative and ROI must be a ratio")
        if not 0.0 <= self.remaining_decision_budget_ratio <= 1.0:
            raise ValueError("remaining decision budget ratio must be in [0,1]")
        object.__setattr__(self, "priority", priority)
        object.__setattr__(self, "roi_ratio", roi)


@dataclass(frozen=True)
class PlatformProjection:
    """Narrow C++ projection injection; Python never derives slope or clearance."""

    traversable_ratio: np.ndarray
    local_traversable_ratio: np.ndarray
    clearance_margin_norm: np.ndarray
    source: str

    def __post_init__(self) -> None:
        if self.source != "test_only/proxy" and not self.source.startswith("cpp_v3/"):
            raise ValueError("projection source must be test_only/proxy or cpp_v3/")
        traversable = _grid("traversable_ratio", self.traversable_ratio, GLOBAL_GEOMETRY.cells)
        local = _grid("local_traversable_ratio", self.local_traversable_ratio, LOCAL_GEOMETRY.cells)
        clearance = _grid("clearance_margin_norm", self.clearance_margin_norm, GLOBAL_GEOMETRY.cells)
        if ((traversable < 0.0) | (traversable > 1.0)).any() or ((local < 0.0) | (local > 1.0)).any():
            raise ValueError("traversability must be ratios")
        if ((clearance < 0.0) | (clearance > 1.0)).any():
            raise ValueError("clearance margin must be normalized to [0,1]")
        object.__setattr__(self, "traversable_ratio", traversable)
        object.__setattr__(self, "local_traversable_ratio", local)
        object.__setattr__(self, "clearance_margin_norm", clearance)


class ObservationBuilderV2:
    """Build one finite, exact-contract NumPy batch from observed state only."""

    def build(self, world: ObservedWorld, mission: MissionRaster, pose_map: Pose2, projection: PlatformProjection, candidates: CandidateBatch, platform_type: str) -> dict[str, np.ndarray]:
        if not isinstance(world, ObservedWorld):
            raise ValueError("network builder requires ObservedWorld, never WorldTruth")
        if pose_map.frame_id != "map":
            raise TransformUnavailable("pose_map must be in map frame")
        if platform_type not in PLATFORM_CONTEXTS:
            raise ValueError("unknown platform type")
        if candidates.features.shape != (64, 12) or candidates.mask.shape != (64,):
            raise ValueError("candidate batch must have 64 exact contract slots")
        observed = world.observed_mask
        elevation = np.where(observed, world.elevation_m, 0.0).astype(np.float32)
        obstacle = np.where(observed, world.physical_obstacle_ratio, 0.0).astype(np.float32)
        traversable = np.where(observed, projection.traversable_ratio, 0.0).astype(np.float32)
        prior = np.stack((elevation, mission.priority, obstacle, traversable), axis=0)[None, ...].astype(np.float32)
        coverage = np.stack((observed.astype(np.float32), mission.roi_ratio, mission.priority * mission.roi_ratio * (~observed)), axis=0)[None, ...].astype(np.float32)
        if world.local_elevation_m is None:
            raise ValueError("true local observed raster is required; global map cannot be promoted to 0.25 m")
        local_elevation, local_observed, local_obstacle = world.local_elevation_m, world.local_observed_mask, world.local_physical_obstacle_ratio
        local = np.stack((
            np.where(local_observed, local_elevation, 0.0), local_observed.astype(np.float32),
            np.where(local_observed, local_obstacle, 0.0), np.where(local_observed, projection.local_traversable_ratio, 0.0),
        ), axis=0)[None, ...].astype(np.float32)
        roi_total = float(mission.roi_ratio.sum())
        observed_ratio = float((observed * mission.roi_ratio).sum() / roi_total) if roi_total else 0.0
        pose_features = np.asarray([[
            (pose_map.x_m - GLOBAL_GEOMETRY.size_m / 2.0) / (GLOBAL_GEOMETRY.size_m / 2.0),
            (pose_map.y_m - GLOBAL_GEOMETRY.size_m / 2.0) / (GLOBAL_GEOMETRY.size_m / 2.0),
            math.sin(pose_map.yaw_rad), math.cos(pose_map.yaw_rad), observed_ratio, mission.remaining_decision_budget_ratio,
        ]], dtype=np.float32)
        result = {
            "prior_channels": prior,
            "coverage_summary": coverage,
            "local_crop": local,
            "frontier_features": candidates.features[None, ...].astype(np.float32, copy=False),
            "pose_features": pose_features,
            "candidate_mask": candidates.mask[None, ...].astype(np.bool_, copy=False),
            "platform_context": np.asarray([PLATFORM_CONTEXTS[platform_type]], dtype=np.float32),
        }
        # This is deliberately the shared V2 validator, not a copied contract.
        validate_observation_inputs(result)
        return result


__all__ = ["MissionRaster", "ObservationBuilderV2", "ObservedWorld", "PlatformProjection", "Pose2", "TransformUnavailable", "resolve_map_pose"]

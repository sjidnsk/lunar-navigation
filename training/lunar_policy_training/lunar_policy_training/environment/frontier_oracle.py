"""Independent observed-only audit of remaining frontier opportunities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .observation_builder import MissionRaster, ObservedWorld, PlatformProjection
from .platform_reachability import CandidateReachabilityResult
from .visibility import SensorGeometry


@dataclass(frozen=True, slots=True)
class FrontierOracleResult:
    frontier_anchor_count: int
    observed_safe_pose_count: int
    platform_reachable_pose_count: int
    opportunity_count: int

    def __post_init__(self) -> None:
        values = (
            self.frontier_anchor_count,
            self.observed_safe_pose_count,
            self.platform_reachable_pose_count,
            self.opportunity_count,
        )
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError("frontier oracle counts must be non-negative integers")
        if not (
            self.opportunity_count <= self.platform_reachable_pose_count
            <= self.observed_safe_pose_count
            <= self.frontier_anchor_count
        ):
            raise ValueError("frontier oracle stage counts are inconsistent")


class FrontierOpportunityOracle:
    """Scan safe observed frontier poses without production ranking heuristics."""

    def __init__(self, visibility_estimator: object) -> None:
        sensor = getattr(visibility_estimator, "sensor", None)
        estimate = getattr(visibility_estimator, "estimate_candidate_gains", None)
        if not isinstance(sensor, SensorGeometry) or not callable(estimate):
            raise TypeError("frontier oracle requires a visibility estimator")
        self._visibility_estimator = visibility_estimator

    def evaluate(
        self,
        world: ObservedWorld,
        mission: MissionRaster,
        projection: PlatformProjection,
        *,
        platform_reachability: object,
    ) -> FrontierOracleResult:
        if (
            not isinstance(world, ObservedWorld)
            or not isinstance(mission, MissionRaster)
            or not isinstance(projection, PlatformProjection)
            or world.canvas != mission.canvas
            or world.canvas != projection.canvas
        ):
            raise ValueError("frontier oracle map identities differ")
        filter_candidates = getattr(platform_reachability, "filter", None)
        if not callable(filter_candidates):
            raise TypeError("frontier oracle requires platform certification")

        observed = world.observed_mask
        unknown_roi = (mission.roi_ratio > 0.0) & ~observed
        adjacent_unknown = np.zeros_like(observed)
        adjacent_unknown[1:] |= unknown_roi[:-1]
        adjacent_unknown[:-1] |= unknown_roi[1:]
        adjacent_unknown[:, 1:] |= unknown_roi[:, :-1]
        adjacent_unknown[:, :-1] |= unknown_roi[:, 1:]
        frontier = observed & (mission.roi_ratio > 0.0) & adjacent_unknown
        frontier_count = int(frontier.sum(dtype=np.int64))
        safe = (
            frontier
            & (world.physical_obstacle_layer.values == 0.0)
            & (projection.traversable_ratio > 0.0)
        )
        safe_cells = np.ascontiguousarray(
            np.column_stack(np.nonzero(safe)), dtype=np.int32
        ).reshape((-1, 2))
        safe_count = int(safe_cells.shape[0])
        if safe_count == 0:
            return FrontierOracleResult(frontier_count, 0, 0, 0)

        reachability = filter_candidates(safe_cells)
        if not isinstance(reachability, CandidateReachabilityResult):
            raise RuntimeError("frontier oracle platform result is invalid")
        reachable_cells = np.ascontiguousarray(
            safe_cells[reachability.accepted_mask], dtype=np.int32
        )
        reachable_count = int(reachable_cells.shape[0])
        if reachable_count == 0:
            return FrontierOracleResult(
                frontier_count, safe_count, 0, 0
            )

        gains = self._visibility_estimator.estimate_candidate_gains(
            np.ascontiguousarray(observed, dtype=np.bool_),
            np.ascontiguousarray(
                world.physical_obstacle_layer.values, dtype=np.float32
            ),
            np.ascontiguousarray(mission.roi_ratio, dtype=np.float32),
            np.ascontiguousarray(
                mission.priority * mission.roi_ratio, dtype=np.float32
            ),
            reachable_cells,
        )
        if (
            not isinstance(gains, np.ndarray)
            or gains.shape != (reachable_count, 2)
            or gains.dtype != np.dtype(np.float32)
            or not gains.flags.c_contiguous
            or not np.isfinite(gains).all()
            or (gains < 0.0).any()
        ):
            raise RuntimeError("frontier oracle visibility result is invalid")
        opportunity_count = int(
            np.count_nonzero(gains[:, 0] > np.float32(0.0))
        )
        return FrontierOracleResult(
            frontier_count,
            safe_count,
            reachable_count,
            opportunity_count,
        )


__all__ = ["FrontierOpportunityOracle", "FrontierOracleResult"]

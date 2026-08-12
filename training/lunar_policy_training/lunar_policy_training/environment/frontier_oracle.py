"""Independent observed-only audit of remaining physical opportunities."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import math

import numpy as np

from .observation_builder import MissionRaster, ObservedWorld, Pose2
from .platform_reachability import PhysicalReachabilityResult
from .visibility import SensorGeometry


_ORACLE_RANGE_M = 30.0
_EMPTY_OPPORTUNITY_SET_SHA256 = sha256(b"").hexdigest()


def _millimetres(value_m: float) -> int:
    scaled = float(value_m) * 1_000.0
    if not math.isfinite(scaled):
        raise ValueError("frontier oracle position is not finite")
    return int(round(scaled))


@dataclass(frozen=True, slots=True)
class FrontierOracleResult:
    frontier_anchor_count: int
    observed_safe_pose_count: int
    platform_reachable_pose_count: int
    opportunity_count: int
    opportunity_set_sha256: str = _EMPTY_OPPORTUNITY_SET_SHA256

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
        digest = self.opportunity_set_sha256
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("frontier oracle opportunity hash is invalid")

    @property
    def oracle_opportunity_count(self) -> int:
        return self.opportunity_count

    @property
    def oracle_opportunity_set_sha256(self) -> str:
        return self.opportunity_set_sha256


class FrontierOpportunityOracle:
    """Enumerate physical poses without production ranking or reserve logic."""

    def __init__(self, visibility_estimator: object) -> None:
        sensor = getattr(visibility_estimator, "sensor", None)
        estimate = getattr(visibility_estimator, "estimate_candidate_gains", None)
        if not isinstance(sensor, SensorGeometry) or not callable(estimate):
            raise TypeError("frontier oracle requires a visibility estimator")
        if (
            not math.isclose(
                sensor.range_m,
                _ORACLE_RANGE_M,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            or not sensor.is_full_circle
        ):
            raise ValueError("frontier oracle requires fixed 30 m / 360 degree LOS")
        self._visibility_estimator = visibility_estimator

    def evaluate_physical(
        self,
        world: ObservedWorld,
        mission: MissionRaster,
        *,
        pose_map: Pose2,
        physical_reachability: PhysicalReachabilityResult,
    ) -> FrontierOracleResult:
        """Audit positive-gain observed poses from physical authority only."""
        if (
            not isinstance(world, ObservedWorld)
            or not isinstance(mission, MissionRaster)
            or not isinstance(pose_map, Pose2)
            or pose_map.frame_id != "map"
            or not isinstance(physical_reachability, PhysicalReachabilityResult)
            or world.canvas != mission.canvas
        ):
            raise ValueError("frontier oracle map identities differ")
        mask = physical_reachability.physical_observation_pose_mask
        if mask.shape != world.observed_mask.shape:
            raise ValueError("frontier oracle physical geometry differs")
        physical_cells = np.ascontiguousarray(
            np.column_stack(np.nonzero(mask)), dtype=np.int32
        ).reshape((-1, 2))
        positions = physical_reachability.observation_positions_m
        opportunity_authority = (
            physical_reachability.hopper_opportunity_authority
        )
        if opportunity_authority is not None:
            physical_cells = np.ascontiguousarray(
                np.column_stack(
                    np.nonzero(opportunity_authority.certified_mask)
                ),
                dtype=np.int32,
            ).reshape((-1, 2))
            positions = opportunity_authority.certified_positions_m
        if len(physical_cells) != len(positions):
            raise ValueError("frontier oracle physical positions differ")
        for cell, position in zip(physical_cells, positions, strict=True):
            try:
                position_cell = world.canvas.world_to_grid(
                    float(position[0]), float(position[1])
                )
            except ValueError as error:
                raise ValueError(
                    "frontier oracle physical position leaves the canvas"
                ) from error
            if position_cell != tuple(cell):
                raise ValueError(
                    "frontier oracle physical position leaves its row-major cell"
                )

        robot_cell = world.canvas.world_to_grid(pose_map.x_m, pose_map.y_m)
        observed = np.ascontiguousarray(world.observed_mask, dtype=np.bool_)
        anchor_count = int(observed.sum(dtype=np.int64))
        observed_safe = np.ascontiguousarray(
            observed & (world.physical_obstacle_layer.values == 0.0)
        )
        observed_safe[robot_cell] = False
        safe_count = int(observed_safe.sum(dtype=np.int64))
        if safe_count == 0:
            return FrontierOracleResult(anchor_count, 0, 0, 0)
        if len(physical_cells) == 0:
            return FrontierOracleResult(anchor_count, safe_count, 0, 0)
        physical_safe = np.ascontiguousarray(
            [observed_safe[tuple(cell)] for cell in physical_cells],
            dtype=np.bool_,
        )
        safe_cells = np.ascontiguousarray(
            physical_cells[physical_safe], dtype=np.int32
        )
        safe_positions = np.ascontiguousarray(
            positions[physical_safe], dtype=np.float64
        )
        if opportunity_authority is None:
            distance_m = np.hypot(
                safe_positions[:, 0] - pose_map.x_m,
                safe_positions[:, 1] - pose_map.y_m,
            )
            within_range = distance_m <= _ORACLE_RANGE_M + 1.0e-12
        else:
            within_range = np.ones(len(safe_positions), dtype=np.bool_)
        reachable_cells = np.ascontiguousarray(
            safe_cells[within_range], dtype=np.int32
        )
        reachable_positions = np.ascontiguousarray(
            safe_positions[within_range], dtype=np.float64
        )
        reachable_count = len(reachable_cells)
        if reachable_count == 0:
            return FrontierOracleResult(
                anchor_count,
                safe_count,
                0,
                0,
            )

        arguments = (
            np.ascontiguousarray(world.observed_mask, dtype=np.bool_),
            np.ascontiguousarray(
                world.physical_obstacle_layer.values, dtype=np.float32
            ),
            np.ascontiguousarray(mission.roi_ratio, dtype=np.float32),
            np.ascontiguousarray(
                mission.priority * mission.roi_ratio, dtype=np.float32
            ),
        )
        exact_gain = getattr(
            self._visibility_estimator,
            "estimate_candidate_gains_at_positions",
            None,
        )
        gains = (
            exact_gain(*arguments, reachable_positions)
            if callable(exact_gain)
            else self._visibility_estimator.estimate_candidate_gains(
                *arguments, reachable_cells
            )
        )
        if (
            not isinstance(gains, np.ndarray)
            or gains.dtype != np.dtype(np.float32)
            or gains.shape != (reachable_count, 2)
            or not gains.flags.c_contiguous
            or not np.isfinite(gains).all()
            or (gains < 0.0).any()
        ):
            raise RuntimeError("frontier oracle visibility result is invalid")
        positive = gains[:, 0] > np.float32(0.0)
        if opportunity_authority is not None:
            positive_mask = np.zeros_like(
                opportunity_authority.certified_mask, dtype=np.bool_
            )
            positive_cells = reachable_cells[positive]
            if len(positive_cells):
                positive_mask[
                    positive_cells[:, 0], positive_cells[:, 1]
                ] = True
            opportunity = opportunity_authority.query(
                np.ascontiguousarray(positive_mask),
                enumerate_all_reachable_opportunities=True,
            )
            connected_positive = np.ascontiguousarray(
                np.flipud(opportunity.reachable_opportunities),
                dtype=np.bool_,
            )
            positive = np.ascontiguousarray(
                [connected_positive[tuple(cell)] for cell in reachable_cells],
                dtype=np.bool_,
            )
        opportunity_keys = sorted(
            (
                f"{int(cell[0])}:{int(cell[1])}:"
                f"{_millimetres(position[0])}:"
                f"{_millimetres(position[1])}:"
                f"{_millimetres(position[2])}"
            )
            for cell, position in zip(
                reachable_cells[positive],
                reachable_positions[positive],
                strict=True,
            )
        )
        opportunity_hash = sha256(
            "\n".join(opportunity_keys).encode("ascii")
        ).hexdigest()
        return FrontierOracleResult(
            anchor_count,
            safe_count,
            reachable_count,
            len(opportunity_keys),
            opportunity_hash,
        )


__all__ = ["FrontierOpportunityOracle", "FrontierOracleResult"]

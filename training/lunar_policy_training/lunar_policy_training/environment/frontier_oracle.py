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


def _oracle_ground_potential_gain_mask(
    unknown_roi: np.ndarray,
    *,
    sensor_range_m: float,
    resolution_m: float,
) -> np.ndarray:
    """Independently coarse-filter impossible ground observation poses."""
    unknown = np.asarray(unknown_roi)
    if (
        unknown.dtype != np.dtype(np.bool_)
        or unknown.ndim != 2
        or min(unknown.shape) <= 0
        or not unknown.flags.c_contiguous
    ):
        raise ValueError("frontier oracle unknown ROI mask is invalid")
    if (
        not math.isfinite(sensor_range_m)
        or sensor_range_m <= 0.0
        or not math.isfinite(resolution_m)
        or resolution_m <= 0.0
    ):
        raise ValueError("frontier oracle gain prefilter geometry is invalid")
    possible = np.zeros_like(unknown)
    if not unknown.any():
        return possible
    rows, columns = unknown.shape
    radius = math.ceil(sensor_range_m / resolution_m)
    if radius >= max(rows - 1, columns - 1):
        return np.ones_like(unknown)
    for row_offset in range(-min(radius, rows - 1), min(radius, rows - 1) + 1):
        source_row_start = max(0, -row_offset)
        source_row_stop = min(rows, rows - row_offset)
        target_row_start = source_row_start + row_offset
        target_row_stop = source_row_stop + row_offset
        for column_offset in range(
            -min(radius, columns - 1), min(radius, columns - 1) + 1
        ):
            source_column_start = max(0, -column_offset)
            source_column_stop = min(columns, columns - column_offset)
            target_column_start = source_column_start + column_offset
            target_column_stop = source_column_stop + column_offset
            possible[
                target_row_start:target_row_stop,
                target_column_start:target_column_stop,
            ] |= unknown[
                source_row_start:source_row_stop,
                source_column_start:source_column_stop,
            ]
    return np.ascontiguousarray(possible, dtype=np.bool_)


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
        platform_type = physical_reachability.platform_type
        if platform_type in ("WHEELED", "LEGGED"):
            reachable_count = len(safe_cells)
            potential_gain = _oracle_ground_potential_gain_mask(
                np.ascontiguousarray(
                    (mission.roi_ratio > 0.0) & ~observed,
                    dtype=np.bool_,
                ),
                sensor_range_m=_ORACLE_RANGE_M,
                resolution_m=world.canvas.geometry.resolution_m,
            )
            within_range = np.ascontiguousarray(
                [potential_gain[tuple(cell)] for cell in safe_cells],
                dtype=np.bool_,
            )
        elif opportunity_authority is None:
            distance_m = np.hypot(
                safe_positions[:, 0] - pose_map.x_m,
                safe_positions[:, 1] - pose_map.y_m,
            )
            within_range = distance_m <= _ORACLE_RANGE_M + 1.0e-12
            reachable_count = int(within_range.sum(dtype=np.int64))
        else:
            within_range = np.ones(len(safe_positions), dtype=np.bool_)
            reachable_count = len(safe_cells)
        reachable_cells = np.ascontiguousarray(
            safe_cells[within_range], dtype=np.int32
        )
        reachable_positions = np.ascontiguousarray(
            safe_positions[within_range], dtype=np.float64
        )
        if reachable_count == 0 or len(reachable_cells) == 0:
            return FrontierOracleResult(
                anchor_count,
                safe_count,
                reachable_count,
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
            or gains.shape != (len(reachable_cells), 2)
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

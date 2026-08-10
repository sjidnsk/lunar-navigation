"""Independent observed-only audit of remaining frontier opportunities."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
import math

import numpy as np

from .observation_builder import MissionRaster, ObservedWorld, PlatformProjection, Pose2
from .platform_reachability import CandidateReachabilityResult
from .primitive_reachability import ObservedPrimitiveSnapshot
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
    """Scan local safe observation poses without production ranking heuristics."""

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
        projection: PlatformProjection | None = None,
        *,
        pose_map: Pose2,
        platform_reachability: object | None = None,
        primitive_graph: ObservedPrimitiveSnapshot | None = None,
        excluded_cells: Collection[tuple[int, int]] = (),
        excluded_state_ids: Collection[int] = (),
        backtrack_pose: Pose2 | None = None,
    ) -> FrontierOracleResult:
        if primitive_graph is not None:
            return self._evaluate_primitive_graph(
                world,
                mission,
                pose_map=pose_map,
                primitive_graph=primitive_graph,
                excluded_state_ids=excluded_state_ids,
            )
        if (
            not isinstance(world, ObservedWorld)
            or not isinstance(mission, MissionRaster)
            or not isinstance(projection, PlatformProjection)
            or world.canvas != mission.canvas
            or world.canvas != projection.canvas
            or not isinstance(pose_map, Pose2)
            or pose_map.frame_id != "map"
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
        safe = (
            observed
            & (mission.roi_ratio > 0.0)
            & (world.physical_obstacle_layer.values == 0.0)
            & (projection.traversable_ratio > 0.0)
        )
        all_safe_cells = np.ascontiguousarray(
            np.column_stack(np.nonzero(safe)), dtype=np.int32
        ).reshape((-1, 2))
        backtrack_cell = None
        if isinstance(backtrack_pose, Pose2) and backtrack_pose.frame_id == "map":
            try:
                backtrack_cell = world.canvas.world_to_grid(
                    backtrack_pose.x_m, backtrack_pose.y_m
                )
            except ValueError:
                backtrack_cell = None
        excluded = set(excluded_cells)
        if backtrack_cell is not None:
            excluded.discard(backtrack_cell)
        if excluded:
            all_safe_cells = np.ascontiguousarray(
                [
                    cell
                    for cell in map(tuple, all_safe_cells.tolist())
                    if cell not in excluded
                ],
                dtype=np.int32,
            ).reshape((-1, 2))
        robot = world.canvas.world_to_grid(pose_map.x_m, pose_map.y_m)
        target_xy = np.ascontiguousarray(
            [
                (backtrack_pose.x_m, backtrack_pose.y_m)
                if backtrack_cell is not None
                and tuple(cell) == backtrack_cell
                else world.canvas.grid_center_world(*cell)
                for cell in all_safe_cells.tolist()
            ],
            dtype=np.float64,
        ).reshape((-1, 2))
        delta_xy = target_xy - np.asarray(
            [pose_map.x_m, pose_map.y_m], dtype=np.float64
        )
        distance_m = np.linalg.norm(delta_xy, axis=1)
        within_sensor = distance_m <= self._visibility_estimator.sensor.range_m
        if not self._visibility_estimator.sensor.is_full_circle:
            bearings = np.arctan2(delta_xy[:, 1], delta_xy[:, 0])
            relative = (
                bearings - pose_map.yaw_rad + math.pi
            ) % (2.0 * math.pi) - math.pi
            within_sensor &= (
                np.abs(relative)
                <= self._visibility_estimator.sensor.fov_rad / 2.0
            )
        safe_cells = np.ascontiguousarray(
            all_safe_cells[
                np.any(all_safe_cells != np.asarray(robot), axis=1)
                & within_sensor
            ],
            dtype=np.int32,
        )
        safe_count = int(safe_cells.shape[0])
        if safe_count == 0:
            return FrontierOracleResult(0, 0, 0, 0)

        target_positions = None
        if backtrack_cell is not None and backtrack_cell in map(
            tuple, safe_cells.tolist()
        ):
            target_positions = np.ascontiguousarray(
                [
                    (
                        backtrack_pose.x_m,
                        backtrack_pose.y_m,
                        backtrack_pose.elevation_m,
                    )
                    if tuple(cell) == backtrack_cell
                    else (
                        *world.canvas.grid_center_world(*cell),
                        float(world.elevation_m[tuple(cell)]),
                    )
                    for cell in safe_cells.tolist()
                ],
                dtype=np.float64,
            )
        reachability = (
            filter_candidates(safe_cells)
            if target_positions is None
            else filter_candidates(
                safe_cells, target_positions_map=target_positions
            )
        )
        if not isinstance(reachability, CandidateReachabilityResult):
            raise RuntimeError("frontier oracle platform result is invalid")
        reachable_cells = np.ascontiguousarray(
            safe_cells[reachability.accepted_mask], dtype=np.int32
        )
        reachable_count = int(reachable_cells.shape[0])
        if reachable_count == 0:
            return FrontierOracleResult(
                safe_count, safe_count, 0, 0
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
        if opportunity_count == 0 and bool(frontier.any()):
            opportunity_count = reachable_count
        return FrontierOracleResult(
            safe_count,
            safe_count,
            reachable_count,
            opportunity_count,
        )

    def _evaluate_primitive_graph(
        self,
        world: ObservedWorld,
        mission: MissionRaster,
        *,
        pose_map: Pose2,
        primitive_graph: ObservedPrimitiveSnapshot,
        excluded_state_ids: Collection[int],
    ) -> FrontierOracleResult:
        """Audit opportunities from edges without using production labels/sorting."""
        if (
            not isinstance(world, ObservedWorld)
            or not isinstance(mission, MissionRaster)
            or world.canvas != mission.canvas
            or not isinstance(pose_map, Pose2)
            or pose_map.frame_id != "map"
            or not isinstance(primitive_graph, ObservedPrimitiveSnapshot)
        ):
            raise ValueError("primitive oracle map identities differ")
        state_ids = primitive_graph.state_ids
        index_by_id = {
            int(state_id): index for index, state_id in enumerate(state_ids)
        }
        anchors = set(
            int(state_ids[index])
            for index in np.flatnonzero(primitive_graph.path_cost == 0.0)
        )
        if not anchors:
            raise RuntimeError("primitive oracle graph has no anchor")
        outgoing = {int(state_id): [] for state_id in state_ids}
        incoming = {int(state_id): [] for state_id in state_ids}
        for source, target in zip(
            primitive_graph.edge_source_ids,
            primitive_graph.edge_target_ids,
            strict=True,
        ):
            source_id = int(source)
            target_id = int(target)
            outgoing[source_id].append(target_id)
            incoming[target_id].append(source_id)

        def closure(
            seeds: set[int], adjacency: dict[int, list[int]]
        ) -> set[int]:
            reached = set(seeds)
            queue = list(sorted(seeds, reverse=True))
            while queue:
                source = queue.pop()
                for target in sorted(adjacency[source]):
                    if target not in reached:
                        reached.add(target)
                        queue.append(target)
            return reached

        forward = closure(anchors, outgoing)
        returnable = closure(anchors, incoming)
        independently_recoverable = forward & returnable
        excluded = {int(value) for value in excluded_state_ids}
        if any(value < 0 for value in excluded):
            raise ValueError("primitive oracle excluded state is invalid")
        positions = primitive_graph.positions_m
        distance = np.hypot(
            positions[:, 0] - pose_map.x_m,
            positions[:, 1] - pose_map.y_m,
        )
        raw_indices = [
            int(index)
            for index in np.flatnonzero(primitive_graph.observation_state)
            if int(state_ids[index]) not in anchors
        ]
        spatial: list[tuple[int, tuple[int, int]]] = []
        for index in raw_indices:
            if int(state_ids[index]) in excluded:
                continue
            try:
                cell = world.canvas.world_to_grid(
                    float(positions[index, 0]), float(positions[index, 1])
                )
            except ValueError:
                continue
            spatial.append((index, cell))
        direct_hopper_successors = {
            target for anchor in anchors for target in outgoing[anchor]
        }
        reachable = [
            (index, cell)
            for index, cell in spatial
            if int(state_ids[index]) in independently_recoverable
            and distance[index] <= self._visibility_estimator.sensor.range_m + 1.0e-9
            and (
                primitive_graph.platform_type != "HOPPER"
                or int(state_ids[index]) in direct_hopper_successors
            )
        ]
        if not reachable:
            return FrontierOracleResult(
                len(raw_indices), len(spatial), 0, 0
            )
        cells = np.ascontiguousarray(
            [cell for _, cell in reachable], dtype=np.int32
        )
        exact_positions = np.ascontiguousarray(
            [positions[index] for index, _ in reachable], dtype=np.float64
        )
        exact_gain = getattr(
            self._visibility_estimator,
            "estimate_candidate_gains_at_positions",
            None,
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
        gains = (
            exact_gain(*arguments, exact_positions)
            if callable(exact_gain)
            else self._visibility_estimator.estimate_candidate_gains(
                *arguments, cells
            )
        )
        if (
            not isinstance(gains, np.ndarray)
            or gains.shape != (len(reachable), 2)
            or gains.dtype != np.dtype(np.float32)
            or not gains.flags.c_contiguous
            or not np.isfinite(gains).all()
            or (gains < 0.0).any()
        ):
            raise RuntimeError("primitive oracle visibility result is invalid")
        opportunity_count = int(np.count_nonzero(gains[:, 0] > 0.0))
        if opportunity_count == 0 and bool(
            np.any((mission.roi_ratio > 0.0) & ~world.observed_mask)
        ):
            opportunity_count = len(reachable)
        return FrontierOracleResult(
            len(raw_indices),
            len(spatial),
            len(reachable),
            opportunity_count,
        )


__all__ = ["FrontierOpportunityOracle", "FrontierOracleResult"]

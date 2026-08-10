"""Observed-only deterministic frontier candidates for the frozen V2 policy."""

from __future__ import annotations

from collections import deque
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
import math

import numpy as np

from lunar_model_contract import ObservationContractV3

from .observation_builder import MissionRaster, ObservedWorld, PlatformProjection, Pose2
from .platform_reachability import PlatformCandidateReachability
from .primitive_reachability import ObservedPrimitiveSnapshot
from .visibility import SensorGeometry, VisibilityEstimator, _ray_cells


_GROUND_PLATFORM_TYPES = frozenset(("WHEELED", "LEGGED"))
_PLATFORM_TYPES = _GROUND_PLATFORM_TYPES | {"HOPPER"}
_MIN_PLATFORM_CANDIDATE_RESERVE = 8


@dataclass(frozen=True, slots=True)
class CandidateDiagnostics:
    frontier_anchor_count: int = 0
    visited_excluded_count: int = 0
    static_infeasible_count: int = 0
    platform_unreachable_count: int = 0
    zero_gain_count: int = 0
    emitted_count: int = 0
    planner_rejected_count: int = 0

    def __post_init__(self) -> None:
        values = (
            self.frontier_anchor_count,
            self.visited_excluded_count,
            self.static_infeasible_count,
            self.platform_unreachable_count,
            self.zero_gain_count,
            self.emitted_count,
            self.planner_rejected_count,
        )
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in values):
            raise ValueError("candidate diagnostics must contain non-negative integers")
        remaining = self.frontier_anchor_count
        for value in (
            self.visited_excluded_count,
            self.static_infeasible_count,
            self.platform_unreachable_count,
            self.zero_gain_count,
        ):
            if value > remaining:
                raise ValueError("candidate diagnostics stage count exceeds input")
            remaining -= value
        if self.emitted_count > remaining:
            raise ValueError("candidate diagnostics emitted count exceeds survivors")
        if self.planner_rejected_count > self.emitted_count:
            raise ValueError("candidate planner rejection exceeds emitted count")


@dataclass(frozen=True)
class CandidateBatch:
    features: np.ndarray
    mask: np.ndarray
    canvas_id: str | None = None
    diagnostics: CandidateDiagnostics = field(default_factory=CandidateDiagnostics)
    target_elevation_m: np.ndarray | None = None
    target_positions_m: np.ndarray | None = None
    target_yaw_rad: np.ndarray | None = None
    primitive_state_ids: np.ndarray | None = None
    primitive_graph_revision: int = 0

    def __post_init__(self) -> None:
        features, mask = np.asarray(self.features, dtype=np.float32), np.asarray(self.mask, dtype=bool)
        if features.shape != (64, len(ObservationContractV3.frontier_fields)) or mask.shape != (64,) or not np.isfinite(features).all():
            raise ValueError("candidate batch must use finite [64,12] and [64]")
        if not isinstance(self.diagnostics, CandidateDiagnostics):
            raise TypeError("candidate diagnostics are required")
        if self.diagnostics.emitted_count != int(mask.sum()):
            raise ValueError("candidate diagnostics emitted count must match mask")
        target_elevation = self.target_elevation_m
        if target_elevation is None:
            target_elevation = np.zeros(64, dtype=np.float64)
        if (
            not isinstance(target_elevation, np.ndarray)
            or target_elevation.dtype != np.dtype(np.float64)
            or target_elevation.shape != (64,)
            or not target_elevation.flags.c_contiguous
            or not np.isfinite(target_elevation).all()
        ):
            raise ValueError("candidate target elevation must be finite float64 [64]")
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "target_elevation_m", target_elevation)
        target_positions = self.target_positions_m
        if target_positions is None:
            target_positions = np.zeros((64, 3), dtype=np.float64)
        target_positions = np.ascontiguousarray(target_positions)
        target_yaw = self.target_yaw_rad
        if target_yaw is None:
            target_yaw = np.zeros(64, dtype=np.float64)
        target_yaw = np.ascontiguousarray(target_yaw)
        state_ids = self.primitive_state_ids
        if state_ids is None:
            state_ids = np.zeros(64, dtype=np.uint64)
        state_ids = np.ascontiguousarray(state_ids)
        if (
            target_positions.dtype != np.dtype(np.float64)
            or target_positions.shape != (64, 3)
            or not np.isfinite(target_positions).all()
            or target_yaw.dtype != np.dtype(np.float64)
            or target_yaw.shape != (64,)
            or not np.isfinite(target_yaw).all()
            or state_ids.dtype != np.dtype(np.uint64)
            or state_ids.shape != (64,)
            or type(self.primitive_graph_revision) is not int
            or self.primitive_graph_revision < 0
            or (state_ids[~mask] != 0).any()
        ):
            raise ValueError("candidate primitive targets are invalid")
        object.__setattr__(self, "target_positions_m", target_positions)
        object.__setattr__(self, "target_yaw_rad", target_yaw)
        object.__setattr__(self, "primitive_state_ids", state_ids)

    @property
    def count(self) -> int:
        return int(self.mask.sum())

    @classmethod
    def empty(
        cls,
        canvas_id: str | None = None,
        diagnostics: CandidateDiagnostics | None = None,
    ) -> "CandidateBatch":
        return cls(
            np.zeros((64, 12), np.float32),
            np.zeros(64, bool),
            canvas_id,
            diagnostics or CandidateDiagnostics(),
        )


def _neighbors(row: int, column: int, cells: int) -> tuple[tuple[int, int], ...]:
    return tuple((row + dr, column + dc) for dr, dc in ((-1, 0), (0, -1), (0, 1), (1, 0)) if 0 <= row + dr < cells and 0 <= column + dc < cells)


def _clear_observed(world: ObservedWorld, cells: list[tuple[int, int]], *, unknown_endpoint_allowed: bool = False) -> bool:
    check = cells[:-1] if unknown_endpoint_allowed else cells
    obstacle_ratio = world.physical_obstacle_layer.values
    return bool(check) and all(world.observed_mask[row, column] and obstacle_ratio[row, column] == 0.0 for row, column in check)


def _ground_reachable_mask(
    world: ObservedWorld,
    projection: PlatformProjection,
    robot: tuple[int, int],
) -> np.ndarray:
    passable = (
        world.observed_mask
        & (world.physical_obstacle_layer.values == 0.0)
        & (projection.traversable_ratio > 0.0)
    )
    reachable = np.zeros_like(passable)
    if not passable[robot]:
        return reachable
    reachable[robot] = True
    queue: deque[tuple[int, int]] = deque((robot,))
    cells = world.canvas.geometry.cells
    while queue:
        row, column = queue.popleft()
        for neighbor in _neighbors(row, column, cells):
            if passable[neighbor] and not reachable[neighbor]:
                reachable[neighbor] = True
                queue.append(neighbor)
    return reachable


def _segments(points: list[tuple[int, int]], cells: int) -> list[list[tuple[int, int]]]:
    remaining, output = set(points), []
    while remaining:
        start = min(remaining); remaining.remove(start); queue, segment = [start], []
        while queue:
            point = queue.pop(); segment.append(point)
            for neighbor in _neighbors(*point, cells):
                if neighbor in remaining: remaining.remove(neighbor); queue.append(neighbor)
        output.append(sorted(segment))
    return output


def _points(mask: np.ndarray) -> list[tuple[int, int]]:
    rows, columns = np.nonzero(mask)
    return sorted((int(row), int(column)) for row, column in zip(rows, columns, strict=True))


def _spaced_anchors(segment: list[tuple[int, int]], spacing_cells: int) -> list[tuple[int, int]]:
    anchors: list[tuple[int, int]] = []
    for point in sorted(segment):
        if all(math.dist(point, anchor) >= spacing_cells for anchor in anchors):
            anchors.append(point)
    return anchors


@dataclass(frozen=True)
class _FeasibleAnchor:
    segment_id: int
    point: tuple[int, int]
    feature: np.ndarray
    elevation_m: float


def _anchor_key(anchor: _FeasibleAnchor) -> tuple[object, ...]:
    return (*tuple(float(value) for value in anchor.feature), anchor.segment_id, *anchor.point)


def _farthest_subset(anchors: list[_FeasibleAnchor], count: int) -> list[_FeasibleAnchor]:
    remaining = sorted(anchors, key=_anchor_key)
    if not remaining or count <= 0:
        return []
    selected = [remaining.pop(0)]
    while remaining and len(selected) < count:
        next_index = min(
            range(len(remaining)),
            key=lambda index: (
                -min(math.dist(remaining[index].point, chosen.point) for chosen in selected),
                _anchor_key(remaining[index]),
            ),
        )
        selected.append(remaining.pop(next_index))
    return selected


def _select_anchors(anchors: list[_FeasibleAnchor], segment_count: int, limit: int = 64) -> list[_FeasibleAnchor]:
    ordered = sorted(anchors, key=lambda anchor: (anchor.segment_id, _anchor_key(anchor)))
    by_segment: list[list[_FeasibleAnchor]] = [[] for _ in range(segment_count)]
    for anchor in ordered:
        by_segment[anchor.segment_id].append(anchor)
    representatives = [min(segment, key=_anchor_key) for segment in by_segment if segment]
    if len(representatives) > limit:
        return sorted(_farthest_subset(representatives, limit), key=_anchor_key)

    representative_keys = {(anchor.segment_id, anchor.point) for anchor in representatives}
    remaining = [anchor for anchor in ordered if (anchor.segment_id, anchor.point) not in representative_keys]
    selected = list(representatives)
    while remaining and len(selected) < limit:
        next_index = min(
            range(len(remaining)),
            key=lambda index: (
                -min(math.dist(remaining[index].point, chosen.point) for chosen in selected),
                _anchor_key(remaining[index]),
            ),
        )
        selected.append(remaining.pop(next_index))
    return sorted(selected, key=_anchor_key)


class CandidateBuilderV2:
    def __init__(self, visibility_estimator: VisibilityEstimator) -> None:
        sensor = getattr(visibility_estimator, "sensor", None)
        estimate = getattr(visibility_estimator, "estimate_candidate_gains", None)
        if not isinstance(sensor, SensorGeometry) or not callable(estimate):
            raise TypeError("candidate builder requires a visibility estimator")
        self._visibility_estimator = visibility_estimator
        self._sensor = sensor

    @property
    def sensor(self) -> SensorGeometry:
        return self._sensor

    def build(
        self,
        world: ObservedWorld,
        mission: MissionRaster,
        pose_map: Pose2,
        projection: PlatformProjection,
        *,
        platform_type: str,
        platform_reachability_filter_enabled: bool = True,
        platform_reachability: PlatformCandidateReachability | None = None,
        excluded_cells: Collection[tuple[int, int]] = (),
        backtrack_pose: Pose2 | None = None,
    ) -> CandidateBatch:
        if platform_type not in _PLATFORM_TYPES:
            raise ValueError("platform_type must be WHEELED, LEGGED, or HOPPER")
        if pose_map.frame_id != "map" or world.canvas != mission.canvas or world.canvas != projection.canvas:
            return CandidateBatch.empty()
        canvas, cells, observed, roi = world.canvas, world.canvas.geometry.cells, world.observed_mask, mission.roi_ratio > 0.0
        try:
            robot = canvas.world_to_grid(pose_map.x_m, pose_map.y_m)
        except ValueError:
            return CandidateBatch.empty()
        unknown_roi = roi & ~observed
        adjacent_unknown = np.zeros_like(observed)
        adjacent_unknown[1:] |= unknown_roi[:-1]
        adjacent_unknown[:-1] |= unknown_roi[1:]
        adjacent_unknown[:, 1:] |= unknown_roi[:, :-1]
        adjacent_unknown[:, :-1] |= unknown_roi[:, 1:]
        boundary = observed & roi & adjacent_unknown
        segments = _segments(_points(boundary), cells)
        spacing = max(1, math.ceil(self._sensor.anchor_spacing_m / canvas.geometry.resolution_m))
        raw_anchors: list[tuple[int, tuple[int, int]]] = []
        total_roi = float(mission.roi_ratio.sum())
        step = max(1, round(self._sensor.standoff_m / canvas.geometry.resolution_m))
        for segment_id, segment in enumerate(segments):
            for row, column in _spaced_anchors(segment, spacing):
                standoff = (row + int(np.sign(robot[0] - row)) * step, column + int(np.sign(robot[1] - column)) * step)
                if not (0 <= standoff[0] < cells and 0 <= standoff[1] < cells and observed[standoff]):
                    standoff = (row, column)
                if self._candidate_within_sensor(canvas, pose_map, standoff):
                    raw_anchors.append((segment_id, standoff))
        qualified_anchors = sorted(set(raw_anchors))
        chosen, diagnostics = self._qualify_anchors(
            qualified_anchors,
            world,
            mission,
            pose_map,
            projection,
            platform_type=platform_type,
            platform_reachability_filter_enabled=(
                platform_reachability_filter_enabled
            ),
            platform_reachability=platform_reachability,
            excluded_cells=excluded_cells,
            total_roi=total_roi,
            allow_zero_gain=False,
        )
        segment_count = len(segments)
        transit_allowed = bool(boundary.any())
        if (
            len(chosen) < _MIN_PLATFORM_CANDIDATE_RESERVE
            and transit_allowed
            and platform_reachability_filter_enabled
        ):
            primary_points = {point for _, point in qualified_anchors}
            reserve_segment_id = segment_count
            reserve_anchors = [
                (reserve_segment_id, point)
                for point in self._fallback_observation_poses(
                    world, mission, pose_map, projection
                )
                if point not in primary_points
            ]
            if reserve_anchors:
                qualified_anchors = [*qualified_anchors, *reserve_anchors]
                chosen, diagnostics = self._qualify_anchors(
                    qualified_anchors,
                    world,
                    mission,
                    pose_map,
                    projection,
                    platform_type=platform_type,
                    platform_reachability_filter_enabled=(
                        platform_reachability_filter_enabled
                    ),
                    platform_reachability=platform_reachability,
                    excluded_cells=excluded_cells,
                    total_roi=total_roi,
                    allow_zero_gain=False,
                    exact_target_poses={},
                )
                segment_count += 1
        if (
            not chosen
            and transit_allowed
            and platform_reachability_filter_enabled
            and qualified_anchors
        ):
            chosen, diagnostics = self._qualify_anchors(
                qualified_anchors,
                world,
                mission,
                pose_map,
                projection,
                platform_type=platform_type,
                platform_reachability_filter_enabled=(
                    platform_reachability_filter_enabled
                ),
                platform_reachability=platform_reachability,
                excluded_cells=excluded_cells,
                total_roi=total_roi,
                allow_zero_gain=True,
                exact_target_poses={},
            )
        if not chosen and platform_reachability_filter_enabled:
            if (
                not chosen
                and transit_allowed
                and isinstance(backtrack_pose, Pose2)
                and backtrack_pose.frame_id == "map"
                and self._position_within_sensor(pose_map, backtrack_pose)
            ):
                try:
                    backtrack_cell = canvas.world_to_grid(
                        backtrack_pose.x_m, backtrack_pose.y_m
                    )
                except ValueError:
                    backtrack_cell = None
                if backtrack_cell is not None:
                    segment_count = max(segment_count, 1)
                    chosen, diagnostics = self._qualify_anchors(
                        [(0, backtrack_cell)],
                        world,
                        mission,
                        pose_map,
                        projection,
                        platform_type=platform_type,
                        platform_reachability_filter_enabled=(
                            platform_reachability_filter_enabled
                        ),
                        platform_reachability=platform_reachability,
                        excluded_cells=(),
                        total_roi=total_roi,
                        allow_zero_gain=True,
                        exact_target_poses={backtrack_cell: backtrack_pose},
                    )
        chosen = _select_anchors(chosen, segment_count)
        output = np.zeros((64, 12), np.float32); mask = np.zeros(64, bool)
        target_elevation = np.zeros(64, dtype=np.float64)
        for index, anchor in enumerate(chosen):
            output[index] = anchor.feature
            target_elevation[index] = anchor.elevation_m
            mask[index] = True
        return CandidateBatch(
            output,
            mask,
            canvas.identity,
            CandidateDiagnostics(
                frontier_anchor_count=diagnostics.frontier_anchor_count,
                visited_excluded_count=diagnostics.visited_excluded_count,
                static_infeasible_count=diagnostics.static_infeasible_count,
                platform_unreachable_count=diagnostics.platform_unreachable_count,
                zero_gain_count=diagnostics.zero_gain_count,
                emitted_count=len(chosen),
            ),
            target_elevation,
        )

    def build_from_primitive_graph(
        self,
        world: ObservedWorld,
        mission: MissionRaster,
        pose_map: Pose2,
        primitive_graph: ObservedPrimitiveSnapshot,
        *,
        platform_type: str,
        excluded_state_ids: Collection[int] = (),
    ) -> CandidateBatch:
        """Build policy targets only from exact current graph states."""
        if platform_type not in _PLATFORM_TYPES:
            raise ValueError("platform_type must be WHEELED, LEGGED, or HOPPER")
        if (
            not isinstance(primitive_graph, ObservedPrimitiveSnapshot)
            or primitive_graph.platform_type != platform_type
        ):
            raise TypeError("candidate primitive graph is invalid")
        if (
            pose_map.frame_id != "map"
            or world.canvas != mission.canvas
            or primitive_graph.revision <= 0
        ):
            return CandidateBatch.empty(world.canvas.identity)
        excluded = {int(value) for value in excluded_state_ids}
        if any(value < 0 for value in excluded):
            raise ValueError("excluded primitive state identity is invalid")
        positions = primitive_graph.positions_m
        distances = np.hypot(
            positions[:, 0] - pose_map.x_m,
            positions[:, 1] - pose_map.y_m,
        )
        raw_indices = np.flatnonzero(primitive_graph.observation_state)
        unvisited_indices = np.asarray(
            [
                index
                for index in raw_indices
                if int(primitive_graph.state_ids[index]) not in excluded
                and primitive_graph.path_cost[index] > 0.0
            ],
            dtype=np.intp,
        )
        visited_excluded = len(raw_indices) - len(unvisited_indices)
        spatial: list[tuple[int, tuple[int, int]]] = []
        for index in unvisited_indices:
            try:
                cell = world.canvas.world_to_grid(
                    float(positions[index, 0]), float(positions[index, 1])
                )
            except ValueError:
                continue
            spatial.append((int(index), cell))
        static_infeasible = len(unvisited_indices) - len(spatial)
        feasible = [
            (index, cell)
            for index, cell in spatial
            if bool(primitive_graph.recoverable[index])
            and distances[index] <= self._sensor.range_m + 1.0e-9
            and (
                platform_type != "HOPPER"
                or bool(primitive_graph.direct_successor[index])
            )
        ]
        platform_unreachable = len(spatial) - len(feasible)
        diagnostics = CandidateDiagnostics(
            frontier_anchor_count=len(raw_indices),
            visited_excluded_count=visited_excluded,
            static_infeasible_count=static_infeasible,
            platform_unreachable_count=platform_unreachable,
        )
        if not feasible:
            return CandidateBatch.empty(world.canvas.identity, diagnostics)

        # A 360-degree sensor makes poses in the same observed 0.2 m cell
        # observationally equivalent. Keep the cheapest stable graph state.
        deduplicated: dict[tuple[int, int], tuple[int, tuple[int, int]]] = {}
        detail_resolution = float(
            getattr(self._visibility_estimator, "resolution_m", 0.2)
        )
        if not math.isfinite(detail_resolution) or detail_resolution <= 0.0:
            detail_resolution = 0.2
        for index, cell in feasible:
            key = (
                int(math.floor(positions[index, 0] / detail_resolution)),
                int(math.floor(positions[index, 1] / detail_resolution)),
            )
            previous = deduplicated.get(key)
            if previous is None or (
                float(primitive_graph.path_cost[index]),
                int(primitive_graph.state_ids[index]),
            ) < (
                float(primitive_graph.path_cost[previous[0]]),
                int(primitive_graph.state_ids[previous[0]]),
            ):
                deduplicated[key] = (index, cell)
        feasible = list(deduplicated.values())
        candidate_cells = np.ascontiguousarray(
            [cell for _, cell in feasible], dtype=np.int32
        ).reshape((-1, 2))
        exact_positions = np.ascontiguousarray(
            [positions[index] for index, _ in feasible], dtype=np.float64
        )
        exact_gain = getattr(
            self._visibility_estimator,
            "estimate_candidate_gains_at_positions",
            None,
        )
        gain_arguments = (
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
            exact_gain(*gain_arguments, exact_positions)
            if callable(exact_gain)
            else self._visibility_estimator.estimate_candidate_gains(
                *gain_arguments, candidate_cells
            )
        )
        if (
            not isinstance(gains, np.ndarray)
            or gains.dtype != np.dtype(np.float32)
            or gains.shape != (len(feasible), 2)
            or not gains.flags.c_contiguous
            or not np.isfinite(gains).all()
            or (gains < 0.0).any()
        ):
            raise RuntimeError("primitive candidate gain result is invalid")
        positive = [
            (index, cell, float(gain), float(priority_gain))
            for (index, cell), (gain, priority_gain) in zip(
                feasible, gains, strict=True
            )
            if gain > 0.0
        ]
        selected = positive
        transit_selected = 0
        if not selected and bool(
            np.any((mission.roi_ratio > 0.0) & ~world.observed_mask)
        ):
            selected = [
                min(
                    (
                        (index, cell, float(gain), float(priority_gain))
                        for (index, cell), (gain, priority_gain) in zip(
                            feasible, gains, strict=True
                        )
                    ),
                    key=lambda item: (
                        float(primitive_graph.path_cost[item[0]]),
                        int(primitive_graph.state_ids[item[0]]),
                    ),
                )
            ]
            transit_selected = 1
        selected.sort(
            key=lambda item: (
                -item[2]
                / (1.0 + float(primitive_graph.path_cost[item[0]])),
                float(primitive_graph.path_cost[item[0]]),
                int(primitive_graph.state_ids[item[0]]),
            )
        )
        selected = selected[:64]
        output = np.zeros((64, 12), dtype=np.float32)
        mask = np.zeros(64, dtype=np.bool_)
        target_positions = np.zeros((64, 3), dtype=np.float64)
        target_yaw = np.zeros(64, dtype=np.float64)
        state_ids = np.zeros(64, dtype=np.uint64)
        target_elevation = np.zeros(64, dtype=np.float64)
        gain_normalizer = max((item[2] for item in selected), default=0.0)
        priority_normalizer = max((item[3] for item in selected), default=0.0)
        total_roi = float(mission.roi_ratio.sum(dtype=np.float64))
        for output_index, (state_index, cell, gain, priority_gain) in enumerate(
            selected
        ):
            output[output_index] = self._primitive_feature(
                world,
                mission,
                pose_map,
                cell,
                positions[state_index],
                gain=gain,
                priority_gain=priority_gain,
                gain_normalizer=gain_normalizer,
                priority_gain_normalizer=priority_normalizer,
                total_roi=total_roi,
            )
            mask[output_index] = True
            target_positions[output_index] = positions[state_index]
            target_yaw[output_index] = primitive_graph.yaw_rad[state_index]
            state_ids[output_index] = primitive_graph.state_ids[state_index]
            target_elevation[output_index] = positions[state_index, 2]
        zero_discarded = max(0, len(feasible) - len(positive) - transit_selected)
        return CandidateBatch(
            output,
            mask,
            world.canvas.identity,
            CandidateDiagnostics(
                frontier_anchor_count=diagnostics.frontier_anchor_count,
                visited_excluded_count=diagnostics.visited_excluded_count,
                static_infeasible_count=diagnostics.static_infeasible_count,
                platform_unreachable_count=diagnostics.platform_unreachable_count,
                zero_gain_count=zero_discarded,
                emitted_count=len(selected),
            ),
            target_elevation,
            target_positions,
            target_yaw,
            state_ids,
            primitive_graph.revision,
        )

    @staticmethod
    def _primitive_feature(
        world: ObservedWorld,
        mission: MissionRaster,
        pose: Pose2,
        point: tuple[int, int],
        target_position_m: np.ndarray,
        *,
        gain: float,
        priority_gain: float,
        gain_normalizer: float,
        priority_gain_normalizer: float,
        total_roi: float,
    ) -> np.ndarray:
        canvas = world.canvas
        x = float(target_position_m[0])
        y = float(target_position_m[1])
        dx = x - pose.x_m
        dy = y - pose.y_m
        distance = math.hypot(dx, dy)
        bearing = math.atan2(dy, dx)
        normal = np.zeros(2, dtype=np.float64)
        for neighbor in _neighbors(*point, canvas.geometry.cells):
            if mission.roi_ratio[neighbor] > 0.0 and not world.observed_mask[neighbor]:
                normal += (
                    neighbor[0] - point[0],
                    neighbor[1] - point[1],
                )
        magnitude = float(np.linalg.norm(normal))
        remaining = (
            float(
                (mission.roi_ratio * ~world.observed_mask).sum(
                    dtype=np.float64
                )
                / total_roi
            )
            if total_roi
            else 0.0
        )
        return np.asarray(
            (
                (x - canvas.bounds_m[0]) / canvas.geometry.size_m,
                (canvas.bounds_m[3] - y) / canvas.geometry.size_m,
                min(
                    1.0,
                    distance
                    / (math.sqrt(2.0) * canvas.geometry.size_m),
                ),
                math.sin(bearing),
                math.cos(bearing),
                min(1.0, gain / gain_normalizer)
                if gain_normalizer
                else 0.0,
                min(1.0, priority_gain / priority_gain_normalizer)
                if priority_gain_normalizer
                else 0.0,
                -normal[0] / magnitude if magnitude else 0.0,
                normal[1] / magnitude if magnitude else 1.0,
                min(1.0, magnitude / 2.0),
                1.0,
                remaining,
            ),
            dtype=np.float32,
        )

    def _qualify_anchors(
        self,
        raw_anchors: list[tuple[int, tuple[int, int]]],
        world: ObservedWorld,
        mission: MissionRaster,
        pose_map: Pose2,
        projection: PlatformProjection,
        *,
        platform_type: str,
        platform_reachability_filter_enabled: bool,
        platform_reachability: PlatformCandidateReachability | None,
        excluded_cells: Collection[tuple[int, int]],
        total_roi: float,
        allow_zero_gain: bool,
        exact_target_poses: Mapping[tuple[int, int], Pose2] | None = None,
    ) -> tuple[list[_FeasibleAnchor], CandidateDiagnostics]:
        exact_target_poses = exact_target_poses or {}
        canvas = world.canvas
        observed = world.observed_mask
        roi = mission.roi_ratio > 0.0
        robot = canvas.world_to_grid(pose_map.x_m, pose_map.y_m)
        unvisited = [
            anchor
            for anchor in raw_anchors
            if anchor[1] != robot and anchor[1] not in excluded_cells
        ]
        visited_excluded = len(raw_anchors) - len(unvisited)
        static_feasible = [
            anchor
            for anchor in unvisited
            if observed[anchor[1]]
            and roi[anchor[1]]
            and world.physical_obstacle_layer.values[anchor[1]] == 0.0
            and projection.traversable_ratio[anchor[1]] > 0.0
        ]
        static_infeasible = len(unvisited) - len(static_feasible)
        reachable = None
        if (
            platform_reachability_filter_enabled
            and platform_reachability is None
            and platform_type in _GROUND_PLATFORM_TYPES
        ):
            reachable = _ground_reachable_mask(world, projection, robot)
        accepted_mask: np.ndarray | None = None
        if platform_reachability_filter_enabled and platform_reachability is not None:
            if not isinstance(platform_reachability, PlatformCandidateReachability):
                raise TypeError("platform reachability filter is invalid")
            candidate_cells = np.ascontiguousarray(
                [point for _, point in static_feasible], dtype=np.int32
            ).reshape((-1, 2))
            target_positions = None
            if exact_target_poses:
                target_positions = np.ascontiguousarray(
                    [
                        (
                            exact_target_poses[point].x_m,
                            exact_target_poses[point].y_m,
                            exact_target_poses[point].elevation_m,
                        )
                        if point in exact_target_poses
                        else (
                            *canvas.grid_center_world(*point),
                            float(world.elevation_m[point]),
                        )
                        for _, point in static_feasible
                    ],
                    dtype=np.float64,
                )
            result = (
                platform_reachability.filter(candidate_cells)
                if target_positions is None
                else platform_reachability.filter(
                    candidate_cells, target_positions_map=target_positions
                )
            )
            accepted_mask = result.accepted_mask
        feasible: list[tuple[int, tuple[int, int]]] = []
        for index, anchor in enumerate(static_feasible):
            point = anchor[1]
            if not platform_reachability_filter_enabled:
                accepted = _clear_observed(world, _ray_cells(robot, point))
            elif accepted_mask is not None:
                accepted = bool(accepted_mask[index])
            elif platform_type in _GROUND_PLATFORM_TYPES:
                assert reachable is not None
                accepted = bool(reachable[point])
            else:
                accepted = bool(
                    observed[point]
                    and roi[point]
                    and world.physical_obstacle_layer.values[point] == 0.0
                )
            if accepted:
                feasible.append(anchor)
        platform_unreachable = len(static_feasible) - len(feasible)
        diagnostics = CandidateDiagnostics(
            frontier_anchor_count=len(raw_anchors),
            visited_excluded_count=visited_excluded,
            static_infeasible_count=static_infeasible,
            platform_unreachable_count=platform_unreachable,
        )
        if not feasible:
            return [], diagnostics
        candidate_cells = np.ascontiguousarray(
            [point for _, point in feasible], dtype=np.int32
        )
        gains = self._visibility_estimator.estimate_candidate_gains(
            np.ascontiguousarray(world.observed_mask, dtype=np.bool_),
            np.ascontiguousarray(
                world.physical_obstacle_layer.values, dtype=np.float32
            ),
            np.ascontiguousarray(mission.roi_ratio, dtype=np.float32),
            np.ascontiguousarray(
                mission.priority * mission.roi_ratio, dtype=np.float32
            ),
            candidate_cells,
        )
        if (
            not isinstance(gains, np.ndarray)
            or gains.shape != (len(feasible), 2)
            or gains.dtype != np.dtype(np.float32)
            or not gains.flags.c_contiguous
            or not np.isfinite(gains).all()
            or (gains < 0.0).any()
        ):
            raise RuntimeError("candidate visibility estimator result is invalid")
        chosen: list[_FeasibleAnchor] = []
        zero_gain_count = 0
        gain_normalizer = float(gains[:, 0].max())
        priority_gain_normalizer = float(gains[:, 1].max())
        admit_zero_gain = allow_zero_gain and not bool(
            np.any(gains[:, 0] > np.float32(0.0))
        )
        for (segment_id, point), (gain, priority_gain) in zip(
            feasible, gains, strict=True
        ):
            feature = self._feature(
                world,
                mission,
                projection,
                pose_map,
                point,
                total_roi,
                float(gain),
                float(priority_gain),
                gain_normalizer,
                priority_gain_normalizer,
                allow_zero_gain=admit_zero_gain,
                target_pose=exact_target_poses.get(point),
            )
            if feature is not None:
                target_pose = exact_target_poses.get(point)
                chosen.append(
                    _FeasibleAnchor(
                        segment_id,
                        point,
                        feature,
                        (
                            float(world.elevation_m[point])
                            if target_pose is None
                            else float(target_pose.elevation_m)
                        ),
                    )
                )
            else:
                zero_gain_count += 1
        return chosen, CandidateDiagnostics(
            frontier_anchor_count=diagnostics.frontier_anchor_count,
            visited_excluded_count=diagnostics.visited_excluded_count,
            static_infeasible_count=diagnostics.static_infeasible_count,
            platform_unreachable_count=diagnostics.platform_unreachable_count,
            zero_gain_count=zero_gain_count,
        )

    def _fallback_observation_poses(
        self,
        world: ObservedWorld,
        mission: MissionRaster,
        pose_map: Pose2,
        projection: PlatformProjection,
    ) -> list[tuple[int, int]]:
        safe = (
            world.observed_mask
            & (mission.roi_ratio > 0.0)
            & (world.physical_obstacle_layer.values == 0.0)
            & (projection.traversable_ratio > 0.0)
        )
        return [
            point
            for point in _points(safe)
            if self._candidate_within_sensor(world.canvas, pose_map, point)
        ]

    def _candidate_within_sensor(
        self,
        canvas,
        pose: Pose2,
        point: tuple[int, int],
    ) -> bool:
        x, y = canvas.grid_center_world(*point)
        return self._position_within_sensor(pose, Pose2(x, y))

    def _position_within_sensor(self, pose: Pose2, target: Pose2) -> bool:
        bearing = math.atan2(target.y_m - pose.y_m, target.x_m - pose.x_m)
        distance = math.hypot(target.x_m - pose.x_m, target.y_m - pose.y_m)
        if distance > self._sensor.range_m:
            return False
        if self._sensor.is_full_circle:
            return True
        relative = math.atan2(
            math.sin(bearing - pose.yaw_rad),
            math.cos(bearing - pose.yaw_rad),
        )
        return abs(relative) <= self._sensor.fov_rad / 2.0

    def _feature(self, world: ObservedWorld, mission: MissionRaster, projection: PlatformProjection, pose: Pose2, point: tuple[int, int], total_roi: float, gain: float, priority_gain: float, gain_normalizer: float, priority_gain_normalizer: float, *, allow_zero_gain: bool = False, target_pose: Pose2 | None = None) -> np.ndarray | None:
        canvas = world.canvas
        x, y = (
            canvas.grid_center_world(*point)
            if target_pose is None
            else (target_pose.x_m, target_pose.y_m)
        )
        dx, dy = x - pose.x_m, y - pose.y_m; distance = math.hypot(dx, dy)
        bearing = math.atan2(dy, dx)
        if gain == 0.0 and not allow_zero_gain:
            return None
        normal = np.array([0.0, 0.0])
        for neighbor in _neighbors(*point, canvas.geometry.cells):
            if mission.roi_ratio[neighbor] > 0 and not world.observed_mask[neighbor]: normal += (neighbor[0] - point[0], neighbor[1] - point[1])
        magnitude = float(np.linalg.norm(normal)); remaining = float((mission.roi_ratio * ~world.observed_mask).sum() / total_roi) if total_roi else 0.0
        return np.asarray(((x - canvas.bounds_m[0]) / canvas.geometry.size_m, (canvas.bounds_m[3] - y) / canvas.geometry.size_m, min(1.0, distance / (math.sqrt(2.0) * canvas.geometry.size_m)), math.sin(bearing), math.cos(bearing), min(1.0, gain / gain_normalizer) if gain_normalizer else 0.0, min(1.0, priority_gain / priority_gain_normalizer) if priority_gain_normalizer else 0.0, -normal[0] / magnitude if magnitude else 0.0, normal[1] / magnitude if magnitude else 1.0, min(1.0, magnitude / 2.0), projection.clearance_margin_norm[point], remaining), dtype=np.float32)


__all__ = [
    "CandidateBatch",
    "CandidateBuilderV2",
    "CandidateDiagnostics",
    "SensorGeometry",
]

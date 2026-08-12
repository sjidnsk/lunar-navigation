"""Observed-only deterministic frontier candidates for the frozen V2 policy."""

from __future__ import annotations

from collections import deque
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from hashlib import sha256
import json
import math

import numpy as np

from lunar_model_contract import ObservationContractV3

from .observation_builder import MissionRaster, ObservedWorld, PlatformProjection, Pose2
from .platform_reachability import (
    HopperOpportunityAuthority,
    PhysicalReachabilityResult,
    PlatformCandidateReachability,
)
from .visibility import SensorGeometry, VisibilityEstimator, _ray_cells


_GROUND_PLATFORM_TYPES = frozenset(("WHEELED", "LEGGED"))
_PLATFORM_TYPES = _GROUND_PLATFORM_TYPES | {"HOPPER"}
_MIN_PLATFORM_CANDIDATE_RESERVE = 8
_MAX_PHYSICAL_CANDIDATES = 4096
_POLICY_CANDIDATE_COUNT = 64
_CANONICAL_INT64_MIN = -(1 << 63)
_CANONICAL_INT64_MAX = (1 << 63) - 1

CANDIDATE_ID_SCHEMA = "lunar-physical-candidate-id/v1"
PHYSICAL_SNAPSHOT_SCHEMA = "lunar-physical-snapshot/v1"


CANDIDATE_DIAGNOSTIC_FIELDS = (
    "physical_snapshot_id",
    "physical_reachability_algorithm_id",
    "physical_candidate_universe_count",
    "selected_policy_candidate_count",
    "available_candidate_count",
    "untried_reserve_count",
    "planner_failed_current_snapshot_count",
    "zero_gain_count",
    "visited_excluded_count",
    "physical_unreachable_count",
)


@dataclass(frozen=True, slots=True)
class CandidateDiagnostics:
    physical_snapshot_id: str = ""
    physical_reachability_algorithm_id: str = ""
    physical_candidate_universe_count: int = 0
    selected_policy_candidate_count: int = 0
    available_candidate_count: int = 0
    untried_reserve_count: int = 0
    planner_failed_current_snapshot_count: int = 0
    zero_gain_count: int = 0
    visited_excluded_count: int = 0
    physical_unreachable_count: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.physical_snapshot_id, str) or not isinstance(
            self.physical_reachability_algorithm_id, str
        ):
            raise TypeError("candidate diagnostic identities must be strings")
        values = (
            self.physical_candidate_universe_count,
            self.selected_policy_candidate_count,
            self.available_candidate_count,
            self.untried_reserve_count,
            self.planner_failed_current_snapshot_count,
            self.zero_gain_count,
            self.visited_excluded_count,
            self.physical_unreachable_count,
        )
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in values):
            raise ValueError("candidate diagnostics must contain non-negative integers")
        if (
            self.physical_candidate_universe_count > _MAX_PHYSICAL_CANDIDATES
            or self.selected_policy_candidate_count > _POLICY_CANDIDATE_COUNT
            or self.available_candidate_count
            > self.physical_candidate_universe_count
            or self.selected_policy_candidate_count
            > self.available_candidate_count
            or (
                self.planner_failed_current_snapshot_count
                + (
                    self.visited_excluded_count
                    if _is_sha256(self.physical_snapshot_id)
                    else 0
                )
                != self.physical_candidate_universe_count
                - self.available_candidate_count
            )
            or self.untried_reserve_count
            != self.available_candidate_count
            - self.selected_policy_candidate_count
        ):
            raise ValueError("candidate selection diagnostics are inconsistent")


@dataclass(frozen=True, slots=True)
class _QualificationDiagnostics:
    frontier_anchor_count: int = 0
    visited_excluded_count: int = 0
    static_infeasible_count: int = 0
    platform_unreachable_count: int = 0
    zero_gain_count: int = 0


def _is_sha256(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_sha256(body: object) -> str:
    return sha256(
        json.dumps(
            body,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _millimetres(value_m: float) -> int:
    scaled = float(value_m) * 1_000.0
    if not math.isfinite(scaled):
        raise ValueError("candidate millimetre key is out of range")
    value_mm = int(round(scaled))
    if not _CANONICAL_INT64_MIN <= value_mm <= _CANONICAL_INT64_MAX:
        raise ValueError("candidate millimetre key is out of range")
    return value_mm


@dataclass(frozen=True, slots=True)
class PhysicalCandidate:
    candidate_id: str
    position_grid_key: tuple[int, int]
    target_position_m: tuple[float, float, float]
    target_yaw_bin: int
    target_yaw_rad: float
    goal_tolerance_mm: int
    feature: np.ndarray
    rank_key: tuple[object, ...]

    def __post_init__(self) -> None:
        if not _is_sha256(self.candidate_id):
            raise ValueError("physical candidate ID must be a lowercase SHA-256")
        if (
            not isinstance(self.position_grid_key, tuple)
            or len(self.position_grid_key) != 2
            or any(type(value) is not int or value < 0 for value in self.position_grid_key)
            or not isinstance(self.target_position_m, tuple)
            or len(self.target_position_m) != 3
            or not all(math.isfinite(float(value)) for value in self.target_position_m)
            or type(self.target_yaw_bin) is not int
            or not 0 <= self.target_yaw_bin < 64
            or not math.isfinite(self.target_yaw_rad)
            or type(self.goal_tolerance_mm) is not int
            or self.goal_tolerance_mm < 0
            or not isinstance(self.rank_key, tuple)
        ):
            raise ValueError("physical candidate fields are invalid")
        feature = np.ascontiguousarray(self.feature, dtype=np.float32)
        if feature.shape != (len(ObservationContractV3.frontier_fields),) or not np.isfinite(feature).all():
            raise ValueError("physical candidate feature must be finite float32 [12]")
        feature = feature.copy()
        feature.setflags(write=False)
        object.__setattr__(self, "feature", feature)


@dataclass(frozen=True, slots=True)
class PhysicalCandidateUniverse:
    physical_snapshot_id: str
    physical_reachability_algorithm_id: str
    candidates: tuple[PhysicalCandidate, ...]
    universe_sha256: str
    diagnostics: CandidateDiagnostics

    def __post_init__(self) -> None:
        if (
            not _is_sha256(self.physical_snapshot_id)
            or not isinstance(self.physical_reachability_algorithm_id, str)
            or not self.physical_reachability_algorithm_id
            or not isinstance(self.candidates, tuple)
            or len(self.candidates) > _MAX_PHYSICAL_CANDIDATES
            or any(not isinstance(candidate, PhysicalCandidate) for candidate in self.candidates)
            or len({candidate.candidate_id for candidate in self.candidates})
            != len(self.candidates)
            or tuple(candidate.rank_key for candidate in self.candidates)
            != tuple(sorted(candidate.rank_key for candidate in self.candidates))
            or not _is_sha256(self.universe_sha256)
            or self.universe_sha256
            != sha256(
                "".join(candidate.candidate_id for candidate in self.candidates).encode("ascii")
            ).hexdigest()
            or not isinstance(self.diagnostics, CandidateDiagnostics)
            or self.diagnostics.physical_snapshot_id != self.physical_snapshot_id
            or self.diagnostics.physical_reachability_algorithm_id
            != self.physical_reachability_algorithm_id
            or self.diagnostics.physical_candidate_universe_count
            != len(self.candidates)
        ):
            raise ValueError("physical candidate universe is invalid")


@dataclass(frozen=True)
class CandidateBatch:
    features: np.ndarray
    mask: np.ndarray
    canvas_id: str | None = None
    diagnostics: CandidateDiagnostics = field(default_factory=CandidateDiagnostics)
    target_elevation_m: np.ndarray | None = None
    target_positions_m: np.ndarray | None = None
    target_yaw_rad: np.ndarray | None = None
    candidate_ids: np.ndarray | None = None

    def __post_init__(self) -> None:
        features, mask = np.asarray(self.features, dtype=np.float32), np.asarray(self.mask, dtype=bool)
        if features.shape != (64, len(ObservationContractV3.frontier_fields)) or mask.shape != (64,) or not np.isfinite(features).all():
            raise ValueError("candidate batch must use finite [64,12] and [64]")
        if not isinstance(self.diagnostics, CandidateDiagnostics):
            raise TypeError("candidate diagnostics are required")
        if self.diagnostics.selected_policy_candidate_count != int(mask.sum()):
            raise ValueError("selected policy candidate count must match mask")
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
        candidate_ids = self.candidate_ids
        if candidate_ids is None:
            candidate_ids = np.full(64, "", dtype="<U64")
        candidate_ids = np.ascontiguousarray(candidate_ids, dtype="<U64")
        if (
            target_positions.dtype != np.dtype(np.float64)
            or target_positions.shape != (64, 3)
            or not np.isfinite(target_positions).all()
            or target_yaw.dtype != np.dtype(np.float64)
            or target_yaw.shape != (64,)
            or not np.isfinite(target_yaw).all()
            or candidate_ids.shape != (64,)
            or (candidate_ids[~mask] != "").any()
            or any(
                value and not _is_sha256(str(value))
                for value in candidate_ids[mask]
            )
        ):
            raise ValueError("candidate targets or identities are invalid")
        object.__setattr__(self, "target_positions_m", target_positions)
        object.__setattr__(self, "target_yaw_rad", target_yaw)
        object.__setattr__(self, "candidate_ids", candidate_ids)

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


@dataclass(frozen=True, slots=True)
class CandidateBuildResult:
    universe: PhysicalCandidateUniverse
    batch: CandidateBatch

    def __post_init__(self) -> None:
        if not isinstance(self.universe, PhysicalCandidateUniverse) or not isinstance(
            self.batch, CandidateBatch
        ):
            raise TypeError("candidate build result is invalid")
        if self.batch.diagnostics != self.universe.diagnostics:
            raise ValueError("candidate result diagnostics differ")
        active_indices = np.flatnonzero(self.batch.mask)
        active_ids = tuple(
            str(self.batch.candidate_ids[index]) for index in active_indices
        )
        if (
            any(not _is_sha256(candidate_id) for candidate_id in active_ids)
            or len(set(active_ids)) != len(active_ids)
        ):
            raise ValueError("formal candidate batch identities are invalid")
        candidates_by_id = {
            candidate.candidate_id: candidate
            for candidate in self.universe.candidates
        }
        if any(candidate_id not in candidates_by_id for candidate_id in active_ids):
            raise ValueError("formal candidate batch identity is outside universe")
        for index, candidate_id in zip(
            active_indices, active_ids, strict=True
        ):
            candidate = candidates_by_id[candidate_id]
            if (
                not np.array_equal(
                    self.batch.features[index], candidate.feature
                )
                or tuple(self.batch.target_positions_m[index])
                != candidate.target_position_m
                or float(self.batch.target_elevation_m[index])
                != candidate.target_position_m[2]
                or float(self.batch.target_yaw_rad[index])
                != candidate.target_yaw_rad
            ):
                raise ValueError("formal candidate batch target is invalid")

        owned_arrays = tuple(
            value.copy(order="C")
            for value in (
                self.batch.features,
                self.batch.mask,
                self.batch.target_elevation_m,
                self.batch.target_positions_m,
                self.batch.target_yaw_rad,
                self.batch.candidate_ids,
            )
        )
        owned_batch = CandidateBatch(
            features=owned_arrays[0],
            mask=owned_arrays[1],
            canvas_id=self.batch.canvas_id,
            diagnostics=self.batch.diagnostics,
            target_elevation_m=owned_arrays[2],
            target_positions_m=owned_arrays[3],
            target_yaw_rad=owned_arrays[4],
            candidate_ids=owned_arrays[5],
        )
        for value in (
            owned_batch.features,
            owned_batch.mask,
            owned_batch.target_elevation_m,
            owned_batch.target_positions_m,
            owned_batch.target_yaw_rad,
            owned_batch.candidate_ids,
        ):
            value.setflags(write=False)
        object.__setattr__(self, "batch", owned_batch)


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


def _source_free_physical_anchors(
    anchors: Collection[tuple[int, tuple[int, int]]],
) -> list[tuple[int, tuple[int, int]]]:
    segment_by_target: dict[tuple[int, int], int] = {}
    for segment_id, target in anchors:
        previous = segment_by_target.get(target)
        if previous is None or segment_id < previous:
            segment_by_target[target] = segment_id
    return sorted(
        (segment_id, target)
        for target, segment_id in segment_by_target.items()
    )


@dataclass(frozen=True)
class _FeasibleAnchor:
    segment_id: int
    point: tuple[int, int]
    feature: np.ndarray
    elevation_m: float
    target_position_m: tuple[float, float, float]


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


def _candidate_distance_squared(
    lhs: PhysicalCandidate, rhs: PhysicalCandidate
) -> float:
    delta_row = lhs.position_grid_key[0] - rhs.position_grid_key[0]
    delta_column = lhs.position_grid_key[1] - rhs.position_grid_key[1]
    return float(delta_row * delta_row + delta_column * delta_column)


def _stable_farthest_candidates(
    candidates: list[PhysicalCandidate],
    count: int,
    *,
    initial: list[PhysicalCandidate] | None = None,
) -> list[PhysicalCandidate]:
    ordered = sorted(candidates, key=lambda candidate: candidate.rank_key)
    selected = list(initial or ())
    selected_ids = {candidate.candidate_id for candidate in selected}
    remaining = [
        candidate
        for candidate in ordered
        if candidate.candidate_id not in selected_ids
    ]
    if not selected and remaining and count > 0:
        selected.append(remaining.pop(0))
    if not remaining or len(selected) >= count:
        return selected[:count]
    minimum_distances = np.asarray(
        [
            min(
                _candidate_distance_squared(candidate, chosen)
                for chosen in selected
            )
            for candidate in remaining
        ],
        dtype=np.float64,
    )
    while remaining and len(selected) < count:
        farthest = float(minimum_distances.max(initial=-1.0))
        next_index = next(
            index
            for index, distance in enumerate(minimum_distances)
            if distance == farthest
        )
        chosen = remaining.pop(next_index)
        selected.append(chosen)
        minimum_distances = np.delete(minimum_distances, next_index)
        if remaining:
            new_distances = np.asarray(
                [
                    _candidate_distance_squared(candidate, chosen)
                    for candidate in remaining
                ],
                dtype=np.float64,
            )
            minimum_distances = np.minimum(minimum_distances, new_distances)
    return selected


def _compress_physical_candidates(
    candidates: list[PhysicalCandidate],
    *,
    canvas_cells: int,
    limit: int = _MAX_PHYSICAL_CANDIDATES,
) -> tuple[PhysicalCandidate, ...]:
    ordered = sorted(candidates, key=lambda candidate: candidate.rank_key)
    if len(ordered) <= limit:
        return tuple(ordered)
    segments = _segments(
        sorted({candidate.position_grid_key for candidate in ordered}),
        canvas_cells,
    )
    segment_by_cell = {
        cell: segment_id
        for segment_id, segment in enumerate(segments)
        for cell in segment
    }
    by_segment: list[list[PhysicalCandidate]] = [
        [] for _ in range(len(segments))
    ]
    for candidate in ordered:
        by_segment[segment_by_cell[candidate.position_grid_key]].append(candidate)
    representatives = [segment[0] for segment in by_segment if segment]
    if len(representatives) > limit:
        selected = _stable_farthest_candidates(representatives, limit)
    else:
        selected = _stable_farthest_candidates(
            ordered,
            limit,
            initial=representatives,
        )
    return tuple(sorted(selected, key=lambda candidate: candidate.rank_key))


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
                physical_snapshot_id="legacy/non-formal",
                physical_reachability_algorithm_id=projection.source,
                physical_candidate_universe_count=len(chosen),
                selected_policy_candidate_count=len(chosen),
                available_candidate_count=len(chosen),
                untried_reserve_count=0,
                planner_failed_current_snapshot_count=0,
                zero_gain_count=diagnostics.zero_gain_count,
                visited_excluded_count=diagnostics.visited_excluded_count,
                physical_unreachable_count=(
                    diagnostics.static_infeasible_count
                    + diagnostics.platform_unreachable_count
                ),
            ),
            target_elevation,
        )

    def build_physical_universe(
        self,
        world: ObservedWorld,
        mission: MissionRaster,
        pose_map: Pose2,
        projection: PlatformProjection,
        *,
        physical_reachability: PhysicalReachabilityResult,
        platform_type: str,
        platform_id: str,
        capability_content_sha256: str,
        mission_revision: int,
        evidence_generation: int,
        physical_evidence_sha256: str,
        physical_reachability_algorithm_id: str,
        goal_tolerance_mm: int,
        excluded_cells: Collection[tuple[int, int]] = (),
        backtrack_pose: Pose2 | None = None,
    ) -> PhysicalCandidateUniverse:
        """Build one primitive-independent, bounded physical opportunity set."""
        del excluded_cells, backtrack_pose
        self._validate_physical_identity(
            platform_type=platform_type,
            platform_id=platform_id,
            capability_content_sha256=capability_content_sha256,
            mission_revision=mission_revision,
            evidence_generation=evidence_generation,
            physical_evidence_sha256=physical_evidence_sha256,
            physical_reachability_algorithm_id=(
                physical_reachability_algorithm_id
            ),
            goal_tolerance_mm=goal_tolerance_mm,
        )
        if (
            not isinstance(world, ObservedWorld)
            or not isinstance(mission, MissionRaster)
            or not isinstance(projection, PlatformProjection)
            or not isinstance(
                physical_reachability, PhysicalReachabilityResult
            )
            or not isinstance(pose_map, Pose2)
            or pose_map.frame_id != "map"
            or world.canvas != mission.canvas
            or world.canvas != projection.canvas
        ):
            raise ValueError("physical candidate inputs must share a map canvas")
        canvas = world.canvas
        cells = canvas.geometry.cells
        physical_mask = physical_reachability.physical_observation_pose_mask
        if (
            physical_reachability.platform_type != platform_type
            or physical_reachability.physical_reachability_algorithm_id
            != physical_reachability_algorithm_id
            or physical_mask.shape != (cells, cells)
        ):
            raise ValueError("physical reachability identity or geometry differs")
        physical_cells = tuple(
            (int(row), int(column))
            for row, column in zip(*np.nonzero(physical_mask), strict=True)
        )
        physical_positions = physical_reachability.observation_positions_m
        if len(physical_cells) != len(physical_positions):
            raise ValueError("physical reachability positions differ from mask")
        exact_target_poses: dict[tuple[int, int], Pose2] = {}
        for cell, position in zip(
            physical_cells, physical_positions, strict=True
        ):
            if canvas.world_to_grid(
                float(position[0]), float(position[1])
            ) != cell:
                raise ValueError(
                    "physical reachability position leaves its row-major cell"
                )
            exact_target_poses[cell] = Pose2(
                float(position[0]),
                float(position[1]),
                elevation_m=float(position[2]),
            )
        try:
            robot = canvas.world_to_grid(pose_map.x_m, pose_map.y_m)
        except ValueError as error:
            raise ValueError("physical candidate pose is outside the map") from error

        observed = world.observed_mask
        roi = mission.roi_ratio > 0.0
        unknown_roi = roi & ~observed
        adjacent_unknown = np.zeros_like(observed)
        adjacent_unknown[1:] |= unknown_roi[:-1]
        adjacent_unknown[:-1] |= unknown_roi[1:]
        adjacent_unknown[:, 1:] |= unknown_roi[:, :-1]
        adjacent_unknown[:, :-1] |= unknown_roi[:, 1:]
        boundary = observed & roi & adjacent_unknown
        segments = _segments(_points(boundary), cells)
        spacing = max(
            1,
            math.ceil(
                self._sensor.anchor_spacing_m
                / canvas.geometry.resolution_m
            ),
        )
        step = max(
            1,
            round(
                self._sensor.standoff_m / canvas.geometry.resolution_m
            ),
        )
        raw_anchors: list[tuple[int, tuple[int, int]]] = []
        for segment_id, segment in enumerate(segments):
            for row, column in _spaced_anchors(segment, spacing):
                standoff = (
                    row + int(np.sign(robot[0] - row)) * step,
                    column + int(np.sign(robot[1] - column)) * step,
                )
                if not (
                    0 <= standoff[0] < cells
                    and 0 <= standoff[1] < cells
                    and observed[standoff]
                ):
                    standoff = (row, column)
                if self._candidate_within_sensor(canvas, pose_map, standoff):
                    raw_anchors.append((segment_id, standoff))
        if boundary.any():
            reserve_segment = len(segments)
            raw_anchors.extend(
                (reserve_segment, point)
                for point in self._fallback_observation_poses(
                    world, mission, pose_map, projection
                )
            )
        hopper_allowed: np.ndarray | None = None
        if platform_type == "HOPPER":
            authority = physical_reachability.hopper_opportunity_authority
            if authority is not None:
                if not isinstance(authority, HopperOpportunityAuthority):
                    raise ValueError("hopper opportunity authority is invalid")
                positive_mask = self.hopper_positive_mask(
                    world, mission, physical_reachability
                )
                remote_positive = np.ascontiguousarray(
                    positive_mask & ~authority.direct_mask, dtype=np.bool_
                )
                opportunity = authority.query(remote_positive)
                progress = np.ascontiguousarray(
                    np.flipud(opportunity.direct_progress), dtype=np.bool_
                )
                hopper_allowed = np.ascontiguousarray(
                    (positive_mask & authority.direct_mask) | progress,
                    dtype=np.bool_,
                )
                for cell, position in zip(
                    zip(*np.nonzero(authority.certified_mask), strict=True),
                    authority.certified_positions_m,
                    strict=True,
                ):
                    if hopper_allowed[cell]:
                        exact_target_poses[cell] = Pose2(
                            float(position[0]),
                            float(position[1]),
                            elevation_m=float(position[2]),
                        )
                raw_anchors.extend(
                    (len(segments), (int(row), int(column)))
                    for row, column in zip(
                        *np.nonzero(hopper_allowed), strict=True
                    )
                )
        qualified_anchors = _source_free_physical_anchors(raw_anchors)
        total_roi = float(mission.roi_ratio.sum(dtype=np.float64))
        chosen, qualification = self._qualify_anchors(
            qualified_anchors,
            world,
            mission,
            pose_map,
            projection,
            platform_type=platform_type,
            platform_reachability_filter_enabled=True,
            platform_reachability=None,
            excluded_cells=(),
            total_roi=total_roi,
            allow_zero_gain=False,
            include_zero_gain=True,
            exact_target_poses=exact_target_poses,
            physical_observation_pose_mask=(
                hopper_allowed
                if hopper_allowed is not None
                else physical_mask
            ),
        )

        if hopper_allowed is not None:
            chosen = [
                anchor for anchor in chosen if hopper_allowed[anchor.point]
            ]

        canonical: dict[str, PhysicalCandidate] = {}
        for anchor in chosen:
            candidate = self._physical_candidate(
                anchor,
                pose_map=pose_map,
                platform_type=platform_type,
                platform_id=platform_id,
                mission_revision=mission_revision,
                goal_tolerance_mm=goal_tolerance_mm,
            )
            previous = canonical.get(candidate.candidate_id)
            if previous is None or candidate.rank_key < previous.rank_key:
                canonical[candidate.candidate_id] = candidate
        candidates = _compress_physical_candidates(
            list(canonical.values()),
            canvas_cells=cells,
        )
        physical_snapshot_id = self._physical_snapshot_id(
            platform_type=platform_type,
            platform_id=platform_id,
            capability_content_sha256=capability_content_sha256,
            mission_revision=mission_revision,
            pose_map=pose_map,
            evidence_generation=evidence_generation,
            physical_evidence_sha256=physical_evidence_sha256,
        )
        positive_count = sum(
            float(candidate.feature[5]) > 0.0 for candidate in candidates
        )
        selected_count = min(
            _POLICY_CANDIDATE_COUNT,
            positive_count if positive_count else len(candidates),
        )
        zero_gain_count = sum(
            float(candidate.feature[5]) == 0.0 for candidate in candidates
        )
        diagnostics = CandidateDiagnostics(
            physical_snapshot_id=physical_snapshot_id,
            physical_reachability_algorithm_id=(
                physical_reachability_algorithm_id
            ),
            physical_candidate_universe_count=len(candidates),
            selected_policy_candidate_count=selected_count,
            available_candidate_count=len(candidates),
            untried_reserve_count=len(candidates) - selected_count,
            planner_failed_current_snapshot_count=0,
            zero_gain_count=zero_gain_count,
            visited_excluded_count=0,
            physical_unreachable_count=(
                qualification.static_infeasible_count
                + qualification.platform_unreachable_count
            ),
        )
        universe_sha256 = sha256(
            "".join(candidate.candidate_id for candidate in candidates).encode(
                "ascii"
            )
        ).hexdigest()
        return PhysicalCandidateUniverse(
            physical_snapshot_id=physical_snapshot_id,
            physical_reachability_algorithm_id=(
                physical_reachability_algorithm_id
            ),
            candidates=candidates,
            universe_sha256=universe_sha256,
            diagnostics=diagnostics,
        )

    def hopper_positive_mask(
        self,
        world: ObservedWorld,
        mission: MissionRaster,
        physical_reachability: PhysicalReachabilityResult,
    ) -> np.ndarray:
        """Estimate production gains across all certified Hopper landings."""
        authority = physical_reachability.hopper_opportunity_authority
        if not isinstance(authority, HopperOpportunityAuthority):
            raise ValueError("hopper opportunity authority is missing")
        certified_cells = np.ascontiguousarray(
            np.column_stack(np.nonzero(authority.certified_mask)),
            dtype=np.int32,
        ).reshape((-1, 2))
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
            exact_gain(*arguments, authority.certified_positions_m)
            if callable(exact_gain)
            else self._visibility_estimator.estimate_candidate_gains(
                *arguments, certified_cells
            )
        )
        gains = np.asarray(gains)
        if (
            gains.dtype != np.dtype(np.float32)
            or gains.shape != (len(certified_cells), 2)
            or not gains.flags.c_contiguous
            or not np.isfinite(gains).all()
            or (gains < 0.0).any()
        ):
            raise RuntimeError("hopper opportunity visibility result is invalid")
        positive_mask = np.zeros_like(authority.certified_mask, dtype=np.bool_)
        if len(certified_cells):
            positive_cells = certified_cells[gains[:, 0] > np.float32(0.0)]
            positive_mask[positive_cells[:, 0], positive_cells[:, 1]] = True
        return np.ascontiguousarray(positive_mask)

    def select_available(
        self,
        universe: PhysicalCandidateUniverse,
        *,
        canvas_id: str,
        failure_snapshot_id: str | None = None,
        planner_failed_candidate_ids: Collection[str] = (),
        excluded_cells: Collection[tuple[int, int]] = (),
        backtrack_pose: Pose2 | None = None,
    ) -> CandidateBuildResult:
        """Apply failures and history, then select observation/transit/backtrack."""
        if not isinstance(universe, PhysicalCandidateUniverse):
            raise TypeError("physical candidate universe is required")
        if not isinstance(canvas_id, str) or not canvas_id:
            raise ValueError("candidate batch canvas identity is missing")
        if isinstance(planner_failed_candidate_ids, (str, bytes)):
            raise TypeError("planner failed candidate IDs must be a collection")
        if isinstance(excluded_cells, (str, bytes)):
            raise TypeError("visited candidate cells must be a collection")
        visited_cells = set(excluded_cells)
        if any(
            not isinstance(cell, tuple)
            or len(cell) != 2
            or any(type(value) is not int or value < 0 for value in cell)
            for cell in visited_cells
        ):
            raise ValueError("visited candidate cell is invalid")
        if backtrack_pose is not None and (
            not isinstance(backtrack_pose, Pose2)
            or backtrack_pose.frame_id != "map"
        ):
            raise ValueError("backtrack pose is invalid")
        failed_values = tuple(planner_failed_candidate_ids)
        current_failures: set[str] = set()
        if failed_values:
            if not _is_sha256(failure_snapshot_id):
                raise ValueError("failure snapshot identity is invalid")
            if failure_snapshot_id == universe.physical_snapshot_id:
                if any(not _is_sha256(value) for value in failed_values):
                    raise ValueError("planner failed candidate identity is invalid")
                current_failures = set(failed_values)
                universe_ids = {
                    candidate.candidate_id for candidate in universe.candidates
                }
                if not current_failures <= universe_ids:
                    raise ValueError(
                        "failed candidate identity is not in the current physical universe"
                    )
        planner_available = tuple(
            candidate
            for candidate in universe.candidates
            if candidate.candidate_id not in current_failures
        )
        unvisited = tuple(
            candidate
            for candidate in planner_available
            if candidate.position_grid_key not in visited_cells
        )
        positive = tuple(
            candidate
            for candidate in unvisited
            if float(candidate.feature[5]) > 0.0
        )
        if positive:
            selected = positive[:_POLICY_CANDIDATE_COUNT]
        elif unvisited:
            selected = unvisited[:_POLICY_CANDIDATE_COUNT]
        else:
            backtrack = ()
            if backtrack_pose is not None:
                exact_position = (
                    float(backtrack_pose.x_m),
                    float(backtrack_pose.y_m),
                    float(backtrack_pose.elevation_m),
                )
                backtrack = tuple(
                    candidate
                    for candidate in planner_available
                    if candidate.target_position_m == exact_position
                )
            selected = backtrack[:1]
        selected_ids = {candidate.candidate_id for candidate in selected}
        history_available = tuple(
            candidate
            for candidate in planner_available
            if candidate.position_grid_key not in visited_cells
            or candidate.candidate_id in selected_ids
        )
        visited_excluded_count = len(planner_available) - len(history_available)
        diagnostics = CandidateDiagnostics(
            physical_snapshot_id=universe.physical_snapshot_id,
            physical_reachability_algorithm_id=(
                universe.physical_reachability_algorithm_id
            ),
            physical_candidate_universe_count=len(universe.candidates),
            selected_policy_candidate_count=len(selected),
            available_candidate_count=len(history_available),
            untried_reserve_count=len(history_available) - len(selected),
            planner_failed_current_snapshot_count=len(current_failures),
            zero_gain_count=universe.diagnostics.zero_gain_count,
            visited_excluded_count=visited_excluded_count,
            physical_unreachable_count=(
                universe.diagnostics.physical_unreachable_count
            ),
        )
        selected_universe = PhysicalCandidateUniverse(
            physical_snapshot_id=universe.physical_snapshot_id,
            physical_reachability_algorithm_id=(
                universe.physical_reachability_algorithm_id
            ),
            candidates=universe.candidates,
            universe_sha256=universe.universe_sha256,
            diagnostics=diagnostics,
        )
        return CandidateBuildResult(
            universe=selected_universe,
            batch=self._candidate_batch(
                selected,
                canvas_id=canvas_id,
                diagnostics=diagnostics,
            ),
        )

    @staticmethod
    def _validate_physical_identity(
        *,
        platform_type: str,
        platform_id: str,
        capability_content_sha256: str,
        mission_revision: int,
        evidence_generation: int,
        physical_evidence_sha256: str,
        physical_reachability_algorithm_id: str,
        goal_tolerance_mm: int,
    ) -> None:
        if platform_type not in _PLATFORM_TYPES:
            raise ValueError("platform_type must be WHEELED, LEGGED, or HOPPER")
        if not isinstance(platform_id, str) or not platform_id:
            raise ValueError("platform ID is missing")
        if not _is_sha256(capability_content_sha256):
            raise ValueError("capability content hash is invalid")
        if type(mission_revision) is not int or mission_revision <= 0:
            raise ValueError("mission revision must be positive")
        if type(evidence_generation) is not int or evidence_generation <= 0:
            raise ValueError("evidence generation must be positive")
        if not _is_sha256(physical_evidence_sha256):
            raise ValueError("physical evidence hash is invalid")
        if (
            not isinstance(physical_reachability_algorithm_id, str)
            or not physical_reachability_algorithm_id
        ):
            raise ValueError("physical reachability algorithm ID is missing")
        if type(goal_tolerance_mm) is not int or goal_tolerance_mm < 0:
            raise ValueError("goal tolerance millimetres are invalid")

    @staticmethod
    def _physical_snapshot_id(
        *,
        platform_type: str,
        platform_id: str,
        capability_content_sha256: str,
        mission_revision: int,
        pose_map: Pose2,
        evidence_generation: int,
        physical_evidence_sha256: str,
    ) -> str:
        normalized_yaw = math.atan2(
            math.sin(pose_map.yaw_rad), math.cos(pose_map.yaw_rad)
        )
        return _canonical_sha256(
            {
                "schema": PHYSICAL_SNAPSHOT_SCHEMA,
                "platform_type": platform_type,
                "platform_id": platform_id,
                "capability_content_sha256": capability_content_sha256,
                "mission_revision": mission_revision,
                "pose_key_mm": [
                    int(round(pose_map.x_m * 1_000.0)),
                    int(round(pose_map.y_m * 1_000.0)),
                    int(round(pose_map.elevation_m * 1_000.0)),
                ],
                "yaw_key_urad": int(round(normalized_yaw * 1_000_000.0)),
                "observed_evidence_generation": evidence_generation,
                "physical_evidence_sha256": physical_evidence_sha256,
            }
        )

    @staticmethod
    def _physical_candidate(
        anchor: _FeasibleAnchor,
        *,
        pose_map: Pose2,
        platform_type: str,
        platform_id: str,
        mission_revision: int,
        goal_tolerance_mm: int,
    ) -> PhysicalCandidate:
        x_m, y_m, z_m = anchor.target_position_m
        position_grid_key = (int(anchor.point[0]), int(anchor.point[1]))
        delta_x = x_m - pose_map.x_m
        delta_y = y_m - pose_map.y_m
        target_yaw = (
            math.atan2(delta_y, delta_x)
            if delta_x != 0.0 or delta_y != 0.0
            else math.atan2(math.sin(pose_map.yaw_rad), math.cos(pose_map.yaw_rad))
        )
        normalized = (target_yaw + math.pi) % (2.0 * math.pi)
        target_yaw_bin = min(
            63, int(math.floor(normalized * 64.0 / (2.0 * math.pi)))
        )
        z_mm = _millimetres(z_m)
        position_key = (
            {
                "landing_key_mm": [
                    _millimetres(x_m),
                    _millimetres(y_m),
                    z_mm,
                ]
            }
            if platform_type == "HOPPER"
            else {
                "row": position_grid_key[0],
                "column": position_grid_key[1],
                "z_mm": z_mm,
            }
        )
        candidate_id = _canonical_sha256(
            {
                "schema": CANDIDATE_ID_SCHEMA,
                "platform_type": platform_type,
                "platform_id": platform_id,
                "mission_revision": mission_revision,
                **position_key,
                "heading_bin_64": target_yaw_bin,
                "goal_tolerance_mm": goal_tolerance_mm,
            }
        )
        distance = math.hypot(delta_x, delta_y)
        rank_key: tuple[object, ...] = (
            -float(anchor.feature[5]),
            -float(anchor.feature[6]),
            distance,
            position_grid_key[0],
            position_grid_key[1],
            z_mm,
            target_yaw_bin,
            goal_tolerance_mm,
            x_m,
            y_m,
            z_m,
            candidate_id,
        )
        return PhysicalCandidate(
            candidate_id=candidate_id,
            position_grid_key=position_grid_key,
            target_position_m=(x_m, y_m, z_m),
            target_yaw_bin=target_yaw_bin,
            target_yaw_rad=target_yaw,
            goal_tolerance_mm=goal_tolerance_mm,
            feature=anchor.feature,
            rank_key=rank_key,
        )

    @staticmethod
    def _candidate_batch(
        candidates: tuple[PhysicalCandidate, ...],
        *,
        canvas_id: str,
        diagnostics: CandidateDiagnostics,
    ) -> CandidateBatch:
        features = np.zeros((64, 12), dtype=np.float32)
        mask = np.zeros(64, dtype=np.bool_)
        elevations = np.zeros(64, dtype=np.float64)
        positions = np.zeros((64, 3), dtype=np.float64)
        yaw = np.zeros(64, dtype=np.float64)
        candidate_ids = np.full(64, "", dtype="<U64")
        for index, candidate in enumerate(candidates):
            features[index] = candidate.feature
            mask[index] = True
            positions[index] = candidate.target_position_m
            elevations[index] = candidate.target_position_m[2]
            yaw[index] = candidate.target_yaw_rad
            candidate_ids[index] = candidate.candidate_id
        return CandidateBatch(
            features=features,
            mask=mask,
            canvas_id=canvas_id,
            diagnostics=diagnostics,
            target_elevation_m=elevations,
            target_positions_m=positions,
            target_yaw_rad=yaw,
            candidate_ids=candidate_ids,
        )

    def build_from_primitive_graph(self, *args, **kwargs) -> CandidateBatch:
        """Deprecated seam retained only to prove formal paths never call it."""
        del args, kwargs
        raise RuntimeError(
            "primitive graph candidate materialization is disabled; "
            "use build_physical_universe"
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
        include_zero_gain: bool = False,
        exact_target_poses: Mapping[tuple[int, int], Pose2] | None = None,
        physical_observation_pose_mask: np.ndarray | None = None,
    ) -> tuple[list[_FeasibleAnchor], _QualificationDiagnostics]:
        exact_target_poses = exact_target_poses or {}
        canvas = world.canvas
        observed = world.observed_mask
        roi = mission.roi_ratio > 0.0
        if physical_observation_pose_mask is not None and (
            not isinstance(physical_observation_pose_mask, np.ndarray)
            or physical_observation_pose_mask.dtype != np.dtype(np.bool_)
            or physical_observation_pose_mask.shape != observed.shape
            or not physical_observation_pose_mask.flags.c_contiguous
            or platform_reachability is not None
        ):
            raise ValueError("physical observation pose mask is invalid")
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
            and physical_observation_pose_mask is None
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
            elif physical_observation_pose_mask is not None:
                accepted = bool(physical_observation_pose_mask[point])
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
        diagnostics = _QualificationDiagnostics(
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
        admit_zero_gain = include_zero_gain or (
            allow_zero_gain
            and not bool(np.any(gains[:, 0] > np.float32(0.0)))
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
                        (
                            (
                                *canvas.grid_center_world(*point),
                                float(world.elevation_m[point]),
                            )
                            if target_pose is None
                            else (
                                float(target_pose.x_m),
                                float(target_pose.y_m),
                                float(target_pose.elevation_m),
                            )
                        ),
                    )
                )
            else:
                zero_gain_count += 1
        return chosen, _QualificationDiagnostics(
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
    "CANDIDATE_ID_SCHEMA",
    "CANDIDATE_DIAGNOSTIC_FIELDS",
    "PHYSICAL_SNAPSHOT_SCHEMA",
    "CandidateBatch",
    "CandidateBuildResult",
    "CandidateBuilderV2",
    "CandidateDiagnostics",
    "PhysicalCandidate",
    "PhysicalCandidateUniverse",
    "SensorGeometry",
]

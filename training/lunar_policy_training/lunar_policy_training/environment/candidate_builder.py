"""Observed-only deterministic frontier candidates for the frozen V2 policy."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, field, replace
from hashlib import sha256
import json
import math
from time import perf_counter

import numpy as np

from lunar_model_contract import ObservationContractV4

from .observation_builder import MissionRaster, ObservedWorld, PlatformProjection, Pose2
from .platform_reachability import (
    HopperSingleHopEnvelope,
    PhysicalReachabilityResult,
    PlatformCandidateReachability,
)
from .task_area import task_roi_diagonal_m
from .visibility import SensorGeometry, VisibilityEstimator, _ray_cells


_GROUND_PLATFORM_TYPES = frozenset(("WHEELED", "LEGGED"))
_PLATFORM_TYPES = _GROUND_PLATFORM_TYPES | {"HOPPER"}
_MIN_PLATFORM_CANDIDATE_RESERVE = 8
_MAX_PHYSICAL_CANDIDATES = 4096
_POLICY_CANDIDATE_COUNT = 64
_CANONICAL_INT64_MIN = -(1 << 63)
_CANONICAL_INT64_MAX = (1 << 63) - 1
_NOMINAL_GLOBAL_COST_PER_M = 1.0
_DETAIL_CELL_COARSE_EQUIVALENT = float(np.float32((0.2 / 4.0) ** 2))
_DETAIL_CELL_AREA_M2 = 0.2**2
_EMPTY_SHA256 = sha256(b"").hexdigest()

CANDIDATE_ID_SCHEMA = "lunar-physical-candidate-id/v3"
_HOPPER_CANDIDATE_ID_SCHEMA = "lunar-physical-candidate-id/v2"
PHYSICAL_SNAPSHOT_SCHEMA = "lunar-physical-snapshot/v1"


class CandidateInvariantError(RuntimeError):
    """A candidate snapshot contains non-finite or contradictory authority."""


@dataclass(frozen=True, slots=True)
class _RawFrontierCandidate:
    sample_rank: int
    frontier_cell: tuple[int, int]
    pose_cell: tuple[int, int]
    target_pose: Pose2 | None = None


@dataclass(frozen=True, slots=True)
class _NarrowFrontierStripSelection:
    """Deterministic detailed witnesses from sampled frontier anchors."""

    pose_cells: tuple[tuple[int, int], ...]
    evaluated_detail_cell_count: int


def _select_narrow_frontier_strip_positions(
    chain: Collection[tuple[int, int]],
    *,
    coarse_observed_mask: np.ndarray,
    coarse_unknown_roi_mask: np.ndarray | None = None,
    observed_detail_mask: np.ndarray,
    physical_safe_detail_mask: np.ndarray,
    clearance_detail: np.ndarray,
    detail_cells_per_coarse: int,
    minimum_standoff_detail_cells: int,
    maximum_standoff_detail_cells: int,
    lateral_half_width_detail_cells: int,
) -> _NarrowFrontierStripSelection:
    """Enumerate observed-safe poses from bounded 0.2 m frontier strips.

    The strip points from each sampled frontier anchor into its observed side,
    determined from four-neighbour unexplored task-ROI cells.  It emits every
    deduplicated safe witness in stable geometric order; later reachability
    and gain gates choose the bounded policy candidates.  It never scans map
    area outside the configured standoff and lateral bounds.
    """
    masks = (
        ("coarse observed", coarse_observed_mask),
        ("detail observed", observed_detail_mask),
        ("detail physical safe", physical_safe_detail_mask),
    )
    if any(
        not isinstance(mask, np.ndarray)
        or mask.dtype != np.dtype(np.bool_)
        or mask.ndim != 2
        or not mask.flags.c_contiguous
        for _, mask in masks
    ):
        raise ValueError("narrow frontier strip masks are invalid")
    if (
        not isinstance(clearance_detail, np.ndarray)
        or clearance_detail.dtype != np.dtype(np.float32)
        or clearance_detail.ndim != 2
        or not clearance_detail.flags.c_contiguous
        or not np.isfinite(clearance_detail).all()
        or (clearance_detail < 0.0).any()
        or observed_detail_mask.shape != physical_safe_detail_mask.shape
        or observed_detail_mask.shape != clearance_detail.shape
        or type(detail_cells_per_coarse) is not int
        or detail_cells_per_coarse <= 0
        or type(minimum_standoff_detail_cells) is not int
        or minimum_standoff_detail_cells < 0
        or type(maximum_standoff_detail_cells) is not int
        or maximum_standoff_detail_cells < minimum_standoff_detail_cells
        or type(lateral_half_width_detail_cells) is not int
        or lateral_half_width_detail_cells < 0
    ):
        raise ValueError("narrow frontier strip geometry is invalid")
    if observed_detail_mask.shape != tuple(
        dimension * detail_cells_per_coarse
        for dimension in coarse_observed_mask.shape
    ):
        raise ValueError("narrow frontier strip resolutions differ")
    if coarse_unknown_roi_mask is None:
        # Compatibility for direct unit callers.  Formal training always
        # supplies the task-ROI mask below, so exterior map cells never steer
        # a production frontier strip.
        coarse_unknown_roi_mask = np.ascontiguousarray(~coarse_observed_mask)
    if (
        not isinstance(coarse_unknown_roi_mask, np.ndarray)
        or coarse_unknown_roi_mask.dtype != np.dtype(np.bool_)
        or coarse_unknown_roi_mask.shape != coarse_observed_mask.shape
        or not coarse_unknown_roi_mask.flags.c_contiguous
        or bool((coarse_unknown_roi_mask & coarse_observed_mask).any())
    ):
        raise ValueError("narrow frontier strip task ROI mask is invalid")

    coarse_rows, coarse_columns = coarse_observed_mask.shape
    detail_rows, detail_columns = observed_detail_mask.shape
    seen: set[tuple[int, int]] = set()
    selected: list[tuple[int, int]] = []
    evaluated = 0
    directions = ((-1, 0), (0, -1), (0, 1), (1, 0))
    center_offset = detail_cells_per_coarse // 2

    for frontier_row, frontier_column in _sample_frontier_chain(chain):
        if not (
            0 <= frontier_row < coarse_rows
            and 0 <= frontier_column < coarse_columns
            and coarse_observed_mask[frontier_row, frontier_column]
        ):
            continue
        unknown_directions = [
            (row_delta, column_delta)
            for row_delta, column_delta in directions
            if 0 <= frontier_row + row_delta < coarse_rows
            and 0 <= frontier_column + column_delta < coarse_columns
            and coarse_unknown_roi_mask[
                frontier_row + row_delta,
                frontier_column + column_delta,
            ]
        ]
        row_normal = int(np.sign(sum(item[0] for item in unknown_directions)))
        column_normal = int(
            np.sign(sum(item[1] for item in unknown_directions))
        )
        anchor_row = frontier_row * detail_cells_per_coarse + center_offset
        anchor_column = (
            frontier_column * detail_cells_per_coarse + center_offset
        )
        if row_normal == 0 and column_normal == 0:
            probe_distance = max(1, minimum_standoff_detail_cells)
            fallback: tuple[tuple[float, int, int], tuple[int, int]] | None = None
            for row_delta, column_delta in directions:
                row = anchor_row + row_delta * probe_distance
                column = anchor_column + column_delta * probe_distance
                if not (0 <= row < detail_rows and 0 <= column < detail_columns):
                    continue
                evaluated += 1
                if (
                    not observed_detail_mask[row, column]
                    or not physical_safe_detail_mask[row, column]
                ):
                    continue
                key = (-float(clearance_detail[row, column]), row, column)
                if fallback is None or key < fallback[0]:
                    fallback = (key, (row_delta, column_delta))
            if fallback is None:
                continue
            observed_row_direction, observed_column_direction = fallback[1]
        else:
            observed_row_direction = -row_normal
            observed_column_direction = -column_normal
        lateral_row_direction = -observed_column_direction
        lateral_column_direction = observed_row_direction
        for distance in range(
            minimum_standoff_detail_cells,
            maximum_standoff_detail_cells + 1,
        ):
            center_row = anchor_row + observed_row_direction * distance
            center_column = anchor_column + observed_column_direction * distance
            for lateral_offset in range(
                -lateral_half_width_detail_cells,
                lateral_half_width_detail_cells + 1,
            ):
                row = center_row + lateral_row_direction * lateral_offset
                column = center_column + lateral_column_direction * lateral_offset
                if not (0 <= row < detail_rows and 0 <= column < detail_columns):
                    continue
                evaluated += 1
                if (
                    not observed_detail_mask[row, column]
                    or not physical_safe_detail_mask[row, column]
                    or (row, column) in seen
                ):
                    continue
                selected.append((row, column))
                seen.add((row, column))

    return _NarrowFrontierStripSelection(
        pose_cells=tuple(selected),
        evaluated_detail_cell_count=evaluated,
    )


def normalize_global_path_cost(
    cost: float,
    roi_diagonal_m: float,
    nominal_cost_per_m: float,
) -> float:
    """Normalize an absolute global cost without candidate-set statistics."""
    values = tuple(float(value) for value in (cost, roi_diagonal_m, nominal_cost_per_m))
    if (
        not all(math.isfinite(value) for value in values)
        or values[0] < 0.0
        or values[1] <= 0.0
        or values[2] <= 0.0
    ):
        raise CandidateInvariantError("GLOBAL_PATH_COST_NONFINITE")
    reference = values[1] * values[2]
    denominator = values[0] + reference
    result = values[0] / denominator
    if not math.isfinite(result) or not 0.0 <= result < 1.0:
        raise CandidateInvariantError("GLOBAL_PATH_COST_NONFINITE")
    return result


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


def _ground_potential_gain_mask(
    unknown_roi: np.ndarray,
    *,
    sensor_range_m: float,
    resolution_m: float,
) -> np.ndarray:
    """Losslessly coarse-filter ground poses before exact visibility."""
    unknown = np.asarray(unknown_roi)
    if (
        unknown.dtype != np.dtype(np.bool_)
        or unknown.ndim != 2
        or min(unknown.shape) <= 0
        or not unknown.flags.c_contiguous
    ):
        raise ValueError("unknown ROI mask is invalid")
    if (
        not math.isfinite(sensor_range_m)
        or sensor_range_m <= 0.0
        or not math.isfinite(resolution_m)
        or resolution_m <= 0.0
    ):
        raise ValueError("ground gain prefilter geometry is invalid")
    possible = np.zeros_like(unknown)
    if not unknown.any():
        return possible
    rows, columns = unknown.shape
    radius = math.ceil(sensor_range_m / resolution_m)
    if radius >= max(rows - 1, columns - 1):
        return np.ones_like(unknown)
    padded = np.pad(unknown.astype(np.int32), ((1, 0), (1, 0)))
    integral = padded.cumsum(axis=0, dtype=np.int64).cumsum(
        axis=1, dtype=np.int64
    )
    row_low = np.maximum(np.arange(rows) - radius, 0)
    row_high = np.minimum(np.arange(rows) + radius + 1, rows)
    column_low = np.maximum(np.arange(columns) - radius, 0)
    column_high = np.minimum(np.arange(columns) + radius + 1, columns)
    sums = (
        integral[row_high[:, None], column_high[None, :]]
        - integral[row_low[:, None], column_high[None, :]]
        - integral[row_high[:, None], column_low[None, :]]
        + integral[row_low[:, None], column_low[None, :]]
    )
    possible = np.ascontiguousarray(sums > 0, dtype=np.bool_)
    return np.ascontiguousarray(possible, dtype=np.bool_)


def _ground_residual_components(
    residual: np.ndarray,
) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Return canonical four-connected observed-only residual components."""
    if (
        not isinstance(residual, np.ndarray)
        or residual.dtype != np.dtype(np.bool_)
        or residual.ndim != 2
        or min(residual.shape) <= 0
        or not residual.flags.c_contiguous
    ):
        raise ValueError("ground residual mask is invalid")
    remaining = np.ascontiguousarray(residual.copy(), dtype=np.bool_)
    rows, columns = remaining.shape
    components: list[tuple[tuple[int, int], ...]] = []
    for start_row, start_column in zip(*np.nonzero(remaining), strict=True):
        start = (int(start_row), int(start_column))
        if not remaining[start]:
            continue
        remaining[start] = False
        queue: deque[tuple[int, int]] = deque((start,))
        cells: list[tuple[int, int]] = []
        while queue:
            row, column = queue.popleft()
            cells.append((row, column))
            for next_row, next_column in (
                (row - 1, column),
                (row, column - 1),
                (row, column + 1),
                (row + 1, column),
            ):
                if (
                    0 <= next_row < rows
                    and 0 <= next_column < columns
                    and remaining[next_row, next_column]
                ):
                    remaining[next_row, next_column] = False
                    queue.append((next_row, next_column))
        components.append(tuple(sorted(cells)))
    return tuple(components)


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
            self.selected_policy_candidate_count > _POLICY_CANDIDATE_COUNT
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
class ExhaustionUpgradeDiagnostics:
    """Observed-only work accounting for one isolated ground exhaustion scan."""

    residual_component_count: int
    reachable_pose_count: int
    endpoint_feasible_pose_count: int
    exact_gain_evaluated_pose_count: int
    positive_pose_count: int
    temporary_array_bytes: int
    elapsed_s: float

    def __post_init__(self) -> None:
        counts = (
            self.residual_component_count,
            self.reachable_pose_count,
            self.endpoint_feasible_pose_count,
            self.exact_gain_evaluated_pose_count,
            self.positive_pose_count,
            self.temporary_array_bytes,
        )
        if (
            any(type(value) is not int or value < 0 for value in counts)
            or self.endpoint_feasible_pose_count > self.reachable_pose_count
            or self.exact_gain_evaluated_pose_count
            != self.endpoint_feasible_pose_count
            or self.positive_pose_count > self.exact_gain_evaluated_pose_count
            or not math.isfinite(self.elapsed_s)
            or self.elapsed_s < 0.0
        ):
            raise ValueError("exhaustion upgrade diagnostics are invalid")


@dataclass(frozen=True, slots=True)
class ExhaustionUpgradeScanResult:
    """A non-policy observed-only scan result; it does not change termination."""

    diagnostics: ExhaustionUpgradeDiagnostics
    positive_pose_cells: tuple[tuple[int, int], ...]
    positive_gain_pairs: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        if (
            len(self.positive_pose_cells)
            != self.diagnostics.positive_pose_count
            or len(self.positive_gain_pairs) != len(self.positive_pose_cells)
            or tuple(sorted(self.positive_pose_cells)) != self.positive_pose_cells
            or len(set(self.positive_pose_cells)) != len(self.positive_pose_cells)
            or any(
                len(pair) != 2
                or not all(math.isfinite(float(value)) and float(value) >= 0.0 for value in pair)
                for pair in self.positive_gain_pairs
            )
        ):
            raise ValueError("exhaustion upgrade result is invalid")


@dataclass(frozen=True, slots=True)
class CandidateDecisionSnapshot:
    snapshot_id: str
    frontier_segment_count: int
    raw_candidate_count: int
    fine_pose_candidate_count: int
    globally_reachable_candidate_count: int
    positive_gain_candidate_count: int
    selected_policy_candidate_count: int
    untried_reserve_count: int
    planner_rejected_current_snapshot_count: int
    candidate_set_sha256: str
    global_search_call_count: int
    global_search_elapsed_s: float
    candidate_refresh_elapsed_s: float
    pipeline_kind: str = "GROUND_FRONTIER"
    representable_landing_sha256: str = _EMPTY_SHA256
    raw_known_landing_count: int = 0
    eligible_landing_count: int = 0
    predicted_positive_landing_count: int = 0
    visited_landing_count: int = 0
    scan_elapsed_s: float = 0.0
    sort_elapsed_s: float = 0.0
    top64_elapsed_s: float = 0.0
    reserve_elapsed_s: float = 0.0
    scan_complete: bool = True
    pagination_closed: bool = True
    capacity_truncated: bool = False

    def __post_init__(self) -> None:
        counts = (
            self.frontier_segment_count,
            self.raw_candidate_count,
            self.fine_pose_candidate_count,
            self.globally_reachable_candidate_count,
            self.positive_gain_candidate_count,
            self.selected_policy_candidate_count,
            self.untried_reserve_count,
            self.planner_rejected_current_snapshot_count,
            self.global_search_call_count,
        )
        if self.pipeline_kind not in {
            "GROUND_FRONTIER",
            "GROUND_EXHAUSTION",
            "HOPPER_LANDING",
        }:
            raise ValueError("candidate decision pipeline kind is invalid")
        common_invalid = (
            not _is_sha256(self.snapshot_id)
            or not _is_sha256(self.candidate_set_sha256)
            or any(type(value) is not int or value < 0 for value in counts)
            or not math.isfinite(self.candidate_refresh_elapsed_s)
            or self.candidate_refresh_elapsed_s < 0.0
        )
        if self.pipeline_kind == "HOPPER_LANDING":
            hopper_counts = (
                self.raw_known_landing_count,
                self.eligible_landing_count,
                self.predicted_positive_landing_count,
                self.visited_landing_count,
            )
            hopper_times = (
                self.scan_elapsed_s,
                self.sort_elapsed_s,
                self.top64_elapsed_s,
                self.reserve_elapsed_s,
            )
            if (
                common_invalid
                or not _is_sha256(self.representable_landing_sha256)
                or any(type(value) is not int or value < 0 for value in hopper_counts)
                or not (
                    self.raw_known_landing_count
                    >= self.eligible_landing_count
                    >= self.predicted_positive_landing_count
                )
                or self.visited_landing_count > self.eligible_landing_count
                or self.globally_reachable_candidate_count
                != self.eligible_landing_count
                or self.positive_gain_candidate_count
                != self.predicted_positive_landing_count
                or self.selected_policy_candidate_count
                > self.eligible_landing_count
                or self.untried_reserve_count
                != self.eligible_landing_count
                - self.selected_policy_candidate_count
                - self.planner_rejected_current_snapshot_count
                or self.global_search_call_count != 0
                or self.global_search_elapsed_s != 0.0
                or any(not math.isfinite(value) or value < 0.0 for value in hopper_times)
                or type(self.scan_complete) is not bool
                or type(self.pagination_closed) is not bool
                or type(self.capacity_truncated) is not bool
                or not self.scan_complete
                or not self.pagination_closed
                or self.capacity_truncated
            ):
                raise ValueError("hopper candidate decision snapshot is invalid")
            return
        if (
            common_invalid
            or not (
                self.raw_candidate_count
                >= self.fine_pose_candidate_count
                >= self.globally_reachable_candidate_count
                >= self.positive_gain_candidate_count
                >= self.selected_policy_candidate_count
            )
            or self.untried_reserve_count
            > self.positive_gain_candidate_count
            - self.selected_policy_candidate_count
            or self.planner_rejected_current_snapshot_count
            > self.positive_gain_candidate_count
            or self.global_search_call_count != 1
            or not math.isfinite(self.global_search_elapsed_s)
            or self.global_search_elapsed_s < 0.0
            or self.representable_landing_sha256 != _EMPTY_SHA256
            or any(
                value != 0
                for value in (
                    self.raw_known_landing_count,
                    self.eligible_landing_count,
                    self.predicted_positive_landing_count,
                    self.visited_landing_count,
                )
            )
            or any(
                value != 0.0
                for value in (
                    self.scan_elapsed_s,
                    self.sort_elapsed_s,
                    self.top64_elapsed_s,
                    self.reserve_elapsed_s,
                )
            )
            or self.scan_complete is not True
            or self.pagination_closed is not True
            or self.capacity_truncated is not False
        ):
            raise ValueError("candidate decision snapshot is invalid")


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
    segment_id: int = 0
    segment_candidate_rank: int = 0
    global_path_cost_m: float = 0.0
    expected_gain_m2: float = 0.0
    expected_priority_gain_m2: float = 0.0
    risk: float = 0.0

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
            or type(self.segment_id) is not int
            or self.segment_id < 0
            or type(self.segment_candidate_rank) is not int
            or not 0 <= self.segment_candidate_rank < 3
            or not math.isfinite(self.global_path_cost_m)
            or self.global_path_cost_m < 0.0
            or not math.isfinite(self.expected_gain_m2)
            or self.expected_gain_m2 < 0.0
            or not math.isfinite(self.expected_priority_gain_m2)
            or self.expected_priority_gain_m2 < 0.0
            or not math.isfinite(self.risk)
            or self.risk < 0.0
        ):
            raise ValueError("physical candidate fields are invalid")
        feature = np.ascontiguousarray(self.feature, dtype=np.float32)
        if feature.shape != (len(ObservationContractV4.frontier_fields),) or not np.isfinite(feature).all():
            raise ValueError("physical candidate feature must be finite float32 [12]")
        feature = feature.copy()
        feature.setflags(write=False)
        object.__setattr__(self, "feature", feature)


def _balanced_candidate_order(
    candidates: Collection[object], *, limit: int = _POLICY_CANDIDATE_COUNT
) -> list[object]:
    """Order Top-N by segment rounds and keep a deterministic reserve."""
    if type(limit) is not int or limit <= 0:
        raise ValueError("candidate policy limit must be positive")
    by_segment: dict[int, list[object]] = {}
    for candidate in candidates:
        segment_id = getattr(candidate, "segment_id", None)
        if type(segment_id) is not int or segment_id < 0:
            raise ValueError("candidate segment identity is invalid")
        by_segment.setdefault(segment_id, []).append(candidate)
    for segment in by_segment.values():
        segment.sort(
            key=lambda candidate: (
                -float(getattr(candidate, "expected_gain_m2")),
                float(getattr(candidate, "global_path_cost_m")),
                float(getattr(candidate, "risk")),
                str(getattr(candidate, "candidate_id")),
            )
        )
    if not by_segment:
        return []
    segment_ids = sorted(by_segment)
    representatives = [by_segment[segment_id][0] for segment_id in segment_ids]
    selected: list[object] = []
    reserve: list[object] = []
    if len(representatives) > limit:
        remaining = sorted(
            representatives,
            key=lambda candidate: str(getattr(candidate, "candidate_id")),
        )
        selected.append(remaining.pop(0))
        while remaining and len(selected) < limit:
            index = min(
                range(len(remaining)),
                key=lambda item: (
                    -min(
                        _candidate_distance_squared(remaining[item], chosen)
                        for chosen in selected
                    ),
                    str(getattr(remaining[item], "candidate_id")),
                ),
            )
            selected.append(remaining.pop(index))
        reserve.extend(
            sorted(
                remaining,
                key=lambda candidate: str(getattr(candidate, "candidate_id")),
            )
        )
        for rank in range(1, 3):
            reserve.extend(
                by_segment[segment_id][rank]
                for segment_id in segment_ids
                if len(by_segment[segment_id]) > rank
            )
        return [*selected, *reserve]

    maximum_rank = max(len(segment) for segment in by_segment.values())
    rounds = [
        by_segment[segment_id][rank]
        for rank in range(maximum_rank)
        for segment_id in segment_ids
        if len(by_segment[segment_id]) > rank
    ]
    return rounds[:limit] + rounds[limit:]


@dataclass(frozen=True, slots=True)
class PhysicalCandidateUniverse:
    physical_snapshot_id: str
    physical_reachability_algorithm_id: str
    candidates: tuple[PhysicalCandidate, ...]
    universe_sha256: str
    diagnostics: CandidateDiagnostics
    decision_snapshot: CandidateDecisionSnapshot | None = None

    def __post_init__(self) -> None:
        if (
            not _is_sha256(self.physical_snapshot_id)
            or not isinstance(self.physical_reachability_algorithm_id, str)
            or not self.physical_reachability_algorithm_id
            or not isinstance(self.candidates, tuple)
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
            or (
                self.decision_snapshot is not None
                and not isinstance(
                    self.decision_snapshot, CandidateDecisionSnapshot
                )
            )
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
        if features.shape != (64, len(ObservationContractV4.frontier_fields)) or mask.shape != (64,) or not np.isfinite(features).all():
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


def _frontier_chains(
    points: Collection[tuple[int, int]], cells: int
) -> list[list[tuple[int, int]]]:
    """Split a four-neighbour frontier graph into canonical maximal chains."""
    point_set = set(points)
    if (
        type(cells) is not int
        or cells <= 0
        or any(
            not isinstance(point, tuple)
            or len(point) != 2
            or any(type(value) is not int for value in point)
            or not 0 <= point[0] < cells
            or not 0 <= point[1] < cells
            for point in point_set
        )
    ):
        raise ValueError("frontier graph is invalid")
    adjacency = {
        point: tuple(
            neighbor
            for neighbor in _neighbors(*point, cells)
            if neighbor in point_set
        )
        for point in point_set
    }
    boundary = {point for point, neighbors in adjacency.items() if len(neighbors) != 2}
    visited_edges: set[frozenset[tuple[int, int]]] = set()
    chains: list[list[tuple[int, int]]] = []

    def edge(lhs: tuple[int, int], rhs: tuple[int, int]) -> frozenset[tuple[int, int]]:
        return frozenset((lhs, rhs))

    for start in sorted(boundary):
        if not adjacency[start]:
            chains.append([start])
            continue
        for neighbor in adjacency[start]:
            if edge(start, neighbor) in visited_edges:
                continue
            chain = [start]
            previous, current = start, neighbor
            visited_edges.add(edge(previous, current))
            while True:
                chain.append(current)
                if current in boundary:
                    break
                following = next(
                    item for item in adjacency[current] if item != previous
                )
                previous, current = current, following
                visited_edges.add(edge(previous, current))
            if chain[-1] < chain[0]:
                chain.reverse()
            elif chain[-1] == chain[0] and tuple(reversed(chain)) < tuple(chain):
                chain.reverse()
            chains.append(chain)

    remaining = point_set - {
        point for chain in chains for point in chain
    }
    while remaining:
        start = min(remaining)
        first = min(adjacency[start])
        chain = [start]
        previous, current = start, first
        visited_edges.add(edge(previous, current))
        while current != start:
            chain.append(current)
            following = next(
                item for item in adjacency[current] if item != previous
            )
            previous, current = current, following
            visited_edges.add(edge(previous, current))
        reverse = [start, *reversed(chain[1:])]
        if tuple(reverse) < tuple(chain):
            chain = reverse
        chains.append(chain)
        remaining.difference_update(chain)
    return [list(chain) for chain in sorted(tuple(chain) for chain in chains)]


def _sample_frontier_chain(
    chain: Collection[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Return stable quarter/half/three-quarter cells without duplication."""
    ordered = list(chain)
    if not ordered:
        return []
    cumulative = [0.0]
    for previous, current in zip(ordered, ordered[1:], strict=False):
        cumulative.append(cumulative[-1] + math.dist(previous, current))
    total = cumulative[-1]
    selected: list[tuple[int, int]] = []
    for fraction in (0.25, 0.5, 0.75):
        target = total * fraction
        index = min(
            range(len(ordered)),
            key=lambda value: (abs(cumulative[value] - target), value),
        )
        if ordered[index] not in selected:
            selected.append(ordered[index])
    return selected


def _map_frontier_chain_candidates(
    chain: Collection[tuple[int, int]],
    *,
    robot: tuple[int, int],
    observed_mask: np.ndarray,
    physical_pose_mask: np.ndarray,
    standoff_cells: int,
) -> list[_RawFrontierCandidate]:
    """Map the three samples once; rejected samples are never backfilled."""
    if (
        not isinstance(observed_mask, np.ndarray)
        or observed_mask.dtype != np.dtype(np.bool_)
        or observed_mask.ndim != 2
        or not isinstance(physical_pose_mask, np.ndarray)
        or physical_pose_mask.dtype != np.dtype(np.bool_)
        or physical_pose_mask.shape != observed_mask.shape
        or type(standoff_cells) is not int
        or standoff_cells < 0
    ):
        raise ValueError("frontier candidate masks are invalid")
    rows, columns = observed_mask.shape
    output: list[_RawFrontierCandidate] = []
    seen: set[tuple[int, int]] = set()
    for sample_rank, frontier_cell in enumerate(_sample_frontier_chain(chain)):
        row, column = frontier_cell
        pose_cell = (
            row + int(np.sign(robot[0] - row)) * standoff_cells,
            column + int(np.sign(robot[1] - column)) * standoff_cells,
        )
        if not (
            0 <= pose_cell[0] < rows
            and 0 <= pose_cell[1] < columns
            and observed_mask[pose_cell]
        ):
            pose_cell = frontier_cell
        if (
            pose_cell in seen
            or not 0 <= pose_cell[0] < rows
            or not 0 <= pose_cell[1] < columns
            or not observed_mask[pose_cell]
            or not physical_pose_mask[pose_cell]
        ):
            continue
        seen.add(pose_cell)
        output.append(
            _RawFrontierCandidate(
                sample_rank=sample_rank,
                frontier_cell=frontier_cell,
                pose_cell=pose_cell,
            )
        )
    return output


def _map_frontier_chain_pose_options(
    chain: Collection[tuple[int, int]],
    *,
    robot: tuple[int, int],
    observed_mask: np.ndarray,
    physical_pose_mask: np.ndarray,
    standoff_cells: int,
) -> list[_RawFrontierCandidate]:
    """Enumerate internal pose options along one chain before slot selection."""
    if (
        not isinstance(observed_mask, np.ndarray)
        or observed_mask.dtype != np.dtype(np.bool_)
        or observed_mask.ndim != 2
        or not isinstance(physical_pose_mask, np.ndarray)
        or physical_pose_mask.dtype != np.dtype(np.bool_)
        or physical_pose_mask.shape != observed_mask.shape
        or type(standoff_cells) is not int
        or standoff_cells < 0
    ):
        raise ValueError("frontier pose-option masks are invalid")
    rows, columns = observed_mask.shape
    output: list[_RawFrontierCandidate] = []
    seen: set[tuple[int, int]] = set()
    for chain_rank, frontier_cell in enumerate(chain):
        row, column = frontier_cell
        pose_cell = (
            row + int(np.sign(robot[0] - row)) * standoff_cells,
            column + int(np.sign(robot[1] - column)) * standoff_cells,
        )
        if not (
            0 <= pose_cell[0] < rows
            and 0 <= pose_cell[1] < columns
            and observed_mask[pose_cell]
        ):
            pose_cell = frontier_cell
        if (
            pose_cell in seen
            or not 0 <= pose_cell[0] < rows
            or not 0 <= pose_cell[1] < columns
            or not observed_mask[pose_cell]
            or not physical_pose_mask[pose_cell]
        ):
            continue
        seen.add(pose_cell)
        output.append(
            _RawFrontierCandidate(
                sample_rank=chain_rank,
                frontier_cell=frontier_cell,
                pose_cell=pose_cell,
            )
        )
    return output


def _select_three_chain_options(
    options: Collection[tuple[_RawFrontierCandidate, int, float]],
    *,
    chain_length: int,
) -> tuple[int, ...]:
    """Choose at most three qualified positions, anchor-first and stable.

    A detailed safe strip can yield many positions for one of the three
    sampled frontier anchors.  It is an internal feasibility fan-out, not a
    request to give that anchor three policy actions.  Select the highest-gain
    qualified position from each distinct anchor before backfilling a missing
    anchor with another qualified witness.
    """
    if type(chain_length) is not int or chain_length <= 0:
        raise ValueError("frontier chain length is invalid")
    remaining = sorted(
        options,
        key=lambda item: (
            item[0].sample_rank,
            -float(item[2]),
            item[0].pose_cell,
            item[1],
        ),
    )
    if any(
        not math.isfinite(float(item[2])) or float(item[2]) < 0.0
        for item in remaining
    ):
        raise ValueError("frontier option gain is invalid")
    by_anchor: dict[int, list[tuple[_RawFrontierCandidate, int, float]]] = {}
    for item in remaining:
        by_anchor.setdefault(item[0].sample_rank, []).append(item)
    anchor_ranks = tuple(sorted(by_anchor))
    selected: list[int] = []
    selected_anchor_ranks: set[int] = set()
    for fraction in (0.25, 0.5, 0.75):
        if not anchor_ranks:
            break
        target_index = min(
            len(anchor_ranks) - 1,
            int(fraction * len(anchor_ranks)),
        )
        rank = anchor_ranks[target_index]
        if rank in selected_anchor_ranks:
            continue
        raw, option_index, _ = by_anchor[rank][0]
        selected.append(option_index)
        selected_anchor_ranks.add(raw.sample_rank)
    for raw, option_index, _ in remaining:
        if len(selected) == 3:
            break
        if option_index not in selected:
            selected.append(option_index)
    return tuple(selected)


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

    def scan_ground_exhaustion_candidates(
        self,
        world: ObservedWorld,
        mission: MissionRaster,
        *,
        reachable_pose_mask: np.ndarray,
        observation_positions_m: np.ndarray | None = None,
        ground_endpoint_feasibility: (
            Callable[[np.ndarray], np.ndarray] | None
        ) = None,
    ) -> ExhaustionUpgradeScanResult:
        """Measure all currently reachable observed-only residual observers.

        This deliberately has no platform action, candidate-batch, checkpoint,
        or terminal side effect.  It is the pre-integration scan used to bound
        the work required when normal frontier candidates are exhausted.
        """
        started = perf_counter()
        if (
            not isinstance(world, ObservedWorld)
            or not isinstance(mission, MissionRaster)
            or world.canvas != mission.canvas
        ):
            raise ValueError("exhaustion scan inputs must share a map canvas")
        reachable = np.asarray(reachable_pose_mask)
        cells = world.canvas.geometry.cells
        if (
            reachable.dtype != np.dtype(np.bool_)
            or reachable.shape != (cells, cells)
            or not reachable.flags.c_contiguous
        ):
            raise ValueError("exhaustion scan reachable pose mask is invalid")
        reachable_count = int(reachable.sum(dtype=np.int64))
        positions = observation_positions_m
        if positions is not None and (
            not isinstance(positions, np.ndarray)
            or positions.dtype != np.dtype(np.float64)
            or positions.shape != (reachable_count, 3)
            or not positions.flags.c_contiguous
            or not np.isfinite(positions).all()
        ):
            raise ValueError("exhaustion scan observation positions are invalid")
        if ground_endpoint_feasibility is not None and positions is None:
            raise ValueError(
                "exhaustion scan endpoint feasibility requires observation positions"
            )
        residual = np.ascontiguousarray(
            (mission.roi_ratio > 0.0) & ~world.observed_mask,
            dtype=np.bool_,
        )
        components = _ground_residual_components(residual)
        potential = _ground_potential_gain_mask(
            residual,
            sensor_range_m=self._sensor.range_m,
            resolution_m=world.canvas.geometry.resolution_m,
        )
        eligible = np.ascontiguousarray(reachable & potential, dtype=np.bool_)
        pose_cells = np.ascontiguousarray(
            np.argwhere(eligible), dtype=np.int32
        ).reshape((-1, 2))
        if positions is None:
            endpoint_positions = np.empty((len(pose_cells), 3), dtype=np.float64)
        else:
            reachable_cells = np.argwhere(reachable)
            reachable_codes = np.ascontiguousarray(
                reachable_cells[:, 0].astype(np.int64) * cells
                + reachable_cells[:, 1],
                dtype=np.int64,
            )
            pose_codes = np.ascontiguousarray(
                pose_cells[:, 0].astype(np.int64) * cells + pose_cells[:, 1],
                dtype=np.int64,
            )
            position_indices = np.searchsorted(reachable_codes, pose_codes)
            if (
                (position_indices >= len(reachable_codes)).any()
                or not np.array_equal(reachable_codes[position_indices], pose_codes)
            ):
                raise CandidateInvariantError(
                    "GROUND_EXHAUSTION_POSITION_ALIGNMENT_INVALID"
                )
            endpoint_positions = np.ascontiguousarray(
                positions[position_indices], dtype=np.float64
            )
        if ground_endpoint_feasibility is None:
            endpoint_mask = np.ones(len(pose_cells), dtype=np.bool_)
        else:
            endpoint_mask = ground_endpoint_feasibility(endpoint_positions)
            if (
                not isinstance(endpoint_mask, np.ndarray)
                or endpoint_mask.dtype != np.dtype(np.bool_)
                or endpoint_mask.shape != (len(pose_cells),)
                or not endpoint_mask.flags.c_contiguous
            ):
                raise CandidateInvariantError(
                    "GROUND_ENDPOINT_FEASIBILITY_INVALID"
                )
        endpoint_cells = np.ascontiguousarray(
            pose_cells[endpoint_mask], dtype=np.int32
        ).reshape((-1, 2))
        endpoint_positions = np.ascontiguousarray(
            endpoint_positions[endpoint_mask], dtype=np.float64
        ).reshape((-1, 3))
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
        exact_gain = getattr(
            self._visibility_estimator,
            "estimate_candidate_gains_at_positions",
            None,
        )
        if not len(endpoint_cells):
            gains = np.zeros((0, 2), dtype=np.float32)
        elif positions is not None and callable(exact_gain):
            gains = exact_gain(*gain_arguments, endpoint_positions)
        else:
            gains = self._visibility_estimator.estimate_candidate_gains(
                *gain_arguments, endpoint_cells
            )
        if (
            not isinstance(gains, np.ndarray)
            or gains.dtype != np.dtype(np.float32)
            or gains.shape != (len(endpoint_cells), 2)
            or not gains.flags.c_contiguous
            or not np.isfinite(gains).all()
            or (gains < 0.0).any()
        ):
            raise CandidateInvariantError("CANDIDATE_GAIN_NONFINITE")
        positive_rows = [
            (int(row), int(column), float(gain[0]), float(gain[1]))
            for (row, column), gain in zip(endpoint_cells, gains, strict=True)
            if float(gain[0]) >= _DETAIL_CELL_COARSE_EQUIVALENT
        ]
        positive = tuple((row, column) for row, column, _, _ in positive_rows)
        positive_gains = tuple(
            (gain, priority_gain)
            for _, _, gain, priority_gain in positive_rows
        )
        temporary_bytes = int(
            residual.nbytes + potential.nbytes + eligible.nbytes
            + pose_cells.nbytes + endpoint_cells.nbytes + endpoint_positions.nbytes
            + gains.nbytes
        )
        return ExhaustionUpgradeScanResult(
            diagnostics=ExhaustionUpgradeDiagnostics(
                residual_component_count=len(components),
                reachable_pose_count=len(pose_cells),
                endpoint_feasible_pose_count=len(endpoint_cells),
                exact_gain_evaluated_pose_count=len(endpoint_cells),
                positive_pose_count=len(positive),
                temporary_array_bytes=temporary_bytes,
                elapsed_s=perf_counter() - started,
            ),
            positive_pose_cells=positive,
            positive_gain_pairs=positive_gains,
        )

    def build_ground_exhaustion_universe(
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
        ground_endpoint_feasibility: (
            Callable[[np.ndarray], np.ndarray] | None
        ),
    ) -> PhysicalCandidateUniverse:
        """Build a formal ground universe from one completed upgrade scan."""
        started = perf_counter()
        if platform_type not in _GROUND_PLATFORM_TYPES:
            raise ValueError("ground exhaustion universe requires a ground platform")
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
            or not isinstance(physical_reachability, PhysicalReachabilityResult)
            or not isinstance(pose_map, Pose2)
            or pose_map.frame_id != "map"
            or world.canvas != mission.canvas
            or world.canvas != projection.canvas
            or physical_reachability.platform_type != platform_type
            or physical_reachability.physical_reachability_algorithm_id
            != physical_reachability_algorithm_id
        ):
            raise ValueError("ground exhaustion universe inputs are inconsistent")
        physical_mask = physical_reachability.physical_observation_pose_mask
        cells = world.canvas.geometry.cells
        if physical_mask.shape != (cells, cells):
            raise ValueError("ground exhaustion mask geometry differs")
        physical_cells = tuple(
            (int(row), int(column))
            for row, column in zip(*np.nonzero(physical_mask), strict=True)
        )
        positions = physical_reachability.observation_positions_m
        if len(physical_cells) != len(positions):
            raise ValueError("ground exhaustion positions differ from mask")
        exact_target_poses: dict[tuple[int, int], Pose2] = {}
        for cell, position in zip(physical_cells, positions, strict=True):
            if world.canvas.world_to_grid(float(position[0]), float(position[1])) != cell:
                raise ValueError("ground exhaustion position leaves its cell")
            exact_target_poses[cell] = Pose2(
                float(position[0]),
                float(position[1]),
                elevation_m=float(position[2]),
            )
        ground_global_search = physical_reachability.ground_global_search
        if ground_global_search is None:
            raise CandidateInvariantError("GLOBAL_PATH_COST_NONFINITE")
        scan = self.scan_ground_exhaustion_candidates(
            world,
            mission,
            reachable_pose_mask=physical_mask,
            observation_positions_m=positions,
            ground_endpoint_feasibility=ground_endpoint_feasibility,
        )
        minimum_cost = ground_global_search.sampled_minimum_cost_m
        total_roi = float(mission.roi_ratio.sum(dtype=np.float64))
        roi_diagonal = task_roi_diagonal_m(
            mission.roi_ratio,
            resolution_m=world.canvas.geometry.resolution_m,
        )
        ranked_scan = sorted(
            (
                (
                    cell,
                    float(gain_pair[0]),
                    float(gain_pair[1]),
                    float(minimum_cost[cell]),
                    max(0.0, 1.0 - float(projection.clearance_margin_norm[cell])),
                )
                for cell, gain_pair in zip(
                    scan.positive_pose_cells,
                    scan.positive_gain_pairs,
                    strict=True,
                )
                if math.isfinite(float(minimum_cost[cell]))
            ),
            key=lambda item: (-item[1], item[3], item[4], item[0]),
        )
        positive_gain_normalizer = max((item[1] for item in ranked_scan), default=0.0)
        priority_gain_normalizer = max((item[2] for item in ranked_scan), default=0.0)
        preliminary: list[PhysicalCandidate] = []
        for segment_id, (cell, gain, priority_gain, global_cost, risk) in enumerate(
            ranked_scan
        ):
            target_pose = exact_target_poses[cell]
            feature = self._feature(
                world,
                mission,
                projection,
                pose_map,
                cell,
                total_roi,
                gain,
                priority_gain,
                positive_gain_normalizer,
                priority_gain_normalizer,
                target_pose=target_pose,
                global_path_cost_norm=normalize_global_path_cost(
                    global_cost,
                    roi_diagonal,
                    _NOMINAL_GLOBAL_COST_PER_M,
                ),
            )
            if feature is None:
                continue
            preliminary.append(
                self._physical_candidate(
                    _FeasibleAnchor(
                        segment_id=segment_id,
                        point=cell,
                        feature=feature,
                        elevation_m=float(target_pose.elevation_m),
                        target_position_m=(
                            float(target_pose.x_m),
                            float(target_pose.y_m),
                            float(target_pose.elevation_m),
                        ),
                    ),
                    pose_map=pose_map,
                    platform_type=platform_type,
                    platform_id=platform_id,
                    mission_revision=mission_revision,
                    goal_tolerance_mm=goal_tolerance_mm,
                    segment_candidate_rank=0,
                    global_path_cost_m=global_cost,
                    expected_gain_m2=(
                        gain / _DETAIL_CELL_COARSE_EQUIVALENT
                    ) * _DETAIL_CELL_AREA_M2,
                    expected_priority_gain_m2=(
                        priority_gain / _DETAIL_CELL_COARSE_EQUIVALENT
                    ) * _DETAIL_CELL_AREA_M2,
                    risk=risk,
                )
            )
        compressed = _compress_physical_candidates(
            preliminary, canvas_cells=cells
        )
        candidates = tuple(
            replace(candidate, rank_key=(index, candidate.candidate_id))
            for index, candidate in enumerate(compressed)
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
        selected_count = min(_POLICY_CANDIDATE_COUNT, len(candidates))
        universe_sha256 = sha256(
            "".join(candidate.candidate_id for candidate in candidates).encode("ascii")
        ).hexdigest()
        diagnostics = CandidateDiagnostics(
            physical_snapshot_id=physical_snapshot_id,
            physical_reachability_algorithm_id=physical_reachability_algorithm_id,
            physical_candidate_universe_count=len(candidates),
            selected_policy_candidate_count=selected_count,
            available_candidate_count=len(candidates),
            untried_reserve_count=len(candidates) - selected_count,
            planner_failed_current_snapshot_count=0,
            zero_gain_count=(
                scan.diagnostics.endpoint_feasible_pose_count
                - scan.diagnostics.positive_pose_count
            ),
            visited_excluded_count=0,
            physical_unreachable_count=(
                physical_reachability.physical_safe_pose_count
                - physical_reachability.physically_reachable_pose_count
            ),
        )
        snapshot = CandidateDecisionSnapshot(
            snapshot_id=physical_snapshot_id,
            frontier_segment_count=scan.diagnostics.residual_component_count,
            raw_candidate_count=scan.diagnostics.reachable_pose_count,
            fine_pose_candidate_count=scan.diagnostics.reachable_pose_count,
            globally_reachable_candidate_count=(
                scan.diagnostics.endpoint_feasible_pose_count
            ),
            positive_gain_candidate_count=len(candidates),
            selected_policy_candidate_count=selected_count,
            untried_reserve_count=len(candidates) - selected_count,
            planner_rejected_current_snapshot_count=0,
            candidate_set_sha256=universe_sha256,
            global_search_call_count=1,
            global_search_elapsed_s=ground_global_search.search_elapsed_s,
            candidate_refresh_elapsed_s=perf_counter() - started,
            pipeline_kind="GROUND_EXHAUSTION",
        )
        return PhysicalCandidateUniverse(
            physical_snapshot_id=physical_snapshot_id,
            physical_reachability_algorithm_id=physical_reachability_algorithm_id,
            candidates=candidates,
            universe_sha256=universe_sha256,
            diagnostics=diagnostics,
            decision_snapshot=snapshot,
        )

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
        ground_endpoint_feasibility: (
            Callable[[np.ndarray], np.ndarray] | None
        ) = None,
        ground_detail_candidate_provider: (
            Callable[
                [
                    list[list[tuple[int, int]]],
                    ObservedWorld,
                ],
                Collection[tuple[int, _RawFrontierCandidate]],
            ]
            | None
        ) = None,
    ) -> PhysicalCandidateUniverse:
        """Build one primitive-independent, bounded physical opportunity set."""
        refresh_started = perf_counter()
        del backtrack_pose
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

        if platform_type == "HOPPER":
            return self._build_hopper_physical_universe(
                world=world,
                mission=mission,
                pose_map=pose_map,
                projection=projection,
                physical_reachability=physical_reachability,
                platform_id=platform_id,
                capability_content_sha256=capability_content_sha256,
                mission_revision=mission_revision,
                evidence_generation=evidence_generation,
                physical_evidence_sha256=physical_evidence_sha256,
                physical_reachability_algorithm_id=(
                    physical_reachability_algorithm_id
                ),
                goal_tolerance_mm=goal_tolerance_mm,
                robot=robot,
                exact_target_poses=exact_target_poses,
                excluded_cells=excluded_cells,
                refresh_started=refresh_started,
            )

        observed = world.observed_mask
        roi = mission.roi_ratio > 0.0
        unknown_roi = roi & ~observed
        adjacent_unknown = np.zeros_like(observed)
        adjacent_unknown[1:] |= unknown_roi[:-1]
        adjacent_unknown[:-1] |= unknown_roi[1:]
        adjacent_unknown[:, 1:] |= unknown_roi[:, :-1]
        adjacent_unknown[:, :-1] |= unknown_roi[:, 1:]
        boundary = observed & roi & adjacent_unknown
        segments = _frontier_chains(_points(boundary), cells)
        if platform_type in _GROUND_PLATFORM_TYPES:
            normal_universe = self._build_ground_physical_universe(
                world=world,
                mission=mission,
                pose_map=pose_map,
                projection=projection,
                physical_reachability=physical_reachability,
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
                robot=robot,
                segments=segments,
                exact_target_poses=exact_target_poses,
                ground_endpoint_feasibility=ground_endpoint_feasibility,
                ground_detail_candidate_provider=(
                    ground_detail_candidate_provider
                ),
                refresh_started=refresh_started,
            )
            normal_snapshot = normal_universe.decision_snapshot
            if (
                normal_snapshot is None
                or normal_snapshot.positive_gain_candidate_count > 0
            ):
                return normal_universe
            return self.build_ground_exhaustion_universe(
                world,
                mission,
                pose_map,
                projection,
                physical_reachability=physical_reachability,
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
                ground_endpoint_feasibility=ground_endpoint_feasibility,
            )
    def _build_hopper_physical_universe(
        self,
        *,
        world: ObservedWorld,
        mission: MissionRaster,
        pose_map: Pose2,
        projection: PlatformProjection,
        physical_reachability: PhysicalReachabilityResult,
        platform_id: str,
        capability_content_sha256: str,
        mission_revision: int,
        evidence_generation: int,
        physical_evidence_sha256: str,
        physical_reachability_algorithm_id: str,
        goal_tolerance_mm: int,
        robot: tuple[int, int],
        exact_target_poses: Mapping[tuple[int, int], Pose2],
        excluded_cells: Collection[tuple[int, int]],
        refresh_started: float,
    ) -> PhysicalCandidateUniverse:
        """Build the complete homogeneous HOPPER landing universe."""
        envelope = physical_reachability.hopper_single_hop_envelope
        if (
            not isinstance(envelope, HopperSingleHopEnvelope)
            or not envelope.complete
        ):
            raise CandidateInvariantError("HOPPER_ENVELOPE_INCOMPLETE")
        canvas = world.canvas
        if (
            envelope.eligible_mask.shape != world.observed_mask.shape
            or not np.array_equal(
                envelope.eligible_mask,
                physical_reachability.physical_observation_pose_mask,
            )
        ):
            raise CandidateInvariantError("HOPPER_ENVELOPE_GEOMETRY_DIFFERS")

        scan_started = perf_counter()
        representable = np.zeros_like(envelope.eligible_mask, dtype=np.bool_)
        current_position = (
            float(pose_map.x_m),
            float(pose_map.y_m),
            float(pose_map.elevation_m),
        )
        eligible_cells: list[tuple[int, int]] = []
        eligible_positions: list[tuple[float, float, float]] = []
        for cell in sorted(exact_target_poses):
            target = exact_target_poses[cell]
            position = (
                float(target.x_m),
                float(target.y_m),
                float(target.elevation_m),
            )
            if (
                not envelope.eligible_mask[cell]
                or mission.roi_ratio[cell] <= 0.0
                or position == current_position
                or not all(world.observed_mask[item] for item in _ray_cells(robot, cell))
            ):
                continue
            representable[cell] = True
            eligible_cells.append(cell)
            eligible_positions.append(position)

        positions_array = np.asarray(
            eligible_positions, dtype=np.float64
        ).reshape((-1, 3))
        gains = self._hopper_predicted_capsule_gains(
            world=world,
            mission=mission,
            pose_map=pose_map,
            target_positions_m=positions_array,
        )
        scan_elapsed = perf_counter() - scan_started
        gain_normalizer = float(gains[:, 0].max(initial=0.0))
        priority_gain_normalizer = float(gains[:, 1].max(initial=0.0))
        total_roi = float(mission.roi_ratio.sum(dtype=np.float64))
        roi_diagonal = task_roi_diagonal_m(
            mission.roi_ratio,
            resolution_m=canvas.geometry.resolution_m,
        )
        preliminary: list[PhysicalCandidate] = []
        for cell, position, gain_pair in zip(
            eligible_cells, eligible_positions, gains, strict=True
        ):
            target_pose = Pose2(
                position[0], position[1], elevation_m=position[2]
            )
            required_delta_v = float(envelope.required_delta_v_mps[cell])
            distance = math.dist(current_position, position)
            feature = self._feature(
                world,
                mission,
                projection,
                pose_map,
                cell,
                total_roi,
                float(gain_pair[0]),
                float(gain_pair[1]),
                gain_normalizer,
                priority_gain_normalizer,
                allow_zero_gain=True,
                target_pose=target_pose,
                global_path_cost_norm=normalize_global_path_cost(
                    required_delta_v,
                    roi_diagonal,
                    _NOMINAL_GLOBAL_COST_PER_M,
                ),
            )
            if feature is None:
                raise CandidateInvariantError("HOPPER_FEATURE_MISSING")
            anchor = _FeasibleAnchor(
                segment_id=0,
                point=cell,
                feature=feature,
                elevation_m=position[2],
                target_position_m=position,
            )
            candidate = self._physical_candidate(
                anchor,
                pose_map=pose_map,
                platform_type="HOPPER",
                platform_id=platform_id,
                mission_revision=mission_revision,
                goal_tolerance_mm=goal_tolerance_mm,
                global_path_cost_m=distance,
                expected_gain_m2=(
                    float(gain_pair[0])
                    * canvas.geometry.resolution_m**2
                ),
                expected_priority_gain_m2=(
                    float(gain_pair[1])
                    * canvas.geometry.resolution_m**2
                ),
                risk=max(
                    0.0,
                    1.0 - float(projection.clearance_margin_norm[cell]),
                ),
            )
            preliminary.append(
                replace(
                    candidate,
                    rank_key=(
                        -float(gain_pair[0]),
                        -float(gain_pair[1]),
                        required_delta_v,
                        candidate.risk,
                        candidate.candidate_id,
                    ),
                )
            )

        sort_started = perf_counter()
        base_order = sorted(preliminary, key=lambda item: item.rank_key)
        sort_elapsed = perf_counter() - sort_started
        top64_started = perf_counter()
        selected_order = _stable_farthest_candidates(
            base_order, min(_POLICY_CANDIDATE_COUNT, len(base_order))
        )
        top64_elapsed = perf_counter() - top64_started
        reserve_started = perf_counter()
        selected_ids = {candidate.candidate_id for candidate in selected_order}
        reserve = [
            candidate
            for candidate in base_order
            if candidate.candidate_id not in selected_ids
        ]
        ordered = [*selected_order, *reserve]
        candidates = tuple(
            replace(candidate, rank_key=(index, candidate.candidate_id))
            for index, candidate in enumerate(ordered)
        )
        reserve_elapsed = perf_counter() - reserve_started
        physical_snapshot_id = self._physical_snapshot_id(
            platform_type="HOPPER",
            platform_id=platform_id,
            capability_content_sha256=capability_content_sha256,
            mission_revision=mission_revision,
            pose_map=pose_map,
            evidence_generation=evidence_generation,
            physical_evidence_sha256=physical_evidence_sha256,
        )
        selected_count = min(_POLICY_CANDIDATE_COUNT, len(candidates))
        positive_count = sum(
            candidate.expected_gain_m2 > 0.0 for candidate in candidates
        )
        visited = set(excluded_cells)
        visited_count = sum(
            candidate.position_grid_key in visited for candidate in candidates
        )
        universe_sha256 = sha256(
            "".join(candidate.candidate_id for candidate in candidates).encode(
                "ascii"
            )
        ).hexdigest()
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
            zero_gain_count=len(candidates) - positive_count,
            visited_excluded_count=0,
            physical_unreachable_count=(
                envelope.raw_known_landing_count - len(candidates)
            ),
        )
        decision_snapshot = CandidateDecisionSnapshot(
            snapshot_id=physical_snapshot_id,
            frontier_segment_count=0,
            raw_candidate_count=envelope.raw_known_landing_count,
            fine_pose_candidate_count=int(
                envelope.certified_mask.sum(dtype=np.int64)
            ),
            globally_reachable_candidate_count=len(candidates),
            positive_gain_candidate_count=positive_count,
            selected_policy_candidate_count=selected_count,
            untried_reserve_count=len(candidates) - selected_count,
            planner_rejected_current_snapshot_count=0,
            candidate_set_sha256=universe_sha256,
            global_search_call_count=0,
            global_search_elapsed_s=0.0,
            candidate_refresh_elapsed_s=perf_counter() - refresh_started,
            pipeline_kind="HOPPER_LANDING",
            representable_landing_sha256=sha256(
                np.packbits(representable, bitorder="little").tobytes()
            ).hexdigest(),
            raw_known_landing_count=envelope.raw_known_landing_count,
            eligible_landing_count=len(candidates),
            predicted_positive_landing_count=positive_count,
            visited_landing_count=visited_count,
            scan_elapsed_s=scan_elapsed,
            sort_elapsed_s=sort_elapsed,
            top64_elapsed_s=top64_elapsed,
            reserve_elapsed_s=reserve_elapsed,
            scan_complete=True,
            pagination_closed=True,
            capacity_truncated=False,
        )
        return PhysicalCandidateUniverse(
            physical_snapshot_id=physical_snapshot_id,
            physical_reachability_algorithm_id=(
                physical_reachability_algorithm_id
            ),
            candidates=candidates,
            universe_sha256=universe_sha256,
            diagnostics=diagnostics,
            decision_snapshot=decision_snapshot,
        )

    def _hopper_predicted_capsule_gains(
        self,
        *,
        world: ObservedWorld,
        mission: MissionRaster,
        pose_map: Pose2,
        target_positions_m: np.ndarray,
    ) -> np.ndarray:
        """Predict no-occlusion 4 m task gain along each horizontal segment."""
        positions = np.ascontiguousarray(target_positions_m, dtype=np.float64)
        gains = np.zeros((len(positions), 2), dtype=np.float32)
        unknown = (mission.roi_ratio > 0.0) & ~world.observed_mask
        if not len(positions) or not unknown.any():
            return gains
        rows, columns = np.nonzero(unknown)
        resolution = world.canvas.geometry.resolution_m
        left, _, _, top = world.canvas.bounds_m
        unknown_x = left + (columns.astype(np.float64) + 0.5) * resolution
        unknown_y = top - (rows.astype(np.float64) + 0.5) * resolution
        roi_weight = mission.roi_ratio[rows, columns].astype(np.float64)
        priority_weight = (
            mission.priority[rows, columns].astype(np.float64) * roi_weight
        )
        radius = float(self._sensor.range_m)
        radius_squared = radius * radius
        start_x = float(pose_map.x_m)
        start_y = float(pose_map.y_m)
        for index, position in enumerate(positions):
            delta_x = float(position[0]) - start_x
            delta_y = float(position[1]) - start_y
            candidate_box = (
                (unknown_x >= min(start_x, float(position[0])) - radius)
                & (unknown_x <= max(start_x, float(position[0])) + radius)
                & (unknown_y >= min(start_y, float(position[1])) - radius)
                & (unknown_y <= max(start_y, float(position[1])) + radius)
            )
            candidate_indices = np.flatnonzero(candidate_box)
            if not len(candidate_indices):
                continue
            length_squared = delta_x * delta_x + delta_y * delta_y
            if length_squared == 0.0:
                projection = np.zeros(len(candidate_indices), dtype=np.float64)
            else:
                projection = np.clip(
                    (
                        (unknown_x[candidate_indices] - start_x) * delta_x
                        + (unknown_y[candidate_indices] - start_y) * delta_y
                    )
                    / length_squared,
                    0.0,
                    1.0,
                )
            nearest_x = start_x + projection * delta_x
            nearest_y = start_y + projection * delta_y
            inside = (
                (unknown_x[candidate_indices] - nearest_x) ** 2
                + (unknown_y[candidate_indices] - nearest_y) ** 2
                <= radius_squared
            )
            included = candidate_indices[inside]
            gains[index, 0] = np.float32(roi_weight[included].sum())
            gains[index, 1] = np.float32(priority_weight[included].sum())
        return np.ascontiguousarray(gains)

    def _build_ground_physical_universe(
        self,
        *,
        world: ObservedWorld,
        mission: MissionRaster,
        pose_map: Pose2,
        projection: PlatformProjection,
        physical_reachability: PhysicalReachabilityResult,
        platform_type: str,
        platform_id: str,
        capability_content_sha256: str,
        mission_revision: int,
        evidence_generation: int,
        physical_evidence_sha256: str,
        physical_reachability_algorithm_id: str,
        goal_tolerance_mm: int,
        robot: tuple[int, int],
        segments: list[list[tuple[int, int]]],
        exact_target_poses: Mapping[tuple[int, int], Pose2],
        ground_endpoint_feasibility: (
            Callable[[np.ndarray], np.ndarray] | None
        ),
        ground_detail_candidate_provider: (
            Callable[
                [
                    list[list[tuple[int, int]]],
                    ObservedWorld,
                ],
                Collection[tuple[int, _RawFrontierCandidate]],
            ]
            | None
        ),
        refresh_started: float,
    ) -> PhysicalCandidateUniverse:
        """Build at most three policy candidates per complete frontier chain."""
        physical_mask = physical_reachability.physical_observation_pose_mask
        step = max(
            1,
            round(
                self._sensor.standoff_m
                / world.canvas.geometry.resolution_m
            ),
        )
        raw_count = 0
        if ground_detail_candidate_provider is None:
            source = (
                (segment_id, candidate)
                for segment_id, segment in enumerate(segments)
                for candidate in _map_frontier_chain_pose_options(
                    segment,
                    robot=robot,
                    observed_mask=world.observed_mask,
                    physical_pose_mask=physical_mask,
                    standoff_cells=step,
                )
            )
        else:
            source = iter(ground_detail_candidate_provider(segments, world))
        mapped: dict[
            tuple[int, int, int, int], tuple[int, _RawFrontierCandidate]
        ] = {}
        for segment_id, candidate in source:
            raw_count += 1
            if not (
                isinstance(segment_id, int)
                and 0 <= segment_id < len(segments)
                and isinstance(candidate, _RawFrontierCandidate)
                and candidate.frontier_cell in segments[segment_id]
            ):
                raise CandidateInvariantError(
                    "GROUND_DETAIL_CANDIDATE_INVALID"
                )
            if not physical_mask[candidate.pose_cell]:
                # Fine traversability only certifies the landing pose.  The
                # existing 4 m physical/global reachability authority remains
                # a separate conservative gate and may reject its parent cell.
                continue
            target_pose = candidate.target_pose or exact_target_poses.get(
                candidate.pose_cell
            )
            if target_pose is None:
                raise CandidateInvariantError(
                    "GROUND_DETAIL_CANDIDATE_POSITION_MISSING"
                )
            if world.canvas.world_to_grid(target_pose.x_m, target_pose.y_m) != (
                candidate.pose_cell
            ):
                raise CandidateInvariantError(
                    "GROUND_DETAIL_CANDIDATE_POSITION_OUTSIDE_CELL"
                )
            mapped_key = (
                candidate.pose_cell[0],
                candidate.pose_cell[1],
                int(round(target_pose.x_m * 1000.0)),
                int(round(target_pose.y_m * 1000.0)),
            )
            previous = mapped.get(mapped_key)
            key = (segment_id, candidate.sample_rank, candidate.pose_cell)
            if previous is None or key < (
                previous[0],
                previous[1].sample_rank,
                previous[1].pose_cell,
            ):
                mapped[mapped_key] = (segment_id, candidate)
        fine = sorted(
            mapped.values(),
            key=lambda item: (
                item[0], item[1].sample_rank, item[1].pose_cell
            ),
        )
        ground_global_search = physical_reachability.ground_global_search
        if ground_global_search is None:
            raise CandidateInvariantError("GLOBAL_PATH_COST_NONFINITE")
        minimum_cost = ground_global_search.sampled_minimum_cost_m
        globally_reachable = [
            (segment_id, raw, float(minimum_cost[raw.pose_cell]))
            for segment_id, raw in fine
            if math.isfinite(float(minimum_cost[raw.pose_cell]))
        ]
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
        target_positions = np.ascontiguousarray(
            [
                (
                    (raw.target_pose or exact_target_poses[raw.pose_cell]).x_m,
                    (raw.target_pose or exact_target_poses[raw.pose_cell]).y_m,
                    (raw.target_pose or exact_target_poses[raw.pose_cell]).elevation_m,
                )
                for _, raw, _ in globally_reachable
            ],
            dtype=np.float64,
        ).reshape((-1, 3))
        if ground_endpoint_feasibility is None:
            endpoint_mask = np.ones(
                len(globally_reachable), dtype=np.bool_
            )
        else:
            endpoint_mask = ground_endpoint_feasibility(target_positions)
            if (
                not isinstance(endpoint_mask, np.ndarray)
                or endpoint_mask.dtype != np.dtype(np.bool_)
                or endpoint_mask.shape != (len(globally_reachable),)
                or not endpoint_mask.flags.c_contiguous
            ):
                raise CandidateInvariantError(
                    "GROUND_ENDPOINT_FEASIBILITY_INVALID"
                )
        endpoint_feasible = [
            item
            for item, accepted in zip(
                globally_reachable, endpoint_mask, strict=True
            )
            if bool(accepted)
        ]
        target_positions = np.ascontiguousarray(
            [
                (
                    (raw.target_pose or exact_target_poses[raw.pose_cell]).x_m,
                    (raw.target_pose or exact_target_poses[raw.pose_cell]).y_m,
                    (raw.target_pose or exact_target_poses[raw.pose_cell]).elevation_m,
                )
                for _, raw, _ in endpoint_feasible
            ],
            dtype=np.float64,
        ).reshape((-1, 3))
        exact_gain = getattr(
            self._visibility_estimator,
            "estimate_candidate_gains_at_positions",
            None,
        )
        if not endpoint_feasible:
            gains = np.zeros((0, 2), dtype=np.float32)
        elif callable(exact_gain):
            gains = exact_gain(*gain_arguments, target_positions)
        else:
            candidate_cells = np.ascontiguousarray(
                [raw.pose_cell for _, raw, _ in endpoint_feasible],
                dtype=np.int32,
            ).reshape((-1, 2))
            gains = self._visibility_estimator.estimate_candidate_gains(
                *gain_arguments, candidate_cells
            )
        if (
            not isinstance(gains, np.ndarray)
            or gains.shape != (len(endpoint_feasible), 2)
            or gains.dtype != np.dtype(np.float32)
            or not gains.flags.c_contiguous
            or not np.isfinite(gains).all()
            or (gains < 0.0).any()
        ):
            raise CandidateInvariantError("CANDIDATE_GAIN_NONFINITE")
        positive_indices = [
            index
            for index, gain in enumerate(gains[:, 0])
            if float(gain) >= _DETAIL_CELL_COARSE_EQUIVALENT
        ]
        total_roi = float(mission.roi_ratio.sum(dtype=np.float64))
        roi_diagonal = task_roi_diagonal_m(
            mission.roi_ratio,
            resolution_m=world.canvas.geometry.resolution_m,
        )
        selectable_by_segment: dict[
            int, list[tuple[_RawFrontierCandidate, int, float]]
        ] = {}
        for index in positive_indices:
            segment_id, raw, _ = endpoint_feasible[index]
            selectable_by_segment.setdefault(segment_id, []).append(
                (raw, index, float(gains[index, 0]))
            )
        selected_positive_indices: list[int] = []
        for segment_id in sorted(selectable_by_segment):
            selected_positive_indices.extend(
                _select_three_chain_options(
                    selectable_by_segment[segment_id],
                    chain_length=len(segments[segment_id]),
                )
            )
        positive_gain_normalizer = max(
            (
                float(gains[index, 0])
                for index in selected_positive_indices
            ),
            default=0.0,
        )
        priority_gain_normalizer = max(
            (
                float(gains[index, 1])
                for index in selected_positive_indices
            ),
            default=0.0,
        )
        preliminary: list[PhysicalCandidate] = []
        for index in selected_positive_indices:
            segment_id, raw, global_cost = endpoint_feasible[index]
            target_pose = raw.target_pose or exact_target_poses[raw.pose_cell]
            gain = float(gains[index, 0])
            priority_gain = float(gains[index, 1])
            feature = self._feature(
                world,
                mission,
                projection,
                pose_map,
                raw.pose_cell,
                total_roi,
                gain,
                priority_gain,
                positive_gain_normalizer,
                priority_gain_normalizer,
                target_pose=target_pose,
                global_path_cost_norm=normalize_global_path_cost(
                    global_cost,
                    roi_diagonal,
                    _NOMINAL_GLOBAL_COST_PER_M,
                ),
            )
            if feature is None:
                continue
            anchor = _FeasibleAnchor(
                segment_id=segment_id,
                point=raw.pose_cell,
                feature=feature,
                elevation_m=float(target_pose.elevation_m),
                target_position_m=(
                    float(target_pose.x_m),
                    float(target_pose.y_m),
                    float(target_pose.elevation_m),
                ),
            )
            preliminary.append(
                self._physical_candidate(
                    anchor,
                    pose_map=pose_map,
                    platform_type=platform_type,
                    platform_id=platform_id,
                    mission_revision=mission_revision,
                    goal_tolerance_mm=goal_tolerance_mm,
                    segment_candidate_rank=0,
                    global_path_cost_m=global_cost,
                    expected_gain_m2=(
                        gain / _DETAIL_CELL_COARSE_EQUIVALENT
                    )
                    * _DETAIL_CELL_AREA_M2,
                    expected_priority_gain_m2=(
                        priority_gain / _DETAIL_CELL_COARSE_EQUIVALENT
                    )
                    * _DETAIL_CELL_AREA_M2,
                    risk=max(
                        0.0,
                        1.0
                        - float(
                            projection.clearance_margin_norm[raw.pose_cell]
                        ),
                    ),
                )
            )
        by_segment: dict[int, list[PhysicalCandidate]] = {}
        for candidate in preliminary:
            by_segment.setdefault(candidate.segment_id, []).append(candidate)
        ranked: list[PhysicalCandidate] = []
        for segment_id in sorted(by_segment):
            segment = sorted(
                by_segment[segment_id],
                key=lambda candidate: (
                    -candidate.expected_gain_m2,
                    candidate.global_path_cost_m,
                    candidate.risk,
                    candidate.candidate_id,
                ),
            )
            ranked.extend(
                replace(candidate, segment_candidate_rank=rank)
                for rank, candidate in enumerate(segment)
            )
        balanced = _balanced_candidate_order(ranked)
        candidates = tuple(
            replace(candidate, rank_key=(index, candidate.candidate_id))
            for index, candidate in enumerate(balanced)
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
        selected_count = min(_POLICY_CANDIDATE_COUNT, len(candidates))
        universe_sha256 = sha256(
            "".join(candidate.candidate_id for candidate in candidates).encode(
                "ascii"
            )
        ).hexdigest()
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
            zero_gain_count=len(endpoint_feasible) - len(positive_indices),
            visited_excluded_count=0,
            physical_unreachable_count=(
                len(fine) - len(endpoint_feasible)
            ),
        )
        snapshot = CandidateDecisionSnapshot(
            snapshot_id=physical_snapshot_id,
            frontier_segment_count=len(segments),
            raw_candidate_count=raw_count,
            fine_pose_candidate_count=len(fine),
            globally_reachable_candidate_count=len(globally_reachable),
            positive_gain_candidate_count=len(candidates),
            selected_policy_candidate_count=selected_count,
            untried_reserve_count=len(candidates) - selected_count,
            planner_rejected_current_snapshot_count=0,
            candidate_set_sha256=universe_sha256,
            global_search_call_count=1,
            global_search_elapsed_s=ground_global_search.search_elapsed_s,
            candidate_refresh_elapsed_s=perf_counter() - refresh_started,
        )
        return PhysicalCandidateUniverse(
            physical_snapshot_id=physical_snapshot_id,
            physical_reachability_algorithm_id=(
                physical_reachability_algorithm_id
            ),
            candidates=candidates,
            universe_sha256=universe_sha256,
            diagnostics=diagnostics,
            decision_snapshot=snapshot,
        )

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
        """Apply current-snapshot failures and select a deterministic Top-64."""
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
        is_hopper = "hopper" in universe.physical_reachability_algorithm_id.lower()
        if is_hopper:
            selected = planner_available[:_POLICY_CANDIDATE_COUNT]
            history_available = planner_available
            visited_excluded_count = 0
        else:
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
            visited_excluded_count = len(planner_available) - len(
                history_available
            )
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
            decision_snapshot=(
                None
                if universe.decision_snapshot is None
                else replace(
                    universe.decision_snapshot,
                    selected_policy_candidate_count=len(selected),
                    untried_reserve_count=(
                        len(history_available) - len(selected)
                    ),
                    planner_rejected_current_snapshot_count=len(
                        current_failures
                    ),
                )
            ),
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
        segment_candidate_rank: int = 0,
        global_path_cost_m: float = 0.0,
        expected_gain_m2: float = 0.0,
        expected_priority_gain_m2: float = 0.0,
        risk: float = 0.0,
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
                "x_mm": _millimetres(x_m),
                "y_mm": _millimetres(y_m),
                "z_mm": z_mm,
            }
        )
        identity_heading = (
            {"heading_bin_64": target_yaw_bin}
            if platform_type == "HOPPER"
            else {"heading_authority": "policy-action"}
        )
        candidate_id = _canonical_sha256(
            {
                "schema": (
                    _HOPPER_CANDIDATE_ID_SCHEMA
                    if platform_type == "HOPPER"
                    else CANDIDATE_ID_SCHEMA
                ),
                "platform_type": platform_type,
                "platform_id": platform_id,
                "mission_revision": mission_revision,
                **position_key,
                **identity_heading,
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
            segment_id=anchor.segment_id,
            segment_candidate_rank=segment_candidate_rank,
            global_path_cost_m=global_path_cost_m,
            expected_gain_m2=expected_gain_m2,
            expected_priority_gain_m2=expected_priority_gain_m2,
            risk=risk,
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
            and (
                platform_type == "HOPPER"
                or roi[anchor[1]]
            )
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

    def _feature(self, world: ObservedWorld, mission: MissionRaster, projection: PlatformProjection, pose: Pose2, point: tuple[int, int], total_roi: float, gain: float, priority_gain: float, gain_normalizer: float, priority_gain_normalizer: float, *, allow_zero_gain: bool = False, target_pose: Pose2 | None = None, global_path_cost_norm: float | None = None) -> np.ndarray | None:
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
        return np.asarray(((x - canvas.bounds_m[0]) / canvas.geometry.size_m, (canvas.bounds_m[3] - y) / canvas.geometry.size_m, min(1.0, distance / (math.sqrt(2.0) * canvas.geometry.size_m)) if global_path_cost_norm is None else global_path_cost_norm, math.sin(bearing), math.cos(bearing), min(1.0, gain / gain_normalizer) if gain_normalizer else 0.0, min(1.0, priority_gain / priority_gain_normalizer) if priority_gain_normalizer else 0.0, -normal[0] / magnitude if magnitude else 0.0, normal[1] / magnitude if magnitude else 1.0, min(1.0, magnitude / 2.0), projection.clearance_margin_norm[point], remaining), dtype=np.float32)


__all__ = [
    "CANDIDATE_ID_SCHEMA",
    "CANDIDATE_DIAGNOSTIC_FIELDS",
    "PHYSICAL_SNAPSHOT_SCHEMA",
    "CandidateBatch",
    "CandidateBuildResult",
    "CandidateBuilderV2",
    "CandidateDecisionSnapshot",
    "CandidateDiagnostics",
    "CandidateInvariantError",
    "PhysicalCandidate",
    "PhysicalCandidateUniverse",
    "SensorGeometry",
    "normalize_global_path_cost",
]

"""Observed-map C++ authority for platform-specific candidate reachability."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import math
from types import MappingProxyType
from typing import Mapping

import numpy as np

from ..polar_data.raster import MapCanvas
from .coverability import PHYSICAL_PROJECTION_SCHEMA
from .observation_builder import Pose2


_PLATFORMS = frozenset(("WHEELED", "LEGGED", "HOPPER"))
_GROUND_PLATFORMS = frozenset(("WHEELED", "LEGGED"))
_GROUND_PHYSICAL_EVIDENCE_ALGORITHM_ID = (
    "cpp-safe-traversability-projection/v1"
)


def ground_point_goal_feasibility(
    hard_feasible: np.ndarray,
    *,
    canvas: MapCanvas,
    target_positions_m: np.ndarray,
    tolerance_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the C++ point-goal/cell-box rule on one north-up mask."""
    hard = np.asarray(hard_feasible)
    targets = np.asarray(target_positions_m)
    cells = canvas.geometry.cells if isinstance(canvas, MapCanvas) else 0
    if (
        hard.dtype != np.dtype(np.bool_)
        or hard.shape != (cells, cells)
        or not hard.flags.c_contiguous
        or targets.dtype != np.dtype(np.float64)
        or targets.ndim != 2
        or targets.shape[1:] != (3,)
        or not targets.flags.c_contiguous
        or not np.isfinite(targets).all()
        or not isinstance(tolerance_m, (int, float))
        or isinstance(tolerance_m, bool)
        or not np.isfinite(float(tolerance_m))
        or float(tolerance_m) < 0.0
    ):
        raise ValueError("ground point-goal feasibility inputs are invalid")
    covered = np.zeros(len(targets), dtype=np.bool_)
    feasible = np.zeros(len(targets), dtype=np.bool_)
    if not len(targets):
        return covered, feasible
    left, _bottom, _right, top = canvas.bounds_m
    resolution = float(canvas.geometry.resolution_m)
    tolerance = float(tolerance_m)
    index_radius = int(math.ceil(tolerance / resolution)) + 1
    for index, target in enumerate(targets):
        x_m = float(target[0])
        y_m = float(target[1])
        base_column = math.floor((x_m - left) / resolution)
        base_row = math.floor((top - y_m) / resolution)
        row0 = max(0, base_row - index_radius)
        row1 = min(cells, base_row + index_radius + 1)
        column0 = max(0, base_column - index_radius)
        column1 = min(cells, base_column + index_radius + 1)
        for row in range(row0, row1):
            cell_top = top - row * resolution
            cell_bottom = cell_top - resolution
            delta_y = max(cell_bottom - y_m, 0.0, y_m - cell_top)
            for column in range(column0, column1):
                cell_left = left + column * resolution
                cell_right = cell_left + resolution
                delta_x = max(cell_left - x_m, 0.0, x_m - cell_right)
                if math.hypot(delta_x, delta_y) <= tolerance + 1.0e-9:
                    covered[index] = True
                    if hard[row, column]:
                        feasible[index] = True
                        break
            if feasible[index]:
                break
    return (
        np.ascontiguousarray(covered),
        np.ascontiguousarray(feasible),
    )


@dataclass(frozen=True, slots=True)
class HopperSingleHopEnvelope:
    """Complete capability-derived screen over one certified landing grid."""

    certified_mask: np.ndarray
    certified_positions_m: np.ndarray
    eligible_mask: np.ndarray
    required_delta_v_mps: np.ndarray
    nominal_flight_time_s: np.ndarray
    algorithm_id: str
    raw_known_landing_count: int
    candidates_evaluated: int
    eligible_count: int
    complete: bool

    def __post_init__(self) -> None:
        shape = (
            self.certified_mask.shape
            if isinstance(self.certified_mask, np.ndarray)
            else ()
        )
        if (
            not isinstance(self.certified_mask, np.ndarray)
            or self.certified_mask.dtype != np.dtype(np.bool_)
            or self.certified_mask.ndim != 2
            or not self.certified_mask.flags.c_contiguous
            or not isinstance(self.certified_positions_m, np.ndarray)
            or self.certified_positions_m.dtype != np.dtype(np.float64)
            or self.certified_positions_m.shape
            != (int(self.certified_mask.sum(dtype=np.int64)), 3)
            or not self.certified_positions_m.flags.c_contiguous
            or not np.isfinite(self.certified_positions_m).all()
            or not isinstance(self.eligible_mask, np.ndarray)
            or self.eligible_mask.dtype != np.dtype(np.bool_)
            or self.eligible_mask.shape != shape
            or not self.eligible_mask.flags.c_contiguous
            or not isinstance(self.required_delta_v_mps, np.ndarray)
            or self.required_delta_v_mps.dtype != np.dtype(np.float64)
            or self.required_delta_v_mps.shape != shape
            or not self.required_delta_v_mps.flags.c_contiguous
            or np.isnan(self.required_delta_v_mps).any()
            or np.isneginf(self.required_delta_v_mps).any()
            or not isinstance(self.nominal_flight_time_s, np.ndarray)
            or self.nominal_flight_time_s.dtype != np.dtype(np.float64)
            or self.nominal_flight_time_s.shape != shape
            or not self.nominal_flight_time_s.flags.c_contiguous
            or not np.isfinite(self.nominal_flight_time_s).all()
            or not isinstance(self.algorithm_id, str)
            or self.algorithm_id != "cpp-hopper-single-hop-envelope/v1"
            or type(self.raw_known_landing_count) is not int
            or type(self.candidates_evaluated) is not int
            or type(self.eligible_count) is not int
            or type(self.complete) is not bool
        ):
            raise ValueError("hopper single-hop envelope geometry is invalid")
        inactive = ~self.eligible_mask
        if (
            (self.eligible_mask & ~self.certified_mask).any()
            or self.raw_known_landing_count
            != int(self.certified_mask.sum(dtype=np.int64))
            or not 0 <= self.candidates_evaluated <= self.raw_known_landing_count
            or self.eligible_count
            != int(self.eligible_mask.sum(dtype=np.int64))
            or not np.isfinite(
                self.required_delta_v_mps[self.eligible_mask]
            ).all()
            or (self.required_delta_v_mps[self.eligible_mask] <= 0.0).any()
            or not np.isposinf(self.required_delta_v_mps[inactive]).all()
            or (self.nominal_flight_time_s[self.eligible_mask] <= 0.0).any()
            or (self.nominal_flight_time_s[inactive] != 0.0).any()
        ):
            raise ValueError("hopper single-hop envelope values differ")
        for value in (
            self.certified_mask,
            self.certified_positions_m,
            self.eligible_mask,
            self.required_delta_v_mps,
            self.nominal_flight_time_s,
        ):
            value.setflags(write=False)


class PlatformReachabilityError(RuntimeError):
    """The native reachability authority violated its public contract."""


@dataclass(frozen=True, slots=True)
class GroundGlobalSearchEvidence:
    """A validated fine planner tree sampled onto the candidate canvas."""

    sampled_minimum_cost_m: np.ndarray
    planner_tree_sha256: str
    planner_tree_cells: int
    planner_start_index: int
    search_elapsed_s: float
    coarse_fine_unreachable_count: int

    def __post_init__(self) -> None:
        cost = self.sampled_minimum_cost_m
        if (
            not isinstance(cost, np.ndarray)
            or cost.dtype != np.dtype(np.float64)
            or cost.ndim != 2
            or cost.shape[0] != cost.shape[1]
            or not cost.flags.c_contiguous
            or cost.flags.writeable
            or np.isnan(cost).any()
            or np.isneginf(cost).any()
            or (cost < 0.0).any()
            or not isinstance(self.planner_tree_sha256, str)
            or len(self.planner_tree_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.planner_tree_sha256
            )
            or type(self.planner_tree_cells) is not int
            or self.planner_tree_cells <= 0
            or type(self.planner_start_index) is not int
            or not 0
            <= self.planner_start_index
            < self.planner_tree_cells * self.planner_tree_cells
            or not isinstance(self.search_elapsed_s, float)
            or not np.isfinite(self.search_elapsed_s)
            or self.search_elapsed_s < 0.0
            or type(self.coarse_fine_unreachable_count) is not int
            or not 0 <= self.coarse_fine_unreachable_count <= cost.size
        ):
            raise ValueError("ground global search evidence is invalid")


def _validate_ground_reachability_projection(
    projection: object,
    *,
    platform_type: str,
    cells: int,
) -> tuple[np.ndarray, str, np.ndarray, np.ndarray, int, float]:
    """Validate and convert one native south-up ground cost tree to north-up."""
    reachable = getattr(projection, "reachable", None)
    if getattr(projection, "platform_type", None) != platform_type:
        raise PlatformReachabilityError(
            "candidate reachability projection platform differs"
        )
    if (
        not isinstance(reachable, np.ndarray)
        or reachable.dtype != np.dtype(np.uint8)
        or reachable.shape != (cells, cells)
        or not reachable.flags.c_contiguous
        or (reachable.size and (reachable > 1).any())
    ):
        raise PlatformReachabilityError(
            "candidate reachability projection geometry differs"
        )
    algorithm_id = getattr(projection, "algorithm_id", None)
    if not isinstance(algorithm_id, str) or not algorithm_id:
        raise PlatformReachabilityError(
            "candidate reachability projection algorithm is invalid"
        )
    minimum_cost = getattr(projection, "minimum_cost", None)
    parent_index = getattr(projection, "parent_index", None)
    start_index = getattr(projection, "start_index", None)
    elapsed_s = getattr(projection, "search_elapsed_s", None)
    if (
        not isinstance(minimum_cost, np.ndarray)
        or minimum_cost.dtype != np.dtype(np.float64)
        or minimum_cost.shape != (cells, cells)
        or not minimum_cost.flags.c_contiguous
        or minimum_cost.flags.writeable
        or np.isnan(minimum_cost).any()
        or np.isneginf(minimum_cost).any()
        or (minimum_cost < 0.0).any()
        or not isinstance(parent_index, np.ndarray)
        or parent_index.dtype != np.dtype(np.int64)
        or parent_index.shape != (cells, cells)
        or not parent_index.flags.c_contiguous
        or parent_index.flags.writeable
        or type(start_index) is not int
        or not 0 <= start_index < cells * cells
        or not isinstance(elapsed_s, float)
        or not np.isfinite(elapsed_s)
        or elapsed_s < 0.0
    ):
        raise PlatformReachabilityError(
            "candidate global cost tree geometry differs"
        )
    raw_cost = minimum_cost.reshape(-1)
    raw_parent = parent_index.reshape(-1)
    finite = np.isfinite(raw_cost)
    if finite.any() and (
        raw_cost[start_index] != 0.0
        or raw_parent[start_index] != start_index
    ):
        raise PlatformReachabilityError(
            "candidate global cost tree start differs"
        )
    if not np.array_equal(reachable.reshape(-1) != 0, finite):
        raise PlatformReachabilityError(
            "candidate global cost tree reachability differs"
        )
    if (
        (raw_parent[~finite] != -1).any()
        or (raw_parent[finite] < 0).any()
        or (raw_parent[finite] >= raw_cost.size).any()
    ):
        raise PlatformReachabilityError(
            "candidate global cost tree parent differs"
        )
    finite_indices = np.flatnonzero(finite)
    non_start = finite_indices[finite_indices != start_index]
    if len(non_start) and not np.all(
        raw_cost[raw_parent[non_start]] < raw_cost[non_start]
    ):
        raise PlatformReachabilityError(
            "candidate global cost tree is not acyclic"
        )
    north_cost = np.ascontiguousarray(np.flipud(minimum_cost))
    north_parent = np.ascontiguousarray(np.flipud(parent_index))
    valid_parent = north_parent >= 0
    parent_rows = np.zeros_like(north_parent)
    parent_columns = np.zeros_like(north_parent)
    parent_rows[valid_parent] = north_parent[valid_parent] // cells
    parent_columns[valid_parent] = north_parent[valid_parent] % cells
    north_parent[valid_parent] = (
        (cells - 1 - parent_rows[valid_parent]) * cells
        + parent_columns[valid_parent]
    )
    raw_start_row, raw_start_column = divmod(start_index, cells)
    north_start_index = (
        (cells - 1 - raw_start_row) * cells + raw_start_column
    )
    north_cost.setflags(write=False)
    north_parent.setflags(write=False)
    return (
        np.ascontiguousarray(np.flipud(reachable).astype(np.bool_)),
        algorithm_id,
        north_cost,
        north_parent,
        north_start_index,
        float(elapsed_s),
    )


@dataclass(frozen=True, slots=True)
class PhysicalReachabilityResult:
    """Primitive-independent physical observation poses for one fixed start."""

    platform_type: str
    physical_observation_pose_mask: np.ndarray
    observation_positions_m: np.ndarray
    physical_projection_schema: str
    physical_reachability_algorithm_id: str
    physical_evidence_algorithm_id: str
    physical_safe_pose_count: int
    physically_reachable_pose_count: int
    ground_global_search: GroundGlobalSearchEvidence | None = None
    hopper_single_hop_envelope: HopperSingleHopEnvelope | None = None

    def __post_init__(self) -> None:
        mask = self.physical_observation_pose_mask
        positions = self.observation_positions_m
        if self.platform_type not in _PLATFORMS:
            raise ValueError("physical reachability platform is invalid")
        if (
            not isinstance(mask, np.ndarray)
            or mask.dtype != np.dtype(np.bool_)
            or mask.ndim != 2
            or not mask.flags.c_contiguous
            or any(dimension <= 0 for dimension in mask.shape)
        ):
            raise ValueError(
                "physical observation pose mask must be C-contiguous bool [H,W]"
            )
        if (
            not isinstance(positions, np.ndarray)
            or positions.dtype != np.dtype(np.float64)
            or positions.ndim != 2
            or positions.shape[1:] != (3,)
            or not positions.flags.c_contiguous
            or not np.isfinite(positions).all()
        ):
            raise ValueError(
                "physical observation positions must be C-contiguous float64 [N,3]"
            )
        reachable_count = int(mask.sum(dtype=np.int64))
        if (
            self.physical_projection_schema != PHYSICAL_PROJECTION_SCHEMA
            or not isinstance(self.physical_reachability_algorithm_id, str)
            or not self.physical_reachability_algorithm_id
            or not isinstance(self.physical_evidence_algorithm_id, str)
            or not self.physical_evidence_algorithm_id
        ):
            raise ValueError("physical reachability algorithm identity is invalid")
        if (
            type(self.physical_safe_pose_count) is not int
            or type(self.physically_reachable_pose_count) is not int
            or self.physical_safe_pose_count < reachable_count
            or self.physically_reachable_pose_count != reachable_count
            or len(positions) != reachable_count
        ):
            raise ValueError("physical reachability counts differ")
        if (
            self.platform_type != "HOPPER"
            and self.hopper_single_hop_envelope is not None
        ):
            raise ValueError("ground physical reachability has hopper authority")
        if self.platform_type == "HOPPER":
            if self.ground_global_search is not None:
                raise ValueError("hopper physical reachability has ground search")
            if (
                not isinstance(
                    self.hopper_single_hop_envelope,
                    HopperSingleHopEnvelope,
                )
                or not self.hopper_single_hop_envelope.complete
                or not np.array_equal(
                    mask, self.hopper_single_hop_envelope.eligible_mask
                )
            ):
                raise ValueError("hopper single-hop envelope is invalid")
            return
        evidence = self.ground_global_search
        if (
            not isinstance(evidence, GroundGlobalSearchEvidence)
            or evidence.sampled_minimum_cost_m.shape != mask.shape
        ):
            raise ValueError("ground global search evidence is invalid")
        if (mask & ~np.isfinite(evidence.sampled_minimum_cost_m)).any():
            raise ValueError("ground physical reachability exceeds planner tree")


@dataclass(frozen=True, slots=True)
class CandidateReachabilityResult:
    accepted_mask: np.ndarray
    reason_counts: Mapping[str, int]

    def __post_init__(self) -> None:
        mask = self.accepted_mask
        if (
            not isinstance(mask, np.ndarray)
            or mask.dtype != np.dtype(np.bool_)
            or mask.ndim != 1
            or not mask.flags.c_contiguous
        ):
            raise ValueError("candidate reachability mask must be C-contiguous bool [N]")
        expected = int((~mask).sum(dtype=np.int64))
        if dict(self.reason_counts) != {"platform_unreachable_count": expected}:
            raise ValueError("candidate reachability reason counts differ")
        object.__setattr__(
            self, "reason_counts", MappingProxyType(dict(self.reason_counts))
        )


class PlatformCandidateReachability:
    """Run the C++ ground component or capability-derived Hopper envelope."""

    def __init__(
        self,
        *,
        platform_type: str,
        canvas: MapCanvas,
        pose_map: Pose2,
        observed_elevation_m: np.ndarray,
        bridge: object,
        request: object,
        maximum_edge_distance_m: float | None = None,
        local_traversability_projection: object | None = None,
        planner_global_canvas: MapCanvas | None = None,
    ) -> None:
        if platform_type not in _PLATFORMS:
            raise ValueError("candidate reachability platform is invalid")
        if not isinstance(canvas, MapCanvas):
            raise TypeError("candidate reachability canvas is invalid")
        if not isinstance(pose_map, Pose2) or pose_map.frame_id != "map":
            raise ValueError("candidate reachability pose is invalid")
        elevation = np.ascontiguousarray(observed_elevation_m, dtype=np.float32)
        if elevation.shape != (
            canvas.geometry.cells,
            canvas.geometry.cells,
        ) or not np.isfinite(elevation).all():
            raise ValueError("candidate reachability elevation is invalid")
        bridge_methods = (
            (
                "project_hopper_landing_evidence",
                "project_hopper_single_hop_envelope",
            )
            if platform_type == "HOPPER"
            else (
                "project_reachability",
                *(
                    ()
                    if local_traversability_projection is not None
                    else ("project_traversability",)
                ),
            )
        )
        if any(
            not callable(getattr(bridge, name, None))
            for name in bridge_methods
        ):
            raise TypeError("candidate reachability bridge is invalid")
        if platform_type == "HOPPER":
            if maximum_edge_distance_m is not None:
                raise ValueError(
                    "hopper reachability has no fixed edge-distance limit"
                )
            resolved_edge_distance_m = None
        else:
            resolved_edge_distance_m = (
                30.0
                if maximum_edge_distance_m is None
                else maximum_edge_distance_m
            )
            if (
                not isinstance(resolved_edge_distance_m, float)
                or not np.isfinite(resolved_edge_distance_m)
                or resolved_edge_distance_m <= 0.0
            ):
                raise ValueError(
                    "candidate reachability edge distance is invalid"
                )
        self._platform_type = platform_type
        self._canvas = canvas
        self._pose = pose_map
        self._elevation = elevation
        self._bridge = bridge
        self._request = request
        self._planner_global_canvas = (
            canvas if planner_global_canvas is None else planner_global_canvas
        )
        if not isinstance(self._planner_global_canvas, MapCanvas):
            raise TypeError("planner global canvas is invalid")
        if platform_type == "HOPPER" and planner_global_canvas is not None:
            raise ValueError("hopper has no ground planner canvas")
        if platform_type != "HOPPER" and planner_global_canvas is not None:
            self._validate_planner_global_map_binding()
        self._maximum_edge_distance_m = resolved_edge_distance_m
        self._ground_endpoint_context: object | None = None
        self._local_traversability = (
            None
            if platform_type == "HOPPER"
            else local_traversability_projection
            if local_traversability_projection is not None
            else bridge.project_traversability(request)
        )

    def filter(
        self,
        candidate_cells: np.ndarray,
        *,
        target_positions_map: np.ndarray | None = None,
    ) -> CandidateReachabilityResult:
        candidates = self._validate_candidate_cells(candidate_cells)
        physical = self.project_physical(
            candidates, target_positions_map=target_positions_map
        )
        accepted = np.ascontiguousarray(
            physical.physical_observation_pose_mask[
                candidates[:, 0], candidates[:, 1]
            ],
            dtype=np.bool_,
        )
        return CandidateReachabilityResult(
            accepted,
            {"platform_unreachable_count": int((~accepted).sum(dtype=np.int64))},
        )

    def project_physical(
        self,
        observed_safe_cells: np.ndarray,
        *,
        target_positions_map: np.ndarray | None = None,
    ) -> PhysicalReachabilityResult:
        """Project one observed-safe cell set into physical observation poses."""
        candidates = self._validate_candidate_cells(observed_safe_cells)
        positions = self._validate_target_positions(
            candidates, target_positions_map
        )
        candidates, positions = self._row_major_unique(candidates, positions)
        if self._platform_type == "HOPPER":
            (
                accepted,
                safe,
                exact_positions,
                reachability_algorithm_id,
                evidence_algorithm_id,
                single_hop_envelope,
            ) = self._project_hopper_candidates(candidates, positions)
            ground_global_search = None
        else:
            assert self._maximum_edge_distance_m is not None
            self._ground_endpoint_context = None
            project_context = getattr(
                self._bridge,
                "project_ground_endpoint_context",
                None,
            )
            if callable(project_context):
                context = project_context(
                    self._request,
                    self._maximum_edge_distance_m,
                )
                projection = getattr(context, "projection", None)
                if projection is None:
                    raise PlatformReachabilityError(
                        "ground endpoint context projection is unavailable"
                    )
                self._ground_endpoint_context = context
            else:
                projection = self._bridge.project_reachability(
                    self._request, self._maximum_edge_distance_m
                )
            (
                reachable,
                reachability_algorithm_id,
                planner_cost,
                planner_parent,
                planner_start_index,
                search_elapsed_s,
            ) = (
                self._validate_reachability_projection(projection)
            )
            planner_cells = np.asarray(
                [
                    self._planner_global_canvas.world_to_grid(
                        float(position[0]), float(position[1])
                    )
                    for position in positions
                ],
                dtype=np.int32,
            ).reshape((-1, 2))
            accepted = np.ascontiguousarray(
                reachable[planner_cells[:, 0], planner_cells[:, 1]],
                dtype=np.bool_,
            )
            sampled_cost = np.full(
                (self._canvas.geometry.cells, self._canvas.geometry.cells),
                np.inf,
                dtype=np.float64,
            )
            sampled_cost[candidates[:, 0], candidates[:, 1]] = planner_cost[
                planner_cells[:, 0], planner_cells[:, 1]
            ]
            coarse_fine_unreachable_count = int(
                (~accepted).sum(dtype=np.int64)
            )
            local_inside, local_accepted = self._ground_local_reachability(
                candidates, positions
            )
            accepted[local_inside] &= local_accepted[local_inside]
            safe = np.ones(len(candidates), dtype=np.bool_)
            exact_positions = positions
            evidence_algorithm_id = (
                _GROUND_PHYSICAL_EVIDENCE_ALGORITHM_ID
            )
            single_hop_envelope = None
            sampled_cost.setflags(write=False)
            tree_identity = sha256()
            tree_identity.update(reachability_algorithm_id.encode("utf-8"))
            tree_identity.update(b"\0")
            tree_identity.update(str(planner_start_index).encode("ascii"))
            tree_identity.update(b"\0")
            tree_identity.update(planner_cost.tobytes(order="C"))
            tree_identity.update(planner_parent.tobytes(order="C"))
            ground_global_search = GroundGlobalSearchEvidence(
                sampled_minimum_cost_m=sampled_cost,
                planner_tree_sha256=tree_identity.hexdigest(),
                planner_tree_cells=self._planner_global_canvas.geometry.cells,
                planner_start_index=planner_start_index,
                search_elapsed_s=search_elapsed_s,
                coarse_fine_unreachable_count=(
                    coarse_fine_unreachable_count
                ),
            )

        mask = np.zeros(
            (self._canvas.geometry.cells, self._canvas.geometry.cells),
            dtype=np.bool_,
        )
        accepted_cells = candidates[accepted]
        if len(accepted_cells):
            mask[accepted_cells[:, 0], accepted_cells[:, 1]] = True
        return PhysicalReachabilityResult(
            platform_type=self._platform_type,
            physical_observation_pose_mask=np.ascontiguousarray(mask),
            observation_positions_m=np.ascontiguousarray(
                exact_positions[accepted], dtype=np.float64
            ).reshape((-1, 3)),
            physical_projection_schema=PHYSICAL_PROJECTION_SCHEMA,
            physical_reachability_algorithm_id=reachability_algorithm_id,
            physical_evidence_algorithm_id=evidence_algorithm_id,
            physical_safe_pose_count=int(safe.sum(dtype=np.int64)),
            physically_reachable_pose_count=int(
                accepted.sum(dtype=np.int64)
            ),
            ground_global_search=ground_global_search,
            hopper_single_hop_envelope=single_hop_envelope,
        )

    def query_exact_ground_endpoints(
        self, target_positions_map: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Query detailed goals against this snapshot's one global cost tree."""
        if self._platform_type not in _GROUND_PLATFORMS:
            raise RuntimeError("exact ground endpoints are ground-only")
        context = self._ground_endpoint_context
        if context is None:
            raise RuntimeError("ground endpoint context is unavailable")
        positions = np.asarray(target_positions_map)
        if (
            positions.dtype != np.dtype(np.float64)
            or positions.ndim != 2
            or positions.shape[1:] != (3,)
            or not positions.flags.c_contiguous
            or not np.isfinite(positions).all()
        ):
            raise ValueError(
                "exact ground endpoint positions must be float64 [N,3]"
            )
        query = getattr(
            self._bridge,
            "query_ground_exact_endpoints",
            None,
        )
        if not callable(query):
            raise RuntimeError("ground endpoint query is unavailable")
        result = query(context, positions, 0.2)
        raw_reachable = getattr(result, "reachable", None)
        raw_cost = getattr(result, "minimum_cost_m", None)
        reasons = getattr(result, "reason_codes", None)
        if (
            not isinstance(raw_reachable, np.ndarray)
            or raw_reachable.dtype != np.dtype(np.uint8)
            or raw_reachable.shape != (len(positions),)
            or not raw_reachable.flags.c_contiguous
            or (raw_reachable.size and (raw_reachable > 1).any())
            or not isinstance(raw_cost, np.ndarray)
            or raw_cost.dtype != np.dtype(np.float64)
            or raw_cost.shape != (len(positions),)
            or not raw_cost.flags.c_contiguous
            or np.isnan(raw_cost).any()
            or np.isneginf(raw_cost).any()
            or (raw_cost < 0.0).any()
            or not isinstance(reasons, tuple)
            or len(reasons) != len(positions)
            or any(not isinstance(reason, str) or not reason for reason in reasons)
        ):
            raise PlatformReachabilityError(
                "ground endpoint query geometry differs"
            )
        reachable = np.ascontiguousarray(
            raw_reachable.astype(np.bool_), dtype=np.bool_
        )
        cost = np.ascontiguousarray(raw_cost, dtype=np.float64)
        if (
            not np.isfinite(cost[reachable]).all()
            or not np.isposinf(cost[~reachable]).all()
        ):
            raise PlatformReachabilityError(
                "ground endpoint query costs differ"
            )
        return reachable, cost

    def _validate_planner_global_map_binding(self) -> None:
        planner_map = getattr(
            getattr(self._request, "world", None), "global_map", None
        )
        canvas = self._planner_global_canvas
        origin = getattr(planner_map, "origin_m", None)
        if (
            planner_map is None
            or getattr(planner_map, "frame_id", None) != "map"
            or getattr(planner_map, "width", None) != canvas.geometry.cells
            or getattr(planner_map, "height", None) != canvas.geometry.cells
            or getattr(planner_map, "resolution_m", None)
            != canvas.geometry.resolution_m
            or origin is None
            or getattr(origin, "x", None) != canvas.bounds_m[0]
            or getattr(origin, "y", None) != canvas.bounds_m[1]
        ):
            raise ValueError("planner global map differs from its canvas")

    def _validate_candidate_cells(
        self, candidate_cells: np.ndarray
    ) -> np.ndarray:
        candidates = np.asarray(candidate_cells)
        cells = self._canvas.geometry.cells
        if (
            candidates.dtype != np.dtype(np.int32)
            or candidates.ndim != 2
            or candidates.shape[1:] != (2,)
            or not candidates.flags.c_contiguous
            or (
                candidates.size
                and ((candidates < 0).any() or (candidates >= cells).any())
            )
        ):
            raise ValueError(
                "candidate reachability cells must be int32 [N,2]"
            )
        return candidates

    def _validate_target_positions(
        self,
        candidates: np.ndarray,
        target_positions_map: np.ndarray | None,
    ) -> np.ndarray:
        if target_positions_map is None:
            return np.asarray(
                [
                    (
                        *self._canvas.grid_center_world(
                            int(row), int(column)
                        ),
                        float(self._elevation[row, column]),
                    )
                    for row, column in candidates
                ],
                dtype=np.float64,
            ).reshape((-1, 3))
        positions = np.asarray(target_positions_map)
        if (
            positions.dtype != np.dtype(np.float64)
            or positions.shape != (len(candidates), 3)
            or not positions.flags.c_contiguous
            or not np.isfinite(positions).all()
        ):
            raise ValueError(
                "candidate target positions must be float64 [N,3]"
            )
        for cell, position in zip(candidates, positions, strict=True):
            try:
                target_cell = self._canvas.world_to_grid(
                    float(position[0]), float(position[1])
                )
            except ValueError as error:
                raise ValueError(
                    "candidate target position leaves its cell"
                ) from error
            if target_cell != tuple(cell):
                raise ValueError("candidate target position leaves its cell")
        return positions

    @staticmethod
    def _row_major_unique(
        candidates: np.ndarray, positions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if not len(candidates):
            return candidates.copy(), positions.copy()
        order = np.lexsort((candidates[:, 1], candidates[:, 0]))
        sorted_cells = np.ascontiguousarray(candidates[order], dtype=np.int32)
        sorted_positions = np.ascontiguousarray(
            positions[order], dtype=np.float64
        )
        keep = np.ones(len(sorted_cells), dtype=np.bool_)
        keep[1:] = np.any(sorted_cells[1:] != sorted_cells[:-1], axis=1)
        duplicate_indices = np.flatnonzero(~keep)
        for index in duplicate_indices:
            if not np.array_equal(
                sorted_positions[index], sorted_positions[index - 1]
            ):
                raise ValueError(
                    "duplicate candidate cell has conflicting positions"
                )
        return (
            np.ascontiguousarray(sorted_cells[keep], dtype=np.int32),
            np.ascontiguousarray(sorted_positions[keep], dtype=np.float64),
        )

    def _validate_reachability_projection(
        self, projection: object
    ) -> tuple[np.ndarray, str, np.ndarray, np.ndarray, int, float]:
        return _validate_ground_reachability_projection(
            projection,
            platform_type=self._platform_type,
            cells=self._planner_global_canvas.geometry.cells,
        )

    def _ground_local_reachability(
        self,
        candidates: np.ndarray,
        target_positions_map: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        projection = self._local_traversability
        local_map = getattr(getattr(self._request, "world", None), "local_map", None)
        transform = getattr(
            getattr(self._request, "world", None), "map_from_odom", None
        )
        if projection is None or local_map is None or transform is None:
            raise RuntimeError("ground local reachability inputs are unavailable")
        width = getattr(local_map, "width", None)
        height = getattr(local_map, "height", None)
        resolution = getattr(local_map, "resolution_m", None)
        origin = getattr(local_map, "origin_m", None)
        if (
            type(width) is not int
            or type(height) is not int
            or width <= 0
            or height <= 0
            or not isinstance(resolution, float)
            or not np.isfinite(resolution)
            or resolution <= 0.0
            or origin is None
            or getattr(local_map, "frame_id", None)
            != getattr(transform, "child_frame", None)
            or getattr(transform, "parent_frame", None) != "map"
        ):
            raise RuntimeError("ground local reachability geometry is invalid")
        hard = np.asarray(getattr(projection, "hard_feasible", None))
        components = np.asarray(getattr(projection, "connected_component", None))
        if (
            hard.shape != (height, width)
            or components.shape != (height, width)
            or hard.dtype != np.dtype(np.uint8)
            or components.dtype != np.dtype(np.int32)
        ):
            raise RuntimeError("ground local reachability projection is invalid")

        if target_positions_map is None:
            target_positions_map = np.asarray(
                [
                    (
                        *self._canvas.grid_center_world(int(row), int(column)),
                        float(self._elevation[row, column]),
                    )
                    for row, column in candidates
                ],
                dtype=np.float64,
            )
        start_map = np.asarray(
            [[self._pose.x_m, self._pose.y_m, self._pose.elevation_m]],
            dtype=np.float64,
        )
        start_local = self._map_to_odom(start_map, transform)[0]
        targets_local = self._map_to_odom(target_positions_map, transform)

        origin_xy = np.asarray(
            [getattr(origin, "x", np.nan), getattr(origin, "y", np.nan)],
            dtype=np.float64,
        )
        if not np.isfinite(origin_xy).all():
            raise RuntimeError("ground local reachability origin is invalid")

        def cells_for(positions: np.ndarray) -> np.ndarray:
            return np.floor((positions[:, :2] - origin_xy) / resolution).astype(
                np.int64
            )

        target_cells = cells_for(targets_local)
        inside = np.ascontiguousarray(
            (target_cells[:, 0] >= 0)
            & (target_cells[:, 0] < width)
            & (target_cells[:, 1] >= 0)
            & (target_cells[:, 1] < height),
            dtype=np.bool_,
        )
        accepted = np.zeros(len(candidates), dtype=np.bool_)
        start_cell = cells_for(start_local.reshape(1, 3))[0]
        if not (
            0 <= start_cell[0] < width and 0 <= start_cell[1] < height
        ):
            return inside, accepted
        start_component = int(components[start_cell[1], start_cell[0]])
        if (
            hard[start_cell[1], start_cell[0]] == 0
            or start_component < 0
        ):
            return inside, accepted

        # The observed detail component is an additional safety restriction
        # inside the local map. It may remove a planner-global target, but the
        # caller must never use it to promote a planner-global rejection.
        # Targets outside this window retain the global C++ tree result.
        indices = np.flatnonzero(inside)
        if len(indices):
            columns = target_cells[indices, 0]
            rows = target_cells[indices, 1]
            accepted[indices] = (
                (hard[rows, columns] != 0)
                & (components[rows, columns] == start_component)
            )
        return inside, np.ascontiguousarray(accepted)

    @staticmethod
    def _map_to_odom(points_map: np.ndarray, transform: object) -> np.ndarray:
        translation = getattr(transform, "translation_m", None)
        rotation = getattr(transform, "rotation", None)
        values = np.asarray(
            [
                getattr(translation, "x", np.nan),
                getattr(translation, "y", np.nan),
                getattr(translation, "z", np.nan),
                getattr(rotation, "x", np.nan),
                getattr(rotation, "y", np.nan),
                getattr(rotation, "z", np.nan),
                getattr(rotation, "w", np.nan),
            ],
            dtype=np.float64,
        )
        if not np.isfinite(values).all():
            raise RuntimeError("ground local reachability transform is invalid")
        tx, ty, tz, qx, qy, qz, qw = values
        norm = float(np.linalg.norm(values[3:]))
        if norm <= np.finfo(np.float64).eps:
            raise RuntimeError("ground local reachability rotation is invalid")
        qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
        rotation_child_to_parent = np.asarray(
            [
                [
                    1.0 - 2.0 * (qy * qy + qz * qz),
                    2.0 * (qx * qy - qz * qw),
                    2.0 * (qx * qz + qy * qw),
                ],
                [
                    2.0 * (qx * qy + qz * qw),
                    1.0 - 2.0 * (qx * qx + qz * qz),
                    2.0 * (qy * qz - qx * qw),
                ],
                [
                    2.0 * (qx * qz - qy * qw),
                    2.0 * (qy * qz + qx * qw),
                    1.0 - 2.0 * (qx * qx + qy * qy),
                ],
            ],
            dtype=np.float64,
        )
        return np.ascontiguousarray(
            (points_map - np.asarray([tx, ty, tz])) @ rotation_child_to_parent
        )

    def _project_hopper_candidates(
        self,
        candidates: np.ndarray,
        target_positions_map: np.ndarray,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        str,
        str,
        HopperSingleHopEnvelope,
    ]:
        import lunar_planner_training_bridge as bridge_api

        start = self._canvas.world_to_grid(self._pose.x_m, self._pose.y_m)
        cells_to_certify = [start, *map(tuple, candidates.tolist())]
        cells_to_certify = list(dict.fromkeys(cells_to_certify))
        exact_targets = {
            tuple(cell): tuple(float(value) for value in position)
            for cell, position in zip(
                candidates.tolist(), target_positions_map.tolist(), strict=True
            )
        }
        targets = np.asarray(
            [
                (
                    (
                        self._pose.x_m,
                        self._pose.y_m,
                        self._pose.elevation_m,
                    )
                    if (row, column) == start
                    else exact_targets[(row, column)]
                    if (row, column) in exact_targets
                    else (
                        *self._canvas.grid_center_world(row, column),
                        float(self._elevation[row, column]),
                    )
                )
                for row, column in cells_to_certify
            ],
            dtype=np.float64,
        )
        landing = self._bridge.project_hopper_landing_evidence(
            self._request, np.ascontiguousarray(targets)
        )
        count = len(cells_to_certify)
        certified = getattr(landing, "certified", None)
        aim = getattr(landing, "aim_positions_m", None)
        boundary = getattr(landing, "boundary_m", None)
        area = getattr(landing, "area_m2", None)
        evidence_algorithm_id = getattr(landing, "algorithm_id", None)
        if (
            not isinstance(certified, np.ndarray)
            or certified.dtype != np.dtype(np.bool_)
            or certified.shape != (count,)
            or not certified.flags.c_contiguous
            or not isinstance(aim, np.ndarray)
            or aim.dtype != np.dtype(np.float64)
            or aim.shape != (count, 3)
            or not aim.flags.c_contiguous
            or not np.isfinite(aim).all()
            or not isinstance(boundary, np.ndarray)
            or boundary.dtype != np.dtype(np.float64)
            or boundary.shape != (count, 4, 3)
            or not boundary.flags.c_contiguous
            or not np.isfinite(boundary).all()
            or not isinstance(area, np.ndarray)
            or area.dtype != np.dtype(np.float64)
            or area.shape != (count,)
            or not area.flags.c_contiguous
            or not np.isfinite(area).all()
            or (area < 0.0).any()
        ):
            raise RuntimeError("hopper landing evidence geometry differs")
        if (
            not isinstance(evidence_algorithm_id, str)
            or not evidence_algorithm_id
        ):
            raise RuntimeError("hopper landing evidence algorithm is invalid")
        for cell, position, is_certified in zip(
            cells_to_certify, aim, certified, strict=True
        ):
            if not bool(is_certified):
                continue
            try:
                aim_cell = self._canvas.world_to_grid(
                    float(position[0]), float(position[1])
                )
            except ValueError as error:
                raise RuntimeError(
                    "certified hopper aim leaves its cell"
                ) from error
            if aim_cell != cell:
                raise RuntimeError("certified hopper aim leaves its cell")
        shape = (self._canvas.geometry.cells, self._canvas.geometry.cells)
        certified = np.zeros(shape, dtype=np.bool_)
        aim = np.zeros((*shape, 3), dtype=np.float64)
        boundary = np.zeros((*shape, 4, 3), dtype=np.float64)
        area = np.zeros(shape, dtype=np.float64)
        for index, (row, column) in enumerate(cells_to_certify):
            certified[row, column] = landing.certified[index]
            aim[row, column] = landing.aim_positions_m[index]
            boundary[row, column] = landing.boundary_m[index]
            area[row, column] = landing.area_m2[index]
        evidence = bridge_api.HopperLandingEvidenceGrid(
            np.ascontiguousarray(np.flipud(certified)),
            np.ascontiguousarray(np.flipud(aim)),
            np.ascontiguousarray(np.flipud(boundary)),
            np.ascontiguousarray(np.flipud(area)),
            evidence_algorithm_id,
        )
        certified_mask = np.ascontiguousarray(certified, dtype=np.bool_)
        certified_positions = np.ascontiguousarray(
            aim[certified_mask], dtype=np.float64
        ).reshape((-1, 3))
        native_envelope = self._bridge.project_hopper_single_hop_envelope(
            self._request, evidence
        )
        envelope_eligible = np.ascontiguousarray(
            np.flipud(np.asarray(native_envelope.eligible, dtype=np.bool_))
        )
        envelope_required_delta_v = np.ascontiguousarray(
            np.flipud(
                np.asarray(
                    native_envelope.required_delta_v_mps,
                    dtype=np.float64,
                )
            )
        )
        envelope_flight_time = np.ascontiguousarray(
            np.flipud(
                np.asarray(
                    native_envelope.nominal_flight_time_s,
                    dtype=np.float64,
                )
            )
        )
        single_hop_envelope = HopperSingleHopEnvelope(
            certified_mask=certified_mask,
            certified_positions_m=certified_positions,
            eligible_mask=envelope_eligible,
            required_delta_v_mps=envelope_required_delta_v,
            nominal_flight_time_s=envelope_flight_time,
            algorithm_id=str(native_envelope.algorithm_id),
            raw_known_landing_count=int(
                native_envelope.raw_known_landing_count
            ),
            candidates_evaluated=int(native_envelope.candidates_evaluated),
            eligible_count=int(native_envelope.eligible_count),
            complete=bool(native_envelope.complete),
        )
        reachability_algorithm_id = single_hop_envelope.algorithm_id
        index_by_cell = {
            cell: index for index, cell in enumerate(cells_to_certify)
        }
        candidate_indices = np.asarray(
            [index_by_cell[tuple(cell)] for cell in candidates],
            dtype=np.int64,
        )
        candidate_certified = np.ascontiguousarray(
            landing.certified[candidate_indices], dtype=np.bool_
        )
        accepted = np.ascontiguousarray(
            envelope_eligible[candidates[:, 0], candidates[:, 1]]
            & candidate_certified,
            dtype=np.bool_,
        )
        exact_positions = np.ascontiguousarray(
            landing.aim_positions_m[candidate_indices], dtype=np.float64
        ).reshape((-1, 3))
        return (
            accepted,
            candidate_certified,
            exact_positions,
            reachability_algorithm_id,
            evidence_algorithm_id,
            single_hop_envelope,
        )


__all__ = [
    "CandidateReachabilityResult",
    "PHYSICAL_PROJECTION_SCHEMA",
    "PhysicalReachabilityResult",
    "PlatformReachabilityError",
    "HopperSingleHopEnvelope",
    "PlatformCandidateReachability",
]

"""Observed-map C++ authority for platform-specific candidate reachability."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np

from ..polar_data.raster import MapCanvas
from .coverability import PHYSICAL_PROJECTION_SCHEMA
from .observation_builder import Pose2


_PLATFORMS = frozenset(("WHEELED", "LEGGED", "HOPPER"))
_GROUND_PHYSICAL_EVIDENCE_ALGORITHM_ID = (
    "cpp-safe-traversability-projection/v1"
)


class HopperOpportunityAuthority:
    """One-boundary certified landing context shared by independent queries."""

    def __init__(
        self,
        *,
        bridge: object,
        context: object,
        certified_mask: np.ndarray,
        certified_positions_m: np.ndarray,
    ) -> None:
        self._bridge = bridge
        self._context = context
        self.certified_mask = np.ascontiguousarray(certified_mask, dtype=np.bool_)
        self.certified_positions_m = np.ascontiguousarray(
            certified_positions_m, dtype=np.float64
        ).reshape((-1, 3))
        if len(self.certified_positions_m) != int(self.certified_mask.sum()):
            raise ValueError("hopper opportunity landing counts differ")
        direct = np.ascontiguousarray(
            np.flipud(np.asarray(context.direct, dtype=np.bool_))
        )
        direct.setflags(write=False)
        self.direct_mask = direct

    @property
    def algorithm_id(self) -> str:
        return str(self._context.algorithm_id)

    def query(self, positive_mask: np.ndarray) -> object:
        positive = np.asarray(positive_mask)
        if (
            positive.dtype != np.dtype(np.bool_)
            or positive.shape != self.certified_mask.shape
        ):
            raise ValueError("hopper opportunity mask differs")
        return self._bridge.query_hopper_opportunity_distance(
            self._context, np.ascontiguousarray(np.flipud(positive))
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
    hopper_opportunity_authority: HopperOpportunityAuthority | None = None

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
        if self.platform_type != "HOPPER" and self.hopper_opportunity_authority is not None:
            raise ValueError("ground physical reachability has hopper authority")


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
    """Run the C++ ground component or Hopper certified directed graph."""

    def __init__(
        self,
        *,
        platform_type: str,
        canvas: MapCanvas,
        pose_map: Pose2,
        observed_elevation_m: np.ndarray,
        bridge: object,
        request: object,
        maximum_edge_distance_m: float = 30.0,
        local_traversability_projection: object | None = None,
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
                "project_hopper_opportunity_context",
                "query_hopper_opportunity_distance",
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
        if (
            not isinstance(maximum_edge_distance_m, float)
            or not np.isfinite(maximum_edge_distance_m)
            or maximum_edge_distance_m <= 0.0
        ):
            raise ValueError("candidate reachability edge distance is invalid")
        self._platform_type = platform_type
        self._canvas = canvas
        self._pose = pose_map
        self._elevation = elevation
        self._bridge = bridge
        self._request = request
        self._maximum_edge_distance_m = maximum_edge_distance_m
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
                opportunity_authority,
            ) = self._project_hopper_candidates(candidates, positions)
        else:
            projection = self._bridge.project_reachability(
                self._request, self._maximum_edge_distance_m
            )
            reachable, reachability_algorithm_id = (
                self._validate_reachability_projection(projection)
            )
            accepted = np.ascontiguousarray(
                reachable[candidates[:, 0], candidates[:, 1]],
                dtype=np.bool_,
            )
            local_inside, local_accepted = self._ground_local_reachability(
                candidates, positions
            )
            accepted[local_inside] = local_accepted[local_inside]
            safe = np.ones(len(candidates), dtype=np.bool_)
            exact_positions = positions
            evidence_algorithm_id = (
                _GROUND_PHYSICAL_EVIDENCE_ALGORITHM_ID
            )
            opportunity_authority = None

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
            hopper_opportunity_authority=opportunity_authority,
        )

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
    ) -> tuple[np.ndarray, str]:
        reachable = getattr(projection, "reachable", None)
        if getattr(projection, "platform_type", None) != self._platform_type:
            raise RuntimeError(
                "candidate reachability projection platform differs"
            )
        cells = self._canvas.geometry.cells
        if (
            not isinstance(reachable, np.ndarray)
            or reachable.dtype != np.dtype(np.uint8)
            or reachable.shape != (cells, cells)
            or not reachable.flags.c_contiguous
            or (reachable.size and (reachable > 1).any())
        ):
            raise RuntimeError(
                "candidate reachability projection geometry differs"
            )
        algorithm_id = getattr(projection, "algorithm_id", None)
        if not isinstance(algorithm_id, str) or not algorithm_id:
            raise RuntimeError(
                "candidate reachability projection algorithm is invalid"
            )
        return (
            np.ascontiguousarray(np.flipud(reachable).astype(np.bool_)),
            algorithm_id,
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

        # The observed detail component is the strongest available authority
        # inside the local map. In particular, it remains valid when the robot's
        # enclosing 4 m global cell is only partially observed and therefore
        # makes the conservative global projection empty. Targets outside this
        # window retain the global C++ component result in filter().
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
        HopperOpportunityAuthority,
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
        opportunity_context = self._bridge.project_hopper_opportunity_context(
            self._request, self._maximum_edge_distance_m, evidence
        )
        direct = np.asarray(getattr(opportunity_context, "direct", None))
        if (
            direct.dtype != np.dtype(np.bool_)
            or direct.shape != shape
            or not direct.flags.c_contiguous
        ):
            raise RuntimeError("hopper opportunity direct geometry differs")
        reachable = np.ascontiguousarray(np.flipud(direct), dtype=np.bool_)
        reachability_algorithm_id = getattr(
            opportunity_context, "algorithm_id", None
        )
        if not isinstance(reachability_algorithm_id, str) or not reachability_algorithm_id:
            raise RuntimeError("hopper opportunity algorithm identity is invalid")
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
            reachable[candidates[:, 0], candidates[:, 1]]
            & candidate_certified,
            dtype=np.bool_,
        )
        exact_positions = np.ascontiguousarray(
            landing.aim_positions_m[candidate_indices], dtype=np.float64
        ).reshape((-1, 3))
        certified_mask = np.ascontiguousarray(certified, dtype=np.bool_)
        certified_positions = np.ascontiguousarray(
            aim[certified_mask], dtype=np.float64
        ).reshape((-1, 3))
        authority = HopperOpportunityAuthority(
            bridge=self._bridge,
            context=opportunity_context,
            certified_mask=certified_mask,
            certified_positions_m=certified_positions,
        )
        return (
            accepted,
            candidate_certified,
            exact_positions,
            reachability_algorithm_id,
            evidence_algorithm_id,
            authority,
        )


__all__ = [
    "CandidateReachabilityResult",
    "PHYSICAL_PROJECTION_SCHEMA",
    "PhysicalReachabilityResult",
    "HopperOpportunityAuthority",
    "PlatformCandidateReachability",
]

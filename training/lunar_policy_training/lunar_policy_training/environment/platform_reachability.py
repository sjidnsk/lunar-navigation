"""Observed-map C++ authority for platform-specific candidate reachability."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np

from ..polar_data.raster import MapCanvas
from .observation_builder import Pose2


_PLATFORMS = frozenset(("WHEELED", "LEGGED", "HOPPER"))


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
                "project_direct_hopper_reachability",
            )
            if platform_type == "HOPPER"
            else ("project_reachability",)
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

    def filter(
        self,
        candidate_cells: np.ndarray,
        *,
        target_positions_map: np.ndarray | None = None,
    ) -> CandidateReachabilityResult:
        candidates = np.asarray(candidate_cells)
        cells = self._canvas.geometry.cells
        if (
            candidates.dtype != np.dtype(np.int32)
            or candidates.ndim != 2
            or candidates.shape[1:] != (2,)
            or not candidates.flags.c_contiguous
            or (candidates.size and ((candidates < 0).any() or (candidates >= cells).any()))
        ):
            raise ValueError("candidate reachability cells must be int32 [N,2]")
        if not len(candidates):
            return CandidateReachabilityResult(
                np.zeros(0, dtype=np.bool_), {"platform_unreachable_count": 0}
            )
        positions = None
        if target_positions_map is not None:
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
                if self._canvas.world_to_grid(
                    float(position[0]), float(position[1])
                ) != tuple(cell):
                    raise ValueError("candidate target position leaves its cell")
        if self._platform_type == "HOPPER":
            projection = self._hopper_projection(candidates, positions)
        else:
            projection = self._bridge.project_reachability(
                self._request, self._maximum_edge_distance_m
            )
        reachable = np.ascontiguousarray(
            np.flipud(projection.reachable).astype(np.bool_)
        )
        if reachable.shape != (cells, cells):
            raise RuntimeError("candidate reachability projection geometry differs")
        accepted = np.ascontiguousarray(
            reachable[candidates[:, 0], candidates[:, 1]], dtype=np.bool_
        )
        return CandidateReachabilityResult(
            accepted,
            {"platform_unreachable_count": int((~accepted).sum(dtype=np.int64))},
        )

    def _hopper_projection(
        self,
        candidates: np.ndarray,
        target_positions_map: np.ndarray | None,
    ) -> object:
        import lunar_planner_training_bridge as bridge_api

        start = self._canvas.world_to_grid(self._pose.x_m, self._pose.y_m)
        cells_to_certify = [start, *map(tuple, candidates.tolist())]
        cells_to_certify = list(dict.fromkeys(cells_to_certify))
        exact_targets = (
            {}
            if target_positions_map is None
            else {
                tuple(cell): tuple(float(value) for value in position)
                for cell, position in zip(
                    candidates.tolist(), target_positions_map.tolist(), strict=True
                )
            }
        )
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
            str(landing.algorithm_id),
        )
        return self._bridge.project_direct_hopper_reachability(
            self._request, self._maximum_edge_distance_m, evidence
        )


__all__ = ["CandidateReachabilityResult", "PlatformCandidateReachability"]

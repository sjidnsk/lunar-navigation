"""Observed-only deterministic frontier candidates for the frozen V2 policy."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from lunar_model_contract import ObservationContractV2

from ..polar_data.raster import GLOBAL_GEOMETRY
from .observation_builder import MissionRaster, ObservedWorld, PlatformProjection, Pose2


@dataclass(frozen=True)
class CandidateBatch:
    """Exactly 64 V2 frontier rows with boolean padding mask."""

    features: np.ndarray
    mask: np.ndarray

    def __post_init__(self) -> None:
        features = np.asarray(self.features, dtype=np.float32)
        mask = np.asarray(self.mask, dtype=bool)
        if features.shape != (64, len(ObservationContractV2.frontier_fields)) or mask.shape != (64,):
            raise ValueError("candidate batch must use [64,12] and [64]")
        if not np.isfinite(features).all():
            raise ValueError("candidate features must be finite")
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "mask", mask)

    @property
    def count(self) -> int:
        return int(self.mask.sum())

    @classmethod
    def empty(cls) -> "CandidateBatch":
        return cls(np.zeros((64, 12), dtype=np.float32), np.zeros((64,), dtype=bool))


def _neighbors(row: int, column: int) -> tuple[tuple[int, int], ...]:
    return tuple((row + dr, column + dc) for dr, dc in ((-1, 0), (0, -1), (0, 1), (1, 0)) if 0 <= row + dr < 256 and 0 <= column + dc < 256)


def _segments(boundary: np.ndarray) -> list[list[tuple[int, int]]]:
    remaining = {(int(row), int(column)) for row, column in np.argwhere(boundary)}
    result: list[list[tuple[int, int]]] = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        queue, component = [start], []
        while queue:
            point = queue.pop()
            component.append(point)
            connected = sorted(set(_neighbors(*point)) & remaining)
            for neighbor in connected:
                remaining.remove(neighbor)
                queue.append(neighbor)
        result.append(sorted(component))
    return result


def _line_of_sight(observed: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> bool:
    """Bresenham LOS using only observed cells, including endpoints."""
    row0, column0 = start
    row1, column1 = end
    delta_column, delta_row = abs(column1 - column0), abs(row1 - row0)
    step_column, step_row = (1 if column0 < column1 else -1), (1 if row0 < row1 else -1)
    error = delta_column - delta_row
    while True:
        if not (0 <= row0 < 256 and 0 <= column0 < 256 and observed[row0, column0]):
            return False
        if (row0, column0) == (row1, column1):
            return True
        doubled = 2 * error
        if doubled > -delta_row:
            error -= delta_row
            column0 += step_column
        if doubled < delta_column:
            error += delta_column
            row0 += step_row


@dataclass(frozen=True)
class _Candidate:
    feature: np.ndarray
    point: tuple[int, int]
    segment_index: int


class CandidateBuilderV2:
    """Contour → anchors/standoff → observed LOS → gains → fixed V2 slots."""

    def build(self, world: ObservedWorld, mission: MissionRaster, pose_map: Pose2, projection: PlatformProjection) -> CandidateBatch:
        if pose_map.frame_id != "map":
            return CandidateBatch.empty()
        observed = world.observed_mask
        roi = mission.roi_ratio > 0.0
        boundary = np.zeros_like(observed)
        for row, column in np.argwhere(observed & roi):
            # A frontier is a contour edge between observed ROI and unobserved ROI.
            boundary[row, column] = any(roi[next_row, next_column] and not observed[next_row, next_column] for next_row, next_column in _neighbors(int(row), int(column)))
        if not boundary.any():
            return CandidateBatch.empty()
        robot = (int(math.floor(pose_map.y_m / GLOBAL_GEOMETRY.resolution_m)), int(math.floor(pose_map.x_m / GLOBAL_GEOMETRY.resolution_m)))
        components = _segments(boundary)
        candidates: list[_Candidate] = []
        total_priority = float((mission.priority * mission.roi_ratio).sum())
        total_roi = float(mission.roi_ratio.sum())
        for segment_index, segment in enumerate(components):
            for row, column in segment:
                # Standoff remains on the observed side of the contour.  Use the
                # adjacent cell toward robot when it is observed; otherwise anchor.
                direction_row = int(np.sign(robot[0] - row))
                direction_column = int(np.sign(robot[1] - column))
                standoff = (row + direction_row, column + direction_column)
                if not (0 <= standoff[0] < 256 and 0 <= standoff[1] < 256 and observed[standoff]):
                    standoff = (row, column)
                if not _line_of_sight(observed, robot, standoff):
                    continue
                candidates.append(_Candidate(self._feature(standoff, robot, observed, roi, mission, projection, total_priority, total_roi), standoff, segment_index))
        if not candidates:
            return CandidateBatch.empty()
        selected = self._segment_representatives(candidates)
        selected = self._farthest_fill(selected, candidates)
        selected.sort(key=self._stable_key)
        selected = selected[:64]
        features = np.zeros((64, 12), dtype=np.float32)
        mask = np.zeros((64,), dtype=bool)
        for index, candidate in enumerate(selected):
            features[index] = candidate.feature
            mask[index] = True
        return CandidateBatch(features, mask)

    @staticmethod
    def _feature(point: tuple[int, int], robot: tuple[int, int], observed: np.ndarray, roi: np.ndarray, mission: MissionRaster, projection: PlatformProjection, total_priority: float, total_roi: float) -> np.ndarray:
        row, column = point
        x, y = (column + 0.5) * 4.0, (row + 0.5) * 4.0
        robot_x, robot_y = (robot[1] + 0.5) * 4.0, (robot[0] + 0.5) * 4.0
        dx, dy = x - robot_x, y - robot_y
        distance = math.hypot(dx, dy)
        unobserved_neighbors = [(next_row - row, next_column - column) for next_row, next_column in _neighbors(row, column) if roi[next_row, next_column] and not observed[next_row, next_column]]
        normal_row = float(sum(item[0] for item in unobserved_neighbors))
        normal_column = float(sum(item[1] for item in unobserved_neighbors))
        normal_norm = math.hypot(normal_row, normal_column)
        radius = 6
        nearby_roi = mission.roi_ratio[max(0, row - radius):row + radius + 1, max(0, column - radius):column + radius + 1]
        nearby_observed = observed[max(0, row - radius):row + radius + 1, max(0, column - radius):column + radius + 1]
        nearby_unobserved = nearby_roi * (~nearby_observed)
        nearby_priority = mission.priority[max(0, row - radius):row + radius + 1, max(0, column - radius):column + radius + 1]
        gain = float(nearby_unobserved.sum() / total_roi) if total_roi else 0.0
        weighted_gain = float((nearby_priority * nearby_unobserved).sum() / total_priority) if total_priority else 0.0
        remaining = float((mission.roi_ratio * ~observed).sum() / mission.roi_ratio.sum()) if mission.roi_ratio.any() else 0.0
        return np.asarray((
            (x - 512.0) / 512.0, (y - 512.0) / 512.0,
            distance / (math.sqrt(2.0) * 512.0), dy / distance if distance else 0.0, dx / distance if distance else 1.0,
            gain, weighted_gain,
            normal_row / normal_norm if normal_norm else 0.0, normal_column / normal_norm if normal_norm else 1.0,
            min(1.0, normal_norm / 2.0), projection.clearance_margin_norm[row, column], remaining,
        ), dtype=np.float32)

    @staticmethod
    def _segment_representatives(candidates: list[_Candidate]) -> list[_Candidate]:
        representatives: list[_Candidate] = []
        for segment in sorted({candidate.segment_index for candidate in candidates}):
            choices = [candidate for candidate in candidates if candidate.segment_index == segment]
            representatives.append(min(choices, key=CandidateBuilderV2._stable_key))
        return representatives[:64]

    @staticmethod
    def _farthest_fill(selected: list[_Candidate], candidates: list[_Candidate]) -> list[_Candidate]:
        selected_ids = {(candidate.segment_index, candidate.point) for candidate in selected}
        available = [candidate for candidate in candidates if (candidate.segment_index, candidate.point) not in selected_ids]
        while available and len(selected) < 64:
            def distance_to_selection(item: _Candidate) -> tuple[float, tuple[float, ...]]:
                nearest = min(math.dist(item.point, chosen.point) for chosen in selected) if selected else float("inf")
                return (-nearest, CandidateBuilderV2._stable_key(item))
            choice_index = min(range(len(available)), key=lambda index: distance_to_selection(available[index]))
            choice = available[choice_index]
            selected.append(choice)
            available.pop(choice_index)
        return selected

    @staticmethod
    def _stable_key(candidate: _Candidate) -> tuple[float, ...]:
        feature = candidate.feature
        return (-float(feature[6]), -float(feature[5]), float(feature[2]), float(feature[1]), float(feature[0]))


__all__ = ["CandidateBatch", "CandidateBuilderV2"]

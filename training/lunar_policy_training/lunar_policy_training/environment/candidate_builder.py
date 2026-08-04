"""Observed-only deterministic frontier candidates for the frozen V2 policy."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from lunar_model_contract import ObservationContractV2

from .observation_builder import MissionRaster, ObservedWorld, PlatformProjection, Pose2


@dataclass(frozen=True)
class SensorGeometry:
    range_m: float
    fov_rad: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.range_m) or self.range_m <= 0.0 or not 0.0 < self.fov_rad <= 2.0 * math.pi:
            raise ValueError("sensor range/FOV are invalid")

    @property
    def anchor_spacing_m(self) -> float:
        return max(4.0, self.range_m / 8.0)

    @property
    def standoff_m(self) -> float:
        return max(4.0, self.range_m / 16.0)


@dataclass(frozen=True)
class CandidateBatch:
    features: np.ndarray
    mask: np.ndarray
    canvas_id: str | None = None

    def __post_init__(self) -> None:
        features, mask = np.asarray(self.features, dtype=np.float32), np.asarray(self.mask, dtype=bool)
        if features.shape != (64, len(ObservationContractV2.frontier_fields)) or mask.shape != (64,) or not np.isfinite(features).all():
            raise ValueError("candidate batch must use finite [64,12] and [64]")
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "mask", mask)

    @property
    def count(self) -> int:
        return int(self.mask.sum())

    @classmethod
    def empty(cls, canvas_id: str | None = None) -> "CandidateBatch":
        return cls(np.zeros((64, 12), np.float32), np.zeros(64, bool), canvas_id)


def _neighbors(row: int, column: int, cells: int) -> tuple[tuple[int, int], ...]:
    return tuple((row + dr, column + dc) for dr, dc in ((-1, 0), (0, -1), (0, 1), (1, 0)) if 0 <= row + dr < cells and 0 <= column + dc < cells)


def _ray_cells(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    row0, column0 = start; row1, column1 = end
    dc, dr = abs(column1 - column0), abs(row1 - row0)
    sc, sr = (1 if column0 < column1 else -1), (1 if row0 < row1 else -1)
    error, result = dc - dr, []
    while True:
        result.append((row0, column0))
        if (row0, column0) == (row1, column1):
            return result
        doubled = 2 * error
        if doubled > -dr: error -= dr; column0 += sc
        if doubled < dc: error += dc; row0 += sr


def _clear_observed(world: ObservedWorld, cells: list[tuple[int, int]], *, unknown_endpoint_allowed: bool = False) -> bool:
    check = cells[:-1] if unknown_endpoint_allowed else cells
    return bool(check) and all(world.observed_mask[row, column] and world.physical_obstacle_ratio[row, column] == 0.0 for row, column in check)


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


class CandidateBuilderV2:
    def __init__(self, sensor: SensorGeometry = SensorGeometry(80.0, 2.0 * math.pi)) -> None:
        self._sensor = sensor

    def build(self, world: ObservedWorld, mission: MissionRaster, pose_map: Pose2, projection: PlatformProjection) -> CandidateBatch:
        if pose_map.frame_id != "map" or world.canvas != mission.canvas or world.canvas != projection.canvas:
            return CandidateBatch.empty()
        canvas, cells, observed, roi = world.canvas, world.canvas.geometry.cells, world.observed_mask, mission.roi_ratio > 0.0
        try:
            robot = canvas.world_to_grid(pose_map.x_m, pose_map.y_m)
        except ValueError:
            return CandidateBatch.empty()
        boundary = np.zeros_like(observed)
        for row, column in np.argwhere(observed & roi):
            boundary[row, column] = any(roi[next_row, next_column] and not observed[next_row, next_column] for next_row, next_column in _neighbors(int(row), int(column), cells))
        spacing = max(1, math.ceil(self._sensor.anchor_spacing_m / canvas.geometry.resolution_m))
        chosen: list[tuple[tuple[int, int], np.ndarray]] = []
        total_roi, total_priority = float(mission.roi_ratio.sum()), float((mission.priority * mission.roi_ratio).sum())
        for index, (row, column) in enumerate(np.argwhere(boundary)):
            if index % spacing:
                continue
            row, column = int(row), int(column)
            step = max(1, round(self._sensor.standoff_m / canvas.geometry.resolution_m))
            standoff = (row + int(np.sign(robot[0] - row)) * step, column + int(np.sign(robot[1] - column)) * step)
            if not (0 <= standoff[0] < cells and 0 <= standoff[1] < cells and observed[standoff]): standoff = (row, column)
            if projection.traversable_ratio[standoff] == 0.0 or not _clear_observed(world, _ray_cells(robot, standoff)):
                continue
            feature = self._feature(world, mission, projection, pose_map, robot, standoff, total_roi, total_priority)
            if feature is not None:
                chosen.append((standoff, feature))
        by_point = {point: feature for point, feature in chosen}
        representatives: list[tuple[tuple[int, int], np.ndarray]] = []
        for segment in _segments(list(by_point), cells):
            representatives.append(min(((point, by_point[point]) for point in segment), key=lambda item: tuple(item[1].tolist())))
        selected = representatives[:64]
        remaining = [(point, feature) for point, feature in chosen if point not in {item[0] for item in selected}]
        while remaining and len(selected) < 64:
            index = max(range(len(remaining)), key=lambda idx: min(math.dist(remaining[idx][0], item[0]) for item in selected))
            selected.append(remaining.pop(index))
        chosen = sorted(selected, key=lambda item: tuple(item[1].tolist()))
        output = np.zeros((64, 12), np.float32); mask = np.zeros(64, bool)
        for index, (_, feature) in enumerate(chosen[:64]): output[index] = feature; mask[index] = True
        return CandidateBatch(output, mask, canvas.identity)

    def _feature(self, world: ObservedWorld, mission: MissionRaster, projection: PlatformProjection, pose: Pose2, robot: tuple[int, int], point: tuple[int, int], total_roi: float, total_priority: float) -> np.ndarray | None:
        canvas = world.canvas; x, y = canvas.grid_center_world(*point)
        dx, dy = x - pose.x_m, y - pose.y_m; distance = math.hypot(dx, dy)
        bearing = math.atan2(dy, dx)
        if distance > self._sensor.range_m or abs(math.atan2(math.sin(bearing - pose.yaw_rad), math.cos(bearing - pose.yaw_rad))) > self._sensor.fov_rad / 2.0:
            return None
        gain = priority_gain = 0.0
        for row, column in np.argwhere((mission.roi_ratio > 0.0) & ~world.observed_mask):
            tx, ty = canvas.grid_center_world(int(row), int(column))
            if math.hypot(tx - x, ty - y) > self._sensor.range_m:
                continue
            target_bearing = math.atan2(ty - y, tx - x)
            if abs(math.atan2(math.sin(target_bearing - bearing), math.cos(target_bearing - bearing))) > self._sensor.fov_rad / 2.0:
                continue
            if _clear_observed(world, _ray_cells(point, (int(row), int(column))), unknown_endpoint_allowed=True):
                gain += float(mission.roi_ratio[row, column]); priority_gain += float(mission.priority[row, column] * mission.roi_ratio[row, column])
        if gain == 0.0:
            return None
        normal = np.array([0.0, 0.0])
        for neighbor in _neighbors(*point, canvas.geometry.cells):
            if mission.roi_ratio[neighbor] > 0 and not world.observed_mask[neighbor]: normal += (neighbor[0] - point[0], neighbor[1] - point[1])
        magnitude = float(np.linalg.norm(normal)); remaining = float((mission.roi_ratio * ~world.observed_mask).sum() / total_roi) if total_roi else 0.0
        return np.asarray(((x - canvas.bounds_m[0]) / canvas.geometry.size_m, (canvas.bounds_m[3] - y) / canvas.geometry.size_m, min(1.0, distance / (math.sqrt(2.0) * canvas.geometry.size_m)), math.sin(bearing), math.cos(bearing), min(1.0, gain / total_roi) if total_roi else 0.0, min(1.0, priority_gain / total_priority) if total_priority else 0.0, -normal[0] / magnitude if magnitude else 0.0, normal[1] / magnitude if magnitude else 1.0, min(1.0, magnitude / 2.0), projection.clearance_margin_norm[point], remaining), dtype=np.float32)


__all__ = ["CandidateBatch", "CandidateBuilderV2", "SensorGeometry"]

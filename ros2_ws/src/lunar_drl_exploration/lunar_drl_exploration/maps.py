"""Read-only cache for GetPolicyMap full snapshots and tile deltas."""

from dataclasses import dataclass
from math import atan2
from typing import Any

import numpy as np

from .contracts import MapSnapshot, Pose, freeze_array

UNKNOWN = 0
FREE = 1
BLOCKED = 2
_TILE_WIDTH = 256
_TILE_CELLS = _TILE_WIDTH * _TILE_WIDTH


def _field(value: Any, name: str, default=None):
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def _wire_origin(value: Any) -> tuple:
    if isinstance(value, (tuple, list, np.ndarray)):
        return tuple(value)
    if value is None:
        raise ValueError("policy map response is missing origin")
    return (float(value.x), float(value.y), float(value.z))


def _wire_pose(response: Any) -> Pose:
    pose = _field(response, "pose")
    if isinstance(pose, Pose): return pose
    if pose is not None:
        return Pose(float(_field(pose, "x")), float(_field(pose, "y")), float(_field(pose, "yaw")))
    anchor = _field(response, "anchor_pose")
    if anchor is None:
        raise ValueError("policy map response is missing anchor pose")
    orientation = anchor.orientation
    return Pose(float(anchor.position.x), float(anchor.position.y),
                float(atan2(2.0 * (orientation.w * orientation.z + orientation.x * orientation.y),
                            1.0 - 2.0 * (orientation.y ** 2 + orientation.z ** 2))))


@dataclass(frozen=True)
class PolicyMapTile:
    states: np.ndarray
    intrinsic_states: np.ndarray
    observed: np.ndarray
    costs: np.ndarray
    elevation_m: np.ndarray

    def __post_init__(self):
        for name, dtype in (("states", np.uint8), ("intrinsic_states", np.uint8),
                            ("observed", np.uint8), ("costs", np.float32),
                            ("elevation_m", np.float32)):
            value = freeze_array(getattr(self, name), dtype=dtype).reshape(_TILE_CELLS)
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class PolicyMapCell:
    state: int
    intrinsic_state: int
    navigation_state: int
    cost: float
    elevation_m: float


@dataclass(frozen=True)
class PolicyMapRaster:
    bounds: tuple
    states: np.ndarray
    intrinsic_states: np.ndarray
    navigation_states: np.ndarray
    costs: np.ndarray
    elevation_m: np.ndarray

    def __post_init__(self):
        for name, dtype in (("states", np.uint8), ("intrinsic_states", np.uint8),
                            ("navigation_states", np.uint8), ("costs", np.float32),
                            ("elevation_m", np.float32)):
            object.__setattr__(self, name, freeze_array(getattr(self, name), dtype=dtype))


class PolicyMapStore:
    """Applies only revision-contiguous deltas; snapshots never mutate."""

    def __init__(self):
        self._snapshot: MapSnapshot | None = None

    def apply(self, response: Any) -> MapSnapshot:
        if not _field(response, "ready", False):
            raise ValueError(_field(response, "reason_code", "policy map is unavailable"))
        epoch = _field(response, "epoch")
        revision = int(_field(response, "fine_revision", _field(response, "revision")))
        full_snapshot = bool(_field(response, "full_snapshot", False))
        if self._snapshot is None and not full_snapshot:
            raise ValueError("initial policy map requires a full snapshot")
        if self._snapshot is not None:
            if epoch != self._snapshot.epoch and not full_snapshot:
                raise ValueError("epoch replacement requires a full snapshot")
            if epoch == self._snapshot.epoch and not full_snapshot and revision != self._snapshot.revision + 1:
                raise ValueError("missed policy-map revision requires a full snapshot")
            tiles = {} if full_snapshot else dict(self._snapshot.tiles)
        else:
            tiles = {}
        for payload in _field(response, "tiles", ()):
            key = (int(_field(payload, "tile_x")), int(_field(payload, "tile_y")))
            tiles[key] = PolicyMapTile(
                states=_field(payload, "states"),
                intrinsic_states=_field(payload, "intrinsic_states"),
                observed=_field(payload, "observed"),
                costs=_field(payload, "costs"),
                elevation_m=_field(payload, "elevation_m"),
            )
        pose = _wire_pose(response)
        connections = _field(response, "start_connections")
        if connections is None:
            xs, ys = _field(response, "start_connection_x", ()), _field(response, "start_connection_y", ())
            if len(xs) != len(ys):
                raise ValueError("policy map start connection arrays differ in length")
            connections = np.column_stack((xs, ys)) if xs else np.empty((0, 2), dtype=np.int64)
        self._snapshot = PolicyMapSnapshot(
            epoch=str(epoch), revision=revision,
            resolution_m=float(_field(response, "resolution_m")),
            origin=_wire_origin(_field(response, "origin")), tiles=tiles, pose=pose,
            start_connections=connections,
            local_bounds=tuple(_field(response, "local_bounds", (0, 0, 0, 0))),
            profile_hash=str(_field(response, "profile_hash", "")),
            start_connection_status=str(_field(response, "start_connection_status", "READY")),
            goal_position_tolerance_m=float(_field(response, "goal_position_tolerance_m", np.nan)),
            goal_yaw_tolerance_rad=float(_field(response, "goal_yaw_tolerance_rad", np.nan)),
        )
        return self._snapshot


class PolicyMapSnapshot(MapSnapshot):
    def cell_at(self, ix: int, iy: int) -> PolicyMapCell:
        tile = self.tiles.get((ix // _TILE_WIDTH, iy // _TILE_WIDTH))
        if tile is None:
            return PolicyMapCell(UNKNOWN, UNKNOWN, UNKNOWN, 0.0, float("nan"))
        offset = (iy % _TILE_WIDTH) * _TILE_WIDTH + ix % _TILE_WIDTH
        return PolicyMapCell(int(tile.observed[offset]), int(tile.intrinsic_states[offset]),
                             int(tile.states[offset]), float(tile.costs[offset]),
                             float(tile.elevation_m[offset]))

    def raster(self, bounds: tuple) -> PolicyMapRaster:
        min_x, min_y, max_x, max_y = bounds
        if max_x < min_x or max_y < min_y:
            raise ValueError("raster bounds are inverted")
        shape = (max_y - min_y, max_x - min_x)
        states = np.full(shape, UNKNOWN, dtype=np.uint8)
        intrinsic = np.full(shape, UNKNOWN, dtype=np.uint8)
        navigation = np.full(shape, UNKNOWN, dtype=np.uint8)
        costs = np.zeros(shape, dtype=np.float32)
        elevations = np.full(shape, np.nan, dtype=np.float32)
        first_tile_x, last_tile_x = min_x // _TILE_WIDTH, (max_x - 1) // _TILE_WIDTH
        first_tile_y, last_tile_y = min_y // _TILE_WIDTH, (max_y - 1) // _TILE_WIDTH
        for tile_y in range(first_tile_y, last_tile_y + 1):
            for tile_x in range(first_tile_x, last_tile_x + 1):
                tile = self.tiles.get((tile_x, tile_y))
                if tile is None: continue
                tile_min_x, tile_min_y = tile_x * _TILE_WIDTH, tile_y * _TILE_WIDTH
                left, right = max(min_x, tile_min_x), min(max_x, tile_min_x + _TILE_WIDTH)
                top, bottom = max(min_y, tile_min_y), min(max_y, tile_min_y + _TILE_WIDTH)
                tile_slice = np.s_[top - tile_min_y:bottom - tile_min_y, left - tile_min_x:right - tile_min_x]
                output_slice = np.s_[top - min_y:bottom - min_y, left - min_x:right - min_x]
                states[output_slice] = tile.observed.reshape(_TILE_WIDTH, _TILE_WIDTH)[tile_slice]
                intrinsic[output_slice] = tile.intrinsic_states.reshape(_TILE_WIDTH, _TILE_WIDTH)[tile_slice]
                navigation[output_slice] = tile.states.reshape(_TILE_WIDTH, _TILE_WIDTH)[tile_slice]
                costs[output_slice] = tile.costs.reshape(_TILE_WIDTH, _TILE_WIDTH)[tile_slice]
                elevations[output_slice] = tile.elevation_m.reshape(_TILE_WIDTH, _TILE_WIDTH)[tile_slice]
        return PolicyMapRaster(bounds, states, intrinsic, navigation, costs, elevations)

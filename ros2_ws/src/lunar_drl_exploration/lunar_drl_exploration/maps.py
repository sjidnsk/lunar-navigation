"""Read-only cache for GetPolicyMap full snapshots and tile deltas."""

from dataclasses import dataclass
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
        pose = _field(response, "pose", Pose(0.0, 0.0, 0.0))
        if not isinstance(pose, Pose):
            pose = Pose(float(_field(pose, "x")), float(_field(pose, "y")),
                        float(_field(pose, "yaw")))
        self._snapshot = PolicyMapSnapshot(
            epoch=str(epoch), revision=revision,
            resolution_m=float(_field(response, "resolution_m")),
            origin=tuple(_field(response, "origin")), tiles=tiles, pose=pose,
            start_connections=_field(response, "start_connections", np.empty((0, 2))),
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
        for row, iy in enumerate(range(min_y, max_y)):
            for col, ix in enumerate(range(min_x, max_x)):
                cell = self.cell_at(ix, iy)
                states[row, col] = cell.state
                intrinsic[row, col] = cell.intrinsic_state
                navigation[row, col] = cell.navigation_state
                costs[row, col] = cell.cost
                elevations[row, col] = cell.elevation_m
        return PolicyMapRaster(bounds, states, intrinsic, navigation, costs, elevations)

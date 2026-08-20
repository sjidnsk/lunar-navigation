"""Immutable, revision-keyed Task3 global-map snapshots.

This module intentionally keeps the SQLite/NumPy boundary and its cache in one
runtime language.  It never copies the complete Task3 map: a snapshot contains
only the tiles read for one frozen task ROI.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Callable

from .t3_sqlite_reader import Task3MapError, Task3MapRead, Task3Tile, read_roi_at_revision


TaskRoi = tuple[float, float, float, float]
TileReader = Callable[[Path | str, TaskRoi, int], Task3MapRead]

_SCALES = (1, 2, 4, 8, 16, 20)
_BASE_RESOLUTION_M = 0.2
_TARGET_AXIS_CELLS = 256


@dataclass(frozen=True)
class SelectedGlobalLevel:
    level: int
    resolution_m: float
    width: int
    height: int


@dataclass(frozen=True)
class GlobalMapSnapshot:
    mission_id: str
    mission_revision: int
    map_revision: int
    roi: TaskRoi
    selected_level: SelectedGlobalLevel
    source: Task3MapRead


def _select_level(roi: TaskRoi) -> SelectedGlobalLevel:
    min_x, min_y, max_x, max_y = roi
    if (
        not all(math.isfinite(value) for value in roi)
        or max_x <= min_x
        or max_y <= min_y
    ):
        raise Task3MapError("TASK_ROI_INVALID")
    for level, scale in enumerate(_SCALES):
        resolution_m = _BASE_RESOLUTION_M * scale
        width = math.ceil((max_x - min_x) / resolution_m)
        height = math.ceil((max_y - min_y) / resolution_m)
        if width <= _TARGET_AXIS_CELLS and height <= _TARGET_AXIS_CELLS:
            return SelectedGlobalLevel(level, resolution_m, width, height)
    raise Task3MapError("GLOBAL_MAP_SCALE_UNSUPPORTED")


class GlobalMapCache:
    """Reuse complete immutable snapshots and bound raw tile residency."""

    def __init__(
        self,
        database_path: Path | str,
        *,
        provider: TileReader = read_roi_at_revision,
        max_cached_tiles: int,
    ) -> None:
        if max_cached_tiles <= 0:
            raise ValueError("TASK3_TILE_CACHE_INVALID")
        self._database_path = Path(database_path)
        self._provider = provider
        self._max_cached_tiles = max_cached_tiles
        self._snapshots: dict[tuple[str, int, TaskRoi, int], GlobalMapSnapshot] = {}
        self._tiles: OrderedDict[tuple[int, int, int], Task3Tile] = OrderedDict()

    @property
    def cached_tile_count(self) -> int:
        return len(self._tiles)

    def refresh(
        self,
        mission_id: str,
        mission_revision: int,
        roi: TaskRoi,
        map_revision: int,
    ) -> GlobalMapSnapshot:
        if not mission_id or mission_revision < 0 or map_revision < 0:
            raise Task3MapError("TASK3_SNAPSHOT_KEY_INVALID")
        key = (mission_id, mission_revision, roi, map_revision)
        previous = self._snapshots.get(key)
        if previous is not None:
            return previous

        selected_level = _select_level(roi)
        source = self._provider(self._database_path, roi, map_revision)
        if source.metadata.map_revision != map_revision:
            raise Task3MapError("TASK3_MAP_REVISION_MISMATCH")
        if not source.tiles:
            raise Task3MapError("TASK3_TILE_MISSING")

        for tile in source.tiles:
            tile_key = (tile.tile_x, tile.tile_y, map_revision)
            self._tiles[tile_key] = tile
            self._tiles.move_to_end(tile_key)
        while len(self._tiles) > self._max_cached_tiles:
            self._tiles.popitem(last=False)

        snapshot = GlobalMapSnapshot(
            mission_id=mission_id,
            mission_revision=mission_revision,
            map_revision=map_revision,
            roi=roi,
            selected_level=selected_level,
            source=source,
        )
        self._snapshots[key] = snapshot
        return snapshot

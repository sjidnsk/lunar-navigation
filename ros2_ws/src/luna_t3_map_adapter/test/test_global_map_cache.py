from __future__ import annotations

from collections import Counter

import numpy as np

from luna_t3_map_adapter.global_map_cache import GlobalMapCache
from luna_t3_map_adapter.t3_sqlite_reader import (
    Task3MapRead,
    Task3Metadata,
    Task3Tile,
)


def _read(revision: int) -> Task3MapRead:
    layers = {
        "occupancy": np.zeros((256, 256), dtype=np.int8),
        "semantic": np.zeros((256, 256), dtype=np.uint8),
        "semantic_confidence": np.zeros((256, 256), dtype=np.uint8),
        "elevation": np.zeros((256, 256), dtype=np.int16),
        "elevation_variance": np.zeros((256, 256), dtype=np.uint16),
        "height_range": np.zeros((256, 256), dtype=np.uint16),
        "roughness": np.zeros((256, 256), dtype=np.uint16),
        "observation_count": np.ones((256, 256), dtype=np.uint16),
    }
    metadata = Task3Metadata(revision, "map", 0.2, 256, 51.2, 0.0, 0.0)
    return Task3MapRead(
        metadata,
        tuple(Task3Tile(x, y, revision, 0.0, layers) for y in range(2) for x in range(2)),
    )


def test_unchanged_snapshot_is_reused_without_another_source_read() -> None:
    calls: Counter[int] = Counter()

    def provider(_path: str, _roi: tuple[float, float, float, float], revision: int) -> Task3MapRead:
        calls[revision] += 1
        return _read(revision)

    cache = GlobalMapCache("/unused.sqlite3", provider=provider, max_cached_tiles=8)
    roi = (0.0, 0.0, 100.0, 100.0)

    first = cache.refresh("mission-a", 1, roi, 7)
    second = cache.refresh("mission-a", 1, roi, 7)

    assert second is first
    assert calls == Counter({7: 1})
    assert first.selected_level.level == 1
    assert first.selected_level.resolution_m == 0.4


def test_new_revision_replaces_snapshot_without_mutating_old_request() -> None:
    def provider(_path: str, _roi: tuple[float, float, float, float], revision: int) -> Task3MapRead:
        return _read(revision)

    cache = GlobalMapCache("/unused.sqlite3", provider=provider, max_cached_tiles=4)
    roi = (0.0, 0.0, 300.0, 300.0)

    old = cache.refresh("mission-a", 1, roi, 7)
    new = cache.refresh("mission-a", 1, roi, 8)

    assert new is not old
    assert old.map_revision == 7
    assert new.map_revision == 8
    assert old.selected_level == new.selected_level


def test_task_revision_uses_a_distinct_immutable_snapshot() -> None:
    cache = GlobalMapCache("/unused.sqlite3", provider=lambda *_args: _read(7), max_cached_tiles=4)
    roi = (0.0, 0.0, 100.0, 100.0)

    first = cache.refresh("mission-a", 1, roi, 7)
    next_task = cache.refresh("mission-a", 2, roi, 7)

    assert next_task is not first
    assert first.mission_revision == 1
    assert next_task.mission_revision == 2


def test_raw_tile_cache_is_bounded_after_a_new_map_revision() -> None:
    cache = GlobalMapCache("/unused.sqlite3", provider=lambda *_args: _read(7), max_cached_tiles=2)
    roi = (0.0, 0.0, 100.0, 100.0)

    cache.refresh("mission-a", 1, roi, 7)

    assert cache.cached_tile_count == 2

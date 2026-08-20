from __future__ import annotations

import numpy as np

from luna_t3_map_adapter.global_map_coordinator import GlobalMapCoordinator
from luna_t3_map_adapter.global_map_cache import GlobalMapCache
from luna_t3_map_adapter.t3_sqlite_reader import Task3MapRead, Task3Metadata, Task3Tile


def _source(_path: str, _roi: tuple[float, float, float, float], revision: int) -> Task3MapRead:
    layers = {
        "occupancy": np.zeros((256, 256), dtype=np.int8),
        "semantic": np.zeros((256, 256), dtype=np.uint8),
        "semantic_confidence": np.full((256, 256), 255, dtype=np.uint8),
        "elevation": np.zeros((256, 256), dtype=np.int16),
        "elevation_variance": np.ones((256, 256), dtype=np.uint16),
        "height_range": np.ones((256, 256), dtype=np.uint16),
        "roughness": np.ones((256, 256), dtype=np.uint16),
        "observation_count": np.ones((256, 256), dtype=np.uint16),
    }
    return Task3MapRead(
        Task3Metadata(revision, "map", 0.2, 256, 51.2, 0.0, 0.0),
        (Task3Tile(0, 0, revision, 0.0, layers),),
    )


def test_publishes_only_after_active_task_and_new_revision() -> None:
    coordinator = GlobalMapCoordinator(
        GlobalMapCache("/unused.sqlite3", provider=_source, max_cached_tiles=4)
    )

    assert coordinator.accept_revision(7) is None
    assert coordinator.accept_task("mission-a", 1, 1, (0.0, 0.0, 100.0, 100.0)) is not None
    assert coordinator.accept_revision(7) is None
    assert coordinator.accept_revision(8) is not None


def test_paused_task_never_publishes_a_new_global_map() -> None:
    coordinator = GlobalMapCoordinator(
        GlobalMapCache("/unused.sqlite3", provider=_source, max_cached_tiles=4)
    )

    coordinator.accept_revision(7)
    assert coordinator.accept_task("mission-a", 1, 2, (0.0, 0.0, 100.0, 100.0)) is None
    assert coordinator.accept_revision(8) is None

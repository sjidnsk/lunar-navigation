from __future__ import annotations

import numpy as np

from luna_t3_map_adapter.global_map_cache import GlobalMapCache
from luna_t3_map_adapter.global_grid_conversion import (
    to_canonical_global_grid,
    to_grid_map_message,
)
from luna_t3_map_adapter.t3_sqlite_reader import Task3MapRead, Task3Metadata, Task3Tile


def _source(_path: str, _roi: tuple[float, float, float, float], revision: int) -> Task3MapRead:
    layers = {
        "occupancy": np.zeros((256, 256), dtype=np.int8),
        "semantic": np.zeros((256, 256), dtype=np.uint8),
        "semantic_confidence": np.full((256, 256), 255, dtype=np.uint8),
        "elevation": np.full((256, 256), 100, dtype=np.int16),
        "elevation_variance": np.full((256, 256), 100, dtype=np.uint16),
        "height_range": np.full((256, 256), 100, dtype=np.uint16),
        "roughness": np.full((256, 256), 100, dtype=np.uint16),
        "observation_count": np.full((256, 256), 3, dtype=np.uint16),
    }
    layers["occupancy"][0, 0] = 100
    layers["elevation"][0, 2] = -32768
    metadata = Task3Metadata(revision, "map", 0.2, 256, 51.2, 0.0, 0.0)
    return Task3MapRead(metadata, (Task3Tile(0, 0, revision, 1.0, layers),))


def test_conversion_has_all_canonical_layers_and_aggregates_safety() -> None:
    cache = GlobalMapCache("/unused.sqlite3", provider=_source, max_cached_tiles=4)
    snapshot = cache.refresh("mission", 1, (0.0, 0.0, 100.0, 100.0), 7)

    grid = to_canonical_global_grid(snapshot)

    assert grid.frame_id == "map"
    assert grid.resolution_m == 0.4
    assert grid.width == grid.height == 250
    assert tuple(grid.layers) == (
        "elevation", "valid_mask", "obstacle", "obstacle_height",
        "observation_age_s", "observation_quality", "elevation_variance",
        "obstacle_variance", "observation_count", "forbidden",
    )
    assert grid.layers["obstacle"][0, 0] == 1.0
    assert grid.layers["valid_mask"][0, 0] == 1.0
    assert grid.layers["valid_mask"][0, 1] == 0.0
    assert grid.layers["forbidden"][0, 1] == 1.0


def test_unknown_task3_cells_never_become_traversable() -> None:
    cache = GlobalMapCache("/unused.sqlite3", provider=_source, max_cached_tiles=4)
    snapshot = cache.refresh("mission", 1, (0.0, 0.0, 100.0, 100.0), 7)
    grid = to_canonical_global_grid(snapshot)

    assert np.all(grid.layers["valid_mask"][grid.layers["forbidden"] == 1.0] == 0.0)


def test_ros_message_has_canonical_geometry_and_unrotated_layout() -> None:
    cache = GlobalMapCache("/unused.sqlite3", provider=_source, max_cached_tiles=4)
    grid = to_canonical_global_grid(cache.refresh("mission", 1, (0.0, 0.0, 100.0, 100.0), 7))

    message = to_grid_map_message(grid)

    assert message.header.frame_id == "map"
    assert message.info.resolution == 0.4
    assert message.info.length_x == message.info.length_y == 100.0
    assert message.info.pose.orientation.w == 1.0
    assert tuple(message.layers) == tuple(grid.layers)
    assert tuple(message.basic_layers) == ("elevation", "valid_mask")
    assert message.outer_start_index == message.inner_start_index == 0
    assert len(message.data) == 10
    assert all(len(layer.data) == 250 * 250 for layer in message.data)

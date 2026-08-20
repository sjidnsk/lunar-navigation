from __future__ import annotations

import numpy as np

from luna_t3_map_adapter.grid_map_codec import decode_layers, encode_grid_map
from luna_t3_map_adapter.local_grid_conversion import LocalMapPolicy, ObservationLedger
from luna_t3_map_adapter.local_map_adapter_node import adapt_local_grid_message
from luna_t3_map_adapter.local_tile_evidence import PlanarTransform
from luna_t3_map_adapter.t3_sqlite_reader import Task3MapRead, Task3Metadata, Task3Tile


def _message():
    message = encode_grid_map(
        frame_id="odom", resolution_m=0.2, origin_x_m=0.0, origin_y_m=0.0,
        layers={
            "occupancy": np.zeros((2, 2), dtype=np.float32),
            "semantic_id": np.zeros((2, 2), dtype=np.float32),
            "elevation": np.ones((2, 2), dtype=np.float32),
            "roughness": np.full((2, 2), 0.02, dtype=np.float32),
        },
        basic_layers=("elevation",),
    )
    message.header.stamp.sec = 4
    return message


def _snapshot() -> Task3MapRead:
    layers = {
        "occupancy": np.zeros((256, 256), dtype=np.int8),
        "semantic": np.zeros((256, 256), dtype=np.uint8),
        "semantic_confidence": np.full((256, 256), 255, dtype=np.uint8),
        "elevation": np.zeros((256, 256), dtype=np.int16),
        "elevation_variance": np.full((256, 256), 100, dtype=np.uint16),
        "height_range": np.full((256, 256), 100, dtype=np.uint16),
        "roughness": np.full((256, 256), 100, dtype=np.uint16),
        "observation_count": np.full((256, 256), 3, dtype=np.uint16),
    }
    return Task3MapRead(
        Task3Metadata(1, "map", 0.2, 256, 51.2, 0.0, 0.0),
        (Task3Tile(0, 0, 1, 0.0, layers),),
    )


def _policy() -> LocalMapPolicy:
    return LocalMapPolicy(50, frozenset({3}), frozenset({4}), 30.0)


def test_adapter_publishes_same_odom_geometry_with_ten_canonical_layers() -> None:
    source = _message()
    output = adapt_local_grid_message(
        source, _snapshot(), PlanarTransform.identity(), _policy(), ObservationLedger()
    )

    assert output.header.frame_id == "odom"
    assert output.header.stamp == source.header.stamp
    assert output.info.resolution == source.info.resolution
    assert output.info.length_x == source.info.length_x
    assert output.info.length_y == source.info.length_y
    layers = decode_layers(output, tuple(output.layers))
    assert len(layers) == 10
    assert np.all(layers["valid_mask"] == 1.0)

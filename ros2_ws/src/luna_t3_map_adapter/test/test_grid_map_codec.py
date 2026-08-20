from __future__ import annotations

import numpy as np
import pytest

from luna_t3_map_adapter.grid_map_codec import (
    GridMapCodecError,
    decode_layers,
    encode_grid_map,
)


def test_round_trip_preserves_02m_shape_values_and_ring_indices() -> None:
    source = {
        "elevation": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "valid_mask": np.array([[1.0, 1.0], [1.0, 0.0]], dtype=np.float32),
    }

    message = encode_grid_map(
        frame_id="odom",
        resolution_m=0.2,
        origin_x_m=-1.0,
        origin_y_m=3.0,
        layers=source,
        basic_layers=("elevation", "valid_mask"),
    )

    decoded = decode_layers(message, ("elevation", "valid_mask"))

    assert message.info.resolution == 0.2
    assert message.outer_start_index == 0
    assert message.inner_start_index == 0
    assert message.header.frame_id == "odom"
    assert decoded["elevation"].tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert decoded["valid_mask"].tolist() == [[1.0, 1.0], [1.0, 0.0]]


def test_decode_rejects_missing_or_wrong_shaped_required_layer() -> None:
    message = encode_grid_map(
        frame_id="odom",
        resolution_m=0.2,
        origin_x_m=0.0,
        origin_y_m=0.0,
        layers={"elevation": np.ones((2, 2), dtype=np.float32)},
        basic_layers=("elevation",),
    )

    with pytest.raises(GridMapCodecError, match="GRID_MAP_LAYER_MISSING"):
        decode_layers(message, ("elevation", "roughness"))

    message.data[0].data.pop()
    with pytest.raises(GridMapCodecError, match="GRID_MAP_LAYER_SHAPE_INVALID"):
        decode_layers(message, ("elevation",))

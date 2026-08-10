from __future__ import annotations

import numpy as np
import pytest
from grid_map_msgs.msg import GridMap
from std_msgs.msg import Float32MultiArray, MultiArrayDimension

from lunar_exploration_policy.grid_map_runtime import (
    GridMapRuntimeError,
    decode_grid_map,
)


LAYERS = (
    "elevation",
    "valid_mask",
    "obstacle",
    "obstacle_height",
    "observation_age_s",
    "observation_quality",
    "elevation_variance",
    "obstacle_variance",
    "observation_count",
    "forbidden",
)


def _encode(values, width: int, height: int, outer: int, inner: int):
    layer = Float32MultiArray()
    layer.layout.dim = [
        MultiArrayDimension(
            label="column_index", size=height, stride=width * height
        ),
        MultiArrayDimension(label="row_index", size=width, stride=width),
    ]
    layer.data = [0.0] * (width * height)
    for y in range(height):
        for x in range(width):
            physical_row = (width - 1 - x + outer) % width
            physical_column = (height - 1 - y + inner) % height
            layer.data[physical_column * width + physical_row] = values[
                y * width + x
            ]
    return layer


def _message(outer: int = 1, inner: int = 1) -> GridMap:
    width, height = 3, 2
    message = GridMap()
    message.header.frame_id = "map"
    message.header.stamp.sec = 10
    message.info.resolution = 1.0
    message.info.length_x = 3.0
    message.info.length_y = 2.0
    message.info.pose.position.x = 1.5
    message.info.pose.position.y = 1.0
    message.info.pose.orientation.w = 1.0
    message.outer_start_index = outer
    message.inner_start_index = inner
    message.layers = list(LAYERS)
    for name in LAYERS:
        values = [0.0] * 6
        if name == "elevation":
            values = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        elif name in ("valid_mask", "observation_quality"):
            values = [1.0] * 6
        elif name == "observation_count":
            values = [2.0] * 6
        message.data.append(_encode(values, width, height, outer, inner))
    return message


def test_decoder_matches_cpp_canonical_south_up_unwrap() -> None:
    decoded = decode_grid_map(_message(), expected_frame="map")

    assert decoded.stamp_ns == 10_000_000_000
    assert decoded.origin_xy_m == pytest.approx((0.0, 0.0))
    np.testing.assert_array_equal(
        decoded.layers["elevation"],
        np.asarray([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], np.float32),
    )
    assert decoded.content_id == decode_grid_map(
        _message(0, 0), expected_frame="map"
    ).content_id


@pytest.mark.parametrize(
    "mutate,message",
    [
        (lambda value: setattr(value.header, "frame_id", "odom"), "frame"),
        (lambda value: value.layers.pop(), "layer"),
        (lambda value: setattr(value.info, "resolution", 0.0), "geometry"),
        (
            lambda value: value.data[0].data.__setitem__(0, float("nan")),
            "finite",
        ),
    ],
)
def test_decoder_rejects_ambiguous_grid_map(mutate, message: str) -> None:
    value = _message()
    mutate(value)

    with pytest.raises(GridMapRuntimeError, match=message):
        decode_grid_map(value, expected_frame="map")

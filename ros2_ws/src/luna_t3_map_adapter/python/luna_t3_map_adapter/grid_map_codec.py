"""Deterministic conversion between canonical NumPy layers and ROS GridMap."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from grid_map_msgs.msg import GridMap
from std_msgs.msg import Float32MultiArray, MultiArrayDimension


class GridMapCodecError(ValueError):
    """Raised when a GridMap cannot represent one canonical layer set."""


def _encode_layer(values: np.ndarray) -> Float32MultiArray:
    if values.ndim != 2 or values.shape[0] <= 0 or values.shape[1] <= 0:
        raise GridMapCodecError("GRID_MAP_LAYER_SHAPE_INVALID")
    height, width = values.shape
    message = Float32MultiArray()
    message.layout.dim = [
        MultiArrayDimension(label="column_index", size=height, stride=width * height),
        MultiArrayDimension(label="row_index", size=width, stride=width),
    ]
    # lunar_planner_ros unwraps the GridMap ring buffer into canonical [y, x].
    encoded = np.asarray(values, dtype=np.float32)[::-1, ::-1]
    message.data = encoded.reshape(-1).tolist()
    return message


def _decode_layer(message: Float32MultiArray) -> np.ndarray:
    if len(message.layout.dim) != 2:
        raise GridMapCodecError("GRID_MAP_LAYER_SHAPE_INVALID")
    height = int(message.layout.dim[0].size)
    width = int(message.layout.dim[1].size)
    if height <= 0 or width <= 0 or len(message.data) != height * width:
        raise GridMapCodecError("GRID_MAP_LAYER_SHAPE_INVALID")
    encoded = np.asarray(message.data, dtype=np.float32).reshape((height, width))
    return encoded[::-1, ::-1].copy()


def decode_layers(message: GridMap, required_layers: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Decode required named layers using the established canonical [y, x] order."""

    if message.info.resolution <= 0.0 or len(message.layers) != len(message.data):
        raise GridMapCodecError("GRID_MAP_LAYER_SHAPE_INVALID")
    if len(set(message.layers)) != len(message.layers):
        raise GridMapCodecError("GRID_MAP_LAYER_DUPLICATE")
    index = {name: position for position, name in enumerate(message.layers)}
    decoded: dict[str, np.ndarray] = {}
    expected_shape: tuple[int, int] | None = None
    for name in required_layers:
        position = index.get(name)
        if position is None:
            raise GridMapCodecError("GRID_MAP_LAYER_MISSING")
        values = _decode_layer(message.data[position])
        if expected_shape is None:
            expected_shape = values.shape
        elif values.shape != expected_shape:
            raise GridMapCodecError("GRID_MAP_LAYER_SHAPE_INVALID")
        decoded[name] = values
    return decoded


def encode_grid_map(
    *,
    frame_id: str,
    resolution_m: float,
    origin_x_m: float,
    origin_y_m: float,
    layers: Mapping[str, np.ndarray],
    basic_layers: tuple[str, ...],
) -> GridMap:
    """Encode equally shaped canonical layers as an unrotated GridMap."""

    if not frame_id or resolution_m <= 0.0 or not layers:
        raise GridMapCodecError("GRID_MAP_GEOMETRY_INVALID")
    names = tuple(layers)
    if len(set(names)) != len(names) or any(not name for name in names):
        raise GridMapCodecError("GRID_MAP_LAYER_DUPLICATE")
    if any(name not in layers for name in basic_layers):
        raise GridMapCodecError("GRID_MAP_BASIC_LAYER_INVALID")

    encoded_layers = [_encode_layer(np.asarray(layers[name])) for name in names]
    height = int(np.asarray(layers[names[0]]).shape[0])
    width = int(np.asarray(layers[names[0]]).shape[1])
    if any(np.asarray(layers[name]).shape != (height, width) for name in names):
        raise GridMapCodecError("GRID_MAP_LAYER_SHAPE_INVALID")

    message = GridMap()
    message.header.frame_id = frame_id
    message.info.resolution = float(resolution_m)
    message.info.length_x = width * float(resolution_m)
    message.info.length_y = height * float(resolution_m)
    message.info.pose.position.x = float(origin_x_m) + message.info.length_x * 0.5
    message.info.pose.position.y = float(origin_y_m) + message.info.length_y * 0.5
    message.info.pose.orientation.w = 1.0
    message.layers = list(names)
    message.basic_layers = list(basic_layers)
    message.data = encoded_layers
    message.outer_start_index = 0
    message.inner_start_index = 0
    return message

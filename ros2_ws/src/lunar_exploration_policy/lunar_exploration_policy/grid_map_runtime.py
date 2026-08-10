"""与 C++ GridMapAdapter 同义的严格 Python 解码边界。"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Mapping

import numpy as np


REQUIRED_LAYERS = (
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
_BINARY = {"valid_mask", "obstacle", "forbidden"}
_NONNEGATIVE = {
    "obstacle_height",
    "observation_age_s",
    "elevation_variance",
    "obstacle_variance",
    "observation_count",
}


class GridMapRuntimeError(ValueError):
    """GridMap 无法与规划器的规范栅格形成一一对应。"""


@dataclass(frozen=True, slots=True)
class DecodedGridMap:
    frame_id: str
    stamp_ns: int
    width: int
    height: int
    resolution_m: float
    origin_xy_m: tuple[float, float]
    layers: Mapping[str, np.ndarray]
    content_id: str


def _cells(length: float, resolution: float) -> int:
    if not math.isfinite(length) or not math.isfinite(resolution) or length <= 0.0 or resolution <= 0.0:
        raise GridMapRuntimeError("grid map geometry is invalid")
    value = length / resolution
    rounded = round(value)
    if rounded < 1 or not math.isclose(value, rounded, rel_tol=1e-6, abs_tol=1e-6):
        raise GridMapRuntimeError("grid map geometry is invalid")
    return int(rounded)


def _unwrap(source, width: int, height: int, outer: int, inner: int) -> np.ndarray:
    dimensions = list(source.layout.dim)
    if len(dimensions) != 2:
        raise GridMapRuntimeError("grid map layer layout is invalid")
    first, second = dimensions
    if first.label == "column_index" and second.label == "row_index":
        row_major = False
        if (first.size, second.size) != (height, width):
            raise GridMapRuntimeError("grid map layer layout is invalid")
    elif first.label == "row_index" and second.label == "column_index":
        row_major = True
        if (first.size, second.size) != (width, height):
            raise GridMapRuntimeError("grid map layer layout is invalid")
    else:
        raise GridMapRuntimeError("grid map layer layout is invalid")
    count = width * height
    if first.stride != count or second.stride != second.size:
        raise GridMapRuntimeError("grid map layer layout is invalid")
    offset = int(source.layout.data_offset)
    values = np.asarray(source.data, dtype=np.float32)
    if offset < 0 or values.size - offset != count:
        raise GridMapRuntimeError("grid map layer layout is invalid")
    result = np.empty((height, width), dtype=np.float32)
    for y in range(height):
        for x in range(width):
            physical_row = (width - 1 - x + outer) % width
            physical_column = (height - 1 - y + inner) % height
            index = (
                physical_row * height + physical_column
                if row_major
                else physical_column * width + physical_row
            )
            result[y, x] = values[offset + index]
    return result


def _validate_layer(name: str, values: np.ndarray) -> None:
    if not np.isfinite(values).all():
        raise GridMapRuntimeError(f"grid map layer {name} must be finite")
    if name in _BINARY and not np.isin(values, (0.0, 1.0)).all():
        raise GridMapRuntimeError(f"grid map layer {name} must be binary")
    if name in _NONNEGATIVE and (values < 0.0).any():
        raise GridMapRuntimeError(f"grid map layer {name} must be non-negative")
    if name == "observation_quality" and ((values < 0.0) | (values > 1.0)).any():
        raise GridMapRuntimeError("grid map observation_quality must be in [0,1]")
    if name == "observation_count" and not np.equal(values, np.trunc(values)).all():
        raise GridMapRuntimeError("grid map observation_count must be integral")


def decode_grid_map(message, *, expected_frame: str) -> DecodedGridMap:
    """解环为 C++ 使用的南向起始 row-major 数组。"""
    if not expected_frame or message.header.frame_id != expected_frame:
        raise GridMapRuntimeError("grid map frame mismatch")
    stamp_ns = int(message.header.stamp.sec) * 1_000_000_000 + int(message.header.stamp.nanosec)
    if stamp_ns <= 0:
        raise GridMapRuntimeError("grid map stamp is invalid")
    width = _cells(float(message.info.length_x), float(message.info.resolution))
    height = _cells(float(message.info.length_y), float(message.info.resolution))
    pose = message.info.pose
    values = (
        pose.position.x, pose.position.y, pose.position.z,
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w,
    )
    if not all(math.isfinite(float(value)) for value in values) or abs(abs(pose.orientation.w) - 1.0) > 1e-6 or any(abs(value) > 1e-6 for value in (pose.orientation.x, pose.orientation.y, pose.orientation.z)):
        raise GridMapRuntimeError("grid map geometry is invalid")
    outer, inner = int(message.outer_start_index), int(message.inner_start_index)
    if not 0 <= outer < width or not 0 <= inner < height:
        raise GridMapRuntimeError("grid map start index is invalid")
    if len(message.layers) != len(message.data):
        raise GridMapRuntimeError("grid map layer count mismatch")
    if len(set(message.layers)) != len(message.layers):
        raise GridMapRuntimeError("grid map layer names must be unique")
    indexed = dict(zip(message.layers, message.data, strict=True))
    missing = set(REQUIRED_LAYERS) - indexed.keys()
    if missing:
        raise GridMapRuntimeError(f"grid map layer missing: {sorted(missing)}")
    layers: dict[str, np.ndarray] = {}
    for name in REQUIRED_LAYERS:
        layer = _unwrap(indexed[name], width, height, outer, inner)
        _validate_layer(name, layer)
        layer.setflags(write=False)
        layers[name] = layer
    origin = (
        float(pose.position.x) - float(message.info.length_x) * 0.5,
        float(pose.position.y) - float(message.info.length_y) * 0.5,
    )
    digest = hashlib.sha256()
    digest.update(repr((expected_frame, width, height, float(message.info.resolution), origin)).encode("utf-8"))
    for name in REQUIRED_LAYERS:
        digest.update(name.encode("utf-8"))
        digest.update(layers[name].tobytes(order="C"))
    return DecodedGridMap(
        frame_id=expected_frame,
        stamp_ns=stamp_ns,
        width=width,
        height=height,
        resolution_m=float(message.info.resolution),
        origin_xy_m=origin,
        layers=layers,
        content_id=digest.hexdigest(),
    )


__all__ = ["DecodedGridMap", "GridMapRuntimeError", "decode_grid_map"]

"""Conservative conversion from a frozen Task3 snapshot to planner layers."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np
from grid_map_msgs.msg import GridMap
from std_msgs.msg import Float32MultiArray, MultiArrayDimension

from .global_map_cache import GlobalMapSnapshot
from .t3_sqlite_reader import Task3MapError


_LAYER_NAMES = (
    "elevation", "valid_mask", "obstacle", "obstacle_height",
    "observation_age_s", "observation_quality", "elevation_variance",
    "obstacle_variance", "observation_count", "forbidden",
)


@dataclass(frozen=True)
class CanonicalGlobalGrid:
    frame_id: str
    origin_x_m: float
    origin_y_m: float
    resolution_m: float
    width: int
    height: int
    layers: Mapping[str, np.ndarray]


def _aligned_cell(value: float, origin: float, resolution: float) -> int:
    coordinate = (value - origin) / resolution
    rounded = round(coordinate)
    if not math.isclose(coordinate, rounded, abs_tol=1.0e-7):
        raise Task3MapError("TASK_ROI_ALIGNMENT_INVALID")
    return int(rounded)


def _source_mosaic(snapshot: GlobalMapSnapshot) -> dict[str, np.ndarray]:
    metadata = snapshot.source.metadata
    min_x, min_y, max_x, max_y = snapshot.roi
    start_x = _aligned_cell(min_x, metadata.origin_x_m, metadata.resolution_m)
    start_y = _aligned_cell(min_y, metadata.origin_y_m, metadata.resolution_m)
    end_x = _aligned_cell(max_x, metadata.origin_x_m, metadata.resolution_m)
    end_y = _aligned_cell(max_y, metadata.origin_y_m, metadata.resolution_m)
    width, height = end_x - start_x, end_y - start_y
    if width <= 0 or height <= 0:
        raise Task3MapError("TASK_ROI_INVALID")

    sentinels = {
        "occupancy": -1, "semantic": 0, "semantic_confidence": 0,
        "elevation": -32768, "elevation_variance": 65535,
        "height_range": 65535, "roughness": 65535, "observation_count": 0,
    }
    result: dict[str, np.ndarray] = {}
    for name, sentinel in sentinels.items():
        dtype = next(iter(snapshot.source.tiles)).layers[name].dtype
        result[name] = np.full((height, width), sentinel, dtype=dtype)
    result["elevation_offset_m"] = np.full((height, width), np.nan, dtype=np.float32)

    cells = metadata.tile_cells
    for tile in snapshot.source.tiles:
        tile_start_x = tile.tile_x * cells
        tile_start_y = tile.tile_y * cells
        source_x0, source_y0 = max(start_x, tile_start_x), max(start_y, tile_start_y)
        source_x1, source_y1 = min(end_x, tile_start_x + cells), min(end_y, tile_start_y + cells)
        if source_x0 >= source_x1 or source_y0 >= source_y1:
            continue
        dest = (slice(source_y0 - start_y, source_y1 - start_y), slice(source_x0 - start_x, source_x1 - start_x))
        src = (slice(source_y0 - tile_start_y, source_y1 - tile_start_y), slice(source_x0 - tile_start_x, source_x1 - tile_start_x))
        for name, values in result.items():
            if name == "elevation_offset_m":
                values[dest] = tile.elevation_offset_m
                continue
            values[dest] = tile.layers[name][src]
    return result


def to_canonical_global_grid(snapshot: GlobalMapSnapshot) -> CanonicalGlobalGrid:
    """Aggregate exact L0 evidence; incomplete blocks are invalid and forbidden."""

    source = _source_mosaic(snapshot)
    factor = round(snapshot.selected_level.resolution_m / 0.2)
    if factor < 1 or not math.isclose(factor * 0.2, snapshot.selected_level.resolution_m):
        raise Task3MapError("GLOBAL_MAP_LEVEL_INVALID")
    height, width = snapshot.selected_level.height, snapshot.selected_level.width
    layers = {name: np.zeros((height, width), dtype=np.float32) for name in _LAYER_NAMES}

    valid = (
        (source["occupancy"] >= 0)
        & (source["elevation"] != -32768)
        & (source["elevation_variance"] != 65535)
        & (source["height_range"] != 65535)
        & (source["roughness"] != 65535)
        & np.isfinite(source["elevation_offset_m"])
    )
    obstacle = (source["occupancy"] >= 50) | (source["semantic"] >= 3)
    elevation_m = (
        source["elevation"].astype(np.float32) * 0.01
        + source["elevation_offset_m"]
    )
    elevation_variance = source["elevation_variance"].astype(np.float32) * 0.0001
    obstacle_height = source["height_range"].astype(np.float32) * 0.001
    obstacle_variance = source["roughness"].astype(np.float32) * 0.001
    quality = source["semantic_confidence"].astype(np.float32) / 255.0

    for y in range(height):
        for x in range(width):
            y0, x0 = y * factor, x * factor
            block_valid = valid[y0:y0 + factor, x0:x0 + factor]
            complete = block_valid.shape == (factor, factor)
            parent_valid = complete and bool(np.all(block_valid))
            if not parent_valid:
                layers["forbidden"][y, x] = 1.0
                continue
            sample = (slice(y0, y0 + factor), slice(x0, x0 + factor))
            elevations = elevation_m[sample]
            layers["elevation"][y, x] = float(np.mean(elevations))
            layers["valid_mask"][y, x] = 1.0
            layers["obstacle"][y, x] = float(np.any(obstacle[sample]))
            layers["obstacle_height"][y, x] = float(np.max(obstacle_height[sample]))
            layers["observation_age_s"][y, x] = 86400.0
            layers["observation_quality"][y, x] = float(np.min(quality[sample]))
            layers["elevation_variance"][y, x] = float(
                np.max(elevation_variance[sample]) + np.var(elevations)
            )
            layers["obstacle_variance"][y, x] = float(np.max(obstacle_variance[sample]))
            layers["observation_count"][y, x] = float(np.min(source["observation_count"][sample]))

    min_x, min_y, _, _ = snapshot.roi
    return CanonicalGlobalGrid(
        frame_id="map", origin_x_m=min_x, origin_y_m=min_y,
        resolution_m=snapshot.selected_level.resolution_m,
        width=width, height=height, layers=layers,
    )


def _encode_layer(values: np.ndarray) -> Float32MultiArray:
    height, width = values.shape
    message = Float32MultiArray()
    message.layout.dim = [
        MultiArrayDimension(label="column_index", size=height, stride=width * height),
        MultiArrayDimension(label="row_index", size=width, stride=width),
    ]
    # grid_map stores its default column-major ring buffer inverted relative to
    # canonical [y, x] values.  This is the inverse of lunar_planner_ros::Unwrap.
    encoded = np.empty((height, width), dtype=np.float32)
    encoded[:, :] = values[::-1, ::-1]
    message.data = encoded.reshape(-1).tolist()
    return message


def to_grid_map_message(grid: CanonicalGlobalGrid) -> GridMap:
    """Encode a canonical grid using the layout accepted by lunar_planner_ros."""

    message = GridMap()
    message.header.frame_id = grid.frame_id
    message.info.resolution = grid.resolution_m
    message.info.length_x = grid.width * grid.resolution_m
    message.info.length_y = grid.height * grid.resolution_m
    message.info.pose.position.x = grid.origin_x_m + message.info.length_x * 0.5
    message.info.pose.position.y = grid.origin_y_m + message.info.length_y * 0.5
    message.info.pose.orientation.w = 1.0
    message.layers = list(_LAYER_NAMES)
    message.basic_layers = ["elevation", "valid_mask"]
    message.data = [_encode_layer(grid.layers[name]) for name in _LAYER_NAMES]
    message.outer_start_index = 0
    message.inner_start_index = 0
    return message

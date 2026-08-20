"""Map direct Task3 odom-local cells to read-only Task3 L0 evidence."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .local_grid_conversion import LocalSourceGrid, TileEvidence
from .t3_sqlite_reader import Task3MapRead


@dataclass(frozen=True)
class PlanarTransform:
    """A map-from-odom transform used for evidence indexing only."""

    translation_x_m: float
    translation_y_m: float
    yaw_rad: float

    @classmethod
    def identity(cls) -> "PlanarTransform":
        return cls(0.0, 0.0, 0.0)

    def apply(self, x_odom_m: float, y_odom_m: float) -> tuple[float, float]:
        cosine, sine = math.cos(self.yaw_rad), math.sin(self.yaw_rad)
        return (
            self.translation_x_m + cosine * x_odom_m - sine * y_odom_m,
            self.translation_y_m + sine * x_odom_m + cosine * y_odom_m,
        )


def tile_evidence_for_local_grid(
    source: LocalSourceGrid,
    snapshot: Task3MapRead,
    map_from_odom: PlanarTransform,
) -> TileEvidence:
    """Return exact available L0 fields, leaving unprovable cells missing."""

    metadata = snapshot.metadata
    if metadata.frame_id != "map" or not np.isclose(metadata.resolution_m, source.resolution_m):
        raise ValueError("LOCAL_MAP_EVIDENCE_GEOMETRY_INVALID")
    height, width = source.occupancy.shape
    shape = (height, width)
    height_range = np.full(shape, np.nan, dtype=np.float32)
    elevation_variance = np.full(shape, np.nan, dtype=np.float32)
    roughness = np.full(shape, np.nan, dtype=np.float32)
    observation_count = np.full(shape, -1, dtype=np.int32)
    semantic_confidence = np.full(shape, np.nan, dtype=np.float32)
    tiles = {(tile.tile_x, tile.tile_y): tile for tile in snapshot.tiles}
    cells = metadata.tile_cells

    for y in range(height):
        y_odom = source.origin_y_m + (y + 0.5) * source.resolution_m
        for x in range(width):
            x_odom = source.origin_x_m + (x + 0.5) * source.resolution_m
            x_map, y_map = map_from_odom.apply(x_odom, y_odom)
            cell_x = math.floor((x_map - metadata.origin_x_m) / metadata.resolution_m)
            cell_y = math.floor((y_map - metadata.origin_y_m) / metadata.resolution_m)
            tile_x, local_x = divmod(cell_x, cells)
            tile_y, local_y = divmod(cell_y, cells)
            tile = tiles.get((tile_x, tile_y))
            if tile is None:
                continue
            raw_height = tile.layers["height_range"][local_y, local_x]
            raw_elevation_variance = tile.layers["elevation_variance"][local_y, local_x]
            raw_roughness = tile.layers["roughness"][local_y, local_x]
            raw_count = tile.layers["observation_count"][local_y, local_x]
            raw_confidence = tile.layers["semantic_confidence"][local_y, local_x]
            if (
                raw_height == 65535
                or raw_elevation_variance == 65535
                or raw_roughness == 65535
            ):
                continue
            height_range[y, x] = float(raw_height) * 0.001
            elevation_variance[y, x] = float(raw_elevation_variance) * 0.0001
            roughness[y, x] = float(raw_roughness) * 0.001
            observation_count[y, x] = int(raw_count)
            semantic_confidence[y, x] = float(raw_confidence) / 255.0

    return TileEvidence(
        height_range=height_range,
        elevation_variance=elevation_variance,
        roughness=roughness,
        observation_count=observation_count,
        semantic_confidence=semantic_confidence,
    )

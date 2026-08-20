"""Read Task3 compact global-map tiles without modifying their SQLite store."""

from __future__ import annotations

from dataclasses import dataclass
import io
import json
import math
from pathlib import Path
import sqlite3
from typing import Mapping
import zlib
import zipfile

import numpy as np


_REQUIRED_LAYERS: Mapping[str, np.dtype] = {
    "occupancy": np.dtype(np.int8),
    "semantic": np.dtype(np.uint8),
    "semantic_confidence": np.dtype(np.uint8),
    "elevation": np.dtype(np.int16),
    "elevation_variance": np.dtype(np.uint16),
    "height_range": np.dtype(np.uint16),
    "roughness": np.dtype(np.uint16),
    "observation_count": np.dtype(np.uint16),
}


class Task3MapError(RuntimeError):
    """A fail-closed Task3 source error with a stable reason code."""


@dataclass(frozen=True)
class Task3Metadata:
    map_revision: int
    frame_id: str
    resolution_m: float
    tile_cells: int
    tile_length_m: float
    origin_x_m: float
    origin_y_m: float


@dataclass(frozen=True)
class Task3Tile:
    tile_x: int
    tile_y: int
    revision: int
    elevation_offset_m: float
    layers: Mapping[str, np.ndarray]


@dataclass(frozen=True)
class Task3MapRead:
    metadata: Task3Metadata
    tiles: tuple[Task3Tile, ...]


def _raise(code: str) -> None:
    raise Task3MapError(code)


def _metadata(connection: sqlite3.Connection) -> Task3Metadata:
    try:
        raw = dict(connection.execute("SELECT key, value FROM metadata"))
        if json.loads(raw["format"]) != "t3_compact_global_grid_map":
            _raise("TASK3_MAP_FORMAT_INVALID")
        if int(json.loads(raw["format_version"])) != 1:
            _raise("TASK3_MAP_FORMAT_INVALID")
        frame_id = str(json.loads(raw["frame_id"]))
        resolution_m = float(json.loads(raw["resolution"]))
        tile_cells = int(json.loads(raw["tile_cells"]))
        tile_length_m = float(json.loads(raw["tile_length"]))
        origin_x_m = float(json.loads(raw["origin_x"]))
        origin_y_m = float(json.loads(raw["origin_y"]))
        map_revision = int(json.loads(raw["map_revision"]))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        _raise("TASK3_MAP_METADATA_INVALID")
    if (
        frame_id != "map"
        or not math.isfinite(resolution_m)
        or resolution_m != 0.2
        or tile_cells != 256
        or not math.isfinite(tile_length_m)
        or tile_length_m != resolution_m * tile_cells
        or not math.isfinite(origin_x_m)
        or not math.isfinite(origin_y_m)
        or map_revision < 0
    ):
        _raise("TASK3_MAP_METADATA_INVALID")
    return Task3Metadata(
        map_revision=map_revision,
        frame_id=frame_id,
        resolution_m=resolution_m,
        tile_cells=tile_cells,
        tile_length_m=tile_length_m,
        origin_x_m=origin_x_m,
        origin_y_m=origin_y_m,
    )


def _decode_payload(payload: bytes) -> Mapping[str, np.ndarray]:
    try:
        compressed = zlib.decompress(payload)
        with np.load(io.BytesIO(compressed), allow_pickle=False) as archive:
            names = set(archive.files)
            if names != set(_REQUIRED_LAYERS):
                _raise("TASK3_TILE_SCHEMA_INVALID")
            layers = {name: archive[name].copy() for name in _REQUIRED_LAYERS}
    except (OSError, ValueError, zipfile.BadZipFile):
        _raise("TASK3_TILE_PAYLOAD_INVALID")
    for name, expected_dtype in _REQUIRED_LAYERS.items():
        layer = layers[name]
        if layer.dtype != expected_dtype or layer.shape != (256, 256):
            _raise("TASK3_TILE_SCHEMA_INVALID")
    return layers


def _tile_bounds(
    roi: tuple[float, float, float, float], metadata: Task3Metadata
) -> tuple[int, int, int, int]:
    min_x, min_y, max_x, max_y = roi
    if not all(math.isfinite(value) for value in roi) or max_x <= min_x or max_y <= min_y:
        _raise("TASK_ROI_INVALID")
    length = metadata.tile_length_m
    return (
        math.floor((min_x - metadata.origin_x_m) / length),
        math.floor((math.nextafter(max_x, -math.inf) - metadata.origin_x_m) / length),
        math.floor((min_y - metadata.origin_y_m) / length),
        math.floor((math.nextafter(max_y, -math.inf) - metadata.origin_y_m) / length),
    )


def read_roi_at_revision(
    database_path: Path | str,
    roi: tuple[float, float, float, float],
    required_revision: int,
) -> Task3MapRead:
    """Return every tile intersecting ``roi`` from one immutable SQLite revision."""

    if required_revision < 0:
        _raise("TASK3_MAP_REVISION_MISMATCH")
    path = Path(database_path)
    try:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except sqlite3.Error:
        _raise("TASK3_SQLITE_OPEN_FAILED")
    try:
        connection.execute("PRAGMA query_only=ON")
        connection.execute("BEGIN")
        metadata = _metadata(connection)
        if metadata.map_revision != required_revision:
            _raise("TASK3_MAP_REVISION_MISMATCH")
        min_tile_x, max_tile_x, min_tile_y, max_tile_y = _tile_bounds(roi, metadata)
        rows = list(
            connection.execute(
                "SELECT tile_x, tile_y, revision, elevation_offset_m, payload "
                "FROM tiles WHERE tile_x BETWEEN ? AND ? AND tile_y BETWEEN ? AND ? "
                "ORDER BY tile_y, tile_x",
                (min_tile_x, max_tile_x, min_tile_y, max_tile_y),
            )
        )
        expected = (max_tile_x - min_tile_x + 1) * (max_tile_y - min_tile_y + 1)
        if len(rows) != expected:
            _raise("TASK3_TILE_MISSING")
        tiles = tuple(
            Task3Tile(
                tile_x=int(tile_x),
                tile_y=int(tile_y),
                revision=int(revision),
                elevation_offset_m=float(elevation_offset_m),
                layers=_decode_payload(payload),
            )
            for tile_x, tile_y, revision, elevation_offset_m, payload in rows
        )
        connection.commit()
        return Task3MapRead(metadata=metadata, tiles=tiles)
    except Task3MapError:
        connection.rollback()
        raise
    except sqlite3.Error:
        connection.rollback()
        _raise("TASK3_SQLITE_READ_FAILED")
    finally:
        connection.close()

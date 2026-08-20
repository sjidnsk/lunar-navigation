import io
import sqlite3
import zlib

import numpy as np
import pytest

from luna_t3_map_adapter.t3_sqlite_reader import (
    Task3MapError,
    read_roi_at_revision,
)


def _payload(occupancy: np.ndarray) -> bytes:
    arrays = {
        "occupancy": occupancy.astype(np.int8),
        "semantic": np.zeros((256, 256), dtype=np.uint8),
        "semantic_confidence": np.zeros((256, 256), dtype=np.uint8),
        "elevation": np.full((256, 256), -32768, dtype=np.int16),
        "elevation_variance": np.full((256, 256), 65535, dtype=np.uint16),
        "height_range": np.full((256, 256), 65535, dtype=np.uint16),
        "roughness": np.full((256, 256), 65535, dtype=np.uint16),
        "observation_count": np.zeros((256, 256), dtype=np.uint16),
    }
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return zlib.compress(buffer.getvalue())


def _database(path, revision: int = 7) -> None:
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE tiles (
          tile_x INTEGER NOT NULL,
          tile_y INTEGER NOT NULL,
          revision INTEGER NOT NULL,
          elevation_offset_m REAL NOT NULL,
          payload BLOB NOT NULL,
          PRIMARY KEY(tile_x, tile_y)
        );
        """
    )
    con.executemany(
        "INSERT INTO metadata(key, value) VALUES (?, ?)",
        [
            ("format", '"t3_compact_global_grid_map"'),
            ("format_version", "1"),
            ("frame_id", '"map"'),
            ("map_revision", str(revision)),
            ("origin_x", "0.0"),
            ("origin_y", "0.0"),
            ("resolution", "0.2"),
            ("tile_cells", "256"),
            ("tile_length", "51.2"),
        ],
    )
    unknown = np.full((256, 256), -1, dtype=np.int8)
    con.execute(
        "INSERT INTO tiles VALUES (?, ?, ?, ?, ?)",
        (0, 0, revision, 0.0, _payload(unknown)),
    )
    con.commit()
    con.close()


def test_reads_exact_revision_and_decodes_all_documented_layers(tmp_path):
    database = tmp_path / "global_grid_map.sqlite3"
    _database(database)

    result = read_roi_at_revision(database, (0.0, 0.0, 51.2, 51.2), 7)

    assert result.metadata.map_revision == 7
    assert result.metadata.frame_id == "map"
    assert result.metadata.resolution_m == 0.2
    assert [(tile.tile_x, tile.tile_y) for tile in result.tiles] == [(0, 0)]
    assert set(result.tiles[0].layers) == {
        "occupancy", "semantic", "semantic_confidence", "elevation",
        "elevation_variance", "height_range", "roughness", "observation_count",
    }
    assert result.tiles[0].layers["occupancy"].dtype == np.int8
    assert result.tiles[0].layers["occupancy"].shape == (256, 256)


def test_rejects_revision_mismatch_and_missing_roi_tile(tmp_path):
    database = tmp_path / "global_grid_map.sqlite3"
    _database(database)

    with pytest.raises(Task3MapError, match="TASK3_MAP_REVISION_MISMATCH"):
        read_roi_at_revision(database, (0.0, 0.0, 51.2, 51.2), 8)
    with pytest.raises(Task3MapError, match="TASK3_TILE_MISSING"):
        read_roi_at_revision(database, (51.2, 0.0, 102.4, 51.2), 7)

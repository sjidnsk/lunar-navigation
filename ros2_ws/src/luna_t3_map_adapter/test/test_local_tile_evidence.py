from __future__ import annotations

import numpy as np

from luna_t3_map_adapter.local_grid_conversion import LocalSourceGrid
from luna_t3_map_adapter.local_tile_evidence import PlanarTransform, tile_evidence_for_local_grid
from luna_t3_map_adapter.t3_sqlite_reader import Task3MapRead, Task3Metadata, Task3Tile


def _source() -> LocalSourceGrid:
    return LocalSourceGrid(
        frame_id="odom", stamp_ns=1, origin_x_m=0.0, origin_y_m=0.0,
        resolution_m=0.2,
        occupancy=np.zeros((2, 2), dtype=np.int8),
        semantic_id=np.zeros((2, 2), dtype=np.uint8),
        elevation=np.zeros((2, 2), dtype=np.float32),
        roughness=np.zeros((2, 2), dtype=np.float32),
    )


def _read() -> Task3MapRead:
    layers = {
        "occupancy": np.zeros((256, 256), dtype=np.int8),
        "semantic": np.zeros((256, 256), dtype=np.uint8),
        "semantic_confidence": np.full((256, 256), 200, dtype=np.uint8),
        "elevation": np.zeros((256, 256), dtype=np.int16),
        "elevation_variance": np.full((256, 256), 400, dtype=np.uint16),
        "height_range": np.full((256, 256), 150, dtype=np.uint16),
        "roughness": np.full((256, 256), 20, dtype=np.uint16),
        "observation_count": np.full((256, 256), 7, dtype=np.uint16),
    }
    layers["height_range"][0, 1] = 250
    return Task3MapRead(
        Task3Metadata(4, "map", 0.2, 256, 51.2, 0.0, 0.0),
        (Task3Tile(0, 0, 4, 0.0, layers),),
    )


def test_identity_map_from_odom_reads_exact_l0_evidence_at_local_cell_centers() -> None:
    evidence = tile_evidence_for_local_grid(_source(), _read(), PlanarTransform.identity())

    np.testing.assert_allclose(evidence.height_range, [[0.15, 0.25], [0.15, 0.15]])
    np.testing.assert_allclose(evidence.elevation_variance, [[0.04, 0.04], [0.04, 0.04]])
    np.testing.assert_allclose(evidence.roughness, [[0.02, 0.02], [0.02, 0.02]])
    assert evidence.observation_count.tolist() == [[7, 7], [7, 7]]
    np.testing.assert_allclose(
        evidence.semantic_confidence,
        [[200 / 255, 200 / 255], [200 / 255, 200 / 255]],
    )


def test_cells_without_a_corresponding_l0_tile_remain_missing() -> None:
    source = _source()
    source = LocalSourceGrid(**{**source.__dict__, "origin_x_m": 100.0})

    evidence = tile_evidence_for_local_grid(source, _read(), PlanarTransform.identity())

    assert np.isnan(evidence.height_range).all()
    assert (evidence.observation_count == -1).all()

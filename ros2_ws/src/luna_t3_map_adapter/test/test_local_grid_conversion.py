from __future__ import annotations

import numpy as np

from luna_t3_map_adapter.local_grid_conversion import (
    CanonicalLocalGrid,
    LocalMapPolicy,
    LocalSourceGrid,
    ObservationLedger,
    TileEvidence,
    convert_local_grid,
    local_source_from_grid,
)
from luna_t3_map_adapter.grid_map_codec import encode_grid_map


def _source() -> LocalSourceGrid:
    return LocalSourceGrid(
        frame_id="odom",
        stamp_ns=1_000_000_000,
        origin_x_m=-30.0,
        origin_y_m=-30.0,
        resolution_m=0.2,
        occupancy=np.array([[0, 0], [0, 80]], dtype=np.int8),
        semantic_id=np.array([[0, 4], [0, 0]], dtype=np.uint8),
        elevation=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        roughness=np.full((2, 2), 0.05, dtype=np.float32),
    )


def _evidence() -> TileEvidence:
    return TileEvidence(
        height_range=np.full((2, 2), 0.1, dtype=np.float32),
        elevation_variance=np.full((2, 2), 0.04, dtype=np.float32),
        roughness=np.full((2, 2), 0.02, dtype=np.float32),
        observation_count=np.array([[5, -1], [-1, -1]], dtype=np.int32),
        semantic_confidence=np.full((2, 2), 0.8, dtype=np.float32),
    )


def _policy() -> LocalMapPolicy:
    return LocalMapPolicy(
        occupancy_obstacle_threshold=50,
        semantic_obstacle_ids=frozenset({3}),
        semantic_forbidden_ids=frozenset({4}),
        max_age_s=30.0,
    )


def test_local_conversion_preserves_02m_grid_shape_and_source_elevation() -> None:
    source = _source()
    output = convert_local_grid(source, _evidence(), _policy(), ObservationLedger())

    assert isinstance(output, CanonicalLocalGrid)
    assert output.frame_id == "odom"
    assert output.resolution_m == 0.2
    assert output.layers["elevation"].shape == (2, 2)
    assert output.layers["elevation"].tolist() == source.elevation.tolist()
    assert set(output.layers) == {
        "elevation", "valid_mask", "obstacle", "obstacle_height",
        "observation_age_s", "observation_quality", "elevation_variance",
        "obstacle_variance", "observation_count", "forbidden",
    }


def test_missing_height_evidence_is_invalid_and_forbidden() -> None:
    evidence = _evidence()
    evidence = TileEvidence(
        height_range=np.array([[0.1, np.nan], [0.1, 0.1]], dtype=np.float32),
        elevation_variance=evidence.elevation_variance,
        roughness=evidence.roughness,
        observation_count=evidence.observation_count,
        semantic_confidence=evidence.semantic_confidence,
    )

    output = convert_local_grid(_source(), evidence, _policy(), ObservationLedger())

    assert output.layers["valid_mask"][0, 1] == 0.0
    assert output.layers["forbidden"][0, 1] == 1.0
    assert output.layers["obstacle"][0, 1] == 1.0


def test_semantic_and_occupancy_safety_mapping_is_conservative() -> None:
    output = convert_local_grid(_source(), _evidence(), _policy(), ObservationLedger())

    assert output.layers["forbidden"][0, 1] == 1.0
    assert output.layers["obstacle"][0, 1] == 0.0
    assert output.layers["obstacle"][1, 1] == 1.0


def test_same_stamp_does_not_increment_local_observation_count_twice() -> None:
    ledger = ObservationLedger()
    source = _source()

    first = convert_local_grid(source, _evidence(), _policy(), ledger)
    second = convert_local_grid(source, _evidence(), _policy(), ledger)

    assert first.layers["observation_count"].tolist() == [[5.0, 1.0], [1.0, 1.0]]
    assert second.layers["observation_count"].tolist() == first.layers["observation_count"].tolist()


def test_grid_map_source_parser_preserves_odom_geometry_and_layers() -> None:
    message = encode_grid_map(
        frame_id="odom",
        resolution_m=0.2,
        origin_x_m=-30.0,
        origin_y_m=-30.0,
        layers={
            "occupancy": np.array([[0, 1]], dtype=np.float32),
            "semantic_id": np.array([[0, 4]], dtype=np.float32),
            "elevation": np.array([[1.0, 2.0]], dtype=np.float32),
            "roughness": np.array([[0.1, 0.2]], dtype=np.float32),
        },
        basic_layers=("elevation",),
    )
    message.header.stamp.sec = 4

    source = local_source_from_grid(message)

    assert source.frame_id == "odom"
    assert source.stamp_ns == 4_000_000_000
    assert source.origin_x_m == -30.0
    assert source.origin_y_m == -30.0
    assert source.occupancy.tolist() == [[0, 1]]


def test_grid_map_source_parser_rejects_non_odom_or_zero_timestamp() -> None:
    message = encode_grid_map(
        frame_id="map",
        resolution_m=0.2,
        origin_x_m=0.0,
        origin_y_m=0.0,
        layers={name: np.zeros((1, 1), dtype=np.float32) for name in (
            "occupancy", "semantic_id", "elevation", "roughness"
        )},
        basic_layers=("elevation",),
    )

    with np.testing.assert_raises_regex(ValueError, "LOCAL_MAP_FRAME_INVALID"):
        local_source_from_grid(message)

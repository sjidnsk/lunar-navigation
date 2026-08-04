from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "model_contract"))

from lunar_model_contract import ObservationContractV2, validate_observation_inputs  # noqa: E402
from lunar_policy_training.environment.candidate_builder import CandidateBatch  # noqa: E402
from lunar_policy_training.environment.observation_builder import (  # noqa: E402
    MissionRaster,
    ObservationBuilderV2,
    ObservedWorld,
    PlatformProjection,
    Pose2,
    TransformUnavailable,
    resolve_map_pose,
)
from lunar_policy_training.polar_data.raster import GLOBAL_GEOMETRY, LOCAL_GEOMETRY, WorldTruth  # noqa: E402


def _observed_world() -> ObservedWorld:
    elevation = np.full((256, 256), 9.0, dtype=np.float32)
    observed = np.zeros((256, 256), dtype=bool)
    observed[100:156, 100:156] = True
    return ObservedWorld(
        elevation_m=elevation,
        observed_mask=observed,
        physical_obstacle_ratio=np.full((256, 256), 0.25, dtype=np.float32),
        local_elevation_m=np.full((32, 32), 11.0, dtype=np.float32),
        local_observed_mask=np.ones((32, 32), dtype=bool),
        local_physical_obstacle_ratio=np.full((32, 32), 0.5, dtype=np.float32),
    )


def _test_only_projection() -> PlatformProjection:
    return PlatformProjection(
        traversable_ratio=np.full((256, 256), 0.75, dtype=np.float32),
        local_traversable_ratio=np.full((32, 32), 0.6, dtype=np.float32),
        clearance_margin_norm=np.full((256, 256), 0.4, dtype=np.float32),
        source="test_only/proxy",
    )


def test_builder_emits_complete_contract_in_contract_field_order_without_truth_leakage() -> None:
    mission = MissionRaster(
        priority=np.ones((256, 256), dtype=np.float32),
        roi_ratio=np.ones((256, 256), dtype=np.float32),
        remaining_decision_budget_ratio=0.5,
    )
    candidates = CandidateBatch.empty()
    observation = ObservationBuilderV2().build(
        _observed_world(), mission, Pose2(512.0, 512.0, yaw_rad=np.pi / 2.0),
        _test_only_projection(), candidates, "WHEELED",
    )

    assert tuple(observation) == ObservationContractV2.input_names
    assert observation["prior_channels"].shape == (1, 4, 256, 256)
    assert observation["local_crop"].shape == (1, 4, 32, 32)
    assert observation["prior_channels"][0, 0, 0, 0] == 0.0
    assert observation["prior_channels"][0, 0, 110, 110] == 9.0
    assert observation["local_crop"][0, 0, 0, 0] == 11.0
    assert observation["platform_context"].tolist() == [[1.0, 0.0, 0.0]]
    validate_observation_inputs(observation)


def test_world_rejects_nodata_marked_observed_and_projection_must_be_narrow_proxy() -> None:
    with pytest.raises(ValueError, match="NoData"):
        ObservedWorld(
            elevation_m=np.full((256, 256), np.nan, dtype=np.float32),
            observed_mask=np.ones((256, 256), dtype=bool),
            physical_obstacle_ratio=np.zeros((256, 256), dtype=np.float32),
        )
    with pytest.raises(ValueError, match="test_only/proxy"):
        PlatformProjection(
            traversable_ratio=np.ones((256, 256), dtype=np.float32),
            local_traversable_ratio=np.ones((32, 32), dtype=np.float32),
            clearance_margin_norm=np.ones((256, 256), dtype=np.float32),
            source="recomputed-from-dem",
        )


def test_builder_rejects_missing_map_to_odom_transform() -> None:
    with pytest.raises(TransformUnavailable, match="map/odom"):
        resolve_map_pose(Pose2(1.0, 2.0, frame_id="odom"), None)
    pose = resolve_map_pose(
        Pose2(1.0, 2.0, frame_id="odom"), Pose2(10.0, 20.0, yaw_rad=np.pi / 2.0, frame_id="map")
    )
    np.testing.assert_allclose((pose.x_m, pose.y_m), (8.0, 21.0))
    assert pose.frame_id == "map"


def test_builder_rejects_full_dem_truth_to_keep_network_input_observed_only() -> None:
    truth = WorldTruth(window_sha256="a" * 64, elevation_m=np.zeros((256, 256), dtype=np.float32))
    with pytest.raises(ValueError, match="ObservedWorld"):
        ObservationBuilderV2().build(
            truth, _mission := MissionRaster(np.ones((256, 256), dtype=np.float32), np.ones((256, 256), dtype=np.float32), 1.0),
            Pose2(512.0, 512.0), _test_only_projection(), CandidateBatch.empty(), "WHEELED",
        )


def test_builder_requires_true_local_observations_and_projection_normalizes_clearance_at_boundary() -> None:
    world = ObservedWorld(
        elevation_m=np.zeros((256, 256), dtype=np.float32), observed_mask=np.ones((256, 256), dtype=bool),
        physical_obstacle_ratio=np.zeros((256, 256), dtype=np.float32),
    )
    with pytest.raises(ValueError, match="local"):
        ObservationBuilderV2().build(world, MissionRaster(np.ones((256, 256), dtype=np.float32), np.ones((256, 256), dtype=np.float32), 1.0), Pose2(512.0, 512.0), _test_only_projection(), CandidateBatch.empty(), "WHEELED")
    with pytest.raises(ValueError, match="clearance"):
        PlatformProjection(
            traversable_ratio=np.ones((256, 256), dtype=np.float32),
            local_traversable_ratio=np.ones((32, 32), dtype=np.float32),
            clearance_margin_norm=np.full((256, 256), 1.1, dtype=np.float32), source="test_only/proxy",
        )

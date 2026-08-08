from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "model_contract"))

from lunar_model_contract import ObservationContractV3, validate_observation_inputs  # noqa: E402
from lunar_policy_training.environment.candidate_builder import CandidateBatch  # noqa: E402
from lunar_policy_training.environment.observation_builder import (  # noqa: E402
    MissionRaster,
    LocalObservation,
    ElevationReference,
    ObservationBuilderV2,
    ObservedWorld,
    PlatformProjection,
    Pose2,
    TransformUnavailable,
    resolve_map_pose,
)
from lunar_policy_training.environment.sensor_observation import TrainingWorldTruth  # noqa: E402
from lunar_policy_training.polar_data.hazards import CanvasRatioLayer  # noqa: E402
from lunar_policy_training.polar_data.raster import GLOBAL_GEOMETRY, LOCAL_GEOMETRY, MapCanvas, WorldTruth  # noqa: E402


def _canvas() -> MapCanvas:
    return MapCanvas.from_roi_bounds("a" * 64, (500.0, 500.0, 524.0, 524.0))


def _obstacle_layer(canvas: MapCanvas, value: float = 0.0) -> CanvasRatioLayer:
    return CanvasRatioLayer(canvas, np.full((256, 256), value, dtype=np.float32))


def _observed_world() -> ObservedWorld:
    elevation = np.full((256, 256), 9.0, dtype=np.float32)
    observed = np.zeros((256, 256), dtype=bool)
    observed[100:156, 100:156] = True
    canvas = _canvas()
    local = LocalObservation(canvas.identity, (508.8, 508.8, 515.2, 515.2), np.full((32, 32), 11.0, dtype=np.float32), np.ones((32, 32), dtype=bool), np.full((32, 32), 0.5, dtype=np.float32))
    return ObservedWorld(canvas, elevation, observed, _obstacle_layer(canvas, 0.25), local)


def _test_only_projection() -> PlatformProjection:
    return PlatformProjection(
        _canvas(),
        traversable_ratio=np.full((256, 256), 0.75, dtype=np.float32),
        local_traversable_ratio=np.full((32, 32), 0.6, dtype=np.float32),
        clearance_margin_norm=np.full((256, 256), 0.4, dtype=np.float32),
        source="test_only/proxy",
    )


def test_builder_emits_complete_contract_in_contract_field_order_without_truth_leakage() -> None:
    mission = MissionRaster(
        _canvas(),
        priority=np.ones((256, 256), dtype=np.float32),
        roi_ratio=np.ones((256, 256), dtype=np.float32),
    )
    candidates = CandidateBatch.empty()
    observation = ObservationBuilderV2().build(
        _observed_world(), mission, Pose2(512.0, 512.0, yaw_rad=np.pi / 2.0),
        _test_only_projection(), candidates, "WHEELED",
    )

    assert tuple(observation) == ObservationContractV3.input_names
    assert observation["prior_channels"].shape == (1, 4, 256, 256)
    assert observation["local_crop"].shape == (1, 4, 32, 32)
    assert observation["prior_channels"][0, 0, 0, 0] == 0.0
    assert observation["prior_channels"][0, 0, 110, 110] == 9.0
    assert observation["local_crop"][0, 0, 0, 0] == 11.0
    assert observation["pose_features"].shape == (1, 5)
    assert observation["platform_context"].tolist() == [[1.0, 0.0, 0.0]]
    validate_observation_inputs(observation)


def test_world_rejects_nodata_marked_observed_and_projection_must_be_narrow_proxy() -> None:
    with pytest.raises(ValueError, match="NoData"):
        ObservedWorld(
            _canvas(),
            elevation_m=np.full((256, 256), np.nan, dtype=np.float32),
            observed_mask=np.ones((256, 256), dtype=bool),
            physical_obstacle_layer=_obstacle_layer(_canvas()),
            local=LocalObservation(_canvas().identity, (508.8, 508.8, 515.2, 515.2), np.zeros((32, 32), np.float32), np.ones((32, 32), bool), np.zeros((32, 32), np.float32)),
        )
    with pytest.raises(ValueError, match="test_only/proxy"):
        PlatformProjection(
            _canvas(),
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
            truth, _mission := MissionRaster(_canvas(), np.ones((256, 256), dtype=np.float32), np.ones((256, 256), dtype=np.float32)),
            Pose2(512.0, 512.0), _test_only_projection(), CandidateBatch.empty(), "WHEELED",
        )

    training_truth = TrainingWorldTruth(
        _canvas(),
        np.zeros((256, 256), dtype=np.float32),
        np.zeros((256, 256), dtype=np.float32),
    )
    with pytest.raises(ValueError, match="ObservedWorld"):
        ObservationBuilderV2().build(
            training_truth,
            _mission,
            Pose2(512.0, 512.0),
            _test_only_projection(),
            CandidateBatch.empty(),
            "WHEELED",
        )


def test_builder_requires_true_local_observations_and_projection_normalizes_clearance_at_boundary() -> None:
    canvas = _canvas()
    local = LocalObservation(canvas.identity, (500.8, 500.8, 507.2, 507.2), np.zeros((32, 32), dtype=np.float32), np.ones((32, 32), dtype=bool), np.zeros((32, 32), dtype=np.float32))
    world = ObservedWorld(canvas, np.zeros((256, 256), dtype=np.float32), np.ones((256, 256), dtype=bool), _obstacle_layer(canvas), local)
    with pytest.raises(ValueError, match="local"):
        ObservationBuilderV2().build(world, MissionRaster(canvas, np.ones((256, 256), dtype=np.float32), np.ones((256, 256), dtype=np.float32)), Pose2(512.0, 512.0), _test_only_projection(), CandidateBatch.empty(), "WHEELED")
    with pytest.raises(ValueError, match="clearance"):
        PlatformProjection(
            _canvas(),
            traversable_ratio=np.ones((256, 256), dtype=np.float32),
            local_traversable_ratio=np.ones((32, 32), dtype=np.float32),
            clearance_margin_norm=np.full((256, 256), 1.1, dtype=np.float32), source="test_only/proxy",
        )


def test_builder_requires_canvas_identity_local_center_and_relative_elevation() -> None:
    canvas = MapCanvas.from_roi_bounds("c" * 64, (1_000.0, 2_000.0, 1_020.0, 2_020.0))
    pose = Pose2(1_012.0, 2_008.0, elevation_m=10.0)
    local = LocalObservation(
        canvas_id=canvas.identity, bounds_m=(1_008.8, 2_004.8, 1_015.2, 2_011.2),
        elevation_m=np.full((32, 32), 12.0, dtype=np.float32), observed_mask=np.ones((32, 32), dtype=bool),
        physical_obstacle_ratio=np.zeros((32, 32), dtype=np.float32),
    )
    world = ObservedWorld(canvas, np.full((256, 256), 15.0, dtype=np.float32), np.ones((256, 256), dtype=bool), _obstacle_layer(canvas), local)
    mission = MissionRaster(canvas, np.ones((256, 256), dtype=np.float32), np.ones((256, 256), dtype=np.float32))
    projection = PlatformProjection(canvas, np.ones((256, 256), dtype=np.float32), np.ones((32, 32), dtype=np.float32), np.ones((256, 256), dtype=np.float32), "test_only/proxy")
    observation = ObservationBuilderV2(ElevationReference(5.0)).build(world, mission, pose, projection, CandidateBatch.empty(), "WHEELED")
    assert observation["prior_channels"][0, 0, 0, 0] == 10.0
    assert observation["local_crop"][0, 0, 0, 0] == 2.0
    assert observation["pose_features"][0, 0] == pytest.approx(514.0 / 1024.0)
    bad_local = LocalObservation("other", local.bounds_m, local.elevation_m, local.observed_mask, local.physical_obstacle_ratio)
    with pytest.raises(ValueError, match="canvas"):
        ObservedWorld(canvas, world.elevation_m, world.observed_mask, world.physical_obstacle_layer, bad_local)


def test_observed_world_rejects_physical_obstacle_layer_from_another_canvas() -> None:
    canvas = _canvas()
    other_canvas = MapCanvas.from_roi_bounds("a" * 64, (1_500.0, 500.0, 1_524.0, 524.0))
    local = LocalObservation(
        canvas.identity,
        (508.8, 508.8, 515.2, 515.2),
        np.zeros((32, 32), dtype=np.float32),
        np.ones((32, 32), dtype=bool),
        np.zeros((32, 32), dtype=np.float32),
    )

    with pytest.raises(ValueError, match="physical obstacle.*canvas identity"):
        ObservedWorld(
            canvas,
            np.zeros((256, 256), dtype=np.float32),
            np.ones((256, 256), dtype=bool),
            _obstacle_layer(other_canvas),
            local,
        )

from __future__ import annotations

import pathlib
import sys

import numpy as np


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "model_contract"))

from lunar_policy_training.environment.candidate_builder import CandidateBatch, CandidateBuilderV2, SensorGeometry  # noqa: E402
from lunar_policy_training.environment.observation_builder import LocalObservation, MissionRaster, ObservedWorld, PlatformProjection, Pose2  # noqa: E402
from lunar_policy_training.polar_data.raster import MapCanvas  # noqa: E402


def _canvas() -> MapCanvas:
    return MapCanvas.from_roi_bounds("d" * 64, (500.0, 500.0, 524.0, 524.0))


def _world_with_frontier(*, occlude: bool = False) -> ObservedWorld:
    observed = np.zeros((256, 256), dtype=bool)
    observed[120:136, 100:128] = True
    if occlude:
        observed[128, 110:120] = False
    canvas = _canvas()
    local = LocalObservation(canvas.identity, (508.0, 508.0, 516.0, 516.0), np.zeros((32, 32), dtype=np.float32), np.ones((32, 32), dtype=bool), np.zeros((32, 32), dtype=np.float32))
    return ObservedWorld(canvas, np.zeros((256, 256), dtype=np.float32), observed, np.zeros((256, 256), dtype=np.float32), local)


def _mission() -> MissionRaster:
    roi = np.zeros((256, 256), dtype=np.float32)
    roi[116:140, 96:144] = 1.0
    return MissionRaster(_canvas(), roi.copy(), roi, 1.0)


def _projection() -> PlatformProjection:
    return PlatformProjection(
        _canvas(),
        traversable_ratio=np.ones((256, 256), dtype=np.float32),
        local_traversable_ratio=np.ones((32, 32), dtype=np.float32),
        clearance_margin_norm=np.full((256, 256), 0.3, dtype=np.float32),
        source="test_only/proxy",
    )


def test_candidate_builder_is_observed_only_uses_exact_12_fields_and_stable_64_padding() -> None:
    batch = CandidateBuilderV2().build(_world_with_frontier(), _mission(), Pose2(450.0, 512.0), _projection())

    assert batch.features.shape == (64, 12)
    assert batch.mask.shape == (64,)
    assert batch.mask.dtype == np.bool_
    assert batch.mask.any()
    assert not batch.mask[batch.count :].any()
    assert np.all(batch.features[~batch.mask] == 0.0)
    valid = batch.features[batch.mask]
    assert np.all(np.isfinite(valid))
    assert np.all(np.abs(valid[:, :2]) <= 1.0)
    assert np.all(valid[:, 10] == 0.3)
    again = CandidateBuilderV2().build(_world_with_frontier(), _mission(), Pose2(450.0, 512.0), _projection())
    np.testing.assert_array_equal(batch.features, again.features)
    np.testing.assert_array_equal(batch.mask, again.mask)


def test_candidate_gain_ratios_use_fractional_roi_area_instead_of_cell_count() -> None:
    mission = _mission()
    mission = MissionRaster(_canvas(), mission.priority, mission.roi_ratio * 0.1, 1.0)
    batch = CandidateBuilderV2().build(_world_with_frontier(), mission, Pose2(450.0, 512.0), _projection())
    valid = batch.features[batch.mask]
    assert np.all(valid[:, 5] <= 1.0)
    assert np.all(valid[:, 6] <= 1.0)
    assert np.all(valid[:, 11] <= 1.0)


def test_candidate_builder_returns_all_false_instead_of_robot_fallback_when_los_has_no_frontier() -> None:
    observed = np.ones((256, 256), dtype=bool)
    canvas = _canvas()
    world = ObservedWorld(canvas, np.zeros((256, 256), dtype=np.float32), observed, np.zeros((256, 256), dtype=np.float32), LocalObservation(canvas.identity, (508.0, 508.0, 516.0, 516.0), np.zeros((32, 32), dtype=np.float32), np.ones((32, 32), dtype=bool), np.zeros((32, 32), dtype=np.float32)))
    batch = CandidateBuilderV2().build(world, _mission(), Pose2(512.0, 512.0), _projection())
    assert batch.count == 0
    np.testing.assert_array_equal(batch.features, CandidateBatch.empty().features)
    np.testing.assert_array_equal(batch.mask, CandidateBatch.empty().mask)


def test_sensor_geometry_and_obstacle_or_zero_traversable_reject_candidates() -> None:
    sensor = SensorGeometry(range_m=24.0, fov_rad=np.pi / 2.0)
    assert sensor.anchor_spacing_m > 0.0
    assert sensor.standoff_m > 0.0
    world = _world_with_frontier()
    blocked = ObservedWorld(world.canvas, world.elevation_m, world.observed_mask, np.where(world.observed_mask, 1.0, 0.0).astype(np.float32), world.local)
    assert CandidateBuilderV2(sensor).build(blocked, _mission(), Pose2(450.0, 512.0), _projection()).count == 0
    projection = _projection()
    zero = PlatformProjection(
        projection.canvas,
        traversable_ratio=np.zeros((256, 256), dtype=np.float32), local_traversable_ratio=projection.local_traversable_ratio,
        clearance_margin_norm=projection.clearance_margin_norm, source="test_only/proxy",
    )
    assert CandidateBuilderV2(sensor).build(world, _mission(), Pose2(450.0, 512.0), zero).count == 0

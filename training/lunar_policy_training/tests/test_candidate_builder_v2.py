from __future__ import annotations

import pathlib
import sys
import math

import numpy as np
import pytest


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "model_contract"))

from lunar_policy_training.environment import candidate_builder as candidate_builder_module  # noqa: E402
from lunar_policy_training.environment.candidate_builder import CandidateBatch, CandidateBuilderV2  # noqa: E402
from lunar_policy_training.environment.observation_builder import LocalObservation, MissionRaster, ObservedWorld, PlatformProjection, Pose2  # noqa: E402
from lunar_policy_training.environment.visibility import NativeVisibilityEstimator, SensorGeometry  # noqa: E402
from lunar_policy_training.polar_data.hazards import CanvasRatioLayer  # noqa: E402
from lunar_policy_training.polar_data.raster import MapCanvas  # noqa: E402
from lunar_policy_training.training_semantics import FORMAL_SENSOR_FOV_RAD, FORMAL_SENSOR_RANGE_M  # noqa: E402


def _canvas() -> MapCanvas:
    return MapCanvas.from_roi_bounds("d" * 64, (500.0, 500.0, 524.0, 524.0))


def _obstacle_layer(canvas: MapCanvas, values: np.ndarray | None = None) -> CanvasRatioLayer:
    if values is None:
        values = np.zeros((256, 256), dtype=np.float32)
    return CanvasRatioLayer(canvas, values)


def _world_with_frontier(*, occlude: bool = False) -> ObservedWorld:
    observed = np.zeros((256, 256), dtype=bool)
    observed[120:136, 100:128] = True
    if occlude:
        observed[128, 110:120] = False
    canvas = _canvas()
    local = LocalObservation(canvas.identity, (508.8, 508.8, 515.2, 515.2), np.zeros((32, 32), dtype=np.float32), np.ones((32, 32), dtype=bool), np.zeros((32, 32), dtype=np.float32))
    return ObservedWorld(canvas, np.zeros((256, 256), dtype=np.float32), observed, _obstacle_layer(canvas), local)


def _world(observed: np.ndarray) -> ObservedWorld:
    canvas = _canvas()
    local = LocalObservation(canvas.identity, (508.8, 508.8, 515.2, 515.2), np.zeros((32, 32), dtype=np.float32), np.ones((32, 32), dtype=bool), np.zeros((32, 32), dtype=np.float32))
    return ObservedWorld(canvas, np.zeros((256, 256), dtype=np.float32), observed, _obstacle_layer(canvas), local)


def _mission_for_roi(roi: np.ndarray) -> MissionRaster:
    ratio = roi.astype(np.float32)
    return MissionRaster(_canvas(), ratio.copy(), ratio)


def _mission() -> MissionRaster:
    roi = np.zeros((256, 256), dtype=np.float32)
    roi[116:140, 96:144] = 1.0
    return MissionRaster(_canvas(), roi.copy(), roi)


def _projection() -> PlatformProjection:
    return PlatformProjection(
        _canvas(),
        traversable_ratio=np.ones((256, 256), dtype=np.float32),
        local_traversable_ratio=np.ones((32, 32), dtype=np.float32),
        clearance_margin_norm=np.full((256, 256), 0.3, dtype=np.float32),
        source="test_only/proxy",
    )


def _builder(
    sensor: SensorGeometry | None = None,
) -> CandidateBuilderV2:
    geometry = sensor or SensorGeometry(
        range_m=FORMAL_SENSOR_RANGE_M,
        fov_rad=FORMAL_SENSOR_FOV_RAD,
    )
    return CandidateBuilderV2(
        NativeVisibilityEstimator(
            geometry,
            resolution_m=_canvas().geometry.resolution_m,
        )
    )


def test_candidate_builder_requires_explicit_sensor_estimator() -> None:
    with pytest.raises(TypeError):
        CandidateBuilderV2()

    builder = _builder()
    assert builder.sensor == SensorGeometry(
        range_m=30.0,
        fov_rad=2.0 * math.pi,
    )


def test_candidate_builder_is_observed_only_uses_exact_12_fields_and_stable_64_padding() -> None:
    batch = _builder().build(_world_with_frontier(), _mission(), Pose2(500.0, 512.0), _projection())

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
    again = _builder().build(_world_with_frontier(), _mission(), Pose2(500.0, 512.0), _projection())
    np.testing.assert_array_equal(batch.features, again.features)
    np.testing.assert_array_equal(batch.mask, again.mask)


def test_candidate_gain_ratios_use_fractional_roi_area_instead_of_cell_count() -> None:
    mission = _mission()
    mission = MissionRaster(_canvas(), mission.priority, mission.roi_ratio * 0.1)
    batch = _builder().build(_world_with_frontier(), mission, Pose2(500.0, 512.0), _projection())
    valid = batch.features[batch.mask]
    assert np.all(valid[:, 5] <= 1.0)
    assert np.all(valid[:, 6] <= 1.0)
    assert np.all(valid[:, 11] <= 1.0)


def test_candidate_builder_returns_all_false_instead_of_robot_fallback_when_los_has_no_frontier() -> None:
    observed = np.ones((256, 256), dtype=bool)
    canvas = _canvas()
    world = ObservedWorld(canvas, np.zeros((256, 256), dtype=np.float32), observed, _obstacle_layer(canvas), LocalObservation(canvas.identity, (508.8, 508.8, 515.2, 515.2), np.zeros((32, 32), dtype=np.float32), np.ones((32, 32), dtype=bool), np.zeros((32, 32), dtype=np.float32)))
    batch = _builder().build(world, _mission(), Pose2(512.0, 512.0), _projection())
    assert batch.count == 0
    np.testing.assert_array_equal(batch.features, CandidateBatch.empty().features)
    np.testing.assert_array_equal(batch.mask, CandidateBatch.empty().mask)


def test_candidate_builder_excludes_the_robot_cell_from_exploration_targets() -> None:
    canvas = _canvas()
    robot_cell = (128, 128)
    robot_x_m, robot_y_m = canvas.grid_center_world(*robot_cell)
    observed = np.zeros((256, 256), dtype=bool)
    observed[robot_cell] = True
    roi = np.zeros((256, 256), dtype=bool)
    roi[127:130, 127:130] = True

    batch = _builder().build(
        _world(observed),
        _mission_for_roi(roi),
        Pose2(robot_x_m, robot_y_m),
        _projection(),
    )

    assert batch.count == 0


def test_candidate_builder_excludes_visited_landing_cells_only() -> None:
    batch = _builder().build(
        _world_with_frontier(),
        _mission(),
        Pose2(500.0, 512.0),
        _projection(),
        excluded_cells={(121, 121)},
    )

    positions = batch.features[batch.mask, :2]
    assert batch.count == 23
    assert not np.any(
        np.all(positions == np.asarray([0.474609375, 0.474609375]), axis=1)
    )
    assert np.any(
        np.all(positions == np.asarray([0.474609375, 0.525390625]), axis=1)
    )


def test_sensor_geometry_and_obstacle_or_zero_traversable_reject_candidates() -> None:
    sensor = SensorGeometry(range_m=24.0, fov_rad=2.0 * np.pi)
    assert sensor.anchor_spacing_m > 0.0
    assert sensor.standoff_m > 0.0
    world = _world_with_frontier()
    blocked = ObservedWorld(world.canvas, world.elevation_m, world.observed_mask, _obstacle_layer(world.canvas, np.where(world.observed_mask, 1.0, 0.0).astype(np.float32)), world.local)
    assert _builder(sensor).build(blocked, _mission(), Pose2(500.0, 512.0), _projection()).count == 0
    projection = _projection()
    zero = PlatformProjection(
        projection.canvas,
        traversable_ratio=np.zeros((256, 256), dtype=np.float32), local_traversable_ratio=projection.local_traversable_ratio,
        clearance_margin_norm=projection.clearance_margin_norm, source="test_only/proxy",
    )
    assert _builder(sensor).build(world, _mission(), Pose2(500.0, 512.0), zero).count == 0


def test_long_single_contour_uses_farthest_fill_after_its_representative() -> None:
    observed = np.zeros((256, 256), dtype=bool)
    observed[112:145, 112:145] = True
    roi = np.zeros((256, 256), dtype=bool)
    roi[108:149, 108:149] = True

    batch = _builder(SensorGeometry(80.0, 2.0 * math.pi)).build(
        _world(observed), _mission_for_roi(roi), Pose2(512.0, 512.0), _projection()
    )

    assert batch.count > 1
    positions = batch.features[batch.mask, :2]
    assert np.unique(positions, axis=0).shape[0] == batch.count


def test_each_disconnected_frontier_segment_keeps_a_representative() -> None:
    observed = np.zeros((256, 256), dtype=bool)
    observed[116:141, 108:149] = True
    roi = observed.copy()
    holes = ((124, 116), (124, 140))
    for hole in holes:
        observed[hole] = False

    batch = _builder(SensorGeometry(80.0, 2.0 * math.pi)).build(
        _world(observed), _mission_for_roi(roi), Pose2(512.0, 512.0), _projection()
    )

    columns = batch.features[batch.mask, 0] * 256.0 - 0.5
    assert np.any(columns < 120.0)
    assert np.any(columns > 136.0)


def test_more_than_64_segment_representatives_use_farthest_subset() -> None:
    observed = np.zeros((256, 256), dtype=bool)
    observed[104:153, 104:153] = True
    roi = observed.copy()
    for index in range(28):
        angle = 2.0 * math.pi * index / 28.0
        hole = (round(128 + 18 * math.sin(angle)), round(128 + 18 * math.cos(angle)))
        observed[hole] = False

    batch = _builder(SensorGeometry(80.0, 2.0 * math.pi)).build(
        _world(observed), _mission_for_roi(roi), Pose2(512.0, 512.0), _projection()
    )

    assert batch.count == 64
    grid_columns = batch.features[batch.mask, 0] * 256.0 - 0.5
    grid_rows = batch.features[batch.mask, 1] * 256.0 - 0.5
    assert grid_columns.min() < 114.0 and grid_columns.max() > 142.0
    assert grid_rows.min() < 114.0 and grid_rows.max() > 142.0


def test_candidate_result_does_not_depend_on_argwhere_traversal(monkeypatch: pytest.MonkeyPatch) -> None:
    world = _world_with_frontier()
    baseline = _builder().build(world, _mission(), Pose2(500.0, 512.0), _projection())
    original_argwhere = np.argwhere
    monkeypatch.setattr(candidate_builder_module.np, "argwhere", lambda values: original_argwhere(values)[::-1])

    perturbed = _builder().build(world, _mission(), Pose2(500.0, 512.0), _projection())

    np.testing.assert_array_equal(perturbed.features, baseline.features)
    np.testing.assert_array_equal(perturbed.mask, baseline.mask)


def test_candidate_builder_batches_all_feasible_anchors_once_and_is_yaw_invariant() -> None:
    class RecordingEstimator:
        def __init__(self) -> None:
            self.sensor = SensorGeometry(30.0, 2.0 * math.pi)
            self.calls: list[np.ndarray] = []

        def estimate_candidate_gains(
            self,
            observed_mask,
            obstacle_ratio,
            roi_ratio,
            priority_weight,
            candidate_cells,
        ) -> np.ndarray:
            self.calls.append(candidate_cells.copy())
            return np.ones((candidate_cells.shape[0], 2), dtype=np.float32)

    estimator = RecordingEstimator()
    builder = CandidateBuilderV2(estimator)
    first = builder.build(
        _world_with_frontier(),
        _mission(),
        Pose2(500.0, 512.0, yaw_rad=-1.2),
        _projection(),
    )
    second = builder.build(
        _world_with_frontier(),
        _mission(),
        Pose2(500.0, 512.0, yaw_rad=2.4),
        _projection(),
    )

    assert len(estimator.calls) == 2
    assert estimator.calls[0].shape[0] > 0
    np.testing.assert_array_equal(estimator.calls[0], estimator.calls[1])
    np.testing.assert_array_equal(first.features, second.features)
    np.testing.assert_array_equal(first.mask, second.mask)

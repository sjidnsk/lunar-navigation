from __future__ import annotations

import pathlib
import sys
import math
from types import SimpleNamespace

import numpy as np
import pytest


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "model_contract"))

from lunar_policy_training.environment import candidate_builder as candidate_builder_module  # noqa: E402
from lunar_policy_training.environment.candidate_builder import CandidateBatch, CandidateBuilderV2  # noqa: E402
from lunar_policy_training.environment.observation_builder import LocalObservation, MissionRaster, ObservedWorld, PlatformProjection, Pose2  # noqa: E402
from lunar_policy_training.environment.platform_reachability import CandidateReachabilityResult, PlatformCandidateReachability  # noqa: E402
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


def _detour_fixture() -> tuple[ObservedWorld, MissionRaster, PlatformProjection, Pose2, tuple[int, int]]:
    canvas = _canvas()
    observed = np.zeros((256, 256), dtype=bool)
    observed[124:133, 124:136] = True
    roi = np.zeros((256, 256), dtype=bool)
    roi[124:133, 124:137] = True
    obstacles = np.zeros((256, 256), dtype=np.float32)
    obstacles[128, 130] = 1.0
    local = LocalObservation(
        canvas.identity,
        (508.8, 508.8, 515.2, 515.2),
        np.zeros((32, 32), dtype=np.float32),
        np.ones((32, 32), dtype=bool),
        np.zeros((32, 32), dtype=np.float32),
    )
    world = ObservedWorld(
        canvas,
        np.zeros((256, 256), dtype=np.float32),
        observed,
        _obstacle_layer(canvas, obstacles),
        local,
    )
    traversable = np.zeros((256, 256), dtype=np.float32)
    traversable[128, 128] = 1.0
    traversable[127, 128:135] = 1.0
    traversable[128, 134] = 1.0
    projection = PlatformProjection(
        canvas,
        traversable_ratio=traversable,
        local_traversable_ratio=np.ones((32, 32), dtype=np.float32),
        clearance_margin_norm=np.full((256, 256), 0.3, dtype=np.float32),
        source="test_only/proxy",
    )
    robot_x_m, robot_y_m = canvas.grid_center_world(128, 128)
    return world, _mission_for_roi(roi), projection, Pose2(robot_x_m, robot_y_m), (128, 134)


class _RecordingEstimator:
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
    batch = _builder().build(_world_with_frontier(), _mission(), Pose2(500.0, 512.0), _projection(), platform_type="WHEELED")

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
    again = _builder().build(_world_with_frontier(), _mission(), Pose2(500.0, 512.0), _projection(), platform_type="WHEELED")
    np.testing.assert_array_equal(batch.features, again.features)
    np.testing.assert_array_equal(batch.mask, again.mask)


def test_candidate_gain_ratios_use_fractional_roi_area_instead_of_cell_count() -> None:
    mission = _mission()
    mission = MissionRaster(_canvas(), mission.priority, mission.roi_ratio * 0.1)
    batch = _builder().build(_world_with_frontier(), mission, Pose2(500.0, 512.0), _projection(), platform_type="WHEELED")
    valid = batch.features[batch.mask]
    assert np.all(valid[:, 5] <= 1.0)
    assert np.all(valid[:, 6] <= 1.0)
    assert np.all(valid[:, 11] <= 1.0)


def test_candidate_gain_features_are_normalized_against_the_reachable_batch() -> None:
    class UnequalGainEstimator(_RecordingEstimator):
        def estimate_candidate_gains(self, *args) -> np.ndarray:
            candidates = args[-1]
            self.calls.append(candidates.copy())
            gains = np.full((len(candidates), 2), (8.0, 16.0), np.float32)
            gains[0] = (2.0, 4.0)
            return gains

    batch = CandidateBuilderV2(UnequalGainEstimator()).build(
        _world_with_frontier(),
        _mission(),
        Pose2(500.0, 512.0),
        _projection(),
        platform_type="WHEELED",
    )

    assert batch.count >= 2
    valid = batch.features[batch.mask]
    np.testing.assert_allclose(np.unique(valid[:, 5]), [0.25, 1.0])
    np.testing.assert_allclose(np.unique(valid[:, 6]), [0.25, 1.0])


def test_candidate_builder_returns_all_false_instead_of_robot_fallback_when_los_has_no_frontier() -> None:
    observed = np.ones((256, 256), dtype=bool)
    canvas = _canvas()
    world = ObservedWorld(canvas, np.zeros((256, 256), dtype=np.float32), observed, _obstacle_layer(canvas), LocalObservation(canvas.identity, (508.8, 508.8, 515.2, 515.2), np.zeros((32, 32), dtype=np.float32), np.ones((32, 32), dtype=bool), np.zeros((32, 32), dtype=np.float32)))
    batch = _builder().build(world, _mission(), Pose2(512.0, 512.0), _projection(), platform_type="WHEELED")
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
        platform_type="WHEELED",
    )

    assert batch.count == 0


def test_candidate_builder_excludes_visited_landing_cells_only() -> None:
    batch = _builder().build(
        _world_with_frontier(),
        _mission(),
        Pose2(500.0, 512.0),
        _projection(),
        platform_type="WHEELED",
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
    assert batch.diagnostics.visited_excluded_count == 1


def test_sensor_geometry_and_obstacle_or_zero_traversable_reject_candidates() -> None:
    sensor = SensorGeometry(range_m=24.0, fov_rad=2.0 * np.pi)
    assert sensor.anchor_spacing_m > 0.0
    assert sensor.standoff_m > 0.0
    world = _world_with_frontier()
    blocked = ObservedWorld(world.canvas, world.elevation_m, world.observed_mask, _obstacle_layer(world.canvas, np.where(world.observed_mask, 1.0, 0.0).astype(np.float32)), world.local)
    assert _builder(sensor).build(blocked, _mission(), Pose2(500.0, 512.0), _projection(), platform_type="WHEELED").count == 0
    projection = _projection()
    zero = PlatformProjection(
        projection.canvas,
        traversable_ratio=np.zeros((256, 256), dtype=np.float32), local_traversable_ratio=projection.local_traversable_ratio,
        clearance_margin_norm=projection.clearance_margin_norm, source="test_only/proxy",
    )
    assert _builder(sensor).build(world, _mission(), Pose2(500.0, 512.0), zero, platform_type="WHEELED").count == 0


def test_long_single_contour_uses_farthest_fill_after_its_representative() -> None:
    observed = np.zeros((256, 256), dtype=bool)
    observed[112:145, 112:145] = True
    roi = np.zeros((256, 256), dtype=bool)
    roi[108:149, 108:149] = True

    batch = _builder(SensorGeometry(80.0, 2.0 * math.pi)).build(
        _world(observed), _mission_for_roi(roi), Pose2(512.0, 512.0), _projection(), platform_type="WHEELED"
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
        _world(observed), _mission_for_roi(roi), Pose2(512.0, 512.0), _projection(), platform_type="WHEELED"
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
        _world(observed), _mission_for_roi(roi), Pose2(512.0, 512.0), _projection(), platform_type="WHEELED"
    )

    assert batch.count == 64
    grid_columns = batch.features[batch.mask, 0] * 256.0 - 0.5
    grid_rows = batch.features[batch.mask, 1] * 256.0 - 0.5
    assert grid_columns.min() < 114.0 and grid_columns.max() > 142.0
    assert grid_rows.min() < 114.0 and grid_rows.max() > 142.0


def test_candidate_result_does_not_depend_on_argwhere_traversal(monkeypatch: pytest.MonkeyPatch) -> None:
    world = _world_with_frontier()
    baseline = _builder().build(world, _mission(), Pose2(500.0, 512.0), _projection(), platform_type="WHEELED")
    original_argwhere = np.argwhere
    monkeypatch.setattr(candidate_builder_module.np, "argwhere", lambda values: original_argwhere(values)[::-1])

    perturbed = _builder().build(world, _mission(), Pose2(500.0, 512.0), _projection(), platform_type="WHEELED")

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
        platform_type="WHEELED",
    )
    second = builder.build(
        _world_with_frontier(),
        _mission(),
        Pose2(500.0, 512.0, yaw_rad=2.4),
        _projection(),
        platform_type="WHEELED",
    )

    assert len(estimator.calls) == 2
    assert estimator.calls[0].shape[0] > 0
    np.testing.assert_array_equal(estimator.calls[0], estimator.calls[1])
    np.testing.assert_array_equal(first.features, second.features)
    np.testing.assert_array_equal(first.mask, second.mask)


def test_sparse_primary_candidates_add_platform_checked_observation_reserves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Would fail when one coarse-reachable goal can end the whole episode."""

    class SparsePrimaryEstimator(_RecordingEstimator):
        def estimate_candidate_gains(self, *args) -> np.ndarray:
            candidates = args[-1]
            self.calls.append(candidates.copy())
            gains = np.ones((len(candidates), 2), dtype=np.float32)
            if len(self.calls) == 1:
                gains.fill(0.0)
                gains[0] = (1.0, 1.0)
            return gains

    reachability_calls: list[np.ndarray] = []

    def platform_filter(
        _self,
        candidate_cells: np.ndarray,
        *,
        target_positions_map: np.ndarray | None = None,
    ) -> CandidateReachabilityResult:
        del target_positions_map
        reachability_calls.append(candidate_cells.copy())
        accepted = np.ascontiguousarray(
            candidate_cells[:, 1] % 2 == 0, dtype=np.bool_
        )
        return CandidateReachabilityResult(
            accepted,
            {
                "platform_unreachable_count": int(
                    (~accepted).sum(dtype=np.int64)
                )
            },
        )

    monkeypatch.setattr(
        PlatformCandidateReachability,
        "filter",
        platform_filter,
    )
    reachability = object.__new__(PlatformCandidateReachability)
    estimator = SparsePrimaryEstimator()

    batch = CandidateBuilderV2(estimator).build(
        _world_with_frontier(),
        _mission(),
        Pose2(500.0, 512.0),
        _projection(),
        platform_type="WHEELED",
        platform_reachability=reachability,
    )

    assert len(estimator.calls) == 2
    assert len(reachability_calls) == 2
    assert estimator.calls[0].shape[0] > 0
    assert estimator.calls[1].shape[0] > estimator.calls[0].shape[0]
    assert set(map(tuple, estimator.calls[0])) < set(
        map(tuple, estimator.calls[1])
    )
    assert batch.count > 1
    assert batch.diagnostics.emitted_count == batch.count


def test_ground_platform_keeps_candidate_when_observed_detour_exists() -> None:
    world, mission, projection, pose, target = _detour_fixture()
    estimator = _RecordingEstimator()
    builder = CandidateBuilderV2(estimator)

    batch = builder.build(
        world,
        mission,
        pose,
        projection,
        platform_type="WHEELED",
    )
    legacy = builder.build(
        world,
        mission,
        pose,
        projection,
        platform_type="WHEELED",
        platform_reachability_filter_enabled=False,
    )

    assert target in map(tuple, estimator.calls[0])
    assert len(estimator.calls) == 2
    assert set(map(tuple, estimator.calls[0])) < set(
        map(tuple, estimator.calls[1])
    )
    assert legacy.count == 0
    assert batch.count > legacy.count
    assert batch.diagnostics.emitted_count == batch.count
    assert batch.diagnostics.frontier_anchor_count >= batch.count
    assert (
        batch.diagnostics.static_infeasible_count
        + batch.diagnostics.platform_unreachable_count
        > 0
    )


def test_hopper_keeps_observed_landing_when_ground_ray_is_blocked() -> None:
    world, mission, projection, pose, target = _detour_fixture()
    estimator = _RecordingEstimator()
    builder = CandidateBuilderV2(estimator)

    batch = builder.build(
        world,
        mission,
        pose,
        projection,
        platform_type="HOPPER",
    )
    legacy = builder.build(
        world,
        mission,
        pose,
        projection,
        platform_type="HOPPER",
        platform_reachability_filter_enabled=False,
    )

    assert target in map(tuple, estimator.calls[0])
    assert len(estimator.calls) == 2
    assert set(map(tuple, estimator.calls[0])) < set(
        map(tuple, estimator.calls[1])
    )
    assert legacy.count == 0
    assert batch.count > legacy.count


def test_hopper_rejects_unobserved_or_projection_infeasible_landings() -> None:
    world, mission, projection, pose, target = _detour_fixture()
    zero_projection = PlatformProjection(
        projection.canvas,
        traversable_ratio=np.zeros((256, 256), dtype=np.float32),
        local_traversable_ratio=projection.local_traversable_ratio,
        clearance_margin_norm=projection.clearance_margin_norm,
        source="test_only/proxy",
    )
    unsafe = CandidateBuilderV2(_RecordingEstimator()).build(
        world,
        mission,
        pose,
        zero_projection,
        platform_type="HOPPER",
    )
    observed = world.observed_mask.copy()
    observed[target] = False
    target_only = np.zeros((256, 256), dtype=np.float32)
    target_only[target] = 1.0
    target_projection = PlatformProjection(
        projection.canvas,
        traversable_ratio=target_only,
        local_traversable_ratio=projection.local_traversable_ratio,
        clearance_margin_norm=projection.clearance_margin_norm,
        source="test_only/proxy",
    )
    unobserved_world = ObservedWorld(
        world.canvas,
        world.elevation_m,
        observed,
        world.physical_obstacle_layer,
        world.local,
    )
    unobserved = CandidateBuilderV2(_RecordingEstimator()).build(
        unobserved_world,
        mission,
        pose,
        target_projection,
        platform_type="HOPPER",
    )

    assert unsafe.count == 0
    assert unsafe.diagnostics.static_infeasible_count > 0
    assert unobserved.count == 0


def test_hopper_falls_back_to_reachable_positive_gain_observation_pose() -> None:
    canvas = _canvas()
    robot = (128, 128)
    robot_x_m, robot_y_m = canvas.grid_center_world(*robot)
    observed = np.zeros((256, 256), dtype=np.bool_)
    observed[127:130, 128:142] = True
    roi = np.zeros((256, 256), dtype=np.bool_)
    roi[127:130, 128:143] = True
    estimator = _RecordingEstimator()

    batch = CandidateBuilderV2(estimator).build(
        _world(observed),
        _mission_for_roi(roi),
        Pose2(robot_x_m, robot_y_m),
        _projection(),
        platform_type="HOPPER",
    )

    assert batch.count > 0
    assert len(estimator.calls) == 1
    frontier_cell = (128, 141)
    assert frontier_cell not in map(tuple, estimator.calls[0])
    candidate_distances_m = np.linalg.norm(
        estimator.calls[0] - np.asarray(robot), axis=1
    ) * canvas.geometry.resolution_m
    assert np.all(candidate_distances_m <= 30.0)
    assert np.any(estimator.calls[0][:, 1] >= 135)


def test_hopper_emits_truthful_zero_gain_transit_when_frontier_is_remote() -> None:
    canvas = _canvas()
    robot = (128, 128)
    robot_x_m, robot_y_m = canvas.grid_center_world(*robot)
    observed = np.zeros((256, 256), dtype=np.bool_)
    observed[127:130, 128:142] = True
    roi = np.zeros((256, 256), dtype=np.bool_)
    roi[127:130, 128:143] = True

    class ZeroGainEstimator(_RecordingEstimator):
        def estimate_candidate_gains(self, *args) -> np.ndarray:
            self.calls.append(args[-1].copy())
            return np.zeros((len(args[-1]), 2), dtype=np.float32)

    batch = CandidateBuilderV2(ZeroGainEstimator()).build(
        _world(observed),
        _mission_for_roi(roi),
        Pose2(robot_x_m, robot_y_m),
        _projection(),
        platform_type="HOPPER",
    )

    assert batch.count > 0
    assert np.all(batch.features[batch.mask, 5:7] == 0.0)
    assert batch.diagnostics.zero_gain_count == 0


def test_hopper_emits_zero_gain_transit_when_frontier_anchors_have_no_gain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = _world_with_frontier()
    mission = _mission()
    projection = _projection()
    pose = Pose2(500.0, 512.0)

    class ZeroGainEstimator(_RecordingEstimator):
        def estimate_candidate_gains(self, *args) -> np.ndarray:
            self.calls.append(args[-1].copy())
            return np.zeros((len(args[-1]), 2), dtype=np.float32)

    def accept(
        _self,
        candidate_cells: np.ndarray,
        *,
        target_positions_map: np.ndarray | None = None,
    ) -> CandidateReachabilityResult:
        del target_positions_map
        return CandidateReachabilityResult(
            np.ones(len(candidate_cells), dtype=np.bool_),
            {"platform_unreachable_count": 0},
        )

    monkeypatch.setattr(PlatformCandidateReachability, "filter", accept)
    reachability = object.__new__(PlatformCandidateReachability)
    batch = CandidateBuilderV2(ZeroGainEstimator()).build(
        world,
        mission,
        pose,
        projection,
        platform_type="HOPPER",
        platform_reachability=reachability,
    )

    assert batch.count > 0
    assert np.all(batch.features[batch.mask, 5:7] == 0.0)


def test_zero_gain_backtrack_preserves_exact_parent_pose(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canvas = _canvas()
    robot = (128, 128)
    robot_x_m, robot_y_m = canvas.grid_center_world(*robot)
    observed = np.zeros((256, 256), dtype=np.bool_)
    observed[127:130, 128:142] = True
    roi = np.zeros((256, 256), dtype=np.bool_)
    roi[127:130, 128:143] = True
    visited = set(map(tuple, np.column_stack(np.nonzero(observed))))

    class ZeroGainEstimator(_RecordingEstimator):
        def estimate_candidate_gains(self, *args) -> np.ndarray:
            self.calls.append(args[-1].copy())
            return np.zeros((len(args[-1]), 2), dtype=np.float32)

    estimator = ZeroGainEstimator()
    parent_cell = (128, 129)
    parent_center_x, parent_center_y = canvas.grid_center_world(*parent_cell)
    parent = Pose2(
        parent_center_x + 0.02,
        parent_center_y - 0.02,
        elevation_m=123.0,
    )
    recorded_positions: list[np.ndarray | None] = []

    def accept(
        _self,
        candidate_cells: np.ndarray,
        target_positions_map: np.ndarray | None = None,
    ) -> CandidateReachabilityResult:
        recorded_positions.append(
            None if target_positions_map is None else target_positions_map.copy()
        )
        return CandidateReachabilityResult(
            np.ones(len(candidate_cells), dtype=np.bool_),
            {"platform_unreachable_count": 0},
        )

    monkeypatch.setattr(PlatformCandidateReachability, "filter", accept)
    reachability = object.__new__(PlatformCandidateReachability)
    batch = CandidateBuilderV2(estimator).build(
        _world(observed),
        _mission_for_roi(roi),
        Pose2(robot_x_m, robot_y_m),
        _projection(),
        platform_type="HOPPER",
        platform_reachability=reachability,
        excluded_cells=visited,
        backtrack_pose=parent,
    )

    assert batch.count == 1
    assert recorded_positions[-1] is not None
    np.testing.assert_array_equal(
        recorded_positions[-1],
        np.asarray([[parent.x_m, parent.y_m, parent.elevation_m]]),
    )
    np.testing.assert_allclose(
        batch.features[0, :2],
        np.asarray(
            [
                (parent.x_m - canvas.bounds_m[0]) / canvas.geometry.size_m,
                (canvas.bounds_m[3] - parent.y_m) / canvas.geometry.size_m,
            ]
        ),
    )
    assert batch.target_elevation_m[0] == 123.0
    assert np.all(batch.features[0, 5:7] == 0.0)


def test_platform_filter_rejects_unknown_platform_and_is_byte_deterministic() -> None:
    world, mission, projection, pose, _ = _detour_fixture()
    builder = CandidateBuilderV2(_RecordingEstimator())

    with pytest.raises(ValueError, match="platform_type"):
        builder.build(world, mission, pose, projection, platform_type="FLYING")

    first = builder.build(world, mission, pose, projection, platform_type="LEGGED")
    second = builder.build(world, mission, pose, projection, platform_type="LEGGED")
    np.testing.assert_array_equal(first.features, second.features)
    np.testing.assert_array_equal(first.mask, second.mask)
    assert first.diagnostics == second.diagnostics


def test_stage_diagnostics_isolate_platform_unreachable_and_zero_gain() -> None:
    world = _world_with_frontier()
    mission = _mission()
    projection = _projection()
    pose = Pose2(500.0, 512.0)

    class RejectingBridge:
        def project_reachability(self, request, maximum_edge_distance_m):
            del request, maximum_edge_distance_m
            return type(
                "Projection",
                (),
                {"reachable": np.zeros((256, 256), dtype=np.uint8)},
            )()

    reachability = PlatformCandidateReachability(
        platform_type="WHEELED",
        canvas=world.canvas,
        pose_map=pose,
        observed_elevation_m=world.elevation_m,
        bridge=RejectingBridge(),
        request=SimpleNamespace(
            world=SimpleNamespace(
                local_map=SimpleNamespace(
                    frame_id="odom",
                    width=256,
                    height=256,
                    resolution_m=4.0,
                    origin_m=SimpleNamespace(x=0.0, y=0.0, z=0.0),
                ),
                map_from_odom=SimpleNamespace(
                    parent_frame="map",
                    child_frame="odom",
                    translation_m=SimpleNamespace(x=0.0, y=0.0, z=0.0),
                    rotation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
                ),
            )
            ),
            local_traversability_projection=SimpleNamespace(
                hard_feasible=np.zeros((256, 256), dtype=np.uint8),
                connected_component=np.full((256, 256), -1, dtype=np.int32),
            ),
        )
    unreachable = CandidateBuilderV2(_RecordingEstimator()).build(
        world,
        mission,
        pose,
        projection,
        platform_type="WHEELED",
        platform_reachability=reachability,
    )

    class ZeroGainEstimator(_RecordingEstimator):
        def estimate_candidate_gains(self, *args) -> np.ndarray:
            candidates = args[-1]
            return np.zeros((len(candidates), 2), dtype=np.float32)

    zero_gain = CandidateBuilderV2(ZeroGainEstimator()).build(
        world,
        mission,
        pose,
        projection,
        platform_type="HOPPER",
        platform_reachability_filter_enabled=False,
    )

    assert unreachable.count == 0
    assert unreachable.diagnostics.platform_unreachable_count > 0
    assert unreachable.diagnostics.zero_gain_count == 0
    assert zero_gain.count == 0
    assert zero_gain.diagnostics.platform_unreachable_count == 0
    assert zero_gain.diagnostics.zero_gain_count > 0


def test_ground_reachability_uses_local_detail_authority_inside_window_and_global_outside() -> None:
    canvas = _canvas()
    pose = Pose2(2.0, 2.0)
    local_shape = (50, 50)
    local_hard = np.ones(local_shape, dtype=np.uint8)
    local_components = np.ones(local_shape, dtype=np.int32)
    local_components[10, 30] = 2

    class RecordingBridge:
        def project_reachability(self, request, maximum_edge_distance_m):
            del request, maximum_edge_distance_m
            reachable = np.zeros((256, 256), dtype=np.uint8)
            reachable[0, 3] = 1
            return SimpleNamespace(
                reachable=reachable
            )

        def project_traversability(self, request):
            del request
            return SimpleNamespace(
                hard_feasible=local_hard,
                connected_component=local_components,
            )

    request = SimpleNamespace(
        world=SimpleNamespace(
            local_map=SimpleNamespace(
                frame_id="odom",
                width=local_shape[1],
                height=local_shape[0],
                resolution_m=0.2,
                origin_m=SimpleNamespace(x=0.0, y=0.0, z=0.0),
            ),
            map_from_odom=SimpleNamespace(
                parent_frame="map",
                child_frame="odom",
                translation_m=SimpleNamespace(x=0.0, y=0.0, z=0.0),
                rotation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
            ),
        )
    )
    reachability = PlatformCandidateReachability(
        platform_type="LEGGED",
        canvas=canvas,
        pose_map=pose,
        observed_elevation_m=np.zeros((256, 256), dtype=np.float32),
        bridge=RecordingBridge(),
        request=request,
    )
    candidates = np.asarray(((255, 0), (255, 1), (255, 3)), dtype=np.int32)
    exact_targets = np.asarray(
        ((3.9, 2.0, 0.0), (6.0, 2.0, 0.0), (12.0, 2.0, 0.0)),
        dtype=np.float64,
    )

    result = reachability.filter(
        candidates, target_positions_map=exact_targets
    )

    # The observed detail component is authoritative inside its window even
    # when the partially observed 4 m start cell makes the global component
    # empty. Targets outside the detail window retain the global result.
    assert result.accepted_mask.tolist() == [True, False, True]
    assert dict(result.reason_counts) == {"platform_unreachable_count": 1}


def test_hopper_reachability_uses_certified_pose_height_for_start() -> None:
    canvas = _canvas()
    start = (128, 128)
    target = (128, 129)
    start_x_m, start_y_m = canvas.grid_center_world(*start)
    elevation = np.full((256, 256), 7.0, dtype=np.float32)

    class RecordingBridge:
        def __init__(self) -> None:
            self.targets = None

        def project_hopper_landing_evidence(self, request, targets):
            del request
            self.targets = targets.copy()
            count = len(targets)
            return SimpleNamespace(
                certified=np.ones(count, dtype=np.bool_),
                aim_positions_m=targets.copy(),
                boundary_m=np.zeros((count, 4, 3), dtype=np.float64),
                area_m2=np.ones(count, dtype=np.float64),
                algorithm_id="test/landing-evidence/v1",
            )

        def project_direct_hopper_reachability(
            self, request, maximum_edge_distance_m, evidence
        ):
            del request, maximum_edge_distance_m, evidence
            return SimpleNamespace(
                reachable=np.ones((256, 256), dtype=np.uint8)
            )

    bridge = RecordingBridge()
    reachability = PlatformCandidateReachability(
        platform_type="HOPPER",
        canvas=canvas,
        pose_map=Pose2(start_x_m, start_y_m, elevation_m=123.0),
        observed_elevation_m=elevation,
        bridge=bridge,
        request=object(),
    )

    target_x_m, target_y_m = canvas.grid_center_world(*target)
    exact_target = np.asarray(
        [[target_x_m + 0.02, target_y_m - 0.02, 321.0]],
        dtype=np.float64,
    )
    result = reachability.filter(
        np.asarray([target], dtype=np.int32),
        target_positions_map=exact_target,
    )

    assert result.accepted_mask.tolist() == [True]
    np.testing.assert_array_equal(
        bridge.targets[0], np.asarray([start_x_m, start_y_m, 123.0])
    )
    np.testing.assert_array_equal(bridge.targets[1], exact_target[0])

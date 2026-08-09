from __future__ import annotations

import math

import numpy as np
import pytest

from lunar_policy_training.environment.multires_observation import (
    MultiresSensorObservationState,
)
from lunar_policy_training.environment.observation_builder import Pose2
from lunar_policy_training.polar_data.hazards import (
    RockCircle,
    VectorHazardScene,
)
from lunar_policy_training.polar_data.multires_scene import (
    MultiResolutionScene,
    SceneTileProvider,
)
from lunar_policy_training.polar_data.raster import MapCanvas


def _state() -> tuple[MultiresSensorObservationState, SceneTileProvider]:
    canvas = MapCanvas("a" * 64, (0.0, 0.0, 1024.0, 1024.0))
    vector = VectorHazardScene(
        seed="b" * 64,
        canvas=canvas,
        rocks=(RockCircle(522.0, 512.0, 0.8, 1.0),),
        craters=(),
        no_go_polygons=(),
    )
    scene = MultiResolutionScene(
        canvas,
        np.full((256, 256), 7.0, np.float32),
        np.ones((256, 256), bool),
        vector,
        scenario_id="c" * 64,
    )
    provider = SceneTileProvider(scene, capacity=8)
    state = MultiresSensorObservationState(
        scene=scene,
        tile_provider=provider,
        mission_roi_ratio=np.ones((256, 256), np.float32),
        mission_priority=np.full((256, 256), 0.5, np.float32),
    )
    return state, provider


def test_reveal_uses_point_zero_four_square_metres_and_deduplicates_overlap() -> None:
    state, _ = _state()
    pose = Pose2(512.0, 512.0, elevation_m=7.0)

    first = state.observe_world(pose, elapsed_s=0.0)
    second = state.observe_world(pose, elapsed_s=1.0)

    assert first.newly_observed_cells > 0
    assert first.mission_observed_delta_m2 == pytest.approx(
        first.newly_observed_cells * 0.04
    )
    assert first.priority_observed_delta_m2 == pytest.approx(
        first.newly_observed_cells * 0.04 * 0.5
    )
    assert second.newly_observed_cells == 0
    assert second.mission_observed_delta_m2 == 0.0
    assert state.detail_observation_count_at(512.0, 512.0) == 2


def test_rock_is_observed_but_occludes_cells_behind_it() -> None:
    state, _ = _state()

    state.observe_world(Pose2(512.0, 512.0, elevation_m=7.0), elapsed_s=0.0)

    assert state.detail_observed_at(521.2, 512.0)
    assert not state.detail_observed_at(525.0, 512.0)


def test_four_central_subcells_gate_conservative_four_metre_validity() -> None:
    state, _ = _state()
    state.observe_world(Pose2(512.0, 512.0, elevation_m=7.0), elapsed_s=0.0)

    pose_row, pose_column = state.observed.canvas.world_to_grid(512.0, 512.0)
    assert state.observed.valid_mask[pose_row, pose_column]
    assert state.detail_block_valid(pose_row, pose_column)[9:11, 9:11].all()

    partial_cells = []
    for row in range(max(0, pose_row - 9), min(256, pose_row + 10)):
        for column in range(
            max(0, pose_column - 9), min(256, pose_column + 10)
        ):
            block = state.detail_block_valid(row, column)
            if block.any() and not block[9:11, 9:11].all():
                partial_cells.append((row, column))
    assert partial_cells
    assert all(not state.observed.valid_mask[cell] for cell in partial_cells)


def test_local_crop_is_exact_six_point_four_metres_and_never_exposes_unknown_truth() -> None:
    state, _ = _state()
    pose = Pose2(512.0, 512.0, elevation_m=7.0)
    state.observe_world(pose, elapsed_s=0.0)

    local = state.local_observation(pose)

    assert local.bounds_m == pytest.approx((508.8, 508.8, 515.2, 515.2))
    assert local.elevation_m.shape == (32, 32)
    assert local.observed_mask.shape == (32, 32)
    assert local.observed_mask.any()
    assert np.all(local.elevation_m[~local.observed_mask] == 0.0)
    assert np.all(local.physical_obstacle_ratio[~local.observed_mask] == 0.0)
    assert np.all(local.elevation_m[local.observed_mask] == 7.0)


def test_detail_state_is_sparse_and_truth_tiles_remain_lru_bounded() -> None:
    state, provider = _state()

    state.observe_world(Pose2(512.0, 512.0, elevation_m=7.0), elapsed_s=0.0)

    assert 1 <= state.allocated_detail_tiles <= 4
    assert provider.cache_size <= provider.capacity
    assert not hasattr(state, "global_detail_grid")


def test_observe_world_rejects_a_pose_without_thirty_metre_scene_margin() -> None:
    state, _ = _state()

    with pytest.raises(ValueError, match="detail window"):
        state.observe_world(Pose2(1.0, 1.0), elapsed_s=0.0)


def test_sensor_geometry_remains_formal_thirty_metre_full_circle() -> None:
    state, _ = _state()

    assert state.visibility_estimator.sensor.range_m == 30.0
    assert state.visibility_estimator.sensor.fov_rad == pytest.approx(2.0 * math.pi)
    assert state.visibility_estimator.resolution_m == 0.2


def test_candidate_gains_use_one_observed_only_detail_window_and_physical_area() -> None:
    state, _ = _state()
    state.observe_world(Pose2(512.0, 512.0, elevation_m=7.0), elapsed_s=0.0)
    calls: list[tuple[np.ndarray, ...]] = []

    class RecordingDetailEstimator:
        sensor = state.visibility_estimator.sensor
        resolution_m = 0.2

        def estimate_candidate_gains(
            self,
            observed_mask: np.ndarray,
            obstacle_ratio: np.ndarray,
            roi_ratio: np.ndarray,
            priority_weight: np.ndarray,
            candidate_cells: np.ndarray,
        ) -> np.ndarray:
            calls.append(
                (
                    observed_mask.copy(),
                    obstacle_ratio.copy(),
                    roi_ratio.copy(),
                    priority_weight.copy(),
                    candidate_cells.copy(),
                )
            )
            return np.ascontiguousarray(
                np.tile(np.asarray([[400.0, 200.0]], np.float32), (len(candidate_cells), 1))
            )

    state.visibility_estimator = RecordingDetailEstimator()
    candidate_cells = np.ascontiguousarray([[127, 127], [128, 128]], np.int32)

    gains = state.estimate_candidate_gains(
        state.observed.valid_mask,
        state.observed.physical_obstacle_ratio,
        state.mission_roi_ratio,
        state.mission_priority * state.mission_roi_ratio,
        candidate_cells,
    )

    assert len(calls) == 1
    observed, obstacles, roi, priority, local_candidates = calls[0]
    assert observed.shape == obstacles.shape == roi.shape == priority.shape
    assert observed.shape[0] >= 301 and observed.shape[1] >= 301
    assert observed.dtype == np.bool_
    assert obstacles.dtype == roi.dtype == priority.dtype == np.float32
    assert local_candidates.dtype == np.int32
    assert np.all(obstacles[~observed] == 0.0)
    np.testing.assert_allclose(gains, [[1.0, 0.5], [1.0, 0.5]])

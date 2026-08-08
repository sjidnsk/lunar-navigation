from __future__ import annotations

import numpy as np
import pytest

from lunar_policy_training.environment.observation_builder import (
    LocalObservation,
    ObservedWorld,
)
from lunar_policy_training.environment.sensor_observation import (
    SensorObservationState,
    TrainingObservedGrid,
    TrainingWorldTruth,
)
from lunar_policy_training.environment.visibility import (
    NativeVisibilityEstimator,
    SensorGeometry,
)
from lunar_policy_training.polar_data.raster import (
    GLOBAL_GEOMETRY,
    GridGeometry,
    MapCanvas,
)


def _small_canvas(identity: str = "a" * 64) -> MapCanvas:
    geometry = GridGeometry(size_m=11.0, resolution_m=1.0, cells=11)
    return MapCanvas(identity, (0.0, 0.0, 11.0, 11.0), geometry)


def _state() -> tuple[SensorObservationState, np.ndarray]:
    canvas = _small_canvas()
    elevation = np.arange(121, dtype=np.float32).reshape(11, 11)
    obstacle = np.zeros((11, 11), dtype=np.float32)
    obstacle[5, 7] = 0.01
    elevation_variance = np.full((11, 11), 0.25, dtype=np.float32)
    obstacle_variance = np.full((11, 11), 0.125, dtype=np.float32)
    truth = TrainingWorldTruth(
        canvas,
        elevation,
        obstacle,
        elevation_variance,
        obstacle_variance,
    )
    observed = TrainingObservedGrid.empty(canvas)
    observed.valid_mask[0, 0] = True
    observed.elevation_m[0, 0] = elevation[0, 0]
    observed.observation_quality[0, 0] = 0.5
    observed.observation_count[0, 0] = 1
    roi = np.ones((11, 11), dtype=np.float32)
    priority = np.full((11, 11), 0.5, dtype=np.float32)
    forbidden = np.zeros((11, 11), dtype=np.bool_)
    forbidden[5, 6] = True
    estimator = NativeVisibilityEstimator(
        SensorGeometry(4.0, 2.0 * np.pi),
        resolution_m=1.0,
    )
    return (
        SensorObservationState(
            truth=truth,
            observed=observed,
            mission_roi_ratio=roi,
            mission_priority=priority,
            forbidden_mask=forbidden,
            visibility_estimator=estimator,
        ),
        forbidden,
    )


def test_observe_copies_truth_without_forbidden_occlusion_and_updates_quality() -> None:
    state, forbidden = _state()

    delta = state.observe((5, 5), elapsed_s=2.0)

    assert forbidden[5, 6]
    assert state.observed.valid_mask[5, 6]
    assert state.observed.valid_mask[5, 7]
    assert not state.observed.valid_mask[5, 8]
    assert state.observed.physical_obstacle_ratio[5, 7] > 0.0
    assert state.observed.elevation_m[5, 7] == state.truth.elevation_m[5, 7]
    assert state.observed.observation_quality[5, 7] == 1.0
    assert state.observed.observation_age_s[5, 7] == 0.0
    assert state.observed.elevation_variance[5, 7] == 0.25
    assert state.observed.obstacle_variance[5, 7] == 0.125
    assert state.observed.observation_age_s[0, 0] == 2.0
    assert delta.newly_observed_cells > 0
    assert delta.mission_observed_delta_m2 == pytest.approx(
        float(delta.newly_observed_cells)
    )
    assert delta.priority_observed_delta_m2 == pytest.approx(
        0.5 * delta.newly_observed_cells
    )


def test_repeated_observation_has_zero_coverage_delta_and_saturates_count() -> None:
    state, _ = _state()
    first = state.observe((5, 5), elapsed_s=0.0)
    assert first.newly_observed_cells > 0
    state.observed.observation_count[5, 5] = np.iinfo(np.uint32).max

    repeated = state.observe((5, 5), elapsed_s=1.0)

    assert repeated.newly_observed_cells == 0
    assert repeated.mission_observed_delta_m2 == 0.0
    assert repeated.priority_observed_delta_m2 == 0.0
    assert state.observed.observation_count[5, 5] == np.iinfo(np.uint32).max


def test_observation_errors_are_fail_closed_before_partial_mutation() -> None:
    state, _ = _state()
    before = state.observed.copy()

    with pytest.raises(ValueError, match="outside"):
        state.observe((11, 0), elapsed_s=1.0)
    with pytest.raises(ValueError, match="elapsed"):
        state.observe((5, 5), elapsed_s=-1.0)

    np.testing.assert_array_equal(
        state.observed.valid_mask, before.valid_mask
    )
    np.testing.assert_array_equal(
        state.observed.observation_count, before.observation_count
    )


def test_truth_is_immutable_and_observed_grid_owns_its_input_arrays() -> None:
    canvas = _small_canvas()
    elevation = np.zeros((11, 11), dtype=np.float32)
    obstacle = np.zeros((11, 11), dtype=np.float32)
    elevation_variance = np.full((11, 11), 0.25, dtype=np.float32)
    obstacle_variance = np.full((11, 11), 0.125, dtype=np.float32)
    truth = TrainingWorldTruth(
        canvas,
        elevation,
        obstacle,
        elevation_variance,
        obstacle_variance,
    )
    elevation[0, 0] = 7.0
    elevation_variance[0, 0] = 9.0

    assert truth.elevation_m[0, 0] == 0.0
    assert truth.elevation_variance[0, 0] == 0.25
    with pytest.raises(ValueError, match="read-only"):
        truth.physical_obstacle_ratio[0, 0] = 1.0

    input_elevation = np.zeros((11, 11), dtype=np.float32)
    observed = TrainingObservedGrid(
        canvas=canvas,
        elevation_m=input_elevation,
        physical_obstacle_ratio=np.zeros((11, 11), dtype=np.float32),
        valid_mask=np.zeros((11, 11), dtype=np.bool_),
        observation_age_s=np.zeros((11, 11), dtype=np.float32),
        observation_quality=np.zeros((11, 11), dtype=np.float32),
        elevation_variance=np.zeros((11, 11), dtype=np.float32),
        obstacle_variance=np.zeros((11, 11), dtype=np.float32),
        observation_count=np.zeros((11, 11), dtype=np.uint32),
    )
    input_elevation[0, 0] = 11.0
    assert observed.elevation_m[0, 0] == 0.0


def test_state_rejects_canvas_and_estimator_geometry_mismatch() -> None:
    canvas = _small_canvas()
    truth = TrainingWorldTruth(
        canvas,
        np.zeros((11, 11), np.float32),
        np.zeros((11, 11), np.float32),
    )
    wrong_observed = TrainingObservedGrid.empty(_small_canvas("b" * 64))
    wrong_resolution = NativeVisibilityEstimator(
        SensorGeometry(4.0, 2.0 * np.pi),
        resolution_m=0.5,
    )

    with pytest.raises(ValueError, match="canvas"):
        SensorObservationState(
            truth=truth,
            observed=wrong_observed,
            mission_roi_ratio=np.ones((11, 11), np.float32),
            mission_priority=np.ones((11, 11), np.float32),
            forbidden_mask=np.zeros((11, 11), bool),
            visibility_estimator=wrong_resolution,
        )

    with pytest.raises(ValueError, match="resolution"):
        SensorObservationState(
            truth=truth,
            observed=TrainingObservedGrid.empty(canvas),
            mission_roi_ratio=np.ones((11, 11), np.float32),
            mission_priority=np.ones((11, 11), np.float32),
            forbidden_mask=np.zeros((11, 11), bool),
            visibility_estimator=wrong_resolution,
        )


def test_observed_grid_converts_explicitly_to_policy_world() -> None:
    canvas = MapCanvas("c" * 64, (0.0, 0.0, 1024.0, 1024.0), GLOBAL_GEOMETRY)
    observed = TrainingObservedGrid.empty(canvas)
    observed.valid_mask[128, 128] = True
    observed.elevation_m[128, 128] = 3.0
    local = LocalObservation(
        canvas.identity,
        (508.0, 508.0, 516.0, 516.0),
        np.zeros((32, 32), np.float32),
        np.ones((32, 32), bool),
        np.zeros((32, 32), np.float32),
    )

    world = observed.to_observed_world(local=local)

    assert isinstance(world, ObservedWorld)
    assert world.observed_mask[128, 128]
    assert world.elevation_m[128, 128] == 3.0

from __future__ import annotations

import math

import numpy as np
import pytest

from lunar_policy_training.environment.multires_observation import (
    MultiresSensorObservationState,
)
from lunar_policy_training.environment.coverability import (
    mask_sha256,
    pack_detail_mask,
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


def _state(
    coverable_detail_mask: np.ndarray | None = None,
) -> tuple[MultiresSensorObservationState, SceneTileProvider]:
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
    coverability = {}
    if coverable_detail_mask is not None:
        coverability = {
            "coverable_detail_shape": coverable_detail_mask.shape,
            "coverable_detail_bits": pack_detail_mask(coverable_detail_mask),
            "coverable_detail_cell_count": int(
                coverable_detail_mask.sum(dtype=np.int64)
            ),
            "coverable_mask_sha256": mask_sha256(coverable_detail_mask),
        }
    state = MultiresSensorObservationState(
        scene=scene,
        tile_provider=provider,
        mission_roi_ratio=np.ones((256, 256), np.float32),
        mission_priority=np.full((256, 256), 0.5, np.float32),
        **coverability,
    )
    return state, provider


def _authoritative_observation_bytes(
    state: MultiresSensorObservationState,
) -> tuple[object, ...]:
    detail_fields = (
        "elevation_m",
        "physical_obstacle_ratio",
        "physical_obstacle_height_m",
        "forbidden_ratio",
        "valid_mask",
        "observation_age_s",
        "observation_quality",
        "observation_count",
    )
    coarse_fields = (
        "elevation_m",
        "physical_obstacle_ratio",
        "valid_mask",
        "observation_age_s",
        "observation_quality",
        "elevation_variance",
        "obstacle_variance",
        "observation_count",
    )
    return (
        state.evidence_generation,
        state.physical_evidence_sha256(),
        state.observed_coverable_detail_cell_count,
        tuple(
            (key, *(getattr(tile, name).tobytes() for name in detail_fields))
            for key, tile in sorted(state._detail_tiles.items())
        ),
        *(getattr(state.observed, name).tobytes() for name in coarse_fields),
        state.coarse_obstacle_height_m.tobytes(),
        state.coarse_forbidden_ratio.tobytes(),
    )


def test_exact_coverable_bits_not_coarse_roi_define_coverage_delta() -> None:
    coverable = np.zeros((5120, 5120), dtype=np.bool_)
    coverable[2560, 2560:2562] = True
    state, _ = _state(coverable)

    delta = state.observe_world(
        Pose2(512.0, 512.0, elevation_m=7.0), elapsed_s=0.0
    )

    assert delta.newly_observed_cells > 2
    assert delta.mission_observed_delta_m2 == pytest.approx(0.08)
    assert state.coverable_detail_cell_count == 2
    assert state.observed_coverable_detail_cell_count == 2
    assert state.remaining_coverable_detail_cell_count == 0


def test_observed_obstacle_evidence_never_adds_coverage_outside_mask() -> None:
    coverable = np.zeros((5120, 5120), dtype=np.bool_)
    coverable[2560, 2560] = True
    state, _ = _state(coverable)

    delta = state.observe_world(
        Pose2(512.0, 512.0, elevation_m=7.0), elapsed_s=0.0
    )

    assert state.detail_observed_at(521.2, 512.0)
    assert delta.mission_observed_delta_m2 == pytest.approx(0.04)


def test_coverability_truth_does_not_change_observed_candidate_gain_inputs() -> None:
    coverable = np.zeros((5120, 5120), dtype=np.bool_)
    coverable[2560, 2560] = True
    first, _ = _state(coverable)
    coverable[2560, 2560] = False
    coverable[3000, 3000] = True
    second, _ = _state(coverable)
    pose = Pose2(512.0, 512.0, elevation_m=7.0)
    first.observe_world(pose, elapsed_s=0.0)
    second.observe_world(pose, elapsed_s=0.0)
    candidates = np.ascontiguousarray([[127, 127], [128, 128]], np.int32)

    first_gain = first.estimate_candidate_gains(
        first.observed.valid_mask,
        first.observed.physical_obstacle_ratio,
        first.mission_roi_ratio,
        first.mission_priority,
        candidates,
    )
    second_gain = second.estimate_candidate_gains(
        second.observed.valid_mask,
        second.observed.physical_obstacle_ratio,
        second.mission_roi_ratio,
        second.mission_priority,
        candidates,
    )

    np.testing.assert_array_equal(first.observed.valid_mask, second.observed.valid_mask)
    np.testing.assert_array_equal(first_gain, second_gain)


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


def test_full_detail_block_gates_stable_four_metre_planning_validity() -> None:
    state, _ = _state()
    state.observe_world(Pose2(512.0, 512.0, elevation_m=7.0), elapsed_s=0.0)

    pose_row, pose_column = state.observed.canvas.world_to_grid(512.0, 512.0)
    assert state.observed.valid_mask[pose_row, pose_column]
    assert state.detail_block_valid(pose_row, pose_column).all()

    partial_cells = []
    central_only_cells = []
    for row in range(max(0, pose_row - 9), min(256, pose_row + 10)):
        for column in range(
            max(0, pose_column - 9), min(256, pose_column + 10)
        ):
            block = state.detail_block_valid(row, column)
            if block.any() and not block.all():
                partial_cells.append((row, column))
            if block[9:11, 9:11].all() and not block.all():
                central_only_cells.append((row, column))
    assert partial_cells
    assert central_only_cells
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


def test_local_crop_pads_task_edge_without_exposing_outside_truth() -> None:
    state, _ = _state()
    state.observe_world(
        Pose2(20.0, 1004.0, elevation_m=7.0), elapsed_s=0.0
    )

    local = state.local_observation(
        Pose2(1.0, 1023.0, elevation_m=7.0)
    )

    assert local.bounds_m == pytest.approx((-2.2, 1019.8, 4.2, 1026.2))
    assert local.elevation_m.shape == (32, 32)
    assert local.observed_mask.any()
    assert not local.observed_mask[:11].any()
    assert not local.observed_mask[:, :11].any()
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


def test_evidence_generation_and_digest_change_only_on_authoritative_sensor_updates() -> None:
    state, _ = _state()
    pose = Pose2(512.0, 512.0, elevation_m=7.0)
    initial_digest = state.physical_evidence_sha256()

    assert state.evidence_generation == 0
    assert len(initial_digest) == 64
    state.local_observation(pose)
    state.planning_observation(pose)
    assert state.evidence_generation == 0
    assert state.physical_evidence_sha256() == initial_digest

    state.observe_world(pose, elapsed_s=0.0)
    first_digest = state.physical_evidence_sha256()
    assert state.evidence_generation == 1
    assert first_digest != initial_digest

    state.planning_observation(pose)
    assert state.evidence_generation == 1
    assert state.physical_evidence_sha256() == first_digest

    state.observe_world(pose, elapsed_s=1.0)
    assert state.evidence_generation == 2
    assert state.physical_evidence_sha256() != first_digest


def test_same_cell_batch_preserves_multires_authoritative_updates() -> None:
    sequential, _ = _state()
    batched, _ = _state()
    pose = Pose2(512.0, 512.0, elevation_m=7.0)
    elapsed_steps_s = (0.0, 0.25, 1.0)
    delegate = batched.visibility_estimator
    reveal_calls = 0

    class CountingDetailEstimator:
        sensor = delegate.sensor
        resolution_m = delegate.resolution_m

        @staticmethod
        def reveal_from_pose(
            physical_obstacle_ratio: np.ndarray,
            pose_cell: tuple[int, int],
        ) -> np.ndarray:
            nonlocal reveal_calls
            reveal_calls += 1
            return delegate.reveal_from_pose(
                physical_obstacle_ratio,
                pose_cell,
            )

    batched.visibility_estimator = CountingDetailEstimator()

    sequential_deltas = tuple(
        sequential.observe_world(pose, elapsed_s=elapsed_s)
        for elapsed_s in elapsed_steps_s
    )
    batch_delta = batched.observe_world_repeated(
        pose, elapsed_steps_s=elapsed_steps_s
    )

    assert batched.evidence_generation == len(elapsed_steps_s)
    assert reveal_calls == 1
    assert _authoritative_observation_bytes(batched) == (
        _authoritative_observation_bytes(sequential)
    )
    assert batch_delta == type(batch_delta)(
        visible_cells=sum(delta.visible_cells for delta in sequential_deltas),
        newly_observed_cells=sum(
            delta.newly_observed_cells for delta in sequential_deltas
        ),
        mission_observed_delta_m2=sum(
            delta.mission_observed_delta_m2 for delta in sequential_deltas
        ),
        priority_observed_delta_m2=sum(
            delta.priority_observed_delta_m2 for delta in sequential_deltas
        ),
    )


def test_ground_path_batch_preserves_sequential_authoritative_updates() -> None:
    sequential, _ = _state()
    batched, _ = _state()
    poses = (
        Pose2(512.0, 512.0, elevation_m=7.0),
        Pose2(512.0, 512.0, elevation_m=7.0),
        Pose2(513.0, 512.0, elevation_m=7.0),
        Pose2(514.0, 512.0, elevation_m=7.0),
    )
    elapsed_steps_s = (0.0, 0.25, 0.5, 1.0)
    delegate = batched.visibility_estimator
    reveal_calls = 0

    class CountingDetailEstimator:
        sensor = delegate.sensor
        resolution_m = delegate.resolution_m

        @staticmethod
        def reveal_from_pose(
            physical_obstacle_ratio: np.ndarray,
            pose_cell: tuple[int, int],
        ) -> np.ndarray:
            nonlocal reveal_calls
            reveal_calls += 1
            return delegate.reveal_from_pose(
                physical_obstacle_ratio,
                pose_cell,
            )

    batched.visibility_estimator = CountingDetailEstimator()
    sequential_deltas = tuple(
        sequential.observe_world(pose, elapsed_s=elapsed_s)
        for pose, elapsed_s in zip(poses, elapsed_steps_s, strict=True)
    )

    batch_delta = batched.observe_world_path(
        tuple(zip(poses, elapsed_steps_s, strict=True))
    )

    assert reveal_calls == 3
    assert batched.evidence_generation == len(poses)
    assert _authoritative_observation_bytes(batched) == (
        _authoritative_observation_bytes(sequential)
    )
    assert batch_delta == type(batch_delta)(
        visible_cells=sum(delta.visible_cells for delta in sequential_deltas),
        newly_observed_cells=sum(
            delta.newly_observed_cells for delta in sequential_deltas
        ),
        mission_observed_delta_m2=sum(
            delta.mission_observed_delta_m2 for delta in sequential_deltas
        ),
        priority_observed_delta_m2=sum(
            delta.priority_observed_delta_m2 for delta in sequential_deltas
        ),
    )


def test_ground_trajectory_deduplicates_nonconsecutive_poses_in_one_native_batch() -> None:
    sequential, _ = _state()
    batched, _ = _state()
    poses = (
        Pose2(512.0, 512.0, elevation_m=7.0),
        Pose2(513.0, 512.0, elevation_m=7.0),
        Pose2(512.0, 512.0, elevation_m=7.0),
        Pose2(514.0, 512.0, elevation_m=7.0),
    )
    elapsed_steps_s = (0.0, 0.25, 0.5, 1.0)
    delegate = batched.visibility_estimator
    batch_sizes: list[int] = []

    class BatchCountingDetailEstimator:
        sensor = delegate.sensor
        resolution_m = delegate.resolution_m

        @staticmethod
        def reveal_from_pose(*args, **kwargs) -> np.ndarray:
            del args, kwargs
            raise AssertionError("trajectory must use the native batch reveal")

        @staticmethod
        def reveal_from_poses(
            truth_obstacle_ratios: np.ndarray,
            pose_cells: np.ndarray,
        ) -> np.ndarray:
            batch_sizes.append(int(truth_obstacle_ratios.shape[0]))
            return delegate.reveal_from_poses(
                truth_obstacle_ratios,
                pose_cells,
            )

    batched.visibility_estimator = BatchCountingDetailEstimator()
    path = tuple(zip(poses, elapsed_steps_s, strict=True))
    sequential_deltas = tuple(
        sequential.observe_world(pose, elapsed_s=elapsed_s)
        for pose, elapsed_s in path
    )

    batch_delta = batched.observe_ground_trajectory(path)

    assert batch_sizes == [3]
    assert batched.evidence_generation == len(path)
    assert _authoritative_observation_bytes(batched) == (
        _authoritative_observation_bytes(sequential)
    )
    assert batch_delta == type(batch_delta)(
        visible_cells=sum(delta.visible_cells for delta in sequential_deltas),
        newly_observed_cells=sum(
            delta.newly_observed_cells for delta in sequential_deltas
        ),
        mission_observed_delta_m2=sum(
            delta.mission_observed_delta_m2 for delta in sequential_deltas
        ),
        priority_observed_delta_m2=sum(
            delta.priority_observed_delta_m2 for delta in sequential_deltas
        ),
    )


def test_ground_path_batch_is_atomic_when_later_visibility_fails() -> None:
    state, _ = _state()
    center = Pose2(512.0, 512.0, elevation_m=7.0)
    state.observe_world(center, elapsed_s=0.0)
    before = _authoritative_observation_bytes(state)
    delegate = state.visibility_estimator
    reveal_calls = 0

    class FailingSecondReveal:
        sensor = delegate.sensor
        resolution_m = delegate.resolution_m

        @staticmethod
        def reveal_from_pose(
            physical_obstacle_ratio: np.ndarray,
            pose_cell: tuple[int, int],
        ) -> np.ndarray:
            nonlocal reveal_calls
            reveal_calls += 1
            if reveal_calls == 2:
                raise RuntimeError("injected second path reveal failure")
            return delegate.reveal_from_pose(
                physical_obstacle_ratio,
                pose_cell,
            )

    state.visibility_estimator = FailingSecondReveal()

    with pytest.raises(RuntimeError, match="second path reveal"):
        state.observe_world_path(
            (
                (Pose2(513.0, 512.0, elevation_m=7.0), 0.5),
                (Pose2(514.0, 512.0, elevation_m=7.0), 1.0),
            )
        )

    assert _authoritative_observation_bytes(state) == before


def test_ground_path_lazy_aging_matches_remote_tile_history() -> None:
    sequential, _ = _state()
    batched, _ = _state()
    history = (
        Pose2(128.0, 128.0, elevation_m=7.0),
        Pose2(512.0, 512.0, elevation_m=7.0),
        Pose2(896.0, 896.0, elevation_m=7.0),
    )
    for state in (sequential, batched):
        for pose in history:
            state.observe_world(pose, elapsed_s=0.125)
    path = (
        (Pose2(512.0, 512.0, elevation_m=7.0), 0.25),
        (Pose2(513.0, 512.0, elevation_m=7.0), 0.5),
        (Pose2(514.0, 513.0, elevation_m=7.0), 0.75),
    )

    sequential_deltas = tuple(
        sequential.observe_world(pose, elapsed_s=elapsed_s)
        for pose, elapsed_s in path
    )
    batch_delta = batched.observe_world_path(path)

    assert _authoritative_observation_bytes(batched) == (
        _authoritative_observation_bytes(sequential)
    )
    assert batch_delta == type(batch_delta)(
        visible_cells=sum(delta.visible_cells for delta in sequential_deltas),
        newly_observed_cells=sum(
            delta.newly_observed_cells for delta in sequential_deltas
        ),
        mission_observed_delta_m2=sum(
            delta.mission_observed_delta_m2 for delta in sequential_deltas
        ),
        priority_observed_delta_m2=sum(
            delta.priority_observed_delta_m2 for delta in sequential_deltas
        ),
    )


def test_cross_path_age_stays_lazy_until_the_remote_planning_window_reads_it() -> None:
    eager, _ = _state()
    lazy, lazy_provider = _state()
    remote = Pose2(128.0, 128.0, elevation_m=7.0)
    current = Pose2(512.0, 512.0, elevation_m=7.0)

    def observe_eager(
        state: MultiresSensorObservationState,
        pose: Pose2,
        elapsed_s: float,
    ) -> None:
        state._apply_prepared_detail_observation(
            state._prepare_detail_observation(pose), elapsed_s=elapsed_s
        )

    observe_eager(eager, remote, 0.0)
    lazy.observe_ground_trajectory(((remote, 0.0),))
    remote_row, remote_column = lazy_provider.world_to_detail(
        remote.x_m, remote.y_m
    )
    tile_cells = lazy_provider.tile_geometry.cells
    remote_identity = remote_row // tile_cells, remote_column // tile_cells
    before_remote_age = lazy._detail_tiles[remote_identity].observation_age_s.tobytes()

    observe_eager(eager, current, 1.0)
    lazy.observe_ground_trajectory(((current, 1.0),))

    assert lazy._detail_tiles[remote_identity].observation_age_s.tobytes() == (
        before_remote_age
    )

    lazy.planning_observation(remote)

    assert lazy._detail_tiles[remote_identity].observation_age_s.tobytes() != (
        before_remote_age
    )
    assert _authoritative_observation_bytes(lazy) == (
        _authoritative_observation_bytes(eager)
    )


def test_rolling_evidence_identity_changes_without_materializing_remote_age() -> None:
    state, provider = _state()
    remote = Pose2(128.0, 128.0, elevation_m=7.0)
    current = Pose2(512.0, 512.0, elevation_m=7.0)
    state.observe_ground_trajectory(((remote, 0.0),))
    remote_row, remote_column = provider.world_to_detail(
        remote.x_m, remote.y_m
    )
    tile_cells = provider.tile_geometry.cells
    remote_identity = remote_row // tile_cells, remote_column // tile_cells
    before_identity = state.physical_evidence_identity_sha256()
    before_remote_age = state._detail_tiles[remote_identity].observation_age_s.tobytes()

    state.observe_ground_trajectory(((current, 1.0),))

    assert state.physical_evidence_identity_sha256() != before_identity
    assert state._observation_age_ledger
    assert state._detail_tiles[remote_identity].observation_age_s.tobytes() == (
        before_remote_age
    )


def test_failed_observation_does_not_advance_evidence_generation() -> None:
    state, _ = _state()
    before = state.physical_evidence_sha256()

    with pytest.raises(ValueError, match="detail window"):
        state.observe_world(Pose2(1.0, 1.0), elapsed_s=0.0)

    assert state.evidence_generation == 0
    assert state.physical_evidence_sha256() == before


@pytest.mark.parametrize("failure_stage", ("detail_window", "reveal"))
def test_failed_observation_is_atomic_after_detail_state_exists(
    failure_stage: str,
) -> None:
    state, _ = _state()
    center = Pose2(512.0, 512.0, elevation_m=7.0)
    state.observe_world(center, elapsed_s=0.0)
    before = _authoritative_observation_bytes(state)

    if failure_stage == "detail_window":
        failed_pose = Pose2(1.0, 1.0, elevation_m=7.0)
        expected_error = ValueError
    else:
        failed_pose = center
        expected_error = RuntimeError

        class FailingReveal:
            sensor = state.visibility_estimator.sensor
            resolution_m = state.visibility_estimator.resolution_m

            @staticmethod
            def reveal_from_pose(*args, **kwargs):
                del args, kwargs
                raise RuntimeError("injected reveal failure")

        state.visibility_estimator = FailingReveal()

    with pytest.raises(expected_error):
        state.observe_world(failed_pose, elapsed_s=2.0)

    assert _authoritative_observation_bytes(state) == before


def test_physical_evidence_digest_uses_sorted_tile_identity_and_observed_bytes() -> None:
    first, _ = _state()
    second, _ = _state()
    low = Pose2(400.0, 400.0, elevation_m=7.0)
    high = Pose2(624.0, 624.0, elevation_m=7.0)

    first.observe_world(low, elapsed_s=0.0)
    first.observe_world(high, elapsed_s=0.0)
    second.observe_world(high, elapsed_s=0.0)
    second.observe_world(low, elapsed_s=0.0)

    assert first.evidence_generation == second.evidence_generation == 2
    assert first.physical_evidence_sha256() == second.physical_evidence_sha256()

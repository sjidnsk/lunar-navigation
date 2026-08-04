from __future__ import annotations

import pathlib
import sys

import numpy as np


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from lunar_policy_training.environment.candidate_builder import CandidateBatch, CandidateBuilderV2  # noqa: E402
from lunar_policy_training.environment.observation_builder import MissionRaster, ObservedWorld, PlatformProjection, Pose2  # noqa: E402


def _world_with_frontier(*, occlude: bool = False) -> ObservedWorld:
    observed = np.zeros((256, 256), dtype=bool)
    observed[120:136, 100:128] = True
    if occlude:
        observed[128, 110:120] = False
    return ObservedWorld(
        elevation_m=np.zeros((256, 256), dtype=np.float32),
        observed_mask=observed,
        physical_obstacle_ratio=np.zeros((256, 256), dtype=np.float32),
    )


def _mission() -> MissionRaster:
    roi = np.zeros((256, 256), dtype=np.float32)
    roi[116:140, 96:144] = 1.0
    return MissionRaster(priority=roi.copy(), roi_ratio=roi, remaining_decision_budget_ratio=1.0)


def _projection() -> PlatformProjection:
    return PlatformProjection(
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
    mission = MissionRaster(priority=mission.priority, roi_ratio=mission.roi_ratio * 0.1, remaining_decision_budget_ratio=1.0)
    batch = CandidateBuilderV2().build(_world_with_frontier(), mission, Pose2(450.0, 512.0), _projection())
    valid = batch.features[batch.mask]
    assert np.all(valid[:, 5] <= 1.0)
    assert np.all(valid[:, 6] <= 1.0)
    assert np.all(valid[:, 11] <= 1.0)


def test_candidate_builder_returns_all_false_instead_of_robot_fallback_when_los_has_no_frontier() -> None:
    observed = np.ones((256, 256), dtype=bool)
    world = ObservedWorld(
        elevation_m=np.zeros((256, 256), dtype=np.float32), observed_mask=observed,
        physical_obstacle_ratio=np.zeros((256, 256), dtype=np.float32),
    )
    batch = CandidateBuilderV2().build(world, _mission(), Pose2(512.0, 512.0), _projection())
    assert batch.count == 0
    np.testing.assert_array_equal(batch.features, CandidateBatch.empty().features)
    np.testing.assert_array_equal(batch.mask, CandidateBatch.empty().mask)

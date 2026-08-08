from __future__ import annotations

import numpy as np
import pytest

from lunar_policy_training.environment.visibility import (
    NativeVisibilityEstimator,
    SensorGeometry,
    SlowVisibilityReference,
)


@pytest.mark.parametrize("seed", range(8))
def test_native_visibility_exactly_matches_slow_bresenham_reference(
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    sensor = SensorGeometry(range_m=4.0, fov_rad=2.0 * np.pi)
    native = NativeVisibilityEstimator(sensor, resolution_m=1.0)
    slow = SlowVisibilityReference(
        sensor,
        resolution_m=1.0,
        test_only=True,
    )
    observed = np.ascontiguousarray(rng.random((11, 13)) > 0.35)
    obstacle = np.ascontiguousarray(
        ((rng.random((11, 13)) > 0.9) & observed).astype(np.float32)
    )
    roi = np.ascontiguousarray(rng.random((11, 13)).astype(np.float32))
    priority = np.ascontiguousarray(
        (rng.random((11, 13)).astype(np.float32) * roi)
    )
    candidates = np.asarray(((0, 0), (5, 6), (10, 12), (3, 9)), np.int32)
    observed[tuple(candidates.T)] = True
    obstacle[tuple(candidates.T)] = 0.0

    native_gains = native.estimate_candidate_gains(
        observed, obstacle, roi, priority, candidates
    )
    slow_gains = slow.estimate_candidate_gains(
        observed, obstacle, roi, priority, candidates
    )
    truth_obstacle = np.ascontiguousarray(
        (rng.random((11, 13)) > 0.88).astype(np.float32)
    )
    pose = (seed % 11, (seed * 3) % 13)

    np.testing.assert_allclose(native_gains, slow_gains, rtol=0.0, atol=1e-7)
    np.testing.assert_array_equal(
        native.reveal_from_pose(truth_obstacle, pose),
        slow.reveal_from_pose(truth_obstacle, pose),
    )


def test_slow_visibility_reference_requires_explicit_test_only_opt_in() -> None:
    with pytest.raises(ValueError, match="test-only"):
        SlowVisibilityReference(
            SensorGeometry(30.0, 2.0 * np.pi),
            resolution_m=0.2,
        )

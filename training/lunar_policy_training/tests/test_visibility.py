from __future__ import annotations

import pathlib
import sys

import numpy as np


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from lunar_policy_training.environment.visibility import (
    SensorGeometry,
    SlowVisibilityReference,
)


def test_predicted_gain_sees_unknown_roi_until_a_known_obstacle() -> None:
    estimator = SlowVisibilityReference(
        SensorGeometry(range_m=6.0, fov_rad=2.0 * np.pi),
        resolution_m=1.0,
        test_only=True,
    )
    observed = np.zeros((7, 7), dtype=np.bool_)
    obstacle = np.zeros((7, 7), dtype=np.float32)
    roi = np.zeros((7, 7), dtype=np.float32)
    priority = np.zeros((7, 7), dtype=np.float32)
    candidates = np.asarray([[3, 0]], dtype=np.int32)
    observed[3, 0] = True
    roi[3, 4], roi[3, 6] = 2.0, 8.0
    priority[3, 4], priority[3, 6] = 3.0, 5.0

    optimistic = estimator.estimate_candidate_gains(
        observed, obstacle, roi, priority, candidates
    )

    np.testing.assert_array_equal(optimistic, [[10.0, 8.0]])

    observed[3, 2] = True
    obstacle[3, 2] = 1.0
    blocked = estimator.estimate_candidate_gains(
        observed, obstacle, roi, priority, candidates
    )
    np.testing.assert_array_equal(blocked, [[0.0, 0.0]])

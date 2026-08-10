from __future__ import annotations

import math

import numpy as np
import pytest

from lunar_exploration_policy.action_selection import (
    DeterministicPolicy,
    PolicyDecisionError,
    select_action,
)


def _outputs() -> dict[str, np.ndarray]:
    logits = np.full((1, 64), -5.0, np.float32)
    logits[0, 1] = 3.0
    logits[0, 7] = 99.0
    theta = np.zeros((1, 64), np.float32)
    theta[0, 1] = 0.75
    return {
        "frontier_logits": logits,
        "theta_mu": theta,
        "theta_kappa": np.ones((1, 64), np.float32),
        "value": np.asarray([0.25], np.float32),
    }


def test_masked_argmax_cannot_select_invalid_large_logit() -> None:
    mask = np.zeros((1, 64), np.bool_)
    mask[0, (0, 1)] = True

    action = select_action(_outputs(), mask, "WHEELED")

    assert action.frontier_index == 1
    assert action.theta_rad == pytest.approx(0.75)
    assert action.value == pytest.approx(0.25)


def test_hopper_keeps_fed9_inactive_theta_semantics() -> None:
    mask = np.zeros((1, 64), np.bool_)
    mask[0, 1] = True

    action = select_action(_outputs(), mask, "HOPPER")

    assert action.frontier_index == 1
    assert action.theta_rad is None


class _CountingRuntime:
    def __init__(self) -> None:
        self.calls = 0

    def infer(self, inputs):
        self.calls += 1
        return _outputs()


def test_all_false_mask_bypasses_runtime_instead_of_inferencing() -> None:
    runtime = _CountingRuntime()
    inputs = {"candidate_mask": np.zeros((1, 64), np.bool_)}

    action = DeterministicPolicy(runtime).decide(inputs, "LEGGED")

    assert action is None
    assert runtime.calls == 0


def test_theta_outside_normalized_interval_is_rejected() -> None:
    outputs = _outputs()
    outputs["theta_mu"][0, 1] = math.pi
    mask = np.zeros((1, 64), np.bool_)
    mask[0, 1] = True

    with pytest.raises(PolicyDecisionError, match="theta_mu"):
        select_action(outputs, mask, "LEGGED")

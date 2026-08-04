from __future__ import annotations

import math

import pytest
from lunar_planner_training_bridge import PlanningOutcome

from lunar_policy_training.reward import (
    DEFAULT_REWARD_WEIGHTS,
    InvalidTransition,
    RewardInputs,
    compute_reward,
    reward_weights_sha256,
)


def _inputs(**overrides: object) -> RewardInputs:
    values: dict[str, object] = {
        "platform_type": "WHEELED",
        "coverage_gain": 0.2,
        "goal_progress": 0.3,
        "normalized_plan_cost": 0.4,
        "normalized_elapsed_time": 0.5,
        "repeated_visit": False,
        "planning_outcome": PlanningOutcome.NEW_REFERENCE_AVAILABLE,
    }
    values.update(overrides)
    return RewardInputs(**values)


def test_reward_uses_only_shared_dimensionless_terms() -> None:
    """Would fail if a platform-specific weight or an omitted term changed reward."""
    reward = compute_reward(_inputs(repeated_visit=True))

    assert reward == pytest.approx(
        0.2 * DEFAULT_REWARD_WEIGHTS.coverage_gain
        + 0.3 * DEFAULT_REWARD_WEIGHTS.goal_progress
        - 0.4 * DEFAULT_REWARD_WEIGHTS.normalized_plan_cost
        - 0.5 * DEFAULT_REWARD_WEIGHTS.normalized_elapsed_time
        - DEFAULT_REWARD_WEIGHTS.repeated_visit
    )
    assert {
        reward_weights_sha256(DEFAULT_REWARD_WEIGHTS, platform_type=platform)
        for platform in ("WHEELED", "LEGGED", "HOPPER")
    } == {reward_weights_sha256(DEFAULT_REWARD_WEIGHTS)}


@pytest.mark.parametrize(
    ("outcome", "expected_penalty"),
    [
        (PlanningOutcome.GOAL_INFEASIBLE, "goal_infeasible"),
        (PlanningOutcome.NO_KNOWN_SAFE_ROUTE, "no_known_safe_route"),
        (PlanningOutcome.RESOURCE_EXHAUSTED, "resource_exhausted"),
    ],
)
def test_semantic_planner_failures_are_negative_reward_samples(
    outcome: PlanningOutcome, expected_penalty: str
) -> None:
    """Would fail if a learnable v3 outcome were discarded or left unpenalized."""
    base = compute_reward(_inputs(coverage_gain=0.0, goal_progress=0.0))

    assert compute_reward(
        _inputs(
            coverage_gain=0.0,
            goal_progress=0.0,
            planning_outcome=outcome,
        )
    ) == pytest.approx(base - getattr(DEFAULT_REWARD_WEIGHTS, expected_penalty))


@pytest.mark.parametrize(
    "outcome",
    [
        PlanningOutcome.INVALID_REQUEST,
        PlanningOutcome.STALE_INPUT,
        PlanningOutcome.NUMERICAL_FAILURE,
        PlanningOutcome.CANCELED,
    ],
)
def test_infrastructure_failure_is_not_a_reward_sample(
    outcome: PlanningOutcome,
) -> None:
    """Would fail if invalid infrastructure transitions entered a rollout."""
    with pytest.raises(InvalidTransition):
        compute_reward(_inputs(planning_outcome=outcome))


@pytest.mark.parametrize(
    "field",
    [
        "coverage_gain",
        "goal_progress",
        "normalized_plan_cost",
        "normalized_elapsed_time",
    ],
)
def test_nonfinite_reward_input_is_rejected(field: str) -> None:
    """Would fail if NaN or infinity were converted into a rollout reward."""
    with pytest.raises(InvalidTransition, match="non-finite"):
        compute_reward(_inputs(**{field: math.nan}))


def test_cpp_exception_is_not_a_reward_sample() -> None:
    """Would fail if an exception marker were treated as a policy failure."""
    with pytest.raises(InvalidTransition, match=r"C\+\+ exception"):
        compute_reward(_inputs(cpp_exception="planner threw"))

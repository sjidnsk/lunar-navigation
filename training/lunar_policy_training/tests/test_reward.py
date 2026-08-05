from __future__ import annotations

import math

import pytest
import torch
from lunar_planner_training_bridge import ExecutionDirective, PlanningOutcome

from lunar_policy_training.environment.macro_step import (
    ExecutionEvents,
    PlannerTransition,
)
from lunar_policy_training.policy.observation import PolicyBatch
from lunar_policy_training.reward import (
    DEFAULT_REWARD_WEIGHTS,
    InvalidTransition,
    RewardInputsV2,
    RewardWeightsV2,
    compute_reward,
    reward_weights_sha256,
)


def _inputs(**overrides: object) -> RewardInputsV2:
    values: dict[str, object] = {
        "platform_type": "WHEELED",
        "mission_observed_delta": 0.0,
        "priority_observed_delta": 0.0,
        "normalized_plan_or_execution_cost": 0.0,
        "normalized_macro_step_time": 0.0,
        "executed_without_new_coverage": False,
        "success_first_crossing": False,
        "episode_ended_without_success": False,
        "hard_safety_violation": False,
        "safety_violation_count": 0,
        "platform_reference_mismatch_count": 0,
        "hopper_commitment_violation_count": 0,
        "terminated": False,
        "planning_outcome": PlanningOutcome.NEW_REFERENCE_AVAILABLE,
    }
    values.update(overrides)
    return RewardInputsV2(**values)


def _observation() -> PolicyBatch:
    return PolicyBatch(
        prior_channels=torch.zeros((1, 4, 256, 256), dtype=torch.float32),
        coverage_summary=torch.zeros((1, 3, 256, 256), dtype=torch.float32),
        local_crop=torch.zeros((1, 4, 32, 32), dtype=torch.float32),
        frontier_features=torch.zeros((1, 64, 12), dtype=torch.float32),
        pose_features=torch.zeros((1, 6), dtype=torch.float32),
        candidate_mask=torch.ones((1, 64), dtype=torch.bool),
        platform_context=torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32),
    )


def test_transition_adapter_consumes_only_explicit_v2_facts() -> None:
    """Would fail if PlannerTransition retained legacy proxy reward fields."""
    transition = PlannerTransition(
        next_observation=_observation(),
        mission_observed_delta=0.2,
        priority_observed_delta=0.1,
        normalized_plan_or_execution_cost=0.4,
        normalized_macro_step_time=0.5,
        executed_without_new_coverage=False,
        success_first_crossing=False,
        episode_ended_without_success=False,
        hard_safety_violation=False,
        planning_outcome=PlanningOutcome.NEW_REFERENCE_AVAILABLE,
        execution_directive=ExecutionDirective.ACTIVATE_NEW_REFERENCE,
        reason_code="OK",
        terminated=False,
        execution_events=ExecutionEvents(),
    )

    inputs = RewardInputsV2.from_transition(transition, platform_type="WHEELED")

    assert inputs.mission_observed_delta == 0.2
    assert inputs.priority_observed_delta == 0.1
    assert inputs.normalized_plan_or_execution_cost == 0.4
    assert inputs.normalized_macro_step_time == 0.5


def test_success_bonus_dominates_maximum_dense_positive_shaping_once() -> None:
    """Would fail if dense shaping reached the 50-point success bonus."""
    dense = compute_reward(
        _inputs(mission_observed_delta=1.0, priority_observed_delta=1.0)
    )
    first = compute_reward(
        _inputs(success_first_crossing=True, terminated=True)
    )
    later = compute_reward(_inputs(success_first_crossing=False))

    assert dense == 25.0
    assert first == 50.0
    assert later == 0.0


def test_shared_v2_reward_uses_every_fixed_term_and_one_platform_independent_hash() -> None:
    """Would fail if one V2 term, sign, weight, or platform-specific hash drifted."""
    reward = compute_reward(
        _inputs(
            mission_observed_delta=0.2,
            priority_observed_delta=0.3,
            normalized_plan_or_execution_cost=0.4,
            normalized_macro_step_time=0.5,
            executed_without_new_coverage=True,
        )
    )

    assert reward == pytest.approx(4.0 + 1.5 - 0.04 - 0.025 - 0.20)
    assert isinstance(DEFAULT_REWARD_WEIGHTS, RewardWeightsV2)
    assert {
        reward_weights_sha256(DEFAULT_REWARD_WEIGHTS, platform_type=platform)
        for platform in ("WHEELED", "LEGGED", "HOPPER")
    } == {reward_weights_sha256(DEFAULT_REWARD_WEIGHTS)}


def test_unsuccessful_and_hard_safety_terminals_are_distinct() -> None:
    """Would fail if hard safety did not add 50 to the normal failed terminal."""
    unsuccessful = compute_reward(
        _inputs(episode_ended_without_success=True, terminated=True)
    )
    hard_safety = compute_reward(
        _inputs(
            episode_ended_without_success=True,
            hard_safety_violation=True,
            safety_violation_count=1,
            terminated=True,
        )
    )

    assert unsuccessful == -10.0
    assert hard_safety == -60.0


@pytest.mark.parametrize(
    "event_field",
    [
        "safety_violation_count",
        "platform_reference_mismatch_count",
        "hopper_commitment_violation_count",
    ],
)
def test_each_hard_safety_gate_event_supports_the_hard_terminal_penalty(
    event_field: str,
) -> None:
    """Would fail if reward rewrote distinct safety gate counters into one."""
    assert compute_reward(
        _inputs(
            **{
                event_field: 1,
                "episode_ended_without_success": True,
                "hard_safety_violation": True,
                "terminated": True,
            }
        )
    ) == -60.0


@pytest.mark.parametrize(
    ("outcome", "expected"),
    [
        (PlanningOutcome.GOAL_INFEASIBLE, -2.0),
        (PlanningOutcome.NO_KNOWN_SAFE_ROUTE, -3.0),
        (PlanningOutcome.ACTIVE_REFERENCE_INVALIDATED, -3.0),
        (PlanningOutcome.RESOURCE_EXHAUSTED, -1.5),
    ],
)
def test_learnable_planner_outcome_penalties_are_exact(
    outcome: PlanningOutcome, expected: float
) -> None:
    """Would fail if a semantic v3 rejection were discarded or misweighted."""
    assert compute_reward(_inputs(planning_outcome=outcome)) == expected


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {
                "hard_safety_violation": True,
                "safety_violation_count": 1,
                "success_first_crossing": True,
                "terminated": True,
            },
            "success",
        ),
        (
            {
                "platform_reference_mismatch_count": 1,
                "episode_ended_without_success": True,
                "terminated": True,
            },
            "safety",
        ),
        (
            {
                "hard_safety_violation": True,
                "safety_violation_count": 0,
                "episode_ended_without_success": True,
                "terminated": True,
            },
            "safety",
        ),
        ({"safety_violation_count": 1}, "safety"),
        ({"episode_ended_without_success": True}, "terminal"),
        ({"terminated": True}, "terminal"),
    ],
)
def test_terminal_and_safety_facts_fail_closed(
    overrides: dict[str, object], message: str
) -> None:
    """Would fail if inconsistent terminal or safety facts entered a rollout."""
    with pytest.raises(InvalidTransition, match=message):
        compute_reward(_inputs(**overrides))


@pytest.mark.parametrize(
    "field",
    [
        "executed_without_new_coverage",
        "success_first_crossing",
        "episode_ended_without_success",
        "hard_safety_violation",
        "terminated",
    ],
)
def test_reward_flags_require_exact_booleans(field: str) -> None:
    """Would fail if integer truthiness changed a frozen reward fact."""
    with pytest.raises(InvalidTransition, match="boolean"):
        compute_reward(_inputs(**{field: 1}))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("mission_observed_delta", -0.01),
        ("mission_observed_delta", 1.01),
        ("priority_observed_delta", -0.01),
        ("priority_observed_delta", 1.01),
        ("normalized_plan_or_execution_cost", -0.01),
        ("normalized_macro_step_time", -0.01),
    ],
)
def test_reward_scalar_ranges_fail_closed(field: str, value: float) -> None:
    """Would fail if out-of-contract ratios or negative costs were clipped silently."""
    with pytest.raises(InvalidTransition, match="range|non-negative"):
        compute_reward(_inputs(**{field: value}))


@pytest.mark.parametrize(
    "field",
    [
        "mission_observed_delta",
        "priority_observed_delta",
        "normalized_plan_or_execution_cost",
        "normalized_macro_step_time",
    ],
)
def test_nonfinite_reward_input_is_rejected(field: str) -> None:
    """Would fail if NaN or infinity were converted into a rollout reward."""
    with pytest.raises(InvalidTransition, match="non-finite"):
        compute_reward(_inputs(**{field: math.nan}))


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


def test_cpp_exception_is_not_a_reward_sample() -> None:
    """Would fail if an exception marker were treated as a policy failure."""
    with pytest.raises(InvalidTransition, match=r"C\+\+ exception"):
        compute_reward(_inputs(cpp_exception="planner threw"))

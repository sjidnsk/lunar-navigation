from __future__ import annotations

from dataclasses import MISSING, fields
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
    RewardInputsV4,
    RewardWeightsV4,
    compute_reward,
    reward_weights_sha256,
)


def _inputs(**overrides: object) -> RewardInputsV4:
    values: dict[str, object] = {
        "platform_type": "WHEELED",
        "mission_observed_delta": 0.0,
        "mission_observed_ratio_after": 0.0,
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
        "cancellation_expected": False,
        "cpp_exception": None,
    }
    values.update(overrides)
    return RewardInputsV4(**values)


def _observation(*, mission_observed_ratio: float = 0.0) -> PolicyBatch:
    observation = PolicyBatch(
        prior_channels=torch.zeros((1, 4, 256, 256), dtype=torch.float32),
        coverage_summary=torch.zeros((1, 3, 256, 256), dtype=torch.float32),
        local_crop=torch.zeros((1, 4, 32, 32), dtype=torch.float32),
        frontier_features=torch.zeros((1, 64, 12), dtype=torch.float32),
        pose_features=torch.zeros((1, 5), dtype=torch.float32),
        candidate_mask=torch.ones((1, 64), dtype=torch.bool),
        platform_context=torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32),
    )
    observation.pose_features[0, 4] = mission_observed_ratio
    return observation


def test_transition_adapter_reads_authoritative_v3_coverage() -> None:
    """Would fail if PlannerTransition retained legacy proxy reward fields."""
    transition = PlannerTransition(
        next_observation=_observation(mission_observed_ratio=0.4),
        mission_observed_delta=0.2,
        priority_observed_delta=0.1,
        normalized_plan_or_execution_cost=0.4,
        normalized_macro_step_time=0.5,
        executed_without_new_coverage=False,
        success_first_crossing=False,
        episode_ended_without_success=False,
        hard_safety_violation=False,
        cancellation_expected=False,
        cpp_exception=None,
        planning_outcome=PlanningOutcome.NEW_REFERENCE_AVAILABLE,
        execution_directive=ExecutionDirective.ACTIVATE_NEW_REFERENCE,
        reason_code="OK",
        terminated=False,
        execution_events=ExecutionEvents(),
    )

    inputs = RewardInputsV4.from_transition(transition, platform_type="WHEELED")

    assert inputs.mission_observed_delta == 0.2
    assert inputs.mission_observed_ratio_after == pytest.approx(0.4)
    assert inputs.priority_observed_delta == 0.1
    assert inputs.normalized_plan_or_execution_cost == 0.4
    assert inputs.normalized_macro_step_time == 0.5


def test_v3_ancillary_facts_have_no_implicit_dataclass_defaults() -> None:
    """Would fail if cancellation, exception, or gate evidence became implicit."""
    transition_defaults = {
        field.name: field.default for field in fields(PlannerTransition)
    }
    reward_defaults = {
        field.name: field.default for field in fields(RewardInputsV4)
    }

    assert transition_defaults["cancellation_expected"] is MISSING
    assert transition_defaults["cpp_exception"] is MISSING
    assert transition_defaults["execution_events"] is MISSING
    assert reward_defaults["cancellation_expected"] is MISSING
    assert reward_defaults["cpp_exception"] is MISSING


def test_standard_transition_adapter_preserves_expected_cancellation() -> None:
    """Would fail if expected CANCELED could only use a non-standard reward path."""
    transition = PlannerTransition(
        next_observation=_observation(),
        mission_observed_delta=0.0,
        priority_observed_delta=0.0,
        normalized_plan_or_execution_cost=0.0,
        normalized_macro_step_time=0.0,
        executed_without_new_coverage=False,
        success_first_crossing=False,
        episode_ended_without_success=False,
        hard_safety_violation=False,
        cancellation_expected=True,
        cpp_exception=None,
        planning_outcome=PlanningOutcome.CANCELED,
        execution_directive=ExecutionDirective.HOLD_POSITION,
        reason_code="EXPECTED_CANCEL",
        terminated=False,
        execution_events=ExecutionEvents(),
    )

    inputs = RewardInputsV4.from_transition(transition, platform_type="WHEELED")

    assert inputs.cancellation_expected is True
    assert inputs.cpp_exception is None
    assert compute_reward(inputs) == 0.0


def test_standard_transition_adapter_rejects_cpp_exception_marker() -> None:
    """Would fail if the adapter dropped an explicit C++ exception fact."""
    transition = PlannerTransition(
        next_observation=_observation(),
        mission_observed_delta=0.0,
        priority_observed_delta=0.0,
        normalized_plan_or_execution_cost=0.0,
        normalized_macro_step_time=0.0,
        executed_without_new_coverage=False,
        success_first_crossing=False,
        episode_ended_without_success=False,
        hard_safety_violation=False,
        cancellation_expected=False,
        cpp_exception="planner threw",
        planning_outcome=PlanningOutcome.NEW_REFERENCE_AVAILABLE,
        execution_directive=ExecutionDirective.ACTIVATE_NEW_REFERENCE,
        reason_code="CPP_EXCEPTION",
        terminated=False,
        execution_events=ExecutionEvents(),
    )

    inputs = RewardInputsV4.from_transition(transition, platform_type="WHEELED")

    with pytest.raises(InvalidTransition, match=r"C\+\+ exception"):
        compute_reward(inputs)


def test_continuous_coverage_and_first_success_have_exact_v4_rewards() -> None:
    """Would fail if coverage or the one-time completion bonus drifted."""
    dense = compute_reward(_inputs(mission_observed_delta=0.02))
    first = compute_reward(
        _inputs(
            mission_observed_delta=0.01,
            mission_observed_ratio_after=0.95,
            success_first_crossing=True,
            terminated=True,
        )
    )
    later = compute_reward(_inputs(success_first_crossing=False))

    assert dense == 2.0
    assert first == 101.0
    assert later == -0.10


def test_zero_gain_first_success_restores_the_full_completion_signal() -> None:
    assert compute_reward(
        _inputs(
            mission_observed_delta=0.0,
            mission_observed_ratio_after=0.95,
            success_first_crossing=True,
            terminated=True,
        )
    ) == pytest.approx(99.9)


def test_v4_reward_ignores_telemetry_and_has_one_platform_independent_hash() -> None:
    """Would fail if cost, time, or priority leaked back into policy reward."""
    baseline = compute_reward(
        _inputs(mission_observed_delta=0.2, mission_observed_ratio_after=0.4)
    )
    changed_telemetry = compute_reward(
        _inputs(
            mission_observed_delta=0.2,
            mission_observed_ratio_after=0.4,
            priority_observed_delta=0.3,
            normalized_plan_or_execution_cost=0.4,
            normalized_macro_step_time=0.5,
            executed_without_new_coverage=True,
        )
    )

    assert baseline == 20.0
    assert changed_telemetry == baseline
    assert isinstance(DEFAULT_REWARD_WEIGHTS, RewardWeightsV4)
    assert {
        reward_weights_sha256(DEFAULT_REWARD_WEIGHTS, platform_type=platform)
        for platform in ("WHEELED", "LEGGED", "HOPPER")
    } == {reward_weights_sha256(DEFAULT_REWARD_WEIGHTS)}
    assert reward_weights_sha256() == (
        "57f481799bb99f803c1f890d44b7ce27682bddb6e83370abbff08b1e4a515c15"
    )
    assert reward_weights_sha256() != (
        "46f0400e934113cfb863c4e4b64174027eac667f8654d14b2e5d333657335bd4"
    )


def test_unsuccessful_terminal_penalty_tracks_remaining_coverage() -> None:
    """Would fail if V3 restored a fixed oversized terminal penalty."""
    unsuccessful = compute_reward(
        _inputs(
            mission_observed_ratio_after=0.4,
            episode_ended_without_success=True,
            terminated=True,
        )
    )

    assert unsuccessful == pytest.approx(-0.10 - 0.60)


@pytest.mark.parametrize(
    "event_field",
    [
        "safety_violation_count",
        "platform_reference_mismatch_count",
        "hopper_commitment_violation_count",
    ],
)
def test_each_hard_safety_gate_event_is_not_a_reward_sample(
    event_field: str,
) -> None:
    """Would fail if a hard invariant breach trained the exploration policy."""
    with pytest.raises(InvalidTransition, match="hard safety"):
        compute_reward(
            _inputs(
                **{
                    event_field: 1,
                    "episode_ended_without_success": True,
                    "hard_safety_violation": True,
                    "terminated": True,
                }
            )
        )


@pytest.mark.parametrize(
    ("outcome", "expected"),
    [
        (PlanningOutcome.GOAL_INFEASIBLE, -0.30),
        (PlanningOutcome.NO_KNOWN_SAFE_ROUTE, -0.40),
        (PlanningOutcome.ACTIVE_REFERENCE_INVALIDATED, -0.40),
    ],
)
def test_learnable_planner_outcome_penalties_are_exact(
    outcome: PlanningOutcome, expected: float
) -> None:
    """Would fail if a semantic v3 rejection were discarded or misweighted."""
    assert compute_reward(_inputs(planning_outcome=outcome)) == pytest.approx(expected)


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
        "mission_observed_ratio_after",
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
        PlanningOutcome.RESOURCE_EXHAUSTED,
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

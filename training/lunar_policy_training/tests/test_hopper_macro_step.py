from __future__ import annotations

from datetime import timedelta
import pathlib
import sys
from unittest.mock import Mock


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "model_contract"))

from lunar_planner_training_bridge import (  # noqa: E402
    ExecutionDirective,
    PlannerOutput,
    PlanningOutcome,
)
import pytest  # noqa: E402

from lunar_policy_training.environment.v3_environment import (  # noqa: E402
    CommittedHopExecutionFeedback,
    EnvironmentInvariantError,
    V3ExplorationEnvironment,
)
from lunar_policy_training.environment.macro_step import (  # noqa: E402
    ExecutionEvents,
    PolicyAction,
)
import torch  # noqa: E402

from lunar_policy_training.policy.observation import (  # noqa: E402
    ObservationIdentity,
    PolicyBatch,
)


class _Bridge:
    def __init__(self, output: PlannerOutput) -> None:
        self.output = output

    def plan(self, request: object) -> PlannerOutput:
        return self.output


class _CommittedHopSimulator:
    def __init__(self, *feedback: CommittedHopExecutionFeedback) -> None:
        self._feedback = list(feedback)

    def __call__(self) -> CommittedHopExecutionFeedback:
        if not self._feedback:
            raise AssertionError("unexpected committed-hop simulation advance")
        return self._feedback.pop(0)


def _hopper_observation(execution_state: str = "GROUND_HOLD") -> PolicyBatch:
    return PolicyBatch(
        prior_channels=torch.zeros((1, 7, 4, 4), dtype=torch.float32),
        coverage_summary=torch.zeros((1, 8, 4, 4), dtype=torch.float32),
        local_crop=torch.zeros((1, 8, 4, 4), dtype=torch.float32),
        frontier_features=torch.zeros((1, 2, 22), dtype=torch.float32),
        pose_features=torch.zeros((1, 5), dtype=torch.float32),
        candidate_mask=torch.tensor([[True, False]], dtype=torch.bool),
        platform_context=torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32),
        observation_identities=(
            ObservationIdentity(
                episode_id="hopper-1",
                mission_revision=1,
                map_snapshot_id="map-1",
                robot_state_id="hopper-state-1",
                state_time_ns=1_000,
                execution_state=execution_state,
                candidate_set_id="candidates-1",
            ),
        ),
    )


def _execution_feedback(
    execution_state: str,
    *,
    marker: float,
    mission_observed_delta: float,
    priority_observed_delta: float,
    normalized_execution_cost_contribution: float,
    normalized_execution_time_contribution: float,
    execution_events: ExecutionEvents = ExecutionEvents(),
) -> CommittedHopExecutionFeedback:
    observation = _hopper_observation(execution_state)
    observation.pose_features[0, 0] = marker
    return CommittedHopExecutionFeedback(
        execution_state=execution_state,
        next_observation=observation,
        mission_observed_delta=mission_observed_delta,
        priority_observed_delta=priority_observed_delta,
        normalized_execution_cost_contribution=(
            normalized_execution_cost_contribution
        ),
        normalized_execution_time_contribution=(
            normalized_execution_time_contribution
        ),
        executed_without_new_coverage=False,
        success_first_crossing=False,
        episode_ended_without_success=False,
        hard_safety_violation=False,
        terminated=False,
        execution_events=execution_events,
    )


def test_prepared_hopper_action_aggregates_until_landed_hold() -> None:
    """Would fail if production returned an intermediate committed-hop sample."""
    output = PlannerOutput()
    output.outcome = PlanningOutcome.SAFE_FRONTIER_REFERENCE_AVAILABLE
    output.directive = ExecutionDirective.CONTINUE_COMMITTED_HOP
    output.reason_code = "COMMITTED_HOP_CONTINUES"
    output.diagnostics.best_cost = 2.0
    output.diagnostics.elapsed = timedelta(milliseconds=500)
    feedback = (
        _execution_feedback(
            "JUMP_COMMITTED",
            marker=1.0,
            mission_observed_delta=0.1,
            priority_observed_delta=0.2,
            normalized_execution_cost_contribution=0.1,
            normalized_execution_time_contribution=0.4,
            execution_events=ExecutionEvents(
                reference_samples_consumed=1,
                selected_action_observed_safe=True,
                hopper_commitment_states=("JUMP_COMMITTED",),
            ),
        ),
        _execution_feedback(
            "IN_FLIGHT",
            marker=2.0,
            mission_observed_delta=0.25,
            priority_observed_delta=0.3,
            normalized_execution_cost_contribution=0.2,
            normalized_execution_time_contribution=0.5,
            execution_events=ExecutionEvents(
                invalid_action_count=1,
                reference_samples_consumed=2,
                hopper_commitment_states=("IN_FLIGHT",),
            ),
        ),
        _execution_feedback(
            "LANDED_HOLD",
            marker=3.0,
            mission_observed_delta=0.4,
            priority_observed_delta=0.5,
            normalized_execution_cost_contribution=0.3,
            normalized_execution_time_contribution=0.6,
            execution_events=ExecutionEvents(
                execution_failure_count=2,
                reference_samples_consumed=3,
                hopper_commitment_states=("LANDED_HOLD",),
            ),
        ),
    )
    hopper_env = V3ExplorationEnvironment(
        platform_type="HOPPER",
        bridge=_Bridge(output),
        request_builder=lambda action: action,
        initial_observation=_hopper_observation(),
        committed_hop_executor=_CommittedHopSimulator(*feedback),
        plan_cost_scale=4.0,
        planner_elapsed_scale_s=2.0,
    )
    prepared = hopper_env.current_observation

    result = hopper_env.advance_prepared_action(
        PolicyAction(frontier_index=0, theta_rad=0.0),
        expected_identity=prepared.observation_identities[0],
    )

    assert result.execution_state == "LANDED_HOLD"
    assert result.policy_decisions_consumed == 1
    assert result.transition is not None
    assert result.transition.next_observation.pose_features[0, 0].item() == 3.0
    assert result.transition.mission_observed_delta == pytest.approx(0.75)
    assert result.transition.priority_observed_delta == pytest.approx(1.0)
    assert result.transition.normalized_plan_or_execution_cost == pytest.approx(1.1)
    assert result.transition.normalized_macro_step_time == pytest.approx(1.75)
    assert result.transition.execution_events == ExecutionEvents(
        invalid_action_count=1,
        execution_failure_count=2,
        reference_samples_consumed=6,
        selected_action_observed_safe=True,
        hopper_commitment_states=(
            "JUMP_COMMITTED",
            "IN_FLIGHT",
            "LANDED_HOLD",
        ),
    )
    assert result.transition.success_first_crossing is False
    assert result.transition.episode_ended_without_success is False
    assert result.transition.hard_safety_violation is False


def test_hopper_does_not_request_policy_while_committed() -> None:
    """Would fail if policy could replace a committed or in-flight hop."""
    landed = _execution_feedback(
        "LANDED_HOLD",
        marker=1.0,
        mission_observed_delta=0.1,
        priority_observed_delta=0.2,
        normalized_execution_cost_contribution=0.0,
        normalized_execution_time_contribution=0.0,
    )
    hopper_env = V3ExplorationEnvironment(
        platform_type="HOPPER",
        bridge=Mock(),
        request_builder=Mock(),
        initial_observation=_hopper_observation(),
        committed_hop_executor=_CommittedHopSimulator(landed),
    )
    policy_spy = Mock()

    hopper_env.begin_committed_hop()
    result = hopper_env.advance_until_decision_boundary(policy_spy)

    assert policy_spy.call_count == 0
    assert result.execution_state == "LANDED_HOLD"


def test_hopper_remains_in_flight_without_landed_feedback() -> None:
    """Would fail if repeated committed advances invented a landing boundary."""
    first_feedback = _execution_feedback(
        "IN_FLIGHT",
        marker=1.0,
        mission_observed_delta=0.1,
        priority_observed_delta=0.2,
        normalized_execution_cost_contribution=0.0,
        normalized_execution_time_contribution=0.0,
    )
    second_feedback = _execution_feedback(
        "IN_FLIGHT",
        marker=2.0,
        mission_observed_delta=0.3,
        priority_observed_delta=0.4,
        normalized_execution_cost_contribution=0.0,
        normalized_execution_time_contribution=0.0,
    )
    hopper_env = V3ExplorationEnvironment(
        platform_type="HOPPER",
        bridge=Mock(),
        request_builder=Mock(),
        initial_observation=_hopper_observation(),
        committed_hop_executor=_CommittedHopSimulator(
            first_feedback, second_feedback
        ),
    )
    policy_spy = Mock()

    hopper_env.begin_committed_hop()
    first = hopper_env.advance_until_decision_boundary(policy_spy)
    second = hopper_env.advance_until_decision_boundary(policy_spy)

    assert policy_spy.call_count == 0
    assert first.execution_state == "IN_FLIGHT"
    assert first.execution_feedback is first_feedback
    assert second.execution_state == "IN_FLIGHT"
    assert second.execution_feedback is second_feedback


def test_hopper_rejects_explicit_action_while_committed() -> None:
    """Would fail if callers could bypass the no-policy commitment boundary."""
    hopper_env = V3ExplorationEnvironment(
        platform_type="HOPPER",
        bridge=Mock(),
        request_builder=Mock(),
        initial_observation=_hopper_observation(),
    )
    hopper_env.begin_committed_hop()

    with pytest.raises(EnvironmentInvariantError, match="committed"):
        hopper_env.step(PolicyAction(frontier_index=0, theta_rad=0.0))

    assert hopper_env.rollout_discarded is True
    assert hopper_env.training_stopped is True


def test_production_hopper_factory_requires_execution_feedback() -> None:
    """Would fail if production could compose a hopper with fake landing state."""
    from lunar_policy_training.environment.v3_environment import (
        create_v3_environment,
    )

    with pytest.raises(ValueError, match="committed hop executor"):
        create_v3_environment(
            platform_type="HOPPER",
            request_builder=Mock(),
            initial_observation=_hopper_observation(),
        )


def test_hopper_resumes_policy_only_after_landed_hold() -> None:
    """Would fail if the policy stayed disabled after the landing boundary."""
    output = PlannerOutput()
    output.outcome = PlanningOutcome.NO_KNOWN_SAFE_ROUTE
    output.directive = ExecutionDirective.NO_SAFE_REFERENCE
    output.reason_code = "LANDED_NO_ROUTE"
    landed = _execution_feedback(
        "LANDED_HOLD",
        marker=3.0,
        mission_observed_delta=0.25,
        priority_observed_delta=0.5,
        normalized_execution_cost_contribution=0.0,
        normalized_execution_time_contribution=0.0,
    )
    hopper_env = V3ExplorationEnvironment(
        platform_type="HOPPER",
        bridge=_Bridge(output),
        request_builder=lambda action: action,
        initial_observation=_hopper_observation(),
        committed_hop_executor=_CommittedHopSimulator(landed),
    )
    policy_spy = Mock(return_value=PolicyAction(frontier_index=0, theta_rad=0.0))

    hopper_env.begin_committed_hop()
    landed_result = hopper_env.advance_until_decision_boundary(policy_spy)
    result = hopper_env.advance_until_decision_boundary(policy_spy)

    assert policy_spy.call_count == 1
    policy_observation = policy_spy.call_args.args[0]
    assert policy_observation is not landed.next_observation
    assert policy_observation.pose_features[0, 0].item() == pytest.approx(3.0)
    assert policy_observation.pose_features.shape == (1, 5)
    assert landed_result.execution_state == "LANDED_HOLD"
    assert landed_result.execution_feedback is landed
    assert landed_result.execution_feedback.mission_observed_delta == 0.25
    assert landed_result.execution_feedback.priority_observed_delta == 0.5
    assert result.execution_state == "LANDED_HOLD"
    assert result.transition is not None
    assert result.transition.reason_code == "LANDED_NO_ROUTE"

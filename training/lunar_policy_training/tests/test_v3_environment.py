from __future__ import annotations

from datetime import timedelta
import pathlib
import sys


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "model_contract"))

from lunar_planner_training_bridge import (  # noqa: E402
    ExecutionDirective,
    MotionReference,
    PlannerDiagnostics,
    PlannerOutput,
    PlanningOutcome,
)
import torch  # noqa: E402
import pytest  # noqa: E402

from lunar_policy_training.environment.macro_step import PolicyAction  # noqa: E402
from lunar_policy_training.environment.v3_environment import (  # noqa: E402
    EnvironmentInvariantError,
    ReferenceExecutionResult,
    V3ExplorationEnvironment,
)
from lunar_policy_training.policy.observation import PolicyBatch  # noqa: E402


def _observation(platform_index: int = 0) -> PolicyBatch:
    platform_context = torch.zeros((1, 3), dtype=torch.float32)
    platform_context[0, platform_index] = 1.0
    return PolicyBatch(
        prior_channels=torch.zeros((1, 7, 4, 4), dtype=torch.float32),
        coverage_summary=torch.zeros((1, 8, 4, 4), dtype=torch.float32),
        local_crop=torch.zeros((1, 8, 4, 4), dtype=torch.float32),
        frontier_features=torch.zeros((1, 2, 22), dtype=torch.float32),
        pose_features=torch.zeros((1, 6), dtype=torch.float32),
        candidate_mask=torch.tensor([[True, False]], dtype=torch.bool),
        platform_context=platform_context,
    )


class _Bridge:
    def __init__(self, output: PlannerOutput) -> None:
        self.output = output

    def plan(self, request: object) -> PlannerOutput:
        return self.output


class _ReferenceExecutor:
    def __init__(self, result: ReferenceExecutionResult) -> None:
        self.result = result
        self.references: list[MotionReference] = []

    def __call__(self, reference: MotionReference) -> ReferenceExecutionResult:
        self.references.append(reference)
        return self.result


def _reference_output(platform_type: str, directive: ExecutionDirective) -> PlannerOutput:
    output = PlannerOutput()
    output.outcome = PlanningOutcome.NEW_REFERENCE_AVAILABLE
    output.directive = directive
    output.reason_code = "REFERENCE_READY"
    output.reference = MotionReference()
    output.reference.platform_type = platform_type
    output.diagnostics.best_cost = 2.0
    output.diagnostics.elapsed = timedelta(milliseconds=500)
    return output


def test_no_reference_holds_state_and_preserves_planner_failure() -> None:
    """Would fail if no-route output were converted into a fake trajectory."""
    output = PlannerOutput()
    output.outcome = PlanningOutcome.NO_KNOWN_SAFE_ROUTE
    output.directive = ExecutionDirective.NO_SAFE_REFERENCE
    output.reason_code = "NO_ROUTE"
    output.diagnostics = PlannerDiagnostics()
    output.diagnostics.elapsed = timedelta(milliseconds=250)
    observation = _observation()
    env = V3ExplorationEnvironment(
        platform_type="WHEELED",
        bridge=_Bridge(output),
        request_builder=lambda action: action,
        initial_observation=observation,
    )

    transition = env.step(PolicyAction(frontier_index=0, theta_rad=0.0))

    assert transition.next_observation is not observation
    assert torch.equal(transition.next_observation.prior_channels, observation.prior_channels)
    assert transition.coverage_delta == 0.0
    assert transition.goal_progress == 0.0
    assert transition.normalized_plan_cost == 0.0
    assert transition.normalized_elapsed_time == 0.25
    assert transition.repeated_visit is False
    assert transition.planning_outcome == PlanningOutcome.NO_KNOWN_SAFE_ROUTE
    assert transition.execution_directive == ExecutionDirective.NO_SAFE_REFERENCE
    assert transition.reason_code == "NO_ROUTE"
    assert transition.terminated is False


@pytest.mark.parametrize(
    ("platform_type", "platform_index"), [("WHEELED", 0), ("LEGGED", 1)]
)
def test_ground_reference_executes_to_next_decision_boundary(
    platform_type: str, platform_index: int
) -> None:
    """Would fail if wheel/legged reference execution were replaced by a hold."""
    next_observation = _observation(platform_index)
    next_observation.pose_features[0, 0] = 0.5
    executor = _ReferenceExecutor(
        ReferenceExecutionResult(
            next_observation=next_observation,
            coverage_delta=0.2,
            goal_progress=0.1,
            repeated_visit=True,
            terminated=False,
            execution_state="DECISION_BOUNDARY",
        )
    )
    env = V3ExplorationEnvironment(
        platform_type=platform_type,
        bridge=_Bridge(
            _reference_output(
                platform_type, ExecutionDirective.ACTIVATE_NEW_REFERENCE
            )
        ),
        request_builder=lambda action: action,
        initial_observation=_observation(platform_index),
        reference_executor=executor,
        plan_cost_scale=4.0,
        planner_elapsed_scale_s=2.0,
    )

    transition = env.step(PolicyAction(frontier_index=0, theta_rad=0.25))

    assert len(executor.references) == 1
    assert transition.next_observation is next_observation
    assert transition.coverage_delta == 0.2
    assert transition.goal_progress == 0.1
    assert transition.normalized_plan_cost == 0.5
    assert transition.normalized_elapsed_time == 0.25
    assert transition.repeated_visit is True
    assert transition.planning_outcome == PlanningOutcome.NEW_REFERENCE_AVAILABLE
    assert transition.execution_directive == ExecutionDirective.ACTIVATE_NEW_REFERENCE
    assert transition.reason_code == "REFERENCE_READY"
    assert transition.terminated is False


def test_mismatched_reference_platform_discards_rollout_and_stops_training() -> None:
    """Would fail if a hopper reference could execute in a wheeled episode."""
    executor = _ReferenceExecutor(
        ReferenceExecutionResult(
            next_observation=_observation(),
            coverage_delta=0.0,
            goal_progress=0.0,
            repeated_visit=False,
            terminated=False,
            execution_state="DECISION_BOUNDARY",
        )
    )
    env = V3ExplorationEnvironment(
        platform_type="WHEELED",
        bridge=_Bridge(
            _reference_output("HOPPER", ExecutionDirective.ACTIVATE_NEW_REFERENCE)
        ),
        request_builder=lambda action: action,
        initial_observation=_observation(),
        reference_executor=executor,
    )

    with pytest.raises(EnvironmentInvariantError, match="platform"):
        env.step(PolicyAction(frontier_index=0, theta_rad=0.0))

    assert executor.references == []
    assert env.rollout_discarded is True
    assert env.training_stopped is True


def test_rejecting_directive_with_reference_fails_closed() -> None:
    """Would fail if a rejected reference were encoded as an ordinary reward."""
    env = V3ExplorationEnvironment(
        platform_type="WHEELED",
        bridge=_Bridge(
            _reference_output("WHEELED", ExecutionDirective.NO_SAFE_REFERENCE)
        ),
        request_builder=lambda action: action,
        initial_observation=_observation(),
        reference_executor=lambda reference: None,
    )

    with pytest.raises(EnvironmentInvariantError, match="directive"):
        env.step(PolicyAction(frontier_index=0, theta_rad=0.0))

    assert env.rollout_discarded is True
    assert env.training_stopped is True


def test_committed_hop_directive_on_ground_platform_fails_closed() -> None:
    """Would fail if a hopper-only directive became a recoverable ground error."""
    output = PlannerOutput()
    output.outcome = PlanningOutcome.SAFE_FRONTIER_REFERENCE_AVAILABLE
    output.directive = ExecutionDirective.CONTINUE_COMMITTED_HOP
    output.reason_code = "COMMITTED_HOP_CONTINUES"
    env = V3ExplorationEnvironment(
        platform_type="LEGGED",
        bridge=_Bridge(output),
        request_builder=lambda action: action,
        initial_observation=_observation(1),
    )

    with pytest.raises(EnvironmentInvariantError, match="directive"):
        env.step(PolicyAction(frontier_index=0, theta_rad=0.0))

    assert env.rollout_discarded is True
    assert env.training_stopped is True

"""In-process macro-step environment backed by the C++ v3 planner bridge."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Callable, Protocol

from lunar_planner_training_bridge import (
    ExecutionDirective,
    MotionReference,
    PlannerOutput,
    PlanningOutcome,
    TrainingPlanRequest,
)

from ..policy.observation import PolicyBatch
from .macro_step import PlannerTransition, PolicyAction


class EnvironmentInvariantError(RuntimeError):
    """A planner/environment mismatch that invalidates the current rollout."""


class PlannerBridgeProtocol(Protocol):
    def plan(self, request: TrainingPlanRequest) -> PlannerOutput:
        raise NotImplementedError


@dataclass(frozen=True)
class DecisionBoundaryResult:
    execution_state: str
    transition: PlannerTransition | None = None


@dataclass(frozen=True)
class ReferenceExecutionResult:
    next_observation: PolicyBatch
    coverage_delta: float
    goal_progress: float
    repeated_visit: bool
    terminated: bool
    execution_state: str


class V3ExplorationEnvironment:
    """One planner call plus execution to the next exploration boundary."""

    def __init__(
        self,
        *,
        platform_type: str,
        bridge: PlannerBridgeProtocol,
        request_builder: Callable[[PolicyAction], TrainingPlanRequest],
        initial_observation: PolicyBatch,
        reference_executor: Callable[
            [MotionReference], ReferenceExecutionResult
        ] | None = None,
        plan_cost_scale: float = 1.0,
        planner_elapsed_scale_s: float = 1.0,
    ) -> None:
        self._platform_type = platform_type
        self._bridge = bridge
        self._request_builder = request_builder
        self._observation = initial_observation
        self._reference_executor = reference_executor
        self._plan_cost_scale = plan_cost_scale
        self._planner_elapsed_scale_s = planner_elapsed_scale_s
        self._rollout_discarded = False
        self._training_stopped = False
        self._execution_state = (
            "GROUND_HOLD" if platform_type == "HOPPER" else "DECISION_BOUNDARY"
        )

    @property
    def rollout_discarded(self) -> bool:
        return self._rollout_discarded

    @property
    def training_stopped(self) -> bool:
        return self._training_stopped

    def step(self, action: PolicyAction) -> PlannerTransition:
        if self._execution_state in {"JUMP_COMMITTED", "IN_FLIGHT"}:
            self._fail_closed("committed hopper cannot accept a new policy action")
        output = self._bridge.plan(self._request_builder(action))
        if output.directive == ExecutionDirective.CONTINUE_COMMITTED_HOP:
            if output.reference is not None:
                self._fail_closed(
                    "committed-hop directive cannot carry a replacement reference"
                )
            return self._advance_committed_hop_without_policy(output)
        if output.reference is None:
            return self._hold_transition(
                outcome=output.outcome,
                directive=output.directive,
                reason_code=output.reason_code,
                planner_elapsed=output.diagnostics.elapsed,
            )
        if output.reference.platform_type != self._platform_type:
            self._fail_closed("reference platform does not match episode platform")
        if output.directive not in {
            ExecutionDirective.ACTIVATE_NEW_REFERENCE,
            ExecutionDirective.CONTINUE_ACTIVE_REFERENCE,
        }:
            self._fail_closed("execution directive does not allow its reference")
        return self._execute_reference_until_decision_boundary(output)

    def begin_committed_hop(self) -> None:
        if self._platform_type != "HOPPER":
            raise ValueError("only HOPPER can enter a committed hop")
        self._execution_state = "JUMP_COMMITTED"

    def advance_until_decision_boundary(
        self, policy: Callable[[PolicyBatch], PolicyAction]
    ) -> DecisionBoundaryResult:
        if self._execution_state in {"JUMP_COMMITTED", "IN_FLIGHT"}:
            self._execution_state = "IN_FLIGHT"
            self._execution_state = "LANDED_HOLD"
            return DecisionBoundaryResult(execution_state=self._execution_state)
        action = policy(self._observation)
        transition = self.step(action)
        return DecisionBoundaryResult(
            execution_state=self._execution_state,
            transition=transition,
        )

    def _advance_committed_hop_without_policy(
        self, output: PlannerOutput
    ) -> PlannerTransition:
        if self._platform_type != "HOPPER":
            self._fail_closed(
                "committed hop directive requires a HOPPER episode"
            )
        self._execution_state = "IN_FLIGHT"
        self._execution_state = "LANDED_HOLD"
        return self._hold_transition(
            outcome=output.outcome,
            directive=output.directive,
            reason_code=output.reason_code,
            planner_elapsed=output.diagnostics.elapsed,
        )

    def _execute_reference_until_decision_boundary(
        self, output: PlannerOutput
    ) -> PlannerTransition:
        if self._reference_executor is None:
            self._fail_closed("reference executor is required for executable output")
        execution = self._reference_executor(output.reference)
        if not isinstance(execution, ReferenceExecutionResult):
            self._fail_closed("reference executor returned an invalid result")
        self._observation = execution.next_observation
        self._execution_state = execution.execution_state
        best_cost = output.diagnostics.best_cost
        return PlannerTransition(
            next_observation=execution.next_observation,
            coverage_delta=execution.coverage_delta,
            goal_progress=execution.goal_progress,
            normalized_plan_cost=(
                0.0 if best_cost is None else best_cost / self._plan_cost_scale
            ),
            normalized_elapsed_time=(
                output.diagnostics.elapsed.total_seconds()
                / self._planner_elapsed_scale_s
            ),
            repeated_visit=execution.repeated_visit,
            planning_outcome=output.outcome,
            execution_directive=output.directive,
            reason_code=output.reason_code,
            terminated=execution.terminated,
        )

    def _fail_closed(self, message: str) -> None:
        self._rollout_discarded = True
        self._training_stopped = True
        raise EnvironmentInvariantError(message)

    def _hold_transition(
        self,
        *,
        outcome: PlanningOutcome,
        directive: ExecutionDirective,
        reason_code: str,
        planner_elapsed: timedelta,
    ) -> PlannerTransition:
        return PlannerTransition(
            next_observation=_clone_observation(self._observation),
            coverage_delta=0.0,
            goal_progress=0.0,
            normalized_plan_cost=0.0,
            normalized_elapsed_time=(
                planner_elapsed.total_seconds() / self._planner_elapsed_scale_s
            ),
            repeated_visit=False,
            planning_outcome=outcome,
            execution_directive=directive,
            reason_code=reason_code,
            terminated=False,
        )


def _clone_observation(observation: PolicyBatch) -> PolicyBatch:
    return PolicyBatch(
        prior_channels=observation.prior_channels.clone(),
        coverage_summary=observation.coverage_summary.clone(),
        local_crop=observation.local_crop.clone(),
        frontier_features=observation.frontier_features.clone(),
        pose_features=observation.pose_features.clone(),
        candidate_mask=observation.candidate_mask.clone(),
        platform_context=observation.platform_context.clone(),
    )


__all__ = [
    "DecisionBoundaryResult",
    "EnvironmentInvariantError",
    "ReferenceExecutionResult",
    "V3ExplorationEnvironment",
]

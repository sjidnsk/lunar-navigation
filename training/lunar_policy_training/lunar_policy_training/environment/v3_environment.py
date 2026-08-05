"""In-process macro-step environment backed by the C++ v3 planner bridge."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Callable, Protocol

from lunar_planner_training_bridge import (
    ExecutionDirective,
    MotionReference,
    PlannerBridge,
    PlannerOutput,
    PlanningOutcome,
    TrainingPlanRequest,
)

from ..policy.observation import ObservationIdentity, PolicyBatch
from .macro_step import ExecutionEvents, PlannerTransition, PolicyAction


_REFERENCE_OUTPUTS = frozenset(
    {
        (
            PlanningOutcome.NEW_REFERENCE_AVAILABLE,
            ExecutionDirective.ACTIVATE_NEW_REFERENCE,
        ),
    }
)
_NO_REFERENCE_OUTPUTS = frozenset(
    {
        (
            PlanningOutcome.SAFE_FRONTIER_REFERENCE_AVAILABLE,
            ExecutionDirective.CONTINUE_COMMITTED_HOP,
        ),
        (PlanningOutcome.GOAL_INFEASIBLE, ExecutionDirective.HOLD_POSITION),
        (PlanningOutcome.CANCELED, ExecutionDirective.HOLD_POSITION),
        (PlanningOutcome.NUMERICAL_FAILURE, ExecutionDirective.HOLD_POSITION),
        (PlanningOutcome.INVALID_REQUEST, ExecutionDirective.NO_SAFE_REFERENCE),
        (
            PlanningOutcome.NO_KNOWN_SAFE_ROUTE,
            ExecutionDirective.NO_SAFE_REFERENCE,
        ),
        (
            PlanningOutcome.RESOURCE_EXHAUSTED,
            ExecutionDirective.NO_SAFE_REFERENCE,
        ),
        (
            PlanningOutcome.ACTIVE_REFERENCE_INVALIDATED,
            ExecutionDirective.NO_SAFE_REFERENCE,
        ),
        (
            PlanningOutcome.NUMERICAL_FAILURE,
            ExecutionDirective.NO_SAFE_REFERENCE,
        ),
    }
)
_REJECTED_ACTION_OUTPUTS = frozenset(
    {
        (PlanningOutcome.GOAL_INFEASIBLE, ExecutionDirective.HOLD_POSITION),
        (
            PlanningOutcome.NO_KNOWN_SAFE_ROUTE,
            ExecutionDirective.NO_SAFE_REFERENCE,
        ),
    }
)


class EnvironmentInvariantError(RuntimeError):
    """A planner/environment mismatch that invalidates the current rollout."""


class PlannerBridgeProtocol(Protocol):
    def plan(self, request: TrainingPlanRequest) -> PlannerOutput:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class PreparedPlanRequest:
    """Planner request explicitly attested to one prepared observation identity."""

    request: TrainingPlanRequest
    identity: ObservationIdentity

    def __post_init__(self) -> None:
        if not isinstance(self.request, TrainingPlanRequest):
            raise ValueError("prepared planner request must use TrainingPlanRequest")
        if not isinstance(self.identity, ObservationIdentity):
            raise ValueError("prepared planner request requires observation identity")


@dataclass(frozen=True)
class CommittedHopExecutionFeedback:
    execution_state: str
    next_observation: PolicyBatch
    coverage_delta: float
    goal_progress: float
    repeated_visit: bool
    terminated: bool
    execution_events: ExecutionEvents = ExecutionEvents()


@dataclass(frozen=True)
class DecisionBoundaryResult:
    execution_state: str
    transition: PlannerTransition | None = None
    execution_feedback: CommittedHopExecutionFeedback | None = None
    decision_budget_consumed: int = 0


@dataclass(frozen=True)
class ReferenceExecutionResult:
    next_observation: PolicyBatch
    coverage_delta: float
    goal_progress: float
    repeated_visit: bool
    terminated: bool
    execution_state: str
    execution_events: ExecutionEvents = ExecutionEvents()


class V3ExplorationEnvironment:
    """One planner call plus execution to the next exploration boundary."""

    def __init__(
        self,
        *,
        platform_type: str,
        bridge: PlannerBridgeProtocol,
        request_builder: Callable[..., object],
        initial_observation: PolicyBatch,
        observation_provider: Callable[[], PolicyBatch] | None = None,
        require_identity_bound_request: bool = False,
        reference_executor: Callable[
            [MotionReference], ReferenceExecutionResult
        ] | None = None,
        committed_hop_executor: Callable[
            [], CommittedHopExecutionFeedback
        ] | None = None,
        total_decision_budget: int = 8,
        remaining_decision_budget: int | None = None,
        plan_cost_scale: float = 1.0,
        planner_elapsed_scale_s: float = 1.0,
    ) -> None:
        self._platform_type = platform_type
        self._bridge = bridge
        self._request_builder = request_builder
        if (
            not isinstance(initial_observation, PolicyBatch)
            or initial_observation.prior_channels.shape[0] != 1
            or initial_observation.observation_identities is None
            or len(initial_observation.observation_identities) != 1
        ):
            raise EnvironmentInvariantError(
                "V3 observation requires one producer-owned observation identity"
            )
        if type(total_decision_budget) is not int or total_decision_budget <= 0:
            raise ValueError("total decision budget must be a positive integer")
        if remaining_decision_budget is None:
            remaining_decision_budget = total_decision_budget
        if (
            type(remaining_decision_budget) is not int
            or remaining_decision_budget < 0
            or remaining_decision_budget > total_decision_budget
        ):
            raise ValueError(
                "remaining decision budget must be within total decision budget"
            )
        if observation_provider is not None and not callable(observation_provider):
            raise ValueError("observation provider must be callable")
        if type(require_identity_bound_request) is not bool:
            raise ValueError("identity-bound request flag must be boolean")
        self._observation = _clone_observation(initial_observation)
        self._rejected_candidates: set[int] = set()
        self._observation_provider = observation_provider
        self._require_identity_bound_request = require_identity_bound_request
        self._total_decision_budget = total_decision_budget
        self._remaining_decisions = remaining_decision_budget
        self._set_network_budget_ratio()
        self._reference_executor = reference_executor
        self._committed_hop_executor = committed_hop_executor
        self._plan_cost_scale = plan_cost_scale
        self._planner_elapsed_scale_s = planner_elapsed_scale_s
        self._rollout_discarded = False
        self._training_stopped = False
        self._committed_output: PlannerOutput | None = None
        self._execution_state = (
            "GROUND_HOLD" if platform_type == "HOPPER" else "DECISION_BOUNDARY"
        )

    @property
    def rollout_discarded(self) -> bool:
        return self._rollout_discarded

    @property
    def training_stopped(self) -> bool:
        return self._training_stopped

    @property
    def current_observation(self) -> PolicyBatch:
        """Return a detached copy of the environment-owned current observation."""
        return _clone_observation(self._observation)

    def step(
        self,
        action: PolicyAction,
        *,
        expected_identity: ObservationIdentity | None = None,
    ) -> PlannerTransition:
        if self._execution_state in {"JUMP_COMMITTED", "IN_FLIGHT"}:
            self._fail_closed("committed hopper cannot accept a new policy action")
        request = self._build_plan_request(action, expected_identity)
        output = self._bridge.plan(request)
        self._validate_output(output)
        if output.directive == ExecutionDirective.CONTINUE_COMMITTED_HOP:
            return self._advance_committed_hop_without_policy(output)
        if output.reference is None:
            if (output.outcome, output.directive) in _REJECTED_ACTION_OUTPUTS:
                self._mask_rejected_candidate(action.frontier_index)
            return self._hold_transition(
                outcome=output.outcome,
                directive=output.directive,
                reason_code=output.reason_code,
                planner_elapsed=output.diagnostics.elapsed,
            )
        return self._execute_reference_until_decision_boundary(output)

    def _validate_output(self, output: PlannerOutput) -> None:
        signature = (output.outcome, output.directive)
        has_reference = output.reference is not None
        legal = (
            signature in _REFERENCE_OUTPUTS
            if has_reference
            else signature in _NO_REFERENCE_OUTPUTS
        )
        if not legal:
            self._fail_closed(
                "planner output outcome/directive/reference combination is invalid"
            )
        if has_reference and output.reference.platform_type != self._platform_type:
            self._fail_closed("reference platform does not match episode platform")
        if (
            output.directive == ExecutionDirective.CONTINUE_COMMITTED_HOP
            and self._platform_type != "HOPPER"
        ):
            self._fail_closed(
                "committed hop directive requires a HOPPER episode"
            )

    def begin_committed_hop(self) -> None:
        if self._platform_type != "HOPPER":
            raise ValueError("only HOPPER can enter a committed hop")
        self._committed_output = None
        self._execution_state = "JUMP_COMMITTED"

    def advance_until_decision_boundary(
        self, policy: Callable[[PolicyBatch], PolicyAction]
    ) -> DecisionBoundaryResult:
        if self._execution_state in {"JUMP_COMMITTED", "IN_FLIGHT"}:
            feedback = self._advance_committed_hop_execution()
            output = self._committed_output
            transition = (
                self._committed_feedback_transition(output, feedback)
                if output is not None
                else None
            )
            if feedback.execution_state == "LANDED_HOLD":
                self._committed_output = None
            return DecisionBoundaryResult(
                execution_state=self._execution_state,
                transition=transition,
                execution_feedback=feedback,
            )
        boundary = self.refresh_decision_boundary()
        if boundary.execution_state != "DECISION_READY":
            return boundary
        action = policy(_clone_observation(self._observation))
        return self.advance_prepared_action(
            action,
            expected_identity=self._observation.observation_identities[0],
        )

    def advance_prepared_action(
        self,
        action: PolicyAction,
        *,
        expected_identity: ObservationIdentity,
    ) -> DecisionBoundaryResult:
        """Apply one action to the exact identity published before policy forward."""
        current_identity = self._observation.observation_identities[0]
        if not isinstance(expected_identity, ObservationIdentity):
            self._fail_closed("prepared action requires an observation identity")
        if expected_identity != current_identity:
            self._fail_closed("prepared action observation identity is stale")
        if not bool(self._observation.candidate_mask.any().item()):
            return DecisionBoundaryResult(execution_state="NO_CANDIDATES")
        if self._remaining_decisions <= 0:
            return DecisionBoundaryResult(execution_state="DECISION_BUDGET_EXHAUSTED")
        self._remaining_decisions -= 1
        self._set_network_budget_ratio()
        transition = self.step(action, expected_identity=expected_identity)
        return DecisionBoundaryResult(
            execution_state=self._execution_state,
            transition=transition,
            decision_budget_consumed=1,
        )

    def refresh_decision_boundary(self) -> DecisionBoundaryResult:
        """Refresh producer input and classify a ground boundary without policy."""
        if self._execution_state in {"JUMP_COMMITTED", "IN_FLIGHT"}:
            self._fail_closed(
                "committed hopper requires execution feedback before refresh"
            )
        self._refresh_ground_observation()
        if not bool(self._observation.candidate_mask.any().item()):
            return DecisionBoundaryResult(execution_state="NO_CANDIDATES")
        if self._remaining_decisions <= 0:
            return DecisionBoundaryResult(execution_state="DECISION_BUDGET_EXHAUSTED")
        return DecisionBoundaryResult(execution_state="DECISION_READY")

    def _mask_rejected_candidate(self, candidate_index: int) -> None:
        mask = self._observation.candidate_mask
        if (
            mask.ndim != 2
            or mask.shape[0] != 1
            or candidate_index < 0
            or candidate_index >= mask.shape[1]
            or not bool(mask[0, candidate_index].item())
        ):
            self._fail_closed(
                "rejected planner action does not identify an active candidate"
            )
        self._rejected_candidates.add(candidate_index)
        masked = _clone_observation(self._observation)
        masked.candidate_mask[0, candidate_index] = False
        self._observation = masked

    def _build_plan_request(
        self,
        action: PolicyAction,
        expected_identity: ObservationIdentity | None,
    ) -> TrainingPlanRequest | object:
        if not self._require_identity_bound_request:
            return self._request_builder(action)
        if expected_identity is None:
            self._fail_closed(
                "identity-bound request requires a prepared observation"
            )
        prepared = self._request_builder(action, expected_identity)
        if not isinstance(prepared, PreparedPlanRequest):
            self._fail_closed(
                "request builder must return PreparedPlanRequest"
            )
        if prepared.identity != expected_identity:
            self._fail_closed("request observation identity is stale")
        if (
            prepared.request.state_time.nanoseconds_since_epoch
            != expected_identity.state_time_ns
        ):
            self._fail_closed("request state time does not match observation identity")
        return prepared.request

    def _install_observation(self, observation: PolicyBatch) -> PolicyBatch:
        if (
            not isinstance(observation, PolicyBatch)
            or observation.prior_channels.shape[0] != 1
            or observation.observation_identities is None
            or len(observation.observation_identities) != 1
        ):
            self._fail_closed(
                "V3 observation requires one producer-owned observation identity"
            )
        current_identity = self._observation.observation_identities[0]
        next_identity = observation.observation_identities[0]
        if next_identity != current_identity:
            self._rejected_candidates.clear()
        installed = _clone_observation(observation)
        for candidate_index in self._rejected_candidates:
            if candidate_index < installed.candidate_mask.shape[1]:
                installed.candidate_mask[0, candidate_index] = False
        self._observation = installed
        self._set_network_budget_ratio()
        return self._observation

    def _refresh_ground_observation(self) -> None:
        if self._observation_provider is None:
            return
        observation = self._observation_provider()
        if not isinstance(observation, PolicyBatch):
            self._fail_closed("observation provider returned invalid data")
        self._install_observation(observation)

    def _set_network_budget_ratio(self) -> None:
        self._observation.pose_features[0, 5] = (
            self._remaining_decisions / self._total_decision_budget
        )

    def _advance_committed_hop_without_policy(
        self, output: PlannerOutput
    ) -> PlannerTransition:
        if self._platform_type != "HOPPER":
            self._fail_closed(
                "committed hop directive requires a HOPPER episode"
            )
        self._committed_output = output
        feedback = self._advance_committed_hop_execution()
        transition = self._committed_feedback_transition(output, feedback)
        if feedback.execution_state == "LANDED_HOLD":
            self._committed_output = None
        return transition

    def _advance_committed_hop_execution(
        self,
    ) -> CommittedHopExecutionFeedback:
        if self._committed_hop_executor is None:
            self._fail_closed(
                "committed hop execution feedback is required before landing"
            )
        feedback = self._committed_hop_executor()
        if not isinstance(feedback, CommittedHopExecutionFeedback):
            self._fail_closed("committed hop executor returned invalid feedback")
        if feedback.execution_state not in {
            "JUMP_COMMITTED",
            "IN_FLIGHT",
            "LANDED_HOLD",
        }:
            self._fail_closed("committed hop feedback has invalid execution state")
        self._install_observation(feedback.next_observation)
        self._execution_state = feedback.execution_state
        return feedback

    def _committed_feedback_transition(
        self,
        output: PlannerOutput,
        feedback: CommittedHopExecutionFeedback,
    ) -> PlannerTransition:
        return PlannerTransition(
            next_observation=self._observation,
            coverage_delta=feedback.coverage_delta,
            goal_progress=feedback.goal_progress,
            normalized_plan_cost=0.0,
            normalized_elapsed_time=(
                output.diagnostics.elapsed.total_seconds()
                / self._planner_elapsed_scale_s
            ),
            repeated_visit=feedback.repeated_visit,
            planning_outcome=output.outcome,
            execution_directive=output.directive,
            reason_code=output.reason_code,
            terminated=feedback.terminated,
            execution_events=feedback.execution_events,
        )

    def _execute_reference_until_decision_boundary(
        self, output: PlannerOutput
    ) -> PlannerTransition:
        if self._reference_executor is None:
            self._fail_closed("reference executor is required for executable output")
        execution = self._reference_executor(output.reference)
        if not isinstance(execution, ReferenceExecutionResult):
            self._fail_closed("reference executor returned an invalid result")
        self._install_observation(execution.next_observation)
        self._execution_state = execution.execution_state
        best_cost = output.diagnostics.best_cost
        return PlannerTransition(
            next_observation=self._observation,
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
            execution_events=execution.execution_events,
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
        observation_identities=observation.observation_identities,
    )


def create_v3_environment(
    *,
    platform_type: str,
    request_builder: Callable[
        [PolicyAction, ObservationIdentity], PreparedPlanRequest
    ],
    initial_observation: PolicyBatch,
    observation_provider: Callable[[], PolicyBatch] | None = None,
    reference_executor: Callable[
        [MotionReference], ReferenceExecutionResult
    ] | None = None,
    committed_hop_executor: Callable[
        [], CommittedHopExecutionFeedback
    ] | None = None,
    total_decision_budget: int = 8,
    remaining_decision_budget: int | None = None,
    plan_cost_scale: float = 1.0,
    planner_elapsed_scale_s: float = 1.0,
) -> V3ExplorationEnvironment:
    """Compose the supported training environment with the real C++ v3 bridge."""
    if platform_type == "HOPPER" and committed_hop_executor is None:
        raise ValueError("production HOPPER requires a committed hop executor")
    return V3ExplorationEnvironment(
        platform_type=platform_type,
        bridge=PlannerBridge(),
        request_builder=request_builder,
        initial_observation=initial_observation,
        observation_provider=observation_provider,
        require_identity_bound_request=True,
        reference_executor=reference_executor,
        committed_hop_executor=committed_hop_executor,
        total_decision_budget=total_decision_budget,
        remaining_decision_budget=remaining_decision_budget,
        plan_cost_scale=plan_cost_scale,
        planner_elapsed_scale_s=planner_elapsed_scale_s,
    )


__all__ = [
    "CommittedHopExecutionFeedback",
    "DecisionBoundaryResult",
    "EnvironmentInvariantError",
    "PreparedPlanRequest",
    "ReferenceExecutionResult",
    "V3ExplorationEnvironment",
    "create_v3_environment",
]

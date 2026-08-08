"""In-process macro-step environment backed by the C++ v3 planner bridge."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
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
from .observation_boundary import (
    BoundaryObservationResult,
    ObservationBoundaryController,
    SensorBoundaryEvidence,
)


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
_MAX_COMMITTED_HOP_FEEDBACK_STEPS = 64
_EXECUTION_EVENT_COUNT_FIELDS = (
    "safety_violation_count",
    "invalid_action_count",
    "platform_reference_mismatch_count",
    "hopper_commitment_violation_count",
    "execution_failure_count",
    "reference_samples_consumed",
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
    mission_observed_delta: float
    priority_observed_delta: float
    normalized_execution_cost_contribution: float
    normalized_execution_time_contribution: float
    executed_without_new_coverage: bool
    success_first_crossing: bool
    episode_ended_without_success: bool
    hard_safety_violation: bool
    terminated: bool
    execution_events: ExecutionEvents = ExecutionEvents()
    sensor_boundary_evidence: SensorBoundaryEvidence | None = None


@dataclass(frozen=True)
class DecisionBoundaryResult:
    execution_state: str
    transition: PlannerTransition | None = None
    execution_feedback: CommittedHopExecutionFeedback | None = None
    policy_decisions_consumed: int = 0


@dataclass(frozen=True)
class ReferenceExecutionResult:
    next_observation: PolicyBatch
    mission_observed_delta: float
    priority_observed_delta: float
    normalized_execution_cost_contribution: float
    normalized_execution_time_contribution: float
    executed_without_new_coverage: bool
    success_first_crossing: bool
    episode_ended_without_success: bool
    hard_safety_violation: bool
    terminated: bool
    execution_state: str
    execution_events: ExecutionEvents = ExecutionEvents()
    sensor_boundary_evidence: SensorBoundaryEvidence | None = None


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
        observation_boundary_controller: (
            ObservationBoundaryController | None
        ) = None,
        require_sensor_closed_loop: bool = False,
        require_identity_bound_request: bool = False,
        reference_executor: Callable[
            [MotionReference], ReferenceExecutionResult
        ] | None = None,
        committed_hop_executor: Callable[
            [], CommittedHopExecutionFeedback
        ] | None = None,
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
        if observation_provider is not None and not callable(observation_provider):
            raise ValueError("observation provider must be callable")
        if type(require_sensor_closed_loop) is not bool:
            raise ValueError("sensor-closed-loop flag must be boolean")
        if require_sensor_closed_loop:
            if not isinstance(
                observation_boundary_controller, ObservationBoundaryController
            ):
                raise ValueError(
                    "sensor-closed environment requires an observation boundary controller"
                )
            if not observation_boundary_controller.initialized:
                raise ValueError(
                    "observation boundary controller must perform its initial reveal"
                )
            if observation_boundary_controller.platform_type != platform_type:
                raise ValueError("observation boundary controller platform mismatch")
            if observation_provider is not None:
                raise ValueError(
                    "sensor-closed environment cannot use an observation provider"
                )
            boundary_observation = (
                observation_boundary_controller.current_observation
            )
            if not _same_observation(
                boundary_observation, initial_observation
            ):
                raise ValueError(
                    "initial observation is not owned by the boundary controller"
                )
        elif observation_boundary_controller is not None:
            raise ValueError(
                "observation boundary controller requires sensor-closed-loop mode"
            )
        if type(require_identity_bound_request) is not bool:
            raise ValueError("identity-bound request flag must be boolean")
        self._observation = _clone_observation(initial_observation)
        self._rejected_candidates: set[int] = set()
        self._observation_provider = observation_provider
        self._observation_boundary_controller = observation_boundary_controller
        self._require_sensor_closed_loop = require_sensor_closed_loop
        self._require_identity_bound_request = require_identity_bound_request
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
    def sensor_closed_loop(self) -> bool:
        return self._require_sensor_closed_loop

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
                self._committed_feedback_transition(
                    output,
                    feedback,
                    include_planner_contribution=False,
                )
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
        transition = self.step(action, expected_identity=expected_identity)
        if self._platform_type == "HOPPER" and self._execution_state in {
            "JUMP_COMMITTED",
            "IN_FLIGHT",
        }:
            transition = self._complete_committed_hop_transition(transition)
        if (
            not transition.terminated
            and not transition.success_first_crossing
            and not transition.hard_safety_violation
            and not bool(
                transition.next_observation.candidate_mask.any().item()
            )
        ):
            transition = replace(
                transition,
                episode_ended_without_success=True,
                terminated=True,
            )
        return DecisionBoundaryResult(
            execution_state=self._execution_state,
            transition=transition,
            policy_decisions_consumed=1,
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
        return self._observation

    def _refresh_ground_observation(self) -> None:
        if self._observation_provider is None:
            return
        observation = self._observation_provider()
        if not isinstance(observation, PolicyBatch):
            self._fail_closed("observation provider returned invalid data")
        self._install_observation(observation)

    def _advance_committed_hop_without_policy(
        self, output: PlannerOutput
    ) -> PlannerTransition:
        if self._platform_type != "HOPPER":
            self._fail_closed(
                "committed hop directive requires a HOPPER episode"
            )
        self._committed_output = output
        feedback = self._advance_committed_hop_execution()
        transition = self._committed_feedback_transition(
            output,
            feedback,
            include_planner_contribution=True,
        )
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
        if self._require_sensor_closed_loop:
            boundary = self._apply_sensor_boundary(
                feedback.execution_state,
                feedback.sensor_boundary_evidence,
            )
            if boundary.updated:
                self._install_observation(boundary.next_observation)
            feedback = replace(
                feedback,
                next_observation=_clone_observation(self._observation),
                mission_observed_delta=boundary.mission_observed_delta,
                priority_observed_delta=boundary.priority_observed_delta,
                executed_without_new_coverage=(
                    boundary.mission_observed_delta == 0.0
                    and boundary.priority_observed_delta == 0.0
                ),
            )
        else:
            self._install_observation(feedback.next_observation)
        self._execution_state = feedback.execution_state
        return feedback

    def _complete_committed_hop_transition(
        self,
        initial_transition: PlannerTransition,
    ) -> PlannerTransition:
        """Drain one production HOPPER action to a finite landed macro-step."""
        transitions = [initial_transition]
        previous_state = self._execution_state
        for _ in range(_MAX_COMMITTED_HOP_FEEDBACK_STEPS):
            if previous_state == "LANDED_HOLD":
                self._committed_output = None
                return self._aggregate_hopper_transitions(transitions)
            if previous_state not in {"JUMP_COMMITTED", "IN_FLIGHT"}:
                self._fail_closed(
                    "committed hopper left the execution state machine before landing"
                )
            if transitions[-1].terminated:
                self._fail_closed(
                    "committed hopper terminated before LANDED_HOLD"
                )
            output = self._committed_output
            if output is None:
                self._fail_closed(
                    "committed hopper lost its planner output before landing"
                )
            feedback = self._advance_committed_hop_execution()
            next_state = feedback.execution_state
            if previous_state == "IN_FLIGHT" and next_state == "JUMP_COMMITTED":
                self._fail_closed("committed hopper execution state regressed")
            transitions.append(
                self._committed_feedback_transition(
                    output,
                    feedback,
                    include_planner_contribution=False,
                )
            )
            previous_state = next_state
            if previous_state == "LANDED_HOLD":
                self._committed_output = None
                return self._aggregate_hopper_transitions(transitions)
        self._fail_closed("committed hopper did not land within the feedback limit")

    def _aggregate_hopper_transitions(
        self,
        transitions: list[PlannerTransition],
    ) -> PlannerTransition:
        mission_observed_delta = sum(
            transition.mission_observed_delta for transition in transitions
        )
        priority_observed_delta = sum(
            transition.priority_observed_delta for transition in transitions
        )
        if not all(
            math.isfinite(float(value))
            for transition in transitions
            for value in (
                transition.mission_observed_delta,
                transition.priority_observed_delta,
                transition.normalized_plan_or_execution_cost,
                transition.normalized_macro_step_time,
            )
        ) or not all(
            math.isfinite(value)
            for value in (mission_observed_delta, priority_observed_delta)
        ):
            self._fail_closed("committed hopper aggregation contains non-finite data")
        success_crossings = sum(
            transition.success_first_crossing for transition in transitions
        )
        if success_crossings > 1:
            self._fail_closed("committed hopper emitted success more than once")
        final = transitions[-1]
        if self._execution_state != "LANDED_HOLD":
            self._fail_closed("committed hopper aggregation requires LANDED_HOLD")
        return PlannerTransition(
            next_observation=_clone_observation(final.next_observation),
            mission_observed_delta=mission_observed_delta,
            priority_observed_delta=priority_observed_delta,
            normalized_plan_or_execution_cost=(
                sum(
                    transition.normalized_plan_or_execution_cost
                    for transition in transitions
                )
            ),
            normalized_macro_step_time=(
                sum(
                    transition.normalized_macro_step_time
                    for transition in transitions
                )
            ),
            executed_without_new_coverage=all(
                transition.executed_without_new_coverage
                for transition in transitions
            ),
            success_first_crossing=success_crossings == 1,
            episode_ended_without_success=(
                final.episode_ended_without_success
            ),
            hard_safety_violation=any(
                transition.hard_safety_violation for transition in transitions
            ),
            cancellation_expected=transitions[0].cancellation_expected,
            cpp_exception=next(
                (
                    transition.cpp_exception
                    for transition in transitions
                    if transition.cpp_exception is not None
                ),
                None,
            ),
            planning_outcome=transitions[0].planning_outcome,
            execution_directive=transitions[0].execution_directive,
            reason_code=transitions[0].reason_code,
            terminated=final.terminated,
            execution_events=self._aggregate_execution_events(transitions),
        )

    def _aggregate_execution_events(
        self,
        transitions: list[PlannerTransition],
    ) -> ExecutionEvents:
        totals = {name: 0 for name in _EXECUTION_EVENT_COUNT_FIELDS}
        commitment_states: list[str] = []
        selected_action_observed_safe = False
        for transition in transitions:
            events = transition.execution_events
            if not isinstance(events, ExecutionEvents):
                self._fail_closed("committed hopper execution events are invalid")
            for name in _EXECUTION_EVENT_COUNT_FIELDS:
                value = getattr(events, name)
                if type(value) is not int or value < 0:
                    self._fail_closed(
                        "committed hopper event count must be non-negative"
                    )
                totals[name] += value
            if type(events.selected_action_observed_safe) is not bool:
                self._fail_closed("committed hopper safe-action fact is invalid")
            selected_action_observed_safe = (
                selected_action_observed_safe
                or events.selected_action_observed_safe
            )
            states = events.hopper_commitment_states
            if not isinstance(states, tuple) or any(
                state not in {"JUMP_COMMITTED", "IN_FLIGHT", "LANDED_HOLD"}
                for state in states
            ):
                self._fail_closed("committed hopper event states are invalid")
            if states:
                commitment_states.extend(states)
            else:
                identity = transition.next_observation.observation_identities[0]
                commitment_states.append(identity.execution_state)
        return ExecutionEvents(
            **totals,
            selected_action_observed_safe=selected_action_observed_safe,
            hopper_commitment_states=tuple(commitment_states),
        )

    def _committed_feedback_transition(
        self,
        output: PlannerOutput,
        feedback: CommittedHopExecutionFeedback,
        *,
        include_planner_contribution: bool,
    ) -> PlannerTransition:
        if type(include_planner_contribution) is not bool:
            self._fail_closed("planner contribution flag must be boolean")
        planner_cost = output.diagnostics.best_cost
        return PlannerTransition(
            next_observation=self._observation,
            mission_observed_delta=feedback.mission_observed_delta,
            priority_observed_delta=feedback.priority_observed_delta,
            normalized_plan_or_execution_cost=(
                feedback.normalized_execution_cost_contribution
                + (
                    0.0
                    if not include_planner_contribution or planner_cost is None
                    else planner_cost / self._plan_cost_scale
                )
            ),
            normalized_macro_step_time=(
                feedback.normalized_execution_time_contribution
                + (
                    output.diagnostics.elapsed.total_seconds()
                    / self._planner_elapsed_scale_s
                    if include_planner_contribution
                    else 0.0
                )
            ),
            executed_without_new_coverage=(
                feedback.executed_without_new_coverage
            ),
            success_first_crossing=feedback.success_first_crossing,
            episode_ended_without_success=(
                feedback.episode_ended_without_success
            ),
            hard_safety_violation=feedback.hard_safety_violation,
            cancellation_expected=False,
            cpp_exception=None,
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
        self._execution_state = execution.execution_state
        if self._require_sensor_closed_loop:
            boundary = self._apply_sensor_boundary(
                execution.execution_state,
                execution.sensor_boundary_evidence,
            )
            if boundary.updated:
                self._install_observation(boundary.next_observation)
            execution = replace(
                execution,
                next_observation=_clone_observation(self._observation),
                mission_observed_delta=boundary.mission_observed_delta,
                priority_observed_delta=boundary.priority_observed_delta,
                executed_without_new_coverage=(
                    boundary.mission_observed_delta == 0.0
                    and boundary.priority_observed_delta == 0.0
                ),
            )
        else:
            self._install_observation(execution.next_observation)
        if self._platform_type == "HOPPER" and self._execution_state in {
            "JUMP_COMMITTED",
            "IN_FLIGHT",
        }:
            self._committed_output = output
        best_cost = output.diagnostics.best_cost
        return PlannerTransition(
            next_observation=self._observation,
            mission_observed_delta=execution.mission_observed_delta,
            priority_observed_delta=execution.priority_observed_delta,
            normalized_plan_or_execution_cost=(
                execution.normalized_execution_cost_contribution
                + (0.0 if best_cost is None else best_cost / self._plan_cost_scale)
            ),
            normalized_macro_step_time=(
                execution.normalized_execution_time_contribution
                + output.diagnostics.elapsed.total_seconds()
                / self._planner_elapsed_scale_s
            ),
            executed_without_new_coverage=(
                execution.executed_without_new_coverage
            ),
            success_first_crossing=execution.success_first_crossing,
            episode_ended_without_success=(
                execution.episode_ended_without_success
            ),
            hard_safety_violation=execution.hard_safety_violation,
            cancellation_expected=False,
            cpp_exception=None,
            planning_outcome=output.outcome,
            execution_directive=output.directive,
            reason_code=output.reason_code,
            terminated=execution.terminated,
            execution_events=execution.execution_events,
        )

    def _apply_sensor_boundary(
        self,
        execution_state: str,
        evidence: SensorBoundaryEvidence | None,
    ) -> BoundaryObservationResult:
        controller = self._observation_boundary_controller
        if not isinstance(controller, ObservationBoundaryController):
            self._fail_closed("observation boundary controller is unavailable")
        try:
            return controller.after_execution(
                platform_type=self._platform_type,
                execution_state=execution_state,
                evidence=evidence,
            )
        except (TypeError, ValueError, RuntimeError) as error:
            self._fail_closed(str(error))

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
            mission_observed_delta=0.0,
            priority_observed_delta=0.0,
            normalized_plan_or_execution_cost=0.0,
            normalized_macro_step_time=(
                planner_elapsed.total_seconds() / self._planner_elapsed_scale_s
            ),
            executed_without_new_coverage=False,
            success_first_crossing=False,
            episode_ended_without_success=False,
            hard_safety_violation=False,
            cancellation_expected=False,
            cpp_exception=None,
            planning_outcome=outcome,
            execution_directive=directive,
            reason_code=reason_code,
            terminated=False,
            execution_events=ExecutionEvents(),
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


def _same_observation(left: PolicyBatch, right: PolicyBatch) -> bool:
    return (
        left.observation_identities == right.observation_identities
        and all(
            getattr(left, name).equal(getattr(right, name))
            for name in left.input_names
        )
    )


def create_v3_environment(
    *,
    platform_type: str,
    request_builder: Callable[
        [PolicyAction, ObservationIdentity], PreparedPlanRequest
    ],
    initial_observation: PolicyBatch,
    observation_provider: Callable[[], PolicyBatch] | None = None,
    observation_boundary_controller: (
        ObservationBoundaryController | None
    ) = None,
    require_sensor_closed_loop: bool = False,
    reference_executor: Callable[
        [MotionReference], ReferenceExecutionResult
    ] | None = None,
    committed_hop_executor: Callable[
        [], CommittedHopExecutionFeedback
    ] | None = None,
    plan_cost_scale: float = 1.0,
    planner_elapsed_scale_s: float = 1.0,
) -> V3ExplorationEnvironment:
    """Compose the supported training environment with the real C++ v3 bridge."""
    if platform_type == "HOPPER" and committed_hop_executor is None:
        raise ValueError("production HOPPER requires a committed hop executor")
    if require_sensor_closed_loop and not isinstance(
        observation_boundary_controller, ObservationBoundaryController
    ):
        raise ValueError(
            "sensor-closed environment requires an observation boundary controller"
        )
    return V3ExplorationEnvironment(
        platform_type=platform_type,
        bridge=PlannerBridge(),
        request_builder=request_builder,
        initial_observation=initial_observation,
        observation_provider=observation_provider,
        observation_boundary_controller=observation_boundary_controller,
        require_sensor_closed_loop=require_sensor_closed_loop,
        require_identity_bound_request=True,
        reference_executor=reference_executor,
        committed_hop_executor=committed_hop_executor,
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

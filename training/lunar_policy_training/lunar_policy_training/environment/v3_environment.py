"""In-process macro-step environment backed by the C++ v3 planner bridge."""

from __future__ import annotations

from hashlib import sha256
import math
from dataclasses import dataclass, replace
from datetime import timedelta
from typing import Callable, Protocol

from lunar_planner_training_bridge import (
    CandidateDisposition,
    ExecutionDirective,
    HopReference,
    MotionReference,
    PlannerBridge,
    PlannerOutput,
    PlanningOutcome,
    TrainingPlanRequest,
)

from ..policy.observation import ObservationIdentity, PolicyBatch
from ..reward_contract import RewardTerminalClass
from ..training_semantics import formal_success_first_crossing
from .candidate_builder import CandidateDecisionSnapshot, CandidateDiagnostics
from .macro_step import (
    ExecutionEvents,
    HopperObservationCommitment,
    PlannerTransition,
    PolicyAction,
    TerminalReason,
    terminal_class_for_reason,
)
from .multires_observation import HopperTrajectoryPoint
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
_MAX_COMMITTED_HOP_FEEDBACK_STEPS = 64
_GROUND_OPTION_TARGET_TOLERANCE_M = 0.2
_GROUND_OPTION_PROGRESS_EPSILON_M = 1.0e-6
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


def _audit_candidate_boundary(
    snapshot: CandidateDecisionSnapshot,
    platform_type: str,
) -> TerminalReason | None:
    """Classify one boundary from the single production candidate pipeline."""
    if not isinstance(snapshot, CandidateDecisionSnapshot):
        raise EnvironmentInvariantError("candidate snapshot is invalid")
    if platform_type not in {"WHEELED", "LEGGED", "HOPPER"}:
        raise EnvironmentInvariantError("candidate snapshot platform is invalid")
    expected_pipeline = (
        "HOPPER_LANDING" if platform_type == "HOPPER" else "GROUND_FRONTIER"
    )
    if snapshot.pipeline_kind != expected_pipeline:
        raise EnvironmentInvariantError("candidate snapshot pipeline is invalid")
    selected = snapshot.selected_policy_candidate_count
    reserve = snapshot.untried_reserve_count
    positive = snapshot.positive_gain_candidate_count
    rejected = snapshot.planner_rejected_current_snapshot_count
    if platform_type == "HOPPER":
        if selected > 0:
            return None
        if reserve > 0:
            raise EnvironmentInvariantError(
                "hopper candidate availability is inconsistent"
            )
        eligible = snapshot.eligible_landing_count
        if eligible == 0:
            return TerminalReason.NO_AVAILABLE_LANDING_CANDIDATE
        if rejected == eligible:
            return TerminalReason.PLANNER_EXHAUSTED
        raise EnvironmentInvariantError("hopper candidate snapshot is incomplete")
    if platform_type != "HOPPER" and snapshot.frontier_segment_count == 0:
        if any(
            (
                snapshot.raw_candidate_count,
                snapshot.fine_pose_candidate_count,
                snapshot.globally_reachable_candidate_count,
                positive,
                selected,
                reserve,
                rejected,
            )
        ):
            raise EnvironmentInvariantError(
                "candidate snapshot has candidates without a frontier"
            )
        return TerminalReason.NO_FRONTIER
    if selected > 0:
        return None
    if reserve > 0:
        raise EnvironmentInvariantError("candidate availability is inconsistent")
    if (
        snapshot.raw_candidate_count > 0
        and snapshot.globally_reachable_candidate_count == 0
    ):
        return TerminalReason.NO_GLOBAL_ROUTE
    if snapshot.globally_reachable_candidate_count > 0 and positive == 0:
        return TerminalReason.ZERO_EXPECTED_GAIN
    if positive > 0 and rejected == positive:
        return TerminalReason.PLANNER_EXHAUSTED
    raise EnvironmentInvariantError("candidate snapshot is incomplete")


class PlannerBridgeProtocol(Protocol):
    def plan(self, request: TrainingPlanRequest) -> PlannerOutput:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class PreparedPlanRequest:
    """Planner request explicitly attested to one prepared observation identity."""

    request: TrainingPlanRequest
    identity: ObservationIdentity
    candidate_id: str
    physical_snapshot_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.request, TrainingPlanRequest):
            raise ValueError("prepared planner request must use TrainingPlanRequest")
        if not isinstance(self.identity, ObservationIdentity):
            raise ValueError("prepared planner request requires observation identity")
        for name, value in (
            ("candidate", self.candidate_id),
            ("physical snapshot", self.physical_snapshot_id),
        ):
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(
                    f"prepared planner request {name} identity is invalid"
                )


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
    coverage_before: float = 0.0
    coverage_after: float = 0.0
    priority_before: float = 0.0
    priority_after: float = 0.0
    executed_path_length_m: float = 0.0
    commitment_id: str | None = None
    trajectory_points: tuple[HopperTrajectoryPoint, ...] = ()
    trajectory_within_certified_flight_tube: bool = True


@dataclass(frozen=True)
class DecisionBoundaryResult:
    execution_state: str
    transition: PlannerTransition | None = None
    execution_feedback: CommittedHopExecutionFeedback | None = None
    policy_decisions_consumed: int = 0
    terminal_reason: TerminalReason | None = None
    remaining_coverable_detail_cell_count: int | None = None
    reward_terminal_class: RewardTerminalClass = RewardTerminalClass.CONTINUE
    task_resample_required: bool = False
    task_resample_reason: str | None = None
    reselection_required: bool = False
    reselection_reason: str | None = None


@dataclass(frozen=True, slots=True)
class GroundReselection:
    """A planner-suppressed ground candidate rejected before any motion."""

    reason_code: str

    def __post_init__(self) -> None:
        if not isinstance(self.reason_code, str) or not self.reason_code:
            raise ValueError("ground reselection reason is invalid")


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
    coverage_before: float = 0.0
    coverage_after: float = 0.0
    priority_before: float = 0.0
    priority_after: float = 0.0
    executed_path_length_m: float = 0.0


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
        planning_failure_refresher: (
            Callable[
                [str, CandidateDisposition, str],
                BoundaryObservationResult,
            ]
            | None
        ) = None,
        require_sensor_closed_loop: bool = False,
        require_identity_bound_request: bool = False,
        reference_executor: Callable[
            [MotionReference], ReferenceExecutionResult
        ] | None = None,
        committed_hop_executor: Callable[
            [], CommittedHopExecutionFeedback
        ] | None = None,
        ground_option_continuation_builder: (
            Callable[[ObservationIdentity], object] | None
        ) = None,
        ground_option_distance_provider: Callable[[], float] | None = None,
        ground_option_clearer: Callable[[], None] | None = None,
        candidate_diagnostics_provider: (
            Callable[[], CandidateDiagnostics] | None
        ) = None,
        candidate_decision_snapshot_provider: (
            Callable[[], CandidateDecisionSnapshot] | None
        ) = None,
        remaining_coverable_detail_cell_count_provider: (
            Callable[[], int | None] | None
        ) = None,
        ground_start_qualification_provider: (
            Callable[[], str | None] | None
        ) = None,
        plan_cost_scale: float = 1.0,
        planner_elapsed_scale_s: float = 1.0,
        include_planner_wall_time_in_reward: bool = True,
        task_scale_m: float = 1.0,
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
            if not callable(planning_failure_refresher):
                raise ValueError(
                    "sensor-closed environment requires a planning failure refresher"
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
        if (
            planning_failure_refresher is not None
            and not callable(planning_failure_refresher)
        ):
            raise ValueError("planning failure refresher must be callable")
        if type(require_identity_bound_request) is not bool:
            raise ValueError("identity-bound request flag must be boolean")
        if type(include_planner_wall_time_in_reward) is not bool:
            raise ValueError("planner wall-time reward flag must be boolean")
        if (
            not isinstance(task_scale_m, (int, float))
            or isinstance(task_scale_m, bool)
            or not math.isfinite(float(task_scale_m))
            or float(task_scale_m) <= 0.0
        ):
            raise ValueError("task scale must be finite and positive")
        for name, callback in (
            ("candidate diagnostics provider", candidate_diagnostics_provider),
            (
                "candidate decision snapshot provider",
                candidate_decision_snapshot_provider,
            ),
            (
                "remaining-coverable provider",
                remaining_coverable_detail_cell_count_provider,
            ),
            (
                "ground start qualification provider",
                ground_start_qualification_provider,
            ),
            (
                "ground option continuation builder",
                ground_option_continuation_builder,
            ),
            ("ground option distance provider", ground_option_distance_provider),
            ("ground option clearer", ground_option_clearer),
        ):
            if callback is not None and not callable(callback):
                raise ValueError(f"{name} must be callable")
        ground_callbacks = (
            ground_option_continuation_builder,
            ground_option_distance_provider,
            ground_option_clearer,
        )
        if any(callback is not None for callback in ground_callbacks) and not all(
            callback is not None for callback in ground_callbacks
        ):
            raise ValueError("ground option callbacks must be configured together")
        if all(callback is not None for callback in ground_callbacks):
            if platform_type not in {"WHEELED", "LEGGED"}:
                raise ValueError("ground option callbacks require a ground platform")
            if not require_identity_bound_request:
                raise ValueError("ground options require identity-bound requests")
        if ground_start_qualification_provider is not None and platform_type not in {
            "WHEELED",
            "LEGGED",
        }:
            raise ValueError(
                "ground start qualification requires a ground platform"
            )
        self._observation = _clone_observation(initial_observation)
        self._observation_provider = observation_provider
        self._observation_boundary_controller = observation_boundary_controller
        self._planning_failure_refresher = planning_failure_refresher
        self._require_sensor_closed_loop = require_sensor_closed_loop
        self._require_identity_bound_request = require_identity_bound_request
        self._reference_executor = reference_executor
        self._committed_hop_executor = committed_hop_executor
        self._ground_option_continuation_builder = (
            ground_option_continuation_builder
        )
        self._ground_option_distance_provider = ground_option_distance_provider
        self._ground_option_clearer = ground_option_clearer
        self._candidate_diagnostics_provider = candidate_diagnostics_provider
        self._candidate_decision_snapshot_provider = (
            candidate_decision_snapshot_provider
        )
        self._remaining_coverable_detail_cell_count_provider = (
            remaining_coverable_detail_cell_count_provider
        )
        self._ground_start_qualification_provider = (
            ground_start_qualification_provider
        )
        self._plan_cost_scale = plan_cost_scale
        self._planner_elapsed_scale_s = planner_elapsed_scale_s
        self._include_planner_wall_time_in_reward = (
            include_planner_wall_time_in_reward
        )
        self._task_scale_m = float(task_scale_m)
        self._cumulative_executed_path_m = 0.0
        self._rollout_discarded = False
        self._training_stopped = False
        initial_ratio = (
            observation_boundary_controller.coverage_ratio
            if require_sensor_closed_loop
            else 0.0
        )
        self._initial_task_resample_pending = formal_success_first_crossing(
            0.0, initial_ratio
        )
        self._episode_terminated = False
        self._committed_output: PlannerOutput | None = None
        self._hopper_commitment_id: str | None = None
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

    def current_candidate_diagnostics(self) -> CandidateDiagnostics:
        """Return authoritative producer diagnostics for this boundary."""
        if self._candidate_diagnostics_provider is None:
            selected = int(self._observation.candidate_mask.sum().item())
            diagnostics = CandidateDiagnostics(
                physical_candidate_universe_count=selected,
                selected_policy_candidate_count=selected,
                available_candidate_count=selected,
            )
        else:
            diagnostics = self._candidate_diagnostics_provider()
            if not isinstance(diagnostics, CandidateDiagnostics):
                self._fail_closed(
                    "candidate diagnostics provider returned invalid data"
                )
        return diagnostics

    def current_candidate_decision_snapshot(self) -> CandidateDecisionSnapshot:
        """Return the one authoritative candidate-pipeline decision snapshot."""
        provider = self._candidate_decision_snapshot_provider
        if provider is not None:
            snapshot = provider()
            if not isinstance(snapshot, CandidateDecisionSnapshot):
                self._fail_closed(
                    "candidate decision snapshot provider returned invalid data"
                )
            return snapshot
        diagnostics = self.current_candidate_diagnostics()
        selected = diagnostics.selected_policy_candidate_count
        universe = diagnostics.physical_candidate_universe_count
        if self._platform_type == "HOPPER":
            raw = universe + diagnostics.physical_unreachable_count
            positive = max(0, universe - diagnostics.zero_gain_count)
            return CandidateDecisionSnapshot(
                snapshot_id="0" * 64,
                frontier_segment_count=0,
                raw_candidate_count=raw,
                fine_pose_candidate_count=universe,
                globally_reachable_candidate_count=universe,
                positive_gain_candidate_count=positive,
                selected_policy_candidate_count=selected,
                untried_reserve_count=diagnostics.untried_reserve_count,
                planner_rejected_current_snapshot_count=(
                    diagnostics.planner_failed_current_snapshot_count
                ),
                candidate_set_sha256="0" * 64,
                global_search_call_count=0,
                global_search_elapsed_s=0.0,
                candidate_refresh_elapsed_s=0.0,
                pipeline_kind="HOPPER_LANDING",
                representable_landing_sha256="0" * 64,
                raw_known_landing_count=raw,
                eligible_landing_count=universe,
                predicted_positive_landing_count=positive,
                visited_landing_count=0,
            )
        frontier = 0
        raw = fine = reachable = positive = 0
        rejected = diagnostics.planner_failed_current_snapshot_count
        reserve = diagnostics.untried_reserve_count
        if selected > 0:
            positive = reachable = fine = raw = max(universe, selected)
            frontier = max(1, (raw + 2) // 3)
        elif diagnostics.zero_gain_count > 0:
            reachable = fine = raw = max(diagnostics.zero_gain_count, 1)
            frontier = max(1, (raw + 2) // 3)
            rejected = reserve = 0
        elif diagnostics.physical_unreachable_count > 0:
            fine = raw = max(diagnostics.physical_unreachable_count, 1)
            frontier = max(1, (raw + 2) // 3)
            rejected = reserve = 0
        elif universe > 0 and rejected == universe:
            positive = reachable = fine = raw = universe
            frontier = max(1, (raw + 2) // 3)
            reserve = 0
        else:
            rejected = reserve = 0
        return CandidateDecisionSnapshot(
            snapshot_id="0" * 64,
            frontier_segment_count=frontier,
            raw_candidate_count=raw,
            fine_pose_candidate_count=fine,
            globally_reachable_candidate_count=reachable,
            positive_gain_candidate_count=positive,
            selected_policy_candidate_count=selected,
            untried_reserve_count=reserve,
            planner_rejected_current_snapshot_count=rejected,
            candidate_set_sha256="0" * 64,
            global_search_call_count=1,
            global_search_elapsed_s=0.0,
            candidate_refresh_elapsed_s=0.0,
        )

    def snapshot_stable_state(self) -> dict[str, object]:
        """Return the dynamic fields not owned by the observation controller."""
        if self._execution_state not in {
            "DECISION_BOUNDARY",
            "GROUND_HOLD",
            "LANDED_HOLD",
        }:
            raise EnvironmentInvariantError(
                "formal snapshot requires a stable execution boundary"
            )
        return {
            "execution_state": self._execution_state,
            "cumulative_executed_path_m": self._cumulative_executed_path_m,
        }

    def restore_stable_state(
        self,
        *,
        execution_state: str,
        cumulative_executed_path_m: float = 0.0,
    ) -> None:
        """Reapply non-reveal state after deterministic observation replay."""
        if execution_state not in {
            "DECISION_BOUNDARY",
            "GROUND_HOLD",
            "LANDED_HOLD",
        }:
            raise EnvironmentInvariantError(
                "formal restore requires a stable execution boundary"
            )
        identity = self._observation.observation_identities[0]
        if identity.execution_state != execution_state:
            raise EnvironmentInvariantError(
                "restored execution state differs from observation identity"
            )
        if (
            not isinstance(cumulative_executed_path_m, (int, float))
            or isinstance(cumulative_executed_path_m, bool)
            or not math.isfinite(float(cumulative_executed_path_m))
            or float(cumulative_executed_path_m) < 0.0
        ):
            raise EnvironmentInvariantError(
                "restored cumulative executed path is invalid"
            )
        self._execution_state = execution_state
        self._cumulative_executed_path_m = float(
            cumulative_executed_path_m
        )

    def step(
        self,
        action: PolicyAction,
        *,
        expected_identity: ObservationIdentity | None = None,
    ) -> PlannerTransition:
        if self._episode_terminated:
            raise RuntimeError("terminated episode requires reset")
        if self._execution_state in {"JUMP_COMMITTED", "IN_FLIGHT"}:
            self._fail_closed("committed hopper cannot accept a new policy action")
        request, prepared = self._build_plan_request(action, expected_identity)
        output = self._bridge.plan(request)
        self._validate_output(output)
        return self._transition_for_output(output, prepared)

    def _transition_for_output(
        self,
        output: PlannerOutput,
        prepared: PreparedPlanRequest | None,
    ) -> PlannerTransition:
        if output.directive == ExecutionDirective.CONTINUE_COMMITTED_HOP:
            return self._advance_committed_hop_without_policy(output)
        if output.reference is None:
            if output.outcome == PlanningOutcome.CANCELED:
                return self._hold_transition(
                    outcome=output.outcome,
                    directive=output.directive,
                    reason_code=output.reason_code,
                    planner_elapsed=output.diagnostics.elapsed,
                    terminal_reason=TerminalReason.CANCELED,
                )
            if output.outcome in {
                PlanningOutcome.INVALID_REQUEST,
                PlanningOutcome.NUMERICAL_FAILURE,
                PlanningOutcome.RESOURCE_EXHAUSTED,
                PlanningOutcome.ACTIVE_REFERENCE_INVALIDATED,
            }:
                return self._hold_transition(
                    outcome=output.outcome,
                    directive=output.directive,
                    reason_code=output.reason_code,
                    planner_elapsed=output.diagnostics.elapsed,
                    terminal_reason=TerminalReason.HARD_FAILURE,
                )
            self._refresh_after_planning_failure(output, prepared)
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
        if self._episode_terminated:
            raise RuntimeError("terminated episode requires reset")
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
                terminal_reason=(
                    None if transition is None else transition.terminal_reason
                ),
                remaining_coverable_detail_cell_count=(
                    None
                    if transition is None
                    else transition.remaining_coverable_detail_cell_count
                ),
                reward_terminal_class=(
                    RewardTerminalClass.CONTINUE
                    if transition is None
                    else transition.reward_terminal_class
                ),
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
            return self._no_candidate_boundary()
        ground_result = (
            self._advance_ground_option(action, expected_identity)
            if self._ground_option_continuation_builder is not None
            else self.step(action, expected_identity=expected_identity)
        )
        if isinstance(ground_result, GroundReselection):
            return DecisionBoundaryResult(
                execution_state=self._execution_state,
                policy_decisions_consumed=1,
                reselection_required=True,
                reselection_reason=ground_result.reason_code,
            )
        transition = ground_result
        if self._platform_type == "HOPPER" and self._execution_state in {
            "JUMP_COMMITTED",
            "IN_FLIGHT",
        }:
            transition = self._complete_committed_hop_transition(transition)
        if transition.terminated and transition.terminal_reason is None:
            transition = self._with_flag_terminal_reason(transition)
        elif not transition.terminated:
            reason, remaining = self._audit_current_candidate_boundary()
            if reason is not None:
                transition = replace(
                    transition,
                    episode_ended_without_success=True,
                    terminated=True,
                    terminal_reason=reason,
                    remaining_coverable_detail_cell_count=remaining,
                    reward_terminal_class=terminal_class_for_reason(reason),
                )
        if transition.terminated:
            self._episode_terminated = True
        return DecisionBoundaryResult(
            execution_state=self._execution_state,
            transition=transition,
            policy_decisions_consumed=1,
            terminal_reason=transition.terminal_reason,
            remaining_coverable_detail_cell_count=(
                transition.remaining_coverable_detail_cell_count
            ),
            reward_terminal_class=transition.reward_terminal_class,
        )

    def refresh_decision_boundary(self) -> DecisionBoundaryResult:
        """Refresh producer input and classify a ground boundary without policy."""
        if self._episode_terminated:
            raise RuntimeError("terminated episode requires reset")
        if self._initial_task_resample_pending:
            self._initial_task_resample_pending = False
            self._episode_terminated = True
            return DecisionBoundaryResult(
                execution_state="TASK_RESAMPLE_REQUIRED",
                reward_terminal_class=RewardTerminalClass.INVALID_TRANSITION,
                task_resample_required=True,
                task_resample_reason="INITIAL_OBSERVATION_ALREADY_SUCCESSFUL",
            )
        if self._execution_state in {"JUMP_COMMITTED", "IN_FLIGHT"}:
            self._fail_closed(
                "committed hopper requires execution feedback before refresh"
            )
        self._refresh_ground_observation()
        start_reason = self._ground_start_qualification_reason()
        if start_reason is not None:
            self._episode_terminated = True
            return DecisionBoundaryResult(
                execution_state="TASK_RESAMPLE_REQUIRED",
                reward_terminal_class=RewardTerminalClass.INVALID_TRANSITION,
                task_resample_required=True,
                task_resample_reason=start_reason,
            )
        reason, remaining = self._audit_current_candidate_boundary()
        if reason is None:
            return DecisionBoundaryResult(execution_state="DECISION_READY")
        self._episode_terminated = True
        return DecisionBoundaryResult(
            execution_state="NO_CANDIDATES",
            terminal_reason=reason,
            remaining_coverable_detail_cell_count=remaining,
            reward_terminal_class=terminal_class_for_reason(reason),
        )

    def _no_candidate_boundary(self) -> DecisionBoundaryResult:
        reason, remaining = self._audit_current_candidate_boundary()
        if reason is None:
            self._fail_closed("candidate availability is inconsistent")
        self._episode_terminated = True
        return DecisionBoundaryResult(
            execution_state="NO_CANDIDATES",
            terminal_reason=reason,
            remaining_coverable_detail_cell_count=remaining,
            reward_terminal_class=terminal_class_for_reason(reason),
        )

    def _audit_current_candidate_boundary(
        self,
    ) -> tuple[TerminalReason | None, int | None]:
        snapshot = self.current_candidate_decision_snapshot()
        try:
            reason = _audit_candidate_boundary(snapshot, self._platform_type)
        except EnvironmentInvariantError as error:
            self._fail_closed(str(error))
        selected = int(self._observation.candidate_mask.sum().item())
        if selected != snapshot.selected_policy_candidate_count:
            self._fail_closed("candidate availability differs from observation")
        return reason, (
            self._remaining_coverable_count() if reason is not None else None
        )

    def _remaining_coverable_count(self) -> int | None:
        provider = self._remaining_coverable_detail_cell_count_provider
        remaining = None if provider is None else provider()
        if remaining is not None and (
            type(remaining) is not int or remaining < 0
        ):
            self._fail_closed("remaining-coverable provider returned invalid data")
        return remaining

    def _ground_start_qualification_reason(self) -> str | None:
        provider = self._ground_start_qualification_provider
        if provider is None:
            return None
        reason = provider()
        expected = {
            "WHEELED": "WHEEL_START_NOT_SAFE",
            "LEGGED": "LEGGED_START_NOT_SAFE",
        }.get(self._platform_type)
        if reason is None:
            return None
        if not isinstance(reason, str) or not reason or reason != expected:
            self._fail_closed(
                "ground start qualification provider returned invalid data"
            )
        return reason

    def _with_flag_terminal_reason(
        self, transition: PlannerTransition
    ) -> PlannerTransition:
        if transition.success_first_crossing:
            reason = TerminalReason.SUCCESS
        elif transition.terminated:
            reason = TerminalReason.HARD_FAILURE
        else:
            return transition
        return replace(
            transition,
            terminal_reason=reason,
            remaining_coverable_detail_cell_count=(
                self._remaining_coverable_count()
            ),
            reward_terminal_class=terminal_class_for_reason(reason),
        )

    def _build_plan_request(
        self,
        action: PolicyAction,
        expected_identity: ObservationIdentity | None,
    ) -> tuple[TrainingPlanRequest | object, PreparedPlanRequest | None]:
        if not self._require_identity_bound_request:
            return self._request_builder(action), None
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
        return prepared.request, prepared

    def _build_ground_continuation_request(
        self, expected_identity: ObservationIdentity
    ) -> PreparedPlanRequest:
        builder = self._ground_option_continuation_builder
        if builder is None:
            self._fail_closed("ground option continuation builder is unavailable")
        prepared = builder(expected_identity)
        if not isinstance(prepared, PreparedPlanRequest):
            self._fail_closed(
                "ground continuation builder must return PreparedPlanRequest"
            )
        if prepared.identity != expected_identity:
            self._fail_closed("ground continuation observation identity is stale")
        if (
            prepared.request.state_time.nanoseconds_since_epoch
            != expected_identity.state_time_ns
        ):
            self._fail_closed(
                "ground continuation state time does not match observation identity"
            )
        return prepared

    def _ground_option_distance_m(self) -> float:
        provider = self._ground_option_distance_provider
        if provider is None:
            self._fail_closed("ground option distance provider is unavailable")
        distance_m = provider()
        if (
            not isinstance(distance_m, (int, float))
            or isinstance(distance_m, bool)
            or not math.isfinite(float(distance_m))
            or float(distance_m) < 0.0
        ):
            self._fail_closed("ground option distance is invalid")
        return float(distance_m)

    def _ground_progress_signature(
        self,
        output: PlannerOutput,
        *,
        candidate_id: str,
    ) -> tuple[object, ...]:
        identity = self._observation.observation_identities[0]
        hierarchical = output.diagnostics.hierarchical
        route_cursor = (
            None if hierarchical is None else int(hierarchical.route_cursor)
        )
        reference = output.reference
        if reference is None:
            self._fail_closed("ground progress signature requires a reference")
        points = getattr(reference.data, "points", None)
        if points:
            position = points[-1].pose.position_m
            endpoint: tuple[object, ...] = (
                "trajectory",
                float(position.x),
                float(position.y),
                float(position.z),
            )
            if not all(
                math.isfinite(value) for value in endpoint[1:]
            ):
                self._fail_closed("ground reference endpoint is non-finite")
        else:
            endpoint = ("reference", str(reference.plan_id))
        return identity, route_cursor, endpoint, candidate_id

    def _advance_ground_option(
        self,
        action: PolicyAction,
        expected_identity: ObservationIdentity,
    ) -> PlannerTransition | GroundReselection:
        clearer = self._ground_option_clearer
        if clearer is None:
            self._fail_closed("ground option clearer is unavailable")
        transitions: list[PlannerTransition] = []
        try:
            request, prepared = self._build_plan_request(
                action, expected_identity
            )
            if not isinstance(prepared, PreparedPlanRequest):
                self._fail_closed(
                    "ground option requires prepared candidate identity"
                )
            locked_candidate_id = prepared.candidate_id
            seen_progress_signatures: set[tuple[object, ...]] = set()
            while True:
                output = self._bridge.plan(request)
                self._validate_output(output)
                if output.reference is None and not transitions:
                    suppressible = (
                        output.outcome
                        in {
                            PlanningOutcome.GOAL_INFEASIBLE,
                            PlanningOutcome.NO_KNOWN_SAFE_ROUTE,
                        }
                        and output.candidate_disposition
                        == CandidateDisposition.SUPPRESS_FOR_CURRENT_PHYSICAL_SNAPSHOT
                    )
                    if not suppressible:
                        self._fail_closed(
                            "GROUND_PRE_MOTION_REJECTION_NOT_SUPPRESSIBLE"
                        )
                    if (
                        not isinstance(output.reason_code, str)
                        or not output.reason_code
                    ):
                        self._fail_closed(
                            "GROUND_PRE_MOTION_REJECTION_REASON_INVALID"
                        )
                    self._refresh_after_planning_failure(output, prepared)
                    return GroundReselection(reason_code=output.reason_code)
                transition = self._transition_for_output(output, prepared)
                transitions.append(transition)
                if output.reference is None:
                    return self._aggregate_ground_transitions(transitions)
                if (
                    transition.terminated
                    or transition.success_first_crossing
                    or transition.hard_safety_violation
                ):
                    return self._aggregate_ground_transitions(transitions)
                current_distance_m = self._ground_option_distance_m()
                if current_distance_m <= (
                    _GROUND_OPTION_TARGET_TOLERANCE_M
                    + _GROUND_OPTION_PROGRESS_EPSILON_M
                ):
                    return self._aggregate_ground_transitions(transitions)
                progress_signature = self._ground_progress_signature(
                    output,
                    candidate_id=locked_candidate_id,
                )
                if progress_signature in seen_progress_signatures:
                    self._fail_closed(
                        "GROUND_OPTION_STALLED"
                    )
                seen_progress_signatures.add(progress_signature)
                continuation_identity = self._observation.observation_identities[0]
                prepared = self._build_ground_continuation_request(
                    continuation_identity
                )
                if prepared.candidate_id != locked_candidate_id:
                    self._fail_closed(
                        "ground option candidate identity changed"
                    )
                request = prepared.request
                request.continuation = output.continuation
        finally:
            clearer()

    def _aggregate_ground_transitions(
        self, transitions: list[PlannerTransition]
    ) -> PlannerTransition:
        if not transitions:
            self._fail_closed("ground option aggregation is empty")
        values = tuple(
            float(value)
            for transition in transitions
            for value in (
                transition.mission_observed_delta,
                transition.priority_observed_delta,
                transition.normalized_plan_or_execution_cost,
                transition.normalized_macro_step_time,
            )
        )
        if not all(math.isfinite(value) for value in values):
            self._fail_closed("ground option aggregation contains non-finite data")
        mission_observed_delta = sum(
            transition.mission_observed_delta for transition in transitions
        )
        priority_observed_delta = sum(
            transition.priority_observed_delta for transition in transitions
        )
        normalized_cost = sum(
            transition.normalized_plan_or_execution_cost
            for transition in transitions
        )
        normalized_time = sum(
            transition.normalized_macro_step_time for transition in transitions
        )
        if not all(
            math.isfinite(float(value))
            for value in (
                mission_observed_delta,
                priority_observed_delta,
                normalized_cost,
                normalized_time,
            )
        ):
            self._fail_closed("ground option totals are non-finite")
        success_crossings = sum(
            transition.success_first_crossing for transition in transitions
        )
        if success_crossings > 1:
            self._fail_closed("ground option emitted success more than once")
        final = transitions[-1]
        first = transitions[0]
        return PlannerTransition(
            next_observation=_clone_observation(final.next_observation),
            mission_observed_delta=mission_observed_delta,
            priority_observed_delta=priority_observed_delta,
            normalized_plan_or_execution_cost=normalized_cost,
            normalized_macro_step_time=normalized_time,
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
            cancellation_expected=any(
                transition.cancellation_expected for transition in transitions
            ),
            cpp_exception=next(
                (
                    transition.cpp_exception
                    for transition in transitions
                    if transition.cpp_exception is not None
                ),
                None,
            ),
            planning_outcome=final.planning_outcome,
            execution_directive=final.execution_directive,
            reason_code=final.reason_code,
            terminated=final.terminated,
            execution_events=self._aggregate_execution_events(
                transitions,
                context="ground option",
                infer_missing_hopper_states=False,
            ),
            terminal_reason=final.terminal_reason,
            remaining_coverable_detail_cell_count=(
                final.remaining_coverable_detail_cell_count
            ),
            reward_terminal_class=final.reward_terminal_class,
            coverage_before=first.coverage_before,
            coverage_after=final.coverage_after,
            priority_before=first.priority_before,
            priority_after=final.priority_after,
            executed_path_length_m=sum(
                transition.executed_path_length_m
                for transition in transitions
            ),
            cumulative_executed_path_m_before=(
                first.cumulative_executed_path_m_before
            ),
            cumulative_executed_path_m_after=(
                final.cumulative_executed_path_m_after
            ),
            task_scale_m=self._task_scale_m,
        )

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
        self._observation = _clone_observation(observation)
        return self._observation

    def _refresh_ground_observation(self) -> None:
        if self._observation_provider is None:
            return
        observation = self._observation_provider()
        if not isinstance(observation, PolicyBatch):
            self._fail_closed("observation provider returned invalid data")
        self._install_observation(observation)

    def _begin_hopper_observation_commitment(
        self, output: PlannerOutput
    ) -> None:
        if not self._require_sensor_closed_loop:
            return
        if self._platform_type != "HOPPER":
            self._fail_closed("hopper observation commitment requires HOPPER")
        controller = self._observation_boundary_controller
        if not isinstance(controller, ObservationBoundaryController):
            self._fail_closed("observation boundary controller is unavailable")
        reference = output.reference
        data = None if reference is None else reference.data
        if not isinstance(data, HopReference) or len(data.segments) != 1:
            self._fail_closed("hopper reference must contain one certified hop")
        segment = data.segments[0]
        landing = segment.nominal_landing_point_m
        flight_time_s = float(segment.flight_time.total_seconds())
        flight_tube_radius_m = float(segment.flight_tube_radius_m)
        values = (
            float(landing.x),
            float(landing.y),
            float(landing.z),
            flight_time_s,
            flight_tube_radius_m,
        )
        if not all(math.isfinite(value) for value in values):
            self._fail_closed("hopper commitment contains non-finite values")
        takeoff = controller.current_pose
        if takeoff is None:
            self._fail_closed("hopper commitment takeoff pose is unavailable")
        landing_pose = type(takeoff)(
            float(landing.x),
            float(landing.y),
            0.0,
            "map",
            float(landing.z),
        )
        identity = self._observation.observation_identities[0]
        token = sha256(
            repr(
                (
                    reference.plan_id,
                    segment.segment_id,
                    identity,
                    takeoff,
                    landing_pose,
                    flight_time_s,
                    flight_tube_radius_m,
                )
            ).encode("utf-8")
        ).hexdigest()
        try:
            controller.begin_hopper_trajectory(
                HopperObservationCommitment(
                    commitment_id=token,
                    takeoff_pose=takeoff,
                    expected_landing_pose=landing_pose,
                    flight_time_s=flight_time_s,
                    flight_tube_radius_m=flight_tube_radius_m,
                )
            )
        except (TypeError, ValueError, RuntimeError) as error:
            self._fail_closed(str(error))
        self._hopper_commitment_id = token

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
            controller = self._observation_boundary_controller
            if not isinstance(controller, ObservationBoundaryController):
                self._fail_closed(
                    "observation boundary controller is unavailable"
                )
            commitment_id = None
            if controller.hopper_commitment_active:
                commitment_id = (
                    self._hopper_commitment_id
                    if feedback.commitment_id is None
                    else feedback.commitment_id
                )
                if not isinstance(commitment_id, str):
                    self._fail_closed("committed hop identity is unavailable")
            if not isinstance(feedback.trajectory_points, tuple) or any(
                not isinstance(point, HopperTrajectoryPoint)
                for point in feedback.trajectory_points
            ):
                self._fail_closed("committed hop trajectory points are invalid")
            if feedback.trajectory_points:
                if not isinstance(commitment_id, str):
                    self._fail_closed("committed hop trajectory has no commitment")
                try:
                    controller.append_hopper_trajectory(
                        commitment_id,
                        feedback.trajectory_points,
                        within_certified_flight_tube=(
                            feedback.trajectory_within_certified_flight_tube
                        ),
                    )
                except (TypeError, ValueError, RuntimeError) as error:
                    self._fail_closed(str(error))
            boundary = self._apply_sensor_boundary(
                feedback.execution_state,
                feedback.sensor_boundary_evidence,
                hopper_commitment_id=commitment_id,
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
                success_first_crossing=boundary.success_first_crossing,
                episode_ended_without_success=(
                    feedback.episode_ended_without_success
                    and not boundary.success_first_crossing
                ),
                terminated=(
                    feedback.terminated or boundary.success_first_crossing
                ),
                coverage_before=boundary.coverage_before,
                coverage_after=boundary.coverage_after,
                priority_before=boundary.priority_before,
                priority_after=boundary.priority_after,
                executed_path_length_m=boundary.executed_path_length_m,
            )
        else:
            self._install_observation(feedback.next_observation)
        self._execution_state = feedback.execution_state
        if feedback.execution_state == "LANDED_HOLD":
            self._hopper_commitment_id = None
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
        first = transitions[0]
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
            terminal_reason=final.terminal_reason,
            remaining_coverable_detail_cell_count=(
                final.remaining_coverable_detail_cell_count
            ),
            reward_terminal_class=final.reward_terminal_class,
            coverage_before=first.coverage_before,
            coverage_after=final.coverage_after,
            priority_before=first.priority_before,
            priority_after=final.priority_after,
            executed_path_length_m=sum(
                transition.executed_path_length_m
                for transition in transitions
            ),
            cumulative_executed_path_m_before=(
                first.cumulative_executed_path_m_before
            ),
            cumulative_executed_path_m_after=(
                final.cumulative_executed_path_m_after
            ),
            task_scale_m=self._task_scale_m,
        )

    def _aggregate_execution_events(
        self,
        transitions: list[PlannerTransition],
        *,
        context: str = "committed hopper",
        infer_missing_hopper_states: bool = True,
    ) -> ExecutionEvents:
        totals = {name: 0 for name in _EXECUTION_EVENT_COUNT_FIELDS}
        commitment_states: list[str] = []
        selected_action_observed_safe = False
        for transition in transitions:
            events = transition.execution_events
            if not isinstance(events, ExecutionEvents):
                self._fail_closed(f"{context} execution events are invalid")
            for name in _EXECUTION_EVENT_COUNT_FIELDS:
                value = getattr(events, name)
                if type(value) is not int or value < 0:
                    self._fail_closed(
                        f"{context} event count must be non-negative"
                    )
                totals[name] += value
            if type(events.selected_action_observed_safe) is not bool:
                self._fail_closed(f"{context} safe-action fact is invalid")
            selected_action_observed_safe = (
                selected_action_observed_safe
                or events.selected_action_observed_safe
            )
            states = events.hopper_commitment_states
            if not isinstance(states, tuple) or any(
                state not in {"JUMP_COMMITTED", "IN_FLIGHT", "LANDED_HOLD"}
                for state in states
            ):
                self._fail_closed(f"{context} event states are invalid")
            if states:
                if not infer_missing_hopper_states:
                    self._fail_closed(
                        "ground option must not emit hopper commitment states"
                    )
                commitment_states.extend(states)
            elif infer_missing_hopper_states:
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
        terminal_reason = (
            TerminalReason.SUCCESS
            if feedback.success_first_crossing
            else (
                TerminalReason.HARD_FAILURE
                if feedback.terminated
                else None
            )
        )
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
                + self._normalized_planner_wall_time(
                    output,
                    include=include_planner_contribution,
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
            terminal_reason=terminal_reason,
            remaining_coverable_detail_cell_count=(
                self._remaining_coverable_count()
                if feedback.terminated
                else None
            ),
            reward_terminal_class=terminal_class_for_reason(terminal_reason),
            **self._transition_execution_facts(feedback),
        )

    def _execute_reference_until_decision_boundary(
        self, output: PlannerOutput
    ) -> PlannerTransition:
        if self._reference_executor is None:
            self._fail_closed("reference executor is required for executable output")
        execution = self._reference_executor(output.reference)
        if not isinstance(execution, ReferenceExecutionResult):
            self._fail_closed("reference executor returned an invalid result")
        if (
            self._platform_type == "HOPPER"
            and execution.execution_state == "JUMP_COMMITTED"
        ):
            self._begin_hopper_observation_commitment(output)
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
                success_first_crossing=boundary.success_first_crossing,
                episode_ended_without_success=(
                    execution.episode_ended_without_success
                    and not boundary.success_first_crossing
                ),
                terminated=(
                    execution.terminated or boundary.success_first_crossing
                ),
                coverage_before=boundary.coverage_before,
                coverage_after=boundary.coverage_after,
                priority_before=boundary.priority_before,
                priority_after=boundary.priority_after,
                executed_path_length_m=boundary.executed_path_length_m,
            )
        else:
            self._install_observation(execution.next_observation)
        if self._platform_type == "HOPPER" and self._execution_state in {
            "JUMP_COMMITTED",
            "IN_FLIGHT",
        }:
            self._committed_output = output
        best_cost = output.diagnostics.best_cost
        terminal_reason = (
            TerminalReason.SUCCESS
            if execution.success_first_crossing
            else (
                TerminalReason.HARD_FAILURE
                if execution.terminated
                else None
            )
        )
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
                + self._normalized_planner_wall_time(output)
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
            terminal_reason=terminal_reason,
            remaining_coverable_detail_cell_count=(
                self._remaining_coverable_count()
                if execution.terminated
                else None
            ),
            reward_terminal_class=terminal_class_for_reason(terminal_reason),
            **self._transition_execution_facts(execution),
        )

    def _transition_execution_facts(
        self,
        execution: ReferenceExecutionResult | CommittedHopExecutionFeedback,
    ) -> dict[str, float]:
        path_length_m = float(execution.executed_path_length_m)
        cumulative_before = self._cumulative_executed_path_m
        cumulative_after = cumulative_before + path_length_m
        if not math.isfinite(cumulative_after):
            self._fail_closed("cumulative executed path is non-finite")
        self._cumulative_executed_path_m = cumulative_after
        return {
            "coverage_before": float(execution.coverage_before),
            "coverage_after": float(execution.coverage_after),
            "priority_before": float(execution.priority_before),
            "priority_after": float(execution.priority_after),
            "executed_path_length_m": path_length_m,
            "cumulative_executed_path_m_before": cumulative_before,
            "cumulative_executed_path_m_after": cumulative_after,
            "task_scale_m": self._task_scale_m,
        }

    def _apply_sensor_boundary(
        self,
        execution_state: str,
        evidence: SensorBoundaryEvidence | None,
        *,
        hopper_commitment_id: str | None = None,
    ) -> BoundaryObservationResult:
        controller = self._observation_boundary_controller
        if not isinstance(controller, ObservationBoundaryController):
            self._fail_closed("observation boundary controller is unavailable")
        try:
            return controller.after_execution(
                platform_type=self._platform_type,
                execution_state=execution_state,
                evidence=evidence,
                hopper_commitment_id=hopper_commitment_id,
            )
        except (TypeError, ValueError, RuntimeError) as error:
            self._fail_closed(str(error))

    def _refresh_after_planning_failure(
        self,
        output: PlannerOutput,
        prepared: PreparedPlanRequest | None,
    ) -> None:
        disposition = output.candidate_disposition
        if disposition not in {
            CandidateDisposition.KEEP,
            CandidateDisposition.SUPPRESS_FOR_CURRENT_PHYSICAL_SNAPSHOT,
        }:
            self._fail_closed("planner candidate disposition is invalid")
        refresher = self._planning_failure_refresher
        if refresher is None:
            return
        if not isinstance(prepared, PreparedPlanRequest):
            self._fail_closed(
                "planning failure refresh requires prepared candidate identity"
            )
        before_identity = self._observation.observation_identities[0]
        try:
            boundary = refresher(
                prepared.candidate_id,
                disposition,
                prepared.physical_snapshot_id,
            )
        except (TypeError, ValueError, RuntimeError) as error:
            self._fail_closed(str(error))
        if not isinstance(boundary, BoundaryObservationResult):
            self._fail_closed("planning failure refresher returned invalid data")
        after_identity = boundary.next_observation.observation_identities[0]
        if (
            not boundary.updated
            or boundary.mission_observed_delta != 0.0
            or boundary.priority_observed_delta != 0.0
            or boundary.success_first_crossing
            or after_identity.state_time_ns != before_identity.state_time_ns
            or after_identity == before_identity
        ):
            self._fail_closed(
                "planning failure refresh must be a zero-evidence revision"
            )
        self._install_observation(boundary.next_observation)

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
        terminal_reason: TerminalReason | None = None,
    ) -> PlannerTransition:
        terminal_class = terminal_class_for_reason(terminal_reason)
        controller = self._observation_boundary_controller
        coverage = (
            controller.coverage_ratio
            if isinstance(controller, ObservationBoundaryController)
            else 0.0
        )
        priority = (
            controller.priority_ratio
            if isinstance(controller, ObservationBoundaryController)
            else 0.0
        )
        return PlannerTransition(
            next_observation=_clone_observation(self._observation),
            mission_observed_delta=0.0,
            priority_observed_delta=0.0,
            normalized_plan_or_execution_cost=0.0,
            normalized_macro_step_time=(
                planner_elapsed.total_seconds() / self._planner_elapsed_scale_s
                if self._include_planner_wall_time_in_reward
                else 0.0
            ),
            executed_without_new_coverage=False,
            success_first_crossing=False,
            episode_ended_without_success=terminal_reason is not None,
            hard_safety_violation=False,
            cancellation_expected=terminal_reason is TerminalReason.CANCELED,
            cpp_exception=None,
            planning_outcome=outcome,
            execution_directive=directive,
            reason_code=reason_code,
            terminated=terminal_reason is not None,
            execution_events=ExecutionEvents(),
            terminal_reason=terminal_reason,
            remaining_coverable_detail_cell_count=(
                self._remaining_coverable_count()
                if terminal_reason is not None
                else None
            ),
            reward_terminal_class=terminal_class,
            coverage_before=coverage,
            coverage_after=coverage,
            priority_before=priority,
            priority_after=priority,
            executed_path_length_m=0.0,
            cumulative_executed_path_m_before=(
                self._cumulative_executed_path_m
            ),
            cumulative_executed_path_m_after=(
                self._cumulative_executed_path_m
            ),
            task_scale_m=self._task_scale_m,
        )

    def _normalized_planner_wall_time(
        self,
        output: PlannerOutput,
        *,
        include: bool = True,
    ) -> float:
        if not include or not self._include_planner_wall_time_in_reward:
            return 0.0
        return (
            output.diagnostics.elapsed.total_seconds()
            / self._planner_elapsed_scale_s
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
    planning_failure_refresher: (
        Callable[
            [str, CandidateDisposition, str], BoundaryObservationResult
        ]
        | None
    ) = None,
    require_sensor_closed_loop: bool = False,
    reference_executor: Callable[
        [MotionReference], ReferenceExecutionResult
    ] | None = None,
    committed_hop_executor: Callable[
        [], CommittedHopExecutionFeedback
    ] | None = None,
    ground_option_continuation_builder: (
        Callable[[ObservationIdentity], PreparedPlanRequest] | None
    ) = None,
    ground_option_distance_provider: Callable[[], float] | None = None,
    ground_option_clearer: Callable[[], None] | None = None,
    candidate_diagnostics_provider: (
        Callable[[], CandidateDiagnostics] | None
    ) = None,
    candidate_decision_snapshot_provider: (
        Callable[[], CandidateDecisionSnapshot] | None
    ) = None,
    remaining_coverable_detail_cell_count_provider: (
        Callable[[], int | None] | None
    ) = None,
    ground_start_qualification_provider: (
        Callable[[], str | None] | None
    ) = None,
    plan_cost_scale: float = 1.0,
    planner_elapsed_scale_s: float = 1.0,
    include_planner_wall_time_in_reward: bool = True,
    task_scale_m: float = 1.0,
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
    if require_sensor_closed_loop and not callable(planning_failure_refresher):
        raise ValueError(
            "sensor-closed environment requires a planning failure refresher"
        )
    return V3ExplorationEnvironment(
        platform_type=platform_type,
        bridge=PlannerBridge(),
        request_builder=request_builder,
        initial_observation=initial_observation,
        observation_provider=observation_provider,
        observation_boundary_controller=observation_boundary_controller,
        planning_failure_refresher=planning_failure_refresher,
        require_sensor_closed_loop=require_sensor_closed_loop,
        require_identity_bound_request=True,
        reference_executor=reference_executor,
        committed_hop_executor=committed_hop_executor,
        ground_option_continuation_builder=(
            ground_option_continuation_builder
        ),
        ground_option_distance_provider=ground_option_distance_provider,
        ground_option_clearer=ground_option_clearer,
        candidate_diagnostics_provider=candidate_diagnostics_provider,
        candidate_decision_snapshot_provider=(
            candidate_decision_snapshot_provider
        ),
        remaining_coverable_detail_cell_count_provider=(
            remaining_coverable_detail_cell_count_provider
        ),
        ground_start_qualification_provider=(
            ground_start_qualification_provider
        ),
        plan_cost_scale=plan_cost_scale,
        planner_elapsed_scale_s=planner_elapsed_scale_s,
        include_planner_wall_time_in_reward=(
            include_planner_wall_time_in_reward
        ),
        task_scale_m=task_scale_m,
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

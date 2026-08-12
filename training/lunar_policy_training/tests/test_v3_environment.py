from __future__ import annotations

from datetime import timedelta
from dataclasses import FrozenInstanceError
import pathlib
import sys
from types import SimpleNamespace


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "model_contract"))

from lunar_planner_training_bridge import (  # noqa: E402
    CandidateDisposition,
    ExecutionDirective,
    HierarchicalPlannerMetrics,
    MotionReference,
    PlannerDiagnostics,
    PlannerOutput,
    PlanningOutcome,
    TrainingPlanRequest,
    TrajectoryPoint,
)
import torch  # noqa: E402
import pytest  # noqa: E402
import lunar_policy_training.environment.v3_environment as v3_module  # noqa: E402

from lunar_policy_training.environment.candidate_builder import (  # noqa: E402
    CandidateDiagnostics,
)
from lunar_policy_training.environment.frontier_oracle import (  # noqa: E402
    FrontierOracleResult,
)
from lunar_policy_training.environment.macro_step import (  # noqa: E402
    ExecutionEvents,
    PolicyAction,
    TerminalReason,
)
from lunar_policy_training.environment.v3_environment import (  # noqa: E402
    CommittedHopExecutionFeedback,
    EnvironmentInvariantError,
    PreparedPlanRequest,
    ReferenceExecutionResult,
    V3ExplorationEnvironment,
)
from lunar_policy_training.environment.observation_boundary import (  # noqa: E402
    BoundaryObservationResult,
    ObservationBoundaryController,
)
from lunar_policy_training.policy.observation import (  # noqa: E402
    ObservationIdentity,
    PolicyBatch,
)


def test_terminal_reason_contract_uses_physical_opportunity_matrix() -> None:
    assert {reason.value for reason in TerminalReason} == {
        "SUCCESS",
        "NO_RECOVERABLE_OBSERVATION_STATE",
        "VISITED_EXHAUSTED",
        "ZERO_GAIN",
        "NO_TRANSIT_OPPORTUNITY",
        "PLANNER_BLOCKED_WITH_OPPORTUNITY",
        "HARD_FAILURE",
        "CANCELED",
    }


def _identity(**changes: object) -> ObservationIdentity:
    values = {
        "episode_id": "episode-1",
        "mission_revision": 1,
        "map_snapshot_id": "map-1",
        "robot_state_id": "robot-state-1",
        "state_time_ns": 1_000,
        "execution_state": "DECISION_BOUNDARY",
        "candidate_set_id": "candidates-1",
    }
    values.update(changes)
    return ObservationIdentity(**values)


def test_observation_identity_is_immutable_non_network_metadata() -> None:
    identity = _identity()
    observation = _observation(identity=identity)

    with pytest.raises(FrozenInstanceError):
        identity.map_snapshot_id = "map-2"  # type: ignore[misc]
    assert observation.input_names == (
        "prior_channels",
        "coverage_summary",
        "local_crop",
        "frontier_features",
        "pose_features",
        "candidate_mask",
        "platform_context",
    )


def _observation(
    platform_index: int = 0,
    *,
    candidate_mask: tuple[bool, ...] = (True, False),
    identity: ObservationIdentity | None = None,
) -> PolicyBatch:
    platform_context = torch.zeros((1, 3), dtype=torch.float32)
    platform_context[0, platform_index] = 1.0
    return PolicyBatch(
        prior_channels=torch.zeros((1, 7, 4, 4), dtype=torch.float32),
        coverage_summary=torch.zeros((1, 8, 4, 4), dtype=torch.float32),
        local_crop=torch.zeros((1, 8, 4, 4), dtype=torch.float32),
        frontier_features=torch.zeros((1, 2, 22), dtype=torch.float32),
        pose_features=torch.zeros((1, 5), dtype=torch.float32),
        candidate_mask=torch.tensor([candidate_mask], dtype=torch.bool),
        platform_context=platform_context,
        observation_identities=(identity or _identity(),),
    )


def _candidate_diagnostics(**changes: object) -> CandidateDiagnostics:
    values = {
        "physical_snapshot_id": "f" * 64,
        "physical_reachability_algorithm_id": "test/reachability",
        "physical_candidate_universe_count": 0,
        "selected_policy_candidate_count": 0,
        "available_candidate_count": 0,
        "untried_reserve_count": 0,
        "planner_failed_current_snapshot_count": 0,
        "zero_gain_count": 0,
        "visited_excluded_count": 0,
        "physical_unreachable_count": 0,
    }
    values.update(changes)
    return CandidateDiagnostics(**values)


class _Bridge:
    def __init__(self, output: PlannerOutput) -> None:
        self.output = output

    def plan(self, request: object) -> PlannerOutput:
        return self.output


class _SequenceBridge:
    def __init__(self, outputs: list[PlannerOutput]) -> None:
        self.outputs = iter(outputs)
        self.requests: list[object] = []

    def plan(self, request: object) -> PlannerOutput:
        self.requests.append(request)
        return next(self.outputs)


class _ReferenceExecutor:
    def __init__(self, result: ReferenceExecutionResult) -> None:
        self.result = result
        self.references: list[MotionReference] = []

    def __call__(self, reference: MotionReference) -> ReferenceExecutionResult:
        self.references.append(reference)
        return self.result


class _SequenceReferenceExecutor:
    def __init__(self, results: list[ReferenceExecutionResult]) -> None:
        self.results = iter(results)
        self.references: list[MotionReference] = []

    def __call__(self, reference: MotionReference) -> ReferenceExecutionResult:
        self.references.append(reference)
        return next(self.results)


class _GroundOptionHarness:
    def __init__(self, distances: list[float]) -> None:
        self.distances = iter(distances)
        self.begin_identities: list[ObservationIdentity] = []
        self.continue_identities: list[ObservationIdentity] = []
        self.clear_calls = 0

    @staticmethod
    def _prepared(identity: ObservationIdentity) -> PreparedPlanRequest:
        request = TrainingPlanRequest()
        request.state_time.nanoseconds_since_epoch = identity.state_time_ns
        return PreparedPlanRequest(
            request=request,
            identity=identity,
            candidate_id="a" * 64,
            physical_snapshot_id="b" * 64,
        )

    def begin(
        self, action: PolicyAction, identity: ObservationIdentity
    ) -> PreparedPlanRequest:
        del action
        self.begin_identities.append(identity)
        return self._prepared(identity)

    def continue_(
        self, identity: ObservationIdentity
    ) -> PreparedPlanRequest:
        self.continue_identities.append(identity)
        return self._prepared(identity)

    def distance(self) -> float:
        return next(self.distances)

    def clear(self) -> None:
        self.clear_calls += 1


class _ObservationProvider:
    def __init__(self, *observations: PolicyBatch) -> None:
        self.observations = list(observations)
        self.calls = 0

    def __call__(self) -> PolicyBatch:
        self.calls += 1
        if not self.observations:
            raise AssertionError("unexpected observation refresh")
        return self.observations.pop(0)


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
    assert transition.mission_observed_delta == 0.0
    assert transition.priority_observed_delta == 0.0
    assert transition.normalized_plan_or_execution_cost == 0.0
    assert transition.normalized_macro_step_time == 0.25
    assert transition.executed_without_new_coverage is False
    assert transition.planning_outcome == PlanningOutcome.NO_KNOWN_SAFE_ROUTE
    assert transition.execution_directive == ExecutionDirective.NO_SAFE_REFERENCE
    assert transition.reason_code == "NO_ROUTE"
    assert transition.terminated is False


def test_initial_planning_failure_uses_prepared_identity_for_zero_evidence_refresh(
) -> None:
    """Would fail if suppression recovered candidate identity from action index."""
    candidate_id = "c" * 64
    physical_snapshot_id = "d" * 64
    initial_identity = _identity()
    refreshed_identity = _identity(
        map_snapshot_id="map-refresh",
        candidate_set_id="candidates-refresh",
    )
    refreshed_observation = _observation(
        candidate_mask=(False, True), identity=refreshed_identity
    )
    output = PlannerOutput()
    output.outcome = PlanningOutcome.NO_KNOWN_SAFE_ROUTE
    output.directive = ExecutionDirective.NO_SAFE_REFERENCE
    output.candidate_disposition = (
        CandidateDisposition.SUPPRESS_FOR_CURRENT_PHYSICAL_SNAPSHOT
    )
    output.reason_code = "NO_ROUTE"
    refresh_calls: list[tuple[str, CandidateDisposition, str]] = []

    def build_request(
        action: PolicyAction, identity: ObservationIdentity
    ) -> PreparedPlanRequest:
        assert action.frontier_index == 1
        request = TrainingPlanRequest()
        request.state_time.nanoseconds_since_epoch = identity.state_time_ns
        return PreparedPlanRequest(
            request=request,
            identity=identity,
            candidate_id=candidate_id,
            physical_snapshot_id=physical_snapshot_id,
        )

    def refresh(
        failed_candidate_id: str,
        disposition: CandidateDisposition,
        failed_physical_snapshot_id: str,
    ) -> BoundaryObservationResult:
        refresh_calls.append(
            (
                failed_candidate_id,
                disposition,
                failed_physical_snapshot_id,
            )
        )
        return BoundaryObservationResult(
            next_observation=refreshed_observation,
            mission_observed_delta=0.0,
            priority_observed_delta=0.0,
            mission_observed_ratio=0.4,
            success_first_crossing=False,
            updated=True,
        )

    env = V3ExplorationEnvironment(
        platform_type="WHEELED",
        bridge=_Bridge(output),
        request_builder=build_request,
        initial_observation=_observation(
            candidate_mask=(True, True), identity=initial_identity
        ),
        require_identity_bound_request=True,
        planning_failure_refresher=refresh,
        include_planner_wall_time_in_reward=False,
    )

    transition = env.step(
        PolicyAction(frontier_index=1, theta_rad=0.0),
        expected_identity=initial_identity,
    )

    assert refresh_calls == [
        (
            candidate_id,
            CandidateDisposition.SUPPRESS_FOR_CURRENT_PHYSICAL_SNAPSHOT,
            physical_snapshot_id,
        )
    ]
    assert transition.next_observation.candidate_mask.tolist() == [
        [False, True]
    ]
    assert transition.next_observation.observation_identities == (
        refreshed_identity,
    )
    assert transition.mission_observed_delta == 0.0
    assert transition.priority_observed_delta == 0.0
    assert transition.normalized_macro_step_time == 0.0
    assert transition.terminated is False


@pytest.mark.parametrize(
    ("outcome", "directive"),
    [
        (PlanningOutcome.GOAL_INFEASIBLE, ExecutionDirective.HOLD_POSITION),
        (
            PlanningOutcome.NO_KNOWN_SAFE_ROUTE,
            ExecutionDirective.NO_SAFE_REFERENCE,
        ),
    ],
)
def test_keep_disposition_consumes_decision_without_suppressing_candidate(
    outcome: PlanningOutcome,
    directive: ExecutionDirective,
) -> None:
    """Would fail if outcome/directive parsing overrode disposition authority."""
    output = PlannerOutput()
    output.outcome = outcome
    output.directive = directive
    output.candidate_disposition = CandidateDisposition.KEEP
    output.reason_code = "CANDIDATE_REJECTED"
    env = V3ExplorationEnvironment(
        platform_type="WHEELED",
        bridge=_Bridge(output),
        request_builder=lambda action: action,
        initial_observation=_observation(candidate_mask=(True, True)),
    )
    policy_masks: list[torch.Tensor] = []

    def policy(observation: PolicyBatch) -> PolicyAction:
        policy_masks.append(observation.candidate_mask.clone())
        return PolicyAction(frontier_index=0, theta_rad=0.25)

    result = env.advance_until_decision_boundary(policy)

    assert result.policy_decisions_consumed == 1
    assert result.transition is not None
    assert policy_masks[0].tolist() == [[True, True]]
    assert result.transition.next_observation.candidate_mask.tolist() == [
        [True, True]
    ]


def test_ninth_policy_decision_is_not_a_task_terminal() -> None:
    rejected = PlannerOutput()
    rejected.outcome = PlanningOutcome.NO_KNOWN_SAFE_ROUTE
    rejected.directive = ExecutionDirective.NO_SAFE_REFERENCE
    rejected.reason_code = "CANDIDATE_REJECTED"
    env = V3ExplorationEnvironment(
        platform_type="WHEELED",
        bridge=_Bridge(rejected),
        request_builder=lambda action: action,
        initial_observation=_observation(candidate_mask=(True,) * 10),
    )

    for frontier_index in range(9):
        result = env.advance_until_decision_boundary(
            lambda _observation, index=frontier_index: PolicyAction(
                frontier_index=index,
                theta_rad=0.0,
            )
        )
        assert result.policy_decisions_consumed == 1
        assert result.transition is not None
        assert result.transition.terminated is False

    assert env.refresh_decision_boundary().execution_state == "DECISION_READY"


@pytest.mark.parametrize(
    ("identity_change", "changed_value"),
    (
        ("episode_id", "episode-2"),
        ("mission_revision", 2),
        ("map_snapshot_id", "map-2"),
        ("robot_state_id", "robot-state-2"),
        ("state_time_ns", 2_000),
        ("execution_state", "GROUND_HOLD"),
        ("candidate_set_id", "candidates-2"),
    ),
)
def test_keep_disposition_never_creates_identity_scoped_suppression(
    identity_change: str, changed_value: object
) -> None:
    """Would fail if environment retained an index mask outside the producer."""
    rejected = PlannerOutput()
    rejected.outcome = PlanningOutcome.NO_KNOWN_SAFE_ROUTE
    rejected.directive = ExecutionDirective.NO_SAFE_REFERENCE
    rejected.reason_code = "CANDIDATE_REJECTED"
    accepted = _reference_output(
        "WHEELED", ExecutionDirective.ACTIVATE_NEW_REFERENCE
    )
    refreshed = _observation(
        candidate_mask=(True, True),
        identity=_identity(**{identity_change: changed_value}),
    )
    env = V3ExplorationEnvironment(
        platform_type="WHEELED",
        bridge=_SequenceBridge([rejected, accepted, rejected]),
        request_builder=lambda action: action,
        initial_observation=_observation(candidate_mask=(True, True)),
        reference_executor=_ReferenceExecutor(
            ReferenceExecutionResult(
                next_observation=refreshed,
                mission_observed_delta=0.1,
                priority_observed_delta=0.1,
                normalized_execution_cost_contribution=0.0,
                normalized_execution_time_contribution=0.0,
                executed_without_new_coverage=False,
                success_first_crossing=False,
                episode_ended_without_success=False,
                hard_safety_violation=False,
                terminated=False,
                execution_state="DECISION_BOUNDARY",
            )
        ),
    )
    seen_masks: list[list[list[bool]]] = []

    def policy(observation: PolicyBatch) -> PolicyAction:
        seen_masks.append(observation.candidate_mask.tolist())
        selected = 0 if observation.candidate_mask[0, 0] else 1
        return PolicyAction(frontier_index=selected, theta_rad=0.0)

    env.advance_until_decision_boundary(policy)
    env.advance_until_decision_boundary(policy)
    env.advance_until_decision_boundary(policy)

    assert seen_masks == [
        [[True, True]],
        [[True, True]],
        [[True, True]],
    ]


def test_same_observation_identity_does_not_create_index_suppression() -> None:
    """Would fail if the environment inferred failure identity from an index."""
    rejected = PlannerOutput()
    rejected.outcome = PlanningOutcome.NO_KNOWN_SAFE_ROUTE
    rejected.directive = ExecutionDirective.NO_SAFE_REFERENCE
    rejected.reason_code = "CANDIDATE_REJECTED"
    accepted = _reference_output("WHEELED", ExecutionDirective.ACTIVATE_NEW_REFERENCE)
    env = V3ExplorationEnvironment(
        platform_type="WHEELED",
        bridge=_SequenceBridge([rejected, accepted]),
        request_builder=lambda action: action,
        initial_observation=_observation(candidate_mask=(True, True)),
        reference_executor=_ReferenceExecutor(
            ReferenceExecutionResult(
                next_observation=_observation(candidate_mask=(True, True)),
                mission_observed_delta=0.0,
                priority_observed_delta=0.0,
                normalized_execution_cost_contribution=0.0,
                normalized_execution_time_contribution=0.0,
                executed_without_new_coverage=False,
                success_first_crossing=False,
                episode_ended_without_success=False,
                hard_safety_violation=False,
                terminated=False,
                execution_state="DECISION_BOUNDARY",
            )
        ),
    )
    seen: list[list[list[bool]]] = []

    def policy(observation: PolicyBatch) -> PolicyAction:
        seen.append(observation.candidate_mask.tolist())
        return PolicyAction(
            frontier_index=0 if observation.candidate_mask[0, 0] else 1,
            theta_rad=0.0,
        )

    env.advance_until_decision_boundary(policy)
    env.advance_until_decision_boundary(policy)

    assert seen == [[[True, True]], [[True, True]]]


def test_rejection_refreshes_changed_producer_identity_before_retry_policy() -> None:
    """Would fail if rejection retry never observed new mission/map/state input."""
    rejected = PlannerOutput()
    rejected.outcome = PlanningOutcome.NO_KNOWN_SAFE_ROUTE
    rejected.directive = ExecutionDirective.NO_SAFE_REFERENCE
    rejected.reason_code = "CANDIDATE_REJECTED"
    first = _observation(candidate_mask=(True, True))
    refreshed = _observation(
        candidate_mask=(True, True),
        identity=_identity(
            mission_revision=2,
            map_snapshot_id="map-2",
            robot_state_id="robot-state-2",
            state_time_ns=2_000,
        ),
    )
    provider = _ObservationProvider(first, refreshed)
    env = V3ExplorationEnvironment(
        platform_type="WHEELED",
        bridge=_SequenceBridge([rejected, rejected]),
        request_builder=lambda action: action,
        initial_observation=first,
        observation_provider=provider,
    )
    seen: list[list[list[bool]]] = []

    def policy(observation: PolicyBatch) -> PolicyAction:
        seen.append(observation.candidate_mask.tolist())
        return PolicyAction(frontier_index=0, theta_rad=0.0)

    env.advance_until_decision_boundary(policy)
    env.advance_until_decision_boundary(policy)

    assert provider.calls == 2
    assert seen == [[[True, True]], [[True, True]]]


def test_provider_refresh_with_same_identity_has_no_environment_index_mask() -> None:
    """Would fail if the consumer reapplied a retired action-index mask."""
    rejected = PlannerOutput()
    rejected.outcome = PlanningOutcome.NO_KNOWN_SAFE_ROUTE
    rejected.directive = ExecutionDirective.NO_SAFE_REFERENCE
    rejected.reason_code = "CANDIDATE_REJECTED"
    first = _observation(candidate_mask=(True, True))
    provider = _ObservationProvider(
        first,
        _observation(candidate_mask=(True, True)),
    )
    env = V3ExplorationEnvironment(
        platform_type="WHEELED",
        bridge=_SequenceBridge([rejected, rejected]),
        request_builder=lambda action: action,
        initial_observation=first,
        observation_provider=provider,
    )
    seen: list[list[list[bool]]] = []

    def policy(observation: PolicyBatch) -> PolicyAction:
        seen.append(observation.candidate_mask.tolist())
        return PolicyAction(
            frontier_index=0 if observation.candidate_mask[0, 0] else 1,
            theta_rad=0.0,
        )

    env.advance_until_decision_boundary(policy)
    env.advance_until_decision_boundary(policy)

    assert seen == [[[True, True]], [[True, True]]]


def test_prepared_action_does_not_refresh_producer_after_policy_boundary() -> None:
    """Would fail if identity could drift between policy forward and planner action."""
    output = PlannerOutput()
    output.outcome = PlanningOutcome.INVALID_REQUEST
    output.directive = ExecutionDirective.NO_SAFE_REFERENCE
    output.reason_code = "INVALID"
    prepared = _observation(candidate_mask=(True, True))
    drifted = _observation(
        candidate_mask=(True, True),
        identity=_identity(map_snapshot_id="map-after-policy"),
    )
    provider = _ObservationProvider(prepared, drifted)
    env = V3ExplorationEnvironment(
        platform_type="WHEELED",
        bridge=_Bridge(output),
        request_builder=lambda action: action,
        initial_observation=prepared,
        observation_provider=provider,
    )

    boundary = env.refresh_decision_boundary()
    result = env.advance_prepared_action(
        PolicyAction(frontier_index=0, theta_rad=0.0),
        expected_identity=env.current_observation.observation_identities[0],
    )

    assert boundary.execution_state == "DECISION_READY"
    assert result.policy_decisions_consumed == 1
    assert provider.calls == 1
    assert result.transition is not None
    assert (
        result.transition.next_observation.observation_identities[0].map_snapshot_id
        == "map-1"
    )


def test_identity_bound_request_rejects_snapshot_drift_before_bridge_plan() -> None:
    """Would fail if builder could swap request snapshot after policy forward."""
    prepared = _observation(candidate_mask=(True, True))
    prepared_identity = prepared.observation_identities[0]
    drifted_identity = _identity(
        mission_revision=2,
        map_snapshot_id="map-after-policy",
        robot_state_id="robot-after-policy",
        state_time_ns=2_000,
    )
    request = TrainingPlanRequest()
    request.state_time.nanoseconds_since_epoch = drifted_identity.state_time_ns

    def drifting_builder(
        action: PolicyAction, expected_identity: ObservationIdentity
    ) -> PreparedPlanRequest:
        assert expected_identity == prepared_identity
        return PreparedPlanRequest(
            request=request,
            identity=drifted_identity,
            candidate_id="a" * 64,
            physical_snapshot_id="b" * 64,
        )

    env = V3ExplorationEnvironment(
        platform_type="WHEELED",
        bridge=_Bridge(PlannerOutput()),
        request_builder=drifting_builder,
        initial_observation=prepared,
        require_identity_bound_request=True,
    )
    env.refresh_decision_boundary()

    with pytest.raises(EnvironmentInvariantError, match="request observation identity"):
        env.advance_prepared_action(
            PolicyAction(frontier_index=0, theta_rad=0.0),
            expected_identity=prepared_identity,
        )


def test_installed_observation_is_cloned_from_producer_owned_tensors() -> None:
    """Would fail if environment state aliased producer-owned tensors."""
    output = PlannerOutput()
    output.outcome = PlanningOutcome.INVALID_REQUEST
    output.directive = ExecutionDirective.NO_SAFE_REFERENCE
    output.reason_code = "INVALID"
    initial = _observation(candidate_mask=(True, True))
    producer_refresh = _observation(
        candidate_mask=(True, True),
        identity=_identity(map_snapshot_id="map-2"),
    )
    producer_refresh.pose_features[0, 4] = 0.625
    provider = _ObservationProvider(producer_refresh)
    env = V3ExplorationEnvironment(
        platform_type="WHEELED",
        bridge=_Bridge(output),
        request_builder=lambda action: action,
        initial_observation=initial,
        observation_provider=provider,
    )

    env.advance_until_decision_boundary(
        lambda observation: PolicyAction(frontier_index=0, theta_rad=0.0)
    )
    published = env.current_observation
    published.pose_features[0, 4] = 0.0

    assert float(producer_refresh.pose_features[0, 4]) == pytest.approx(0.625)
    assert producer_refresh.candidate_mask.tolist() == [[True, True]]
    assert float(env.current_observation.pose_features[0, 4]) == pytest.approx(0.625)


def test_policy_decision_count_is_metadata_not_a_remaining_resource() -> None:
    output = PlannerOutput()
    output.outcome = PlanningOutcome.NO_KNOWN_SAFE_ROUTE
    output.directive = ExecutionDirective.NO_SAFE_REFERENCE
    output.reason_code = "NO_ROUTE"
    env = V3ExplorationEnvironment(
        platform_type="WHEELED",
        bridge=_Bridge(output),
        request_builder=lambda action: action,
        initial_observation=_observation(candidate_mask=(True, True)),
    )

    result = env.advance_until_decision_boundary(
        lambda _observation: PolicyAction(frontier_index=0, theta_rad=0.0)
    )

    assert result.policy_decisions_consumed == 1
    assert result.transition is not None
    assert result.transition.next_observation.pose_features.shape == (1, 5)
    assert result.transition.terminated is False


def test_all_false_candidates_bypass_policy_without_fallback() -> None:
    """Would fail if the environment sampled policy or invented robot position."""
    output = PlannerOutput()
    output.outcome = PlanningOutcome.NO_KNOWN_SAFE_ROUTE
    output.directive = ExecutionDirective.NO_SAFE_REFERENCE
    env = V3ExplorationEnvironment(
        platform_type="WHEELED",
        bridge=_Bridge(output),
        request_builder=lambda action: action,
        initial_observation=_observation(candidate_mask=(False, False)),
    )
    policy_called = False

    def policy(observation: PolicyBatch) -> PolicyAction:
        nonlocal policy_called
        policy_called = True
        return PolicyAction(frontier_index=0, theta_rad=0.0)

    result = env.advance_until_decision_boundary(policy)

    assert policy_called is False
    assert result.execution_state == "NO_CANDIDATES"
    assert result.transition is None
    assert result.policy_decisions_consumed == 0


@pytest.mark.parametrize(
    ("diagnostics", "oracle_count", "candidate_mask", "expected_reason"),
    (
        (
            _candidate_diagnostics(
                physical_candidate_universe_count=2,
                selected_policy_candidate_count=1,
                available_candidate_count=1,
                planner_failed_current_snapshot_count=1,
            ),
            2,
            (True, False),
            None,
        ),
        (
            _candidate_diagnostics(
                physical_candidate_universe_count=3,
                selected_policy_candidate_count=1,
                available_candidate_count=3,
                untried_reserve_count=2,
            ),
            3,
            (True, False),
            None,
        ),
        (
            _candidate_diagnostics(physical_unreachable_count=4),
            0,
            (False, False),
            "NO_RECOVERABLE_OBSERVATION_STATE",
        ),
        (
            _candidate_diagnostics(
                physical_candidate_universe_count=2,
                planner_failed_current_snapshot_count=2,
                zero_gain_count=2,
            ),
            0,
            (False, False),
            "ZERO_GAIN",
        ),
        (
            _candidate_diagnostics(
                physical_candidate_universe_count=5,
                visited_excluded_count=5,
            ),
            0,
            (False, False),
            "VISITED_EXHAUSTED",
        ),
        (
            _candidate_diagnostics(),
            0,
            (False, False),
            "NO_TRANSIT_OPPORTUNITY",
        ),
        (
            _candidate_diagnostics(
                physical_candidate_universe_count=146,
                planner_failed_current_snapshot_count=146,
            ),
            146,
            (False, False),
            "PLANNER_BLOCKED_WITH_OPPORTUNITY",
        ),
        (
            _candidate_diagnostics(
                physical_candidate_universe_count=146,
                planner_failed_current_snapshot_count=100,
                visited_excluded_count=46,
            ),
            146,
            (False, False),
            "PLANNER_BLOCKED_WITH_OPPORTUNITY",
        ),
    ),
)
def test_candidate_boundary_matrix_continues_or_terminates_from_observed_facts(
    diagnostics: CandidateDiagnostics,
    oracle_count: int,
    candidate_mask: tuple[bool, ...],
    expected_reason: str | None,
) -> None:
    env = V3ExplorationEnvironment(
        platform_type="WHEELED",
        bridge=_Bridge(PlannerOutput()),
        request_builder=lambda action: action,
        initial_observation=_observation(candidate_mask=candidate_mask),
        candidate_diagnostics_provider=lambda: diagnostics,
        frontier_oracle=lambda: FrontierOracleResult(
            oracle_count, oracle_count, oracle_count, oracle_count
        ),
        remaining_coverable_detail_cell_count_provider=lambda: 1234,
    )

    result = env.refresh_decision_boundary()

    assert result.execution_state == (
        "DECISION_READY" if expected_reason is None else "NO_CANDIDATES"
    )
    assert (
        None if result.terminal_reason is None else result.terminal_reason.value
    ) == expected_reason
    assert result.oracle_opportunity_count == (
        0 if expected_reason is None else oracle_count
    )
    assert result.remaining_coverable_detail_cell_count == (
        None if expected_reason is None else 1234
    )


@pytest.mark.parametrize(
    ("diagnostics", "oracle_count", "candidate_mask"),
    (
        (_candidate_diagnostics(), 1, (False, False)),
        (
            _candidate_diagnostics(
                physical_candidate_universe_count=1,
                selected_policy_candidate_count=1,
                available_candidate_count=1,
            ),
            0,
            (True, False),
        ),
        (
            _candidate_diagnostics(
                physical_candidate_universe_count=2,
                planner_failed_current_snapshot_count=0,
                visited_excluded_count=2,
            ),
            2,
            (False, False),
        ),
    ),
)
def test_candidate_oracle_or_availability_mismatch_fails_closed(
    diagnostics: CandidateDiagnostics,
    oracle_count: int,
    candidate_mask: tuple[bool, ...],
) -> None:
    env = V3ExplorationEnvironment(
        platform_type="WHEELED",
        bridge=_Bridge(PlannerOutput()),
        request_builder=lambda action: action,
        initial_observation=_observation(candidate_mask=candidate_mask),
        candidate_diagnostics_provider=lambda: diagnostics,
        frontier_oracle=lambda: FrontierOracleResult(
            oracle_count, oracle_count, oracle_count, oracle_count
        ),
    )

    with pytest.raises(
        EnvironmentInvariantError,
        match="CANDIDATE_ORACLE_MISMATCH|availability",
    ):
        env.refresh_decision_boundary()

    assert env.rollout_discarded is True
    assert env.training_stopped is True


def test_oracle_146_refills_last_reserve_then_reports_planner_blocked() -> None:
    rejected = PlannerOutput()
    rejected.outcome = PlanningOutcome.NO_KNOWN_SAFE_ROUTE
    rejected.directive = ExecutionDirective.NO_SAFE_REFERENCE
    rejected.candidate_disposition = (
        CandidateDisposition.SUPPRESS_FOR_CURRENT_PHYSICAL_SNAPSHOT
    )
    rejected.reason_code = "NO_ROUTE"
    oracle_calls = 0
    diagnostics = _candidate_diagnostics(
        physical_candidate_universe_count=26,
        selected_policy_candidate_count=1,
        available_candidate_count=1,
        planner_failed_current_snapshot_count=25,
    )
    initial = _observation(candidate_mask=(True, False))
    refreshed = _observation(
        candidate_mask=(False, False),
        identity=_identity(
            map_snapshot_id="map-planner-failed",
            candidate_set_id="candidates-empty",
        ),
    )

    def oracle() -> FrontierOracleResult:
        nonlocal oracle_calls
        oracle_calls += 1
        return FrontierOracleResult(146, 146, 146, 146)

    def request_builder(
        _action: PolicyAction, identity: ObservationIdentity
    ) -> PreparedPlanRequest:
        request = TrainingPlanRequest()
        request.state_time.nanoseconds_since_epoch = identity.state_time_ns
        return PreparedPlanRequest(
            request=request,
            identity=identity,
            candidate_id="a" * 64,
            physical_snapshot_id="f" * 64,
        )

    def refresh(
        _candidate_id: str,
        _disposition: CandidateDisposition,
        _physical_snapshot_id: str,
    ) -> BoundaryObservationResult:
        nonlocal diagnostics
        diagnostics = _candidate_diagnostics(
            physical_candidate_universe_count=26,
            selected_policy_candidate_count=0,
            available_candidate_count=0,
            planner_failed_current_snapshot_count=26,
        )
        return BoundaryObservationResult(
            next_observation=refreshed,
            mission_observed_delta=0.0,
            priority_observed_delta=0.0,
            mission_observed_ratio=0.5,
            success_first_crossing=False,
            updated=True,
        )

    env = V3ExplorationEnvironment(
        platform_type="WHEELED",
        bridge=_Bridge(rejected),
        request_builder=request_builder,
        initial_observation=initial,
        require_identity_bound_request=True,
        planning_failure_refresher=refresh,
        candidate_diagnostics_provider=lambda: diagnostics,
        frontier_oracle=oracle,
        remaining_coverable_detail_cell_count_provider=lambda: 88,
    )

    ready = env.refresh_decision_boundary()
    assert ready.execution_state == "DECISION_READY"
    assert ready.terminal_reason is None

    result = env.advance_prepared_action(
        PolicyAction(frontier_index=0, theta_rad=0.0),
        expected_identity=env.current_observation.observation_identities[0],
    )

    assert oracle_calls == 2
    assert result.transition is not None
    assert result.transition.terminated is True
    assert result.transition.terminal_reason is not None
    assert (
        result.transition.terminal_reason.value
        == "PLANNER_BLOCKED_WITH_OPPORTUNITY"
    )
    assert result.transition.oracle_opportunity_count == 146
    assert result.transition.remaining_coverable_detail_cell_count == 88
    assert result.transition.success_first_crossing is False
    assert result.transition.terminal_reason not in {
        TerminalReason.VISITED_EXHAUSTED,
        TerminalReason.NO_TRANSIT_OPPORTUNITY,
    }


@pytest.mark.parametrize(
    ("outcome", "directive", "expected_reason"),
    (
        (
            PlanningOutcome.INVALID_REQUEST,
            ExecutionDirective.NO_SAFE_REFERENCE,
            TerminalReason.HARD_FAILURE,
        ),
        (
            PlanningOutcome.NUMERICAL_FAILURE,
            ExecutionDirective.NO_SAFE_REFERENCE,
            TerminalReason.HARD_FAILURE,
        ),
        (
            PlanningOutcome.RESOURCE_EXHAUSTED,
            ExecutionDirective.NO_SAFE_REFERENCE,
            TerminalReason.HARD_FAILURE,
        ),
        (
            PlanningOutcome.CANCELED,
            ExecutionDirective.HOLD_POSITION,
            TerminalReason.CANCELED,
        ),
    ),
)
def test_planner_hard_outcomes_never_become_legal_exploration_exhaustion(
    outcome: PlanningOutcome,
    directive: ExecutionDirective,
    expected_reason: TerminalReason,
) -> None:
    output = PlannerOutput()
    output.outcome = outcome
    output.directive = directive
    output.reason_code = "HARD_OUTCOME"
    env = V3ExplorationEnvironment(
        platform_type="WHEELED",
        bridge=_Bridge(output),
        request_builder=lambda action: action,
        initial_observation=_observation(candidate_mask=(True, False)),
        remaining_coverable_detail_cell_count_provider=lambda: 456,
    )

    result = env.advance_prepared_action(
        PolicyAction(frontier_index=0, theta_rad=0.0),
        expected_identity=env.current_observation.observation_identities[0],
    )

    assert result.transition is not None
    assert result.transition.terminated is True
    assert result.transition.terminal_reason is expected_reason
    assert result.transition.remaining_coverable_detail_cell_count == 456


def test_oracle_146_hard_keep_failure_does_not_refresh_or_report_blocked() -> None:
    output = PlannerOutput()
    output.outcome = PlanningOutcome.INVALID_REQUEST
    output.directive = ExecutionDirective.NO_SAFE_REFERENCE
    output.candidate_disposition = CandidateDisposition.KEEP
    output.reason_code = "INFRASTRUCTURE_FAILURE"
    refresh_calls = 0

    def unexpected_refresh(*_args: object) -> BoundaryObservationResult:
        nonlocal refresh_calls
        refresh_calls += 1
        raise AssertionError("hard KEEP must not update the failure set")

    diagnostics = _candidate_diagnostics(
        physical_candidate_universe_count=1,
        selected_policy_candidate_count=1,
        available_candidate_count=1,
    )
    env = V3ExplorationEnvironment(
        platform_type="WHEELED",
        bridge=_Bridge(output),
        request_builder=lambda action: action,
        initial_observation=_observation(candidate_mask=(True, False)),
        planning_failure_refresher=unexpected_refresh,
        candidate_diagnostics_provider=lambda: diagnostics,
    )

    result = env.advance_prepared_action(
        PolicyAction(0, 0.0),
        expected_identity=env.current_observation.observation_identities[0],
    )

    assert refresh_calls == 0
    assert env.current_candidate_diagnostics() == diagnostics
    assert result.transition is not None
    assert result.transition.terminal_reason is TerminalReason.HARD_FAILURE
    assert (
        result.transition.terminal_reason
        is not TerminalReason.PLANNER_BLOCKED_WITH_OPPORTUNITY
    )


def test_execution_success_crossing_precedes_candidate_oracle_audit() -> None:
    output = _reference_output(
        "WHEELED", ExecutionDirective.ACTIVATE_NEW_REFERENCE
    )
    oracle_calls = 0

    def contradictory_oracle() -> FrontierOracleResult:
        nonlocal oracle_calls
        oracle_calls += 1
        return FrontierOracleResult(1, 1, 1, 9)

    env = V3ExplorationEnvironment(
        platform_type="WHEELED",
        bridge=_Bridge(output),
        request_builder=lambda action: action,
        initial_observation=_observation(candidate_mask=(True, False)),
        reference_executor=_ReferenceExecutor(
            ReferenceExecutionResult(
                next_observation=_observation(candidate_mask=(False, False)),
                mission_observed_delta=0.01,
                priority_observed_delta=0.0,
                normalized_execution_cost_contribution=0.0,
                normalized_execution_time_contribution=0.0,
                executed_without_new_coverage=False,
                success_first_crossing=True,
                episode_ended_without_success=False,
                hard_safety_violation=False,
                terminated=True,
                execution_state="DECISION_BOUNDARY",
            )
        ),
        candidate_diagnostics_provider=_candidate_diagnostics,
        frontier_oracle=contradictory_oracle,
    )

    result = env.advance_prepared_action(
        PolicyAction(0, 0.0),
        expected_identity=env.current_observation.observation_identities[0],
    )

    assert oracle_calls == 0
    assert result.transition is not None
    assert result.transition.success_first_crossing is True
    assert result.transition.terminal_reason is TerminalReason.SUCCESS


def test_initial_sensor_crossing_succeeds_before_planner_and_only_once() -> None:
    initial = _observation(candidate_mask=(True, False))
    controller = object.__new__(ObservationBoundaryController)
    controller._platform_type = "WHEELED"
    controller._current_observation = initial
    controller._mission_observed_area_m2 = 0.96
    controller._mission_area_m2 = 1.0
    planner_calls = 0

    class CountingBridge:
        def plan(self, request: object) -> PlannerOutput:
            del request
            nonlocal planner_calls
            planner_calls += 1
            return PlannerOutput()

    env = V3ExplorationEnvironment(
        platform_type="WHEELED",
        bridge=CountingBridge(),
        request_builder=lambda action: action,
        initial_observation=initial,
        observation_boundary_controller=controller,
        planning_failure_refresher=lambda *_args: None,
        require_sensor_closed_loop=True,
    )
    policy_calls = 0

    def policy(_observation: PolicyBatch) -> PolicyAction:
        nonlocal policy_calls
        policy_calls += 1
        return PolicyAction(0, 0.0)

    result = env.advance_until_decision_boundary(policy)

    assert result.terminal_reason is TerminalReason.SUCCESS
    assert result.policy_decisions_consumed == 0
    assert planner_calls == 0
    assert policy_calls == 0
    with pytest.raises(RuntimeError, match="reset"):
        env.advance_until_decision_boundary(policy)


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
            mission_observed_delta=0.2,
            priority_observed_delta=0.1,
            normalized_execution_cost_contribution=0.3,
            normalized_execution_time_contribution=0.4,
            executed_without_new_coverage=True,
            success_first_crossing=False,
            episode_ended_without_success=False,
            hard_safety_violation=False,
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
    assert transition.next_observation is not next_observation
    assert transition.next_observation.pose_features[0, 0].item() == pytest.approx(0.5)
    assert transition.mission_observed_delta == 0.2
    assert transition.priority_observed_delta == 0.1
    assert transition.normalized_plan_or_execution_cost == pytest.approx(0.8)
    assert transition.normalized_macro_step_time == pytest.approx(0.65)
    assert transition.executed_without_new_coverage is True
    assert transition.planning_outcome == PlanningOutcome.NEW_REFERENCE_AVAILABLE
    assert transition.execution_directive == ExecutionDirective.ACTIVATE_NEW_REFERENCE
    assert transition.reason_code == "REFERENCE_READY"
    assert transition.terminated is False


def _ground_result(
    identity: ObservationIdentity,
    *,
    mission_delta: float,
    priority_delta: float,
    execution_cost: float,
    execution_time: float,
    sample_count: int,
    success_first_crossing: bool = False,
) -> ReferenceExecutionResult:
    return ReferenceExecutionResult(
        next_observation=_observation(identity=identity),
        mission_observed_delta=mission_delta,
        priority_observed_delta=priority_delta,
        normalized_execution_cost_contribution=execution_cost,
        normalized_execution_time_contribution=execution_time,
        executed_without_new_coverage=mission_delta == 0.0,
        success_first_crossing=success_first_crossing,
        episode_ended_without_success=False,
        hard_safety_violation=False,
        terminated=success_first_crossing,
        execution_state="DECISION_BOUNDARY",
        execution_events=ExecutionEvents(
            reference_samples_consumed=sample_count,
            selected_action_observed_safe=True,
        ),
    )


def test_one_ground_policy_action_aggregates_three_rolling_references() -> None:
    identities = [
        _identity(
            map_snapshot_id=f"map-{index}",
            robot_state_id=f"robot-{index}",
            state_time_ns=1_000 * index,
        )
        for index in range(1, 5)
    ]
    harness = _GroundOptionHarness([6.0, 2.0, 0.1])
    executor = _SequenceReferenceExecutor(
        [
            _ground_result(
                identities[index + 1],
                mission_delta=0.1 * (index + 1),
                priority_delta=0.01 * (index + 1),
                execution_cost=0.2 * (index + 1),
                execution_time=0.3 * (index + 1),
                sample_count=index + 2,
            )
            for index in range(3)
        ]
    )
    bridge = _SequenceBridge(
        [
            _reference_output(
                "WHEELED", ExecutionDirective.ACTIVATE_NEW_REFERENCE
            )
            for _ in range(3)
        ]
    )
    env = V3ExplorationEnvironment(
        platform_type="WHEELED",
        bridge=bridge,
        request_builder=harness.begin,
        initial_observation=_observation(identity=identities[0]),
        require_identity_bound_request=True,
        reference_executor=executor,
        ground_option_continuation_builder=harness.continue_,
        ground_option_distance_provider=harness.distance,
        ground_option_clearer=harness.clear,
        plan_cost_scale=4.0,
        planner_elapsed_scale_s=2.0,
    )
    policy_calls = 0

    def policy(_observation: PolicyBatch) -> PolicyAction:
        nonlocal policy_calls
        policy_calls += 1
        return PolicyAction(0, 0.25)

    result = env.advance_until_decision_boundary(policy)

    assert policy_calls == 1
    assert result.policy_decisions_consumed == 1
    assert len(executor.references) == 3
    assert harness.begin_identities == [identities[0]]
    assert harness.continue_identities == identities[1:3]
    assert harness.clear_calls == 1
    transition = result.transition
    assert transition is not None
    assert transition.next_observation.observation_identities == (identities[3],)
    assert transition.mission_observed_delta == pytest.approx(0.6)
    assert transition.priority_observed_delta == pytest.approx(0.06)
    assert transition.normalized_plan_or_execution_cost == pytest.approx(2.7)
    assert transition.normalized_macro_step_time == pytest.approx(2.55)
    assert transition.execution_events.reference_samples_consumed == 9
    assert transition.execution_events.selected_action_observed_safe is True


def test_ground_rolling_requests_carry_the_previous_opaque_continuation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identities = [
        _identity(
            map_snapshot_id=f"continuation-map-{index}",
            robot_state_id=f"continuation-robot-{index}",
            state_time_ns=1_000 * index,
        )
        for index in range(1, 4)
    ]
    harness = _GroundOptionHarness([4.0, 0.1])
    monkeypatch.setattr(v3_module, "TrainingPlanRequest", SimpleNamespace)
    harness._prepared = lambda identity: PreparedPlanRequest(
        request=SimpleNamespace(
            state_time=SimpleNamespace(
                nanoseconds_since_epoch=identity.state_time_ns
            ),
            continuation=None,
        ),
        identity=identity,
        candidate_id="a" * 64,
        physical_snapshot_id="b" * 64,
    )
    outputs = [
        _reference_output("WHEELED", ExecutionDirective.ACTIVATE_NEW_REFERENCE)
        for _ in range(2)
    ]
    handles = [object(), object()]
    outputs = [
        SimpleNamespace(
            outcome=output.outcome,
            directive=output.directive,
            candidate_disposition=output.candidate_disposition,
            reason_code=output.reason_code,
            reference=output.reference,
            diagnostics=output.diagnostics,
            continuation=handle,
        )
        for output, handle in zip(outputs, handles, strict=True)
    ]
    bridge = _SequenceBridge(outputs)
    env = V3ExplorationEnvironment(
        platform_type="WHEELED",
        bridge=bridge,
        request_builder=harness.begin,
        initial_observation=_observation(identity=identities[0]),
        require_identity_bound_request=True,
        reference_executor=_SequenceReferenceExecutor(
            [
                _ground_result(
                    identities[1],
                    mission_delta=0.1,
                    priority_delta=0.0,
                    execution_cost=0.1,
                    execution_time=0.1,
                    sample_count=2,
                ),
                _ground_result(
                    identities[2],
                    mission_delta=0.1,
                    priority_delta=0.0,
                    execution_cost=0.1,
                    execution_time=0.1,
                    sample_count=2,
                ),
            ]
        ),
        ground_option_continuation_builder=harness.continue_,
        ground_option_distance_provider=harness.distance,
        ground_option_clearer=harness.clear,
    )

    env.advance_prepared_action(
        PolicyAction(0, 0.0), expected_identity=identities[0]
    )

    assert bridge.requests[0].continuation is None
    assert bridge.requests[1].continuation is handles[0]


def test_ground_option_returns_aggregated_progress_when_refresh_rejects_goal() -> None:
    first = _identity()
    progressed = _identity(
        map_snapshot_id="map-progress",
        robot_state_id="robot-progress",
        state_time_ns=2_000,
    )
    harness = _GroundOptionHarness([5.0])
    rejected = PlannerOutput()
    rejected.outcome = PlanningOutcome.NO_KNOWN_SAFE_ROUTE
    rejected.directive = ExecutionDirective.NO_SAFE_REFERENCE
    rejected.reason_code = "REFRESHED_GOAL_REJECTED"
    rejected.diagnostics.elapsed = timedelta(milliseconds=250)
    env = V3ExplorationEnvironment(
        platform_type="WHEELED",
        bridge=_SequenceBridge(
            [
                _reference_output(
                    "WHEELED", ExecutionDirective.ACTIVATE_NEW_REFERENCE
                ),
                rejected,
            ]
        ),
        request_builder=harness.begin,
        initial_observation=_observation(identity=first),
        require_identity_bound_request=True,
        reference_executor=_SequenceReferenceExecutor(
            [
                _ground_result(
                    progressed,
                    mission_delta=0.2,
                    priority_delta=0.1,
                    execution_cost=0.3,
                    execution_time=0.4,
                    sample_count=2,
                )
            ]
        ),
        ground_option_continuation_builder=harness.continue_,
        ground_option_distance_provider=harness.distance,
        ground_option_clearer=harness.clear,
        planner_elapsed_scale_s=1.0,
    )

    result = env.advance_prepared_action(
        PolicyAction(0, 0.0), expected_identity=first
    )

    assert result.policy_decisions_consumed == 1
    assert result.transition is not None
    assert result.transition.mission_observed_delta == pytest.approx(0.2)
    assert result.transition.planning_outcome == PlanningOutcome.NO_KNOWN_SAFE_ROUTE
    assert result.transition.reason_code == "REFRESHED_GOAL_REJECTED"
    assert result.transition.terminated is False
    assert harness.clear_calls == 1


def test_rolling_planning_failure_refreshes_after_aggregating_each_reference_once(
) -> None:
    identities = [
        _identity(
            map_snapshot_id=f"rolling-map-{index}",
            robot_state_id=f"rolling-robot-{index}",
            state_time_ns=1_000 * index,
            candidate_set_id=f"rolling-candidates-{index}",
        )
        for index in range(1, 5)
    ]
    refreshed_identity = _identity(
        map_snapshot_id="rolling-map-refresh",
        robot_state_id=identities[-1].robot_state_id,
        state_time_ns=identities[-1].state_time_ns,
        candidate_set_id="rolling-candidates-refresh",
    )
    harness = _GroundOptionHarness([7.0, 4.0, 1.0])
    executor = _SequenceReferenceExecutor(
        [
            _ground_result(
                identities[index + 1],
                mission_delta=0.1 * (index + 1),
                priority_delta=0.01 * (index + 1),
                execution_cost=0.2 * (index + 1),
                execution_time=0.3 * (index + 1),
                sample_count=index + 2,
            )
            for index in range(3)
        ]
    )
    rejected = PlannerOutput()
    rejected.outcome = PlanningOutcome.NO_KNOWN_SAFE_ROUTE
    rejected.directive = ExecutionDirective.NO_SAFE_REFERENCE
    rejected.candidate_disposition = (
        CandidateDisposition.SUPPRESS_FOR_CURRENT_PHYSICAL_SNAPSHOT
    )
    rejected.reason_code = "ROLLING_GOAL_REJECTED"
    refresh_calls: list[tuple[str, CandidateDisposition, str, int]] = []

    def refresh(
        candidate_id: str,
        disposition: CandidateDisposition,
        physical_snapshot_id: str,
    ) -> BoundaryObservationResult:
        refresh_calls.append(
            (
                candidate_id,
                disposition,
                physical_snapshot_id,
                len(executor.references),
            )
        )
        return BoundaryObservationResult(
            next_observation=_observation(
                candidate_mask=(False, True), identity=refreshed_identity
            ),
            mission_observed_delta=0.0,
            priority_observed_delta=0.0,
            mission_observed_ratio=0.7,
            success_first_crossing=False,
            updated=True,
        )

    env = V3ExplorationEnvironment(
        platform_type="WHEELED",
        bridge=_SequenceBridge(
            [
                _reference_output(
                    "WHEELED", ExecutionDirective.ACTIVATE_NEW_REFERENCE
                )
                for _ in range(3)
            ]
            + [rejected]
        ),
        request_builder=harness.begin,
        initial_observation=_observation(identity=identities[0]),
        require_identity_bound_request=True,
        reference_executor=executor,
        ground_option_continuation_builder=harness.continue_,
        ground_option_distance_provider=harness.distance,
        ground_option_clearer=harness.clear,
        planning_failure_refresher=refresh,
        include_planner_wall_time_in_reward=False,
        plan_cost_scale=4.0,
    )

    result = env.advance_prepared_action(
        PolicyAction(1, 0.0), expected_identity=identities[0]
    )

    assert refresh_calls == [
        (
            "a" * 64,
            CandidateDisposition.SUPPRESS_FOR_CURRENT_PHYSICAL_SNAPSHOT,
            "b" * 64,
            3,
        )
    ]
    assert harness.clear_calls == 1
    assert harness.begin_identities == [identities[0]]
    assert harness.continue_identities == identities[1:]
    transition = result.transition
    assert transition is not None
    assert transition.next_observation.observation_identities == (
        refreshed_identity,
    )
    assert transition.mission_observed_delta == pytest.approx(0.6)
    assert transition.priority_observed_delta == pytest.approx(0.06)
    assert transition.normalized_plan_or_execution_cost == pytest.approx(2.7)
    assert transition.normalized_macro_step_time == pytest.approx(1.8)
    assert transition.execution_events.reference_samples_consumed == 9
    assert transition.terminated is False


def test_rolling_success_crossing_skips_fourth_plan_and_failure_refresh() -> None:
    identities = [
        _identity(
            map_snapshot_id=f"success-map-{index}",
            robot_state_id=f"success-robot-{index}",
            state_time_ns=1_000 * index,
        )
        for index in range(1, 5)
    ]
    harness = _GroundOptionHarness([7.0, 4.0])
    executor = _SequenceReferenceExecutor(
        [
            _ground_result(
                identities[index + 1],
                mission_delta=0.1,
                priority_delta=0.01,
                execution_cost=0.0,
                execution_time=0.1,
                sample_count=2,
                success_first_crossing=index == 2,
            )
            for index in range(3)
        ]
    )
    bridge = _SequenceBridge(
        [
            _reference_output(
                "WHEELED", ExecutionDirective.ACTIVATE_NEW_REFERENCE
            )
            for _ in range(3)
        ]
    )
    refresh_calls = 0

    def unexpected_refresh(*_args) -> BoundaryObservationResult:
        nonlocal refresh_calls
        refresh_calls += 1
        raise AssertionError("success must not refresh planning failure")

    env = V3ExplorationEnvironment(
        platform_type="WHEELED",
        bridge=bridge,
        request_builder=harness.begin,
        initial_observation=_observation(identity=identities[0]),
        require_identity_bound_request=True,
        reference_executor=executor,
        ground_option_continuation_builder=harness.continue_,
        ground_option_distance_provider=harness.distance,
        ground_option_clearer=harness.clear,
        planning_failure_refresher=unexpected_refresh,
    )

    result = env.advance_prepared_action(
        PolicyAction(0, 0.0), expected_identity=identities[0]
    )

    assert len(executor.references) == 3
    assert refresh_calls == 0
    assert result.transition is not None
    assert result.transition.success_first_crossing is True
    assert result.transition.terminated is True


def test_ground_option_allows_a_detour_before_reaching_the_target() -> None:
    identities = [
        _identity(
            map_snapshot_id=f"detour-map-{index}",
            robot_state_id=f"detour-robot-{index}",
            state_time_ns=1_000 * index,
        )
        for index in range(1, 3)
    ]
    harness = _GroundOptionHarness([])
    executor = _SequenceReferenceExecutor(
        [
            _ground_result(
                identities[index],
                mission_delta=0.1,
                priority_delta=0.01,
                execution_cost=0.2,
                execution_time=0.3,
                sample_count=73,
            )
            for index in range(2)
        ]
    )
    bridge = _SequenceBridge(
        [
            _reference_output(
                "LEGGED", ExecutionDirective.ACTIVATE_NEW_REFERENCE
            )
            for _ in range(2)
        ]
    )
    target_distances_m = (
        21.15892246783659,
        22.466419385387436,
        0.1,
    )

    def target_distance_m() -> float:
        return target_distances_m[len(executor.references)]

    env = V3ExplorationEnvironment(
        platform_type="LEGGED",
        bridge=bridge,
        request_builder=harness.begin,
        initial_observation=_observation(identity=identities[0]),
        require_identity_bound_request=True,
        reference_executor=executor,
        ground_option_continuation_builder=harness.continue_,
        ground_option_distance_provider=target_distance_m,
        ground_option_clearer=harness.clear,
    )

    result = env.advance_prepared_action(
        PolicyAction(0, 0.0), expected_identity=identities[0]
    )

    assert result.policy_decisions_consumed == 1
    assert len(executor.references) == 2
    assert harness.continue_identities == [identities[0]]
    assert harness.clear_calls == 1
    assert result.transition is not None
    assert result.transition.mission_observed_delta == pytest.approx(0.2)
    assert result.transition.execution_events.reference_samples_consumed == 146


def test_ground_option_can_reach_target_after_more_than_64_references() -> None:
    identities = [
        _identity(
            map_snapshot_id=f"map-{index}",
            robot_state_id=f"robot-{index}",
            state_time_ns=1_000 * index,
        )
        for index in range(67)
    ]
    harness = _GroundOptionHarness([10.0] * 65 + [0.1])
    executor = _SequenceReferenceExecutor(
        [
            _ground_result(
                identities[index + 1],
                mission_delta=0.01,
                priority_delta=0.001,
                execution_cost=0.02,
                execution_time=0.1,
                sample_count=2,
            )
            for index in range(66)
        ]
    )
    env = V3ExplorationEnvironment(
        platform_type="WHEELED",
        bridge=_SequenceBridge(
            [
                _reference_output(
                    "WHEELED", ExecutionDirective.ACTIVATE_NEW_REFERENCE
                )
                for _ in range(66)
            ]
        ),
        request_builder=harness.begin,
        initial_observation=_observation(identity=identities[0]),
        require_identity_bound_request=True,
        reference_executor=executor,
        ground_option_continuation_builder=harness.continue_,
        ground_option_distance_provider=harness.distance,
        ground_option_clearer=harness.clear,
    )

    result = env.advance_prepared_action(
        PolicyAction(0, 0.0), expected_identity=identities[0]
    )

    assert result.policy_decisions_consumed == 1
    assert len(executor.references) == 66
    assert result.transition is not None
    assert result.transition.mission_observed_delta == pytest.approx(0.66)
    assert result.transition.priority_observed_delta == pytest.approx(0.066)
    assert result.transition.execution_events.reference_samples_consumed == 132
    assert harness.clear_calls == 1


def test_ground_option_repeated_progress_signature_fails_closed() -> None:
    identity = _identity()
    harness = _GroundOptionHarness([10.0, 10.0])
    outputs = [
        _reference_output("WHEELED", ExecutionDirective.ACTIVATE_NEW_REFERENCE)
        for _ in range(2)
    ]
    for output in outputs:
        hierarchical = HierarchicalPlannerMetrics()
        hierarchical.route_cursor = 7
        output.diagnostics.hierarchical = hierarchical
        point = TrajectoryPoint()
        point.pose.position_m.x = 12.0
        point.pose.position_m.y = 34.0
        point.pose.position_m.z = 5.0
        output.reference.data.points = [point]
    executor = _SequenceReferenceExecutor(
        [
            _ground_result(
                identity,
                mission_delta=0.0,
                priority_delta=0.0,
                execution_cost=0.0,
                execution_time=0.1,
                sample_count=2,
            )
            for _ in range(2)
        ]
    )
    env = V3ExplorationEnvironment(
        platform_type="WHEELED",
        bridge=_SequenceBridge(outputs),
        request_builder=harness.begin,
        initial_observation=_observation(identity=identity),
        require_identity_bound_request=True,
        reference_executor=executor,
        ground_option_continuation_builder=harness.continue_,
        ground_option_distance_provider=harness.distance,
        ground_option_clearer=harness.clear,
    )

    with pytest.raises(EnvironmentInvariantError, match="GROUND_OPTION_STALLED"):
        env.advance_prepared_action(
            PolicyAction(0, 0.0), expected_identity=identity
        )

    assert len(executor.references) == 2
    assert harness.clear_calls == 1


def test_mismatched_reference_platform_discards_rollout_and_stops_training() -> None:
    """Would fail if a hopper reference could execute in a wheeled episode."""
    executor = _ReferenceExecutor(
        ReferenceExecutionResult(
            next_observation=_observation(),
            mission_observed_delta=0.0,
            priority_observed_delta=0.0,
            normalized_execution_cost_contribution=0.0,
            normalized_execution_time_contribution=0.0,
            executed_without_new_coverage=False,
            success_first_crossing=False,
            episode_ended_without_success=False,
            hard_safety_violation=False,
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


def test_rejecting_outcome_with_executable_reference_fails_closed() -> None:
    """Would fail if a failure outcome could smuggle an executable reference."""
    output = _reference_output(
        "WHEELED", ExecutionDirective.ACTIVATE_NEW_REFERENCE
    )
    output.outcome = PlanningOutcome.INVALID_REQUEST
    executor = _ReferenceExecutor(
        ReferenceExecutionResult(
            next_observation=_observation(),
            mission_observed_delta=1.0,
            priority_observed_delta=1.0,
            normalized_execution_cost_contribution=0.0,
            normalized_execution_time_contribution=0.0,
            executed_without_new_coverage=False,
            success_first_crossing=False,
            episode_ended_without_success=False,
            hard_safety_violation=False,
            terminated=False,
            execution_state="DECISION_BOUNDARY",
        )
    )
    env = V3ExplorationEnvironment(
        platform_type="WHEELED",
        bridge=_Bridge(output),
        request_builder=lambda action: action,
        initial_observation=_observation(),
        reference_executor=executor,
    )

    with pytest.raises(EnvironmentInvariantError, match="outcome"):
        env.step(PolicyAction(frontier_index=0, theta_rad=0.0))

    assert executor.references == []
    assert env.rollout_discarded is True
    assert env.training_stopped is True


def test_success_outcome_without_reference_fails_closed() -> None:
    """Would fail if a missing successful reference were disguised as a hold."""
    output = PlannerOutput()
    output.outcome = PlanningOutcome.NEW_REFERENCE_AVAILABLE
    output.directive = ExecutionDirective.ACTIVATE_NEW_REFERENCE
    output.reason_code = "BROKEN_SUCCESS"
    env = V3ExplorationEnvironment(
        platform_type="WHEELED",
        bridge=_Bridge(output),
        request_builder=lambda action: action,
        initial_observation=_observation(),
    )

    with pytest.raises(EnvironmentInvariantError, match="reference"):
        env.step(PolicyAction(frontier_index=0, theta_rad=0.0))

    assert env.rollout_discarded is True
    assert env.training_stopped is True


@pytest.mark.parametrize(
    ("outcome", "directive"),
    [
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
        (PlanningOutcome.GOAL_INFEASIBLE, ExecutionDirective.HOLD_POSITION),
        (PlanningOutcome.CANCELED, ExecutionDirective.HOLD_POSITION),
        (PlanningOutcome.NUMERICAL_FAILURE, ExecutionDirective.HOLD_POSITION),
    ],
)
def test_current_cpp_v3_no_reference_outputs_hold_without_rejection(
    outcome: PlanningOutcome, directive: ExecutionDirective
) -> None:
    """Would fail if the matrix rejected a combination produced by C++ v3."""
    output = PlannerOutput()
    output.outcome = outcome
    output.directive = directive
    output.reason_code = "CPP_V3_HOLD"
    env = V3ExplorationEnvironment(
        platform_type="WHEELED",
        bridge=_Bridge(output),
        request_builder=lambda action: action,
        initial_observation=_observation(),
    )

    transition = env.step(PolicyAction(frontier_index=0, theta_rad=0.0))

    assert transition.planning_outcome == outcome
    assert transition.execution_directive == directive
    assert transition.reason_code == "CPP_V3_HOLD"


def test_current_cpp_v3_committed_hop_output_uses_execution_feedback() -> None:
    """Would fail if the legal committed-hop output were rejected or held."""
    output = PlannerOutput()
    output.outcome = PlanningOutcome.SAFE_FRONTIER_REFERENCE_AVAILABLE
    output.directive = ExecutionDirective.CONTINUE_COMMITTED_HOP
    output.reason_code = "COMMITTED_HOP_CONTINUES"
    observation = _observation(2)
    feedback = CommittedHopExecutionFeedback(
        execution_state="IN_FLIGHT",
        next_observation=observation,
        mission_observed_delta=0.2,
        priority_observed_delta=0.3,
        normalized_execution_cost_contribution=0.0,
        normalized_execution_time_contribution=0.0,
        executed_without_new_coverage=True,
        success_first_crossing=False,
        episode_ended_without_success=False,
        hard_safety_violation=False,
        terminated=False,
    )
    env = V3ExplorationEnvironment(
        platform_type="HOPPER",
        bridge=_Bridge(output),
        request_builder=lambda action: action,
        initial_observation=_observation(2),
        committed_hop_executor=lambda: feedback,
    )

    transition = env.step(PolicyAction(frontier_index=0, theta_rad=0.0))

    assert transition.next_observation is not observation
    assert transition.mission_observed_delta == 0.2
    assert transition.priority_observed_delta == 0.3
    assert transition.executed_without_new_coverage is True
    assert transition.planning_outcome == PlanningOutcome.SAFE_FRONTIER_REFERENCE_AVAILABLE
    assert transition.execution_directive == ExecutionDirective.CONTINUE_COMMITTED_HOP


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


def test_production_factory_step_uses_real_cpp_v3_bridge() -> None:
    """Would fail if production composition bypassed the in-process C++ planner."""
    from lunar_policy_training.environment.v3_environment import (
        create_v3_environment,
    )

    request = TrainingPlanRequest()
    request.request_id = "production-composition-invalid-map"

    def build_request(action, identity):
        request.state_time.nanoseconds_since_epoch = identity.state_time_ns
        return PreparedPlanRequest(
            request=request,
            identity=identity,
            candidate_id="a" * 64,
            physical_snapshot_id="b" * 64,
        )

    env = create_v3_environment(
        platform_type="WHEELED",
        request_builder=build_request,
        initial_observation=_observation(),
    )

    env.refresh_decision_boundary()
    transition = env.advance_prepared_action(
        PolicyAction(frontier_index=0, theta_rad=0.0),
        expected_identity=env.current_observation.observation_identities[0],
    ).transition

    assert transition is not None
    assert transition.planning_outcome == PlanningOutcome.INVALID_REQUEST
    assert transition.execution_directive == ExecutionDirective.NO_SAFE_REFERENCE
    assert transition.reason_code == "MISSING_MAP_LAYER_ELEVATION"

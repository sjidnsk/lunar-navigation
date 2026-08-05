from __future__ import annotations

from datetime import timedelta
from dataclasses import FrozenInstanceError
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
    TrainingPlanRequest,
)
import torch  # noqa: E402
import pytest  # noqa: E402

from lunar_policy_training.environment.macro_step import PolicyAction  # noqa: E402
from lunar_policy_training.environment.v3_environment import (  # noqa: E402
    CommittedHopExecutionFeedback,
    EnvironmentInvariantError,
    PreparedPlanRequest,
    ReferenceExecutionResult,
    V3ExplorationEnvironment,
)
from lunar_policy_training.policy.observation import (  # noqa: E402
    ObservationIdentity,
    PolicyBatch,
)


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
    candidate_mask: tuple[bool, bool] = (True, False),
    identity: ObservationIdentity | None = None,
) -> PolicyBatch:
    platform_context = torch.zeros((1, 3), dtype=torch.float32)
    platform_context[0, platform_index] = 1.0
    return PolicyBatch(
        prior_channels=torch.zeros((1, 7, 4, 4), dtype=torch.float32),
        coverage_summary=torch.zeros((1, 8, 4, 4), dtype=torch.float32),
        local_crop=torch.zeros((1, 8, 4, 4), dtype=torch.float32),
        frontier_features=torch.zeros((1, 2, 22), dtype=torch.float32),
        pose_features=torch.zeros((1, 6), dtype=torch.float32),
        candidate_mask=torch.tensor([candidate_mask], dtype=torch.bool),
        platform_context=platform_context,
        observation_identities=(identity or _identity(),),
    )


class _Bridge:
    def __init__(self, output: PlannerOutput) -> None:
        self.output = output

    def plan(self, request: object) -> PlannerOutput:
        return self.output


class _SequenceBridge:
    def __init__(self, outputs: list[PlannerOutput]) -> None:
        self.outputs = iter(outputs)

    def plan(self, request: object) -> PlannerOutput:
        return next(self.outputs)


class _ReferenceExecutor:
    def __init__(self, result: ReferenceExecutionResult) -> None:
        self.result = result
        self.references: list[MotionReference] = []

    def __call__(self, reference: MotionReference) -> ReferenceExecutionResult:
        self.references.append(reference)
        return self.result


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
def test_normal_v3_rejection_consumes_decision_and_masks_only_selected_candidate(
    outcome: PlanningOutcome,
    directive: ExecutionDirective,
) -> None:
    """Would fail if one rejected frontier poisoned unrelated candidates."""
    output = PlannerOutput()
    output.outcome = outcome
    output.directive = directive
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

    assert result.decision_budget_consumed == 1
    assert result.transition is not None
    assert policy_masks[0].tolist() == [[True, True]]
    assert result.transition.next_observation.candidate_mask.tolist() == [
        [False, True]
    ]


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
def test_new_observation_identity_component_clears_temporary_rejections(
    identity_change: str, changed_value: object
) -> None:
    """Would fail if rejection leaked across any producer identity component."""
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
        [[False, True]],
        [[True, True]],
    ]


def test_same_observation_identity_preserves_temporary_rejections() -> None:
    """Would fail if a tensor refresh silently forgot same-boundary rejection state."""
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

    assert seen == [[[True, True]], [[False, True]]]


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


def test_rejection_refresh_with_same_identity_keeps_temporary_mask() -> None:
    """Would fail if every producer refresh erased same-boundary rejection state."""
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

    assert seen == [[[True, True]], [[False, True]]]


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
    assert result.decision_budget_consumed == 1
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
        return PreparedPlanRequest(request=request, identity=drifted_identity)

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


def test_installed_observation_is_cloned_before_private_budget_update() -> None:
    """Would fail if environment budget writes mutated producer-owned tensors."""
    output = PlannerOutput()
    output.outcome = PlanningOutcome.INVALID_REQUEST
    output.directive = ExecutionDirective.NO_SAFE_REFERENCE
    output.reason_code = "INVALID"
    initial = _observation(candidate_mask=(True, True))
    producer_refresh = _observation(
        candidate_mask=(True, True),
        identity=_identity(map_snapshot_id="map-2"),
    )
    producer_refresh.pose_features[0, 5] = 0.625
    provider = _ObservationProvider(producer_refresh)
    env = V3ExplorationEnvironment(
        platform_type="WHEELED",
        bridge=_Bridge(output),
        request_builder=lambda action: action,
        initial_observation=initial,
        observation_provider=provider,
        total_decision_budget=4,
        remaining_decision_budget=3,
    )

    env.advance_until_decision_boundary(
        lambda observation: PolicyAction(frontier_index=0, theta_rad=0.0)
    )
    published = env.current_observation
    published.pose_features[0, 5] = 0.0

    assert float(producer_refresh.pose_features[0, 5]) == pytest.approx(0.625)
    assert producer_refresh.candidate_mask.tolist() == [[True, True]]
    assert float(env.current_observation.pose_features[0, 5]) == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("total", "remaining", "expected_ratio"),
    ((8, 8, 1.0), (8, 3, 0.375), (1, 0, 0.0)),
)
def test_initial_and_resumed_decision_budget_publish_exact_ratio(
    total: int, remaining: int, expected_ratio: float
) -> None:
    producer = _observation(candidate_mask=(True, True))
    producer.pose_features[0, 5] = 0.123

    env = V3ExplorationEnvironment(
        platform_type="WHEELED",
        bridge=_Bridge(PlannerOutput()),
        request_builder=lambda action: action,
        initial_observation=producer,
        total_decision_budget=total,
        remaining_decision_budget=remaining,
    )

    assert float(env.current_observation.pose_features[0, 5]) == pytest.approx(
        expected_ratio
    )
    assert float(producer.pose_features[0, 5]) == pytest.approx(0.123)


def test_policy_decision_consumes_real_budget_and_updates_network_ratio() -> None:
    """Would fail if budget accounting remained helper-only metadata."""
    output = PlannerOutput()
    output.outcome = PlanningOutcome.NO_KNOWN_SAFE_ROUTE
    output.directive = ExecutionDirective.NO_SAFE_REFERENCE
    output.reason_code = "NO_ROUTE"
    env = V3ExplorationEnvironment(
        platform_type="WHEELED",
        bridge=_Bridge(output),
        request_builder=lambda action: action,
        initial_observation=_observation(candidate_mask=(True, True)),
        total_decision_budget=4,
        remaining_decision_budget=4,
    )
    ratios: list[float] = []

    result = env.advance_until_decision_boundary(
        lambda observation: (
            ratios.append(float(observation.pose_features[0, 5].item()))
            or PolicyAction(frontier_index=0, theta_rad=0.0)
        )
    )

    assert result.decision_budget_consumed == 1
    assert ratios == pytest.approx([1.0])
    assert result.transition is not None
    assert float(result.transition.next_observation.pose_features[0, 5]) == pytest.approx(0.75)


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
    assert result.decision_budget_consumed == 0


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
    assert next_observation.pose_features[0, 5].item() == pytest.approx(0.0)
    assert transition.mission_observed_delta == 0.2
    assert transition.priority_observed_delta == 0.1
    assert transition.normalized_plan_or_execution_cost == 0.5
    assert transition.normalized_macro_step_time == 0.25
    assert transition.executed_without_new_coverage is True
    assert transition.planning_outcome == PlanningOutcome.NEW_REFERENCE_AVAILABLE
    assert transition.execution_directive == ExecutionDirective.ACTIVATE_NEW_REFERENCE
    assert transition.reason_code == "REFERENCE_READY"
    assert transition.terminated is False


def test_mismatched_reference_platform_discards_rollout_and_stops_training() -> None:
    """Would fail if a hopper reference could execute in a wheeled episode."""
    executor = _ReferenceExecutor(
        ReferenceExecutionResult(
            next_observation=_observation(),
            mission_observed_delta=0.0,
            priority_observed_delta=0.0,
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
    assert observation.pose_features[0, 5].item() == pytest.approx(0.0)
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
        return PreparedPlanRequest(request=request, identity=identity)

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

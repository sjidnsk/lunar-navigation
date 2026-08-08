from __future__ import annotations

from datetime import timedelta

from lunar_planner_training_bridge import (
    ExecutionDirective,
    MotionReference,
    PlannerOutput,
    PlanningOutcome,
)
import numpy as np
import pytest
import torch

from lunar_policy_training.environment.macro_step import PolicyAction
from lunar_policy_training.environment.observation_boundary import (
    ObservationBoundaryController,
    SensorBoundaryEvidence,
)
from lunar_policy_training.environment.observation_builder import Pose2
from lunar_policy_training.environment.sensor_observation import (
    SensorObservationState,
    TrainingObservedGrid,
    TrainingWorldTruth,
)
from lunar_policy_training.environment.v3_environment import (
    CommittedHopExecutionFeedback,
    EnvironmentInvariantError,
    ReferenceExecutionResult,
    V3ExplorationEnvironment,
    create_v3_environment,
)
from lunar_policy_training.environment.visibility import (
    NativeVisibilityEstimator,
    SensorGeometry,
)
from lunar_policy_training.polar_data.raster import (
    GridGeometry,
    MapCanvas,
)
from lunar_policy_training.policy.observation import PolicyBatch


_PLATFORM_INDEX = {"WHEELED": 0, "LEGGED": 1, "HOPPER": 2}


class _PolicyBuilder:
    def __init__(self, platform_type: str) -> None:
        self.platform_type = platform_type
        self.visible_counts: list[int] = []

    def __call__(
        self, observed: TrainingObservedGrid, pose_map: Pose2
    ) -> PolicyBatch:
        visible_count = int(np.count_nonzero(observed.valid_mask))
        self.visible_counts.append(visible_count)
        platform_context = torch.zeros((1, 3), dtype=torch.float32)
        platform_context[0, _PLATFORM_INDEX[self.platform_type]] = 1.0
        pose = torch.zeros((1, 5), dtype=torch.float32)
        pose[0, 0] = float(visible_count) / 1000.0
        pose[0, 1] = pose_map.x_m / 100.0
        return PolicyBatch(
            prior_channels=torch.zeros((1, 4, 256, 256), dtype=torch.float32),
            coverage_summary=torch.zeros(
                (1, 3, 256, 256), dtype=torch.float32
            ),
            local_crop=torch.zeros((1, 4, 32, 32), dtype=torch.float32),
            frontier_features=torch.zeros((1, 64, 12), dtype=torch.float32),
            pose_features=pose,
            candidate_mask=torch.tensor(
                [[True] + [False] * 63], dtype=torch.bool
            ),
            platform_context=platform_context,
        )


def _controller(
    platform_type: str,
) -> tuple[ObservationBoundaryController, _PolicyBuilder, MapCanvas]:
    geometry = GridGeometry(size_m=11.0, resolution_m=1.0, cells=11)
    canvas = MapCanvas("d" * 64, (0.0, 0.0, 11.0, 11.0), geometry)
    truth = TrainingWorldTruth(
        canvas,
        np.zeros((11, 11), dtype=np.float32),
        np.zeros((11, 11), dtype=np.float32),
    )
    sensor_state = SensorObservationState(
        truth=truth,
        observed=TrainingObservedGrid.empty(canvas),
        mission_roi_ratio=np.ones((11, 11), dtype=np.float32),
        mission_priority=np.ones((11, 11), dtype=np.float32),
        forbidden_mask=np.zeros((11, 11), dtype=np.bool_),
        visibility_estimator=NativeVisibilityEstimator(
            SensorGeometry(2.0, 2.0 * np.pi), resolution_m=1.0
        ),
    )
    builder = _PolicyBuilder(platform_type)
    return (
        ObservationBoundaryController(
            platform_type=platform_type,
            sensor_state=sensor_state,
            policy_observation_builder=builder,
            episode_id="sensor-episode",
            mission_revision=3,
        ),
        builder,
        canvas,
    )


def _pose(canvas: MapCanvas, row: int, column: int) -> Pose2:
    x_m, y_m = canvas.grid_center_world(row, column)
    return Pose2(x_m=x_m, y_m=y_m)


def _reference_output(platform_type: str) -> PlannerOutput:
    output = PlannerOutput()
    output.outcome = PlanningOutcome.NEW_REFERENCE_AVAILABLE
    output.directive = ExecutionDirective.ACTIVATE_NEW_REFERENCE
    output.reason_code = "REFERENCE_READY"
    output.reference = MotionReference()
    output.reference.platform_type = platform_type
    output.diagnostics.best_cost = 0.0
    output.diagnostics.elapsed = timedelta(0)
    return output


class _Bridge:
    def __init__(self, output: PlannerOutput) -> None:
        self.output = output

    def plan(self, request: object) -> PlannerOutput:
        return self.output


def _execution_result(
    next_observation: PolicyBatch,
    *,
    execution_state: str,
    evidence: SensorBoundaryEvidence | None,
) -> ReferenceExecutionResult:
    return ReferenceExecutionResult(
        next_observation=next_observation,
        mission_observed_delta=0.99,
        priority_observed_delta=0.88,
        normalized_execution_cost_contribution=0.0,
        normalized_execution_time_contribution=0.0,
        executed_without_new_coverage=False,
        success_first_crossing=False,
        episode_ended_without_success=False,
        hard_safety_violation=False,
        terminated=False,
        execution_state=execution_state,
        sensor_boundary_evidence=evidence,
    )


def test_reset_reveals_before_candidates_and_ground_boundary_uses_actual_area() -> None:
    controller, builder, canvas = _controller("WHEELED")

    initial = controller.reset(_pose(canvas, 5, 2))
    updated = controller.after_execution(
        platform_type="WHEELED",
        execution_state="DECISION_BOUNDARY",
        evidence=SensorBoundaryEvidence(_pose(canvas, 5, 8), 2.0),
    )

    assert initial.updated is True
    assert initial.mission_observed_delta == 0.0
    assert initial.priority_observed_delta == 0.0
    assert len(builder.visible_counts) == 2
    assert builder.visible_counts[0] > 0
    assert builder.visible_counts[1] > builder.visible_counts[0]
    newly_visible = builder.visible_counts[1] - builder.visible_counts[0]
    assert updated.mission_observed_delta == pytest.approx(newly_visible / 121.0)
    assert updated.priority_observed_delta == pytest.approx(newly_visible / 121.0)
    identity = updated.next_observation.observation_identities[0]
    assert identity.execution_state == "DECISION_BOUNDARY"
    assert identity.state_time_ns == 2_000_000_000


def test_hopper_updates_once_at_landed_hold_and_never_in_flight() -> None:
    controller, builder, canvas = _controller("HOPPER")
    controller.reset(_pose(canvas, 5, 2))

    committed = controller.after_execution(
        platform_type="HOPPER",
        execution_state="JUMP_COMMITTED",
        evidence=None,
    )
    flying = controller.after_execution(
        platform_type="HOPPER",
        execution_state="IN_FLIGHT",
        evidence=None,
    )
    landed = controller.after_execution(
        platform_type="HOPPER",
        execution_state="LANDED_HOLD",
        evidence=SensorBoundaryEvidence(_pose(canvas, 5, 8), 4.0),
    )

    assert committed.updated is False
    assert flying.updated is False
    assert committed.mission_observed_delta == 0.0
    assert flying.priority_observed_delta == 0.0
    assert landed.updated is True
    assert landed.mission_observed_delta > 0.0
    assert len(builder.visible_counts) == 2

    with pytest.raises(ValueError, match="in-flight"):
        controller.after_execution(
            platform_type="HOPPER",
            execution_state="IN_FLIGHT",
            evidence=SensorBoundaryEvidence(_pose(canvas, 5, 8), 1.0),
        )


def test_sensor_closed_ground_environment_ignores_executor_coverage_claims() -> None:
    controller, _, canvas = _controller("LEGGED")
    initial = controller.reset(_pose(canvas, 5, 2)).next_observation
    fake_next = _PolicyBuilder("LEGGED")(
        TrainingObservedGrid.empty(controller.sensor_state.truth.canvas),
        _pose(canvas, 5, 2),
    )
    fake_next.pose_features[0, 0] = 0.999
    execution = _execution_result(
        fake_next,
        execution_state="DECISION_BOUNDARY",
        evidence=SensorBoundaryEvidence(_pose(canvas, 5, 8), 1.0),
    )
    environment = V3ExplorationEnvironment(
        platform_type="LEGGED",
        bridge=_Bridge(_reference_output("LEGGED")),
        request_builder=lambda action: action,
        initial_observation=initial,
        reference_executor=lambda reference: execution,
        observation_boundary_controller=controller,
        require_sensor_closed_loop=True,
    )

    transition = environment.step(PolicyAction(0, 0.0))

    assert transition.mission_observed_delta != 0.99
    assert transition.priority_observed_delta != 0.88
    assert transition.mission_observed_delta > 0.0
    assert transition.next_observation.pose_features[0, 0].item() != pytest.approx(
        0.999
    )
    assert transition.executed_without_new_coverage is False


def test_sensor_closed_environment_fails_before_accepting_missing_evidence() -> None:
    controller, _, canvas = _controller("WHEELED")
    initial = controller.reset(_pose(canvas, 5, 2)).next_observation
    execution = _execution_result(
        initial,
        execution_state="DECISION_BOUNDARY",
        evidence=None,
    )
    environment = V3ExplorationEnvironment(
        platform_type="WHEELED",
        bridge=_Bridge(_reference_output("WHEELED")),
        request_builder=lambda action: action,
        initial_observation=initial,
        reference_executor=lambda reference: execution,
        observation_boundary_controller=controller,
        require_sensor_closed_loop=True,
    )

    with pytest.raises(EnvironmentInvariantError, match="sensor boundary evidence"):
        environment.step(PolicyAction(0, 0.0))

    assert environment.rollout_discarded
    assert environment.training_stopped


def test_sensor_closed_hopper_ignores_in_flight_claims_and_reveals_on_landing() -> None:
    controller, builder, canvas = _controller("HOPPER")
    initial = controller.reset(_pose(canvas, 5, 2)).next_observation
    output = PlannerOutput()
    output.outcome = PlanningOutcome.SAFE_FRONTIER_REFERENCE_AVAILABLE
    output.directive = ExecutionDirective.CONTINUE_COMMITTED_HOP
    output.reason_code = "COMMITTED_HOP_CONTINUES"
    feedback = iter(
        (
            CommittedHopExecutionFeedback(
                execution_state="JUMP_COMMITTED",
                next_observation=initial,
                mission_observed_delta=0.9,
                priority_observed_delta=0.9,
                normalized_execution_cost_contribution=0.0,
                normalized_execution_time_contribution=0.1,
                executed_without_new_coverage=False,
                success_first_crossing=False,
                episode_ended_without_success=False,
                hard_safety_violation=False,
                terminated=False,
            ),
            CommittedHopExecutionFeedback(
                execution_state="IN_FLIGHT",
                next_observation=initial,
                mission_observed_delta=0.9,
                priority_observed_delta=0.9,
                normalized_execution_cost_contribution=0.0,
                normalized_execution_time_contribution=0.1,
                executed_without_new_coverage=False,
                success_first_crossing=False,
                episode_ended_without_success=False,
                hard_safety_violation=False,
                terminated=False,
            ),
            CommittedHopExecutionFeedback(
                execution_state="LANDED_HOLD",
                next_observation=initial,
                mission_observed_delta=0.9,
                priority_observed_delta=0.9,
                normalized_execution_cost_contribution=0.0,
                normalized_execution_time_contribution=0.1,
                executed_without_new_coverage=False,
                success_first_crossing=False,
                episode_ended_without_success=False,
                hard_safety_violation=False,
                terminated=False,
                sensor_boundary_evidence=SensorBoundaryEvidence(
                    _pose(canvas, 5, 8), 3.0
                ),
            ),
        )
    )
    environment = V3ExplorationEnvironment(
        platform_type="HOPPER",
        bridge=_Bridge(output),
        request_builder=lambda action: action,
        initial_observation=initial,
        committed_hop_executor=lambda: next(feedback),
        observation_boundary_controller=controller,
        require_sensor_closed_loop=True,
    )

    result = environment.advance_prepared_action(
        PolicyAction(0, 0.0),
        expected_identity=initial.observation_identities[0],
    )

    assert result.execution_state == "LANDED_HOLD"
    assert result.transition is not None
    assert len(builder.visible_counts) == 2
    newly_visible = builder.visible_counts[1] - builder.visible_counts[0]
    assert result.transition.mission_observed_delta == pytest.approx(
        newly_visible / 121.0
    )
    assert result.transition.priority_observed_delta == pytest.approx(
        newly_visible / 121.0
    )
    assert result.transition.mission_observed_delta != pytest.approx(2.7)


def test_formal_factory_requires_initialized_sensor_closed_controller() -> None:
    controller, _, canvas = _controller("WHEELED")
    initial = controller.reset(_pose(canvas, 5, 2)).next_observation

    with pytest.raises(ValueError, match="observation boundary controller"):
        create_v3_environment(
            platform_type="WHEELED",
            request_builder=lambda action, identity: None,
            initial_observation=initial,
            require_sensor_closed_loop=True,
        )

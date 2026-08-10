from __future__ import annotations

import numpy as np
import pytest

from lunar_exploration_policy.action_selection import SelectedAction
from lunar_exploration_policy.coordinator import (
    ClosedLoopCoordinator,
    CoordinatorError,
    CoordinatorState,
    ExecutionFeedback,
    PlannerResult,
)
from lunar_exploration_policy.identity import DecisionIdentity
from lunar_exploration_policy.observation_runtime import ObservationSnapshot
from lunar_policy_training.environment.candidate_builder import (
    CandidateBatch,
    CandidateDiagnostics,
)
from lunar_policy_training.polar_data.raster import MapCanvas


class _FixedPolicy:
    def __init__(self, index: int, theta: float | None) -> None:
        self.index = index
        self.theta = theta

    def decide(self, inputs, platform_type):
        if not inputs["candidate_mask"].any():
            return None
        return SelectedAction(self.index, self.theta, 0.5)


def _snapshot(
    platform: str,
    *,
    revision: int = 1,
    map_id: str = "map-1",
    state_id: str = "state-1",
    candidate_id: str = "candidates-1",
    x_norm: float = 0.25,
    y_norm: float = 0.75,
) -> ObservationSnapshot:
    canvas = MapCanvas.from_roi_bounds("d" * 64, (500.0, 500.0, 524.0, 524.0))
    features = np.zeros((64, 12), np.float32)
    mask = np.zeros(64, np.bool_)
    mask[0] = True
    features[0, 0] = x_norm
    features[0, 1] = y_norm
    candidates = CandidateBatch(
        features,
        mask,
        canvas.identity,
        CandidateDiagnostics(1, 0, 1),
    )
    identity = DecisionIdentity(
        mission_revision=revision,
        map_snapshot_id=map_id,
        robot_state_id=state_id,
        state_time_ns=revision * 1000,
        execution_state="DECISION_BOUNDARY",
        candidate_set_id=candidate_id,
    )
    return ObservationSnapshot(
        identity=identity,
        arrays={"candidate_mask": mask[None]},
        candidates=candidates,
        candidate_set_id=candidate_id,
        canvas=canvas,
        elevation_m=np.full((256, 256), 12.5, np.float32),
        platform_type=platform,
    )


@pytest.mark.parametrize("platform", ("WHEELED", "LEGGED"))
def test_ground_platform_feedback_allows_second_fresh_decision(platform: str) -> None:
    coordinator = ClosedLoopCoordinator(_FixedPolicy(0, 0.4))
    first = coordinator.start_decision(_snapshot(platform), mission_id="mission-A")
    assert first.target_xyz == pytest.approx((256.0, 256.0, 12.5))
    assert first.tolerance_m == pytest.approx(0.2)
    assert first.theta_rad == pytest.approx(0.4)
    coordinator.accept_planner_result(
        PlannerResult(first.request_id, True, "plan-1", "REFERENCE_READY")
    )

    accepted = coordinator.accept_feedback(
        ExecutionFeedback(1, platform, "plan-1", "SEGMENT_COMPLETE")
    )
    second = coordinator.start_decision(
        _snapshot(
            platform,
            revision=2,
            map_id="map-2",
            state_id="state-2",
            candidate_id="candidates-2",
            x_norm=0.75,
            y_norm=0.25,
        ),
        mission_id="mission-A",
    )

    assert accepted is True
    assert second.request_id != first.request_id
    assert second.target_xyz == pytest.approx((768.0, 768.0, 12.5))
    assert second.identity.map_snapshot_id == "map-2"


def test_hopper_landing_rebuilds_state_before_second_goal() -> None:
    coordinator = ClosedLoopCoordinator(_FixedPolicy(0, None))
    first = coordinator.start_decision(_snapshot("HOPPER"), mission_id="mission-H")
    assert first.tolerance_m == 0.0
    assert first.theta_rad is None
    coordinator.accept_planner_result(
        PlannerResult(first.request_id, True, "hop-1", "REFERENCE_READY")
    )
    assert coordinator.state is CoordinatorState.LANDED_HOLD

    assert coordinator.accept_feedback(
        ExecutionFeedback(1, "HOPPER", "hop-1", "LANDED_HOLD")
    )
    second = coordinator.start_decision(
        _snapshot(
            "HOPPER",
            revision=2,
            map_id="landed-map",
            state_id="landed-pose",
            candidate_id="landing-candidates",
        ),
        mission_id="mission-H",
    )

    assert second.identity.robot_state_id == "landed-pose"
    assert second.request_id != first.request_id


def test_foreign_or_late_feedback_does_not_clear_current_plan() -> None:
    coordinator = ClosedLoopCoordinator(_FixedPolicy(0, 0.0))
    goal = coordinator.start_decision(_snapshot("WHEELED"), mission_id="mission")
    coordinator.accept_planner_result(
        PlannerResult(goal.request_id, True, "plan-current", "REFERENCE_READY")
    )

    assert not coordinator.accept_feedback(
        ExecutionFeedback(1, "WHEELED", "foreign", "SEGMENT_COMPLETE")
    )
    assert not coordinator.accept_feedback(
        ExecutionFeedback(0, "WHEELED", "plan-current", "SEGMENT_COMPLETE")
    )
    assert coordinator.state is CoordinatorState.EXECUTING
    assert coordinator.active_plan_id == "plan-current"


def test_planner_infeasible_waits_for_new_boundary_without_fake_reference() -> None:
    coordinator = ClosedLoopCoordinator(_FixedPolicy(0, 0.0))
    goal = coordinator.start_decision(_snapshot("LEGGED"), mission_id="mission")

    coordinator.accept_planner_result(
        PlannerResult(goal.request_id, False, "", "NO_KNOWN_SAFE_ROUTE")
    )

    assert coordinator.state is CoordinatorState.HOLD_ERROR
    assert coordinator.active_plan_id is None
    assert coordinator.last_reason == "NO_KNOWN_SAFE_ROUTE"


def test_new_decision_is_forbidden_before_matching_execution_feedback() -> None:
    coordinator = ClosedLoopCoordinator(_FixedPolicy(0, 0.0))
    goal = coordinator.start_decision(_snapshot("WHEELED"), mission_id="mission")
    coordinator.accept_planner_result(
        PlannerResult(goal.request_id, True, "plan", "REFERENCE_READY")
    )

    with pytest.raises(CoordinatorError, match="EXECUTING"):
        coordinator.start_decision(
            _snapshot("WHEELED", revision=2), mission_id="mission"
        )

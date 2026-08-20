"""Coordinator decisions before ROS message adaptation."""

from __future__ import annotations

from dataclasses import dataclass

from luna_wheeled_controller.coordinator import (
    ActiveMission,
    ExecutionGoal,
    PlanResult,
    decide_plan_result,
    make_plan_request,
)


@dataclass(frozen=True)
class Reference:
    platform_type: int
    plan_id: str


def test_active_mission_and_goal_form_one_versioned_plan_request() -> None:
    request = make_plan_request(
        ActiveMission(mission_id="mission-a", revision=7),
        ExecutionGoal(region="goal-region"),
        request_id="wheel-1",
    )

    assert (request.request_id, request.mission_id, request.mission_revision) == ("wheel-1", "mission-a", 7)
    assert request.region == "goal-region"


def test_only_wheeled_activation_is_forwarded() -> None:
    reference = Reference(platform_type=1, plan_id="wheel-1")
    decision = decide_plan_result(
        PlanResult(has_reference=True, execution_directive=0, reference=reference)
    )

    assert decision.reference is reference
    assert decision.cancel_reason is None


def test_hold_result_cancels_without_reusing_a_previous_reference() -> None:
    decision = decide_plan_result(
        PlanResult(has_reference=False, execution_directive=2, reference=None)
    )

    assert decision.reference is None
    assert decision.cancel_reason == "REFERENCE_WITHDRAWN"


def test_non_wheeled_activation_is_cancelled() -> None:
    decision = decide_plan_result(
        PlanResult(has_reference=True, execution_directive=0, reference=Reference(platform_type=2, plan_id="leg-1"))
    )

    assert decision.reference is None
    assert decision.cancel_reason == "REFERENCE_WITHDRAWN"

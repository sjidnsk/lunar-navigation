"""ROS-independent decisions for forwarding WHEELED planning references."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar


ReferenceT = TypeVar("ReferenceT")

WHEELED_PLATFORM = 1
ACTIVATE_NEW_REFERENCE = 0


@dataclass(frozen=True)
class ActiveMission:
    mission_id: str
    revision: int


@dataclass(frozen=True)
class ExecutionGoal:
    region: object


@dataclass(frozen=True)
class PlanRequest:
    request_id: str
    mission_id: str
    mission_revision: int
    region: object


@dataclass(frozen=True)
class PlanResult(Generic[ReferenceT]):
    has_reference: bool
    execution_directive: int
    reference: ReferenceT | None


@dataclass(frozen=True)
class CoordinatorDecision(Generic[ReferenceT]):
    reference: ReferenceT | None
    cancel_reason: str | None


def make_plan_request(
    mission: ActiveMission,
    goal: ExecutionGoal,
    *,
    request_id: str,
) -> PlanRequest:
    if not mission.mission_id or mission.revision < 0 or not request_id:
        raise ValueError("mission and request identity must be valid")
    return PlanRequest(
        request_id=request_id,
        mission_id=mission.mission_id,
        mission_revision=mission.revision,
        region=goal.region,
    )


def decide_plan_result(result: PlanResult[ReferenceT]) -> CoordinatorDecision[ReferenceT]:
    reference = result.reference
    if (
        result.has_reference
        and result.execution_directive == ACTIVATE_NEW_REFERENCE
        and reference is not None
        and getattr(reference, "platform_type", None) == WHEELED_PLATFORM
        and bool(getattr(reference, "plan_id", ""))
    ):
        return CoordinatorDecision(reference=reference, cancel_reason=None)
    return CoordinatorDecision(reference=None, cancel_reason="REFERENCE_WITHDRAWN")

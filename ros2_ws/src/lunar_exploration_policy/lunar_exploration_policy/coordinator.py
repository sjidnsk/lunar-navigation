"""平台无关的单决策闭环状态机。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
import math
from typing import Protocol

from .action_selection import SelectedAction
from .identity import DecisionIdentity
from .observation_runtime import ObservationSnapshot


class CoordinatorError(RuntimeError):
    """调用顺序会混用观测、规划或执行身份。"""


class CoordinatorState(Enum):
    WAITING_INPUTS = auto()
    READY = auto()
    INFERENCING = auto()
    PLANNING = auto()
    EXECUTING = auto()
    LANDED_HOLD = auto()
    HOLD_ERROR = auto()


class DecisionPolicy(Protocol):
    def decide(
        self, inputs, platform_type: str
    ) -> SelectedAction | None: ...


@dataclass(frozen=True, slots=True)
class PlannerGoal:
    request_id: str
    mission_id: str
    mission_revision: int
    identity: DecisionIdentity
    frontier_index: int
    target_xyz: tuple[float, float, float]
    tolerance_m: float
    theta_rad: float | None
    yaw_tolerance_rad: float


@dataclass(frozen=True, slots=True)
class PlannerResult:
    request_id: str
    has_reference: bool
    plan_id: str
    reason_code: str


@dataclass(frozen=True, slots=True)
class ExecutionFeedback:
    sequence: int
    platform_type: str
    plan_id: str
    state: str


class ClosedLoopCoordinator:
    """保证一次动作、PlanMotion 和反馈共享同一生产者身份。"""

    def __init__(self, policy: DecisionPolicy) -> None:
        if not hasattr(policy, "decide"):
            raise TypeError("policy must provide decide()")
        self._policy = policy
        self.state = CoordinatorState.WAITING_INPUTS
        self.active_plan_id: str | None = None
        self.last_reason = "WAITING_INPUTS"
        self._pending_request_id: str | None = None
        self._platform_type: str | None = None
        self._last_feedback_sequence = 0
        self._last_identity: DecisionIdentity | None = None

    def start_decision(
        self,
        snapshot: ObservationSnapshot,
        *,
        mission_id: str,
    ) -> PlannerGoal | None:
        if self.state not in (
            CoordinatorState.WAITING_INPUTS,
            CoordinatorState.HOLD_ERROR,
        ):
            raise CoordinatorError(
                f"cannot start a decision while {self.state.name}"
            )
        if not isinstance(snapshot, ObservationSnapshot):
            raise TypeError("snapshot must use ObservationSnapshot")
        if not isinstance(mission_id, str) or not mission_id:
            raise ValueError("mission_id must be non-empty")
        if snapshot.platform_type not in ("WHEELED", "LEGGED", "HOPPER"):
            raise ValueError("snapshot platform type is invalid")
        if self.state is CoordinatorState.HOLD_ERROR and snapshot.identity == self._last_identity:
            raise CoordinatorError("HOLD_ERROR requires a new decision boundary")
        self.state = CoordinatorState.READY
        self.state = CoordinatorState.INFERENCING
        action = self._policy.decide(snapshot.arrays, snapshot.platform_type)
        self._last_identity = snapshot.identity
        self._platform_type = snapshot.platform_type
        if action is None:
            self.state = CoordinatorState.HOLD_ERROR
            self.last_reason = "NO_CANDIDATE"
            return None
        goal = self._goal(snapshot, mission_id, action)
        self._pending_request_id = goal.request_id
        self.state = CoordinatorState.PLANNING
        self.last_reason = "PLANNING"
        return goal

    def _goal(
        self,
        snapshot: ObservationSnapshot,
        mission_id: str,
        action: SelectedAction,
    ) -> PlannerGoal:
        snapshot.require_identity(snapshot.identity)
        if not 0 <= action.frontier_index < snapshot.candidates.features.shape[0]:
            raise CoordinatorError("selected frontier index is out of range")
        if not snapshot.candidates.mask[action.frontier_index]:
            raise CoordinatorError("selected frontier index is masked")
        feature = snapshot.candidates.features[action.frontier_index]
        left, _, _, top = snapshot.canvas.bounds_m
        x_m = left + float(feature[0]) * snapshot.canvas.geometry.size_m
        y_m = top - float(feature[1]) * snapshot.canvas.geometry.size_m
        try:
            row, column = snapshot.canvas.world_to_grid(x_m, y_m)
        except ValueError as error:
            raise CoordinatorError("selected target is outside the canvas") from error
        z_m = float(snapshot.elevation_m[row, column])
        if not math.isfinite(z_m):
            raise CoordinatorError("selected target elevation is unknown")
        identity = snapshot.identity
        request_id = (
            f"interface-v1/{mission_id}/{identity.mission_revision}/"
            f"{identity.map_snapshot_id}/{identity.robot_state_id}/"
            f"{identity.candidate_set_id}/{action.frontier_index}"
        )
        hopper = snapshot.platform_type == "HOPPER"
        return PlannerGoal(
            request_id=request_id,
            mission_id=mission_id,
            mission_revision=identity.mission_revision,
            identity=identity,
            frontier_index=action.frontier_index,
            target_xyz=(x_m, y_m, z_m),
            tolerance_m=0.0 if hopper else 0.2,
            theta_rad=None if hopper else action.theta_rad,
            yaw_tolerance_rad=0.0 if hopper else math.pi / 24.0,
        )

    def accept_planner_result(self, result: PlannerResult) -> None:
        if self.state is not CoordinatorState.PLANNING:
            raise CoordinatorError("planner result arrived outside PLANNING")
        if not isinstance(result, PlannerResult):
            raise TypeError("result must use PlannerResult")
        if result.request_id != self._pending_request_id:
            raise CoordinatorError("planner result request_id mismatch")
        self._pending_request_id = None
        self.last_reason = result.reason_code
        if not result.has_reference:
            self.active_plan_id = None
            self.state = CoordinatorState.HOLD_ERROR
            return
        if not result.plan_id:
            raise CoordinatorError("planner reference requires a plan_id")
        self.active_plan_id = result.plan_id
        self._last_feedback_sequence = 0
        self.state = (
            CoordinatorState.LANDED_HOLD
            if self._platform_type == "HOPPER"
            else CoordinatorState.EXECUTING
        )

    def accept_feedback(self, feedback: ExecutionFeedback) -> bool:
        if not isinstance(feedback, ExecutionFeedback):
            raise TypeError("feedback must use ExecutionFeedback")
        if self.state not in (
            CoordinatorState.EXECUTING,
            CoordinatorState.LANDED_HOLD,
        ):
            return False
        if (
            feedback.platform_type != self._platform_type
            or feedback.plan_id != self.active_plan_id
            or type(feedback.sequence) is not int
            or feedback.sequence <= self._last_feedback_sequence
        ):
            return False
        if feedback.state in ("FAILED", "CANCELED"):
            self._last_feedback_sequence = feedback.sequence
            self.active_plan_id = None
            self._platform_type = None
            self.state = CoordinatorState.HOLD_ERROR
            self.last_reason = feedback.state
            return False
        terminal = (
            feedback.state == "LANDED_HOLD"
            if self._platform_type == "HOPPER"
            else feedback.state == "SEGMENT_COMPLETE"
        )
        if not terminal:
            return False
        self._last_feedback_sequence = feedback.sequence
        self.active_plan_id = None
        self._platform_type = None
        self.state = CoordinatorState.WAITING_INPUTS
        self.last_reason = feedback.state
        return True

    def reset(self, reason: str) -> None:
        """生命周期停用或任务暂停时丢弃本节点拥有的上下文。"""
        if not isinstance(reason, str) or not reason:
            raise ValueError("reset reason must be non-empty")
        self.active_plan_id = None
        self._pending_request_id = None
        self._platform_type = None
        self._last_feedback_sequence = 0
        self.state = CoordinatorState.WAITING_INPUTS
        self.last_reason = reason


__all__ = [
    "ClosedLoopCoordinator",
    "CoordinatorError",
    "CoordinatorState",
    "ExecutionFeedback",
    "PlannerGoal",
    "PlannerResult",
]

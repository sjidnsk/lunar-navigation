"""Planner-scale transitions for the shared seven-input PPO policy."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

from lunar_planner_training_bridge import ExecutionDirective, PlanningOutcome

from ..policy.observation import PolicyBatch
from ..reward_contract import RewardTerminalClass
from .candidate_builder import CandidateDiagnostics
from .multires_observation import HopperTrajectoryPoint
from .observation_builder import Pose2


@dataclass(frozen=True, slots=True)
class HopperObservationCommitment:
    """Identity and certified endpoint facts frozen before one physical hop."""

    commitment_id: str
    takeoff_pose: Pose2
    expected_landing_pose: Pose2
    flight_time_s: float
    flight_tube_radius_m: float

    def __post_init__(self) -> None:
        if (
            not isinstance(self.commitment_id, str)
            or len(self.commitment_id) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.commitment_id
            )
        ):
            raise ValueError("hopper commitment identity is invalid")
        for name, pose in (
            ("takeoff", self.takeoff_pose),
            ("landing", self.expected_landing_pose),
        ):
            if not isinstance(pose, Pose2) or pose.frame_id != "map":
                raise ValueError(f"hopper commitment {name} pose is invalid")
            if any(
                not math.isfinite(float(value))
                for value in (pose.x_m, pose.y_m, pose.yaw_rad, pose.elevation_m)
            ):
                raise ValueError(f"hopper commitment {name} pose is non-finite")
        for name, value in (
            ("flight time", self.flight_time_s),
            ("flight tube radius", self.flight_tube_radius_m),
        ):
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise ValueError(f"hopper commitment {name} is invalid")


class HopperTrajectoryBuffer:
    """Private executed-trajectory journal that is invisible before landing."""

    def __init__(self, commitment: HopperObservationCommitment) -> None:
        if not isinstance(commitment, HopperObservationCommitment):
            raise ValueError("hopper trajectory buffer commitment is invalid")
        self._commitment = commitment
        self._points: tuple[HopperTrajectoryPoint, ...] = (
            HopperTrajectoryPoint(commitment.takeoff_pose, 0.0),
        )
        self._sealed = False

    @property
    def commitment(self) -> HopperObservationCommitment:
        return self._commitment

    @property
    def points(self) -> tuple[HopperTrajectoryPoint, ...]:
        return self._points

    @property
    def sealed(self) -> bool:
        return self._sealed

    def append(
        self,
        commitment_id: str,
        points: tuple[HopperTrajectoryPoint, ...],
        *,
        within_certified_flight_tube: bool,
    ) -> None:
        if self._sealed:
            raise ValueError("hopper trajectory buffer is sealed")
        if commitment_id != self._commitment.commitment_id:
            raise ValueError("hopper trajectory commitment identity differs")
        if type(within_certified_flight_tube) is not bool:
            raise ValueError("hopper trajectory flight-tube fact is invalid")
        if not within_certified_flight_tube:
            raise ValueError("hopper trajectory left the certified flight tube")
        if not isinstance(points, tuple) or any(
            not isinstance(point, HopperTrajectoryPoint) for point in points
        ):
            raise ValueError("hopper trajectory points are invalid")
        if not points:
            return
        previous_time = float(self._points[-1].time_s)
        times = tuple(float(point.time_s) for point in points)
        if times[0] < previous_time or any(
            right < left for left, right in zip(times, times[1:])
        ):
            raise ValueError("hopper trajectory time regressed")
        if times[-1] > float(self._commitment.flight_time_s) + 1.0e-9:
            raise ValueError("hopper trajectory time exceeds the commitment")
        self._points = (*self._points, *points)

    def finalize(
        self,
        commitment_id: str,
        *,
        landing_pose: Pose2,
        elapsed_s: float,
    ) -> tuple[HopperTrajectoryPoint, ...]:
        if commitment_id != self._commitment.commitment_id:
            raise ValueError("hopper trajectory commitment identity differs")
        if not isinstance(landing_pose, Pose2) or landing_pose.frame_id != "map":
            raise ValueError("hopper trajectory landing pose is invalid")
        expected = self._commitment.expected_landing_pose
        if math.dist(
            (landing_pose.x_m, landing_pose.y_m, landing_pose.elevation_m),
            (expected.x_m, expected.y_m, expected.elevation_m),
        ) > 1.0e-6:
            raise ValueError("hopper trajectory landing differs from commitment")
        if (
            not isinstance(elapsed_s, (int, float))
            or isinstance(elapsed_s, bool)
            or not math.isfinite(float(elapsed_s))
            or abs(float(elapsed_s) - float(self._commitment.flight_time_s))
            > 1.0e-9
        ):
            raise ValueError("hopper trajectory landing time differs from commitment")
        if self._sealed:
            return self._points
        final = HopperTrajectoryPoint(landing_pose, float(elapsed_s))
        points = self._points
        if points[-1] != final:
            if float(final.time_s) < float(points[-1].time_s):
                raise ValueError("hopper trajectory time regressed")
            points = (*points, final)
        self._points = points
        self._sealed = True
        return self._points


class TerminalReason(str, Enum):
    """One authoritative reason for every completed episode."""

    SUCCESS = "SUCCESS"
    NO_FRONTIER = "NO_FRONTIER"
    NO_GLOBAL_ROUTE = "NO_GLOBAL_ROUTE"
    ZERO_EXPECTED_GAIN = "ZERO_EXPECTED_GAIN"
    PLANNER_EXHAUSTED = "PLANNER_EXHAUSTED"
    NO_AVAILABLE_LANDING_CANDIDATE = "NO_AVAILABLE_LANDING_CANDIDATE"
    TRUNCATED = "TRUNCATED"
    HARD_FAILURE = "HARD_FAILURE"
    CANCELED = "CANCELED"


_VALID_INCOMPLETE_TERMINALS = frozenset(
    {
        TerminalReason.NO_FRONTIER,
        TerminalReason.NO_GLOBAL_ROUTE,
        TerminalReason.ZERO_EXPECTED_GAIN,
        TerminalReason.PLANNER_EXHAUSTED,
        TerminalReason.NO_AVAILABLE_LANDING_CANDIDATE,
        TerminalReason.TRUNCATED,
    }
)


def terminal_class_for_reason(
    reason: TerminalReason | None,
) -> RewardTerminalClass:
    """Map one environment terminal fact to the frozen Reward V4 class."""
    if reason is None:
        return RewardTerminalClass.CONTINUE
    if reason is TerminalReason.SUCCESS:
        return RewardTerminalClass.SUCCESS
    if reason in _VALID_INCOMPLETE_TERMINALS:
        return RewardTerminalClass.VALID_INCOMPLETE_TERMINAL
    if reason in {TerminalReason.HARD_FAILURE, TerminalReason.CANCELED}:
        return RewardTerminalClass.INVALID_TRANSITION
    raise ValueError("terminal reason is invalid")


@dataclass(frozen=True, slots=True)
class TerminalAudit:
    """Out-of-band terminal evidence that never enters policy or reward."""

    reason: TerminalReason
    candidate_diagnostics: CandidateDiagnostics
    remaining_coverable_detail_cell_count: int | None

    def __post_init__(self) -> None:
        if not isinstance(self.reason, TerminalReason):
            raise ValueError("terminal audit reason is invalid")
        if not isinstance(self.candidate_diagnostics, CandidateDiagnostics):
            raise ValueError("terminal candidate diagnostics are invalid")
        remaining = self.remaining_coverable_detail_cell_count
        if remaining is not None and (
            type(remaining) is not int or remaining < 0
        ):
            raise ValueError("terminal remaining-coverable count is invalid")


@dataclass(frozen=True)
class PolicyAction:
    frontier_index: int
    theta_rad: float


@dataclass(frozen=True)
class ExecutionEvents:
    """Observed execution facts carried from a worker into evaluation."""

    safety_violation_count: int = 0
    invalid_action_count: int = 0
    platform_reference_mismatch_count: int = 0
    hopper_commitment_violation_count: int = 0
    execution_failure_count: int = 0
    reference_samples_consumed: int = 0
    selected_action_observed_safe: bool = False
    hopper_commitment_states: tuple[str, ...] = ()


@dataclass(frozen=True)
class PlannerTransition:
    next_observation: PolicyBatch
    mission_observed_delta: float
    priority_observed_delta: float
    normalized_plan_or_execution_cost: float
    normalized_macro_step_time: float
    executed_without_new_coverage: bool
    success_first_crossing: bool
    episode_ended_without_success: bool
    hard_safety_violation: bool
    cancellation_expected: bool
    cpp_exception: str | None
    planning_outcome: PlanningOutcome
    execution_directive: ExecutionDirective
    reason_code: str
    terminated: bool
    execution_events: ExecutionEvents
    terminal_reason: TerminalReason | None = None
    remaining_coverable_detail_cell_count: int | None = None
    reward_terminal_class: RewardTerminalClass = RewardTerminalClass.CONTINUE
    coverage_before: float = 0.0
    coverage_after: float = 0.0
    priority_before: float = 0.0
    priority_after: float = 0.0
    executed_path_length_m: float = 0.0
    cumulative_executed_path_m_before: float = 0.0
    cumulative_executed_path_m_after: float = 0.0
    task_scale_m: float = 1.0

    def __post_init__(self) -> None:
        validate_planner_transition(self)


def validate_planner_transition(
    transition: PlannerTransition,
) -> PlannerTransition:
    """Reject a transition without an environment-owned terminal class."""
    if not isinstance(transition, PlannerTransition):
        raise ValueError("planner transition type is invalid")
    if not isinstance(transition.reward_terminal_class, RewardTerminalClass):
        raise ValueError("planner transition terminal class is invalid")
    for name in (
        "coverage_before",
        "coverage_after",
        "priority_before",
        "priority_after",
    ):
        value = getattr(transition, name)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
        ):
            raise ValueError(f"planner transition {name} is invalid")
    if transition.coverage_after < transition.coverage_before:
        raise ValueError("planner transition coverage regressed")
    if transition.priority_after < transition.priority_before:
        raise ValueError("planner transition priority coverage regressed")
    for name in (
        "executed_path_length_m",
        "cumulative_executed_path_m_before",
        "cumulative_executed_path_m_after",
        "task_scale_m",
    ):
        value = getattr(transition, name)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError(f"planner transition {name} is invalid")
    if transition.task_scale_m <= 0.0:
        raise ValueError("planner transition task scale must be positive")
    if not math.isclose(
        transition.cumulative_executed_path_m_after
        - transition.cumulative_executed_path_m_before,
        transition.executed_path_length_m,
        rel_tol=1.0e-12,
        abs_tol=1.0e-9,
    ):
        raise ValueError("planner transition cumulative path is inconsistent")
    return transition


__all__ = [
    "ExecutionEvents",
    "HopperObservationCommitment",
    "HopperTrajectoryBuffer",
    "PlannerTransition",
    "PolicyAction",
    "TerminalAudit",
    "TerminalReason",
    "terminal_class_for_reason",
    "validate_planner_transition",
]

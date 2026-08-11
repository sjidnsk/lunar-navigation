"""Planner-scale transitions for the shared seven-input PPO policy."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from lunar_planner_training_bridge import ExecutionDirective, PlanningOutcome

from ..policy.observation import PolicyBatch
from .candidate_builder import CandidateDiagnostics


class TerminalReason(str, Enum):
    """One authoritative reason for every completed episode."""

    SUCCESS = "SUCCESS"
    NO_RECOVERABLE_OBSERVATION_STATE = "NO_RECOVERABLE_OBSERVATION_STATE"
    VISITED_EXHAUSTED = "VISITED_EXHAUSTED"
    ZERO_GAIN = "ZERO_GAIN"
    NO_TRANSIT_OPPORTUNITY = "NO_TRANSIT_OPPORTUNITY"
    PLANNER_BLOCKED_WITH_OPPORTUNITY = "PLANNER_BLOCKED_WITH_OPPORTUNITY"
    HARD_FAILURE = "HARD_FAILURE"
    CANCELED = "CANCELED"


@dataclass(frozen=True, slots=True)
class TerminalAudit:
    """Out-of-band terminal evidence that never enters policy or reward."""

    reason: TerminalReason
    oracle_opportunity_count: int
    candidate_diagnostics: CandidateDiagnostics
    remaining_coverable_detail_cell_count: int | None

    def __post_init__(self) -> None:
        if not isinstance(self.reason, TerminalReason):
            raise ValueError("terminal audit reason is invalid")
        if (
            type(self.oracle_opportunity_count) is not int
            or self.oracle_opportunity_count < 0
        ):
            raise ValueError("terminal oracle opportunity count is invalid")
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
    oracle_opportunity_count: int = 0
    remaining_coverable_detail_cell_count: int | None = None


__all__ = [
    "ExecutionEvents",
    "PlannerTransition",
    "PolicyAction",
    "TerminalAudit",
    "TerminalReason",
]

"""Planner-scale transitions for the shared seven-input PPO policy."""

from __future__ import annotations

from dataclasses import dataclass

from lunar_planner_training_bridge import ExecutionDirective, PlanningOutcome

from ..policy.observation import PolicyBatch


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


__all__ = ["ExecutionEvents", "PlannerTransition", "PolicyAction"]

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
    coverage_delta: float
    goal_progress: float
    normalized_plan_cost: float
    normalized_elapsed_time: float
    repeated_visit: bool
    planning_outcome: PlanningOutcome
    execution_directive: ExecutionDirective
    reason_code: str
    terminated: bool
    execution_events: ExecutionEvents = ExecutionEvents()


__all__ = ["ExecutionEvents", "PlannerTransition", "PolicyAction"]

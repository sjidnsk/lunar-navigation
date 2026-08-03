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


__all__ = ["PlannerTransition", "PolicyAction"]

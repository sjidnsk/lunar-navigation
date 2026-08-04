"""Shared dimensionless reward for v3 planner transitions."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass

from lunar_planner_training_bridge import PlanningOutcome

from .environment.macro_step import PlannerTransition


class InvalidTransition(RuntimeError):
    """A planner/infrastructure transition must not enter a rollout."""


@dataclass(frozen=True, slots=True)
class RewardWeights:
    coverage_gain: float = 10.0
    goal_progress: float = 2.0
    normalized_plan_cost: float = 1.0
    normalized_elapsed_time: float = 0.5
    repeated_visit: float = 1.0
    goal_infeasible: float = 2.0
    no_known_safe_route: float = 3.0
    resource_exhausted: float = 1.5


DEFAULT_REWARD_WEIGHTS = RewardWeights()


@dataclass(frozen=True, slots=True)
class RewardInputs:
    platform_type: str
    coverage_gain: float
    goal_progress: float
    normalized_plan_cost: float
    normalized_elapsed_time: float
    repeated_visit: bool
    planning_outcome: PlanningOutcome
    cancellation_expected: bool = False
    cpp_exception: str | None = None

    @classmethod
    def from_transition(
        cls, transition: PlannerTransition, *, platform_type: str
    ) -> "RewardInputs":
        if not isinstance(transition, PlannerTransition):
            raise InvalidTransition("reward input is not a PlannerTransition")
        return cls(
            platform_type=platform_type,
            coverage_gain=transition.coverage_delta,
            goal_progress=transition.goal_progress,
            normalized_plan_cost=transition.normalized_plan_cost,
            normalized_elapsed_time=transition.normalized_elapsed_time,
            repeated_visit=transition.repeated_visit,
            planning_outcome=transition.planning_outcome,
        )


_PLATFORMS = frozenset({"WHEELED", "LEGGED", "HOPPER"})
_INVALID_OUTCOMES = frozenset(
    {
        PlanningOutcome.INVALID_REQUEST,
        PlanningOutcome.STALE_INPUT,
        PlanningOutcome.NUMERICAL_FAILURE,
    }
)


def reward_weights_sha256(
    weights: RewardWeights = DEFAULT_REWARD_WEIGHTS,
    *,
    platform_type: str | None = None,
) -> str:
    """Return one platform-independent identity for the shared weight set."""
    if platform_type is not None and platform_type not in _PLATFORMS:
        raise ValueError("platform type must be WHEELED, LEGGED, or HOPPER")
    payload = json.dumps(
        asdict(weights), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def compute_reward(
    inputs: RewardInputs,
    weights: RewardWeights = DEFAULT_REWARD_WEIGHTS,
) -> float:
    """Compute one shared reward or reject an invalid infrastructure sample."""
    if not isinstance(inputs, RewardInputs):
        raise InvalidTransition("reward inputs must use RewardInputs")
    if inputs.platform_type not in _PLATFORMS:
        raise InvalidTransition("reward platform type is invalid")
    if not isinstance(weights, RewardWeights):
        raise InvalidTransition("reward weights must use RewardWeights")
    if inputs.cpp_exception is not None:
        raise InvalidTransition("C++ exception transition cannot enter rollout")
    values = (
        inputs.coverage_gain,
        inputs.goal_progress,
        inputs.normalized_plan_cost,
        inputs.normalized_elapsed_time,
    )
    if not all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        for value in values
    ):
        raise InvalidTransition("reward input contains a non-finite scalar")
    if inputs.normalized_plan_cost < 0.0 or inputs.normalized_elapsed_time < 0.0:
        raise InvalidTransition("normalized reward costs cannot be negative")
    if not isinstance(inputs.repeated_visit, bool):
        raise InvalidTransition("repeated_visit must be boolean")
    if not isinstance(inputs.planning_outcome, PlanningOutcome):
        raise InvalidTransition("planning outcome is invalid")
    if inputs.planning_outcome in _INVALID_OUTCOMES:
        raise InvalidTransition(
            f"{inputs.planning_outcome.name.lower()} transition cannot enter rollout"
        )
    if (
        inputs.planning_outcome == PlanningOutcome.CANCELED
        and not inputs.cancellation_expected
    ):
        raise InvalidTransition("unexpected canceled transition cannot enter rollout")

    reward = (
        max(0.0, float(inputs.coverage_gain)) * weights.coverage_gain
        + max(0.0, float(inputs.goal_progress)) * weights.goal_progress
        - float(inputs.normalized_plan_cost) * weights.normalized_plan_cost
        - float(inputs.normalized_elapsed_time) * weights.normalized_elapsed_time
    )
    if inputs.repeated_visit:
        reward -= weights.repeated_visit
    if inputs.planning_outcome == PlanningOutcome.GOAL_INFEASIBLE:
        reward -= weights.goal_infeasible
    elif inputs.planning_outcome in {
        PlanningOutcome.NO_KNOWN_SAFE_ROUTE,
        PlanningOutcome.ACTIVE_REFERENCE_INVALIDATED,
    }:
        reward -= weights.no_known_safe_route
    elif inputs.planning_outcome == PlanningOutcome.RESOURCE_EXHAUSTED:
        reward -= weights.resource_exhausted
    if not math.isfinite(reward):
        raise InvalidTransition("computed reward is non-finite")
    return reward


def compute_transition_reward(transition: PlannerTransition) -> float:
    """Pool-safe shared reward adapter deriving the episode platform one-hot."""
    if not isinstance(transition, PlannerTransition):
        raise InvalidTransition("reward input is not a PlannerTransition")
    context = transition.next_observation.platform_context
    if context.shape != (1, 3):
        raise InvalidTransition("transition platform context shape is invalid")
    values = context.detach().cpu().tolist()[0]
    try:
        platform_type = ("WHEELED", "LEGGED", "HOPPER")[values.index(1.0)]
    except (ValueError, IndexError) as error:
        raise InvalidTransition("transition platform context is invalid") from error
    if values.count(1.0) != 1 or any(value not in (0.0, 1.0) for value in values):
        raise InvalidTransition("transition platform context is invalid")
    return compute_reward(
        RewardInputs.from_transition(transition, platform_type=platform_type)
    )


__all__ = [
    "DEFAULT_REWARD_WEIGHTS",
    "InvalidTransition",
    "RewardInputs",
    "RewardWeights",
    "compute_reward",
    "compute_transition_reward",
    "reward_weights_sha256",
]

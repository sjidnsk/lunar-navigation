"""Frozen coverage-first V3 reward for v3 planner transitions."""

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
class RewardWeightsV3:
    mission_observed_delta: float = 100.0
    executed_without_new_mission_coverage: float = 0.10
    goal_infeasible: float = 0.20
    no_known_safe_route: float = 0.30
    unsuccessful_remaining_coverage: float = 1.00
    success_first_crossing: float = 5.00


REWARD_SCHEMA_VERSION = "lunar-reward/v3"
DEFAULT_REWARD_WEIGHTS = RewardWeightsV3()


@dataclass(frozen=True, slots=True)
class RewardInputsV3:
    platform_type: str
    mission_observed_delta: float
    mission_observed_ratio_after: float
    priority_observed_delta: float
    normalized_plan_or_execution_cost: float
    normalized_macro_step_time: float
    executed_without_new_coverage: bool
    success_first_crossing: bool
    episode_ended_without_success: bool
    hard_safety_violation: bool
    safety_violation_count: int
    platform_reference_mismatch_count: int
    hopper_commitment_violation_count: int
    terminated: bool
    planning_outcome: PlanningOutcome
    cancellation_expected: bool
    cpp_exception: str | None

    @classmethod
    def from_transition(
        cls, transition: PlannerTransition, *, platform_type: str
    ) -> "RewardInputsV3":
        if not isinstance(transition, PlannerTransition):
            raise InvalidTransition("reward input is not a PlannerTransition")
        return cls(
            platform_type=platform_type,
            mission_observed_delta=transition.mission_observed_delta,
            mission_observed_ratio_after=float(
                transition.next_observation.pose_features[0, 4]
                .detach()
                .cpu()
                .item()
            ),
            priority_observed_delta=transition.priority_observed_delta,
            normalized_plan_or_execution_cost=(
                transition.normalized_plan_or_execution_cost
            ),
            normalized_macro_step_time=transition.normalized_macro_step_time,
            executed_without_new_coverage=(
                transition.executed_without_new_coverage
            ),
            success_first_crossing=transition.success_first_crossing,
            episode_ended_without_success=(
                transition.episode_ended_without_success
            ),
            hard_safety_violation=transition.hard_safety_violation,
            safety_violation_count=(
                transition.execution_events.safety_violation_count
            ),
            platform_reference_mismatch_count=(
                transition.execution_events.platform_reference_mismatch_count
            ),
            hopper_commitment_violation_count=(
                transition.execution_events.hopper_commitment_violation_count
            ),
            terminated=transition.terminated,
            planning_outcome=transition.planning_outcome,
            cancellation_expected=transition.cancellation_expected,
            cpp_exception=transition.cpp_exception,
        )


_PLATFORMS = frozenset({"WHEELED", "LEGGED", "HOPPER"})
_INVALID_OUTCOMES = frozenset(
    {
        PlanningOutcome.INVALID_REQUEST,
        PlanningOutcome.STALE_INPUT,
        PlanningOutcome.NUMERICAL_FAILURE,
        PlanningOutcome.RESOURCE_EXHAUSTED,
    }
)
_BOOLEAN_FIELDS = (
    "executed_without_new_coverage",
    "success_first_crossing",
    "episode_ended_without_success",
    "hard_safety_violation",
    "terminated",
    "cancellation_expected",
)


def reward_weights_sha256(
    weights: RewardWeightsV3 = DEFAULT_REWARD_WEIGHTS,
    *,
    platform_type: str | None = None,
) -> str:
    """Return one platform-independent identity for the complete V3 reward."""
    if not isinstance(weights, RewardWeightsV3):
        raise ValueError("reward weights must use RewardWeightsV3")
    if platform_type is not None and platform_type not in _PLATFORMS:
        raise ValueError("platform type must be WHEELED, LEGGED, or HOPPER")
    values = tuple(asdict(weights).values())
    if any(
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) < 0.0
        for value in values
    ):
        raise ValueError("reward weights must be finite and non-negative")
    payload = json.dumps(
        {
            "schema_version": REWARD_SCHEMA_VERSION,
            "weights": asdict(weights),
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def compute_reward(
    inputs: RewardInputsV3,
    weights: RewardWeightsV3 = DEFAULT_REWARD_WEIGHTS,
) -> float:
    """Compute the frozen shared V3 reward or reject an invalid sample."""
    if not isinstance(inputs, RewardInputsV3):
        raise InvalidTransition("reward inputs must use RewardInputsV3")
    if inputs.platform_type not in _PLATFORMS:
        raise InvalidTransition("reward platform type is invalid")
    if not isinstance(weights, RewardWeightsV3) or weights != DEFAULT_REWARD_WEIGHTS:
        raise InvalidTransition("reward weights differ from the frozen V3 baseline")
    if inputs.cpp_exception is not None:
        raise InvalidTransition("C++ exception transition cannot enter rollout")
    if any(type(getattr(inputs, field)) is not bool for field in _BOOLEAN_FIELDS):
        raise InvalidTransition("reward flags must use exact boolean values")

    values = (
        inputs.mission_observed_delta,
        inputs.mission_observed_ratio_after,
        inputs.priority_observed_delta,
        inputs.normalized_plan_or_execution_cost,
        inputs.normalized_macro_step_time,
    )
    if not all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        for value in values
    ):
        raise InvalidTransition("reward input contains a non-finite scalar")
    if not 0.0 <= float(inputs.mission_observed_delta) <= 1.0:
        raise InvalidTransition("mission observed delta is outside [0,1] range")
    if not 0.0 <= float(inputs.mission_observed_ratio_after) <= 1.0:
        raise InvalidTransition("mission observed ratio is outside [0,1] range")
    if not 0.0 <= float(inputs.priority_observed_delta) <= 1.0:
        raise InvalidTransition("priority observed delta is outside [0,1] range")
    if (
        inputs.normalized_plan_or_execution_cost < 0.0
        or inputs.normalized_macro_step_time < 0.0
    ):
        raise InvalidTransition("normalized reward costs must be non-negative")
    hard_safety_event_counts = (
        inputs.safety_violation_count,
        inputs.platform_reference_mismatch_count,
        inputs.hopper_commitment_violation_count,
    )
    if any(type(value) is not int or value < 0 for value in hard_safety_event_counts):
        raise InvalidTransition("hard safety gate event count is invalid")
    if inputs.hard_safety_violation != any(
        value > 0 for value in hard_safety_event_counts
    ):
        raise InvalidTransition("hard safety flag and safety gate event disagree")
    if inputs.success_first_crossing and inputs.hard_safety_violation:
        raise InvalidTransition("success and hard safety cannot coexist")
    if inputs.success_first_crossing and inputs.episode_ended_without_success:
        raise InvalidTransition("success and unsuccessful terminal cannot coexist")
    terminal_fact = (
        inputs.success_first_crossing or inputs.episode_ended_without_success
    )
    if inputs.terminated != terminal_fact:
        raise InvalidTransition("terminal state and reward terminal facts disagree")
    if inputs.hard_safety_violation and not inputs.episode_ended_without_success:
        raise InvalidTransition("hard safety requires an unsuccessful terminal")
    if inputs.hard_safety_violation:
        raise InvalidTransition("hard safety transition cannot enter rollout")
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
    if inputs.planning_outcome == PlanningOutcome.CANCELED:
        return 0.0

    reward = float(inputs.mission_observed_delta) * weights.mission_observed_delta
    if inputs.mission_observed_delta == 0.0:
        reward -= weights.executed_without_new_mission_coverage
    if inputs.planning_outcome == PlanningOutcome.GOAL_INFEASIBLE:
        reward -= weights.goal_infeasible
    elif inputs.planning_outcome in {
        PlanningOutcome.NO_KNOWN_SAFE_ROUTE,
        PlanningOutcome.ACTIVE_REFERENCE_INVALIDATED,
    }:
        reward -= weights.no_known_safe_route
    if inputs.success_first_crossing:
        reward += weights.success_first_crossing
    if inputs.episode_ended_without_success:
        reward -= weights.unsuccessful_remaining_coverage * (
            1.0 - float(inputs.mission_observed_ratio_after)
        )
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
        RewardInputsV3.from_transition(transition, platform_type=platform_type)
    )


__all__ = [
    "DEFAULT_REWARD_WEIGHTS",
    "InvalidTransition",
    "REWARD_SCHEMA_VERSION",
    "RewardInputsV3",
    "RewardWeightsV3",
    "compute_reward",
    "compute_transition_reward",
    "reward_weights_sha256",
]

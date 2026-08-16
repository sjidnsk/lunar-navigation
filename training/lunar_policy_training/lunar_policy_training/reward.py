"""Pure, auditable Reward V4 computation for formal macro transitions."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

from .environment.macro_step import PlannerTransition
from .reward_contract import (
    DEFAULT_REWARD_CONFIG,
    REWARD_SCHEMA_VERSION,
    RewardStage,
    RewardTerminalClass,
    RewardWeightsV4,
    reward_config_sha256,
)


class InvalidTransition(RuntimeError):
    """An invalid environment transition must not enter a rollout."""


DEFAULT_REWARD_WEIGHTS = DEFAULT_REWARD_CONFIG.initial_weights


@dataclass(frozen=True, slots=True)
class RewardInputsV4:
    """Environment-owned cumulative facts consumed by Reward V4."""

    platform_type: str
    coverage_before: float
    coverage_after: float
    priority_before: float
    priority_after: float
    path_before_m: float
    path_after_m: float
    task_scale_m: float
    terminal_class: RewardTerminalClass
    success_first_crossing: bool

    @classmethod
    def from_transition(
        cls, transition: PlannerTransition, *, platform_type: str
    ) -> "RewardInputsV4":
        if not isinstance(transition, PlannerTransition):
            raise InvalidTransition("reward input is not a PlannerTransition")
        return cls(
            platform_type=platform_type,
            coverage_before=transition.coverage_before,
            coverage_after=transition.coverage_after,
            priority_before=transition.priority_before,
            priority_after=transition.priority_after,
            path_before_m=transition.cumulative_executed_path_m_before,
            path_after_m=transition.cumulative_executed_path_m_after,
            task_scale_m=transition.task_scale_m,
            terminal_class=transition.reward_terminal_class,
            success_first_crossing=transition.success_first_crossing,
        )


@dataclass(frozen=True, slots=True)
class RewardComponentsV4:
    """Named Reward V4 terms retained for diagnostics and exact tests."""

    coverage: float
    success: float
    terminal_gap: float
    priority: float
    path: float
    total: float


_PLATFORMS = frozenset({"WHEELED", "LEGGED", "HOPPER"})
_RATIO_FIELDS = (
    "coverage_before",
    "coverage_after",
    "priority_before",
    "priority_after",
)
_PATH_FIELDS = ("path_before_m", "path_after_m")


def reward_weights_sha256(
    weights: RewardWeightsV4 = DEFAULT_REWARD_WEIGHTS,
    *,
    platform_type: str | None = None,
) -> str:
    """Hash the complete Reward V4 config, independent of platform."""
    if not isinstance(weights, RewardWeightsV4):
        raise ValueError("reward weights must use RewardWeightsV4")
    if platform_type is not None and platform_type not in _PLATFORMS:
        raise ValueError("platform type must be WHEELED, LEGGED, or HOPPER")
    config = replace(
        DEFAULT_REWARD_CONFIG,
        coverage_weight=weights.coverage,
        success_bonus=weights.success,
        terminal_gap_weight=weights.terminal_gap,
        priority_weight=weights.priority,
        path_weight=weights.path,
    )
    return reward_config_sha256(config)


def _finite_scalar(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _validate_inputs(inputs: RewardInputsV4) -> None:
    if not isinstance(inputs, RewardInputsV4):
        raise InvalidTransition("reward inputs must use RewardInputsV4")
    if inputs.platform_type not in _PLATFORMS:
        raise InvalidTransition("reward platform type is invalid")
    if not isinstance(inputs.terminal_class, RewardTerminalClass):
        raise InvalidTransition("reward terminal class is invalid")
    if type(inputs.success_first_crossing) is not bool:
        raise InvalidTransition("reward success fact must use an exact boolean")

    for name in _RATIO_FIELDS:
        value = getattr(inputs, name)
        if not _finite_scalar(value) or not 0.0 <= float(value) <= 1.0:
            raise InvalidTransition(f"reward {name} is outside [0,1]")
    for name in _PATH_FIELDS:
        value = getattr(inputs, name)
        if not _finite_scalar(value) or float(value) < 0.0:
            raise InvalidTransition(f"reward {name} must be finite and non-negative")
    if not _finite_scalar(inputs.task_scale_m) or inputs.task_scale_m <= 0.0:
        raise InvalidTransition("reward task scale must be finite and positive")

    if inputs.coverage_after < inputs.coverage_before:
        raise InvalidTransition("reward coverage facts regress")
    if inputs.priority_after < inputs.priority_before:
        raise InvalidTransition("reward priority facts regress")
    if inputs.path_after_m < inputs.path_before_m:
        raise InvalidTransition("reward path facts regress")

    if inputs.terminal_class is RewardTerminalClass.INVALID_TRANSITION:
        raise InvalidTransition("invalid transition cannot enter reward computation")

    threshold = DEFAULT_REWARD_CONFIG.success_threshold
    actual_first_crossing = (
        inputs.coverage_before < threshold <= inputs.coverage_after
    )
    declared_success = inputs.terminal_class is RewardTerminalClass.SUCCESS
    if (
        inputs.success_first_crossing != actual_first_crossing
        or declared_success != actual_first_crossing
    ):
        raise InvalidTransition("success must be one real first threshold crossing")
    if not declared_success and inputs.coverage_after >= threshold:
        raise InvalidTransition("non-success transition reached success threshold")


def compute_reward_components(
    inputs: RewardInputsV4,
    stage: RewardStage,
    weights: RewardWeightsV4 = DEFAULT_REWARD_WEIGHTS,
) -> RewardComponentsV4:
    """Compute the five Reward V4 terms from one authoritative boundary."""
    _validate_inputs(inputs)
    if not isinstance(stage, RewardStage):
        raise InvalidTransition("reward stage is invalid")
    if not isinstance(weights, RewardWeightsV4):
        raise InvalidTransition("reward weights must use RewardWeightsV4")

    coverage = math.fsum(
        (
            weights.coverage * float(inputs.coverage_after),
            -weights.coverage * float(inputs.coverage_before),
        )
    )
    success = (
        weights.success
        if inputs.terminal_class is RewardTerminalClass.SUCCESS
        else 0.0
    )
    terminal_gap = (
        -weights.terminal_gap * (1.0 - float(inputs.coverage_after))
        if inputs.terminal_class
        is RewardTerminalClass.VALID_INCOMPLETE_TERMINAL
        else 0.0
    )

    priority = 0.0
    path = 0.0
    if stage is RewardStage.R2:
        priority = math.fsum(
            (
                weights.priority * float(inputs.priority_after),
                -weights.priority * float(inputs.priority_before),
            )
        )
        scale = float(inputs.task_scale_m)
        normalized_before = 1.0 - math.exp(-float(inputs.path_before_m) / scale)
        normalized_after = 1.0 - math.exp(-float(inputs.path_after_m) / scale)
        path = -weights.path * (normalized_after - normalized_before)

    total = math.fsum((coverage, success, terminal_gap, priority, path))
    if not all(
        math.isfinite(value)
        for value in (coverage, success, terminal_gap, priority, path, total)
    ):
        raise InvalidTransition("computed Reward V4 component is non-finite")
    return RewardComponentsV4(
        coverage=coverage,
        success=success,
        terminal_gap=terminal_gap,
        priority=priority,
        path=path,
        total=total,
    )


def compute_reward(
    inputs: RewardInputsV4,
    weights: RewardWeightsV4 = DEFAULT_REWARD_WEIGHTS,
    *,
    stage: RewardStage = RewardStage.R1,
) -> float:
    """Return the scalar Reward V4 total for compatibility callers."""
    return compute_reward_components(inputs, stage, weights).total


def compute_transition_reward(
    transition: PlannerTransition,
    *,
    stage: RewardStage = RewardStage.R1,
    weights: RewardWeightsV4 = DEFAULT_REWARD_WEIGHTS,
) -> float:
    """Pool-safe adapter deriving the platform from the frozen one-hot."""
    if not isinstance(transition, PlannerTransition):
        raise InvalidTransition("reward input is not a PlannerTransition")
    context = transition.next_observation.platform_context
    if context.shape != (1, 3):
        raise InvalidTransition("transition platform context shape is invalid")
    values = context.detach().cpu().tolist()[0]
    if values.count(1.0) != 1 or any(value not in (0.0, 1.0) for value in values):
        raise InvalidTransition("transition platform context is invalid")
    platform_type = ("WHEELED", "LEGGED", "HOPPER")[values.index(1.0)]
    inputs = RewardInputsV4.from_transition(
        transition, platform_type=platform_type
    )
    return compute_reward(inputs, weights, stage=stage)


# Source compatibility only; hashes and runtime validation are V4.
RewardInputsV3 = RewardInputsV4
RewardWeightsV3 = RewardWeightsV4


__all__ = [
    "DEFAULT_REWARD_WEIGHTS",
    "InvalidTransition",
    "REWARD_SCHEMA_VERSION",
    "RewardComponentsV4",
    "RewardInputsV3",
    "RewardInputsV4",
    "RewardWeightsV3",
    "RewardWeightsV4",
    "compute_reward",
    "compute_reward_components",
    "compute_transition_reward",
    "reward_weights_sha256",
]

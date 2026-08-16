"""Capability-driven, platform-local Reward V4 curriculum state machine."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import math
from types import MappingProxyType
from typing import Mapping, Sequence

from .reward_contract import (
    DEFAULT_REWARD_CONFIG,
    RewardConfigV4,
    RewardStage,
    RewardWeightsV4,
)


REWARD_CURRICULUM_SCHEMA_VERSION = "lunar-reward-curriculum-state/v1"


class PlatformType(str, Enum):
    WHEELED = "WHEELED"
    LEGGED = "LEGGED"
    HOPPER = "HOPPER"


class TrainingStage(str, Enum):
    GROUND_R1 = "GROUND_R1"
    GROUND_R2_HOPPER_R1 = "GROUND_R2_HOPPER_R1"
    THREE_PLATFORM_R2 = "THREE_PLATFORM_R2"


_PLATFORM_ORDER = tuple(PlatformType)
_STAGE_ORDER = tuple(TrainingStage)


@dataclass(frozen=True, slots=True)
class PlatformGateMetrics:
    success_rate: float
    mean_final_coverage: float
    hard_error_count: int

    def __post_init__(self) -> None:
        for name in ("success_rate", "mean_final_coverage"):
            value = getattr(self, name)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise ValueError(f"curriculum {name} is invalid")
            object.__setattr__(self, name, float(value))
        if type(self.hard_error_count) is not int or self.hard_error_count < 0:
            raise ValueError("curriculum hard error count is invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            "success_rate": self.success_rate,
            "mean_final_coverage": self.mean_final_coverage,
            "hard_error_count": self.hard_error_count,
        }

    @classmethod
    def from_mapping(cls, value: object) -> "PlatformGateMetrics":
        if not isinstance(value, Mapping) or set(value) != {
            "success_rate",
            "mean_final_coverage",
            "hard_error_count",
        }:
            raise ValueError("curriculum gate metrics structure is invalid")
        return cls(
            success_rate=value["success_rate"],
            mean_final_coverage=value["mean_final_coverage"],
            hard_error_count=value["hard_error_count"],
        )


@dataclass(frozen=True, slots=True)
class PlatformRewardState:
    stage: RewardStage
    weights: RewardWeightsV4
    next_r2_weights: RewardWeightsV4
    consecutive_passes: int
    consecutive_relative_failures: int
    rollback_count: int
    efficiency_locked_off: bool
    frozen_r1_baseline: PlatformGateMetrics | None

    def __post_init__(self) -> None:
        if not isinstance(self.stage, RewardStage):
            raise ValueError("platform reward stage is invalid")
        if not isinstance(self.weights, RewardWeightsV4) or not isinstance(
            self.next_r2_weights, RewardWeightsV4
        ):
            raise ValueError("platform reward weights are invalid")
        for name in (
            "consecutive_passes",
            "consecutive_relative_failures",
            "rollback_count",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"platform reward {name} is invalid")
        if type(self.efficiency_locked_off) is not bool:
            raise ValueError("platform reward efficiency lock is invalid")
        if self.efficiency_locked_off and self.stage is not RewardStage.R1:
            raise ValueError("locked platform must remain in Reward V4 R1")
        if self.frozen_r1_baseline is not None and not isinstance(
            self.frozen_r1_baseline, PlatformGateMetrics
        ):
            raise ValueError("platform R1 baseline is invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            "stage": self.stage.value,
            "weights": _weights_to_mapping(self.weights),
            "next_r2_weights": _weights_to_mapping(self.next_r2_weights),
            "consecutive_passes": self.consecutive_passes,
            "consecutive_relative_failures": (
                self.consecutive_relative_failures
            ),
            "rollback_count": self.rollback_count,
            "efficiency_locked_off": self.efficiency_locked_off,
            "frozen_r1_baseline": (
                None
                if self.frozen_r1_baseline is None
                else self.frozen_r1_baseline.to_dict()
            ),
        }

    @classmethod
    def from_mapping(cls, value: object) -> "PlatformRewardState":
        if not isinstance(value, Mapping) or set(value) != {
            "stage",
            "weights",
            "next_r2_weights",
            "consecutive_passes",
            "consecutive_relative_failures",
            "rollback_count",
            "efficiency_locked_off",
            "frozen_r1_baseline",
        }:
            raise ValueError("platform reward state structure is invalid")
        try:
            stage = RewardStage(value["stage"])
        except (TypeError, ValueError) as error:
            raise ValueError("platform reward stage is invalid") from error
        baseline = value["frozen_r1_baseline"]
        return cls(
            stage=stage,
            weights=_weights_from_mapping(value["weights"]),
            next_r2_weights=_weights_from_mapping(value["next_r2_weights"]),
            consecutive_passes=value["consecutive_passes"],
            consecutive_relative_failures=value[
                "consecutive_relative_failures"
            ],
            rollback_count=value["rollback_count"],
            efficiency_locked_off=value["efficiency_locked_off"],
            frozen_r1_baseline=(
                None
                if baseline is None
                else PlatformGateMetrics.from_mapping(baseline)
            ),
        )


@dataclass(frozen=True, slots=True)
class RewardCurriculumState:
    stage: TrainingStage
    platforms: Mapping[PlatformType, PlatformRewardState]
    current_update_id: int
    recovery_generation: int
    restored_from_checkpoint_sha256: str | None
    pending_rollback_platforms: tuple[PlatformType, ...]
    worker_restart_required: bool

    def __post_init__(self) -> None:
        if not isinstance(self.stage, TrainingStage):
            raise ValueError("Reward V4 training stage is invalid")
        if not isinstance(self.platforms, Mapping) or set(self.platforms) != set(
            _PLATFORM_ORDER
        ):
            raise ValueError("Reward V4 platform states are invalid")
        frozen_platforms = {
            platform: self.platforms[platform] for platform in _PLATFORM_ORDER
        }
        if any(
            not isinstance(value, PlatformRewardState)
            for value in frozen_platforms.values()
        ):
            raise ValueError("Reward V4 platform state is invalid")
        object.__setattr__(
            self, "platforms", MappingProxyType(frozen_platforms)
        )
        if type(self.current_update_id) is not int or self.current_update_id < 0:
            raise ValueError("Reward V4 current update ID is invalid")
        if type(self.recovery_generation) is not int or self.recovery_generation < 0:
            raise ValueError("Reward V4 recovery generation is invalid")
        if self.restored_from_checkpoint_sha256 is not None and not _is_sha256(
            self.restored_from_checkpoint_sha256
        ):
            raise ValueError("Reward V4 restored checkpoint identity is invalid")
        if (
            not isinstance(self.pending_rollback_platforms, tuple)
            or len(set(self.pending_rollback_platforms))
            != len(self.pending_rollback_platforms)
            or any(
                not isinstance(platform, PlatformType)
                for platform in self.pending_rollback_platforms
            )
            or self.pending_rollback_platforms
            != _ordered_platforms(self.pending_rollback_platforms)
        ):
            raise ValueError("Reward V4 rollback trigger order is invalid")
        if type(self.worker_restart_required) is not bool:
            raise ValueError("Reward V4 worker restart flag is invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": REWARD_CURRICULUM_SCHEMA_VERSION,
            "stage": self.stage.value,
            "platforms": {
                platform.value: self.platforms[platform].to_dict()
                for platform in _PLATFORM_ORDER
            },
            "current_update_id": self.current_update_id,
            "recovery_generation": self.recovery_generation,
            "restored_from_checkpoint_sha256": (
                self.restored_from_checkpoint_sha256
            ),
            "pending_rollback_platforms": [
                platform.value for platform in self.pending_rollback_platforms
            ],
            "worker_restart_required": self.worker_restart_required,
        }

    @classmethod
    def from_mapping(cls, value: object) -> "RewardCurriculumState":
        if not isinstance(value, Mapping) or set(value) != {
            "schema_version",
            "stage",
            "platforms",
            "current_update_id",
            "recovery_generation",
            "restored_from_checkpoint_sha256",
            "pending_rollback_platforms",
            "worker_restart_required",
        }:
            raise ValueError("Reward V4 curriculum state structure is invalid")
        if value["schema_version"] != REWARD_CURRICULUM_SCHEMA_VERSION:
            raise ValueError("Reward V4 curriculum state schema is invalid")
        try:
            stage = TrainingStage(value["stage"])
        except (TypeError, ValueError) as error:
            raise ValueError("Reward V4 training stage is invalid") from error
        platform_values = value["platforms"]
        pending_values = value["pending_rollback_platforms"]
        if not isinstance(platform_values, Mapping) or not isinstance(
            pending_values, list
        ):
            raise ValueError("Reward V4 curriculum arrays are invalid")
        try:
            platforms = {
                PlatformType(name): PlatformRewardState.from_mapping(state)
                for name, state in platform_values.items()
            }
            pending = tuple(PlatformType(name) for name in pending_values)
        except (TypeError, ValueError) as error:
            raise ValueError("Reward V4 curriculum platform is invalid") from error
        return cls(
            stage=stage,
            platforms=platforms,
            current_update_id=value["current_update_id"],
            recovery_generation=value["recovery_generation"],
            restored_from_checkpoint_sha256=value[
                "restored_from_checkpoint_sha256"
            ],
            pending_rollback_platforms=pending,
            worker_restart_required=value["worker_restart_required"],
        )


@dataclass(frozen=True, slots=True)
class AcceptedRewardCheckpoint:
    update_id: int
    payload_sha256: str
    curriculum_state: RewardCurriculumState

    def __post_init__(self) -> None:
        if type(self.update_id) is not int or self.update_id < 0:
            raise ValueError("accepted Reward V4 checkpoint update is invalid")
        if not _is_sha256(self.payload_sha256):
            raise ValueError("accepted Reward V4 checkpoint hash is invalid")
        if not isinstance(self.curriculum_state, RewardCurriculumState):
            raise ValueError("accepted Reward V4 curriculum state is invalid")
        if self.curriculum_state.current_update_id != self.update_id:
            raise ValueError("accepted Reward V4 checkpoint update differs")


def initial_reward_curriculum_state(
    config: RewardConfigV4 = DEFAULT_REWARD_CONFIG,
) -> RewardCurriculumState:
    if not isinstance(config, RewardConfigV4):
        raise TypeError("Reward V4 curriculum config is invalid")
    r1 = _r1_weights(config)
    r2 = config.initial_weights
    platforms = {
        platform: PlatformRewardState(
            stage=RewardStage.R1,
            weights=r1,
            next_r2_weights=r2,
            consecutive_passes=0,
            consecutive_relative_failures=0,
            rollback_count=0,
            efficiency_locked_off=False,
            frozen_r1_baseline=None,
        )
        for platform in _PLATFORM_ORDER
    }
    return RewardCurriculumState(
        stage=TrainingStage.GROUND_R1,
        platforms=platforms,
        current_update_id=0,
        recovery_generation=0,
        restored_from_checkpoint_sha256=None,
        pending_rollback_platforms=(),
        worker_restart_required=False,
    )


def apply_evaluation(
    state: RewardCurriculumState,
    *,
    metrics_by_platform: Mapping[PlatformType, PlatformGateMetrics],
    update_id: int,
    config: RewardConfigV4 = DEFAULT_REWARD_CONFIG,
) -> RewardCurriculumState:
    """Apply one fixed-task point estimate at a complete checkpoint boundary."""
    _require_state_and_config(state, config)
    if type(update_id) is not int or update_id <= state.current_update_id:
        raise ValueError("Reward V4 evaluation update must be monotonic")
    if state.pending_rollback_platforms:
        raise ValueError("Reward V4 rollback must be applied before evaluation")
    active = (
        {PlatformType.WHEELED, PlatformType.LEGGED}
        if state.stage is TrainingStage.GROUND_R1
        else set(_PLATFORM_ORDER)
    )
    if (
        not isinstance(metrics_by_platform, Mapping)
        or not metrics_by_platform
        or not set(metrics_by_platform) <= active
        or any(
            not isinstance(platform, PlatformType)
            or not isinstance(metrics, PlatformGateMetrics)
            for platform, metrics in metrics_by_platform.items()
        )
    ):
        raise ValueError("Reward V4 evaluation platform metrics are invalid")

    platforms = dict(state.platforms)
    triggers: list[PlatformType] = []
    for platform in _PLATFORM_ORDER:
        metrics = metrics_by_platform.get(platform)
        if metrics is None:
            continue
        current = platforms[platform]
        if current.stage is RewardStage.R1:
            platforms[platform] = _apply_r1_evaluation(
                current, metrics=metrics, config=config
            )
            continue
        updated, trigger = _apply_r2_evaluation(
            current, metrics=metrics, config=config
        )
        platforms[platform] = updated
        if trigger:
            triggers.append(platform)

    stage = state.stage
    restart_required = state.worker_restart_required
    if (
        stage is TrainingStage.GROUND_R1
        and platforms[PlatformType.WHEELED].stage is RewardStage.R2
        and platforms[PlatformType.LEGGED].stage is RewardStage.R2
    ):
        stage = TrainingStage.GROUND_R2_HOPPER_R1
        restart_required = True
    if (
        stage is TrainingStage.GROUND_R2_HOPPER_R1
        and platforms[PlatformType.HOPPER].stage is RewardStage.R2
    ):
        stage = TrainingStage.THREE_PLATFORM_R2
        restart_required = True
    return RewardCurriculumState(
        stage=stage,
        platforms=platforms,
        current_update_id=update_id,
        recovery_generation=state.recovery_generation,
        restored_from_checkpoint_sha256=(
            state.restored_from_checkpoint_sha256
        ),
        pending_rollback_platforms=_ordered_platforms(triggers),
        worker_restart_required=restart_required,
    )


def apply_rollback(
    state: RewardCurriculumState,
    *,
    triggers: Sequence[PlatformType],
    checkpoint: AcceptedRewardCheckpoint,
    current_update_id: int,
    config: RewardConfigV4 = DEFAULT_REWARD_CONFIG,
) -> RewardCurriculumState:
    """Restore common accepted state and reduce only triggering platforms."""
    _require_state_and_config(state, config)
    if not isinstance(checkpoint, AcceptedRewardCheckpoint):
        raise TypeError("Reward V4 rollback checkpoint is invalid")
    if (
        type(current_update_id) is not int
        or current_update_id < state.current_update_id
        or checkpoint.update_id > current_update_id
    ):
        raise ValueError("Reward V4 rollback update must remain monotonic")
    ordered = _ordered_platforms(triggers)
    if not ordered or len(ordered) != len(tuple(triggers)):
        raise ValueError("Reward V4 rollback triggers are invalid")
    if state.pending_rollback_platforms and ordered != state.pending_rollback_platforms:
        raise ValueError("Reward V4 rollback triggers differ from evaluation")

    restored = dict(checkpoint.curriculum_state.platforms)
    for platform in ordered:
        current = state.platforms[platform]
        checkpoint_platform = restored[platform]
        rollback_count = max(
            current.rollback_count, checkpoint_platform.rollback_count
        ) + 1
        locked = rollback_count > config.maximum_efficiency_rollbacks
        next_weights = (
            checkpoint_platform.next_r2_weights
            if locked
            else _reentry_weights(config, rollback_count)
        )
        restored[platform] = PlatformRewardState(
            stage=RewardStage.R1,
            weights=_r1_weights(config),
            next_r2_weights=next_weights,
            consecutive_passes=0,
            consecutive_relative_failures=0,
            rollback_count=rollback_count,
            efficiency_locked_off=locked,
            frozen_r1_baseline=None,
        )
    stage = _STAGE_ORDER[
        max(
            _STAGE_ORDER.index(state.stage),
            _STAGE_ORDER.index(checkpoint.curriculum_state.stage),
        )
    ]
    return RewardCurriculumState(
        stage=stage,
        platforms=restored,
        current_update_id=current_update_id,
        recovery_generation=state.recovery_generation + 1,
        restored_from_checkpoint_sha256=checkpoint.payload_sha256,
        pending_rollback_platforms=(),
        worker_restart_required=True,
    )


def worker_allocation_for_stage(
    stage: TrainingStage,
    config: RewardConfigV4 = DEFAULT_REWARD_CONFIG,
) -> dict[str, int]:
    if not isinstance(stage, TrainingStage) or not isinstance(
        config, RewardConfigV4
    ):
        raise ValueError("Reward V4 worker allocation input is invalid")
    source = (
        config.ground_worker_counts
        if stage is TrainingStage.GROUND_R1
        else config.joint_worker_counts
    )
    allocation = dict(source)
    if sum(allocation.values()) != 24:
        raise ValueError("Reward V4 worker allocation must total 24")
    return allocation


def _apply_r1_evaluation(
    state: PlatformRewardState,
    *,
    metrics: PlatformGateMetrics,
    config: RewardConfigV4,
) -> PlatformRewardState:
    if state.efficiency_locked_off:
        return replace(
            state,
            consecutive_passes=0,
            consecutive_relative_failures=0,
        )
    if not _passes_capability(metrics, config):
        return replace(
            state,
            consecutive_passes=0,
            frozen_r1_baseline=None,
        )
    passes = state.consecutive_passes + 1
    baseline = _better_baseline(state.frozen_r1_baseline, metrics)
    if passes < config.consecutive_gate_confirmations:
        return replace(
            state,
            consecutive_passes=passes,
            frozen_r1_baseline=baseline,
        )
    return replace(
        state,
        stage=RewardStage.R2,
        weights=state.next_r2_weights,
        consecutive_passes=0,
        consecutive_relative_failures=0,
        frozen_r1_baseline=baseline,
    )


def _apply_r2_evaluation(
    state: PlatformRewardState,
    *,
    metrics: PlatformGateMetrics,
    config: RewardConfigV4,
) -> tuple[PlatformRewardState, bool]:
    if _hard_failure(metrics, config):
        return replace(state, consecutive_relative_failures=0), True
    baseline = state.frozen_r1_baseline
    if baseline is None:
        raise ValueError("Reward V4 R2 platform lacks frozen R1 baseline")
    relative_failure = (
        baseline.success_rate - metrics.success_rate
        > config.relative_success_drop_tolerance
        or baseline.mean_final_coverage - metrics.mean_final_coverage
        > config.relative_coverage_drop_tolerance
    )
    failures = state.consecutive_relative_failures + 1 if relative_failure else 0
    return (
        replace(state, consecutive_relative_failures=failures),
        failures >= config.consecutive_gate_confirmations,
    )


def _passes_capability(
    metrics: PlatformGateMetrics, config: RewardConfigV4
) -> bool:
    return (
        metrics.hard_error_count == 0
        and metrics.success_rate >= config.capability_success_rate
        and metrics.mean_final_coverage
        >= config.capability_mean_final_coverage
    )


def _hard_failure(
    metrics: PlatformGateMetrics, config: RewardConfigV4
) -> bool:
    return not _passes_capability(metrics, config)


def _better_baseline(
    current: PlatformGateMetrics | None,
    candidate: PlatformGateMetrics,
) -> PlatformGateMetrics:
    if current is None:
        return candidate
    return max(
        (current, candidate),
        key=lambda item: (
            item.success_rate,
            item.mean_final_coverage,
            -item.hard_error_count,
        ),
    )


def _r1_weights(config: RewardConfigV4) -> RewardWeightsV4:
    return RewardWeightsV4(
        coverage=config.coverage_weight,
        success=config.success_bonus,
        terminal_gap=config.terminal_gap_weight,
        priority=0.0,
        path=0.0,
    )


def _reentry_weights(
    config: RewardConfigV4, rollback_count: int
) -> RewardWeightsV4:
    index = rollback_count - 1
    if not 0 <= index < config.maximum_efficiency_rollbacks:
        raise ValueError("Reward V4 rollback count has no reentry weights")
    return RewardWeightsV4(
        coverage=config.coverage_weight,
        success=config.success_bonus,
        terminal_gap=config.terminal_gap_weight,
        priority=config.reentry_priority_weights[index],
        path=config.reentry_path_weights[index],
    )


def _weights_to_mapping(weights: RewardWeightsV4) -> dict[str, float]:
    return {
        "coverage": weights.coverage,
        "success": weights.success,
        "terminal_gap": weights.terminal_gap,
        "priority": weights.priority,
        "path": weights.path,
    }


def _weights_from_mapping(value: object) -> RewardWeightsV4:
    if not isinstance(value, Mapping) or set(value) != {
        "coverage",
        "success",
        "terminal_gap",
        "priority",
        "path",
    }:
        raise ValueError("Reward V4 curriculum weights structure is invalid")
    return RewardWeightsV4(
        coverage=value["coverage"],
        success=value["success"],
        terminal_gap=value["terminal_gap"],
        priority=value["priority"],
        path=value["path"],
    )


def _ordered_platforms(
    platforms: Sequence[PlatformType],
) -> tuple[PlatformType, ...]:
    values = tuple(platforms)
    if any(not isinstance(platform, PlatformType) for platform in values):
        raise ValueError("Reward V4 platform trigger is invalid")
    return tuple(platform for platform in _PLATFORM_ORDER if platform in values)


def _require_state_and_config(
    state: RewardCurriculumState, config: RewardConfigV4
) -> None:
    if not isinstance(state, RewardCurriculumState):
        raise TypeError("Reward V4 curriculum state is invalid")
    if not isinstance(config, RewardConfigV4):
        raise TypeError("Reward V4 curriculum config is invalid")


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


__all__ = [
    "AcceptedRewardCheckpoint",
    "PlatformGateMetrics",
    "PlatformRewardState",
    "PlatformType",
    "REWARD_CURRICULUM_SCHEMA_VERSION",
    "RewardCurriculumState",
    "TrainingStage",
    "apply_evaluation",
    "apply_rollback",
    "initial_reward_curriculum_state",
    "worker_allocation_for_stage",
]

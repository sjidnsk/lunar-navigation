"""Single frozen identity for Reward V4 training semantics."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from enum import Enum
import hashlib
import json
import math
from typing import Mapping, Sequence

from .training_semantics import FORMAL_SUCCESS_COVERAGE_RATIO


REWARD_SCHEMA_VERSION = "lunar-reward/v4"
_DEFAULT_CONFIG_SENTINEL = object()


class RewardTerminalClass(str, Enum):
    """Environment-owned terminal fact consumed directly by Reward V4."""

    CONTINUE = "CONTINUE"
    SUCCESS = "SUCCESS"
    VALID_INCOMPLETE_TERMINAL = "VALID_INCOMPLETE_TERMINAL"
    INVALID_TRANSITION = "INVALID_TRANSITION"


class RewardStage(str, Enum):
    """Platform-local reward stage in the shared-model curriculum."""

    R1 = "R1"
    R2 = "R2"


class TaskScaleBucket(str, Enum):
    """Frozen four-way task-area stratification."""

    M100_200 = "100_200"
    M200_300 = "200_300"
    M300_400 = "300_400"
    M400_500 = "400_500"


@dataclass(frozen=True, slots=True)
class RewardWeightsV4:
    """The five auditable Reward V4 components."""

    coverage: float = 100.0
    success: float = 100.0
    terminal_gap: float = 1.0
    priority: float = 5.0
    path: float = 1.0

    def __post_init__(self) -> None:
        for value in asdict(self).values():
            if (
                type(value) is not float
                or not math.isfinite(value)
                or value < 0.0
            ):
                raise ValueError("Reward V4 weights must be finite non-negative floats")


@dataclass(frozen=True, slots=True)
class RewardConfigV4:
    """Every approved Reward V4 rule that participates in run identity."""

    success_threshold: float = FORMAL_SUCCESS_COVERAGE_RATIO
    coverage_weight: float = 100.0
    success_bonus: float = 100.0
    terminal_gap_weight: float = 1.0
    priority_weight: float = 5.0
    path_weight: float = 1.0
    gamma: float = 0.995
    gae_lambda: float = 0.95
    macro_actions_per_worker: int = 1
    evaluation_seeds: tuple[int, ...] = (4081, 4082, 4083)
    evaluation_interval_gpu_s: int = 10_800
    evaluation_tasks_per_bucket: int = 3
    capability_success_rate: float = 0.80
    capability_mean_final_coverage: float = 0.90
    consecutive_gate_confirmations: int = 2
    relative_success_drop_tolerance: float = 0.02
    relative_coverage_drop_tolerance: float = 0.01
    reentry_priority_weights: tuple[float, ...] = (2.5, 1.25)
    reentry_path_weights: tuple[float, ...] = (0.5, 0.25)
    maximum_efficiency_rollbacks: int = 2
    task_alignment_m: float = 4.0
    priority_side_divisor: int = 4
    advantage_std_floor: float = 1.0e-8
    scale_bucket_bounds_m: tuple[tuple[float, float], ...] = (
        (100.0, 200.0),
        (200.0, 300.0),
        (300.0, 400.0),
        (400.0, 500.0),
    )
    scale_bucket_weights: tuple[float, ...] = (0.25, 0.25, 0.25, 0.25)
    ground_platform_weights: tuple[tuple[str, float], ...] = (
        ("WHEELED", 0.5),
        ("LEGGED", 0.5),
    )
    joint_platform_weights: tuple[tuple[str, float], ...] = (
        ("WHEELED", 1.0 / 3.0),
        ("LEGGED", 1.0 / 3.0),
        ("HOPPER", 1.0 / 3.0),
    )
    ground_worker_counts: tuple[tuple[str, int], ...] = (
        ("WHEELED", 12),
        ("LEGGED", 12),
    )
    joint_worker_counts: tuple[tuple[str, int], ...] = (
        ("WHEELED", 8),
        ("LEGGED", 8),
        ("HOPPER", 8),
    )

    def __post_init__(self) -> None:
        _validate_reward_config(self)

    @property
    def initial_weights(self) -> RewardWeightsV4:
        return RewardWeightsV4(
            coverage=self.coverage_weight,
            success=self.success_bonus,
            terminal_gap=self.terminal_gap_weight,
            priority=self.priority_weight,
            path=self.path_weight,
        )


def reward_config_as_mapping(
    config: RewardConfigV4 | object = _DEFAULT_CONFIG_SENTINEL,
) -> dict[str, object]:
    """Return the canonical YAML/manifest representation of the contract."""
    if config is _DEFAULT_CONFIG_SENTINEL:
        config = DEFAULT_REWARD_CONFIG
    if not isinstance(config, RewardConfigV4):
        raise ValueError("reward config must use RewardConfigV4")
    _require_reward_config(config)
    return json.loads(
        json.dumps(
            asdict(config),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )


def reward_config_from_mapping(raw: Mapping[str, object]) -> RewardConfigV4:
    """Parse one exact explicit mapping without hidden field defaults."""
    if not isinstance(raw, Mapping):
        raise ValueError("Reward V4 config must be a mapping")
    expected = {field.name for field in fields(RewardConfigV4)}
    if set(raw) != expected:
        raise ValueError("Reward V4 config must contain exactly the frozen fields")
    tuple_fields = {
        "evaluation_seeds",
        "reentry_priority_weights",
        "reentry_path_weights",
        "scale_bucket_bounds_m",
        "scale_bucket_weights",
        "ground_platform_weights",
        "joint_platform_weights",
        "ground_worker_counts",
        "joint_worker_counts",
    }
    values: dict[str, object] = dict(raw)
    for name in tuple_fields:
        value = values[name]
        if not isinstance(value, list):
            raise ValueError(f"Reward V4 field {name} must be an explicit list")
        values[name] = _freeze_sequence(value)
    try:
        return RewardConfigV4(**values)
    except (TypeError, ValueError) as error:
        raise ValueError("Reward V4 config is invalid") from error


def reward_config_sha256(
    config: RewardConfigV4 | object = _DEFAULT_CONFIG_SENTINEL,
) -> str:
    """Hash the schema and complete canonical configuration."""
    if config is _DEFAULT_CONFIG_SENTINEL:
        config = DEFAULT_REWARD_CONFIG
    if not isinstance(config, RewardConfigV4):
        raise ValueError("reward config must use RewardConfigV4")
    payload = json.dumps(
        {
            "schema_version": REWARD_SCHEMA_VERSION,
            "config": reward_config_as_mapping(config),
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _freeze_sequence(value: Sequence[object]) -> tuple[object, ...]:
    return tuple(
        _freeze_sequence(item) if isinstance(item, list) else item
        for item in value
    )


def _require_reward_config(config: RewardConfigV4) -> None:
    if not isinstance(config, RewardConfigV4):
        raise ValueError("reward config must use RewardConfigV4")
    _validate_reward_config(config)


def _validate_reward_config(config: RewardConfigV4) -> None:
    ratio_fields = (
        "success_threshold",
        "gamma",
        "gae_lambda",
        "capability_success_rate",
        "capability_mean_final_coverage",
        "relative_success_drop_tolerance",
        "relative_coverage_drop_tolerance",
    )
    for name in ratio_fields:
        value = getattr(config, name)
        if type(value) is not float or not math.isfinite(value) or not 0.0 < value <= 1.0:
            raise ValueError(f"Reward V4 field {name} must be a finite ratio")
    nonnegative_float_fields = (
        "coverage_weight",
        "success_bonus",
        "terminal_gap_weight",
        "priority_weight",
        "path_weight",
        "task_alignment_m",
        "advantage_std_floor",
    )
    for name in nonnegative_float_fields:
        value = getattr(config, name)
        if type(value) is not float or not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"Reward V4 field {name} must be a positive float")
    positive_int_fields = (
        "macro_actions_per_worker",
        "evaluation_interval_gpu_s",
        "evaluation_tasks_per_bucket",
        "consecutive_gate_confirmations",
        "maximum_efficiency_rollbacks",
        "priority_side_divisor",
    )
    for name in positive_int_fields:
        value = getattr(config, name)
        if type(value) is not int or value <= 0:
            raise ValueError(f"Reward V4 field {name} must be a positive integer")
    if (
        type(config.evaluation_seeds) is not tuple
        or not config.evaluation_seeds
        or any(type(seed) is not int or seed < 0 for seed in config.evaluation_seeds)
        or len(set(config.evaluation_seeds)) != len(config.evaluation_seeds)
    ):
        raise ValueError("Reward V4 evaluation seeds are invalid")
    _validate_float_tuple(config.reentry_priority_weights, 2, "priority reentry")
    _validate_float_tuple(config.reentry_path_weights, 2, "path reentry")
    _validate_float_tuple(config.scale_bucket_weights, 4, "scale weights")
    if not math.isclose(sum(config.scale_bucket_weights), 1.0, abs_tol=1.0e-12):
        raise ValueError("Reward V4 scale weights must sum to one")
    bounds = config.scale_bucket_bounds_m
    if type(bounds) is not tuple or len(bounds) != 4:
        raise ValueError("Reward V4 scale bounds are invalid")
    previous_high: float | None = None
    for pair in bounds:
        if (
            type(pair) is not tuple
            or len(pair) != 2
            or any(type(value) is not float or not math.isfinite(value) for value in pair)
            or pair[0] >= pair[1]
            or (previous_high is not None and pair[0] != previous_high)
        ):
            raise ValueError("Reward V4 scale bounds are invalid")
        previous_high = pair[1]
    if bounds[0][0] != 100.0 or bounds[-1][1] != 500.0:
        raise ValueError("Reward V4 scale bounds must span 100 to 500 metres")
    _validate_weight_pairs(
        config.ground_platform_weights,
        ("WHEELED", "LEGGED"),
        "ground platform weights",
    )
    _validate_weight_pairs(
        config.joint_platform_weights,
        ("WHEELED", "LEGGED", "HOPPER"),
        "joint platform weights",
    )
    _validate_worker_pairs(
        config.ground_worker_counts,
        ("WHEELED", "LEGGED"),
        "ground worker counts",
    )
    _validate_worker_pairs(
        config.joint_worker_counts,
        ("WHEELED", "LEGGED", "HOPPER"),
        "joint worker counts",
    )


def _validate_float_tuple(value: object, length: int, name: str) -> None:
    if (
        type(value) is not tuple
        or len(value) != length
        or any(
            type(item) is not float or not math.isfinite(item) or item < 0.0
            for item in value
        )
    ):
        raise ValueError(f"Reward V4 {name} are invalid")


def _validate_weight_pairs(
    pairs: object, expected_platforms: tuple[str, ...], name: str
) -> None:
    if type(pairs) is not tuple or len(pairs) != len(expected_platforms):
        raise ValueError(f"Reward V4 {name} are invalid")
    weights: list[float] = []
    platforms: list[str] = []
    for pair in pairs:
        if (
            type(pair) is not tuple
            or len(pair) != 2
            or type(pair[0]) is not str
            or type(pair[1]) is not float
            or not math.isfinite(pair[1])
            or pair[1] <= 0.0
        ):
            raise ValueError(f"Reward V4 {name} are invalid")
        platforms.append(pair[0])
        weights.append(pair[1])
    if tuple(platforms) != expected_platforms:
        raise ValueError(f"Reward V4 {name} are invalid")
    if not math.isclose(sum(weights), 1.0, abs_tol=1.0e-12):
        raise ValueError(f"Reward V4 {name} must sum to one")


def _validate_worker_pairs(
    pairs: object, expected_platforms: tuple[str, ...], name: str
) -> None:
    if type(pairs) is not tuple or len(pairs) != len(expected_platforms):
        raise ValueError(f"Reward V4 {name} are invalid")
    if any(
        type(pair) is not tuple
        or len(pair) != 2
        or type(pair[0]) is not str
        or type(pair[1]) is not int
        or pair[1] <= 0
        for pair in pairs
    ):
        raise ValueError(f"Reward V4 {name} are invalid")
    if tuple(pair[0] for pair in pairs) != expected_platforms:
        raise ValueError(f"Reward V4 {name} are invalid")
    if sum(pair[1] for pair in pairs) != 24:
        raise ValueError(f"Reward V4 {name} must total 24")


DEFAULT_REWARD_CONFIG = RewardConfigV4()


__all__ = [
    "DEFAULT_REWARD_CONFIG",
    "REWARD_SCHEMA_VERSION",
    "RewardConfigV4",
    "RewardStage",
    "RewardTerminalClass",
    "RewardWeightsV4",
    "TaskScaleBucket",
    "reward_config_as_mapping",
    "reward_config_from_mapping",
    "reward_config_sha256",
]

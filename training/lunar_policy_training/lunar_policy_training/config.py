"""Validated immutable runtime configuration for Volume 3 training."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import yaml


PLATFORMS = ("WHEELED", "LEGGED", "HOPPER")
WORKER_CANDIDATES = (18, 24, 30)
FORMAL_WORKER_CANDIDATES = (24,)
ROLLOUT_HORIZON_CANDIDATES = (16, 32, 64)
FORMAL_WORKER_RESPONSE_TIMEOUT_SECONDS = 1200.0
TASK_AREA_SAMPLING_ALGORITHM = "deterministic-uniform-square/v1"
_CONFIG_FIELDS = {
    "parallel",
    "task_area",
    "ppo",
    "checkpoint_interval_seconds",
    "candidate_checkpoint_interval_seconds",
    "total_gpu_budget_seconds",
    "formal_training_seeds",
    "run_kind",
}


class TrainingConfigError(ValueError):
    """A training configuration violates the frozen Task 3 contract."""


_FROZEN_PPO_FIELDS: tuple[tuple[str, type[object], object], ...] = (
    ("gamma", float, 0.995),
    ("gae_lambda", float, 0.95),
    ("policy_clip", float, 0.20),
    ("value_clip", float, 0.20),
    ("learning_rate", float, 3.0e-4),
    ("optimizer", str, "AdamW"),
    ("weight_decay", float, 1.0e-4),
    ("adam_epsilon", float, 1.0e-5),
    ("epochs_per_update", int, 4),
    ("target_kl", float, 0.03),
    ("value_loss_coefficient", float, 0.5),
    ("frontier_entropy_coef", float, 0.01),
    ("theta_entropy_coef", float, 0.001),
    ("max_grad_norm", float, 0.5),
    ("dtype", str, "float32"),
)


@dataclass(frozen=True, slots=True)
class ParallelConfig:
    worker_candidates: tuple[int, ...]
    preferred_workers: int
    joint_workers: Mapping[str, int]
    nested_compute_threads: int
    gpu_memory_fraction_max: float


@dataclass(frozen=True, slots=True)
class TaskAreaConfig:
    minimum_size_m: float
    maximum_size_m: float
    sampling_algorithm: str

    def __post_init__(self) -> None:
        validate_task_area_config(self)


@dataclass(frozen=True, slots=True)
class PPOConfig:
    gamma: float
    gae_lambda: float
    policy_clip: float
    value_clip: float
    learning_rate: float
    optimizer: str
    weight_decay: float
    adam_epsilon: float
    epochs_per_update: int
    target_kl: float
    value_loss_coefficient: float
    frontier_entropy_coef: float
    theta_entropy_coef: float
    max_grad_norm: float
    rollout_horizon: int
    dtype: str

    def __post_init__(self) -> None:
        validate_ppo_config(self)


@dataclass(frozen=True, slots=True)
class ResolvedTrainingConfig:
    parallel: ParallelConfig
    task_area: TaskAreaConfig
    ppo: PPOConfig
    checkpoint_interval_seconds: int
    candidate_checkpoint_interval_seconds: int
    total_gpu_budget_seconds: int
    formal_training_seeds: tuple[int, ...]
    run_kind: str

    def as_frozen_dict(self) -> dict[str, object]:
        return {
            "parallel": {
                "worker_candidates": list(self.parallel.worker_candidates),
                "preferred_workers": self.parallel.preferred_workers,
                "joint_workers": dict(self.parallel.joint_workers),
                "nested_compute_threads": self.parallel.nested_compute_threads,
                "gpu_memory_fraction_max": (
                    self.parallel.gpu_memory_fraction_max
                ),
            },
            "task_area": {
                "minimum_size_m": self.task_area.minimum_size_m,
                "maximum_size_m": self.task_area.maximum_size_m,
                "sampling_algorithm": self.task_area.sampling_algorithm,
            },
            "ppo": {
                "gamma": self.ppo.gamma,
                "gae_lambda": self.ppo.gae_lambda,
                "policy_clip": self.ppo.policy_clip,
                "value_clip": self.ppo.value_clip,
                "learning_rate": self.ppo.learning_rate,
                "optimizer": self.ppo.optimizer,
                "weight_decay": self.ppo.weight_decay,
                "adam_epsilon": self.ppo.adam_epsilon,
                "epochs_per_update": self.ppo.epochs_per_update,
                "target_kl": self.ppo.target_kl,
                "value_loss_coefficient": self.ppo.value_loss_coefficient,
                "frontier_entropy_coef": self.ppo.frontier_entropy_coef,
                "theta_entropy_coef": self.ppo.theta_entropy_coef,
                "max_grad_norm": self.ppo.max_grad_norm,
                "rollout_horizon": self.ppo.rollout_horizon,
                "dtype": self.ppo.dtype,
            },
            "checkpoint_interval_seconds": self.checkpoint_interval_seconds,
            "candidate_checkpoint_interval_seconds": (
                self.candidate_checkpoint_interval_seconds
            ),
            "total_gpu_budget_seconds": self.total_gpu_budget_seconds,
            "formal_training_seeds": list(self.formal_training_seeds),
            "run_kind": self.run_kind,
        }


def load_training_config(path: str | Path) -> ResolvedTrainingConfig:
    """Load one explicit YAML file and enforce the fixed Task 3 values."""
    target = Path(path)
    if not target.is_file():
        raise TrainingConfigError("training config file is missing")
    try:
        raw = yaml.safe_load(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise TrainingConfigError("training config could not be loaded") from error
    if not isinstance(raw, dict):
        raise TrainingConfigError("training config must be a mapping")
    return resolve_training_config(raw)


def resolve_training_config(raw: Mapping[str, object]) -> ResolvedTrainingConfig:
    """Resolve an already-loaded run-manifest config through the same gates."""
    if not isinstance(raw, Mapping):
        raise TrainingConfigError("training config must be a mapping")
    if "task_area" not in raw:
        raise TrainingConfigError("task area config is required")
    if set(raw) != _CONFIG_FIELDS:
        raise TrainingConfigError(
            "training config must contain exactly the frozen top-level fields"
        )
    parallel_raw = raw.get("parallel")
    if not isinstance(parallel_raw, Mapping):
        raise TrainingConfigError("parallel config must be a mapping")
    joint_raw = parallel_raw.get("joint_workers")
    if not isinstance(joint_raw, Mapping):
        raise TrainingConfigError("joint worker allocation must be a mapping")
    task_area_raw = raw.get("task_area")
    task_area_fields = {
        "minimum_size_m",
        "maximum_size_m",
        "sampling_algorithm",
    }
    if not isinstance(task_area_raw, Mapping) or set(task_area_raw) != task_area_fields:
        raise TrainingConfigError(
            "task area config must contain exactly the frozen fields"
        )
    ppo_raw = raw.get("ppo")
    ppo_fields = {
        "gamma",
        "gae_lambda",
        "policy_clip",
        "value_clip",
        "learning_rate",
        "optimizer",
        "weight_decay",
        "adam_epsilon",
        "epochs_per_update",
        "target_kl",
        "value_loss_coefficient",
        "frontier_entropy_coef",
        "theta_entropy_coef",
        "max_grad_norm",
        "rollout_horizon",
        "dtype",
    }
    if not isinstance(ppo_raw, Mapping) or set(ppo_raw) != ppo_fields:
        raise TrainingConfigError("PPO config must contain exactly the frozen fields")
    config = ResolvedTrainingConfig(
        parallel=ParallelConfig(
            worker_candidates=_integer_tuple(
                parallel_raw.get("worker_candidates"), "worker candidates"
            ),
            preferred_workers=_integer(
                parallel_raw.get("preferred_workers"), "preferred workers"
            ),
            joint_workers=MappingProxyType(
                {
                    platform: _integer(joint_raw.get(platform), platform)
                    for platform in PLATFORMS
                }
            ),
            nested_compute_threads=_integer(
                parallel_raw.get("nested_compute_threads"),
                "nested compute threads",
            ),
            gpu_memory_fraction_max=_number(
                parallel_raw.get("gpu_memory_fraction_max"),
                "GPU memory fraction",
            ),
        ),
        task_area=TaskAreaConfig(
            minimum_size_m=_exact_float(
                task_area_raw["minimum_size_m"], "task area minimum"
            ),
            maximum_size_m=_exact_float(
                task_area_raw["maximum_size_m"], "task area maximum"
            ),
            sampling_algorithm=_string(
                task_area_raw["sampling_algorithm"], "task area sampling"
            ),
        ),
        ppo=PPOConfig(
            gamma=_number(ppo_raw["gamma"], "PPO gamma"),
            gae_lambda=_number(ppo_raw["gae_lambda"], "PPO GAE lambda"),
            policy_clip=_number(ppo_raw["policy_clip"], "PPO policy clip"),
            value_clip=_number(ppo_raw["value_clip"], "PPO value clip"),
            learning_rate=_number(
                ppo_raw["learning_rate"], "PPO learning rate"
            ),
            optimizer=_string(ppo_raw["optimizer"], "PPO optimizer"),
            weight_decay=_number(
                ppo_raw["weight_decay"], "PPO weight decay"
            ),
            adam_epsilon=_number(
                ppo_raw["adam_epsilon"], "PPO Adam epsilon"
            ),
            epochs_per_update=_integer(
                ppo_raw["epochs_per_update"], "PPO epochs per update"
            ),
            target_kl=_number(ppo_raw["target_kl"], "PPO target KL"),
            value_loss_coefficient=_number(
                ppo_raw["value_loss_coefficient"],
                "PPO value loss coefficient",
            ),
            frontier_entropy_coef=_number(
                ppo_raw["frontier_entropy_coef"],
                "PPO frontier entropy coefficient",
            ),
            theta_entropy_coef=_number(
                ppo_raw["theta_entropy_coef"],
                "PPO theta entropy coefficient",
            ),
            max_grad_norm=_number(
                ppo_raw["max_grad_norm"], "PPO max gradient norm"
            ),
            rollout_horizon=_integer(
                ppo_raw["rollout_horizon"], "PPO rollout horizon"
            ),
            dtype=_string(ppo_raw["dtype"], "PPO dtype"),
        ),
        checkpoint_interval_seconds=_integer(
            raw.get("checkpoint_interval_seconds"), "checkpoint interval"
        ),
        candidate_checkpoint_interval_seconds=_integer(
            raw.get("candidate_checkpoint_interval_seconds"),
            "candidate checkpoint interval",
        ),
        total_gpu_budget_seconds=_integer(
            raw.get("total_gpu_budget_seconds"), "total GPU budget"
        ),
        formal_training_seeds=_integer_tuple(
            raw.get("formal_training_seeds"), "formal training seeds"
        ),
        run_kind=_string(raw.get("run_kind"), "run kind"),
    )
    _validate_frozen_values(config)
    return config


def _validate_frozen_values(config: ResolvedTrainingConfig) -> None:
    validate_ppo_config(config.ppo)
    if config.run_kind == "formal":
        worker_candidates = FORMAL_WORKER_CANDIDATES
        preferred_workers = 24
        joint_workers = {"WHEELED": 8, "LEGGED": 8, "HOPPER": 8}
    elif config.run_kind == "development-smoke":
        worker_candidates = WORKER_CANDIDATES
        preferred_workers = 30
        joint_workers = {"WHEELED": 10, "LEGGED": 10, "HOPPER": 10}
    else:
        raise TrainingConfigError("run kind must be formal or development-smoke")
    expected = {
        "worker_candidates": worker_candidates,
        "preferred_workers": preferred_workers,
        "joint_workers": joint_workers,
        "nested_compute_threads": 1,
        "gpu_memory_fraction_max": 0.90,
        "checkpoint_interval_seconds": 1800,
        "candidate_checkpoint_interval_seconds": 3600,
        "total_gpu_budget_seconds": 86400,
        "formal_training_seeds": (4080,),
        "task_area": (
            100.0,
            500.0,
            TASK_AREA_SAMPLING_ALGORITHM,
        ),
    }
    actual = {
        "worker_candidates": config.parallel.worker_candidates,
        "preferred_workers": config.parallel.preferred_workers,
        "joint_workers": dict(config.parallel.joint_workers),
        "nested_compute_threads": config.parallel.nested_compute_threads,
        "gpu_memory_fraction_max": config.parallel.gpu_memory_fraction_max,
        "checkpoint_interval_seconds": config.checkpoint_interval_seconds,
        "candidate_checkpoint_interval_seconds": (
            config.candidate_checkpoint_interval_seconds
        ),
        "total_gpu_budget_seconds": config.total_gpu_budget_seconds,
        "formal_training_seeds": config.formal_training_seeds,
        "task_area": (
            config.task_area.minimum_size_m,
            config.task_area.maximum_size_m,
            config.task_area.sampling_algorithm,
        ),
    }
    if actual != expected:
        raise TrainingConfigError("training config changes a frozen Task 3 value")


def validate_ppo_config(config: PPOConfig) -> PPOConfig:
    """Reject any typed PPO configuration outside the exact frozen baseline."""
    if not isinstance(config, PPOConfig):
        raise TrainingConfigError("PPO config must use typed PPOConfig")
    for field, expected_type, expected_value in _FROZEN_PPO_FIELDS:
        actual = getattr(config, field)
        if type(actual) is not expected_type:
            raise TrainingConfigError(
                f"PPO field {field} must have exact type {expected_type.__name__}"
            )
        if actual != expected_value:
            raise TrainingConfigError(f"PPO field {field} changes a frozen value")
    if type(config.rollout_horizon) is not int:
        raise TrainingConfigError("PPO field rollout_horizon must have exact type int")
    if config.rollout_horizon not in ROLLOUT_HORIZON_CANDIDATES:
        raise TrainingConfigError(
            "PPO rollout horizon must be one of the calibrated candidates"
        )
    return config


def validate_task_area_config(config: TaskAreaConfig) -> TaskAreaConfig:
    """Reject task-area drift before it can change the training distribution."""
    if not isinstance(config, TaskAreaConfig):
        raise TrainingConfigError("task area config must use typed TaskAreaConfig")
    if type(config.minimum_size_m) is not float:
        raise TrainingConfigError("task area minimum must have exact type float")
    if type(config.maximum_size_m) is not float:
        raise TrainingConfigError("task area maximum must have exact type float")
    if config.minimum_size_m != 100.0 or config.maximum_size_m != 500.0:
        raise TrainingConfigError("task area changes the frozen 100-500 m range")
    if (
        type(config.sampling_algorithm) is not str
        or config.sampling_algorithm != TASK_AREA_SAMPLING_ALGORITHM
    ):
        raise TrainingConfigError("task area sampling algorithm changes a frozen value")
    return config


def with_rollout_horizon(
    config: ResolvedTrainingConfig, rollout_horizon: int
) -> ResolvedTrainingConfig:
    """Return the immutable run config selected by equal-work calibration."""
    if not isinstance(config, ResolvedTrainingConfig):
        raise TrainingConfigError("rollout horizon selection requires resolved config")
    return replace(
        config,
        ppo=replace(config.ppo, rollout_horizon=rollout_horizon),
    )


def _integer(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise TrainingConfigError(f"{name} must be a positive integer")
    return value


def _integer_tuple(value: object, name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise TrainingConfigError(f"{name} must be a non-empty list")
    return tuple(_integer(item, name) for item in value)


def _number(value: object, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TrainingConfigError(f"{name} must be numeric")
    return float(value)


def _exact_float(value: object, name: str) -> float:
    if type(value) is not float:
        raise TrainingConfigError(f"{name} must have exact type float")
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TrainingConfigError(f"{name} must be a non-empty string")
    return value


__all__ = [
    "FORMAL_WORKER_RESPONSE_TIMEOUT_SECONDS",
    "FORMAL_WORKER_CANDIDATES",
    "PLATFORMS",
    "TASK_AREA_SAMPLING_ALGORITHM",
    "ParallelConfig",
    "PPOConfig",
    "ResolvedTrainingConfig",
    "TaskAreaConfig",
    "TrainingConfigError",
    "load_training_config",
    "resolve_training_config",
    "validate_task_area_config",
    "validate_ppo_config",
]

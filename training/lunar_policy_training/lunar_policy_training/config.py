"""Validated immutable runtime configuration for Volume 3 training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import yaml


PLATFORMS = ("WHEELED", "LEGGED", "HOPPER")


class TrainingConfigError(ValueError):
    """A training configuration violates the frozen Task 3 contract."""


@dataclass(frozen=True, slots=True)
class ParallelConfig:
    worker_candidates: tuple[int, ...]
    preferred_workers: int
    joint_workers: Mapping[str, int]
    nested_compute_threads: int
    gpu_memory_fraction_max: float


@dataclass(frozen=True, slots=True)
class ResolvedTrainingConfig:
    parallel: ParallelConfig
    checkpoint_interval_seconds: int
    candidate_checkpoint_interval_seconds: int
    total_gpu_budget_seconds: int
    formal_training_seeds: tuple[int, ...]
    proxy: bool

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
            "checkpoint_interval_seconds": self.checkpoint_interval_seconds,
            "candidate_checkpoint_interval_seconds": (
                self.candidate_checkpoint_interval_seconds
            ),
            "total_gpu_budget_seconds": self.total_gpu_budget_seconds,
            "formal_training_seeds": list(self.formal_training_seeds),
            "proxy": self.proxy,
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
    parallel_raw = raw.get("parallel")
    if not isinstance(parallel_raw, Mapping):
        raise TrainingConfigError("parallel config must be a mapping")
    joint_raw = parallel_raw.get("joint_workers")
    if not isinstance(joint_raw, Mapping):
        raise TrainingConfigError("joint worker allocation must be a mapping")
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
        proxy=raw.get("proxy"),
    )
    _validate_frozen_values(config)
    return config


def _validate_frozen_values(config: ResolvedTrainingConfig) -> None:
    expected = {
        "worker_candidates": (18, 24),
        "preferred_workers": 24,
        "joint_workers": {"WHEELED": 8, "LEGGED": 8, "HOPPER": 8},
        "nested_compute_threads": 1,
        "gpu_memory_fraction_max": 0.90,
        "checkpoint_interval_seconds": 1800,
        "candidate_checkpoint_interval_seconds": 3600,
        "total_gpu_budget_seconds": 86400,
        "formal_training_seeds": (4080,),
        "proxy": True,
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
        "proxy": config.proxy,
    }
    if actual != expected:
        raise TrainingConfigError("training config changes a frozen Task 3 value")


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


__all__ = [
    "PLATFORMS",
    "ParallelConfig",
    "ResolvedTrainingConfig",
    "TrainingConfigError",
    "load_training_config",
    "resolve_training_config",
]

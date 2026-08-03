"""Atomic complete-run checkpoints layered over the PPO core validators."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from .ppo.checkpoint import (
    OBSERVATION_CONTRACT_VERSION,
    CheckpointError,
    _capture_rng_state,
    _cpu_copy,
    _semantic_sha256,
    _restore_rng_state,
    _validate_finite_tensors,
    _validated_rng_state,
)


CHECKPOINT_SCHEMA_VERSION = "lunar-ppo-checkpoint/v2"
_BODY_FIELDS = {
    "schema_version",
    "contract_version",
    "model_state",
    "optimizer_state",
    "scheduler_state",
    "global_step",
    "curriculum_phase",
    "normalization",
    "rng_state",
    "frozen_config",
    "config_hash",
    "source_commit",
    "consumed_gpu_seconds",
}


@dataclass(frozen=True, slots=True)
class TrainingCheckpointV2:
    schema_version: str
    contract_version: str
    model_state: Mapping[object, object]
    optimizer_state: Mapping[object, object]
    scheduler_state: Mapping[object, object]
    global_step: int
    curriculum_phase: str
    normalization: Mapping[object, object]
    rng_state: Mapping[object, object]
    frozen_config: dict[str, object]
    config_hash: str
    source_commit: str
    consumed_gpu_seconds: float
    payload_sha256: str


def build_training_checkpoint(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    global_step: int,
    curriculum_phase: str,
    normalization: Mapping[object, object],
    frozen_config: Mapping[str, object],
    source_commit: str,
    consumed_gpu_seconds: float,
) -> TrainingCheckpointV2:
    """Capture a complete run state with the PPO core's strict validators."""
    if not isinstance(model, nn.Module):
        raise CheckpointError("model must be a Torch module")
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise CheckpointError("optimizer must be a Torch optimizer")
    if not isinstance(scheduler, torch.optim.lr_scheduler.LRScheduler):
        raise CheckpointError("scheduler must be a Torch LR scheduler")
    body = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "contract_version": OBSERVATION_CONTRACT_VERSION,
        "model_state": _cpu_copy(model.state_dict()),
        "optimizer_state": _cpu_copy(optimizer.state_dict()),
        "scheduler_state": _cpu_copy(scheduler.state_dict()),
        "global_step": global_step,
        "curriculum_phase": curriculum_phase,
        "normalization": _cpu_copy(dict(normalization)),
        "rng_state": _capture_rng_state(),
        "frozen_config": _json_copy(frozen_config),
        "config_hash": config_sha256(frozen_config),
        "source_commit": source_commit,
        "consumed_gpu_seconds": consumed_gpu_seconds,
    }
    _validate_body(body)
    return _checkpoint_from_body(body, _semantic_sha256(body))


def save_checkpoint_atomic(
    path: str | Path,
    checkpoint: TrainingCheckpointV2,
    *,
    overwrite: bool = True,
) -> None:
    """Flush one same-directory temporary file before atomically replacing target."""
    if not isinstance(checkpoint, TrainingCheckpointV2):
        raise CheckpointError("checkpoint must use TrainingCheckpointV2")
    target = _validated_target(path)
    if not overwrite and target.exists():
        raise CheckpointError("immutable checkpoint already exists")
    body = _body_from_checkpoint(checkpoint)
    _validate_body(body)
    payload_sha256 = _semantic_sha256(body)
    if payload_sha256 != checkpoint.payload_sha256:
        raise CheckpointError("checkpoint payload hash mismatch")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(
                {"body": body, "body_sha256": payload_sha256},
                stream,
            )
            stream.flush()
            os.fsync(stream.fileno())
        if not overwrite and target.exists():
            raise CheckpointError("immutable checkpoint already exists")
        os.replace(temporary, target)
        directory_fd = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception as error:
        if temporary.exists():
            temporary.unlink()
        if isinstance(error, CheckpointError):
            raise
        raise CheckpointError("checkpoint file could not be written atomically") from error


def load_checkpoint(path: str | Path) -> TrainingCheckpointV2:
    """Load and validate an inert checkpoint payload without mutating live state."""
    target = Path(path)
    if target.is_symlink() or not target.is_file():
        raise CheckpointError("checkpoint must be an existing regular file")
    try:
        payload = torch.load(target, map_location="cpu", weights_only=True)
    except Exception as error:
        raise CheckpointError("checkpoint restricted loader rejected payload") from error
    if not isinstance(payload, Mapping) or set(payload) != {
        "body",
        "body_sha256",
    }:
        raise CheckpointError("checkpoint payload structure is invalid")
    body = payload["body"]
    payload_sha256 = payload["body_sha256"]
    _validate_body(body)
    if not _is_sha256(payload_sha256):
        raise CheckpointError("checkpoint payload hash is invalid")
    if _semantic_sha256(body) != payload_sha256:
        raise CheckpointError("checkpoint payload hash mismatch")
    return _checkpoint_from_body(body, payload_sha256)


def load_checkpoint_for_resume(
    path: str | Path,
    *,
    expected_contract_version: str,
    expected_config_hash: str,
    expected_source_commit: str,
) -> TrainingCheckpointV2:
    """Reject any run identity drift before live state can be mutated."""
    checkpoint = load_checkpoint(path)
    if checkpoint.contract_version != expected_contract_version:
        raise CheckpointError("checkpoint contract version mismatch")
    if checkpoint.config_hash != expected_config_hash:
        raise CheckpointError("checkpoint config hash mismatch")
    if checkpoint.source_commit != expected_source_commit:
        raise CheckpointError("checkpoint source commit mismatch")
    return checkpoint


def restore_training_state(
    checkpoint: TrainingCheckpointV2,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> None:
    """Restore complete train/RNG state, rolling live objects back on failure."""
    if not isinstance(checkpoint, TrainingCheckpointV2):
        raise CheckpointError("checkpoint must use TrainingCheckpointV2")
    if not isinstance(model, nn.Module):
        raise CheckpointError("model must be a Torch module")
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise CheckpointError("optimizer must be a Torch optimizer")
    if not isinstance(scheduler, torch.optim.lr_scheduler.LRScheduler):
        raise CheckpointError("scheduler must be a Torch LR scheduler")
    live_model = _cpu_copy(model.state_dict())
    live_optimizer = _cpu_copy(optimizer.state_dict())
    live_scheduler = _cpu_copy(scheduler.state_dict())
    live_rng = _capture_rng_state()
    try:
        model.load_state_dict(dict(checkpoint.model_state), strict=True)
        optimizer.load_state_dict(dict(checkpoint.optimizer_state))
        scheduler.load_state_dict(dict(checkpoint.scheduler_state))
        _restore_rng_state(checkpoint.rng_state)
    except Exception as error:
        try:
            model.load_state_dict(dict(live_model), strict=True)
            optimizer.load_state_dict(dict(live_optimizer))
            scheduler.load_state_dict(dict(live_scheduler))
            _restore_rng_state(live_rng)
        except Exception as rollback_error:
            raise CheckpointError("checkpoint restore rollback failed") from rollback_error
        raise CheckpointError("checkpoint train state cannot be restored") from error


def config_sha256(config: Mapping[str, object]) -> str:
    """Hash a JSON-compatible frozen configuration deterministically."""
    frozen = _json_copy(config)
    encoded = json.dumps(
        frozen,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_body(body: object) -> None:
    if not isinstance(body, Mapping) or set(body) != _BODY_FIELDS:
        raise CheckpointError("checkpoint body structure is invalid")
    if body["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointError("checkpoint schema version mismatch")
    if body["contract_version"] != OBSERVATION_CONTRACT_VERSION:
        raise CheckpointError("checkpoint contract version mismatch")
    for name in ("model_state", "optimizer_state", "scheduler_state"):
        if not isinstance(body[name], Mapping):
            raise CheckpointError(f"checkpoint {name} is invalid")
        _validate_finite_tensors(body[name], state_name=name)
    if type(body["global_step"]) is not int or body["global_step"] < 0:
        raise CheckpointError("checkpoint global step is invalid")
    if not isinstance(body["curriculum_phase"], str) or not body["curriculum_phase"]:
        raise CheckpointError("checkpoint curriculum phase is invalid")
    if not isinstance(body["normalization"], Mapping):
        raise CheckpointError("checkpoint normalization state is invalid")
    _validate_finite_tensors(body["normalization"], state_name="normalization")
    _validated_rng_state(body["rng_state"])
    if not isinstance(body["frozen_config"], Mapping):
        raise CheckpointError("checkpoint frozen config is invalid")
    if config_sha256(body["frozen_config"]) != body["config_hash"]:
        raise CheckpointError("checkpoint config hash mismatch")
    if not _is_source_commit(body["source_commit"]):
        raise CheckpointError("checkpoint source commit is invalid")
    consumed = body["consumed_gpu_seconds"]
    if (
        not isinstance(consumed, (int, float))
        or isinstance(consumed, bool)
        or not math.isfinite(float(consumed))
        or consumed < 0.0
    ):
        raise CheckpointError("checkpoint consumed GPU seconds are invalid")


def _checkpoint_from_body(
    body: Mapping[object, object], payload_sha256: str
) -> TrainingCheckpointV2:
    return TrainingCheckpointV2(
        schema_version=body["schema_version"],
        contract_version=body["contract_version"],
        model_state=body["model_state"],
        optimizer_state=body["optimizer_state"],
        scheduler_state=body["scheduler_state"],
        global_step=body["global_step"],
        curriculum_phase=body["curriculum_phase"],
        normalization=body["normalization"],
        rng_state=body["rng_state"],
        frozen_config=dict(body["frozen_config"]),
        config_hash=body["config_hash"],
        source_commit=body["source_commit"],
        consumed_gpu_seconds=float(body["consumed_gpu_seconds"]),
        payload_sha256=payload_sha256,
    )


def _body_from_checkpoint(checkpoint: TrainingCheckpointV2) -> dict[str, object]:
    return {
        name: getattr(checkpoint, name)
        for name in _BODY_FIELDS
    }


def _validated_target(path: str | Path) -> Path:
    target = Path(path)
    if target.is_symlink():
        raise CheckpointError("checkpoint target must not be a symbolic link")
    if target.exists() and not target.is_file():
        raise CheckpointError("checkpoint target must be a regular file")
    if not target.parent.is_dir():
        raise CheckpointError("checkpoint parent directory is missing")
    return target


def _json_copy(value: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise CheckpointError("frozen config must be a mapping")
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True))
    except (TypeError, ValueError) as error:
        raise CheckpointError("frozen config must be JSON-compatible") from error


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_source_commit(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "TrainingCheckpointV2",
    "build_training_checkpoint",
    "config_sha256",
    "load_checkpoint",
    "load_checkpoint_for_resume",
    "restore_training_state",
    "save_checkpoint_atomic",
]

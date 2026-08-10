"""Atomic complete-run checkpoints layered over the PPO core validators."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import tempfile
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from lunar_model_contract import ObservationContractV2, ObservationContractV3

from .budget import (
    BUDGET_EXTENSION_BLOCK_SECONDS,
    INITIAL_GPU_BUDGET_SECONDS,
)
from .environment.formal_episode_state import (
    FORMAL_ENVIRONMENT_STATE_SCHEMA_VERSION,
    FormalWorkerState,
)
from .ppo.checkpoint import (
    CheckpointError,
    _capture_rng_state,
    _cpu_copy,
    _semantic_sha256,
    _restore_rng_state,
    _validate_finite_tensors,
    _validated_rng_state,
)
from .reward import reward_weights_sha256
from .training_semantics import training_semantics_sha256


CHECKPOINT_SCHEMA_VERSION = "lunar-ppo-checkpoint/v6"
OBSERVATION_CONTRACT_VERSION = ObservationContractV3.version
_LEGACY_V5_SCHEMA_VERSION = "lunar-ppo-checkpoint/v5"
_LEGACY_V4_SCHEMA_VERSION = "lunar-ppo-checkpoint/v4"
_LEGACY_V3_SCHEMA_VERSION = "lunar-ppo-checkpoint/v3"
_LEGACY_V2_SCHEMA_VERSION = "lunar-ppo-checkpoint/v2"
_LEGACY_OBSERVATION_CONTRACT_VERSION = "ObservationContractV1"
_LEGACY_V2_OBSERVATION_CONTRACT_VERSION = ObservationContractV2.version
POLICY_WARM_START_PREFIXES = (
    "global_encoder",
    "local_encoder",
    "pose_encoder",
    "platform_encoder",
    "frontier_encoder",
    "frontier_position_encoder",
    "cross_attention_blocks",
    "action_output_mlp",
    "frontier_logit_head",
    "theta_sin_head",
    "theta_cos_head",
    "theta_kappa_head",
)
_POLICY_WARM_START_EXCLUDED_PREFIX = "value_mlp"
_RUN_IDENTITY_FIELDS = (
    "run_kind",
    "data_sha256",
    "split_sha256",
    "generator_sha256",
    "capability_sha256",
    "reward_sha256",
    "v3_sha256",
    "training_semantics_sha256",
)
_LEGACY_V3_RUN_IDENTITY_FIELDS = _RUN_IDENTITY_FIELDS[:-1]
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
    "environment_state",
    "run_identity",
    "config_hash",
    "source_commit",
    "consumed_gpu_seconds",
    "budget_extension_blocks",
    "total_gpu_budget_seconds",
    "worker_allocation",
    "micro_batch_size",
    "latest_checkpoint_gpu_seconds",
    "candidate_checkpoint_gpu_seconds",
}
_V4_BODY_FIELDS = _BODY_FIELDS - {"environment_state"}
_V2_BODY_FIELDS = _V4_BODY_FIELDS - {"run_identity"}


@dataclass(frozen=True, slots=True)
class RunIdentity:
    run_kind: str
    data_sha256: str
    split_sha256: str
    generator_sha256: str
    capability_sha256: str
    reward_sha256: str
    v3_sha256: str
    training_semantics_sha256: str

    def __post_init__(self) -> None:
        if self.run_kind not in ("formal", "development-smoke"):
            raise CheckpointError("run identity run kind is invalid")
        for field in _RUN_IDENTITY_FIELDS[1:]:
            if not _is_sha256(getattr(self, field)):
                raise CheckpointError(
                    f"run identity {field.replace('_', ' ')} is invalid"
                )

    def to_dict(self) -> dict[str, str]:
        return {field: getattr(self, field) for field in _RUN_IDENTITY_FIELDS}

    @classmethod
    def from_mapping(cls, value: object) -> "RunIdentity":
        return _run_identity_from_mapping(value)


@dataclass(frozen=True, slots=True)
class PolicyWarmStartEvidence:
    """Auditable evidence for a policy-only import into a fresh run."""

    parent_checkpoint_sha256: str
    parent_payload_sha256: str
    parent_global_step: int
    loaded_prefixes: tuple[str, ...]
    value_head_reinitialization_sha256: str
    value_head_seed: int

    def __post_init__(self) -> None:
        if not _is_sha256(self.parent_checkpoint_sha256) or not _is_sha256(
            self.parent_payload_sha256
        ):
            raise CheckpointError("policy warm-start parent digest is invalid")
        if type(self.parent_global_step) is not int or self.parent_global_step < 0:
            raise CheckpointError("policy warm-start parent step is invalid")
        if (
            not isinstance(self.loaded_prefixes, tuple)
            or not self.loaded_prefixes
            or len(set(self.loaded_prefixes)) != len(self.loaded_prefixes)
            or any(
                not isinstance(prefix, str) or not prefix
                for prefix in self.loaded_prefixes
            )
        ):
            raise CheckpointError("policy warm-start prefix evidence is invalid")
        if not _is_sha256(self.value_head_reinitialization_sha256):
            raise CheckpointError("policy warm-start value-head digest is invalid")
        if type(self.value_head_seed) is not int or self.value_head_seed < 0:
            raise CheckpointError("policy warm-start value-head seed is invalid")

    def to_manifest_dict(self) -> dict[str, object]:
        return {
            "schema_version": "lunar-policy-warm-start/v1",
            "mode": "policy-only",
            "warm_start_parent_checkpoint_sha256": (
                self.parent_checkpoint_sha256
            ),
            "warm_start_parent_payload_sha256": self.parent_payload_sha256,
            "warm_start_parent_global_step": self.parent_global_step,
            "loaded_parameter_prefixes": list(self.loaded_prefixes),
            "value_head_reinitialized": True,
            "value_head_reinitialization_sha256": (
                self.value_head_reinitialization_sha256
            ),
            "value_head_seed": self.value_head_seed,
            "fresh_training_state": {
                "global_step": 0,
                "optimizer": True,
                "scheduler": True,
                "normalization": True,
                "rng": True,
                "worker_episode_state": True,
                "metrics_journal": True,
            },
        }


@dataclass(frozen=True, slots=True)
class TrainingCheckpointV6:
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
    environment_state: dict[str, object]
    run_identity: RunIdentity
    config_hash: str
    source_commit: str
    consumed_gpu_seconds: float
    budget_extension_blocks: int
    total_gpu_budget_seconds: float
    worker_allocation: dict[str, int]
    micro_batch_size: int
    latest_checkpoint_gpu_seconds: float
    candidate_checkpoint_gpu_seconds: float
    payload_sha256: str


@dataclass(frozen=True, slots=True)
class TrainingCheckpointV5:
    """Read-only legacy state using ObservationContractV2."""

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
    environment_state: dict[str, object]
    run_identity: RunIdentity
    config_hash: str
    source_commit: str
    consumed_gpu_seconds: float
    budget_extension_blocks: int
    total_gpu_budget_seconds: float
    worker_allocation: dict[str, int]
    micro_batch_size: int
    latest_checkpoint_gpu_seconds: float
    candidate_checkpoint_gpu_seconds: float
    payload_sha256: str


@dataclass(frozen=True, slots=True)
class TrainingCheckpointV4:
    """Read-only legacy state without exact environment episode cursors."""

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
    run_identity: RunIdentity
    config_hash: str
    source_commit: str
    consumed_gpu_seconds: float
    budget_extension_blocks: int
    total_gpu_budget_seconds: float
    worker_allocation: dict[str, int]
    micro_batch_size: int
    latest_checkpoint_gpu_seconds: float
    candidate_checkpoint_gpu_seconds: float
    payload_sha256: str


@dataclass(frozen=True, slots=True)
class LegacyRunIdentityV3:
    """Historical identity without an explicit training-semantics hash."""

    run_kind: str
    data_sha256: str
    split_sha256: str
    generator_sha256: str
    capability_sha256: str
    reward_sha256: str
    v3_sha256: str


@dataclass(frozen=True, slots=True)
class TrainingCheckpointV3:
    """Read-only legacy state with unknown observation/action semantics."""

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
    run_identity: LegacyRunIdentityV3
    config_hash: str
    source_commit: str
    consumed_gpu_seconds: float
    budget_extension_blocks: int
    total_gpu_budget_seconds: float
    worker_allocation: dict[str, int]
    micro_batch_size: int
    latest_checkpoint_gpu_seconds: float
    candidate_checkpoint_gpu_seconds: float
    payload_sha256: str


@dataclass(frozen=True, slots=True)
class TrainingCheckpointV2:
    """Read-only legacy state; accepted only by explicit development smoke."""

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
    budget_extension_blocks: int
    total_gpu_budget_seconds: float
    worker_allocation: dict[str, int]
    micro_batch_size: int
    latest_checkpoint_gpu_seconds: float
    candidate_checkpoint_gpu_seconds: float
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
    run_identity: RunIdentity,
    source_commit: str,
    consumed_gpu_seconds: float,
    budget_extension_blocks: int,
    total_gpu_budget_seconds: float,
    worker_allocation: Mapping[str, int],
    micro_batch_size: int,
    latest_checkpoint_gpu_seconds: float,
    candidate_checkpoint_gpu_seconds: float,
    environment_state: Mapping[str, object] | None = None,
) -> TrainingCheckpointV6:
    """Capture a complete run state with the PPO core's strict validators."""
    if not isinstance(model, nn.Module):
        raise CheckpointError("model must be a Torch module")
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise CheckpointError("optimizer must be a Torch optimizer")
    if not isinstance(scheduler, torch.optim.lr_scheduler.LRScheduler):
        raise CheckpointError("scheduler must be a Torch LR scheduler")
    if not isinstance(run_identity, RunIdentity):
        raise CheckpointError("checkpoint run identity must use RunIdentity")
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
        "environment_state": _json_copy(
            {} if environment_state is None else environment_state
        ),
        "run_identity": run_identity.to_dict(),
        "config_hash": config_sha256(frozen_config),
        "source_commit": source_commit,
        "consumed_gpu_seconds": consumed_gpu_seconds,
        "budget_extension_blocks": budget_extension_blocks,
        "total_gpu_budget_seconds": float(total_gpu_budget_seconds),
        "worker_allocation": dict(worker_allocation),
        "micro_batch_size": micro_batch_size,
        "latest_checkpoint_gpu_seconds": latest_checkpoint_gpu_seconds,
        "candidate_checkpoint_gpu_seconds": candidate_checkpoint_gpu_seconds,
    }
    _validate_body(body)
    return _checkpoint_from_body(body, _semantic_sha256(body))


def migrate_checkpoint_source_commit(
    checkpoint: TrainingCheckpointV6,
    *,
    expected_source_commit: str,
    new_source_commit: str,
) -> TrainingCheckpointV6:
    """Rebind one validated checkpoint after an explicit in-line code repair.

    The caller remains responsible for preserving the original checkpoint and
    recording operational evidence.  This pure transform changes no model,
    optimizer, RNG, environment, progress, budget, or identity field.
    """
    if not isinstance(checkpoint, TrainingCheckpointV6):
        raise CheckpointError("source migration requires a v6 checkpoint")
    if checkpoint.source_commit != expected_source_commit:
        raise CheckpointError("checkpoint source commit differs from migration")
    if not _is_source_commit(new_source_commit):
        raise CheckpointError("new checkpoint source commit is invalid")
    if new_source_commit == expected_source_commit:
        raise CheckpointError("checkpoint source commit did not change")
    body = _body_from_checkpoint(checkpoint)
    body["source_commit"] = new_source_commit
    _validate_body(body)
    return _checkpoint_from_body(body, _semantic_sha256(body))


def save_checkpoint_atomic(
    path: str | Path,
    checkpoint: TrainingCheckpointV6,
    *,
    overwrite: bool = True,
) -> None:
    """Flush one same-directory temporary file before atomically replacing target."""
    if not isinstance(checkpoint, TrainingCheckpointV6):
        raise CheckpointError("checkpoint must use TrainingCheckpointV6")
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


def load_checkpoint(
    path: str | Path, *, run_kind: str | None = None
) -> (
    TrainingCheckpointV6
    | TrainingCheckpointV5
    | TrainingCheckpointV4
    | TrainingCheckpointV3
    | TrainingCheckpointV2
):
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
    schema = body.get("schema_version") if isinstance(body, Mapping) else None
    if schema == CHECKPOINT_SCHEMA_VERSION:
        _validate_body(body)
    elif schema == _LEGACY_V5_SCHEMA_VERSION:
        _require_explicit_legacy_development_reader(run_kind, "v5")
        _validate_v5_body(body)
    elif schema == _LEGACY_V4_SCHEMA_VERSION:
        _require_explicit_legacy_development_reader(run_kind, "v4")
        _validate_v4_body(body)
    elif schema == _LEGACY_V3_SCHEMA_VERSION:
        _require_explicit_legacy_development_reader(run_kind, "v3")
        _validate_v3_body(body)
    elif schema == _LEGACY_V2_SCHEMA_VERSION:
        _require_explicit_legacy_development_reader(run_kind, "v2")
        _validate_v2_body(body)
    else:
        raise CheckpointError("checkpoint schema version mismatch")
    if not _is_sha256(payload_sha256):
        raise CheckpointError("checkpoint payload hash is invalid")
    if _semantic_sha256(body) != payload_sha256:
        raise CheckpointError("checkpoint payload hash mismatch")
    if schema == _LEGACY_V3_SCHEMA_VERSION:
        return _checkpoint_v3_from_body(body, payload_sha256)
    if schema == _LEGACY_V2_SCHEMA_VERSION:
        return _checkpoint_v2_from_body(body, payload_sha256)
    if schema == _LEGACY_V4_SCHEMA_VERSION:
        return _checkpoint_v4_from_body(body, payload_sha256)
    if schema == _LEGACY_V5_SCHEMA_VERSION:
        return _checkpoint_v5_from_body(body, payload_sha256)
    return _checkpoint_from_body(body, payload_sha256)


def load_policy_warm_start(
    path: str | Path,
    model: nn.Module,
    *,
    value_head_seed: int,
) -> PolicyWarmStartEvidence:
    """Import only the approved policy tensors from an inert formal parent."""
    if not isinstance(model, nn.Module):
        raise CheckpointError("policy warm-start target must be a Torch module")
    if type(value_head_seed) is not int or value_head_seed < 0:
        raise CheckpointError("policy warm-start value-head seed is invalid")
    target_state = model.state_dict()
    if not target_state or any(
        not isinstance(name, str) or not isinstance(value, torch.Tensor)
        for name, value in target_state.items()
    ):
        raise CheckpointError("policy warm-start target state is invalid")
    if any(value.device.type != "cpu" for value in target_state.values()):
        raise CheckpointError("policy warm-start target must remain on CPU")

    allowed_names = {
        name
        for name in target_state
        if any(
            name.startswith(f"{prefix}.")
            for prefix in POLICY_WARM_START_PREFIXES
        )
    }
    value_names = {
        name
        for name in target_state
        if name.startswith(f"{_POLICY_WARM_START_EXCLUDED_PREFIX}.")
    }
    if (
        allowed_names | value_names != set(target_state)
        or allowed_names & value_names
        or not value_names
        or any(
            not any(name.startswith(f"{prefix}.") for name in allowed_names)
            for prefix in POLICY_WARM_START_PREFIXES
        )
    ):
        raise CheckpointError(
            "policy warm-start target architecture is outside the allow-list"
        )

    body, parent_file_sha256, parent_payload_sha256 = (
        _load_policy_warm_start_parent(path)
    )
    parent_state = body["model_state"]
    if set(parent_state) != set(target_state) or any(
        not isinstance(name, str) for name in parent_state
    ):
        raise CheckpointError("policy warm-start model keys differ")
    for name, target_value in target_state.items():
        parent_value = parent_state[name]
        if not isinstance(parent_value, torch.Tensor):
            raise CheckpointError("policy warm-start model tensor is invalid")
        if parent_value.layout != target_value.layout:
            raise CheckpointError("policy warm-start tensor layout differs")
        if parent_value.shape != target_value.shape:
            raise CheckpointError("policy warm-start tensor shape differs")
        if parent_value.dtype != target_value.dtype:
            raise CheckpointError("policy warm-start tensor dtype differs")

    live_state = _cpu_copy(target_state)
    try:
        with torch.no_grad(), torch.random.fork_rng(devices=[]):
            for name in sorted(allowed_names):
                target_state[name].copy_(parent_state[name])
            torch.random.default_generator.manual_seed(value_head_seed)
            value_module = getattr(
                model, _POLICY_WARM_START_EXCLUDED_PREFIX, None
            )
            if not isinstance(value_module, nn.Module):
                raise CheckpointError(
                    "policy warm-start value head is unavailable"
                )
            for module in value_module.modules():
                reset_parameters = getattr(module, "reset_parameters", None)
                if callable(reset_parameters):
                    reset_parameters()
        loaded_state = model.state_dict()
        _validate_finite_tensors(
            loaded_state, state_name="policy warm-start model"
        )
        value_digest = _semantic_sha256(
            {
                name: loaded_state[name].detach().cpu().clone()
                for name in sorted(value_names)
            }
        )
    except Exception as error:
        try:
            model.load_state_dict(dict(live_state), strict=True)
        except Exception as rollback_error:
            raise CheckpointError(
                "policy warm-start rollback failed"
            ) from rollback_error
        if isinstance(error, CheckpointError):
            raise
        raise CheckpointError("policy warm-start could not be applied") from error

    return PolicyWarmStartEvidence(
        parent_checkpoint_sha256=parent_file_sha256,
        parent_payload_sha256=parent_payload_sha256,
        parent_global_step=body["global_step"],
        loaded_prefixes=POLICY_WARM_START_PREFIXES,
        value_head_reinitialization_sha256=value_digest,
        value_head_seed=value_head_seed,
    )


def _load_policy_warm_start_parent(
    path: str | Path,
) -> tuple[Mapping[object, object], str, str]:
    """Read old formal state without admitting its episode state to resume."""
    target = Path(path)
    if target.is_symlink() or not target.is_file():
        raise CheckpointError(
            "policy warm-start checkpoint must be an existing regular file"
        )
    try:
        checkpoint_bytes = target.read_bytes()
        payload = torch.load(
            io.BytesIO(checkpoint_bytes),
            map_location="cpu",
            weights_only=True,
        )
    except Exception as error:
        raise CheckpointError(
            "policy warm-start restricted loader rejected payload"
        ) from error
    if not isinstance(payload, Mapping) or set(payload) != {
        "body",
        "body_sha256",
    }:
        raise CheckpointError("policy warm-start payload structure is invalid")
    body = payload["body"]
    payload_sha256 = payload["body_sha256"]
    if not isinstance(body, Mapping) or set(body) != _BODY_FIELDS:
        raise CheckpointError("policy warm-start body structure is invalid")
    schema = body.get("schema_version")
    contract = body.get("contract_version")
    if schema not in (CHECKPOINT_SCHEMA_VERSION, _LEGACY_V5_SCHEMA_VERSION):
        raise CheckpointError("policy warm-start checkpoint schema is unsupported")
    if (
        schema == CHECKPOINT_SCHEMA_VERSION
        and contract != OBSERVATION_CONTRACT_VERSION
    ) or (
        schema == _LEGACY_V5_SCHEMA_VERSION
        and contract != _LEGACY_V2_OBSERVATION_CONTRACT_VERSION
    ):
        raise CheckpointError("policy warm-start observation contract differs")
    if not _is_sha256(payload_sha256):
        raise CheckpointError("policy warm-start payload hash is invalid")
    try:
        computed_payload_sha256 = _semantic_sha256(body)
    except CheckpointError as error:
        raise CheckpointError("policy warm-start payload cannot be hashed") from error
    if computed_payload_sha256 != payload_sha256:
        raise CheckpointError("policy warm-start payload hash mismatch")
    identity = _run_identity_from_mapping(body.get("run_identity"))
    if identity.run_kind != "formal":
        raise CheckpointError("policy warm-start parent must be formal")
    if identity.reward_sha256 != reward_weights_sha256():
        raise CheckpointError("policy warm-start parent reward identity differs")
    if (
        identity.training_semantics_sha256
        != training_semantics_sha256()
    ):
        raise CheckpointError(
            "policy warm-start parent training semantics differ"
        )
    if type(body.get("global_step")) is not int or body["global_step"] < 0:
        raise CheckpointError("policy warm-start parent step is invalid")
    model_state = body.get("model_state")
    if not isinstance(model_state, Mapping):
        raise CheckpointError("policy warm-start model state is invalid")
    _validate_finite_tensors(
        model_state, state_name="policy warm-start model"
    )
    return (
        body,
        hashlib.sha256(checkpoint_bytes).hexdigest(),
        payload_sha256,
    )


def _require_explicit_legacy_development_reader(
    run_kind: str | None, version: str
) -> None:
    if run_kind is None:
        raise CheckpointError(
            f"{version} checkpoint requires explicit development-smoke run kind"
        )
    if run_kind == "formal":
        raise CheckpointError(f"formal run rejects {version} checkpoint")
    if run_kind != "development-smoke":
        raise CheckpointError("checkpoint run kind is invalid")


def load_checkpoint_for_resume(
    path: str | Path,
    *,
    expected_contract_version: str,
    expected_config_hash: str,
    expected_source_commit: str,
    expected_run_identity: RunIdentity,
    expected_worker_allocation: Mapping[str, int] | None = None,
    expected_micro_batch_size: int | None = None,
    expected_budget_extension_blocks: int | None = None,
    expected_total_gpu_budget_seconds: float | None = None,
) -> TrainingCheckpointV6:
    """Reject any run identity drift before live state can be mutated."""
    if not isinstance(expected_run_identity, RunIdentity):
        raise CheckpointError("expected run identity must use RunIdentity")
    checkpoint = load_checkpoint(path, run_kind=expected_run_identity.run_kind)
    if not isinstance(checkpoint, TrainingCheckpointV6):
        raise CheckpointError("legacy checkpoint is read-only and cannot resume")
    if checkpoint.contract_version != expected_contract_version:
        raise CheckpointError("checkpoint contract version mismatch")
    if checkpoint.config_hash != expected_config_hash:
        raise CheckpointError("checkpoint config hash mismatch")
    if checkpoint.source_commit != expected_source_commit:
        raise CheckpointError("checkpoint source commit mismatch")
    for field in _RUN_IDENTITY_FIELDS:
        if getattr(checkpoint.run_identity, field) != getattr(
            expected_run_identity, field
        ):
            raise CheckpointError(
                f"checkpoint {field.replace('_', ' ')} mismatch"
            )
    if (
        expected_worker_allocation is not None
        and checkpoint.worker_allocation != dict(expected_worker_allocation)
    ):
        raise CheckpointError("checkpoint worker allocation mismatch")
    if (
        expected_micro_batch_size is not None
        and checkpoint.micro_batch_size != expected_micro_batch_size
    ):
        raise CheckpointError("checkpoint micro-batch mismatch")
    if (expected_budget_extension_blocks is None) != (
        expected_total_gpu_budget_seconds is None
    ):
        raise CheckpointError("expected budget identity is incomplete")
    if expected_budget_extension_blocks is not None:
        if (
            type(expected_budget_extension_blocks) is not int
            or expected_budget_extension_blocks < 0
            or not isinstance(expected_total_gpu_budget_seconds, (int, float))
            or isinstance(expected_total_gpu_budget_seconds, bool)
            or not math.isfinite(float(expected_total_gpu_budget_seconds))
            or float(expected_total_gpu_budget_seconds)
            != INITIAL_GPU_BUDGET_SECONDS
            + expected_budget_extension_blocks * BUDGET_EXTENSION_BLOCK_SECONDS
        ):
            raise CheckpointError("expected budget identity is invalid")
        if checkpoint.budget_extension_blocks > expected_budget_extension_blocks:
            raise CheckpointError("checkpoint budget exceeds run manifest budget")
    return checkpoint


def restore_training_state(
    checkpoint: TrainingCheckpointV6,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    *,
    normalization_state: MutableMapping[object, object] | None = None,
) -> None:
    """Restore complete train/RNG state, rolling live objects back on failure."""
    if not isinstance(checkpoint, TrainingCheckpointV6):
        raise CheckpointError("checkpoint must use TrainingCheckpointV6")
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
    if normalization_state is not None and not isinstance(
        normalization_state, MutableMapping
    ):
        raise CheckpointError("live normalization state must be mutable")
    live_normalization = (
        _cpu_copy(dict(normalization_state))
        if normalization_state is not None
        else None
    )
    try:
        model.load_state_dict(dict(checkpoint.model_state), strict=True)
        optimizer.load_state_dict(dict(checkpoint.optimizer_state))
        scheduler.load_state_dict(dict(checkpoint.scheduler_state))
        if normalization_state is not None:
            normalization_state.clear()
            normalization_state.update(_cpu_copy(dict(checkpoint.normalization)))
        _restore_rng_state(checkpoint.rng_state)
    except Exception as error:
        try:
            model.load_state_dict(dict(live_model), strict=True)
            optimizer.load_state_dict(dict(live_optimizer))
            scheduler.load_state_dict(dict(live_scheduler))
            if normalization_state is not None:
                normalization_state.clear()
                normalization_state.update(live_normalization)
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
    run_identity = _run_identity_from_mapping(body["run_identity"])
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
    blocks = body["budget_extension_blocks"]
    total = body["total_gpu_budget_seconds"]
    if type(blocks) is not int or blocks < 0:
        raise CheckpointError("checkpoint budget extension blocks are invalid")
    expected_total = (
        INITIAL_GPU_BUDGET_SECONDS + blocks * BUDGET_EXTENSION_BLOCK_SECONDS
    )
    if (
        not isinstance(total, (int, float))
        or isinstance(total, bool)
        or not math.isfinite(float(total))
        or float(total) != expected_total
        or float(consumed) > float(total)
    ):
        raise CheckpointError("checkpoint total GPU budget is invalid")
    allocation = body["worker_allocation"]
    if (
        not isinstance(allocation, Mapping)
        or not allocation
        or not set(allocation) <= {"WHEELED", "LEGGED", "HOPPER"}
        or any(type(value) is not int or value <= 0 for value in allocation.values())
        or sum(allocation.values()) not in (18, 24)
    ):
        raise CheckpointError("checkpoint worker allocation is invalid")
    _validate_environment_state(
        body["environment_state"],
        run_kind=run_identity.run_kind,
        worker_count=sum(allocation.values()),
    )
    if type(body["micro_batch_size"]) is not int or body["micro_batch_size"] <= 0:
        raise CheckpointError("checkpoint micro-batch is invalid")
    for name in (
        "latest_checkpoint_gpu_seconds",
        "candidate_checkpoint_gpu_seconds",
    ):
        marker = body[name]
        if (
            not isinstance(marker, (int, float))
            or isinstance(marker, bool)
            or not math.isfinite(float(marker))
            or marker < 0.0
            or marker > float(consumed)
        ):
            raise CheckpointError(f"checkpoint {name} is invalid")


def _validate_environment_state(
    value: object,
    *,
    run_kind: str,
    worker_count: int,
) -> None:
    if value == {} and run_kind == "development-smoke":
        return
    required = {
        "schema_version",
        "scenario_schedule_id",
        "worker_episode_states",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise CheckpointError("checkpoint environment state structure is invalid")
    if value["schema_version"] != FORMAL_ENVIRONMENT_STATE_SCHEMA_VERSION:
        raise CheckpointError("checkpoint environment state schema is invalid")
    schedule_id = value["scenario_schedule_id"]
    if not isinstance(schedule_id, str) or not schedule_id:
        raise CheckpointError("checkpoint environment state schedule is invalid")
    states = value["worker_episode_states"]
    if not isinstance(states, list) or len(states) != worker_count:
        raise CheckpointError("checkpoint environment worker states are invalid")
    try:
        parsed = tuple(FormalWorkerState.from_dict(state) for state in states)
    except ValueError as error:
        raise CheckpointError("checkpoint environment worker state is invalid") from error
    if (
        tuple(state.worker_index for state in parsed)
        != tuple(range(worker_count))
        or any(state.scenario_schedule_id != schedule_id for state in parsed)
    ):
        raise CheckpointError("checkpoint environment worker identity is invalid")


def _validate_v2_body(body: object) -> None:
    if not isinstance(body, Mapping) or set(body) != _V2_BODY_FIELDS:
        raise CheckpointError("v2 checkpoint body structure is invalid")
    if body["contract_version"] not in (
        OBSERVATION_CONTRACT_VERSION,
        _LEGACY_V2_OBSERVATION_CONTRACT_VERSION,
        _LEGACY_OBSERVATION_CONTRACT_VERSION,
    ):
        raise CheckpointError("checkpoint contract version mismatch")
    promoted = dict(body)
    promoted["schema_version"] = CHECKPOINT_SCHEMA_VERSION
    promoted["contract_version"] = OBSERVATION_CONTRACT_VERSION
    promoted["run_identity"] = RunIdentity(
        run_kind="development-smoke",
        data_sha256="0" * 64,
        split_sha256="0" * 64,
        generator_sha256="0" * 64,
        capability_sha256="0" * 64,
        reward_sha256="0" * 64,
        v3_sha256="0" * 64,
        training_semantics_sha256="0" * 64,
    ).to_dict()
    promoted["environment_state"] = {}
    _validate_body(promoted)


def _validate_v5_body(body: object) -> None:
    if not isinstance(body, Mapping) or set(body) != _BODY_FIELDS:
        raise CheckpointError("v5 checkpoint body structure is invalid")
    if body["schema_version"] != _LEGACY_V5_SCHEMA_VERSION:
        raise CheckpointError("v5 checkpoint schema version mismatch")
    if body["contract_version"] != _LEGACY_V2_OBSERVATION_CONTRACT_VERSION:
        raise CheckpointError("v5 checkpoint contract version mismatch")
    promoted = dict(body)
    promoted["schema_version"] = CHECKPOINT_SCHEMA_VERSION
    promoted["contract_version"] = OBSERVATION_CONTRACT_VERSION
    _validate_body(promoted)


def _validate_v4_body(body: object) -> None:
    if not isinstance(body, Mapping) or set(body) != _V4_BODY_FIELDS:
        raise CheckpointError("v4 checkpoint body structure is invalid")
    if body["schema_version"] != _LEGACY_V4_SCHEMA_VERSION:
        raise CheckpointError("v4 checkpoint schema version mismatch")
    if body["contract_version"] not in (
        OBSERVATION_CONTRACT_VERSION,
        _LEGACY_V2_OBSERVATION_CONTRACT_VERSION,
    ):
        raise CheckpointError("v4 checkpoint contract version mismatch")
    promoted = dict(body)
    promoted["schema_version"] = CHECKPOINT_SCHEMA_VERSION
    promoted["contract_version"] = OBSERVATION_CONTRACT_VERSION
    promoted["environment_state"] = {}
    _validate_body(promoted)


def _validate_v3_body(body: object) -> None:
    if not isinstance(body, Mapping) or set(body) != _V4_BODY_FIELDS:
        raise CheckpointError("v3 checkpoint body structure is invalid")
    if body["contract_version"] not in (
        OBSERVATION_CONTRACT_VERSION,
        _LEGACY_V2_OBSERVATION_CONTRACT_VERSION,
        _LEGACY_OBSERVATION_CONTRACT_VERSION,
    ):
        raise CheckpointError("checkpoint contract version mismatch")
    legacy_identity = _legacy_v3_run_identity_from_mapping(body["run_identity"])
    promoted = dict(body)
    promoted["schema_version"] = CHECKPOINT_SCHEMA_VERSION
    promoted["contract_version"] = OBSERVATION_CONTRACT_VERSION
    promoted["run_identity"] = {
        **{
            field: getattr(legacy_identity, field)
            for field in _LEGACY_V3_RUN_IDENTITY_FIELDS
        },
        "training_semantics_sha256": "0" * 64,
    }
    promoted["environment_state"] = {}
    _validate_body(promoted)


def _checkpoint_from_body(
    body: Mapping[object, object], payload_sha256: str
) -> TrainingCheckpointV6:
    return TrainingCheckpointV6(
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
        environment_state=dict(body["environment_state"]),
        run_identity=_run_identity_from_mapping(body["run_identity"]),
        config_hash=body["config_hash"],
        source_commit=body["source_commit"],
        consumed_gpu_seconds=float(body["consumed_gpu_seconds"]),
        budget_extension_blocks=body["budget_extension_blocks"],
        total_gpu_budget_seconds=float(body["total_gpu_budget_seconds"]),
        worker_allocation=dict(body["worker_allocation"]),
        micro_batch_size=body["micro_batch_size"],
        latest_checkpoint_gpu_seconds=float(
            body["latest_checkpoint_gpu_seconds"]
        ),
        candidate_checkpoint_gpu_seconds=float(
            body["candidate_checkpoint_gpu_seconds"]
        ),
        payload_sha256=payload_sha256,
    )


def _checkpoint_v5_from_body(
    body: Mapping[object, object], payload_sha256: str
) -> TrainingCheckpointV5:
    return TrainingCheckpointV5(
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
        environment_state=dict(body["environment_state"]),
        run_identity=_run_identity_from_mapping(body["run_identity"]),
        config_hash=body["config_hash"],
        source_commit=body["source_commit"],
        consumed_gpu_seconds=float(body["consumed_gpu_seconds"]),
        budget_extension_blocks=body["budget_extension_blocks"],
        total_gpu_budget_seconds=float(body["total_gpu_budget_seconds"]),
        worker_allocation=dict(body["worker_allocation"]),
        micro_batch_size=body["micro_batch_size"],
        latest_checkpoint_gpu_seconds=float(
            body["latest_checkpoint_gpu_seconds"]
        ),
        candidate_checkpoint_gpu_seconds=float(
            body["candidate_checkpoint_gpu_seconds"]
        ),
        payload_sha256=payload_sha256,
    )


def _checkpoint_v4_from_body(
    body: Mapping[object, object], payload_sha256: str
) -> TrainingCheckpointV4:
    return TrainingCheckpointV4(
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
        run_identity=_run_identity_from_mapping(body["run_identity"]),
        config_hash=body["config_hash"],
        source_commit=body["source_commit"],
        consumed_gpu_seconds=float(body["consumed_gpu_seconds"]),
        budget_extension_blocks=body["budget_extension_blocks"],
        total_gpu_budget_seconds=float(body["total_gpu_budget_seconds"]),
        worker_allocation=dict(body["worker_allocation"]),
        micro_batch_size=body["micro_batch_size"],
        latest_checkpoint_gpu_seconds=float(
            body["latest_checkpoint_gpu_seconds"]
        ),
        candidate_checkpoint_gpu_seconds=float(
            body["candidate_checkpoint_gpu_seconds"]
        ),
        payload_sha256=payload_sha256,
    )


def _checkpoint_v3_from_body(
    body: Mapping[object, object], payload_sha256: str
) -> TrainingCheckpointV3:
    return TrainingCheckpointV3(
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
        run_identity=_legacy_v3_run_identity_from_mapping(
            body["run_identity"]
        ),
        config_hash=body["config_hash"],
        source_commit=body["source_commit"],
        consumed_gpu_seconds=float(body["consumed_gpu_seconds"]),
        budget_extension_blocks=body["budget_extension_blocks"],
        total_gpu_budget_seconds=float(body["total_gpu_budget_seconds"]),
        worker_allocation=dict(body["worker_allocation"]),
        micro_batch_size=body["micro_batch_size"],
        latest_checkpoint_gpu_seconds=float(
            body["latest_checkpoint_gpu_seconds"]
        ),
        candidate_checkpoint_gpu_seconds=float(
            body["candidate_checkpoint_gpu_seconds"]
        ),
        payload_sha256=payload_sha256,
    )


def _checkpoint_v2_from_body(
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
        budget_extension_blocks=body["budget_extension_blocks"],
        total_gpu_budget_seconds=float(body["total_gpu_budget_seconds"]),
        worker_allocation=dict(body["worker_allocation"]),
        micro_batch_size=body["micro_batch_size"],
        latest_checkpoint_gpu_seconds=float(
            body["latest_checkpoint_gpu_seconds"]
        ),
        candidate_checkpoint_gpu_seconds=float(
            body["candidate_checkpoint_gpu_seconds"]
        ),
        payload_sha256=payload_sha256,
    )


def _body_from_checkpoint(checkpoint: TrainingCheckpointV6) -> dict[str, object]:
    return {
        name: (
            checkpoint.run_identity.to_dict()
            if name == "run_identity"
            else getattr(checkpoint, name)
        )
        for name in _BODY_FIELDS
    }


def _run_identity_from_mapping(value: object) -> RunIdentity:
    if not isinstance(value, Mapping) or set(value) != set(_RUN_IDENTITY_FIELDS):
        raise CheckpointError("checkpoint run identity structure is invalid")
    return RunIdentity(**{field: value[field] for field in _RUN_IDENTITY_FIELDS})


def _legacy_v3_run_identity_from_mapping(
    value: object,
) -> LegacyRunIdentityV3:
    if not isinstance(value, Mapping) or set(value) != set(
        _LEGACY_V3_RUN_IDENTITY_FIELDS
    ):
        raise CheckpointError("v3 checkpoint run identity structure is invalid")
    identity = LegacyRunIdentityV3(
        **{
            field: value[field]
            for field in _LEGACY_V3_RUN_IDENTITY_FIELDS
        }
    )
    if identity.run_kind not in ("formal", "development-smoke"):
        raise CheckpointError("v3 checkpoint run kind is invalid")
    for field in _LEGACY_V3_RUN_IDENTITY_FIELDS[1:]:
        if not _is_sha256(getattr(identity, field)):
            raise CheckpointError(
                f"v3 checkpoint {field.replace('_', ' ')} is invalid"
            )
    return identity


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
    "OBSERVATION_CONTRACT_VERSION",
    "FORMAL_ENVIRONMENT_STATE_SCHEMA_VERSION",
    "POLICY_WARM_START_PREFIXES",
    "LegacyRunIdentityV3",
    "PolicyWarmStartEvidence",
    "RunIdentity",
    "TrainingCheckpointV2",
    "TrainingCheckpointV3",
    "TrainingCheckpointV4",
    "TrainingCheckpointV5",
    "TrainingCheckpointV6",
    "build_training_checkpoint",
    "config_sha256",
    "load_checkpoint",
    "load_checkpoint_for_resume",
    "load_policy_warm_start",
    "restore_training_state",
    "save_checkpoint_atomic",
]

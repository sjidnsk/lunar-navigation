"""Atomic complete-run checkpoints layered over the PPO core validators."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import os
import tempfile
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass, replace
from pathlib import Path

import torch
from torch import nn
from lunar_model_contract import (
    ObservationContractV2,
    ObservationContractV3,
    ObservationContractV4,
)

from .budget import (
    BUDGET_EXTENSION_BLOCK_SECONDS,
    INITIAL_GPU_BUDGET_SECONDS,
)
from .config import WORKER_CANDIDATES
from .environment.formal_episode_state import (
    FORMAL_ENVIRONMENT_STATE_SCHEMA_VERSION,
    FormalWorkerState,
)
from .environment.task_area import WorkerStratum
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
from .reward_contract import TaskScaleBucket
from .reward_curriculum import (
    RewardCurriculumState,
    TrainingStage,
    worker_allocation_for_stage,
)
from .training_semantics import training_semantics_sha256


CHECKPOINT_SCHEMA_VERSION = "lunar-ppo-checkpoint/v13"
OBSERVATION_CONTRACT_VERSION = ObservationContractV4.version
_LEGACY_V12_SCHEMA_VERSION = "lunar-ppo-checkpoint/v12"
_LEGACY_V11_SCHEMA_VERSION = "lunar-ppo-checkpoint/v11"
_LEGACY_V10_SCHEMA_VERSION = "lunar-ppo-checkpoint/v10"
_LEGACY_V9_SCHEMA_VERSION = "lunar-ppo-checkpoint/v9"
_LEGACY_V5_SCHEMA_VERSION = "lunar-ppo-checkpoint/v5"
_LEGACY_V4_SCHEMA_VERSION = "lunar-ppo-checkpoint/v4"
_LEGACY_V3_SCHEMA_VERSION = "lunar-ppo-checkpoint/v3"
_LEGACY_V2_SCHEMA_VERSION = "lunar-ppo-checkpoint/v2"
_LEGACY_OBSERVATION_CONTRACT_VERSION = "ObservationContractV1"
_LEGACY_V2_OBSERVATION_CONTRACT_VERSION = ObservationContractV2.version
_LEGACY_V9_OBSERVATION_CONTRACT_VERSION = ObservationContractV3.version
POLICY_WARM_START_PREFIXES = (
    "global_encoder",
    "local_encoder",
    "pose_encoder",
    "platform_encoder",
    "cross_attention_blocks",
    "frontier_logit_head",
    "theta_sin_head",
    "theta_cos_head",
    "theta_kappa_head",
)
POLICY_WARM_START_RESET_PREFIXES = (
    "frontier_encoder",
    "frontier_position_encoder",
    "action_output_mlp",
    "value_mlp",
)
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
_V10_BODY_FIELDS = {
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
_BODY_FIELDS = _V10_BODY_FIELDS | {
    "update_recovery_state",
    "reward_curriculum_state",
    "best_checkpoint_state",
}
_V4_BODY_FIELDS = _V10_BODY_FIELDS - {"environment_state"}
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
class UpdateRecoveryState:
    """Checkpoint-owned identity for one optimizer-applied sealed update."""

    update_id: int
    policy_version: int
    next_slot_by_worker: tuple[int, ...]
    worker_strata: tuple[WorkerStratum, ...]
    journal_state: str
    journal_sha256: str
    metrics_record: dict[str, object]
    metrics_record_sha256: str
    recovery_generation: int
    restored_from_checkpoint_sha256: str | None

    def __post_init__(self) -> None:
        if type(self.update_id) is not int or self.update_id < 0:
            raise CheckpointError("checkpoint update recovery ID is invalid")
        if type(self.policy_version) is not int or self.policy_version < 0:
            raise CheckpointError(
                "checkpoint update recovery policy version is invalid"
            )
        if (
            not isinstance(self.worker_strata, tuple)
            or not self.worker_strata
            or any(
                not isinstance(item, WorkerStratum)
                for item in self.worker_strata
            )
            or tuple(item.worker_index for item in self.worker_strata)
            != tuple(range(len(self.worker_strata)))
        ):
            raise CheckpointError(
                "checkpoint update recovery worker strata are invalid"
            )
        if (
            not isinstance(self.next_slot_by_worker, tuple)
            or len(self.next_slot_by_worker) != len(self.worker_strata)
            or any(
                type(value) is not int or value < 0
                for value in self.next_slot_by_worker
            )
        ):
            raise CheckpointError(
                "checkpoint update recovery worker cursors are invalid"
            )
        if self.journal_state not in ("SEALED", "LEGACY_COMPATIBILITY"):
            raise CheckpointError(
                "checkpoint update recovery journal state is invalid"
            )
        if not _is_sha256(self.journal_sha256):
            raise CheckpointError(
                "checkpoint update recovery journal hash is invalid"
            )
        frozen_metrics = _json_copy(self.metrics_record)
        if (
            type(frozen_metrics.get("global_step")) is not int
            or frozen_metrics["global_step"] != self.update_id
        ):
            raise CheckpointError(
                "checkpoint update recovery metrics step is invalid"
            )
        if (
            not _is_sha256(self.metrics_record_sha256)
            or _json_sha256(frozen_metrics) != self.metrics_record_sha256
        ):
            raise CheckpointError(
                "checkpoint update recovery metrics hash is invalid"
            )
        object.__setattr__(self, "metrics_record", frozen_metrics)
        if type(self.recovery_generation) is not int or self.recovery_generation < 0:
            raise CheckpointError(
                "checkpoint update recovery generation is invalid"
            )
        if self.restored_from_checkpoint_sha256 is not None and not _is_sha256(
            self.restored_from_checkpoint_sha256
        ):
            raise CheckpointError(
                "checkpoint restored-from identity is invalid"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "update_id": self.update_id,
            "policy_version": self.policy_version,
            "next_slot_by_worker": list(self.next_slot_by_worker),
            "worker_strata": [
                _worker_stratum_to_mapping(item) for item in self.worker_strata
            ],
            "journal_state": self.journal_state,
            "journal_sha256": self.journal_sha256,
            "metrics_record": _json_copy(self.metrics_record),
            "metrics_record_sha256": self.metrics_record_sha256,
            "recovery_generation": self.recovery_generation,
            "restored_from_checkpoint_sha256": (
                self.restored_from_checkpoint_sha256
            ),
        }

    def with_rollback_lineage(
        self, *, restored_from_checkpoint_sha256: str
    ) -> "UpdateRecoveryState":
        if not _is_sha256(restored_from_checkpoint_sha256):
            raise CheckpointError(
                "checkpoint rollback parent identity is invalid"
            )
        return replace(
            self,
            recovery_generation=self.recovery_generation + 1,
            restored_from_checkpoint_sha256=(
                restored_from_checkpoint_sha256
            ),
        )

    @classmethod
    def from_mapping(cls, value: object) -> "UpdateRecoveryState":
        if not isinstance(value, Mapping) or set(value) != {
            "update_id",
            "policy_version",
            "next_slot_by_worker",
            "worker_strata",
            "journal_state",
            "journal_sha256",
            "metrics_record",
            "metrics_record_sha256",
            "recovery_generation",
            "restored_from_checkpoint_sha256",
        }:
            raise CheckpointError(
                "checkpoint update recovery structure is invalid"
            )
        slots = value["next_slot_by_worker"]
        strata = value["worker_strata"]
        if not isinstance(slots, list) or not isinstance(strata, list):
            raise CheckpointError(
                "checkpoint update recovery arrays are invalid"
            )
        return cls(
            update_id=value["update_id"],
            policy_version=value["policy_version"],
            next_slot_by_worker=tuple(slots),
            worker_strata=tuple(
                _worker_stratum_from_mapping(item) for item in strata
            ),
            journal_state=value["journal_state"],
            journal_sha256=value["journal_sha256"],
            metrics_record=value["metrics_record"],
            metrics_record_sha256=value["metrics_record_sha256"],
            recovery_generation=value["recovery_generation"],
            restored_from_checkpoint_sha256=value[
                "restored_from_checkpoint_sha256"
            ],
        )


@dataclass(frozen=True, slots=True)
class PolicyWarmStartParameterEvidence:
    """One target parameter's audited warm-start disposition."""

    name: str
    status: str
    parent_sha256: str | None
    target_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise CheckpointError("policy warm-start parameter name is invalid")
        if self.status not in ("loaded", "reset", "random_fallback"):
            raise CheckpointError("policy warm-start parameter status is invalid")
        if self.parent_sha256 is not None and not _is_sha256(
            self.parent_sha256
        ):
            raise CheckpointError(
                "policy warm-start parent parameter digest is invalid"
            )
        if not _is_sha256(self.target_sha256):
            raise CheckpointError("policy warm-start target parameter digest is invalid")

    def to_manifest_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "status": self.status,
            "parent_sha256": self.parent_sha256,
            "target_sha256": self.target_sha256,
        }


@dataclass(frozen=True, slots=True)
class PolicyWarmStartEvidence:
    """Auditable evidence for a policy-only import into a fresh run."""

    parent_checkpoint_sha256: str
    parent_payload_sha256: str
    parent_global_step: int
    loaded_prefixes: tuple[str, ...]
    value_head_reinitialization_sha256: str
    value_head_seed: int
    parameter_evidence: tuple[PolicyWarmStartParameterEvidence, ...]
    parent_checkpoint_path: str
    parent_schema_version: str
    parent_contract_version: str
    parent_run_identity: Mapping[str, str]
    fallback_reason: str | None = None

    def __post_init__(self) -> None:
        if not _is_sha256(self.parent_checkpoint_sha256) or not _is_sha256(
            self.parent_payload_sha256
        ):
            raise CheckpointError("policy warm-start parent digest is invalid")
        if type(self.parent_global_step) is not int or self.parent_global_step < 0:
            raise CheckpointError("policy warm-start parent step is invalid")
        parent_path = Path(self.parent_checkpoint_path)
        if not self.parent_checkpoint_path or not parent_path.is_absolute():
            raise CheckpointError("policy warm-start parent path is invalid")
        if self.parent_schema_version not in (
            CHECKPOINT_SCHEMA_VERSION,
            _LEGACY_V12_SCHEMA_VERSION,
            _LEGACY_V11_SCHEMA_VERSION,
            _LEGACY_V10_SCHEMA_VERSION,
            _LEGACY_V9_SCHEMA_VERSION,
        ):
            raise CheckpointError("policy warm-start parent schema is invalid")
        if self.parent_contract_version not in (
            OBSERVATION_CONTRACT_VERSION,
            _LEGACY_V9_OBSERVATION_CONTRACT_VERSION,
        ):
            raise CheckpointError("policy warm-start parent contract is invalid")
        try:
            parent_identity = _run_identity_from_mapping(
                self.parent_run_identity
            ).to_dict()
        except CheckpointError as error:
            raise CheckpointError(
                "policy warm-start parent identity is invalid"
            ) from error
        object.__setattr__(self, "parent_run_identity", parent_identity)
        if (
            not isinstance(self.loaded_prefixes, tuple)
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
        if (
            not isinstance(self.parameter_evidence, tuple)
            or not self.parameter_evidence
        ):
            raise CheckpointError("policy warm-start parameter evidence is invalid")
        if any(
            not isinstance(entry, PolicyWarmStartParameterEvidence)
            for entry in self.parameter_evidence
        ) or len({entry.name for entry in self.parameter_evidence}) != len(
            self.parameter_evidence
        ):
            raise CheckpointError("policy warm-start parameter evidence is invalid")
        if self.fallback_reason is None:
            if self.loaded_prefixes != POLICY_WARM_START_PREFIXES or {
                entry.status for entry in self.parameter_evidence
            } != {"loaded", "reset"}:
                raise CheckpointError("policy warm-start loaded evidence differs")
        elif (
            not isinstance(self.fallback_reason, str)
            or not self.fallback_reason
            or self.loaded_prefixes
            or {entry.status for entry in self.parameter_evidence}
            != {"random_fallback"}
        ):
            raise CheckpointError("policy warm-start fallback evidence differs")

    def to_manifest_dict(self) -> dict[str, object]:
        return {
            "schema_version": "lunar-policy-warm-start/v2",
            "mode": (
                "random-initialization"
                if self.fallback_reason is not None
                else "policy-partial"
            ),
            "warm_start_parent_checkpoint_sha256": (
                self.parent_checkpoint_sha256
            ),
            "warm_start_parent_payload_sha256": self.parent_payload_sha256,
            "warm_start_parent_global_step": self.parent_global_step,
            "warm_start_parent_path": self.parent_checkpoint_path,
            "warm_start_parent_schema_version": self.parent_schema_version,
            "warm_start_parent_contract_version": self.parent_contract_version,
            "warm_start_parent_run_identity": dict(
                self.parent_run_identity
            ),
            "loaded_parameter_prefixes": list(self.loaded_prefixes),
            "reset_parameter_prefixes": list(
                POLICY_WARM_START_RESET_PREFIXES
            ),
            "parameter_evidence": [
                entry.to_manifest_dict()
                for entry in self.parameter_evidence
            ],
            "fallback_reason": self.fallback_reason,
            "value_head_reinitialized": True,
            "value_head_reinitialization_sha256": (
                self.value_head_reinitialization_sha256
            ),
            "value_head_seed": self.value_head_seed,
            "fresh_training_state": {
                "global_step": 0,
                "candidate_encoder_input": True,
                "value_head": True,
                "optimizer": True,
                "scheduler": True,
                "normalization": True,
                "gae_rollout": True,
                "rng": True,
                "worker_episode_state": True,
                "hopper_private_observation_buffer": True,
                "candidate_suppression": True,
                "reward_stage_gate": True,
                "metrics_journal": True,
            },
            "compatibility_checks": {
                "formal_parent": True,
                "supported_parent_schema": True,
                "observation_contract_supported": True,
                "policy_architecture_compatible": self.fallback_reason is None,
                "policy_allowlist_only": True,
                "training_state_reset": True,
                "parent_reward_identity_reused": False,
                "parent_training_semantics_reused": False,
            },
        }

    @property
    def value_head_reset(self) -> bool:
        return True

    @property
    def optimizer_loaded(self) -> bool:
        return False


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
    update_recovery_state: UpdateRecoveryState
    reward_curriculum_state: dict[str, object]
    best_checkpoint_state: dict[str, object]
    payload_sha256: str


@dataclass(frozen=True, slots=True)
class TrainingCheckpointV10:
    """Read-only pre-Reward-V4 checkpoint; never accepted for strict resume."""

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
    update_recovery_state: UpdateRecoveryState | None = None,
    reward_curriculum_state: (
        RewardCurriculumState | Mapping[str, object] | None
    ) = None,
    best_checkpoint_state: Mapping[str, object] | None = None,
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
    if update_recovery_state is None:
        if "reward_v4" in frozen_config:
            raise CheckpointError(
                "Reward V4 checkpoint requires update recovery state"
            )
        update_recovery_state = _legacy_compatibility_recovery_state(
            global_step=global_step,
            worker_allocation=worker_allocation,
        )
    if not isinstance(update_recovery_state, UpdateRecoveryState):
        raise CheckpointError(
            "checkpoint update recovery state must use UpdateRecoveryState"
        )
    if isinstance(reward_curriculum_state, RewardCurriculumState):
        curriculum_state = reward_curriculum_state.to_dict()
    else:
        curriculum_state = _json_copy(
            reward_curriculum_state
            or {
                "schema_version": "lunar-reward-curriculum-state/v1",
                "state": "LEGACY_COMPATIBILITY",
            }
        )
    best_state = _json_copy(
        best_checkpoint_state
        or {
            "schema_version": "lunar-best-checkpoint-state/v1",
            "state": "LEGACY_COMPATIBILITY",
        }
    )
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
        "update_recovery_state": update_recovery_state.to_dict(),
        "reward_curriculum_state": curriculum_state,
        "best_checkpoint_state": best_state,
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
        raise CheckpointError("source migration requires a v7 checkpoint")
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


def replace_checkpoint_alias_atomic(
    path: str | Path,
    immutable_checkpoint: str | Path,
) -> None:
    """Atomically point a same-directory alias at one persisted checkpoint."""
    target = _validated_target(path)
    source = Path(immutable_checkpoint)
    if (
        source.is_symlink()
        or not source.is_file()
        or source.parent.resolve() != target.parent.resolve()
        or source.resolve() == target.resolve()
    ):
        raise CheckpointError("checkpoint alias source is invalid")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".link",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.unlink()
        os.link(source, temporary)
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
        raise CheckpointError(
            "checkpoint alias could not be replaced atomically"
        ) from error


def load_checkpoint(
    path: str | Path, *, run_kind: str | None = None
) -> (
    TrainingCheckpointV6
    | TrainingCheckpointV10
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
    elif schema == _LEGACY_V10_SCHEMA_VERSION:
        _validate_v10_body(body)
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
    if schema == _LEGACY_V10_SCHEMA_VERSION:
        return _checkpoint_v10_from_body(body, payload_sha256)
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
    reset_names = {
        name
        for name in target_state
        if any(
            name.startswith(f"{prefix}.")
            for prefix in POLICY_WARM_START_RESET_PREFIXES
        )
    }
    if (
        allowed_names | reset_names != set(target_state)
        or allowed_names & reset_names
        or not reset_names
        or any(
            not any(name.startswith(f"{prefix}.") for name in allowed_names)
            for prefix in POLICY_WARM_START_PREFIXES
        )
        or any(
            not any(name.startswith(f"{prefix}.") for name in reset_names)
            for prefix in POLICY_WARM_START_RESET_PREFIXES
        )
    ):
        raise CheckpointError(
            "policy warm-start target architecture is outside the allow-list"
        )

    body, parent_file_sha256, parent_payload_sha256 = (
        _load_policy_warm_start_parent(path)
    )
    parent_checkpoint_path = str(Path(path).resolve(strict=True))
    parent_state = body["model_state"]
    fallback_reason = _policy_warm_start_compatibility_reason(
        parent_state, target_state
    )
    if fallback_reason is not None:
        return _policy_warm_start_evidence(
            body=body,
            parent_state=parent_state,
            target_state=target_state,
            parent_file_sha256=parent_file_sha256,
            parent_payload_sha256=parent_payload_sha256,
            parent_checkpoint_path=parent_checkpoint_path,
            loaded_prefixes=(),
            reset_names=reset_names,
            value_head_seed=value_head_seed,
            fallback_reason=fallback_reason,
        )

    live_state = _cpu_copy(target_state)
    try:
        prepared_state = _cpu_copy(target_state)
        with torch.no_grad(), torch.random.fork_rng(devices=[]):
            for name in sorted(allowed_names):
                prepared_state[name] = parent_state[name].detach().cpu().clone()
            torch.random.default_generator.manual_seed(value_head_seed)
            for prefix in POLICY_WARM_START_RESET_PREFIXES:
                source_module = getattr(model, prefix, None)
                if not isinstance(source_module, nn.Module):
                    raise CheckpointError(
                        "policy warm-start reset module is unavailable"
                    )
                reset_module = copy.deepcopy(source_module).cpu()
                for module in reset_module.modules():
                    reset_parameters = getattr(
                        module, "reset_parameters", None
                    )
                    if callable(reset_parameters):
                        reset_parameters()
                reset_state = reset_module.state_dict()
                expected_names = {
                    name.removeprefix(f"{prefix}.")
                    for name in reset_names
                    if name.startswith(f"{prefix}.")
                }
                if set(reset_state) != expected_names:
                    raise CheckpointError(
                        "policy warm-start reset state differs"
                    )
                for name, value in reset_state.items():
                    prepared_state[f"{prefix}.{name}"] = (
                        value.detach().cpu().clone()
                    )
        _validate_finite_tensors(
            prepared_state, state_name="policy warm-start model"
        )
        model.load_state_dict(dict(prepared_state), strict=True)
        loaded_state = model.state_dict()
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

    return _policy_warm_start_evidence(
        body=body,
        parent_state=parent_state,
        target_state=loaded_state,
        parent_file_sha256=parent_file_sha256,
        parent_payload_sha256=parent_payload_sha256,
        parent_checkpoint_path=parent_checkpoint_path,
        loaded_prefixes=POLICY_WARM_START_PREFIXES,
        reset_names=reset_names,
        value_head_seed=value_head_seed,
        fallback_reason=None,
    )


def _policy_warm_start_compatibility_reason(
    parent_state: Mapping[object, object],
    target_state: Mapping[str, torch.Tensor],
) -> str | None:
    if set(parent_state) != set(target_state) or any(
        not isinstance(name, str) for name in parent_state
    ):
        return "model keys differ"
    for name, target_value in target_state.items():
        parent_value = parent_state[name]
        if not isinstance(parent_value, torch.Tensor):
            return f"model tensor is invalid: {name}"
        if parent_value.layout != target_value.layout:
            return f"tensor layout differs: {name}"
        if parent_value.shape != target_value.shape:
            return f"tensor shape differs: {name}"
        if parent_value.dtype != target_value.dtype:
            return f"tensor dtype differs: {name}"
    return None


def _tensor_sha256(value: torch.Tensor) -> str:
    return _semantic_sha256({"value": value.detach().cpu().clone()})


def _policy_warm_start_evidence(
    *,
    body: Mapping[object, object],
    parent_state: Mapping[object, object],
    target_state: Mapping[str, torch.Tensor],
    parent_file_sha256: str,
    parent_payload_sha256: str,
    parent_checkpoint_path: str,
    loaded_prefixes: tuple[str, ...],
    reset_names: set[str],
    value_head_seed: int,
    fallback_reason: str | None,
) -> PolicyWarmStartEvidence:
    status_by_name = {
        name: (
            "random_fallback"
            if fallback_reason is not None
            else ("reset" if name in reset_names else "loaded")
        )
        for name in target_state
    }
    parameter_evidence = tuple(
        PolicyWarmStartParameterEvidence(
            name=name,
            status=status_by_name[name],
            parent_sha256=(
                _tensor_sha256(parent_state[name])
                if isinstance(parent_state.get(name), torch.Tensor)
                else None
            ),
            target_sha256=_tensor_sha256(target_state[name]),
        )
        for name in sorted(target_state)
    )
    value_names = sorted(
        name for name in target_state if name.startswith("value_mlp.")
    )
    return PolicyWarmStartEvidence(
        parent_checkpoint_sha256=parent_file_sha256,
        parent_payload_sha256=parent_payload_sha256,
        parent_global_step=body["global_step"],
        loaded_prefixes=loaded_prefixes,
        value_head_reinitialization_sha256=_semantic_sha256(
            {
                name: target_state[name].detach().cpu().clone()
                for name in value_names
            }
        ),
        value_head_seed=value_head_seed,
        parameter_evidence=parameter_evidence,
        parent_checkpoint_path=parent_checkpoint_path,
        parent_schema_version=body["schema_version"],
        parent_contract_version=body["contract_version"],
        parent_run_identity=dict(body["run_identity"]),
        fallback_reason=fallback_reason,
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
    if not isinstance(body, Mapping):
        raise CheckpointError("policy warm-start body structure is invalid")
    schema = body.get("schema_version")
    if schema not in (
        CHECKPOINT_SCHEMA_VERSION,
        _LEGACY_V12_SCHEMA_VERSION,
        _LEGACY_V11_SCHEMA_VERSION,
        _LEGACY_V10_SCHEMA_VERSION,
        _LEGACY_V9_SCHEMA_VERSION,
    ):
        raise CheckpointError("policy warm-start checkpoint schema is unsupported")
    expected_fields = (
        _BODY_FIELDS
        if schema in (
            CHECKPOINT_SCHEMA_VERSION,
            _LEGACY_V12_SCHEMA_VERSION,
            _LEGACY_V11_SCHEMA_VERSION,
        )
        else _V10_BODY_FIELDS
    )
    if set(body) != expected_fields:
        raise CheckpointError("policy warm-start body structure is invalid")
    contract = body.get("contract_version")
    if (
        schema
        in (
            CHECKPOINT_SCHEMA_VERSION,
            _LEGACY_V12_SCHEMA_VERSION,
            _LEGACY_V11_SCHEMA_VERSION,
            _LEGACY_V10_SCHEMA_VERSION,
        )
        and contract != OBSERVATION_CONTRACT_VERSION
    ) or (
        schema == _LEGACY_V9_SCHEMA_VERSION
        and contract != _LEGACY_V9_OBSERVATION_CONTRACT_VERSION
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
    # Reward/training identity deliberately does not authorize reuse here.
    # It is recorded in PolicyWarmStartEvidence while only the allow-listed
    # policy tensors cross the new-run boundary.
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
        raise CheckpointError(
            "legacy checkpoint is read-only and cannot strict resume"
        )
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
    body = _body_from_checkpoint(checkpoint)
    _validate_body(body)
    if _semantic_sha256(body) != checkpoint.payload_sha256:
        raise CheckpointError("checkpoint payload hash mismatch")
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


def _json_sha256(value: Mapping[str, object]) -> str:
    copied = _json_copy(value)
    encoded = json.dumps(
        copied,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _worker_stratum_to_mapping(value: WorkerStratum) -> dict[str, object]:
    if not isinstance(value, WorkerStratum):
        raise CheckpointError("checkpoint worker stratum is invalid")
    return {
        "worker_index": value.worker_index,
        "platform_type": value.platform_type,
        "platform_worker_index": value.platform_worker_index,
        "platform_worker_count": value.platform_worker_count,
        "scale_bucket": value.scale_bucket.value,
    }


def _worker_stratum_from_mapping(value: object) -> WorkerStratum:
    if not isinstance(value, Mapping) or set(value) != {
        "worker_index",
        "platform_type",
        "platform_worker_index",
        "platform_worker_count",
        "scale_bucket",
    }:
        raise CheckpointError("checkpoint worker stratum structure is invalid")
    try:
        result = WorkerStratum(
            worker_index=value["worker_index"],
            platform_type=value["platform_type"],
            platform_worker_index=value["platform_worker_index"],
            platform_worker_count=value["platform_worker_count"],
            scale_bucket=TaskScaleBucket(value["scale_bucket"]),
        )
    except (TypeError, ValueError) as error:
        raise CheckpointError("checkpoint worker stratum is invalid") from error
    if (
        type(result.worker_index) is not int
        or result.worker_index < 0
        or result.platform_type not in {"WHEELED", "LEGGED", "HOPPER"}
        or type(result.platform_worker_index) is not int
        or type(result.platform_worker_count) is not int
        or result.platform_worker_count <= 0
        or not 0
        <= result.platform_worker_index
        < result.platform_worker_count
    ):
        raise CheckpointError("checkpoint worker stratum is invalid")
    return result


def _legacy_compatibility_recovery_state(
    *,
    global_step: int,
    worker_allocation: Mapping[str, int],
) -> UpdateRecoveryState:
    strata: list[WorkerStratum] = []
    worker_index = 0
    buckets = tuple(TaskScaleBucket)
    for platform_type, count in worker_allocation.items():
        if type(count) is not int or count <= 0:
            raise CheckpointError("checkpoint worker allocation is invalid")
        for lane in range(count):
            strata.append(
                WorkerStratum(
                    worker_index=worker_index,
                    platform_type=platform_type,
                    platform_worker_index=lane,
                    platform_worker_count=count,
                    scale_bucket=buckets[
                        min(len(buckets) - 1, lane * len(buckets) // count)
                    ],
                )
            )
            worker_index += 1
    metrics = {
        "schema_version": "lunar-training-update-metrics/legacy-compatibility",
        "global_step": global_step,
    }
    return UpdateRecoveryState(
        update_id=global_step,
        policy_version=max(0, global_step - 1),
        next_slot_by_worker=(0,) * len(strata),
        worker_strata=tuple(strata),
        journal_state="LEGACY_COMPATIBILITY",
        journal_sha256="0" * 64,
        metrics_record=metrics,
        metrics_record_sha256=_json_sha256(metrics),
        recovery_generation=0,
        restored_from_checkpoint_sha256=None,
    )


def _add_legacy_v11_fields(body: dict[object, object]) -> None:
    recovery = _legacy_compatibility_recovery_state(
        global_step=body["global_step"],
        worker_allocation=body["worker_allocation"],
    )
    body["update_recovery_state"] = recovery.to_dict()
    body["reward_curriculum_state"] = {
        "schema_version": "lunar-reward-curriculum-state/v1",
        "state": "LEGACY_COMPATIBILITY",
    }
    body["best_checkpoint_state"] = {
        "schema_version": "lunar-best-checkpoint-state/v1",
        "state": "LEGACY_COMPATIBILITY",
    }


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
        or sum(allocation.values()) not in WORKER_CANDIDATES
    ):
        raise CheckpointError("checkpoint worker allocation is invalid")
    _validate_environment_state(
        body["environment_state"],
        run_kind=run_identity.run_kind,
        worker_count=sum(allocation.values()),
    )
    recovery = UpdateRecoveryState.from_mapping(body["update_recovery_state"])
    if recovery.update_id != body["global_step"]:
        raise CheckpointError(
            "checkpoint update recovery differs from global step"
        )
    if len(recovery.worker_strata) != sum(allocation.values()) or {
        platform: sum(
            item.platform_type == platform
            for item in recovery.worker_strata
        )
        for platform in allocation
    } != dict(allocation):
        raise CheckpointError(
            "checkpoint update recovery worker allocation differs"
        )
    reward_v4 = "reward_v4" in body["frozen_config"]
    if reward_v4 and recovery.journal_state != "SEALED":
        raise CheckpointError(
            "Reward V4 checkpoint requires sealed journal identity"
        )
    curriculum_value = body["reward_curriculum_state"]
    if reward_v4:
        try:
            curriculum = RewardCurriculumState.from_mapping(curriculum_value)
        except ValueError as error:
            raise CheckpointError(
                "Reward V4 checkpoint curriculum state is invalid"
            ) from error
        expected_allocation = worker_allocation_for_stage(curriculum.stage)
        metrics_phase_raw = recovery.metrics_record.get("curriculum_phase")
        transition_allocation = False
        if curriculum.worker_restart_required and isinstance(
            metrics_phase_raw, str
        ):
            try:
                metrics_phase = TrainingStage(metrics_phase_raw)
                transition_allocation = (
                    tuple(TrainingStage).index(curriculum.stage)
                    == tuple(TrainingStage).index(metrics_phase) + 1
                    and worker_allocation_for_stage(metrics_phase)
                    == dict(allocation)
                )
            except ValueError:
                transition_allocation = False
        if (
            curriculum.current_update_id != body["global_step"]
            or curriculum.stage.value != body["curriculum_phase"]
            or curriculum.recovery_generation
            != recovery.recovery_generation
            or curriculum.restored_from_checkpoint_sha256
            != recovery.restored_from_checkpoint_sha256
            or (
                expected_allocation != dict(allocation)
                and not transition_allocation
            )
        ):
            raise CheckpointError(
                "Reward V4 checkpoint curriculum identity differs"
            )
    elif not isinstance(curriculum_value, Mapping) or not curriculum_value:
        raise CheckpointError("checkpoint reward_curriculum_state is invalid")
    best_value = body["best_checkpoint_state"]
    if not isinstance(best_value, Mapping) or not best_value:
        raise CheckpointError("checkpoint best_checkpoint_state is invalid")
    _json_copy(curriculum_value)
    _json_copy(best_value)
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
    _add_legacy_v11_fields(promoted)
    _validate_body(promoted)


def _validate_v10_body(body: object) -> None:
    if not isinstance(body, Mapping) or set(body) != _V10_BODY_FIELDS:
        raise CheckpointError("v10 checkpoint body structure is invalid")
    if body["schema_version"] != _LEGACY_V10_SCHEMA_VERSION:
        raise CheckpointError("v10 checkpoint schema version mismatch")
    if body["contract_version"] != OBSERVATION_CONTRACT_VERSION:
        raise CheckpointError("v10 checkpoint contract version mismatch")
    promoted = dict(body)
    promoted["schema_version"] = CHECKPOINT_SCHEMA_VERSION
    _add_legacy_v11_fields(promoted)
    _validate_body(promoted)


def _validate_v5_body(body: object) -> None:
    if not isinstance(body, Mapping) or set(body) != _V10_BODY_FIELDS:
        raise CheckpointError("v5 checkpoint body structure is invalid")
    if body["schema_version"] != _LEGACY_V5_SCHEMA_VERSION:
        raise CheckpointError("v5 checkpoint schema version mismatch")
    if body["contract_version"] != _LEGACY_V2_OBSERVATION_CONTRACT_VERSION:
        raise CheckpointError("v5 checkpoint contract version mismatch")
    promoted = dict(body)
    promoted["schema_version"] = CHECKPOINT_SCHEMA_VERSION
    promoted["contract_version"] = OBSERVATION_CONTRACT_VERSION
    _add_legacy_v11_fields(promoted)
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
    _add_legacy_v11_fields(promoted)
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
    _add_legacy_v11_fields(promoted)
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
        update_recovery_state=UpdateRecoveryState.from_mapping(
            body["update_recovery_state"]
        ),
        reward_curriculum_state=dict(body["reward_curriculum_state"]),
        best_checkpoint_state=dict(body["best_checkpoint_state"]),
        payload_sha256=payload_sha256,
    )


def _checkpoint_v10_from_body(
    body: Mapping[object, object], payload_sha256: str
) -> TrainingCheckpointV10:
    return TrainingCheckpointV10(
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
            else checkpoint.update_recovery_state.to_dict()
            if name == "update_recovery_state"
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
    "POLICY_WARM_START_RESET_PREFIXES",
    "POLICY_WARM_START_PREFIXES",
    "LegacyRunIdentityV3",
    "PolicyWarmStartParameterEvidence",
    "PolicyWarmStartEvidence",
    "RunIdentity",
    "TrainingCheckpointV2",
    "TrainingCheckpointV3",
    "TrainingCheckpointV4",
    "TrainingCheckpointV5",
    "TrainingCheckpointV6",
    "TrainingCheckpointV10",
    "UpdateRecoveryState",
    "build_training_checkpoint",
    "config_sha256",
    "load_checkpoint",
    "load_checkpoint_for_resume",
    "load_policy_warm_start",
    "restore_training_state",
    "save_checkpoint_atomic",
]

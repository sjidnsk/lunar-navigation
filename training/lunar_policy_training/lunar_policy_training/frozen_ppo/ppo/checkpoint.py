"""Stage 4 完整、原子且可恢复的 PPO checkpoint。"""

from __future__ import annotations
from typing import TYPE_CHECKING

# Security scope: descriptor/path identity checks reject testable same-host runner
# path replacement. They do not claim protection from an administrator or kernel
# that can subvert open handles or filesystem identity reporting.

import copy
import hashlib
import io
import os
import random
import re
import stat
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch
from torch import nn

if TYPE_CHECKING:
    from typing import Any as SafetyContract
from .trainer import policy_state_sha256
if TYPE_CHECKING:
    from typing import Any as ArtifactStore
if TYPE_CHECKING:
    from typing import Any as DurableParentGuard
    from typing import Any as PathSecurityError
    from typing import Any as durable_fsync_directory
    from typing import Any as durable_makedirs
    from typing import Any as durable_mkdir
    from typing import Any as durable_rename
    from typing import Any as durable_replace
    from typing import Any as durable_rmdir
    from typing import Any as durable_unlink
    from typing import Any as guarded_file_identity
    from typing import Any as lexical_absolute
    from typing import Any as path_identity
    from typing import Any as require_plain_path


CHECKPOINT_SCHEMA_VERSION: Final = "stage4_complete_checkpoint/v1"
CHECKPOINT_MANIFEST_SCHEMA_VERSION: Final = "stage4_checkpoint_manifest/v1"
CHECKPOINT_COMPLETE_SCHEMA_VERSION: Final = "stage4_checkpoint_complete/v1"
CHECKPOINT_LATEST_SCHEMA_VERSION: Final = "stage4_checkpoint_latest/v1"
STAGE6_CHECKPOINT_SCHEMA_VERSION: Final = "stage6_complete_checkpoint/v1"
STAGE6_CHECKPOINT_MANIFEST_SCHEMA_VERSION: Final = "stage6_checkpoint_manifest/v1"
STAGE6_CHECKPOINT_COMPLETE_SCHEMA_VERSION: Final = "stage6_checkpoint_complete/v1"
STAGE6_CHECKPOINT_LATEST_SCHEMA_VERSION: Final = "stage6_checkpoint_latest/v1"
_CHECKPOINT_NAME = "checkpoint.pt"
_MANIFEST_NAME = "manifest.json"
_COMPLETE_NAME = "complete.json"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_UPDATE_DIRECTORY_RE = re.compile(r"^update-(?P<step>[0-9]{8})$")
_PENDING_DIRECTORY_RE = re.compile(
    r"^\.pending-(?P<directory>update-[0-9]{8})-[0-9a-f]{32}$"
)
_REQUIRED_VERSIONS = frozenset(
    {
        "observation_schema_version",
        "action_space_version",
        "network_architecture_version",
        "reward_version",
        "frontier_version",
        "planner_version",
    }
)
_REQUIRED_LINEAGE = frozenset(
    {
        "stage3_commit",
        "stage3_gate_sha256",
        "stage3_approval_sha256",
    }
)
_STAGE6_REQUIRED_LINEAGE = frozenset(
    {
        "stage5_commit",
        "stage5_gate_sha256",
        "stage5_manifest_sha256",
        "stage4_checkpoint_sha256",
        "stage4_policy_state_sha256",
        "source_set_sha256",
        "prospective_tree_sha256",
        "data_sha256",
        "safety_contract_sha256",
    }
)
_SAFETY_BINDING_FIELDS: Final = (
    "safety_contract",
    "safety_contract_sha256",
    "safety_contract_source",
    "safety_contract_config_sha256",
)
_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")


class CheckpointError(RuntimeError):
    """Checkpoint 不完整、损坏或与当前训练 lineage 不兼容。"""


@dataclass(frozen=True, slots=True)
class CheckpointReceipt:
    update_step: int
    directory: Path
    checkpoint_path: Path
    manifest_path: Path
    complete_marker_path: Path
    checkpoint_sha256: str
    manifest_sha256: str
    policy_state_sha256: str


@dataclass(frozen=True, slots=True)
class PreparedCheckpoint:
    """Attempt-scoped checkpoint bytes that are not yet canonically visible."""

    update_step: int
    root: Path
    pending_directory: Path
    destination: Path
    pending_checkpoint_path: Path
    pending_manifest_path: Path
    pending_complete_marker_path: Path
    checkpoint_sha256: str
    manifest_sha256: str
    complete_marker_sha256: str
    policy_state_sha256: str
    _directory_identity: tuple[int, int, int, int]
    _member_identities: tuple[
        tuple[str, tuple[int, int, int, int, int, int]], ...
    ]


@dataclass(frozen=True, slots=True)
class LoadedCheckpoint:
    update_step: int
    directory: Path
    checkpoint_sha256: str
    manifest_sha256: str
    policy_state_sha256: str
    normalization_stats: dict[str, Any]
    scenario_sampler_state: dict[str, Any]
    vector_env_states: tuple[dict[str, Any], ...]
    best_record: dict[str, Any]
    versions: dict[str, str]
    top_m_config: dict[str, Any]
    scale_profile: str
    training_config: dict[str, Any]
    config_sha256: str
    lineage: dict[str, Any]
    eval_metrics: dict[str, Any]
    safety_contract: dict[str, float] | None


@dataclass(frozen=True, slots=True)
class _ValidatedCandidate:
    directory: Path
    checkpoint_sha256: str
    manifest_sha256: str
    payload: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _CheckpointProfile:
    payload_schema: str
    manifest_schema: str
    complete_schema: str
    latest_schema: str
    required_lineage: frozenset[str]
    requires_safety_contract: bool


_STAGE4_PROFILE = _CheckpointProfile(
    CHECKPOINT_SCHEMA_VERSION,
    CHECKPOINT_MANIFEST_SCHEMA_VERSION,
    CHECKPOINT_COMPLETE_SCHEMA_VERSION,
    CHECKPOINT_LATEST_SCHEMA_VERSION,
    _REQUIRED_LINEAGE,
    False,
)
_STAGE6_PROFILE = _CheckpointProfile(
    STAGE6_CHECKPOINT_SCHEMA_VERSION,
    STAGE6_CHECKPOINT_MANIFEST_SCHEMA_VERSION,
    STAGE6_CHECKPOINT_COMPLETE_SCHEMA_VERSION,
    STAGE6_CHECKPOINT_LATEST_SCHEMA_VERSION,
    _STAGE6_REQUIRED_LINEAGE,
    True,
)


def _profile_for_schema(schema_version: str) -> _CheckpointProfile:
    if schema_version == CHECKPOINT_SCHEMA_VERSION:
        return _STAGE4_PROFILE
    if schema_version == STAGE6_CHECKPOINT_SCHEMA_VERSION:
        return _STAGE6_PROFILE
    raise CheckpointError("checkpoint schema profile is unsupported")


class CheckpointManager:
    """只暴露带 manifest 与 complete marker 的不可覆盖 update。"""

    def __init__(
        self,
        root: str | Path,
        *,
        fault_injector: Callable[[str], None] | None = None,
        schema_version: str = CHECKPOINT_SCHEMA_VERSION,
    ) -> None:
        unresolved = lexical_absolute(root)
        if os.name == "nt" and unresolved.drive.upper() != "D:":
            raise CheckpointError("Stage 4 checkpoints must be stored on the D drive")
        try:
            require_plain_path(
                unresolved,
                allow_missing=True,
                label="checkpoint root",
            )
            durable_makedirs(unresolved)
            self.root = require_plain_path(
                unresolved,
                leaf_kind="directory",
                label="checkpoint root",
            )
        except (OSError, PathSecurityError) as exc:
            raise CheckpointError(
                "checkpoint root contains a link or reparse point"
            ) from exc
        self._unresolved_root = unresolved
        self._root_identity = path_identity(unresolved)
        self._store = ArtifactStore(self.root)
        self._fault_injector = fault_injector
        self._profile = _profile_for_schema(schema_version)

    def _require_root_current(self) -> None:
        try:
            resolved = require_plain_path(
                self._unresolved_root,
                leaf_kind="directory",
                label="checkpoint root",
            )
            current = path_identity(self._unresolved_root)
        except PathSecurityError as exc:
            raise CheckpointError(
                "checkpoint root contains a link or reparse point"
            ) from exc
        if resolved != self.root or (
            current[0], current[1], current[2], current[4]
        ) != (
            self._root_identity[0],
            self._root_identity[1],
            self._root_identity[2],
            self._root_identity[4],
        ):
            raise CheckpointError("checkpoint root identity changed")

    def _require_checkpoint_directory_current(
        self,
        directory: Path,
        *,
        directory_identity: tuple[int, int, int, int],
        member_identities: Mapping[str, tuple[int, int, int, int, int, int]],
    ) -> None:
        self._require_root_current()
        try:
            current_directory = require_plain_path(
                directory,
                base=self.root,
                leaf_kind="directory",
                label="checkpoint directory",
            )
            if (
                current_directory != directory
                or _directory_path_identity(directory) != directory_identity
            ):
                raise CheckpointError("checkpoint directory identity changed")
            members = tuple(directory.iterdir())
            if {member.name for member in members} != set(member_identities):
                raise CheckpointError("checkpoint member identity changed")
            for member in members:
                current_member = require_plain_path(
                    member,
                    base=self.root,
                    leaf_kind="file",
                    label="checkpoint member",
                )
                if (
                    current_member != member
                    or _regular_path_identity(member)
                    != member_identities[member.name]
                ):
                    raise CheckpointError("checkpoint member identity changed")
        except CheckpointError:
            raise
        except (OSError, PathSecurityError) as exc:
            raise CheckpointError(
                "checkpoint directory or member identity changed"
            ) from exc
        self._require_root_current()

    def _write_latest(self, value: Mapping[str, Any]) -> None:
        payload = ArtifactStore.canonical_json_bytes(dict(value))
        temporary = self.root / f".latest-{uuid.uuid4().hex}.tmp"
        destination = self.root / "latest.json"
        temporary_identity: tuple[int, int, int, int, int, int] | None = None
        self._require_root_current()
        try:
            temporary_identity = _write_new_fsynced(temporary, payload)
            self._require_root_current()
            if _regular_path_identity(temporary) != temporary_identity:
                raise CheckpointError("checkpoint latest temporary identity changed")
            self._inject("before_latest_replace")
            self._require_root_current()
            if os.path.lexists(destination):
                require_plain_path(
                    destination,
                    base=self.root,
                    leaf_kind="file",
                    label="checkpoint latest index",
                )
            durable_replace(
                temporary,
                destination,
                expected_source_identity=temporary_identity,
                replace_existing=True,
            )
            temporary_identity = None
            self._require_root_current()
            if _secure_read_bytes(destination) != payload:
                raise CheckpointError("checkpoint latest index changed while written")
            self._require_root_current()
        except (OSError, PathSecurityError) as exc:
            raise CheckpointError("checkpoint latest index write failed closed") from exc
        finally:
            if temporary_identity is not None:
                _remove_file_if_identity(temporary, temporary_identity)

    def _discard_stale_pending(self, directory_name: str) -> None:
        self._require_root_current()
        for candidate in tuple(self.root.iterdir()):
            match = _PENDING_DIRECTORY_RE.fullmatch(candidate.name)
            if match is None or match.group("directory") != directory_name:
                continue
            try:
                pending = require_plain_path(
                    candidate,
                    base=self.root,
                    leaf_kind="directory",
                    label="stale checkpoint pending directory",
                )
                directory_identity = _directory_path_identity(pending)
                members = tuple(pending.iterdir())
                if not {member.name for member in members}.issubset(
                    {_CHECKPOINT_NAME, _MANIFEST_NAME, _COMPLETE_NAME}
                ):
                    raise CheckpointError(
                        "stale checkpoint pending members are not exact"
                    )
                member_identities = {
                    member.name: _regular_path_identity(
                        require_plain_path(
                            member,
                            base=self.root,
                            leaf_kind="file",
                            label="stale checkpoint pending member",
                        )
                    )
                    for member in members
                }
            except (OSError, PathSecurityError) as exc:
                raise CheckpointError(
                    "stale checkpoint pending path failed closed"
                ) from exc
            _remove_known_pending_files(
                pending,
                directory_identity=directory_identity,
                member_identities=member_identities,
                strict=True,
            )
            if os.path.lexists(pending):
                raise CheckpointError(
                    "stale checkpoint pending identity changed during cleanup"
                )
        self._require_root_current()

    def _prepared_member_identities(
        self,
        prepared: PreparedCheckpoint,
    ) -> dict[str, tuple[int, int, int, int, int, int]]:
        if (
            not isinstance(prepared, PreparedCheckpoint)
            or prepared.root != self.root
            or prepared.pending_directory.parent != self.root
            or prepared.destination.parent != self.root
            or prepared.destination.name
            != _update_directory_name(prepared.update_step)
        ):
            raise CheckpointError("prepared checkpoint handle is invalid")
        identities = dict(prepared._member_identities)
        if set(identities) != {
            _CHECKPOINT_NAME,
            _MANIFEST_NAME,
            _COMPLETE_NAME,
        }:
            raise CheckpointError("prepared checkpoint member identity is invalid")
        self._require_checkpoint_directory_current(
            prepared.pending_directory,
            directory_identity=prepared._directory_identity,
            member_identities=identities,
        )
        expected_paths = {
            _CHECKPOINT_NAME: prepared.pending_checkpoint_path,
            _MANIFEST_NAME: prepared.pending_manifest_path,
            _COMPLETE_NAME: prepared.pending_complete_marker_path,
        }
        if any(
            path.parent != prepared.pending_directory or path.name != name
            for name, path in expected_paths.items()
        ):
            raise CheckpointError("prepared checkpoint path binding drifted")
        if (
            _sha256(_secure_read_bytes(prepared.pending_checkpoint_path))
            != prepared.checkpoint_sha256
            or _sha256(_secure_read_bytes(prepared.pending_manifest_path))
            != prepared.manifest_sha256
            or _sha256(_secure_read_bytes(prepared.pending_complete_marker_path))
            != prepared.complete_marker_sha256
        ):
            raise CheckpointError("prepared checkpoint bytes changed")
        self._require_checkpoint_directory_current(
            prepared.pending_directory,
            directory_identity=prepared._directory_identity,
            member_identities=identities,
        )
        return identities

    def prepare_complete(
        self,
        *,
        policy: nn.Module,
        optimizer: torch.optim.Optimizer,
        update_step: int,
        normalization_stats: Mapping[str, Any],
        scenario_sampler_state: Mapping[str, Any],
        vector_env_states: Sequence[Mapping[str, Any]],
        best_record: Mapping[str, Any],
        versions: Mapping[str, str],
        top_m_config: Mapping[str, Any],
        scale_profile: str,
        training_config: Mapping[str, Any],
        config_sha256: str,
        lineage: Mapping[str, Any],
        eval_metrics: Mapping[str, Any],
        safety_contract: SafetyContract | None = None,
    ) -> PreparedCheckpoint:
        self._require_root_current()
        metadata = _validated_metadata(
            policy=policy,
            optimizer=optimizer,
            update_step=update_step,
            normalization_stats=normalization_stats,
            scenario_sampler_state=scenario_sampler_state,
            vector_env_states=vector_env_states,
            best_record=best_record,
            versions=versions,
            top_m_config=top_m_config,
            scale_profile=scale_profile,
            training_config=training_config,
            config_sha256=config_sha256,
            lineage=lineage,
            eval_metrics=eval_metrics,
            safety_contract=safety_contract,
            profile=self._profile,
        )
        directory_name = _update_directory_name(update_step)
        destination = self.root / directory_name
        if os.path.lexists(destination):
            raise CheckpointError(f"checkpoint update {update_step} already exists")

        policy_hash = policy_state_sha256(policy)
        model_state_dict = policy.state_dict()
        optimizer_state_dict = optimizer.state_dict()
        _validate_finite_tensors(model_state_dict, state_name="model")
        _validate_finite_tensors(optimizer_state_dict, state_name="optimizer")

        self._discard_stale_pending(directory_name)
        if os.path.lexists(destination):
            raise CheckpointError("checkpoint destination appeared before prepare")

        payload = {
            "schema_version": self._profile.payload_schema,
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": optimizer_state_dict,
            "update_step": update_step,
            "policy_state_sha256": policy_hash,
            "rng_state": _capture_rng_state(),
            **metadata,
        }
        payload_buffer = io.BytesIO()
        torch.save(payload, payload_buffer)
        payload_bytes = payload_buffer.getvalue()
        checkpoint_hash = _sha256(payload_bytes)
        lineage_hash = _canonical_sha256(metadata["lineage"])
        manifest = {
            "schema_version": self._profile.manifest_schema,
            "update_step": update_step,
            "checkpoint": {
                "path": _CHECKPOINT_NAME,
                "sha256": checkpoint_hash,
                "size_bytes": len(payload_bytes),
            },
            "policy_state_sha256": policy_hash,
            "config_sha256": config_sha256,
            "lineage_sha256": lineage_hash,
        }
        if self._profile.requires_safety_contract:
            manifest.update(
                {
                    name: metadata[name]
                    for name in _SAFETY_BINDING_FIELDS
                }
            )
        manifest_bytes = ArtifactStore.canonical_json_bytes(manifest)
        manifest_hash = _sha256(manifest_bytes)
        complete = {
            "schema_version": self._profile.complete_schema,
            "update_step": update_step,
            "manifest_sha256": manifest_hash,
            "checkpoint_sha256": checkpoint_hash,
        }
        complete_bytes = ArtifactStore.canonical_json_bytes(complete)

        pending = self.root / f".pending-{directory_name}-{uuid.uuid4().hex}"
        self._require_root_current()
        try:
            durable_mkdir(pending)
        except (OSError, PathSecurityError) as exc:
            raise CheckpointError(
                "checkpoint pending directory durable creation failed"
            ) from exc
        pending_identity = _directory_path_identity(pending)
        member_identities: dict[
            str,
            tuple[int, int, int, int, int, int],
        ] = {}
        try:
            member_identities[_CHECKPOINT_NAME] = _write_new_fsynced(
                pending / _CHECKPOINT_NAME,
                payload_bytes,
            )
            self._inject("after_payload_write")
            member_identities[_MANIFEST_NAME] = _write_new_fsynced(
                pending / _MANIFEST_NAME,
                manifest_bytes,
            )
            self._inject("after_manifest_write")
            member_identities[_COMPLETE_NAME] = _write_new_fsynced(
                pending / _COMPLETE_NAME,
                complete_bytes,
            )
            self._inject("after_pending_complete_write")
            try:
                durable_fsync_directory(pending)
            except (OSError, PathSecurityError) as exc:
                raise CheckpointError(
                    "checkpoint pending directory durability flush failed"
                ) from exc
            self._require_checkpoint_directory_current(
                pending,
                directory_identity=pending_identity,
                member_identities=member_identities,
            )
        except BaseException:
            _remove_known_pending_files(
                pending,
                directory_identity=pending_identity,
                member_identities=member_identities,
            )
            raise

        return PreparedCheckpoint(
            update_step=update_step,
            root=self.root,
            pending_directory=pending,
            destination=destination,
            pending_checkpoint_path=pending / _CHECKPOINT_NAME,
            pending_manifest_path=pending / _MANIFEST_NAME,
            pending_complete_marker_path=pending / _COMPLETE_NAME,
            checkpoint_sha256=checkpoint_hash,
            manifest_sha256=manifest_hash,
            complete_marker_sha256=_sha256(complete_bytes),
            policy_state_sha256=policy_hash,
            _directory_identity=pending_identity,
            _member_identities=tuple(sorted(member_identities.items())),
        )

    def abort_prepared(self, prepared: PreparedCheckpoint) -> bool:
        if not isinstance(prepared, PreparedCheckpoint) or prepared.root != self.root:
            raise CheckpointError("prepared checkpoint handle is invalid")
        if not os.path.lexists(prepared.pending_directory):
            return False
        identities = self._prepared_member_identities(prepared)
        _remove_known_pending_files(
            prepared.pending_directory,
            directory_identity=prepared._directory_identity,
            member_identities=identities,
            strict=True,
        )
        if os.path.lexists(prepared.pending_directory):
            raise CheckpointError("prepared checkpoint abort failed closed")
        self._require_root_current()
        return True

    def publish_prepared(
        self,
        prepared: PreparedCheckpoint,
    ) -> CheckpointReceipt:
        identities = self._prepared_member_identities(prepared)
        destination = prepared.destination
        self._inject("before_publish_rename")
        self._prepared_member_identities(prepared)
        if os.path.lexists(destination):
            raise CheckpointError("checkpoint destination appeared before publish")
        try:
            durable_rename(
                prepared.pending_directory,
                destination,
                expected_source_identity=prepared._directory_identity,
                replace_existing=False,
            )
        except (OSError, PathSecurityError) as exc:
            raise CheckpointError(
                "checkpoint publish namespace durability failed"
            ) from exc
        self._require_checkpoint_directory_current(
            destination,
            directory_identity=prepared._directory_identity,
            member_identities=identities,
        )
        self._inject("after_publish_rename")
        self._require_checkpoint_directory_current(
            destination,
            directory_identity=prepared._directory_identity,
            member_identities=identities,
        )
        self._write_latest(
            {
                "schema_version": self._profile.latest_schema,
                "update_step": prepared.update_step,
                "directory": destination.name,
                "manifest_sha256": prepared.manifest_sha256,
                "checkpoint_sha256": prepared.checkpoint_sha256,
            }
        )
        self._require_checkpoint_directory_current(
            destination,
            directory_identity=prepared._directory_identity,
            member_identities=identities,
        )
        return CheckpointReceipt(
            update_step=prepared.update_step,
            directory=destination,
            checkpoint_path=destination / _CHECKPOINT_NAME,
            manifest_path=destination / _MANIFEST_NAME,
            complete_marker_path=destination / _COMPLETE_NAME,
            checkpoint_sha256=prepared.checkpoint_sha256,
            manifest_sha256=prepared.manifest_sha256,
            policy_state_sha256=prepared.policy_state_sha256,
        )

    def recover_accepted_checkpoint(
        self,
        *,
        update_step: int,
        checkpoint_sha256: str,
        complete_marker_sha256: str,
        policy_state_sha256: str,
    ) -> CheckpointReceipt:
        """Publish the exact hidden checkpoint named by durable acceptance."""

        if type(update_step) is not int or update_step <= 0:
            raise CheckpointError("accepted checkpoint update must be positive")
        _require_sha256(checkpoint_sha256, "accepted checkpoint")
        _require_sha256(complete_marker_sha256, "accepted complete marker")
        _require_sha256(policy_state_sha256, "accepted policy state")
        self._require_root_current()
        directory_name = _update_directory_name(update_step)
        destination = self.root / directory_name
        if os.path.lexists(destination):
            candidate = self._validate_candidate(destination)
            complete_path = destination / _COMPLETE_NAME
            if (
                candidate.checkpoint_sha256 != checkpoint_sha256
                or _sha256(_secure_read_bytes(complete_path))
                != complete_marker_sha256
                or candidate.payload["policy_state_sha256"]
                != policy_state_sha256
            ):
                raise CheckpointError("accepted canonical checkpoint identity mismatch")
            expected_latest = {
                "schema_version": self._profile.latest_schema,
                "update_step": update_step,
                "directory": directory_name,
                "manifest_sha256": candidate.manifest_sha256,
                "checkpoint_sha256": checkpoint_sha256,
            }
            latest_path = self.root / "latest.json"
            repair_latest = not os.path.lexists(latest_path)
            if not repair_latest:
                latest, _ = _read_canonical_json(latest_path)
                latest_fields = {
                    "schema_version",
                    "update_step",
                    "directory",
                    "manifest_sha256",
                    "checkpoint_sha256",
                }
                if (
                    set(latest) != latest_fields
                    or latest.get("schema_version") != self._profile.latest_schema
                    or type(latest.get("update_step")) is not int
                    or not isinstance(latest.get("directory"), str)
                    or not isinstance(latest.get("manifest_sha256"), str)
                    or _SHA256_RE.fullmatch(str(latest["manifest_sha256"])) is None
                    or not isinstance(latest.get("checkpoint_sha256"), str)
                    or _SHA256_RE.fullmatch(str(latest["checkpoint_sha256"])) is None
                ):
                    raise CheckpointError("checkpoint latest index drifted")
                if latest == expected_latest:
                    repair_latest = False
                elif int(latest["update_step"]) < update_step:
                    prior_directory = self.root / str(latest["directory"])
                    if latest["directory"] != _update_directory_name(
                        int(latest["update_step"])
                    ):
                        raise CheckpointError("checkpoint latest index drifted")
                    prior = self._validate_candidate(prior_directory)
                    if (
                        prior.checkpoint_sha256 != latest["checkpoint_sha256"]
                        or prior.manifest_sha256 != latest["manifest_sha256"]
                    ):
                        raise CheckpointError("checkpoint latest index drifted")
                    repair_latest = True
                else:
                    raise CheckpointError("checkpoint latest index drifted")
            if repair_latest:
                self._write_latest(expected_latest)
            self._require_root_current()
            return CheckpointReceipt(
                update_step=update_step,
                directory=destination,
                checkpoint_path=destination / _CHECKPOINT_NAME,
                manifest_path=destination / _MANIFEST_NAME,
                complete_marker_path=complete_path,
                checkpoint_sha256=checkpoint_sha256,
                manifest_sha256=candidate.manifest_sha256,
                policy_state_sha256=policy_state_sha256,
            )
        pending_candidates = tuple(
            path
            for path in self.root.iterdir()
            if (
                (match := _PENDING_DIRECTORY_RE.fullmatch(path.name)) is not None
                and match.group("directory") == directory_name
            )
        )
        if len(pending_candidates) != 1:
            raise CheckpointError("accepted checkpoint pending directory is not exact")
        try:
            pending = require_plain_path(
                pending_candidates[0],
                base=self.root,
                leaf_kind="directory",
                label="accepted checkpoint pending directory",
            )
            members = tuple(pending.iterdir())
            if {member.name for member in members} != {
                _CHECKPOINT_NAME,
                _MANIFEST_NAME,
                _COMPLETE_NAME,
            }:
                raise CheckpointError("accepted checkpoint pending members are not exact")
            by_name = {
                member.name: require_plain_path(
                    member,
                    base=self.root,
                    leaf_kind="file",
                    label="accepted checkpoint pending member",
                )
                for member in members
            }
        except PathSecurityError as exc:
            raise CheckpointError(
                "accepted checkpoint pending path contains a link or reparse point"
            ) from exc
        directory_identity = _directory_path_identity(pending)
        member_identities = {
            name: _regular_path_identity(path) for name, path in by_name.items()
        }
        manifest, manifest_bytes = _read_canonical_json(by_name[_MANIFEST_NAME])
        complete, complete_bytes = _read_canonical_json(by_name[_COMPLETE_NAME])
        checkpoint_bytes = _secure_read_bytes(by_name[_CHECKPOINT_NAME])
        self._require_checkpoint_directory_current(
            pending,
            directory_identity=directory_identity,
            member_identities=member_identities,
        )
        _validate_manifest(
            manifest,
            expected_step=update_step,
            profile=self._profile,
        )
        _validate_complete(
            complete,
            expected_step=update_step,
            profile=self._profile,
        )
        manifest_sha256 = _sha256(manifest_bytes)
        payload_sha256 = _sha256(checkpoint_bytes)
        if (
            payload_sha256 != checkpoint_sha256
            or _sha256(complete_bytes) != complete_marker_sha256
            or complete["checkpoint_sha256"] != checkpoint_sha256
            or complete["manifest_sha256"] != manifest_sha256
            or manifest["checkpoint"]["sha256"] != checkpoint_sha256
            or manifest["checkpoint"]["size_bytes"] != len(checkpoint_bytes)
            or manifest["policy_state_sha256"] != policy_state_sha256
        ):
            raise CheckpointError("accepted checkpoint identity mismatch")
        try:
            payload = safe_load_checkpoint_payload(checkpoint_bytes)
        except Exception as exc:
            raise CheckpointError("accepted checkpoint payload cannot be decoded") from exc
        _validate_loaded_payload(
            payload,
            expected_step=update_step,
            manifest=manifest,
            profile=self._profile,
        )
        if payload["policy_state_sha256"] != policy_state_sha256:
            raise CheckpointError("accepted checkpoint policy identity mismatch")
        prepared = PreparedCheckpoint(
            update_step=update_step,
            root=self.root,
            pending_directory=pending,
            destination=destination,
            pending_checkpoint_path=by_name[_CHECKPOINT_NAME],
            pending_manifest_path=by_name[_MANIFEST_NAME],
            pending_complete_marker_path=by_name[_COMPLETE_NAME],
            checkpoint_sha256=checkpoint_sha256,
            manifest_sha256=manifest_sha256,
            complete_marker_sha256=complete_marker_sha256,
            policy_state_sha256=policy_state_sha256,
            _directory_identity=directory_identity,
            _member_identities=tuple(sorted(member_identities.items())),
        )
        return self.publish_prepared(prepared)

    def save_complete(
        self,
        *,
        policy: nn.Module,
        optimizer: torch.optim.Optimizer,
        update_step: int,
        normalization_stats: Mapping[str, Any],
        scenario_sampler_state: Mapping[str, Any],
        vector_env_states: Sequence[Mapping[str, Any]],
        best_record: Mapping[str, Any],
        versions: Mapping[str, str],
        top_m_config: Mapping[str, Any],
        scale_profile: str,
        training_config: Mapping[str, Any],
        config_sha256: str,
        lineage: Mapping[str, Any],
        eval_metrics: Mapping[str, Any],
        safety_contract: SafetyContract | None = None,
    ) -> CheckpointReceipt:
        """Stage 4-compatible prepare-and-publish wrapper."""

        prepared = self.prepare_complete(
            policy=policy,
            optimizer=optimizer,
            update_step=update_step,
            normalization_stats=normalization_stats,
            scenario_sampler_state=scenario_sampler_state,
            vector_env_states=vector_env_states,
            best_record=best_record,
            versions=versions,
            top_m_config=top_m_config,
            scale_profile=scale_profile,
            training_config=training_config,
            config_sha256=config_sha256,
            lineage=lineage,
            eval_metrics=eval_metrics,
            safety_contract=safety_contract,
        )
        try:
            return self.publish_prepared(prepared)
        except BaseException:
            self.abort_prepared(prepared)
            raise

    def load_last_complete(
        self,
        *,
        policy: nn.Module,
        optimizer: torch.optim.Optimizer,
        expected_config_sha256: str,
        expected_lineage: Mapping[str, Any],
        expected_safety_contract: SafetyContract | None = None,
    ) -> LoadedCheckpoint:
        self._require_root_current()
        if not isinstance(policy, nn.Module):
            raise CheckpointError("policy must be a torch module")
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise CheckpointError("optimizer must be a torch optimizer")
        _require_sha256(expected_config_sha256, "expected config")
        expected_lineage_dict = _validated_json_mapping(
            expected_lineage,
            "expected lineage",
        )
        _validate_lineage(expected_lineage_dict, self._profile)
        expected_safety_binding = _validated_safety_argument(
            expected_safety_contract,
            config_sha256=expected_config_sha256,
            lineage=expected_lineage_dict,
            profile=self._profile,
        )
        if not self._profile.required_lineage.issubset(expected_lineage_dict):
            raise CheckpointError("expected lineage is incomplete")

        rejection_reasons: list[str] = []
        root_members = tuple(self.root.iterdir())
        for path in root_members:
            try:
                require_plain_path(
                    path,
                    base=self.root,
                    leaf_kind="directory" if path.name.startswith("update-") else None,
                    label="checkpoint root member",
                )
            except PathSecurityError as exc:
                raise CheckpointError(
                    "checkpoint root member is a link or reparse point"
                ) from exc
        candidate_directories = sorted(
            (
                path
                for path in root_members
                if path.is_dir() and _UPDATE_DIRECTORY_RE.fullmatch(path.name)
            ),
            key=lambda path: int(
                _UPDATE_DIRECTORY_RE.fullmatch(path.name).group("step")  # type: ignore[union-attr]
            ),
            reverse=True,
        )
        selected: _ValidatedCandidate | None = None
        for directory in candidate_directories:
            try:
                candidate = self._validate_candidate(directory)
                payload = candidate.payload
                if payload["config_sha256"] != expected_config_sha256:
                    raise CheckpointError("config hash mismatch")
                if payload["lineage"] != expected_lineage_dict:
                    raise CheckpointError("lineage mismatch")
                _require_expected_safety_binding(
                    payload,
                    expected_safety_binding,
                )
                selected = candidate
                break
            except Exception as exc:
                rejection_reasons.append(f"{directory.name}: {exc}")
        if selected is None:
            detail = "; ".join(rejection_reasons) or "no completed update directories"
            raise CheckpointError(
                f"no compatible complete checkpoint: {detail}"
            )

        return self._restore_candidate(
            selected,
            policy=policy,
            optimizer=optimizer,
        )

    def load_complete(
        self,
        *,
        update_step: int,
        policy: nn.Module,
        optimizer: torch.optim.Optimizer,
        expected_config_sha256: str,
        expected_lineage: Mapping[str, Any],
        expected_safety_contract: SafetyContract | None = None,
    ) -> LoadedCheckpoint:
        """加载一个已验证且仍被 retention 保留的指定 update。"""

        if type(update_step) is not int or update_step <= 0:
            raise CheckpointError("specific checkpoint update must be positive")
        if not isinstance(policy, nn.Module) or not isinstance(
            optimizer, torch.optim.Optimizer
        ):
            raise CheckpointError("specific checkpoint restore target drifted")
        _require_sha256(expected_config_sha256, "expected config")
        expected_lineage_dict = _validated_json_mapping(
            expected_lineage,
            "expected lineage",
        )
        _validate_lineage(expected_lineage_dict, self._profile)
        expected_safety_binding = _validated_safety_argument(
            expected_safety_contract,
            config_sha256=expected_config_sha256,
            lineage=expected_lineage_dict,
            profile=self._profile,
        )
        directory = self.root / _update_directory_name(update_step)
        candidate = self._validate_candidate(directory)
        if candidate.payload["config_sha256"] != expected_config_sha256:
            raise CheckpointError("config hash mismatch")
        if candidate.payload["lineage"] != expected_lineage_dict:
            raise CheckpointError("lineage mismatch")
        _require_expected_safety_binding(
            candidate.payload,
            expected_safety_binding,
        )
        return self._restore_candidate(
            candidate,
            policy=policy,
            optimizer=optimizer,
        )

    def inspect_complete(
        self,
        *,
        update_step: int,
        expected_config_sha256: str,
        expected_lineage: Mapping[str, Any],
        expected_safety_contract: SafetyContract | None = None,
    ) -> CheckpointReceipt:
        """只读验证一个完整 checkpoint，不修改任何 runtime 状态。"""

        if type(update_step) is not int or update_step <= 0:
            raise CheckpointError("specific checkpoint update must be positive")
        _require_sha256(expected_config_sha256, "expected config")
        expected_lineage_dict = _validated_json_mapping(
            expected_lineage,
            "expected lineage",
        )
        _validate_lineage(expected_lineage_dict, self._profile)
        expected_safety_binding = _validated_safety_argument(
            expected_safety_contract,
            config_sha256=expected_config_sha256,
            lineage=expected_lineage_dict,
            profile=self._profile,
        )
        directory = self.root / _update_directory_name(update_step)
        candidate = self._validate_candidate(directory)
        if candidate.payload["config_sha256"] != expected_config_sha256:
            raise CheckpointError("config hash mismatch")
        if candidate.payload["lineage"] != expected_lineage_dict:
            raise CheckpointError("lineage mismatch")
        _require_expected_safety_binding(
            candidate.payload,
            expected_safety_binding,
        )
        return CheckpointReceipt(
            update_step=update_step,
            directory=directory,
            checkpoint_path=directory / _CHECKPOINT_NAME,
            manifest_path=directory / _MANIFEST_NAME,
            complete_marker_path=directory / _COMPLETE_NAME,
            checkpoint_sha256=candidate.checkpoint_sha256,
            manifest_sha256=candidate.manifest_sha256,
            policy_state_sha256=str(candidate.payload["policy_state_sha256"]),
        )

    @staticmethod
    def _restore_candidate(
        selected: _ValidatedCandidate,
        *,
        policy: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> LoadedCheckpoint:

        payload = selected.payload
        _validate_model_state(policy, payload["model_state_dict"])
        policy_before = copy.deepcopy(policy.state_dict())
        optimizer_before = copy.deepcopy(optimizer.state_dict())
        rng_before = copy.deepcopy(_capture_rng_state())
        try:
            policy.load_state_dict(payload["model_state_dict"], strict=True)
            restored_hash = policy_state_sha256(policy)
            if restored_hash != payload["policy_state_sha256"]:
                raise CheckpointError("restored policy hash mismatch")
            optimizer.load_state_dict(payload["optimizer_state_dict"])
            _restore_rng_state(payload["rng_state"])
        except BaseException:
            try:
                policy.load_state_dict(policy_before, strict=True)
                optimizer.load_state_dict(optimizer_before)
                _restore_rng_state(rng_before)
            except BaseException as rollback_error:
                raise CheckpointError(
                    "checkpoint restore failed and target rollback was not exact"
                ) from rollback_error
            raise
        return LoadedCheckpoint(
            update_step=payload["update_step"],
            directory=selected.directory,
            checkpoint_sha256=selected.checkpoint_sha256,
            manifest_sha256=selected.manifest_sha256,
            policy_state_sha256=restored_hash,
            normalization_stats=dict(payload["normalization_stats"]),
            scenario_sampler_state=dict(payload["scenario_sampler_state"]),
            vector_env_states=tuple(
                dict(state) for state in payload["vector_env_states"]
            ),
            best_record=dict(payload["best_record"]),
            versions=dict(payload["versions"]),
            top_m_config=dict(payload["top_m_config"]),
            scale_profile=payload["scale_profile"],
            training_config=dict(payload["training_config"]),
            config_sha256=payload["config_sha256"],
            lineage=dict(payload["lineage"]),
            eval_metrics=dict(payload["eval_metrics"]),
            safety_contract=(
                dict(payload["safety_contract"])
                if "safety_contract" in payload
                else None
            ),
        )

    def _validate_candidate(self, directory: Path) -> _ValidatedCandidate:
        self._require_root_current()
        match = _UPDATE_DIRECTORY_RE.fullmatch(directory.name)
        if match is None:
            raise CheckpointError("invalid checkpoint directory name")
        expected_step = int(match.group("step"))
        try:
            directory = require_plain_path(
                directory,
                base=self.root,
                leaf_kind="directory",
                label="checkpoint update directory",
            )
            members = tuple(directory.iterdir())
            if {member.name for member in members} != {
                _CHECKPOINT_NAME,
                _MANIFEST_NAME,
                _COMPLETE_NAME,
            }:
                raise CheckpointError(
                    "checkpoint update contains unknown or incomplete members"
                )
            by_name = {member.name: member for member in members}
            checkpoint_path = require_plain_path(
                by_name[_CHECKPOINT_NAME],
                base=self.root,
                leaf_kind="file",
                label="checkpoint payload",
            )
            manifest_path = require_plain_path(
                by_name[_MANIFEST_NAME],
                base=self.root,
                leaf_kind="file",
                label="checkpoint manifest",
            )
            complete_path = require_plain_path(
                by_name[_COMPLETE_NAME],
                base=self.root,
                leaf_kind="file",
                label="checkpoint complete marker",
            )
        except PathSecurityError as exc:
            raise CheckpointError(
                "checkpoint path contains a link or reparse point"
            ) from exc
        directory_identity = _directory_path_identity(directory)
        identities = {
            path.name: _regular_path_identity(path)
            for path in (checkpoint_path, manifest_path, complete_path)
        }

        manifest, manifest_bytes = _read_canonical_json(
            manifest_path,
            after_read=lambda: self._inject("after_manifest_read"),
        )
        complete, _ = _read_canonical_json(
            complete_path,
            after_read=lambda: self._inject("after_complete_read"),
        )
        _validate_manifest(
            manifest,
            expected_step=expected_step,
            profile=self._profile,
        )
        _validate_complete(
            complete,
            expected_step=expected_step,
            profile=self._profile,
        )
        manifest_hash = _sha256(manifest_bytes)
        if complete["manifest_sha256"] != manifest_hash:
            raise CheckpointError("complete marker manifest hash mismatch")

        checkpoint_bytes = _secure_read_bytes(
            checkpoint_path,
            after_read=lambda: self._inject("after_checkpoint_read"),
        )
        self._require_checkpoint_directory_current(
            directory,
            directory_identity=directory_identity,
            member_identities=identities,
        )
        checkpoint_hash = _sha256(checkpoint_bytes)
        checkpoint_record = manifest["checkpoint"]
        if (
            checkpoint_record["sha256"] != checkpoint_hash
            or complete["checkpoint_sha256"] != checkpoint_hash
            or checkpoint_record["size_bytes"] != len(checkpoint_bytes)
        ):
            raise CheckpointError("checkpoint payload hash or size mismatch")
        try:
            payload = safe_load_checkpoint_payload(checkpoint_bytes)
        except Exception as exc:
            raise CheckpointError("checkpoint payload cannot be decoded") from exc
        _validate_loaded_payload(
            payload,
            expected_step=expected_step,
            manifest=manifest,
            profile=self._profile,
        )
        return _ValidatedCandidate(
            directory=directory,
            checkpoint_sha256=checkpoint_hash,
            manifest_sha256=manifest_hash,
            payload=payload,
        )

    def _inject(self, phase: str) -> None:
        if self._fault_injector is not None:
            self._fault_injector(phase)


def inspect_complete_checkpoint_snapshot(
    *,
    directory: str | Path,
    update_step: int,
    schema_version: str,
    member_names: Sequence[str],
    checkpoint_bytes: bytes,
    manifest_bytes: bytes,
    complete_bytes: bytes,
    expected_config_sha256: str,
    expected_lineage: Mapping[str, Any],
    expected_safety_contract: SafetyContract | None = None,
) -> CheckpointReceipt:
    """Inspect one complete checkpoint from an already-bound byte snapshot."""

    profile = _profile_for_schema(schema_version)
    candidate = _validated_candidate_from_snapshot(
        directory=directory,
        update_step=update_step,
        member_names=member_names,
        checkpoint_bytes=checkpoint_bytes,
        manifest_bytes=manifest_bytes,
        complete_bytes=complete_bytes,
        profile=profile,
    )
    _require_snapshot_candidate_expectations(
        candidate,
        expected_config_sha256=expected_config_sha256,
        expected_lineage=expected_lineage,
        expected_safety_contract=expected_safety_contract,
        profile=profile,
    )
    return CheckpointReceipt(
        update_step=update_step,
        directory=candidate.directory,
        checkpoint_path=candidate.directory / _CHECKPOINT_NAME,
        manifest_path=candidate.directory / _MANIFEST_NAME,
        complete_marker_path=candidate.directory / _COMPLETE_NAME,
        checkpoint_sha256=candidate.checkpoint_sha256,
        manifest_sha256=candidate.manifest_sha256,
        policy_state_sha256=str(candidate.payload["policy_state_sha256"]),
    )


def load_complete_checkpoint_snapshot(
    *,
    directory: str | Path,
    update_step: int,
    schema_version: str,
    member_names: Sequence[str],
    checkpoint_bytes: bytes,
    manifest_bytes: bytes,
    complete_bytes: bytes,
    policy: nn.Module,
    optimizer: torch.optim.Optimizer,
    expected_config_sha256: str,
    expected_lineage: Mapping[str, Any],
    expected_safety_contract: SafetyContract | None = None,
) -> LoadedCheckpoint:
    """Restore one checkpoint using only an already-bound byte snapshot."""

    if not isinstance(policy, nn.Module) or not isinstance(
        optimizer, torch.optim.Optimizer
    ):
        raise CheckpointError("specific checkpoint restore target drifted")
    profile = _profile_for_schema(schema_version)
    candidate = _validated_candidate_from_snapshot(
        directory=directory,
        update_step=update_step,
        member_names=member_names,
        checkpoint_bytes=checkpoint_bytes,
        manifest_bytes=manifest_bytes,
        complete_bytes=complete_bytes,
        profile=profile,
    )
    _require_snapshot_candidate_expectations(
        candidate,
        expected_config_sha256=expected_config_sha256,
        expected_lineage=expected_lineage,
        expected_safety_contract=expected_safety_contract,
        profile=profile,
    )
    return CheckpointManager._restore_candidate(
        candidate,
        policy=policy,
        optimizer=optimizer,
    )


def _validated_candidate_from_snapshot(
    *,
    directory: str | Path,
    update_step: int,
    member_names: Sequence[str],
    checkpoint_bytes: bytes,
    manifest_bytes: bytes,
    complete_bytes: bytes,
    profile: _CheckpointProfile,
) -> _ValidatedCandidate:
    if type(update_step) is not int or update_step <= 0:
        raise CheckpointError("specific checkpoint update must be positive")
    logical_directory = Path(directory)
    if logical_directory.name != _update_directory_name(update_step):
        raise CheckpointError("invalid checkpoint directory name")
    expected_members = tuple(sorted((_CHECKPOINT_NAME, _MANIFEST_NAME, _COMPLETE_NAME)))
    if (
        not isinstance(member_names, Sequence)
        or isinstance(member_names, (str, bytes, bytearray))
        or tuple(member_names) != expected_members
        or any(not isinstance(name, str) for name in member_names)
        or any(
            type(payload) is not bytes or not payload
            for payload in (checkpoint_bytes, manifest_bytes, complete_bytes)
        )
    ):
        raise CheckpointError("checkpoint snapshot membership drifted")
    manifest = _read_canonical_json_bytes(manifest_bytes, label="manifest")
    complete = _read_canonical_json_bytes(complete_bytes, label="complete marker")
    _validate_manifest(manifest, expected_step=update_step, profile=profile)
    _validate_complete(complete, expected_step=update_step, profile=profile)
    manifest_hash = _sha256(manifest_bytes)
    checkpoint_hash = _sha256(checkpoint_bytes)
    checkpoint_record = manifest["checkpoint"]
    if (
        complete["manifest_sha256"] != manifest_hash
        or checkpoint_record["sha256"] != checkpoint_hash
        or complete["checkpoint_sha256"] != checkpoint_hash
        or checkpoint_record["size_bytes"] != len(checkpoint_bytes)
    ):
        raise CheckpointError("checkpoint payload hash or size mismatch")
    try:
        payload = safe_load_checkpoint_payload(checkpoint_bytes)
    except Exception as exc:
        raise CheckpointError("checkpoint payload cannot be decoded") from exc
    _validate_loaded_payload(
        payload,
        expected_step=update_step,
        manifest=manifest,
        profile=profile,
    )
    return _ValidatedCandidate(
        directory=logical_directory,
        checkpoint_sha256=checkpoint_hash,
        manifest_sha256=manifest_hash,
        payload=payload,
    )


def _require_snapshot_candidate_expectations(
    candidate: _ValidatedCandidate,
    *,
    expected_config_sha256: str,
    expected_lineage: Mapping[str, Any],
    expected_safety_contract: SafetyContract | None,
    profile: _CheckpointProfile,
) -> None:
    _require_sha256(expected_config_sha256, "expected config")
    expected_lineage_dict = _validated_json_mapping(
        expected_lineage,
        "expected lineage",
    )
    _validate_lineage(expected_lineage_dict, profile)
    expected_safety_binding = _validated_safety_argument(
        expected_safety_contract,
        config_sha256=expected_config_sha256,
        lineage=expected_lineage_dict,
        profile=profile,
    )
    if not profile.required_lineage.issubset(expected_lineage_dict):
        raise CheckpointError("expected lineage is incomplete")
    if candidate.payload["config_sha256"] != expected_config_sha256:
        raise CheckpointError("config hash mismatch")
    if candidate.payload["lineage"] != expected_lineage_dict:
        raise CheckpointError("lineage mismatch")
    _require_expected_safety_binding(candidate.payload, expected_safety_binding)


def _read_canonical_json_bytes(
    raw: bytes,
    *,
    label: str,
) -> dict[str, Any]:
    if type(raw) is not bytes or not raw:
        raise CheckpointError(f"invalid JSON checkpoint artifact: {label}")
    try:
        import json

        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise CheckpointError(f"invalid JSON checkpoint artifact: {label}") from exc
    if not isinstance(value, dict):
        raise CheckpointError(f"checkpoint JSON artifact is not an object: {label}")
    try:
        canonical = ArtifactStore.canonical_json_bytes(value)
    except (TypeError, ValueError) as exc:
        raise CheckpointError(f"invalid JSON checkpoint artifact: {label}") from exc
    if raw != canonical:
        raise CheckpointError(f"checkpoint JSON artifact is not canonical: {label}")
    return value


def _validated_metadata(
    *,
    policy: nn.Module,
    optimizer: torch.optim.Optimizer,
    update_step: int,
    normalization_stats: Mapping[str, Any],
    scenario_sampler_state: Mapping[str, Any],
    vector_env_states: Sequence[Mapping[str, Any]],
    best_record: Mapping[str, Any],
    versions: Mapping[str, str],
    top_m_config: Mapping[str, Any],
    scale_profile: str,
    training_config: Mapping[str, Any],
    config_sha256: str,
    lineage: Mapping[str, Any],
    eval_metrics: Mapping[str, Any],
    safety_contract: SafetyContract | None,
    profile: _CheckpointProfile,
) -> dict[str, Any]:
    if not isinstance(policy, nn.Module):
        raise CheckpointError("policy must be a torch module")
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise CheckpointError("optimizer must be a torch optimizer")
    if type(update_step) is not int or update_step <= 0:
        raise CheckpointError("update_step must be a positive integer")
    if not isinstance(vector_env_states, Sequence) or len(vector_env_states) != 8:
        raise CheckpointError("checkpoint requires exactly 8 vector env states")
    vector_states = tuple(
        _validated_json_mapping(state, "vector env state")
        for state in vector_env_states
    )
    version_dict = _validated_string_mapping(versions, "versions")
    if set(version_dict) != _REQUIRED_VERSIONS:
        raise CheckpointError("checkpoint versions must contain the exact frozen fields")
    if not isinstance(scale_profile, str) or not scale_profile:
        raise CheckpointError("scale_profile must be non-empty")
    _require_sha256(config_sha256, "config")
    lineage_dict = _validated_json_mapping(lineage, "lineage")
    _validate_lineage(lineage_dict, profile)
    if not profile.required_lineage.issubset(lineage_dict):
        raise CheckpointError("checkpoint lineage is incomplete")
    metadata = {
        "normalization_stats": _validated_json_mapping(
            normalization_stats,
            "normalization stats",
        ),
        "scenario_sampler_state": _validated_json_mapping(
            scenario_sampler_state,
            "scenario sampler state",
        ),
        "vector_env_states": vector_states,
        "best_record": _validated_json_mapping(best_record, "best record"),
        "versions": version_dict,
        "top_m_config": _validated_json_mapping(top_m_config, "top-M config"),
        "scale_profile": scale_profile,
        "training_config": _validated_json_mapping(
            training_config,
            "training config",
        ),
        "config_sha256": config_sha256,
        "lineage": lineage_dict,
        "eval_metrics": _validated_json_mapping(eval_metrics, "eval metrics"),
    }
    safety_binding = _validated_safety_argument(
        safety_contract,
        config_sha256=config_sha256,
        lineage=lineage_dict,
        profile=profile,
    )
    if safety_binding is not None:
        metadata.update(safety_binding)
    try:
        ArtifactStore.canonical_json_bytes(metadata)
    except (TypeError, ValueError) as exc:
        raise CheckpointError("checkpoint metadata must be canonical JSON data") from exc
    return metadata


def _validate_lineage(
    lineage: Mapping[str, Any],
    profile: _CheckpointProfile,
) -> None:
    if not profile.required_lineage.issubset(lineage):
        raise CheckpointError("checkpoint lineage is incomplete")
    if profile is _STAGE6_PROFILE:
        if (
            not isinstance(lineage.get("stage5_commit"), str)
            or _SHA1_RE.fullmatch(str(lineage["stage5_commit"])) is None
        ):
            raise CheckpointError("Stage 6 lineage Stage 5 commit is invalid")
        for name in profile.required_lineage - {"stage5_commit"}:
            _require_sha256(lineage.get(name), f"Stage 6 lineage {name}")


def _validated_json_mapping(
    value: Mapping[str, Any],
    name: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise CheckpointError(f"{name} must be a string-keyed mapping")
    result = dict(value)
    try:
        ArtifactStore.canonical_json_bytes(result)
    except (TypeError, ValueError) as exc:
        raise CheckpointError(f"{name} must contain canonical JSON values") from exc
    return result


def _validated_string_mapping(
    value: Mapping[str, str],
    name: str,
) -> dict[str, str]:
    result = _validated_json_mapping(value, name)
    if any(not isinstance(item, str) or not item for item in result.values()):
        raise CheckpointError(f"{name} values must be non-empty strings")
    return result


def _capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            tuple(torch.cuda.get_rng_state_all()) if torch.cuda.is_available() else ()
        ),
    }


def _restore_rng_state(state: object) -> None:
    if not isinstance(state, Mapping) or set(state) != {
        "python",
        "numpy",
        "torch_cpu",
        "torch_cuda",
    }:
        raise CheckpointError("checkpoint RNG state is incomplete")
    try:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch_cpu"])
        cuda_states = tuple(state["torch_cuda"])
        if cuda_states:
            if not torch.cuda.is_available():
                raise CheckpointError(
                    "checkpoint contains CUDA RNG state but CUDA is unavailable"
                )
            if len(cuda_states) != torch.cuda.device_count():
                raise CheckpointError("CUDA RNG device count mismatch")
            torch.cuda.set_rng_state_all(list(cuda_states))
    except CheckpointError:
        raise
    except Exception as exc:
        raise CheckpointError("checkpoint RNG state cannot be restored") from exc


def _validate_finite_tensors(value: object, *, state_name: str) -> None:
    visited: set[int] = set()

    def visit(item: object) -> None:
        if isinstance(item, torch.Tensor):
            if not (item.is_floating_point() or item.is_complex()):
                return
            tensor = item.detach()
            try:
                if tensor.layout == torch.sparse_coo:
                    tensor = tensor.coalesce().values()
                elif tensor.layout != torch.strided:
                    tensor = tensor.values()
                is_finite = bool(torch.isfinite(tensor).all().item())
            except (RuntimeError, TypeError, NotImplementedError) as exc:
                raise CheckpointError(
                    f"checkpoint {state_name} state tensor finiteness "
                    "cannot be verified"
                ) from exc
            if not is_finite:
                raise CheckpointError(
                    f"checkpoint {state_name} state contains non-finite tensor"
                )
            return

        if isinstance(item, Mapping):
            identity = id(item)
            if identity in visited:
                return
            visited.add(identity)
            for nested in item.values():
                visit(nested)
            return

        if isinstance(item, Sequence) and not isinstance(
            item,
            (str, bytes, bytearray),
        ):
            identity = id(item)
            if identity in visited:
                return
            visited.add(identity)
            for nested in item:
                visit(nested)

    visit(value)


def _validate_loaded_payload(
    payload: object,
    *,
    expected_step: int,
    manifest: dict[str, Any],
    profile: _CheckpointProfile,
) -> None:
    required = {
        "schema_version",
        "model_state_dict",
        "optimizer_state_dict",
        "update_step",
        "policy_state_sha256",
        "rng_state",
        "normalization_stats",
        "scenario_sampler_state",
        "vector_env_states",
        "best_record",
        "versions",
        "top_m_config",
        "scale_profile",
        "training_config",
        "config_sha256",
        "lineage",
        "eval_metrics",
    }
    if profile.requires_safety_contract:
        required.update(_SAFETY_BINDING_FIELDS)
    if not isinstance(payload, dict) or set(payload) != required:
        raise CheckpointError("checkpoint payload fields are not exact")
    if (
        payload["schema_version"] != profile.payload_schema
        or payload["update_step"] != expected_step
    ):
        raise CheckpointError("checkpoint schema or update step mismatch")
    _require_sha256(payload["policy_state_sha256"], "policy state")
    _require_sha256(payload["config_sha256"], "config")
    if (
        payload["policy_state_sha256"] != manifest["policy_state_sha256"]
        or payload["config_sha256"] != manifest["config_sha256"]
        or _canonical_sha256(payload["lineage"]) != manifest["lineage_sha256"]
    ):
        raise CheckpointError("checkpoint metadata does not match manifest")
    if profile.requires_safety_contract:
        payload_safety = _validated_safety_binding(
            payload,
            expected_config_sha256=payload["config_sha256"],
        )
        manifest_safety = _validated_safety_binding(
            manifest,
            expected_config_sha256=manifest["config_sha256"],
        )
        if payload_safety != manifest_safety:
            raise CheckpointError("Stage 6 checkpoint safety contract drifted")
        if (
            payload["lineage"]["safety_contract_sha256"]
            != payload_safety["safety_contract_sha256"]
        ):
            raise CheckpointError("Stage 6 checkpoint safety lineage drifted")
    if not isinstance(payload["model_state_dict"], Mapping):
        raise CheckpointError("model state dict is invalid")
    if not isinstance(payload["optimizer_state_dict"], Mapping):
        raise CheckpointError("optimizer state dict is invalid")
    _validate_finite_tensors(payload["model_state_dict"], state_name="model")
    _validate_finite_tensors(
        payload["optimizer_state_dict"],
        state_name="optimizer",
    )
    if not isinstance(payload["vector_env_states"], (tuple, list)) or len(
        payload["vector_env_states"]
    ) != 8:
        raise CheckpointError("checkpoint vector env state count is invalid")
    version_dict = _validated_string_mapping(payload["versions"], "versions")
    if set(version_dict) != _REQUIRED_VERSIONS:
        raise CheckpointError("checkpoint versions are incomplete")
    for name in (
        "normalization_stats",
        "scenario_sampler_state",
        "best_record",
        "top_m_config",
        "training_config",
        "lineage",
        "eval_metrics",
    ):
        _validated_json_mapping(payload[name], name)
    for state in payload["vector_env_states"]:
        _validated_json_mapping(state, "vector env state")
    _validate_lineage(payload["lineage"], profile)


def _validate_model_state(
    policy: nn.Module,
    saved_state: Mapping[str, Any],
) -> None:
    current = policy.state_dict()
    if set(current) != set(saved_state):
        raise CheckpointError("checkpoint model keys do not match policy")
    for name, current_tensor in current.items():
        saved_tensor = saved_state[name]
        if not isinstance(saved_tensor, torch.Tensor):
            raise CheckpointError("checkpoint model state contains non-tensor value")
        if (
            current_tensor.shape != saved_tensor.shape
            or current_tensor.dtype != saved_tensor.dtype
        ):
            raise CheckpointError(f"checkpoint model tensor mismatch: {name}")


def _validate_manifest(
    manifest: object,
    *,
    expected_step: int,
    profile: _CheckpointProfile,
) -> None:
    required = {
        "schema_version",
        "update_step",
        "checkpoint",
        "policy_state_sha256",
        "config_sha256",
        "lineage_sha256",
    }
    if profile.requires_safety_contract:
        required.update(_SAFETY_BINDING_FIELDS)
    if not isinstance(manifest, dict) or set(manifest) != required:
        raise CheckpointError("checkpoint manifest fields are not exact")
    if (
        manifest["schema_version"] != profile.manifest_schema
        or manifest["update_step"] != expected_step
    ):
        raise CheckpointError("checkpoint manifest schema or step mismatch")
    checkpoint = manifest["checkpoint"]
    if not isinstance(checkpoint, dict) or set(checkpoint) != {
        "path",
        "sha256",
        "size_bytes",
    }:
        raise CheckpointError("checkpoint manifest payload record is invalid")
    if (
        checkpoint["path"] != _CHECKPOINT_NAME
        or type(checkpoint["size_bytes"]) is not int
        or checkpoint["size_bytes"] <= 0
    ):
        raise CheckpointError("checkpoint manifest path or size is invalid")
    for key in ("policy_state_sha256", "config_sha256", "lineage_sha256"):
        _require_sha256(manifest[key], key)
    _require_sha256(checkpoint["sha256"], "checkpoint")
    if profile.requires_safety_contract:
        _validated_safety_binding(
            manifest,
            expected_config_sha256=manifest["config_sha256"],
        )


def _validated_safety_argument(
    value: object,
    *,
    config_sha256: str,
    lineage: Mapping[str, Any],
    profile: _CheckpointProfile,
) -> dict[str, object] | None:
    if not profile.requires_safety_contract:
        if value is not None:
            raise CheckpointError("Stage 4 checkpoint rejects a safety contract")
        return None
    if not isinstance(value, SafetyContract):
        raise CheckpointError(
            "Stage 6 checkpoint requires a safety contract (SafetyContract)"
        )
    try:
        binding = value.binding(config_sha256=config_sha256)
        SafetyContract.from_binding(
            binding,
            expected_config_sha256=config_sha256,
        )
    except (TypeError, ValueError) as exc:
        raise CheckpointError("Stage 6 checkpoint safety binding drifted") from exc
    if lineage.get("safety_contract_sha256") != value.sha256:
        raise CheckpointError("Stage 6 checkpoint safety lineage drifted")
    return binding


def _validated_safety_binding(
    value: Mapping[str, Any],
    *,
    expected_config_sha256: str,
) -> dict[str, object]:
    binding = {name: value.get(name) for name in _SAFETY_BINDING_FIELDS}
    try:
        contract = SafetyContract.from_binding(
            binding,
            expected_config_sha256=expected_config_sha256,
        )
    except (TypeError, ValueError) as exc:
        raise CheckpointError("Stage 6 checkpoint safety binding drifted") from exc
    return contract.binding(config_sha256=expected_config_sha256)


def _require_expected_safety_binding(
    payload: Mapping[str, Any],
    expected: Mapping[str, object] | None,
) -> None:
    if expected is None:
        return
    actual = {name: payload.get(name) for name in _SAFETY_BINDING_FIELDS}
    if actual != dict(expected):
        raise CheckpointError("Stage 6 checkpoint safety contract mismatch")


def _validate_complete(
    complete: object,
    *,
    expected_step: int,
    profile: _CheckpointProfile,
) -> None:
    if not isinstance(complete, dict) or set(complete) != {
        "schema_version",
        "update_step",
        "manifest_sha256",
        "checkpoint_sha256",
    }:
        raise CheckpointError("checkpoint complete marker fields are not exact")
    if (
        complete["schema_version"] != profile.complete_schema
        or complete["update_step"] != expected_step
    ):
        raise CheckpointError("checkpoint complete marker schema or step mismatch")
    _require_sha256(complete["manifest_sha256"], "manifest")
    _require_sha256(complete["checkpoint_sha256"], "checkpoint")


def _read_canonical_json(
    path: Path,
    *,
    after_read: Callable[[], None] | None = None,
) -> tuple[dict[str, Any], bytes]:
    raw = _secure_read_bytes(path, after_read=after_read)
    try:
        import json

        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise CheckpointError(f"invalid JSON checkpoint artifact: {path.name}") from exc
    if not isinstance(value, dict):
        raise CheckpointError(f"checkpoint JSON artifact is not an object: {path.name}")
    try:
        canonical = ArtifactStore.canonical_json_bytes(value)
    except (TypeError, ValueError) as exc:
        raise CheckpointError(
            f"invalid JSON checkpoint artifact: {path.name}"
        ) from exc
    if raw != canonical:
        raise CheckpointError(f"checkpoint JSON artifact is not canonical: {path.name}")
    return value, raw


def safe_load_checkpoint_payload(payload_bytes: bytes) -> object:
    """Decode trusted checkpoint bytes with PyTorch's restricted unpickler."""

    if not isinstance(payload_bytes, bytes) or not payload_bytes:
        raise CheckpointError("checkpoint payload bytes are invalid")
    safe_numpy_globals = (
        np.ndarray,
        np._core.multiarray._reconstruct,  # type: ignore[attr-defined]
        np.dtype,
        type(np.dtype(np.uint32)),
    )
    with torch.serialization.safe_globals(list(safe_numpy_globals)):
        return torch.load(
            io.BytesIO(payload_bytes),
            map_location="cpu",
            weights_only=True,
        )


def _secure_read_bytes(
    path: Path,
    *,
    after_read: Callable[[], None] | None = None,
) -> bytes:
    descriptor = -1
    try:
        require_plain_path(path, leaf_kind="file", label="checkpoint member")
        descriptor = os.open(path, _secure_open_flags(os.O_RDONLY))
        before = _regular_descriptor_identity(descriptor)
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            payload = stream.read()
        after = _regular_descriptor_identity(descriptor)
        if before != after or len(payload) != after[3]:
            raise CheckpointError("checkpoint member identity changed while read")
    except CheckpointError:
        raise
    except (OSError, PathSecurityError) as exc:
        raise CheckpointError("checkpoint member read failed closed") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)

    try:
        if after_read is not None:
            after_read()
        require_plain_path(path, leaf_kind="file", label="checkpoint member")
        if _regular_path_identity(path) != after:
            raise CheckpointError("checkpoint member identity changed while read")
    except CheckpointError:
        raise
    except (OSError, PathSecurityError) as exc:
        raise CheckpointError("checkpoint member identity changed while read") from exc
    return payload


def _write_new_fsynced(
    path: Path,
    payload: bytes,
) -> tuple[int, int, int, int, int, int]:
    if not isinstance(payload, bytes):
        raise CheckpointError("checkpoint write payload must be bytes")
    identity: tuple[int, int, int, int, int, int] | None = None
    try:
        try:
            require_plain_path(
                path,
                allow_missing=True,
                label="checkpoint new member",
            )
            with DurableParentGuard(path.parent) as parent_guard:
                descriptor = parent_guard.open_file(
                    path,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                )
                identity = guarded_file_identity(
                    descriptor,
                    path,
                    parent_guard=parent_guard,
                )
                primary_error: BaseException | None = None
                try:
                    view = memoryview(payload)
                    written = 0
                    while written < len(view):
                        count = os.write(descriptor, view[written:])
                        if count <= 0:
                            raise OSError(
                                "checkpoint descriptor write made no progress"
                            )
                        written += count
                        guarded_file_identity(
                            descriptor,
                            path,
                            parent_guard=parent_guard,
                        )
                    os.fsync(descriptor)
                    identity = guarded_file_identity(
                        descriptor,
                        path,
                        parent_guard=parent_guard,
                    )
                    if identity[3] != len(payload):
                        raise CheckpointError(
                            "checkpoint member size changed while written"
                        )
                except CheckpointError as exc:
                    primary_error = exc
                    try:
                        identity = guarded_file_identity(
                            descriptor,
                            path,
                            parent_guard=parent_guard,
                        )
                    except BaseException as identity_error:
                        exc.add_note(
                            "suppressed checkpoint identity refresh failure: "
                            f"{identity_error}"
                        )
                    raise
                except (OSError, PathSecurityError) as exc:
                    wrapped = CheckpointError(
                        "checkpoint member write failed closed"
                    )
                    primary_error = wrapped
                    try:
                        identity = guarded_file_identity(
                            descriptor,
                            path,
                            parent_guard=parent_guard,
                        )
                    except BaseException as identity_error:
                        wrapped.add_note(
                            "suppressed checkpoint identity refresh failure: "
                            f"{identity_error}"
                        )
                    raise wrapped from exc
                finally:
                    try:
                        os.close(descriptor)
                    except BaseException as close_error:
                        if primary_error is None:
                            raise
                        primary_error.add_note(
                            "suppressed checkpoint descriptor close failure: "
                            f"{close_error}"
                        )
            require_plain_path(
                path,
                leaf_kind="file",
                label="checkpoint new member",
            )
            if identity is None or _regular_path_identity(path) != identity:
                raise CheckpointError(
                    "checkpoint member identity changed while written"
                )
        except CheckpointError:
            raise
        except (OSError, PathSecurityError) as exc:
            raise CheckpointError(
                "checkpoint member identity changed while written"
            ) from exc
    except CheckpointError:
        if identity is not None:
            _remove_file_if_identity(path, identity)
        raise
    if identity is None:
        raise CheckpointError("checkpoint member identity was not captured")
    return identity


def _secure_open_flags(flags: int) -> int:
    for name in ("O_BINARY", "O_CLOEXEC", "O_NOINHERIT", "O_NOFOLLOW"):
        flags |= int(getattr(os, name, 0))
    return flags


def _regular_descriptor_identity(
    descriptor: int,
) -> tuple[int, int, int, int, int, int]:
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode):
        raise CheckpointError("checkpoint descriptor is not a regular file")
    return _regular_stat_identity(metadata)


def _regular_path_identity(path: Path) -> tuple[int, int, int, int, int, int]:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise CheckpointError("checkpoint member identity is unavailable") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise CheckpointError("checkpoint member is not a regular file")
    return _regular_stat_identity(metadata)


def _regular_stat_identity(
    metadata: os.stat_result,
) -> tuple[int, int, int, int, int, int]:
    if int(metadata.st_nlink) != 1:
        raise CheckpointError("checkpoint member hard link count must equal one")
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(stat.S_IFMT(metadata.st_mode)),
        int(metadata.st_size),
        int(getattr(metadata, "st_file_attributes", 0)),
        int(metadata.st_nlink),
    )


def _directory_path_identity(path: Path) -> tuple[int, int, int, int]:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise CheckpointError("checkpoint directory identity is unavailable") from exc
    if not stat.S_ISDIR(metadata.st_mode):
        raise CheckpointError("checkpoint directory is not a directory")
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(stat.S_IFMT(metadata.st_mode)),
        int(getattr(metadata, "st_file_attributes", 0)),
    )


def _remove_file_if_identity(
    path: Path,
    expected_identity: tuple[int, int, int, int, int, int],
) -> None:
    try:
        if os.path.lexists(path):
            durable_unlink(path, expected_identity=expected_identity)
    except (OSError, CheckpointError, PathSecurityError):
        return


def _remove_known_pending_files(
    directory: Path,
    *,
    directory_identity: tuple[int, int, int, int],
    member_identities: Mapping[str, tuple[int, int, int, int, int, int]],
    strict: bool = False,
) -> None:
    try:
        if (
            not os.path.lexists(directory)
            or _directory_path_identity(directory) != directory_identity
        ):
            return
        members = tuple(directory.iterdir())
        if {member.name for member in members} != set(member_identities):
            return
        if any(
            _regular_path_identity(member) != member_identities[member.name]
            for member in members
        ):
            return
        for member in members:
            durable_unlink(
                member,
                expected_identity=member_identities[member.name],
            )
        if (
            _directory_path_identity(directory) == directory_identity
            and not any(directory.iterdir())
        ):
            durable_rmdir(directory, expected_identity=directory_identity)
    except (OSError, CheckpointError, PathSecurityError) as exc:
        if strict:
            raise CheckpointError(
                "checkpoint pending cleanup durability failed"
            ) from exc


def _update_directory_name(update_step: int) -> str:
    return f"update-{update_step:08d}"


def _canonical_sha256(value: object) -> str:
    return _sha256(ArtifactStore.canonical_json_bytes(value))


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(value: object, name: str) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise CheckpointError(f"{name} SHA-256 is invalid")


__all__ = [
    "CHECKPOINT_COMPLETE_SCHEMA_VERSION",
    "CHECKPOINT_LATEST_SCHEMA_VERSION",
    "CHECKPOINT_MANIFEST_SCHEMA_VERSION",
    "CHECKPOINT_SCHEMA_VERSION",
    "CheckpointError",
    "CheckpointManager",
    "CheckpointReceipt",
    "LoadedCheckpoint",
    "PreparedCheckpoint",
    "inspect_complete_checkpoint_snapshot",
    "load_complete_checkpoint_snapshot",
    "safe_load_checkpoint_payload",
]

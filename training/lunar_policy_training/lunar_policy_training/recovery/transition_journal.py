"""Fsync-backed, append-only journal for committed Reward V4 macro actions."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import re
import shutil
import threading
from types import MappingProxyType
from typing import Mapping, Sequence
import uuid

import torch

from ..environment.task_area import WorkerStratum
from ..reward import (
    RewardComponentsV4,
    RewardInputsV4,
    compute_reward_components,
)
from ..reward_contract import (
    RewardStage,
    RewardTerminalClass,
    RewardWeightsV4,
    TaskScaleBucket,
)


PAYLOAD_SCHEMA_VERSION = "lunar-transition-payload/v4"
INDEX_SCHEMA_VERSION = "lunar-transition-index/v1"
SEAL_SCHEMA_VERSION = "lunar-transition-seal/v1"
APPLIED_SCHEMA_VERSION = "lunar-transition-applied/v1"
_RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_UPDATE_ROOT_PATTERN = re.compile(r"update-(\d{8})")
_PLATFORMS = frozenset(("WHEELED", "LEGGED", "HOPPER"))
_FAULTS = frozenset(("before_rename", "after_rename_before_index"))
_PAYLOAD_FIELDS = frozenset(
    (
        "schema_version",
        "update_id",
        "worker_index",
        "slot_index",
        "policy_version",
        "platform_type",
        "scale_bucket",
        "episode_id",
        "episode_transition_index",
        "transition_id",
        "observation",
        "action",
        "old_log_prob",
        "old_value",
        "reward_stage",
        "reward_weights",
        "reward_components",
        "reward_inputs",
        "done",
        "terminal_class",
        "pre_worker_audit",
        "post_worker_state",
    )
)
_INDEX_FIELDS = frozenset(
    (
        "schema_version",
        "event",
        "run_id",
        "update_id",
        "worker_index",
        "slot_index",
        "policy_version",
        "platform_type",
        "scale_bucket",
        "episode_id",
        "episode_transition_index",
        "transition_id",
        "payload_path",
        "payload_sha256",
        "file_sha256",
        "record_sha256",
    )
)
_SEAL_FIELDS = frozenset(
    (
        "schema_version",
        "state",
        "run_id",
        "update_id",
        "worker_count",
        "slots_per_worker",
        "policy_version",
        "expected_slots",
        "journal_sha256",
        "record_sha256",
    )
)
_APPLIED_FIELDS = frozenset(
    (
        "schema_version",
        "state",
        "run_id",
        "update_id",
        "journal_sha256",
        "checkpoint_payload_sha256",
        "metrics_record_sha256",
        "record_sha256",
    )
)


class JournalError(RuntimeError):
    """Base class for transition journal failures."""


class JournalValidationError(JournalError):
    """A proposed payload or seal does not satisfy the frozen contract."""


class JournalConflictError(JournalError):
    """An immutable slot or transition identity has conflicting content."""


class JournalCorruptionError(JournalError):
    """Committed on-disk evidence is incomplete or corrupted."""


class InjectedJournalFault(JournalError):
    """Test-only crash injected at a named write boundary."""


@dataclass(frozen=True, slots=True, eq=False)
class MacroTransitionPayload:
    """One complete PPO sample, compact pre-audit, and recoverable post-state."""

    update_id: int
    worker_index: int
    slot_index: int
    policy_version: int
    platform_type: str
    scale_bucket: TaskScaleBucket
    episode_id: str
    episode_transition_index: int
    transition_id: str
    observation: Mapping[str, object]
    action: Mapping[str, object]
    old_log_prob: torch.Tensor
    old_value: torch.Tensor
    reward_stage: RewardStage
    reward_weights: RewardWeightsV4
    reward_components: RewardComponentsV4
    reward_inputs: RewardInputsV4
    done: bool
    terminal_class: RewardTerminalClass
    pre_worker_audit: Mapping[str, object]
    post_worker_state: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "observation", _freeze_mapping(self.observation))
        object.__setattr__(self, "action", _freeze_mapping(self.action))
        object.__setattr__(self, "old_log_prob", _clone_tensor(self.old_log_prob))
        object.__setattr__(self, "old_value", _clone_tensor(self.old_value))
        object.__setattr__(
            self, "pre_worker_audit", _freeze_mapping(self.pre_worker_audit)
        )
        object.__setattr__(
            self, "post_worker_state", _freeze_mapping(self.post_worker_state)
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MacroTransitionPayload):
            return NotImplemented
        return _semantic_sha256(_payload_to_mapping(self)) == _semantic_sha256(
            _payload_to_mapping(other)
        )


@dataclass(frozen=True, slots=True)
class CommittedTransition:
    update_id: int
    worker_index: int
    slot_index: int
    policy_version: int
    platform_type: str
    scale_bucket: TaskScaleBucket
    episode_id: str
    episode_transition_index: int
    transition_id: str
    payload_sha256: str
    file_sha256: str
    payload_path: Path
    payload: MacroTransitionPayload


@dataclass(frozen=True, slots=True)
class LoadedUpdate:
    update_id: int
    state: str
    committed: dict[tuple[int, int], CommittedTransition]
    orphaned_payloads: tuple[Path, ...]
    policy_version: int | None
    journal_sha256: str | None
    checkpoint_payload_sha256: str | None = None
    metrics_record_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class RecoveredWorkerBoundary:
    worker_index: int
    next_slot: int
    last_transition_id: str
    episode_id: str
    episode_transition_index: int
    done: bool
    post_worker_state: Mapping[str, object]


class TransitionJournal:
    """Single-writer journal whose index is the only commit authority."""

    def __init__(
        self,
        root: str | Path,
        *,
        run_id: str,
        worker_count: int = 24,
        slots_per_worker: int = 12,
        worker_strata: Sequence[WorkerStratum] | None = None,
        fault_injection: str | None = None,
    ) -> None:
        if not isinstance(root, (str, Path)):
            raise JournalValidationError("journal root is invalid")
        if not isinstance(run_id, str) or _RUN_ID_PATTERN.fullmatch(run_id) is None:
            raise JournalValidationError("journal run ID is invalid")
        if type(worker_count) is not int or worker_count <= 0:
            raise JournalValidationError("journal worker count is invalid")
        if type(slots_per_worker) is not int or slots_per_worker <= 0:
            raise JournalValidationError("journal slots per worker is invalid")
        if fault_injection is not None and fault_injection not in _FAULTS:
            raise JournalValidationError("journal fault injection is invalid")

        self._base_root = Path(root).expanduser().resolve()
        self.run_id = run_id
        self.worker_count = worker_count
        self.slots_per_worker = slots_per_worker
        self.recovery_root = self._base_root / run_id / "recovery"
        self.expected_slots = frozenset(
            (worker, slot)
            for worker in range(worker_count)
            for slot in range(slots_per_worker)
        )
        self._fault_injection = fault_injection
        self._lock = threading.RLock()
        self._worker_strata = _validate_worker_strata(
            worker_strata, worker_count=worker_count
        )

    def commit(self, payload: MacroTransitionPayload) -> CommittedTransition:
        """Durably commit one payload; identical retries return the first record."""
        with self._lock:
            self._validate_payload(payload)
            loaded = self.load_update(payload.update_id)
            if loaded.state != "OPEN":
                raise JournalConflictError("sealed update cannot accept another slot")

            body = _payload_to_mapping(payload)
            payload_sha256 = _semantic_sha256(body)
            key = (payload.worker_index, payload.slot_index)
            existing = loaded.committed.get(key)
            if existing is not None:
                if (
                    existing.payload_sha256 == payload_sha256
                    and existing.transition_id == payload.transition_id
                ):
                    return existing
                raise JournalConflictError("journal slot already has different payload")
            if any(
                item.transition_id == payload.transition_id
                for item in loaded.committed.values()
            ):
                raise JournalConflictError(
                    "journal transition ID is already committed to another slot"
                )

            payload_path = self._payload_path(payload)
            payload_path.parent.mkdir(parents=True, exist_ok=True)
            if payload_path.exists():
                file_sha256, orphan = self._read_payload_file(payload_path)
                if _semantic_sha256(_payload_to_mapping(orphan)) != payload_sha256:
                    raise JournalConflictError(
                        "uncommitted journal slot contains different payload"
                    )
            else:
                file_sha256 = self._write_payload_atomic(payload_path, body)

            if self._fault_injection == "after_rename_before_index":
                raise InjectedJournalFault("after_rename_before_index")

            record = self._index_record(
                payload,
                payload_path=payload_path,
                payload_sha256=payload_sha256,
                file_sha256=file_sha256,
            )
            self._append_index(self._index_path(payload.update_id), record)
            committed = self.load_update(payload.update_id).committed.get(key)
            if committed is None:
                raise JournalCorruptionError("committed slot is not index reachable")
            return committed

    def load_update(self, update_id: int) -> LoadedUpdate:
        """Load only index-reachable, hash-valid committed transitions."""
        with self._lock:
            _require_nonnegative_int(update_id, "update ID")
            update_root = self._update_root(update_id)
            records = self._read_index(self._index_path(update_id))
            committed: dict[tuple[int, int], CommittedTransition] = {}
            transition_ids: set[str] = set()
            referenced: set[Path] = set()
            for record in records:
                item = self._load_committed_record(record, update_id=update_id)
                key = (item.worker_index, item.slot_index)
                if key in committed:
                    raise JournalCorruptionError("journal index repeats a slot")
                if item.transition_id in transition_ids:
                    raise JournalCorruptionError("journal index repeats a transition ID")
                committed[key] = item
                transition_ids.add(item.transition_id)
                referenced.add(item.payload_path.resolve())

            orphaned = tuple(
                sorted(
                    (
                        path
                        for path in update_root.glob("worker-*/slot-*.pt")
                        if path.resolve() not in referenced
                    ),
                    key=lambda path: path.as_posix(),
                )
            )
            versions = {item.policy_version for item in committed.values()}
            policy_version = next(iter(versions)) if len(versions) == 1 else None
            state = "OPEN"
            journal_sha256: str | None = None
            checkpoint_payload_sha256: str | None = None
            metrics_record_sha256: str | None = None
            seal_path = self._seal_path(update_id)
            if seal_path.exists():
                seal = self._read_seal(seal_path)
                journal_sha256 = self._validate_seal(
                    seal,
                    update_id=update_id,
                    committed=committed,
                    orphaned=orphaned,
                )
                state = "SEALED"
                policy_version = int(seal["policy_version"])
                applied_path = self._applied_path(update_id)
                if applied_path.exists():
                    applied = self._read_applied(applied_path)
                    (
                        checkpoint_payload_sha256,
                        metrics_record_sha256,
                    ) = self._validate_applied(
                        applied,
                        update_id=update_id,
                        journal_sha256=journal_sha256,
                    )
                    state = "APPLIED"
            return LoadedUpdate(
                update_id=update_id,
                state=state,
                committed=committed,
                orphaned_payloads=orphaned,
                policy_version=policy_version,
                journal_sha256=journal_sha256,
                checkpoint_payload_sha256=checkpoint_payload_sha256,
                metrics_record_sha256=metrics_record_sha256,
            )

    def seal_update(
        self,
        *,
        update_id: int,
        expected_slots: set[tuple[int, int]] | frozenset[tuple[int, int]],
    ) -> LoadedUpdate:
        """Atomically seal one complete, internally continuous update."""
        with self._lock:
            _require_nonnegative_int(update_id, "update ID")
            if not isinstance(expected_slots, (set, frozenset)):
                raise JournalValidationError("journal expected slots are invalid")
            if frozenset(expected_slots) != self.expected_slots:
                raise JournalValidationError(
                    "journal expected slots do not match the configured grid"
                )
            loaded = self.load_update(update_id)
            if loaded.state in ("SEALED", "APPLIED"):
                return loaded
            if set(loaded.committed) != set(self.expected_slots):
                raise JournalValidationError("journal update is incomplete")
            if loaded.orphaned_payloads:
                raise JournalValidationError("journal update contains orphan payloads")
            policy_version = self._validate_sealable(loaded.committed)
            journal_sha256 = _journal_sha256(loaded.committed)
            seal_body: dict[str, object] = {
                "schema_version": SEAL_SCHEMA_VERSION,
                "state": "SEALED",
                "run_id": self.run_id,
                "update_id": update_id,
                "worker_count": self.worker_count,
                "slots_per_worker": self.slots_per_worker,
                "policy_version": policy_version,
                "expected_slots": [
                    [worker, slot] for worker, slot in sorted(self.expected_slots)
                ],
                "journal_sha256": journal_sha256,
            }
            seal_body["record_sha256"] = _json_sha256(seal_body)
            self._write_json_atomic(self._seal_path(update_id), seal_body)
            return self.load_update(update_id)

    def mark_update_applied(
        self,
        *,
        update_id: int,
        checkpoint_payload_sha256: str,
        metrics_record_sha256: str,
    ) -> LoadedUpdate:
        """Bind one sealed journal to its immutable checkpoint and metrics."""
        with self._lock:
            _require_nonnegative_int(update_id, "update ID")
            if not _is_sha256(checkpoint_payload_sha256) or not _is_sha256(
                metrics_record_sha256
            ):
                raise JournalValidationError(
                    "journal applied artifact hash is invalid"
                )
            loaded = self.load_update(update_id)
            if loaded.state == "OPEN":
                raise JournalValidationError(
                    "open update cannot be marked applied"
                )
            if loaded.state == "APPLIED":
                if (
                    loaded.checkpoint_payload_sha256
                    != checkpoint_payload_sha256
                    or loaded.metrics_record_sha256
                    != metrics_record_sha256
                ):
                    raise JournalConflictError(
                        "applied update has different artifact identity"
                    )
                return loaded
            body: dict[str, object] = {
                "schema_version": APPLIED_SCHEMA_VERSION,
                "state": "APPLIED",
                "run_id": self.run_id,
                "update_id": update_id,
                "journal_sha256": loaded.journal_sha256,
                "checkpoint_payload_sha256": checkpoint_payload_sha256,
                "metrics_record_sha256": metrics_record_sha256,
            }
            body["record_sha256"] = _json_sha256(body)
            self._write_json_atomic(self._applied_path(update_id), body)
            return self.load_update(update_id)

    def prune_applied_updates(self, *, keep_latest: int) -> tuple[int, ...]:
        """Remove only superseded, fully applied update directories.

        Callers invoke this only after the corresponding run manifest has
        advanced, so the remaining newest applied update and any open update
        preserve the crash-recovery boundary.
        """
        if type(keep_latest) is not int or keep_latest < 1:
            raise JournalValidationError("journal retention count is invalid")
        with self._lock:
            if not self.recovery_root.exists():
                return ()
            recovery_root = self.recovery_root.resolve()
            applied: list[tuple[int, Path]] = []
            for path in sorted(self.recovery_root.iterdir(), key=lambda item: item.name):
                matched = _UPDATE_ROOT_PATTERN.fullmatch(path.name)
                if matched is None:
                    continue
                if path.is_symlink() or not path.is_dir():
                    raise JournalCorruptionError("journal update root is invalid")
                update_id = int(matched.group(1))
                applied_path = path / "applied.json"
                if not applied_path.exists():
                    continue
                marker = self._read_applied(applied_path)
                seal = self._read_seal(path / "seal.json")
                if (
                    marker["run_id"] != self.run_id
                    or marker["update_id"] != update_id
                    or marker["state"] != "APPLIED"
                    or seal["run_id"] != self.run_id
                    or seal["update_id"] != update_id
                    or seal["state"] != "SEALED"
                    or marker["journal_sha256"] != seal["journal_sha256"]
                ):
                    raise JournalCorruptionError("journal applied root is invalid")
                applied.append((update_id, path))

            prunable = applied[:-keep_latest]
            removed: list[int] = []
            for update_id, path in prunable:
                target = path.resolve()
                if target.parent != recovery_root or target == recovery_root:
                    raise JournalCorruptionError("journal prune target is invalid")
                shutil.rmtree(target)
                _fsync_directory(recovery_root)
                removed.append(update_id)
            return tuple(removed)

    def recover_worker_boundaries(
        self, *, update_id: int
    ) -> dict[int, RecoveredWorkerBoundary]:
        """Return the last contiguous committed boundary for every active worker."""
        loaded = self.load_update(update_id)
        by_worker: dict[int, list[CommittedTransition]] = {}
        for item in loaded.committed.values():
            by_worker.setdefault(item.worker_index, []).append(item)
        recovered: dict[int, RecoveredWorkerBoundary] = {}
        for worker, items in by_worker.items():
            ordered = sorted(items, key=lambda item: item.slot_index)
            if [item.slot_index for item in ordered] != list(range(len(ordered))):
                raise JournalCorruptionError(
                    "worker committed slots are not a contiguous prefix"
                )
            last = ordered[-1]
            recovered[worker] = RecoveredWorkerBoundary(
                worker_index=worker,
                next_slot=last.slot_index + 1,
                last_transition_id=last.transition_id,
                episode_id=last.episode_id,
                episode_transition_index=last.episode_transition_index,
                done=last.payload.done,
                post_worker_state=last.payload.post_worker_state,
            )
        return recovered

    def _validate_payload(self, payload: MacroTransitionPayload) -> None:
        _validate_payload(payload)
        if not 0 <= payload.worker_index < self.worker_count:
            raise JournalValidationError("journal worker index is out of range")
        if not 0 <= payload.slot_index < self.slots_per_worker:
            raise JournalValidationError("journal slot index is out of range")
        expected = self._worker_strata.get(payload.worker_index)
        if expected is not None and (
            payload.platform_type != expected[0]
            or payload.scale_bucket is not expected[1]
        ):
            raise JournalValidationError("journal worker stratum does not match")

    def _validate_sealable(
        self, committed: Mapping[tuple[int, int], CommittedTransition]
    ) -> int:
        versions = {item.policy_version for item in committed.values()}
        if len(versions) != 1:
            raise JournalValidationError("journal policy versions do not match")
        for worker in range(self.worker_count):
            ordered = [committed[(worker, slot)] for slot in range(self.slots_per_worker)]
            strata = {(item.platform_type, item.scale_bucket) for item in ordered}
            if len(strata) != 1:
                raise JournalValidationError("journal worker stratum changed within update")
            expected = self._worker_strata.get(worker)
            if expected is not None and next(iter(strata)) != expected:
                raise JournalValidationError("journal worker stratum does not match")
            for previous, current in zip(ordered, ordered[1:]):
                if previous.episode_id == current.episode_id:
                    if previous.payload.done or (
                        current.episode_transition_index
                        != previous.episode_transition_index + 1
                    ):
                        raise JournalValidationError(
                            "journal episode transition identity is discontinuous"
                        )
                elif (
                    not previous.payload.done
                    or current.episode_transition_index != 0
                ):
                    raise JournalValidationError(
                        "journal episode transition identity is discontinuous"
                    )
        return next(iter(versions))

    def _payload_path(self, payload: MacroTransitionPayload) -> Path:
        return (
            self._update_root(payload.update_id)
            / f"worker-{payload.worker_index:03d}"
            / f"slot-{payload.slot_index:03d}.pt"
        )

    def _update_root(self, update_id: int) -> Path:
        return self.recovery_root / f"update-{update_id:08d}"

    def _index_path(self, update_id: int) -> Path:
        return self._update_root(update_id) / "index.jsonl"

    def _seal_path(self, update_id: int) -> Path:
        return self._update_root(update_id) / "seal.json"

    def _applied_path(self, update_id: int) -> Path:
        return self._update_root(update_id) / "applied.json"

    def _write_payload_atomic(
        self, payload_path: Path, body: Mapping[str, object]
    ) -> str:
        temporary = payload_path.with_name(
            f".{payload_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )
        try:
            with temporary.open("xb") as stream:
                torch.save(dict(body), stream)
                stream.flush()
                os.fsync(stream.fileno())
            if self._fault_injection == "before_rename":
                raise InjectedJournalFault("before_rename")
            os.replace(temporary, payload_path)
            _fsync_directory(payload_path.parent)
            return _file_sha256(payload_path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def _index_record(
        self,
        payload: MacroTransitionPayload,
        *,
        payload_path: Path,
        payload_sha256: str,
        file_sha256: str,
    ) -> dict[str, object]:
        relative_path = payload_path.relative_to(self.recovery_root).as_posix()
        record: dict[str, object] = {
            "schema_version": INDEX_SCHEMA_VERSION,
            "event": "COMMITTED",
            "run_id": self.run_id,
            "update_id": payload.update_id,
            "worker_index": payload.worker_index,
            "slot_index": payload.slot_index,
            "policy_version": payload.policy_version,
            "platform_type": payload.platform_type,
            "scale_bucket": payload.scale_bucket.value,
            "episode_id": payload.episode_id,
            "episode_transition_index": payload.episode_transition_index,
            "transition_id": payload.transition_id,
            "payload_path": relative_path,
            "payload_sha256": payload_sha256,
            "file_sha256": file_sha256,
        }
        record["record_sha256"] = _json_sha256(record)
        return record

    def _append_index(self, index_path: Path, record: Mapping[str, object]) -> None:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        encoded = _canonical_json(record) + b"\n"
        descriptor = os.open(
            index_path,
            os.O_WRONLY | os.O_CREAT | os.O_APPEND,
            0o600,
        )
        try:
            offset = 0
            while offset < len(encoded):
                written = os.write(descriptor, encoded[offset:])
                if written <= 0:
                    raise OSError("journal index write made no progress")
                offset += written
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        _fsync_directory(index_path.parent)

    def _read_index(self, index_path: Path) -> list[dict[str, object]]:
        if not index_path.exists():
            return []
        records: list[dict[str, object]] = []
        try:
            lines = index_path.read_bytes().splitlines()
        except OSError as error:
            raise JournalCorruptionError("journal index cannot be read") from error
        for line_number, line in enumerate(lines, start=1):
            if not line:
                raise JournalCorruptionError("journal index contains an empty record")
            try:
                record = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise JournalCorruptionError(
                    f"journal index record {line_number} is invalid"
                ) from error
            if not isinstance(record, dict) or set(record) != _INDEX_FIELDS:
                raise JournalCorruptionError("journal index record structure is invalid")
            expected_sha = record["record_sha256"]
            unsigned = dict(record)
            del unsigned["record_sha256"]
            if not _is_sha256(expected_sha) or _json_sha256(unsigned) != expected_sha:
                raise JournalCorruptionError("journal index record hash is invalid")
            records.append(record)
        return records

    def _load_committed_record(
        self, record: Mapping[str, object], *, update_id: int
    ) -> CommittedTransition:
        if (
            record["schema_version"] != INDEX_SCHEMA_VERSION
            or record["event"] != "COMMITTED"
            or record["run_id"] != self.run_id
            or record["update_id"] != update_id
        ):
            raise JournalCorruptionError("journal index identity is invalid")
        worker = _index_int(record["worker_index"], "worker")
        slot = _index_int(record["slot_index"], "slot")
        if not 0 <= worker < self.worker_count or not 0 <= slot < self.slots_per_worker:
            raise JournalCorruptionError("journal index slot is out of range")
        expected_path = (
            Path(f"update-{update_id:08d}")
            / f"worker-{worker:03d}"
            / f"slot-{slot:03d}.pt"
        )
        if record["payload_path"] != expected_path.as_posix():
            raise JournalCorruptionError("journal payload path is invalid")
        payload_path = self.recovery_root / expected_path
        if not payload_path.is_file():
            raise JournalCorruptionError("journal committed payload is missing")
        expected_file_sha = record["file_sha256"]
        if not _is_sha256(expected_file_sha):
            raise JournalCorruptionError("journal file hash is invalid")
        if _file_sha256(payload_path) != expected_file_sha:
            raise JournalCorruptionError("journal payload file hash mismatch")
        _, payload = self._read_payload_file(payload_path)
        self._validate_payload(payload)
        payload_sha = _semantic_sha256(_payload_to_mapping(payload))
        if not _is_sha256(record["payload_sha256"]) or payload_sha != record["payload_sha256"]:
            raise JournalCorruptionError("journal payload body hash mismatch")
        expected_values = {
            "update_id": payload.update_id,
            "worker_index": payload.worker_index,
            "slot_index": payload.slot_index,
            "policy_version": payload.policy_version,
            "platform_type": payload.platform_type,
            "scale_bucket": payload.scale_bucket.value,
            "episode_id": payload.episode_id,
            "episode_transition_index": payload.episode_transition_index,
            "transition_id": payload.transition_id,
        }
        if any(record[name] != value for name, value in expected_values.items()):
            raise JournalCorruptionError("journal index and payload identity disagree")
        return CommittedTransition(
            update_id=payload.update_id,
            worker_index=payload.worker_index,
            slot_index=payload.slot_index,
            policy_version=payload.policy_version,
            platform_type=payload.platform_type,
            scale_bucket=payload.scale_bucket,
            episode_id=payload.episode_id,
            episode_transition_index=payload.episode_transition_index,
            transition_id=payload.transition_id,
            payload_sha256=payload_sha,
            file_sha256=str(expected_file_sha),
            payload_path=payload_path,
            payload=payload,
        )

    def _read_payload_file(
        self, payload_path: Path
    ) -> tuple[str, MacroTransitionPayload]:
        try:
            body = torch.load(
                payload_path,
                weights_only=True,
                map_location="cpu",
            )
        except (
            OSError,
            RuntimeError,
            EOFError,
            ValueError,
            pickle.UnpicklingError,
        ) as error:
            raise JournalCorruptionError("journal payload cannot be loaded safely") from error
        try:
            payload = _payload_from_mapping(body)
        except JournalValidationError as error:
            raise JournalCorruptionError("journal payload structure is invalid") from error
        return _file_sha256(payload_path), payload

    def _write_json_atomic(
        self, target: Path, body: Mapping[str, object]
    ) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(
            f".{target.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )
        try:
            with temporary.open("xb") as stream:
                stream.write(_canonical_json(body) + b"\n")
                stream.flush()
                os.fsync(stream.fileno())
            if target.exists():
                raise JournalConflictError("journal update is already sealed")
            os.replace(temporary, target)
            _fsync_directory(target.parent)
        finally:
            if temporary.exists():
                temporary.unlink()

    def _read_seal(self, path: Path) -> dict[str, object]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise JournalCorruptionError("journal seal cannot be read") from error
        if not isinstance(value, dict) or set(value) != _SEAL_FIELDS:
            raise JournalCorruptionError("journal seal structure is invalid")
        expected_sha = value["record_sha256"]
        unsigned = dict(value)
        del unsigned["record_sha256"]
        if not _is_sha256(expected_sha) or _json_sha256(unsigned) != expected_sha:
            raise JournalCorruptionError("journal seal hash is invalid")
        return value

    def _read_applied(self, path: Path) -> dict[str, object]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise JournalCorruptionError(
                "journal applied marker cannot be read"
            ) from error
        if not isinstance(value, dict) or set(value) != _APPLIED_FIELDS:
            raise JournalCorruptionError(
                "journal applied marker structure is invalid"
            )
        expected_sha = value["record_sha256"]
        unsigned = dict(value)
        del unsigned["record_sha256"]
        if not _is_sha256(expected_sha) or _json_sha256(unsigned) != expected_sha:
            raise JournalCorruptionError(
                "journal applied marker hash is invalid"
            )
        return value

    def _validate_seal(
        self,
        seal: Mapping[str, object],
        *,
        update_id: int,
        committed: Mapping[tuple[int, int], CommittedTransition],
        orphaned: tuple[Path, ...],
    ) -> str:
        if (
            seal["schema_version"] != SEAL_SCHEMA_VERSION
            or seal["state"] != "SEALED"
            or seal["run_id"] != self.run_id
            or seal["update_id"] != update_id
            or seal["worker_count"] != self.worker_count
            or seal["slots_per_worker"] != self.slots_per_worker
            or seal["expected_slots"]
            != [[worker, slot] for worker, slot in sorted(self.expected_slots)]
        ):
            raise JournalCorruptionError("journal seal identity is invalid")
        if set(committed) != set(self.expected_slots) or orphaned:
            raise JournalCorruptionError("sealed journal is incomplete")
        try:
            policy_version = self._validate_sealable(committed)
        except JournalValidationError as error:
            raise JournalCorruptionError("sealed journal semantics are invalid") from error
        if seal["policy_version"] != policy_version:
            raise JournalCorruptionError("journal seal policy version is invalid")
        journal_sha256 = _journal_sha256(committed)
        if seal["journal_sha256"] != journal_sha256:
            raise JournalCorruptionError("journal seal content hash is invalid")
        return journal_sha256

    def _validate_applied(
        self,
        applied: Mapping[str, object],
        *,
        update_id: int,
        journal_sha256: str,
    ) -> tuple[str, str]:
        if (
            applied["schema_version"] != APPLIED_SCHEMA_VERSION
            or applied["state"] != "APPLIED"
            or applied["run_id"] != self.run_id
            or applied["update_id"] != update_id
            or applied["journal_sha256"] != journal_sha256
            or not _is_sha256(applied["checkpoint_payload_sha256"])
            or not _is_sha256(applied["metrics_record_sha256"])
        ):
            raise JournalCorruptionError(
                "journal applied marker identity is invalid"
            )
        return (
            str(applied["checkpoint_payload_sha256"]),
            str(applied["metrics_record_sha256"]),
        )


def _validate_worker_strata(
    worker_strata: Sequence[WorkerStratum] | None, *, worker_count: int
) -> dict[int, tuple[str, TaskScaleBucket]]:
    if worker_strata is None:
        return {}
    if not isinstance(worker_strata, Sequence) or isinstance(worker_strata, (str, bytes)):
        raise JournalValidationError("journal worker strata are invalid")
    parsed: dict[int, tuple[str, TaskScaleBucket]] = {}
    for item in worker_strata:
        if not isinstance(item, WorkerStratum):
            raise JournalValidationError("journal worker stratum is invalid")
        if item.worker_index in parsed:
            raise JournalValidationError("journal worker strata repeat an index")
        parsed[item.worker_index] = (item.platform_type, item.scale_bucket)
    if set(parsed) != set(range(worker_count)):
        raise JournalValidationError("journal worker strata are incomplete")
    return parsed


_PRE_WORKER_AUDIT_FIELDS = frozenset(
    (
        "worker_index",
        "platform_type",
        "episode_id",
        "platform_task_key_sha256",
        "physical_snapshot_id",
        "coverable_detail_cell_count",
        "observed_coverable_detail_cell_count",
        "observed_coverable_mask_sha256",
        "candidate_refresh_elapsed_s",
        "global_search_elapsed_s",
        "fine_pose_candidate_count",
        "globally_reachable_candidate_count",
    )
)


def build_pre_worker_audit(state: Mapping[str, object]) -> Mapping[str, object]:
    """Project just the pre-action facts used by durable update audits."""
    if not isinstance(state, Mapping):
        raise JournalValidationError("pre worker state is invalid")
    try:
        identity = state["observation_identity"]
        decision = state["candidate_decision_snapshot"]
        if not isinstance(identity, Mapping) or not isinstance(decision, Mapping):
            raise TypeError
        audit: dict[str, object] = {
            "worker_index": state["worker_index"],
            "platform_type": state["platform_type"],
            "episode_id": identity["episode_id"],
            "platform_task_key_sha256": state["platform_task_key_sha256"],
            "physical_snapshot_id": state["physical_snapshot_id"],
            "coverable_detail_cell_count": state["coverable_detail_cell_count"],
            "observed_coverable_detail_cell_count": state[
                "observed_coverable_detail_cell_count"
            ],
            "observed_coverable_mask_sha256": state[
                "observed_coverable_mask_sha256"
            ],
            "candidate_refresh_elapsed_s": decision[
                "candidate_refresh_elapsed_s"
            ],
            "global_search_elapsed_s": decision["global_search_elapsed_s"],
            "fine_pose_candidate_count": decision["fine_pose_candidate_count"],
            "globally_reachable_candidate_count": decision[
                "globally_reachable_candidate_count"
            ],
        }
    except (KeyError, TypeError) as error:
        raise JournalValidationError("pre worker state lacks audit facts") from error
    _validate_pre_worker_audit(audit)
    return audit


def _validate_pre_worker_audit(value: object) -> None:
    if not isinstance(value, Mapping) or set(value) != _PRE_WORKER_AUDIT_FIELDS:
        raise JournalValidationError("pre worker audit is invalid")
    worker = value["worker_index"]
    if type(worker) is not int or worker < 0:
        raise JournalValidationError("pre worker audit worker is invalid")
    if value["platform_type"] not in _PLATFORMS:
        raise JournalValidationError("pre worker audit platform is invalid")
    episode_id = value["episode_id"]
    if not isinstance(episode_id, str) or not episode_id:
        raise JournalValidationError("pre worker audit episode is invalid")
    for name in (
        "platform_task_key_sha256",
        "physical_snapshot_id",
        "observed_coverable_mask_sha256",
    ):
        if not _is_sha256(value[name]):
            raise JournalValidationError("pre worker audit hash is invalid")
    total = value["coverable_detail_cell_count"]
    observed = value["observed_coverable_detail_cell_count"]
    if (
        type(total) is not int
        or total <= 0
        or type(observed) is not int
        or not 0 <= observed <= total
    ):
        raise JournalValidationError("pre worker audit coverage is invalid")
    for name in (
        "candidate_refresh_elapsed_s",
        "global_search_elapsed_s",
    ):
        elapsed = value[name]
        if (
            not isinstance(elapsed, (int, float))
            or isinstance(elapsed, bool)
            or not math.isfinite(float(elapsed))
            or elapsed < 0.0
        ):
            raise JournalValidationError("pre worker audit elapsed is invalid")
    for name in (
        "fine_pose_candidate_count",
        "globally_reachable_candidate_count",
    ):
        count = value[name]
        if type(count) is not int or count < 0:
            raise JournalValidationError("pre worker audit count is invalid")


def _validate_payload(payload: MacroTransitionPayload) -> None:
    if not isinstance(payload, MacroTransitionPayload):
        raise JournalValidationError("journal payload type is invalid")
    for name in (
        "update_id",
        "worker_index",
        "slot_index",
        "policy_version",
        "episode_transition_index",
    ):
        _require_nonnegative_int(getattr(payload, name), name.replace("_", " "))
    if payload.platform_type not in _PLATFORMS:
        raise JournalValidationError("journal payload platform is invalid")
    if not isinstance(payload.scale_bucket, TaskScaleBucket):
        raise JournalValidationError("journal payload scale bucket is invalid")
    for name in ("episode_id", "transition_id"):
        value = getattr(payload, name)
        if not isinstance(value, str) or not value or len(value) > 512:
            raise JournalValidationError(f"journal payload {name} is invalid")
    _validate_safe_mapping(payload.observation, "observation", require_tensor=True)
    _validate_safe_mapping(payload.action, "action", require_tensor=True)
    _validate_scalar_tensor(payload.old_log_prob, "old log probability")
    _validate_scalar_tensor(payload.old_value, "old value")
    if not isinstance(payload.reward_stage, RewardStage):
        raise JournalValidationError("journal payload reward stage is invalid")
    if not isinstance(payload.reward_weights, RewardWeightsV4):
        raise JournalValidationError("journal payload reward weights are invalid")
    _validate_reward_components(payload.reward_components, payload.reward_stage)
    if not isinstance(payload.reward_inputs, RewardInputsV4):
        raise JournalValidationError("journal reward inputs are invalid")
    try:
        expected_components = compute_reward_components(
            payload.reward_inputs,
            payload.reward_stage,
            payload.reward_weights,
        )
    except (TypeError, ValueError, RuntimeError) as error:
        raise JournalValidationError("journal reward inputs are invalid") from error
    if expected_components != payload.reward_components:
        raise JournalValidationError(
            "journal reward inputs and components disagree"
        )
    if payload.reward_inputs.platform_type != payload.platform_type:
        raise JournalValidationError("journal reward input platform differs")
    if type(payload.done) is not bool:
        raise JournalValidationError("journal payload done flag is invalid")
    if not isinstance(payload.terminal_class, RewardTerminalClass):
        raise JournalValidationError("journal payload terminal class is invalid")
    if payload.terminal_class is RewardTerminalClass.INVALID_TRANSITION:
        raise JournalValidationError("invalid transition cannot enter journal")
    terminal = payload.terminal_class in {
        RewardTerminalClass.SUCCESS,
        RewardTerminalClass.VALID_INCOMPLETE_TERMINAL,
    }
    if payload.done != terminal:
        raise JournalValidationError("journal done and terminal class disagree")
    _validate_pre_worker_audit(payload.pre_worker_audit)
    if (
        payload.pre_worker_audit["worker_index"] != payload.worker_index
        or payload.pre_worker_audit["platform_type"] != payload.platform_type
        or payload.pre_worker_audit["episode_id"] != payload.episode_id
    ):
        raise JournalValidationError("pre worker audit identity differs")
    _validate_safe_mapping(payload.post_worker_state, "post worker state")


def _validate_reward_components(
    components: RewardComponentsV4, stage: RewardStage
) -> None:
    if not isinstance(components, RewardComponentsV4):
        raise JournalValidationError("journal reward components are invalid")
    values = (
        components.coverage,
        components.success,
        components.terminal_gap,
        components.priority,
        components.path,
        components.total,
    )
    if any(
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        for value in values
    ):
        raise JournalValidationError("journal reward component is non-finite")
    expected_total = math.fsum(float(value) for value in values[:5])
    if not math.isclose(float(components.total), expected_total, rel_tol=1e-12, abs_tol=1e-12):
        raise JournalValidationError("journal reward component total is inconsistent")
    if stage is RewardStage.R1 and (components.priority != 0.0 or components.path != 0.0):
        raise JournalValidationError("R1 journal payload contains efficiency reward")


def _validate_safe_mapping(
    value: object, name: str, *, require_tensor: bool = False
) -> None:
    if not isinstance(value, Mapping) or not value:
        raise JournalValidationError(f"journal {name} mapping is invalid")
    tensor_count = _validate_safe_value(value, name)
    if require_tensor and tensor_count == 0:
        raise JournalValidationError(f"journal {name} contains no tensor")


def _validate_safe_value(value: object, name: str) -> int:
    if isinstance(value, torch.Tensor):
        _validate_tensor(value, name)
        return 1
    if value is None or type(value) in (bool, int, str):
        return 0
    if type(value) is float:
        if not math.isfinite(value):
            raise JournalValidationError(f"journal {name} contains non-finite data")
        return 0
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) or not key for key in value):
            raise JournalValidationError(f"journal {name} mapping key is invalid")
        return sum(_validate_safe_value(item, name) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_validate_safe_value(item, name) for item in value)
    raise JournalValidationError(f"journal {name} contains unsafe data")


def _validate_tensor(value: torch.Tensor, name: str) -> None:
    if value.layout is not torch.strided or value.is_quantized:
        raise JournalValidationError(f"journal {name} tensor layout is unsafe")
    if (value.is_floating_point() or value.is_complex()) and not bool(
        torch.isfinite(value).all().item()
    ):
        raise JournalValidationError(f"journal {name} tensor is non-finite")


def _validate_scalar_tensor(value: object, name: str) -> None:
    if not isinstance(value, torch.Tensor):
        raise JournalValidationError(f"journal {name} must be a tensor")
    _validate_tensor(value, name)
    if value.numel() != 1 or not value.is_floating_point():
        raise JournalValidationError(f"journal {name} must be one floating scalar")


def _payload_to_mapping(payload: MacroTransitionPayload) -> dict[str, object]:
    components = payload.reward_components
    return {
        "schema_version": PAYLOAD_SCHEMA_VERSION,
        "update_id": payload.update_id,
        "worker_index": payload.worker_index,
        "slot_index": payload.slot_index,
        "policy_version": payload.policy_version,
        "platform_type": payload.platform_type,
        "scale_bucket": payload.scale_bucket.value,
        "episode_id": payload.episode_id,
        "episode_transition_index": payload.episode_transition_index,
        "transition_id": payload.transition_id,
        "observation": _storage_value(payload.observation),
        "action": _storage_value(payload.action),
        "old_log_prob": _storage_value(payload.old_log_prob),
        "old_value": _storage_value(payload.old_value),
        "reward_stage": payload.reward_stage.value,
        "reward_weights": {
            "coverage": payload.reward_weights.coverage,
            "success": payload.reward_weights.success,
            "terminal_gap": payload.reward_weights.terminal_gap,
            "priority": payload.reward_weights.priority,
            "path": payload.reward_weights.path,
        },
        "reward_components": {
            "coverage": components.coverage,
            "success": components.success,
            "terminal_gap": components.terminal_gap,
            "priority": components.priority,
            "path": components.path,
            "total": components.total,
        },
        "reward_inputs": {
            "platform_type": payload.reward_inputs.platform_type,
            "coverage_before": payload.reward_inputs.coverage_before,
            "coverage_after": payload.reward_inputs.coverage_after,
            "priority_before": payload.reward_inputs.priority_before,
            "priority_after": payload.reward_inputs.priority_after,
            "path_before_m": payload.reward_inputs.path_before_m,
            "path_after_m": payload.reward_inputs.path_after_m,
            "task_scale_m": payload.reward_inputs.task_scale_m,
            "terminal_class": payload.reward_inputs.terminal_class.value,
            "success_first_crossing": (
                payload.reward_inputs.success_first_crossing
            ),
        },
        "done": payload.done,
        "terminal_class": payload.terminal_class.value,
        "pre_worker_audit": _storage_value(payload.pre_worker_audit),
        "post_worker_state": _storage_value(payload.post_worker_state),
    }


def _payload_from_mapping(value: object) -> MacroTransitionPayload:
    if not isinstance(value, Mapping) or set(value) != _PAYLOAD_FIELDS:
        raise JournalValidationError("journal payload mapping is invalid")
    if value["schema_version"] != PAYLOAD_SCHEMA_VERSION:
        raise JournalValidationError("journal payload schema is invalid")
    reward_raw = value["reward_components"]
    reward_inputs_raw = value["reward_inputs"]
    reward_weights_raw = value["reward_weights"]
    reward_fields = {
        "coverage",
        "success",
        "terminal_gap",
        "priority",
        "path",
        "total",
    }
    if not isinstance(reward_raw, Mapping) or set(reward_raw) != reward_fields:
        raise JournalValidationError("journal reward component mapping is invalid")
    reward_input_fields = {
        "platform_type",
        "coverage_before",
        "coverage_after",
        "priority_before",
        "priority_after",
        "path_before_m",
        "path_after_m",
        "task_scale_m",
        "terminal_class",
        "success_first_crossing",
    }
    if (
        not isinstance(reward_inputs_raw, Mapping)
        or set(reward_inputs_raw) != reward_input_fields
    ):
        raise JournalValidationError("journal reward input mapping is invalid")
    reward_weight_fields = {
        "coverage",
        "success",
        "terminal_gap",
        "priority",
        "path",
    }
    if (
        not isinstance(reward_weights_raw, Mapping)
        or set(reward_weights_raw) != reward_weight_fields
    ):
        raise JournalValidationError("journal reward weight mapping is invalid")
    try:
        payload = MacroTransitionPayload(
            update_id=value["update_id"],
            worker_index=value["worker_index"],
            slot_index=value["slot_index"],
            policy_version=value["policy_version"],
            platform_type=value["platform_type"],
            scale_bucket=TaskScaleBucket(value["scale_bucket"]),
            episode_id=value["episode_id"],
            episode_transition_index=value["episode_transition_index"],
            transition_id=value["transition_id"],
            observation=value["observation"],
            action=value["action"],
            old_log_prob=value["old_log_prob"],
            old_value=value["old_value"],
            reward_stage=RewardStage(value["reward_stage"]),
            reward_weights=RewardWeightsV4(
                coverage=reward_weights_raw["coverage"],
                success=reward_weights_raw["success"],
                terminal_gap=reward_weights_raw["terminal_gap"],
                priority=reward_weights_raw["priority"],
                path=reward_weights_raw["path"],
            ),
            reward_components=RewardComponentsV4(
                coverage=reward_raw["coverage"],
                success=reward_raw["success"],
                terminal_gap=reward_raw["terminal_gap"],
                priority=reward_raw["priority"],
                path=reward_raw["path"],
                total=reward_raw["total"],
            ),
            reward_inputs=RewardInputsV4(
                platform_type=reward_inputs_raw["platform_type"],
                coverage_before=reward_inputs_raw["coverage_before"],
                coverage_after=reward_inputs_raw["coverage_after"],
                priority_before=reward_inputs_raw["priority_before"],
                priority_after=reward_inputs_raw["priority_after"],
                path_before_m=reward_inputs_raw["path_before_m"],
                path_after_m=reward_inputs_raw["path_after_m"],
                task_scale_m=reward_inputs_raw["task_scale_m"],
                terminal_class=RewardTerminalClass(
                    reward_inputs_raw["terminal_class"]
                ),
                success_first_crossing=reward_inputs_raw[
                    "success_first_crossing"
                ],
            ),
            done=value["done"],
            terminal_class=RewardTerminalClass(value["terminal_class"]),
            pre_worker_audit=value["pre_worker_audit"],
            post_worker_state=value["post_worker_state"],
        )
    except (TypeError, ValueError) as error:
        raise JournalValidationError("journal payload enum is invalid") from error
    _validate_payload(payload)
    return payload


def _freeze_mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise JournalValidationError("journal payload mapping is invalid")
    return MappingProxyType(
        {key: _freeze_value(item) for key, item in value.items()}
    )


def _freeze_value(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return _clone_tensor(value)
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_value(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    return value


def _clone_tensor(value: object) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise JournalValidationError("journal payload tensor is invalid")
    return value.detach().cpu().contiguous().clone()


def _storage_value(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return _clone_tensor(value)
    if isinstance(value, Mapping):
        return {
            key: _storage_value(value[key])
            for key in sorted(value)
        }
    if isinstance(value, (list, tuple)):
        return [_storage_value(item) for item in value]
    return value


def _semantic_sha256(value: object) -> str:
    descriptor = _semantic_descriptor(value)
    return hashlib.sha256(_canonical_json(descriptor)).hexdigest()


def _semantic_descriptor(value: object) -> object:
    if isinstance(value, torch.Tensor):
        _validate_tensor(value, "payload")
        contiguous = value.detach().cpu().contiguous().reshape(-1)
        raw = (
            b""
            if contiguous.numel() == 0
            else contiguous.view(torch.uint8).numpy().tobytes()
        )
        return [
            "tensor",
            str(value.dtype),
            list(value.shape),
            hashlib.sha256(raw).hexdigest(),
        ]
    if isinstance(value, Mapping):
        return [
            "mapping",
            [
                [key, _semantic_descriptor(value[key])]
                for key in sorted(value)
            ],
        ]
    if isinstance(value, (list, tuple)):
        return ["sequence", [_semantic_descriptor(item) for item in value]]
    if value is None:
        return ["none"]
    if type(value) is bool:
        return ["bool", value]
    if type(value) is int:
        return ["int", str(value)]
    if type(value) is float:
        if not math.isfinite(value):
            raise JournalValidationError("journal payload contains non-finite data")
        return ["float", value.hex()]
    if type(value) is str:
        return ["str", value]
    raise JournalValidationError("journal payload contains unsafe data")


def _journal_sha256(
    committed: Mapping[tuple[int, int], CommittedTransition]
) -> str:
    rows = [
        {
            "worker_index": key[0],
            "slot_index": key[1],
            "transition_id": item.transition_id,
            "payload_sha256": item.payload_sha256,
            "file_sha256": item.file_sha256,
        }
        for key, item in sorted(committed.items())
    ]
    return _json_sha256(rows)


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise JournalValidationError("journal JSON value is invalid") from error


def _json_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise JournalCorruptionError("journal payload cannot be hashed") from error
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_nonnegative_int(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise JournalValidationError(f"journal {name} is invalid")
    return value


def _index_int(value: object, name: str) -> int:
    try:
        return _require_nonnegative_int(value, name)
    except JournalValidationError as error:
        raise JournalCorruptionError(f"journal index {name} is invalid") from error


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as error:
        raise JournalError("journal directory could not be synchronized") from error


__all__ = [
    "APPLIED_SCHEMA_VERSION",
    "CommittedTransition",
    "INDEX_SCHEMA_VERSION",
    "InjectedJournalFault",
    "JournalConflictError",
    "JournalCorruptionError",
    "JournalError",
    "JournalValidationError",
    "LoadedUpdate",
    "MacroTransitionPayload",
    "PAYLOAD_SCHEMA_VERSION",
    "RecoveredWorkerBoundary",
    "SEAL_SCHEMA_VERSION",
    "TransitionJournal",
]

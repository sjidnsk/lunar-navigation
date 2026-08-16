#!/usr/bin/env python3
"""Validate and report one Reward V4 policy-only atomic cutover."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Iterator, NamedTuple


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPOSITORY_ROOT / "training/lunar_policy_training"
sys.path.insert(0, str(PACKAGE_ROOT))

import torch  # noqa: E402

from lunar_policy_training.checkpoint import (  # noqa: E402
    TrainingCheckpointV6,
    UpdateRecoveryState,
    load_checkpoint,
    load_policy_warm_start,
)
from lunar_policy_training.config import (  # noqa: E402
    load_training_config,
    require_reward_v4_config,
)
from lunar_policy_training.evaluation.report import (  # noqa: E402
    FormalEvaluationBatch,
)
from lunar_policy_training.evaluation.reward_v4_runtime import (  # noqa: E402
    evaluate_reward_v4_fixed_grid,
)
from lunar_policy_training.policy.cross_attention import (  # noqa: E402
    CrossAttentionPolicy,
)
from lunar_policy_training.recovery.transition_journal import (  # noqa: E402
    LoadedUpdate,
    TransitionJournal,
)
from lunar_policy_training.reward_contract import (  # noqa: E402
    TaskScaleBucket,
)
from lunar_policy_training.reward_curriculum import (  # noqa: E402
    PlatformType,
)
from lunar_policy_training.reward_evaluation import (  # noqa: E402
    CheckpointScore,
    build_reward_v4_evaluation_manifest,
    checkpoint_score_sha256,
    select_checkpoint_lexicographically,
)
from lunar_policy_training.training_metrics import (  # noqa: E402
    REWARD_V4_TRAINING_UPDATE_METRICS_SCHEMA_V6,
    TRAINING_UPDATE_METRICS_SCHEMA,
    training_metrics_record_sha256,
)
import lunar_policy_training.cli as training_cli  # noqa: E402


SCHEMA_VERSION = "lunar-live-training-cutover/v1"
AUDIT_SCHEMA_VERSION = "lunar-live-training-update-audit/v1"
_UPDATE_PATTERN = re.compile(r"update-([0-9]{8})$")
_SHA40_PATTERN = re.compile(r"[0-9a-f]{40}")
_INVALID_START_VALUES = frozenset(
    {
        "TASK_RESAMPLE_REQUIRED",
        "WHEEL_START_NOT_SAFE",
        "LEGGED_START_NOT_SAFE",
    }
)
_INVALID_START_KEYS = frozenset(
    {
        "invalid_task_audit",
        "task_resample_reason",
        "invalid_start_reason",
    }
)
_FROZEN_V6_FIELDS = frozenset(
    {
        "schema_version",
        "global_step",
        "timestamp_utc",
        "curriculum_phase",
        "platform_allocation",
        "actions_per_worker",
        "transition_count",
        "macro_transitions",
        "strata",
        "ppo",
        "timing",
        "curriculum_events",
        "checkpoint_decision",
    }
)


class CutoverError(RuntimeError):
    """An immutable cutover identity is invalid or inconsistent."""


class CutoverPending(CutoverError):
    """The requested update has not reached an APPLIED boundary yet."""


class _ValidatedUpdate(NamedTuple):
    update_id: int
    checkpoint_path: Path
    checkpoint: TrainingCheckpointV6
    loaded: LoadedUpdate
    metrics_record: dict[str, object]


CheckpointEvaluator = Callable[
    [Path, TrainingCheckpointV6, int], CheckpointScore
]


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as error:
        raise CutoverError("cutover report is not canonical JSON") from error


def _atomic_write_json(path: Path, value: Mapping[str, object]) -> None:
    target = path.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = (_canonical_json(dict(value)) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        directory = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_json(path: Path, *, name: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise CutoverError(f"{name} is missing or unsafe")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CutoverError(f"{name} is invalid") from error
    if not isinstance(value, dict):
        raise CutoverError(f"{name} is not a JSON object")
    return value


def _read_metrics(
    path: Path, *, allow_frozen_v6: bool = False
) -> tuple[dict[str, object], ...]:
    if path.is_symlink() or not path.is_file():
        raise CutoverError("training metrics journal is missing or unsafe")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise CutoverError("training metrics journal cannot be read") from error
    rows: list[dict[str, object]] = []
    steps: list[int] = []
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise CutoverError("training metrics journal contains invalid JSON") from error
        if not isinstance(row, dict):
            raise CutoverError("training metrics row is invalid")
        try:
            _metrics_record_sha256(row, allow_frozen_v6=allow_frozen_v6)
        except Exception as error:
            raise CutoverError("training metrics row is invalid") from error
        rows.append(row)
        steps.append(int(row["global_step"]))
    if not rows or any(right != left + 1 for left, right in zip(steps, steps[1:])):
        raise CutoverError("training metrics steps are incomplete")
    return tuple(rows)


def _metrics_record_sha256(
    record: Mapping[str, object], *, allow_frozen_v6: bool
) -> str:
    try:
        return training_metrics_record_sha256(record)
    except Exception:
        if not allow_frozen_v6:
            raise
    if (
        record.get("schema_version")
        != REWARD_V4_TRAINING_UPDATE_METRICS_SCHEMA_V6
        or set(record) != _FROZEN_V6_FIELDS
        or type(record.get("global_step")) is not int
        or int(record["global_step"]) <= 0
        or not isinstance(record.get("timestamp_utc"), str)
        or not str(record["timestamp_utc"]).endswith("Z")
        or record.get("curriculum_phase")
        not in {"GROUND_R1", "GROUND_R2_HOPPER_R1", "THREE_PLATFORM_R2"}
        or type(record.get("actions_per_worker")) is not int
        or int(record["actions_per_worker"]) <= 0
        or not isinstance(record.get("platform_allocation"), Mapping)
        or not record["platform_allocation"]
        or any(
            not isinstance(platform, str)
            or not platform
            or type(count) is not int
            or count <= 0
            for platform, count in record["platform_allocation"].items()
        )
        or type(record.get("transition_count")) is not int
        or not isinstance(record.get("macro_transitions"), list)
        or int(record["transition_count"]) != len(record["macro_transitions"])
        or int(record["transition_count"])
        != sum(int(value) for value in record["platform_allocation"].values())
        * int(record["actions_per_worker"])
        or any(
            not isinstance(row, Mapping)
            or type(row.get("worker_index")) is not int
            or type(row.get("slot_index")) is not int
            or not isinstance(row.get("transition_id"), str)
            or not row["transition_id"]
            for row in record["macro_transitions"]
        )
        or len(
            {
                str(row["transition_id"])
                for row in record["macro_transitions"]
            }
        )
        != len(record["macro_transitions"])
        or not isinstance(record.get("strata"), Mapping)
        or not isinstance(record.get("ppo"), Mapping)
        or not record["ppo"]
        or not isinstance(record.get("timing"), Mapping)
        or not record["timing"]
        or not isinstance(record.get("curriculum_events"), list)
        or (
            record.get("checkpoint_decision") is not None
            and not isinstance(record.get("checkpoint_decision"), Mapping)
        )
        or "candidate" in record
        or "oracle" in record
    ):
        raise CutoverError("frozen Reward V4 v6 metrics structure is invalid")
    _validate_finite(record)
    return hashlib.sha256(_canonical_json(dict(record)).encode("utf-8")).hexdigest()


def _discover_journal(
    run_root: Path,
    *,
    worker_count: int,
    slots_per_worker: int,
) -> TransitionJournal:
    journal_root = run_root / "journal"
    try:
        candidates = tuple(
            child
            for child in journal_root.iterdir()
            if child.is_dir() and not child.is_symlink() and (child / "recovery").is_dir()
        )
    except OSError as error:
        raise CutoverError("transition journal root cannot be read") from error
    if len(candidates) != 1:
        raise CutoverError("transition journal run identity is ambiguous")
    return TransitionJournal(
        journal_root,
        run_id=candidates[0].name,
        worker_count=worker_count,
        slots_per_worker=slots_per_worker,
    )


def _update_ids(journal: TransitionJournal) -> tuple[int, ...]:
    values: list[int] = []
    try:
        paths = tuple(journal.recovery_root.glob("update-*"))
    except OSError as error:
        raise CutoverError("transition update inventory cannot be read") from error
    for path in paths:
        match = _UPDATE_PATTERN.fullmatch(path.name)
        if path.is_dir() and match is not None:
            values.append(int(match.group(1)))
    if not values or len(values) != len(set(values)):
        raise CutoverError("transition update inventory is invalid")
    return tuple(sorted(values))


def _manifest_runtime_shape(
    manifest: Mapping[str, object],
    *,
    expected_worker_count: int | None = None,
    expected_slots_per_worker: int | None = None,
) -> tuple[int, int]:
    strata = manifest.get("worker_strata")
    manifest_workers = len(strata) if isinstance(strata, list) and strata else 24
    slots = manifest.get("macro_actions_per_worker", 1)
    if type(slots) is not int or slots <= 0:
        raise CutoverError("run manifest macro-action count is invalid")
    workers = manifest_workers if expected_worker_count is None else expected_worker_count
    expected_slots = slots if expected_slots_per_worker is None else expected_slots_per_worker
    if type(workers) is not int or workers <= 0:
        raise CutoverError("expected worker count is invalid")
    if type(expected_slots) is not int or expected_slots <= 0:
        raise CutoverError("expected slot count is invalid")
    if manifest_workers != workers or slots != expected_slots:
        raise CutoverError("run manifest worker-slot shape differs")
    return workers, expected_slots


def _validate_source(
    manifest: Mapping[str, object], *, expected_source_commit: str
) -> None:
    if _SHA40_PATTERN.fullmatch(expected_source_commit) is None:
        raise CutoverError("expected source commit is invalid")
    if manifest.get("source_commit") != expected_source_commit:
        raise CutoverError("run manifest source commit differs")


def _validate_applied_update(
    *,
    run_root: Path,
    journal: TransitionJournal,
    update_id: int,
    manifest: Mapping[str, object],
    metrics_by_step: Mapping[int, dict[str, object]],
    strict_manifest_boundary: bool,
    allow_frozen_v6: bool,
) -> _ValidatedUpdate:
    try:
        loaded = journal.load_update(update_id)
    except Exception as error:
        raise CutoverError(f"update {update_id} journal is invalid") from error
    if loaded.state != "APPLIED":
        raise CutoverPending(f"update {update_id} is not APPLIED")
    checkpoint_path = run_root / "checkpoints" / f"update-{update_id:08d}.pt"
    try:
        checkpoint = load_checkpoint(checkpoint_path)
    except Exception as error:
        raise CutoverError(f"update {update_id} checkpoint is invalid") from error
    if not isinstance(checkpoint, TrainingCheckpointV6):
        raise CutoverError(f"update {update_id} checkpoint is not resumable")
    recovery = checkpoint.update_recovery_state
    if not isinstance(recovery, UpdateRecoveryState):
        raise CutoverError(f"update {update_id} recovery state is missing")
    if (
        checkpoint.global_step != update_id
        or recovery.update_id != update_id
        or recovery.journal_state != "SEALED"
        or recovery.journal_sha256 != loaded.journal_sha256
        or recovery.policy_version != loaded.policy_version
        or loaded.checkpoint_payload_sha256 != checkpoint.payload_sha256
        or loaded.metrics_record_sha256 != recovery.metrics_record_sha256
        or _metrics_record_sha256(
            recovery.metrics_record,
            allow_frozen_v6=allow_frozen_v6,
        )
        != recovery.metrics_record_sha256
    ):
        raise CutoverError(f"update {update_id} artifact triangle identity differs")
    metrics = metrics_by_step.get(update_id)
    if metrics is None or (
        _metrics_record_sha256(
            metrics,
            allow_frozen_v6=allow_frozen_v6,
        )
        != recovery.metrics_record_sha256
        or metrics != recovery.metrics_record
    ):
        raise CutoverError(f"update {update_id} metrics identity differs")
    if len(recovery.next_slot_by_worker) != journal.worker_count or (
        tuple(recovery.next_slot_by_worker)
        != (journal.slots_per_worker,) * journal.worker_count
    ):
        raise CutoverError(f"update {update_id} worker cursors differ")
    if checkpoint.source_commit != manifest.get("source_commit"):
        raise CutoverError(f"update {update_id} checkpoint source differs")
    if strict_manifest_boundary:
        _validate_manifest_boundary(
            run_root=run_root,
            manifest=manifest,
            update_id=update_id,
            recovery=recovery,
        )
    return _ValidatedUpdate(
        update_id=update_id,
        checkpoint_path=checkpoint_path.resolve(),
        checkpoint=checkpoint,
        loaded=loaded,
        metrics_record=dict(metrics),
    )


def _validate_manifest_boundary(
    *,
    run_root: Path,
    manifest: Mapping[str, object],
    update_id: int,
    recovery: UpdateRecoveryState,
) -> None:
    if manifest.get("global_step") != update_id:
        raise CutoverError("run manifest update boundary differs")
    cursor = manifest.get("journal_cursor")
    if not isinstance(cursor, Mapping):
        raise CutoverError("run manifest journal cursor is missing")
    expected = recovery.to_dict()
    if any(cursor.get(name) != expected[name] for name in expected):
        raise CutoverError("run manifest journal cursor identity differs")
    summary = manifest.get("training_metrics")
    metrics_path = run_root / "metrics/train.jsonl"
    if not isinstance(summary, Mapping) or (
        summary.get("path") != "metrics/train.jsonl"
        or summary.get("last_global_step") != update_id
        or summary.get("sha256") != hashlib.sha256(metrics_path.read_bytes()).hexdigest()
    ):
        raise CutoverError("run manifest metrics summary differs")


def _metrics_by_step(rows: Sequence[dict[str, object]]) -> dict[int, dict[str, object]]:
    result = {int(row["global_step"]): dict(row) for row in rows}
    if len(result) != len(rows):
        raise CutoverError("training metrics repeat an update")
    return result


def _contains_invalid_start(value: object) -> bool:
    if isinstance(value, str):
        return value in _INVALID_START_VALUES
    if isinstance(value, Mapping):
        return any(
            key in _INVALID_START_KEYS
            or _contains_invalid_start(key)
            or _contains_invalid_start(item)
            for key, item in value.items()
        )
    if isinstance(value, (tuple, list)):
        return any(_contains_invalid_start(item) for item in value)
    return False


def _validate_finite(value: object) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise CutoverError("update contains non-finite training value")
        return
    if isinstance(value, Mapping):
        for item in value.values():
            _validate_finite(item)
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            _validate_finite(item)
        return
    if isinstance(value, torch.Tensor) and not bool(torch.isfinite(value).all()):
        raise CutoverError("update contains non-finite tensor")


def _repair_source_commit() -> str:
    try:
        import subprocess

        value = subprocess.run(
            ("git", "-C", str(REPOSITORY_ROOT), "rev-parse", "HEAD"),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise CutoverError("repair source commit cannot be resolved") from error
    if _SHA40_PATTERN.fullmatch(value) is None:
        raise CutoverError("repair source commit is invalid")
    return value


@contextmanager
def _fixed_evaluator(
    *,
    config_path: Path,
    cache_manifest_path: Path,
    evaluation_seed: int,
) -> Iterator[CheckpointEvaluator]:
    config = load_training_config(config_path)
    _validate_fixed_evaluation_seed(config, evaluation_seed)
    capability = training_cli._formal_capability_preflight(REPOSITORY_ROOT)
    with training_cli._FormalTaskCacheRuntime(
        cache_manifest_path=cache_manifest_path,
        capability_bundle=capability,
        task_area=config.task_area,
        source_commit=_repair_source_commit(),
        builder_workers=1,
    ) as task_runtime:
        _, assembly = training_cli._build_formal_environment(
            cache_manifest_path,
            capability_bundle=capability,
            repository_root=REPOSITORY_ROOT,
            split="train",
            task_area=config.task_area,
            task_cache_client=task_runtime.client,
        )
        manifest = build_reward_v4_evaluation_manifest(
            platforms=tuple(PlatformType),
            scale_buckets=tuple(TaskScaleBucket),
            evaluation_seeds=(evaluation_seed,),
        )
        batch = FormalEvaluationBatch(
            split="validation",
            factory=assembly.factory,
            observation_template=assembly.observation_template,
            scenario_seeds=(evaluation_seed,),
        )

        def evaluate(
            checkpoint_path: Path,
            checkpoint: TrainingCheckpointV6,
            seed: int,
        ) -> CheckpointScore:
            policy = CrossAttentionPolicy()
            evidence = load_policy_warm_start(
                checkpoint_path,
                policy,
                value_head_seed=seed,
            )
            if evidence.parent_payload_sha256 != checkpoint.payload_sha256:
                raise CutoverError("policy-only warm-start parent identity differs")
            report = evaluate_reward_v4_fixed_grid(
                policy,
                device="cuda",
                checkpoint_payload_sha256=checkpoint.payload_sha256,
                manifest=manifest,
                active_platforms=(PlatformType.WHEELED, PlatformType.LEGGED),
                batch=batch,
                bootstrap_seed=seed,
                bootstrap_resample_count=2_000,
                enabled_r2_platforms=(),
            )
            return report.checkpoint_score

        yield evaluate


def _validate_fixed_evaluation_seed(config: object, evaluation_seed: int):
    try:
        reward = require_reward_v4_config(config)
    except (TypeError, ValueError) as error:
        raise CutoverError("fixed evaluation config is invalid") from error
    if (
        getattr(config, "run_kind", None) != "formal"
        or type(evaluation_seed) is not int
        or evaluation_seed not in reward.evaluation_seeds
    ):
        raise CutoverError("fixed evaluation seed is outside the frozen set")
    return reward


def select_policy(
    *,
    old_run_root: Path,
    expected_old_source_commit: str,
    evaluation_seed: int,
    maximum_candidates: int,
    output: Path,
    evaluator: CheckpointEvaluator | None = None,
    config_path: Path | None = None,
    cache_manifest_path: Path | None = None,
) -> dict[str, object]:
    root = old_run_root.expanduser().resolve()
    if type(evaluation_seed) is not int or evaluation_seed < 0:
        raise CutoverError("evaluation seed is invalid")
    if type(maximum_candidates) is not int or not 1 <= maximum_candidates <= 2:
        raise CutoverError("maximum candidate count is invalid")
    manifest = _read_json(root / "run-manifest.json", name="run manifest")
    _validate_source(manifest, expected_source_commit=expected_old_source_commit)
    workers, slots = _manifest_runtime_shape(manifest)
    journal = _discover_journal(root, worker_count=workers, slots_per_worker=slots)
    rows = _read_metrics(root / "metrics/train.jsonl", allow_frozen_v6=True)
    by_step = _metrics_by_step(rows)
    update_ids = _update_ids(journal)
    applied: list[_ValidatedUpdate] = []
    ignored: list[int] = []
    for update_id in reversed(update_ids):
        try:
            state = journal.load_update(update_id).state
        except Exception as error:
            raise CutoverError(f"update {update_id} journal is invalid") from error
        if state != "APPLIED":
            ignored.append(update_id)
            continue
        applied.append(
            _validate_applied_update(
                run_root=root,
                journal=journal,
                update_id=update_id,
                manifest=manifest,
                metrics_by_step=by_step,
                strict_manifest_boundary=False,
                allow_frozen_v6=True,
            )
        )
        if len(applied) == maximum_candidates:
            break
    if not applied:
        raise CutoverPending("old run has no APPLIED checkpoint")
    latest_applied = max(item.update_id for item in applied)
    if manifest.get("global_step") != latest_applied or max(by_step) != latest_applied:
        raise CutoverPending("old run latest APPLIED boundary is changing")
    selected_candidates = tuple(sorted(applied, key=lambda item: item.update_id))
    if evaluator is None:
        if config_path is None or cache_manifest_path is None:
            raise CutoverError("fixed evaluator inputs are missing")
        with _fixed_evaluator(
            config_path=config_path,
            cache_manifest_path=cache_manifest_path,
            evaluation_seed=evaluation_seed,
        ) as fixed_evaluator:
            return select_policy(
                old_run_root=root,
                expected_old_source_commit=expected_old_source_commit,
                evaluation_seed=evaluation_seed,
                maximum_candidates=maximum_candidates,
                output=output,
                evaluator=fixed_evaluator,
            )
    scored: list[tuple[_ValidatedUpdate, CheckpointScore]] = []
    for candidate in selected_candidates:
        score = evaluator(
            candidate.checkpoint_path,
            candidate.checkpoint,
            evaluation_seed,
        )
        if not isinstance(score, CheckpointScore) or (
            score.payload_sha256 != candidate.checkpoint.payload_sha256
        ):
            raise CutoverError("fixed evaluator checkpoint identity differs")
        scored.append((candidate, score))
    winner = select_checkpoint_lexicographically(tuple(score for _, score in scored))
    matches = tuple(item for item, score in scored if score == winner)
    if len(matches) != 1:
        raise CutoverError("checkpoint selection result is ambiguous")
    selected = matches[0]
    report: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "passed": True,
        "old_run_root": str(root),
        "expected_old_source_commit": expected_old_source_commit,
        "repair_source_commit": _repair_source_commit(),
        "evaluation_seed": evaluation_seed,
        "maximum_candidates": maximum_candidates,
        "ignored_non_applied_updates": sorted(ignored),
        "candidates": [
            {
                "update_id": item.update_id,
                "checkpoint": str(item.checkpoint_path),
                "checkpoint_payload_sha256": item.checkpoint.payload_sha256,
                "journal_sha256": item.loaded.journal_sha256,
                "metrics_record_sha256": item.loaded.metrics_record_sha256,
                "score": score.to_dict(),
                "score_sha256": checkpoint_score_sha256(score),
            }
            for item, score in scored
        ],
        "selected_update_id": selected.update_id,
        "selected_checkpoint": str(selected.checkpoint_path),
        "selected_checkpoint_payload_sha256": selected.checkpoint.payload_sha256,
        "selection_algorithm": "select_checkpoint_lexicographically",
        "policy_only": True,
    }
    _atomic_write_json(output, report)
    return report


def audit_update(
    *,
    run_root: Path,
    expected_source_commit: str,
    update_id: int,
    expected_worker_count: int,
    expected_slots_per_worker: int,
    output: Path,
) -> dict[str, object]:
    if type(update_id) is not int or update_id <= 0:
        raise CutoverError("audit update ID is invalid")
    root = run_root.expanduser().resolve()
    manifest = _read_json(root / "run-manifest.json", name="run manifest")
    _validate_source(manifest, expected_source_commit=expected_source_commit)
    workers, slots = _manifest_runtime_shape(
        manifest,
        expected_worker_count=expected_worker_count,
        expected_slots_per_worker=expected_slots_per_worker,
    )
    journal = _discover_journal(root, worker_count=workers, slots_per_worker=slots)
    rows = _read_metrics(root / "metrics/train.jsonl")
    validated = _validate_applied_update(
        run_root=root,
        journal=journal,
        update_id=update_id,
        manifest=manifest,
        metrics_by_step=_metrics_by_step(rows),
        strict_manifest_boundary=True,
        allow_frozen_v6=False,
    )
    committed = validated.loaded.committed
    expected_keys = {
        (worker, slot)
        for worker in range(workers)
        for slot in range(slots)
    }
    if set(committed) != expected_keys:
        raise CutoverError("audited update worker slots are incomplete")
    invalid = tuple(
        item
        for item in committed.values()
        if _contains_invalid_start(item.payload.observation)
        or _contains_invalid_start(item.payload.action)
        or _contains_invalid_start(item.payload.pre_worker_state)
        or _contains_invalid_start(item.payload.post_worker_state)
    )
    if invalid:
        raise CutoverError("audited update contains invalid-start transition")
    _validate_finite(validated.metrics_record)
    for item in committed.values():
        _validate_finite(item.payload.observation)
        _validate_finite(item.payload.action)
        _validate_finite(item.payload.old_log_prob)
        _validate_finite(item.payload.old_value)
    if validated.metrics_record.get("schema_version") == TRAINING_UPDATE_METRICS_SCHEMA:
        if validated.metrics_record.get("transition_count") != len(committed):
            raise CutoverError("audited update metrics transition count differs")
    invalid_tasks = validated.metrics_record.get("invalid_tasks")
    invalid_start_task_count = (
        int(invalid_tasks.get("invalid_start_task_count", 0))
        if isinstance(invalid_tasks, Mapping)
        else 0
    )
    report: dict[str, object] = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "passed": True,
        "run_root": str(root),
        "source_commit": expected_source_commit,
        "update_id": update_id,
        "policy_version": validated.loaded.policy_version,
        "checkpoint": str(validated.checkpoint_path),
        "checkpoint_payload_sha256": validated.checkpoint.payload_sha256,
        "journal_sha256": validated.loaded.journal_sha256,
        "metrics_record_sha256": validated.loaded.metrics_record_sha256,
        "worker_count": workers,
        "slots_per_worker": slots,
        "transition_count": len(committed),
        "invalid_start_transition_count": 0,
        "resampled_invalid_start_task_count": invalid_start_task_count,
        "optimizer_apply_count": 1,
        "optimizer_apply_authority": "single-applied-checkpoint-triangle",
        "finite_training_values": True,
    }
    _atomic_write_json(output, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    select = commands.add_parser("select-policy")
    select.add_argument("--old-run-root", required=True, type=Path)
    select.add_argument("--expected-old-source-commit", required=True)
    select.add_argument("--config", required=True, type=Path)
    select.add_argument("--cache-manifest", required=True, type=Path)
    select.add_argument("--evaluation-seed", required=True, type=int)
    select.add_argument("--maximum-candidates", required=True, type=int)
    select.add_argument("--output", required=True, type=Path)
    audit = commands.add_parser("audit-update")
    audit.add_argument("--run-root", required=True, type=Path)
    audit.add_argument("--expected-source-commit", required=True)
    audit.add_argument("--update-id", required=True, type=int)
    audit.add_argument("--expected-worker-count", type=int, default=24)
    audit.add_argument("--expected-slots-per-worker", type=int, default=1)
    audit.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "select-policy":
            report = select_policy(
                old_run_root=arguments.old_run_root,
                expected_old_source_commit=arguments.expected_old_source_commit,
                evaluation_seed=arguments.evaluation_seed,
                maximum_candidates=arguments.maximum_candidates,
                output=arguments.output,
                config_path=arguments.config,
                cache_manifest_path=arguments.cache_manifest,
            )
        else:
            report = audit_update(
                run_root=arguments.run_root,
                expected_source_commit=arguments.expected_source_commit,
                update_id=arguments.update_id,
                expected_worker_count=arguments.expected_worker_count,
                expected_slots_per_worker=arguments.expected_slots_per_worker,
                output=arguments.output,
            )
    except CutoverPending as error:
        print(str(error), file=sys.stderr)
        return 3
    except (CutoverError, OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2
    print(_canonical_json(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

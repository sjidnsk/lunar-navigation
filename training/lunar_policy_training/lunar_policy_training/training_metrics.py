"""Durable per-update metrics for one formal PPO training run."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from lunar_planner_training_bridge import PlanningOutcome

from .environment.candidate_builder import (
    CANDIDATE_DIAGNOSTIC_FIELDS,
    CandidateDiagnostics,
)
from .environment.macro_step import TerminalAudit, TerminalReason
from .ppo.trainer import PPOUpdateMetrics


TRAINING_UPDATE_METRICS_SCHEMA = "lunar-training-update-metrics/v3"


class TrainingMetricsError(ValueError):
    """Training metrics are incomplete, ambiguous, or unsafe to append."""


def build_training_update_record(
    *,
    global_step: int,
    timestamp_utc: str,
    curriculum_phase: str,
    platform_allocation: Mapping[str, int],
    worker_platforms: Sequence[str],
    rollout_horizon: int,
    raw_rewards: np.ndarray,
    dones: np.ndarray,
    start_coverage: np.ndarray,
    end_coverage: np.ndarray,
    success_first_crossings: Sequence[bool],
    planning_outcomes: Sequence[PlanningOutcome],
    candidate_diagnostics: Sequence[CandidateDiagnostics],
    no_candidate_terminations: Sequence[bool],
    terminal_audits: Sequence[TerminalAudit | None],
    ppo_metrics: PPOUpdateMetrics,
    collect_wall_seconds: float,
    update_wall_seconds: float,
    worker_wait_seconds: float,
) -> dict[str, object]:
    """Build one finite JSON-compatible record from an update boundary."""
    if type(global_step) is not int or global_step <= 0:
        raise TrainingMetricsError("metrics global step must be positive")
    if not isinstance(timestamp_utc, str) or not timestamp_utc.endswith("Z"):
        raise TrainingMetricsError("metrics timestamp must be UTC")
    if not isinstance(curriculum_phase, str) or not curriculum_phase:
        raise TrainingMetricsError("metrics curriculum phase is invalid")
    allocation = dict(platform_allocation)
    if (
        not allocation
        or any(
            not isinstance(name, str)
            or not name
            or type(count) is not int
            or count <= 0
            for name, count in allocation.items()
        )
    ):
        raise TrainingMetricsError("metrics platform allocation is invalid")
    platforms = tuple(worker_platforms)
    worker_count = sum(allocation.values())
    if (
        len(platforms) != worker_count
        or Counter(platforms) != Counter(allocation)
    ):
        raise TrainingMetricsError("metrics worker platforms differ from allocation")
    if type(rollout_horizon) is not int or rollout_horizon <= 0:
        raise TrainingMetricsError("metrics rollout horizon must be positive")
    if (
        not isinstance(raw_rewards, np.ndarray)
        or raw_rewards.shape != (rollout_horizon, worker_count)
        or not np.issubdtype(raw_rewards.dtype, np.floating)
        or not np.isfinite(raw_rewards).all()
    ):
        raise TrainingMetricsError("metrics rewards must be finite [T,E]")
    if (
        not isinstance(dones, np.ndarray)
        or dones.dtype != np.bool_
        or dones.shape != raw_rewards.shape
    ):
        raise TrainingMetricsError("metrics dones must be boolean [T,E]")
    coverage = (start_coverage, end_coverage)
    if any(
        not isinstance(value, np.ndarray)
        or value.shape != (worker_count,)
        or not np.issubdtype(value.dtype, np.floating)
        or not np.isfinite(value).all()
        or ((value < 0.0) | (value > 1.0)).any()
        for value in coverage
    ):
        raise TrainingMetricsError("metrics coverage must be finite [E] in [0,1]")
    success = tuple(success_first_crossings)
    outcomes = tuple(planning_outcomes)
    candidates = tuple(candidate_diagnostics)
    no_candidates = tuple(no_candidate_terminations)
    audits = tuple(terminal_audits)
    transition_count = rollout_horizon * worker_count
    if (
        len(success) != transition_count
        or any(type(value) is not bool for value in success)
        or len(outcomes) != transition_count
        or any(not isinstance(value, PlanningOutcome) for value in outcomes)
        or len(candidates) != transition_count
        or any(not isinstance(value, CandidateDiagnostics) for value in candidates)
    ):
        raise TrainingMetricsError("metrics transition diagnostics are incomplete")
    if (
        not no_candidates
        or len(no_candidates) % worker_count != 0
        or any(type(value) is not bool for value in no_candidates)
    ):
        raise TrainingMetricsError(
            "metrics no-candidate diagnostics are incomplete"
        )
    if (
        not audits
        or len(audits) % worker_count != 0
        or any(
            audit is not None and not isinstance(audit, TerminalAudit)
            for audit in audits
        )
    ):
        raise TrainingMetricsError("metrics terminal audits are incomplete")
    materialized_audits = tuple(
        audit for audit in audits if audit is not None
    )
    if len(materialized_audits) != int(dones.sum(dtype=np.int64)):
        raise TrainingMetricsError(
            "metrics terminal audits differ from rollout terminals"
        )
    if sum(
        audit.reason is TerminalReason.SUCCESS
        for audit in materialized_audits
    ) != sum(success):
        raise TrainingMetricsError(
            "metrics success crossings differ from terminal reasons"
        )
    if not isinstance(ppo_metrics, PPOUpdateMetrics):
        raise TrainingMetricsError("metrics require PPOUpdateMetrics")
    timings = (
        collect_wall_seconds,
        update_wall_seconds,
        worker_wait_seconds,
    )
    if any(not _finite_nonnegative(value) for value in timings):
        raise TrainingMetricsError("metrics wall times must be finite and nonnegative")

    reward64 = raw_rewards.astype(np.float64, copy=False)
    per_platform_mean = {
        platform: float(
            reward64[
                :,
                [
                    index
                    for index, value in enumerate(platforms)
                    if value == platform
                ],
            ].mean()
        )
        for platform in allocation
    }
    outcome_counts = Counter(value.name for value in outcomes)
    outcome_counts_by_platform = {
        platform: dict(
            sorted(
                Counter(
                    outcomes[index].name
                    for index in range(transition_count)
                    if platforms[index % worker_count] == platform
                ).items()
            )
        )
        for platform in sorted(allocation)
    }
    candidate_by_platform: dict[str, dict[str, int]] = {
        platform: {
            **{name: 0 for name in CANDIDATE_DIAGNOSTIC_FIELDS},
            "no_candidate_termination_count": 0,
            "planner_rejected_exhaustion_count": 0,
        }
        for platform in sorted(allocation)
    }
    for index, diagnostics in enumerate(candidates):
        values = candidate_by_platform[platforms[index % worker_count]]
        for name in CANDIDATE_DIAGNOSTIC_FIELDS:
            values[name] += getattr(diagnostics, name)
    for index, terminated_without_candidates in enumerate(no_candidates):
        if terminated_without_candidates:
            candidate_by_platform[platforms[index % worker_count]][
                "no_candidate_termination_count"
            ] += 1
    terminal_reason_counts: Counter[str] = Counter()
    terminal_reason_counts_by_platform: dict[str, Counter[str]] = {
        platform: Counter() for platform in sorted(allocation)
    }
    terminal_candidate_by_platform = {
        platform: {
            name: 0
            for name in CANDIDATE_DIAGNOSTIC_FIELDS
        }
        for platform in sorted(allocation)
    }
    remaining_counts: list[int] = []
    oracle_opportunity_count = 0
    oracle_contradiction_count = 0
    for index, audit in enumerate(audits):
        if audit is None:
            continue
        platform = platforms[index % worker_count]
        reason = audit.reason.value
        terminal_reason_counts[reason] += 1
        terminal_reason_counts_by_platform[platform][reason] += 1
        for name in terminal_candidate_by_platform[platform]:
            terminal_candidate_by_platform[platform][name] += getattr(
                audit.candidate_diagnostics, name
            )
        oracle_opportunity_count += audit.oracle_opportunity_count
        if audit.oracle_opportunity_count > 0:
            oracle_contradiction_count += 1
        if audit.remaining_coverable_detail_cell_count is not None:
            remaining_counts.append(
                audit.remaining_coverable_detail_cell_count
            )
        if audit.reason is TerminalReason.PLANNER_REJECTED_ALL:
            candidate_by_platform[platform][
                "planner_rejected_exhaustion_count"
            ] += 1
    start_mean = float(start_coverage.astype(np.float64).mean())
    end_mean = float(end_coverage.astype(np.float64).mean())
    record: dict[str, object] = {
        "schema_version": TRAINING_UPDATE_METRICS_SCHEMA,
        "global_step": global_step,
        "timestamp_utc": timestamp_utc,
        "curriculum_phase": curriculum_phase,
        "platform_allocation": allocation,
        "rollout_horizon": rollout_horizon,
        "transition_count": transition_count,
        "reward": {
            "mean": float(reward64.mean()),
            "std": float(reward64.std()),
            "min": float(reward64.min()),
            "max": float(reward64.max()),
            "per_platform_mean": per_platform_mean,
        },
        "coverage": {
            "start_mean": start_mean,
            "end_mean": end_mean,
            "delta_mean": end_mean - start_mean,
        },
        "terminal_count": int(dones.sum(dtype=np.int64)),
        "success_first_crossing_count": sum(success),
        "candidate": {"by_platform": candidate_by_platform},
        "terminal": {
            "reason_counts": dict(sorted(terminal_reason_counts.items())),
            "reason_counts_by_platform": {
                platform: dict(
                    sorted(terminal_reason_counts_by_platform[platform].items())
                )
                for platform in sorted(allocation)
            },
            "candidate_diagnostics_by_platform": (
                terminal_candidate_by_platform
            ),
            "oracle_opportunity_count": oracle_opportunity_count,
            "oracle_contradiction_count": oracle_contradiction_count,
            "remaining_coverable_detail_cell_count": {
                "count": len(remaining_counts),
                "min": min(remaining_counts) if remaining_counts else None,
                "max": max(remaining_counts) if remaining_counts else None,
            },
        },
        "planner": {
            "outcome_counts": dict(sorted(outcome_counts.items())),
            "outcome_counts_by_platform": outcome_counts_by_platform,
            "success_rate": (
                outcome_counts[PlanningOutcome.NEW_REFERENCE_AVAILABLE.name]
                / transition_count
            ),
        },
        "ppo": {
            "total_loss": ppo_metrics.total_loss,
            "policy_loss": ppo_metrics.policy_loss,
            "value_loss": ppo_metrics.value_loss,
            "frontier_entropy": ppo_metrics.frontier_entropy,
            "theta_entropy": ppo_metrics.theta_entropy,
            "approx_kl": ppo_metrics.approx_kl,
            "gradient_norm": ppo_metrics.gradient_norm,
            "clipped_gradient_norm": ppo_metrics.clipped_gradient_norm,
            "parameter_change_l2": ppo_metrics.parameter_change_l2,
            "optimizer_steps": ppo_metrics.optimizer_steps,
            "epochs_completed": ppo_metrics.epochs_completed,
            "target_kl_early_stopped": ppo_metrics.target_kl_early_stopped,
        },
        "timing": {
            "collect_wall_seconds": float(collect_wall_seconds),
            "update_wall_seconds": float(update_wall_seconds),
            "worker_wait_seconds": float(worker_wait_seconds),
        },
    }
    _validate_finite_json(record)
    return record


class TrainingMetricsJournal:
    """Single-writer, fsync-backed formal training update journal."""

    def __init__(self, path: Path, *, resume_global_step: int) -> None:
        if not isinstance(path, Path) or not path.is_absolute():
            raise TrainingMetricsError("metrics journal path must be absolute")
        if type(resume_global_step) is not int or resume_global_step < 0:
            raise TrainingMetricsError("metrics resume global step is invalid")
        if path.is_symlink():
            raise TrainingMetricsError("metrics journal cannot be a symlink")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(self.path, flags, 0o600)
        os.close(descriptor)
        rows = self._read_rows()
        if rows:
            steps = [row["global_step"] for row in rows]
            if any(right != left + 1 for left, right in zip(steps, steps[1:])):
                raise TrainingMetricsError("metrics global steps are not contiguous")
            if steps[-1] > resume_global_step:
                raise TrainingMetricsError("metrics journal is ahead of checkpoint")
            if steps[-1] != resume_global_step:
                raise TrainingMetricsError("metrics journal does not reach checkpoint")
        self._last_step = resume_global_step

    def append(self, record: Mapping[str, object]) -> None:
        if not isinstance(record, Mapping):
            raise TrainingMetricsError("metrics record must be a mapping")
        payload = dict(record)
        _validate_record(payload)
        if payload["global_step"] != self._last_step + 1:
            raise TrainingMetricsError("metrics record is not the next global step")
        try:
            encoded = (
                json.dumps(
                    payload,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8")
        except (TypeError, ValueError) as error:
            raise TrainingMetricsError("metrics record is not JSON-compatible") from error
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(self.path, flags, 0o600)
        try:
            written = os.write(descriptor, encoded)
            if written != len(encoded):
                raise TrainingMetricsError("metrics journal write was incomplete")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        self._last_step = int(payload["global_step"])

    def checkpoint_summary(self, *, artifact_root: Path) -> dict[str, object]:
        if not isinstance(artifact_root, Path) or not artifact_root.is_absolute():
            raise TrainingMetricsError("metrics artifact root must be absolute")
        try:
            relative = self.path.relative_to(artifact_root)
            payload = self.path.read_bytes()
        except (ValueError, OSError) as error:
            raise TrainingMetricsError("metrics journal is outside artifact root") from error
        return {
            "schema_version": "lunar-training-metrics-summary/v1",
            "path": relative.as_posix(),
            "last_global_step": self._last_step,
            "sha256": hashlib.sha256(payload).hexdigest(),
        }

    def _read_rows(self) -> list[dict[str, object]]:
        if not self.path.exists():
            return []
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError) as error:
            raise TrainingMetricsError("metrics journal cannot be read") from error
        rows: list[dict[str, object]] = []
        for line in lines:
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise TrainingMetricsError("metrics journal contains invalid JSON") from error
            _validate_record(row)
            rows.append(row)
        return rows


def _validate_record(record: object) -> None:
    if not isinstance(record, dict):
        raise TrainingMetricsError("metrics record must be a JSON object")
    if record.get("schema_version") != TRAINING_UPDATE_METRICS_SCHEMA:
        raise TrainingMetricsError("metrics record schema differs")
    if type(record.get("global_step")) is not int or record["global_step"] <= 0:
        raise TrainingMetricsError("metrics record global step is invalid")
    candidate = record.get("candidate")
    candidate_by_platform = (
        candidate.get("by_platform")
        if isinstance(candidate, Mapping)
        else None
    )
    if not isinstance(candidate_by_platform, Mapping) or not candidate_by_platform:
        raise TrainingMetricsError("metrics candidate diagnostics are missing")
    aggregate_fields = set(CANDIDATE_DIAGNOSTIC_FIELDS) | {
        "no_candidate_termination_count",
        "planner_rejected_exhaustion_count",
    }
    for values in candidate_by_platform.values():
        _validate_diagnostic_counts(values, aggregate_fields)
    terminal = record.get("terminal")
    terminal_by_platform = (
        terminal.get("candidate_diagnostics_by_platform")
        if isinstance(terminal, Mapping)
        else None
    )
    if not isinstance(terminal_by_platform, Mapping) or not terminal_by_platform:
        raise TrainingMetricsError("metrics terminal diagnostics are missing")
    for values in terminal_by_platform.values():
        _validate_diagnostic_counts(values, set(CANDIDATE_DIAGNOSTIC_FIELDS))
    _validate_finite_json(record)


def _validate_diagnostic_counts(
    value: object, expected_fields: set[str]
) -> None:
    if (
        not isinstance(value, Mapping)
        or set(value) != expected_fields
        or any(
            type(count) is not int or count < 0
            for count in value.values()
        )
    ):
        raise TrainingMetricsError("metrics candidate diagnostics are invalid")


def _validate_finite_json(value: object) -> None:
    if isinstance(value, Mapping):
        for nested in value.values():
            _validate_finite_json(nested)
        return
    if isinstance(value, (list, tuple)):
        for nested in value:
            _validate_finite_json(nested)
        return
    if isinstance(value, float) and not math.isfinite(value):
        raise TrainingMetricsError("metrics numeric values must be finite")


def _finite_nonnegative(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


__all__ = [
    "TRAINING_UPDATE_METRICS_SCHEMA",
    "TrainingMetricsError",
    "TrainingMetricsJournal",
    "build_training_update_record",
]

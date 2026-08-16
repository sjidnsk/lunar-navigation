"""Durable per-update metrics for one formal PPO training run."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np
from lunar_planner_training_bridge import PlanningOutcome

from .environment.candidate_builder import (
    CANDIDATE_DIAGNOSTIC_FIELDS,
    CandidateDiagnostics,
)
from .environment.macro_step import TerminalAudit, TerminalReason
from .environment.formal_episode_state import FormalWorkerState
from .environment.parallel_pool import InvalidTaskAudit
from .environment.task_area import WorkerStratum, formal_worker_strata
from .ppo.rollout import PLATFORM_ID_BY_TYPE, SCALE_BUCKET_ID_BY_BUCKET
from .ppo.collector import CommittedMacroRollout
from .ppo.trainer import PPOUpdateMetrics, StratumPPOUpdateMetrics
from .reward import RewardComponentsV4
from .reward_contract import (
    DEFAULT_REWARD_CONFIG,
    RewardStage,
    RewardTerminalClass,
    RewardWeightsV4,
    TaskScaleBucket,
)


LEGACY_TRAINING_UPDATE_METRICS_SCHEMA = "lunar-training-update-metrics/v5"
REWARD_V4_TRAINING_UPDATE_METRICS_SCHEMA_V6 = (
    "lunar-training-update-metrics/v6"
)
TRAINING_UPDATE_METRICS_SCHEMA = "lunar-training-update-metrics/v7"
_CANDIDATE_IDENTITY_FIELDS = CANDIDATE_DIAGNOSTIC_FIELDS[:2]
_CANDIDATE_COUNT_FIELDS = CANDIDATE_DIAGNOSTIC_FIELDS[2:]
_PHYSICAL_EXHAUSTION_REASONS = frozenset(
    {
        TerminalReason.NO_FRONTIER,
        TerminalReason.NO_GLOBAL_ROUTE,
        TerminalReason.ZERO_EXPECTED_GAIN,
    }
)


class TrainingMetricsError(ValueError):
    """Training metrics are incomplete, ambiguous, or unsafe to append."""


def _thaw_journal_worker_state(value: object) -> object:
    """Restore the JSON sequence form hidden by journal immutability."""

    if isinstance(value, Mapping):
        return {
            key: _thaw_journal_worker_state(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_thaw_journal_worker_state(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class RuntimeWorkerIdentity:
    """All task and physical-snapshot identities observed for one worker."""

    worker_index: int
    platform_type: str
    episode_ids: tuple[str, ...]
    task_ids: tuple[str, ...]
    physical_snapshot_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.worker_index) is not int or self.worker_index < 0:
            raise TrainingMetricsError("runtime worker index is invalid")
        if self.platform_type not in PLATFORM_ID_BY_TYPE:
            raise TrainingMetricsError("runtime worker platform is invalid")
        for name in ("episode_ids", "task_ids", "physical_snapshot_ids"):
            values = getattr(self, name)
            if (
                type(values) is not tuple
                or not values
                or values != tuple(sorted(set(values)))
            ):
                raise TrainingMetricsError(
                    f"runtime worker {name} are invalid"
                )
        if any(not isinstance(value, str) or not value for value in self.episode_ids):
            raise TrainingMetricsError("runtime worker episode identity is invalid")
        for values, label in (
            (self.task_ids, "task"),
            (self.physical_snapshot_ids, "physical snapshot"),
        ):
            for value in values:
                _require_sha256(value, f"runtime worker {label}")

    def to_dict(self) -> dict[str, object]:
        return {
            "worker_index": self.worker_index,
            "platform": self.platform_type,
            "episode_ids": list(self.episode_ids),
            "task_ids": list(self.task_ids),
            "physical_snapshot_ids": list(self.physical_snapshot_ids),
        }

    @classmethod
    def from_dict(cls, value: object) -> "RuntimeWorkerIdentity":
        if not isinstance(value, Mapping) or set(value) != {
            "worker_index",
            "platform",
            "episode_ids",
            "task_ids",
            "physical_snapshot_ids",
        }:
            raise TrainingMetricsError("runtime worker identity is invalid")
        try:
            return cls(
                worker_index=value["worker_index"],
                platform_type=value["platform"],
                episode_ids=tuple(value["episode_ids"]),
                task_ids=tuple(value["task_ids"]),
                physical_snapshot_ids=tuple(value["physical_snapshot_ids"]),
            )
        except (TypeError, ValueError) as error:
            raise TrainingMetricsError(
                "runtime worker identity is invalid"
            ) from error


@dataclass(frozen=True, slots=True)
class RewardV4RuntimeDiagnostics:
    """Identity-bound, non-policy runtime facts for one Reward V4 update."""

    update_id: int
    worker_identities: tuple[RuntimeWorkerIdentity, ...]
    invalid_start_task_count: int
    coarse_fine_reachability_mismatch_count: int
    planner_suppressed_candidate_count: int
    zero_motion_reselection_count: int
    macro_action_elapsed_s_by_worker: tuple[float, ...]
    slowest_worker: Mapping[str, object]
    terminal_reason_counts: Mapping[str, int]
    invalid_task_reason_counts: Mapping[str, int]
    hard_error_reason_counts: Mapping[str, int]

    def __post_init__(self) -> None:
        if type(self.update_id) is not int or self.update_id <= 0:
            raise TrainingMetricsError("runtime diagnostics update ID is invalid")
        identities = self.worker_identities
        if (
            type(identities) is not tuple
            or not identities
            or any(not isinstance(item, RuntimeWorkerIdentity) for item in identities)
            or tuple(item.worker_index for item in identities)
            != tuple(range(len(identities)))
        ):
            raise TrainingMetricsError("runtime worker identities are incomplete")
        for name in (
            "invalid_start_task_count",
            "coarse_fine_reachability_mismatch_count",
            "planner_suppressed_candidate_count",
            "zero_motion_reselection_count",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise TrainingMetricsError(f"runtime {name} is invalid")
        elapsed = self.macro_action_elapsed_s_by_worker
        if (
            type(elapsed) is not tuple
            or len(elapsed) != len(identities)
            or any(not _finite_nonnegative(value) for value in elapsed)
        ):
            raise TrainingMetricsError("runtime macro elapsed times are invalid")
        slowest = dict(self.slowest_worker)
        slowest_index = max(range(len(elapsed)), key=lambda index: (elapsed[index], -index))
        if set(slowest) != {"worker_index", "elapsed_s"} or (
            slowest["worker_index"] != slowest_index
            or slowest["elapsed_s"] != elapsed[slowest_index]
        ):
            raise TrainingMetricsError("runtime slowest worker differs")
        object.__setattr__(self, "slowest_worker", MappingProxyType(slowest))
        for name in (
            "terminal_reason_counts",
            "invalid_task_reason_counts",
            "hard_error_reason_counts",
        ):
            counts = dict(getattr(self, name))
            if any(
                not isinstance(reason, str)
                or not reason
                or type(count) is not int
                or count <= 0
                for reason, count in counts.items()
            ):
                raise TrainingMetricsError(f"runtime {name} are invalid")
            object.__setattr__(
                self, name, MappingProxyType(dict(sorted(counts.items())))
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "update_id": self.update_id,
            "worker_identities": [item.to_dict() for item in self.worker_identities],
            "invalid_start_task_count": self.invalid_start_task_count,
            "coarse_fine_reachability_mismatch_count": (
                self.coarse_fine_reachability_mismatch_count
            ),
            "planner_suppressed_candidate_count": (
                self.planner_suppressed_candidate_count
            ),
            "zero_motion_reselection_count": self.zero_motion_reselection_count,
            "macro_action_elapsed_s_by_worker": list(
                self.macro_action_elapsed_s_by_worker
            ),
            "slowest_worker": dict(self.slowest_worker),
            "terminal_reason_counts": dict(self.terminal_reason_counts),
            "invalid_task_reason_counts": dict(self.invalid_task_reason_counts),
            "hard_error_reason_counts": dict(self.hard_error_reason_counts),
        }

    @classmethod
    def from_dict(cls, value: object) -> "RewardV4RuntimeDiagnostics":
        expected = {
            "update_id",
            "worker_identities",
            "invalid_start_task_count",
            "coarse_fine_reachability_mismatch_count",
            "planner_suppressed_candidate_count",
            "zero_motion_reselection_count",
            "macro_action_elapsed_s_by_worker",
            "slowest_worker",
            "terminal_reason_counts",
            "invalid_task_reason_counts",
            "hard_error_reason_counts",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise TrainingMetricsError("runtime diagnostics structure is invalid")
        try:
            return cls(
                update_id=value["update_id"],
                worker_identities=tuple(
                    RuntimeWorkerIdentity.from_dict(item)
                    for item in value["worker_identities"]
                ),
                invalid_start_task_count=value["invalid_start_task_count"],
                coarse_fine_reachability_mismatch_count=value[
                    "coarse_fine_reachability_mismatch_count"
                ],
                planner_suppressed_candidate_count=value[
                    "planner_suppressed_candidate_count"
                ],
                zero_motion_reselection_count=value[
                    "zero_motion_reselection_count"
                ],
                macro_action_elapsed_s_by_worker=tuple(
                    value["macro_action_elapsed_s_by_worker"]
                ),
                slowest_worker=value["slowest_worker"],
                terminal_reason_counts=value["terminal_reason_counts"],
                invalid_task_reason_counts=value["invalid_task_reason_counts"],
                hard_error_reason_counts=value["hard_error_reason_counts"],
            )
        except (TypeError, ValueError) as error:
            raise TrainingMetricsError(
                "runtime diagnostics structure is invalid"
            ) from error

    @classmethod
    def from_rollout(
        cls, rollout: CommittedMacroRollout
    ) -> "RewardV4RuntimeDiagnostics":
        if not isinstance(rollout, CommittedMacroRollout):
            raise TrainingMetricsError("runtime diagnostics rollout is invalid")
        worker_count = len(rollout.slot_counts)
        if set(rollout.slot_counts) != set(range(worker_count)):
            raise TrainingMetricsError("runtime rollout workers are incomplete")
        states_by_worker: dict[int, dict[tuple[str, str, str], FormalWorkerState]] = {
            worker: {} for worker in range(worker_count)
        }

        def add_state(value: Mapping[str, object]) -> FormalWorkerState:
            try:
                state = FormalWorkerState.from_dict(
                    _thaw_journal_worker_state(value)
                )
            except ValueError as error:
                raise TrainingMetricsError(
                    "runtime worker state is invalid"
                ) from error
            if state.worker_index not in states_by_worker:
                raise TrainingMetricsError("runtime worker state index differs")
            key = (
                state.observation_identity.episode_id,
                state.platform_task_key_sha256,
                state.physical_snapshot_id,
            )
            states_by_worker[state.worker_index][key] = state
            return state

        terminal_identities: set[tuple[int, str, str]] = set()
        terminal_counts: Counter[str] = Counter()

        def add_terminal(state: FormalWorkerState) -> None:
            if state.terminal_reason is None:
                return
            try:
                reason = TerminalReason(state.terminal_reason).value
            except ValueError as error:
                raise TrainingMetricsError(
                    "runtime terminal reason is invalid"
                ) from error
            identity = (
                state.worker_index,
                state.observation_identity.episode_id,
                reason,
            )
            if identity not in terminal_identities:
                terminal_identities.add(identity)
                terminal_counts[reason] += 1

        for committed in rollout.committed:
            pre = add_state(committed.payload.pre_worker_state)
            post = add_state(committed.payload.post_worker_state)
            if pre.platform_type != committed.platform_type or (
                post.platform_type != committed.platform_type
            ):
                raise TrainingMetricsError("runtime committed platform differs")
            add_terminal(post)
        for item in rollout.invalid_tasks:
            add_state(item.post_worker_state)
        for item in rollout.rejected_actions:
            add_state(item.worker_state)
        for item in rollout.task_outcomes:
            state = add_state(item.post_worker_state)
            add_terminal(state)

        mismatch_by_snapshot: dict[tuple[int, str], int] = {}
        identities: list[RuntimeWorkerIdentity] = []
        for worker in range(worker_count):
            values = tuple(states_by_worker[worker].values())
            if not values:
                raise TrainingMetricsError("runtime worker has no bound identity")
            platforms = {state.platform_type for state in values}
            if len(platforms) != 1:
                raise TrainingMetricsError("runtime worker platform drifts")
            for state in values:
                decision = state.candidate_decision_snapshot
                if state.platform_type == "HOPPER":
                    mismatch = 0
                else:
                    mismatch = (
                        decision.fine_pose_candidate_count
                        - decision.globally_reachable_candidate_count
                    )
                key = (worker, state.physical_snapshot_id)
                previous = mismatch_by_snapshot.setdefault(key, mismatch)
                if previous != mismatch:
                    raise TrainingMetricsError(
                        "runtime coarse/fine audit drifts within one snapshot"
                    )
            identities.append(
                RuntimeWorkerIdentity(
                    worker_index=worker,
                    platform_type=next(iter(platforms)),
                    episode_ids=tuple(
                        sorted(
                            {
                                state.observation_identity.episode_id
                                for state in values
                            }
                        )
                    ),
                    task_ids=tuple(
                        sorted({state.platform_task_key_sha256 for state in values})
                    ),
                    physical_snapshot_ids=tuple(
                        sorted({state.physical_snapshot_id for state in values})
                    ),
                )
            )
        elapsed = rollout.macro_action_elapsed_s_by_worker
        if not elapsed:
            elapsed = (0.0,) * worker_count
        invalid_counts = Counter(item.reason_code for item in rollout.invalid_tasks)
        slowest_index = max(
            range(worker_count), key=lambda index: (elapsed[index], -index)
        )
        rejected_count = len(rollout.rejected_actions)
        hard_error_counts = {
            reason: count
            for reason, count in terminal_counts.items()
            if reason in {
                TerminalReason.HARD_FAILURE.value,
                TerminalReason.CANCELED.value,
            }
        }
        return cls(
            update_id=rollout.update_id,
            worker_identities=tuple(identities),
            invalid_start_task_count=sum(
                invalid_counts.get(reason, 0)
                for reason in ("WHEEL_START_NOT_SAFE", "LEGGED_START_NOT_SAFE")
            ),
            coarse_fine_reachability_mismatch_count=sum(
                mismatch_by_snapshot.values()
            ),
            planner_suppressed_candidate_count=rejected_count,
            zero_motion_reselection_count=rejected_count,
            macro_action_elapsed_s_by_worker=tuple(float(value) for value in elapsed),
            slowest_worker={
                "worker_index": slowest_index,
                "elapsed_s": float(elapsed[slowest_index]),
            },
            terminal_reason_counts=dict(terminal_counts),
            invalid_task_reason_counts=dict(invalid_counts),
            hard_error_reason_counts=hard_error_counts,
        )


@dataclass(frozen=True, slots=True)
class RewardV4MacroAudit:
    """One fully committed Reward V4 macro action prepared for metrics."""

    worker_index: int
    slot_index: int
    transition_id: str
    transition_sha256: str
    payload_sha256: str
    platform_type: str
    scale_bucket: TaskScaleBucket
    reward_stage: RewardStage
    reward_weights: RewardWeightsV4
    terminal_class: RewardTerminalClass
    coverage_before: float
    coverage_after: float
    priority_before: float
    priority_after: float
    executed_path_m: float
    cumulative_path_before_m: float
    cumulative_path_after_m: float
    task_scale_m: float
    coverage_cell_count_before: int
    coverage_cell_count_after: int
    coverage_mask_sha256: str
    next_coverage_mask_sha256: str
    reward_components: RewardComponentsV4
    advantage: float
    done: bool
    success_first_crossing: bool

    def __post_init__(self) -> None:
        for name in ("worker_index", "slot_index"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise TrainingMetricsError(f"Reward V4 macro {name} is invalid")
        if not isinstance(self.transition_id, str) or not self.transition_id:
            raise TrainingMetricsError("Reward V4 transition ID is invalid")
        for name in (
            "transition_sha256",
            "payload_sha256",
            "coverage_mask_sha256",
            "next_coverage_mask_sha256",
        ):
            _require_sha256(getattr(self, name), f"Reward V4 macro {name}")
        if self.platform_type not in PLATFORM_ID_BY_TYPE:
            raise TrainingMetricsError("Reward V4 macro platform is invalid")
        if not isinstance(self.scale_bucket, TaskScaleBucket):
            raise TrainingMetricsError("Reward V4 macro scale bucket is invalid")
        if not isinstance(self.reward_stage, RewardStage) or not isinstance(
            self.reward_weights, RewardWeightsV4
        ):
            raise TrainingMetricsError("Reward V4 macro reward stage is invalid")
        if not isinstance(self.terminal_class, RewardTerminalClass) or (
            self.terminal_class is RewardTerminalClass.INVALID_TRANSITION
        ):
            raise TrainingMetricsError("Reward V4 macro terminal class is invalid")
        for name in (
            "coverage_before",
            "coverage_after",
            "priority_before",
            "priority_after",
        ):
            value = getattr(self, name)
            if not _finite_ratio(value):
                raise TrainingMetricsError(f"Reward V4 macro {name} is invalid")
        if self.coverage_after < self.coverage_before:
            raise TrainingMetricsError("Reward V4 macro coverage regressed")
        if self.priority_after < self.priority_before:
            raise TrainingMetricsError("Reward V4 macro priority regressed")
        for name in (
            "executed_path_m",
            "cumulative_path_before_m",
            "cumulative_path_after_m",
        ):
            if not _finite_nonnegative(getattr(self, name)):
                raise TrainingMetricsError(f"Reward V4 macro {name} is invalid")
        if (
            not _finite_nonnegative(self.task_scale_m)
            or self.task_scale_m <= 0.0
            or not math.isclose(
                self.cumulative_path_after_m - self.cumulative_path_before_m,
                self.executed_path_m,
                rel_tol=1.0e-12,
                abs_tol=1.0e-9,
            )
        ):
            raise TrainingMetricsError("Reward V4 macro path facts differ")
        for name in (
            "coverage_cell_count_before",
            "coverage_cell_count_after",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise TrainingMetricsError(f"Reward V4 macro {name} is invalid")
        if self.coverage_cell_count_after < self.coverage_cell_count_before:
            raise TrainingMetricsError("Reward V4 macro coverage count regressed")
        if not isinstance(self.reward_components, RewardComponentsV4):
            raise TrainingMetricsError("Reward V4 macro components are invalid")
        component_total = math.fsum(
            (
                self.reward_components.coverage,
                self.reward_components.success,
                self.reward_components.terminal_gap,
                self.reward_components.priority,
                self.reward_components.path,
            )
        )
        if not math.isclose(
            component_total,
            self.reward_components.total,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ):
            raise TrainingMetricsError("Reward V4 macro component total differs")
        if not isinstance(self.advantage, (int, float)) or isinstance(
            self.advantage, bool
        ) or not math.isfinite(float(self.advantage)):
            raise TrainingMetricsError("Reward V4 macro advantage is invalid")
        if type(self.done) is not bool or type(self.success_first_crossing) is not bool:
            raise TrainingMetricsError("Reward V4 macro terminal facts are invalid")
        actual_crossing = (
            self.coverage_before < DEFAULT_REWARD_CONFIG.success_threshold
            <= self.coverage_after
        )
        if self.success_first_crossing != actual_crossing or (
            self.terminal_class is RewardTerminalClass.SUCCESS
        ) != actual_crossing:
            raise TrainingMetricsError("Reward V4 macro success facts differ")
        if self.done != (self.terminal_class is not RewardTerminalClass.CONTINUE):
            raise TrainingMetricsError("Reward V4 macro done fact differs")

    def to_dict(self) -> dict[str, object]:
        components = self.reward_components
        weights = self.reward_weights
        return {
            "worker_index": self.worker_index,
            "slot_index": self.slot_index,
            "transition_id": self.transition_id,
            "transition_sha256": self.transition_sha256,
            "payload_sha256": self.payload_sha256,
            "platform": self.platform_type,
            "scale_bucket": self.scale_bucket.value,
            "reward_stage": self.reward_stage.value,
            "reward_weights": {
                "coverage": weights.coverage,
                "success": weights.success,
                "terminal_gap": weights.terminal_gap,
                "priority": weights.priority,
                "path": weights.path,
            },
            "terminal_class": self.terminal_class.value,
            "done": self.done,
            "success_first_crossing": self.success_first_crossing,
            "coverage": {
                "before": self.coverage_before,
                "after": self.coverage_after,
                "delta": self.coverage_after - self.coverage_before,
                "cell_count_before": self.coverage_cell_count_before,
                "cell_count_after": self.coverage_cell_count_after,
                "mask_sha256_before": self.coverage_mask_sha256,
                "mask_sha256_after": self.next_coverage_mask_sha256,
            },
            "priority": {
                "before": self.priority_before,
                "after": self.priority_after,
                "delta": self.priority_after - self.priority_before,
            },
            "path": {
                "executed_m": self.executed_path_m,
                "cumulative_before_m": self.cumulative_path_before_m,
                "cumulative_after_m": self.cumulative_path_after_m,
                "task_scale_m": self.task_scale_m,
                "normalized_before": (
                    self.cumulative_path_before_m / self.task_scale_m
                ),
                "normalized_after": (
                    self.cumulative_path_after_m / self.task_scale_m
                ),
            },
            "reward_components": {
                "coverage": components.coverage,
                "success": components.success,
                "terminal_gap": components.terminal_gap,
                "priority": components.priority,
                "path": components.path,
                "total": components.total,
            },
            "advantage": float(self.advantage),
        }


def build_reward_v4_update_record(
    *,
    global_step: int,
    timestamp_utc: str,
    curriculum_phase: str,
    platform_allocation: Mapping[str, int],
    worker_strata: Sequence[WorkerStratum],
    actions_per_worker: int,
    macro_audits: Sequence[RewardV4MacroAudit],
    ppo_metrics: PPOUpdateMetrics,
    collect_wall_seconds: float,
    update_wall_seconds: float,
    worker_wait_seconds: float,
    curriculum_events: Sequence[Mapping[str, object]] = (),
    checkpoint_decision: Mapping[str, object] | None = None,
    invalid_tasks: Sequence[InvalidTaskAudit] = (),
    runtime_diagnostics: RewardV4RuntimeDiagnostics | None = None,
) -> dict[str, object]:
    """Build one complete Reward V4 metrics update record."""
    if type(global_step) is not int or global_step <= 0:
        raise TrainingMetricsError("Reward V4 metrics global step is invalid")
    if not isinstance(timestamp_utc, str) or not timestamp_utc.endswith("Z"):
        raise TrainingMetricsError("Reward V4 metrics timestamp is invalid")
    try:
        expected_strata = formal_worker_strata(curriculum_phase)
    except ValueError as error:
        raise TrainingMetricsError("Reward V4 metrics stage is invalid") from error
    strata = tuple(worker_strata)
    allocation = dict(platform_allocation)
    if strata != expected_strata or Counter(
        stratum.platform_type for stratum in strata
    ) != Counter(allocation):
        raise TrainingMetricsError("Reward V4 metrics worker strata differ")
    if actions_per_worker != DEFAULT_REWARD_CONFIG.macro_actions_per_worker:
        raise TrainingMetricsError("Reward V4 metrics action count differs")
    audits = tuple(macro_audits)
    expected_slots = tuple(
        (worker, slot)
        for worker in range(len(strata))
        for slot in range(actions_per_worker)
    )
    if (
        len(audits) != len(expected_slots)
        or any(not isinstance(audit, RewardV4MacroAudit) for audit in audits)
        or tuple((audit.worker_index, audit.slot_index) for audit in audits)
        != expected_slots
        or len({audit.transition_id for audit in audits}) != len(audits)
        or len({audit.payload_sha256 for audit in audits}) != len(audits)
    ):
        raise TrainingMetricsError("Reward V4 macro audit grid is invalid")
    for audit in audits:
        stratum = strata[audit.worker_index]
        if (
            audit.platform_type != stratum.platform_type
            or audit.scale_bucket is not stratum.scale_bucket
        ):
            raise TrainingMetricsError("Reward V4 macro audit stratum differs")
    if not isinstance(ppo_metrics, PPOUpdateMetrics):
        raise TrainingMetricsError("Reward V4 metrics require PPO metrics")
    timings = (collect_wall_seconds, update_wall_seconds, worker_wait_seconds)
    if any(not _finite_nonnegative(value) for value in timings):
        raise TrainingMetricsError("Reward V4 metrics timings are invalid")
    events = tuple(curriculum_events)
    invalid = tuple(invalid_tasks)
    if any(not isinstance(event, Mapping) for event in events) or (
        checkpoint_decision is not None
        and not isinstance(checkpoint_decision, Mapping)
    ):
        raise TrainingMetricsError("Reward V4 curriculum events are invalid")
    if any(not isinstance(item, InvalidTaskAudit) for item in invalid):
        raise TrainingMetricsError("Reward V4 invalid-task audits are invalid")
    invalid_reason_counts = Counter(item.reason_code for item in invalid)

    grouped: dict[tuple[str, TaskScaleBucket], list[RewardV4MacroAudit]] = {}
    for audit in audits:
        grouped.setdefault((audit.platform_type, audit.scale_bucket), []).append(audit)
    ppo_by_stratum = _reward_v4_ppo_strata(ppo_metrics)
    if set(ppo_by_stratum) != set(grouped):
        raise TrainingMetricsError("Reward V4 PPO strata differ")
    stratum_records = {
        _reward_v4_stratum_key(platform, bucket): _reward_v4_stratum_record(
            values,
            ppo_by_stratum[(platform, bucket)],
        )
        for (platform, bucket), values in sorted(
            grouped.items(),
            key=lambda item: (
                PLATFORM_ID_BY_TYPE[item[0][0]],
                SCALE_BUCKET_ID_BY_BUCKET[item[0][1]],
            ),
        )
    }
    if runtime_diagnostics is not None:
        if (
            not isinstance(runtime_diagnostics, RewardV4RuntimeDiagnostics)
            or runtime_diagnostics.update_id != global_step
            or len(runtime_diagnostics.worker_identities) != len(strata)
            or any(
                identity.worker_index != stratum.worker_index
                or identity.platform_type != stratum.platform_type
                for identity, stratum in zip(
                    runtime_diagnostics.worker_identities, strata
                )
            )
        ):
            raise TrainingMetricsError(
                "Reward V4 runtime diagnostic identity differs"
            )
    record: dict[str, object] = {
        "schema_version": (
            TRAINING_UPDATE_METRICS_SCHEMA
            if runtime_diagnostics is not None
            else REWARD_V4_TRAINING_UPDATE_METRICS_SCHEMA_V6
        ),
        "global_step": global_step,
        "timestamp_utc": timestamp_utc,
        "curriculum_phase": curriculum_phase,
        "platform_allocation": allocation,
        "actions_per_worker": actions_per_worker,
        "transition_count": len(audits),
        "macro_transitions": [audit.to_dict() for audit in audits],
        "strata": stratum_records,
        "ppo": _ppo_record(ppo_metrics),
        "timing": {
            "collect_wall_seconds": float(collect_wall_seconds),
            "update_wall_seconds": float(update_wall_seconds),
            "worker_wait_seconds": float(worker_wait_seconds),
        },
        "curriculum_events": [dict(event) for event in events],
        "checkpoint_decision": (
            None if checkpoint_decision is None else dict(checkpoint_decision)
        ),
        "invalid_tasks": {
            "invalid_start_task_count": sum(
                invalid_reason_counts.get(reason, 0)
                for reason in (
                    "WHEEL_START_NOT_SAFE",
                    "LEGGED_START_NOT_SAFE",
                )
            ),
            "reason_counts": dict(sorted(invalid_reason_counts.items())),
        },
    }
    if runtime_diagnostics is not None:
        record["runtime_diagnostics"] = runtime_diagnostics.to_dict()
    _validate_record(record)
    return record


def _materialize_identity_sets(
    values: Mapping[str, Mapping[str, set[str]]],
) -> dict[str, dict[str, list[str]]]:
    return {
        platform: {
            name: sorted(identities)
            for name, identities in fields.items()
        }
        for platform, fields in values.items()
    }


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
            **{name: 0 for name in _CANDIDATE_COUNT_FIELDS},
            "no_candidate_termination_count": 0,
            "physical_exhaustion_count": 0,
            "planner_exhausted_count": 0,
        }
        for platform in sorted(allocation)
    }
    candidate_identity_sets = {
        platform: {
            "physical_snapshot_ids": set(),
            "physical_reachability_algorithm_ids": set(),
        }
        for platform in sorted(allocation)
    }
    for index, diagnostics in enumerate(candidates):
        platform = platforms[index % worker_count]
        values = candidate_by_platform[platform]
        for name in _CANDIDATE_COUNT_FIELDS:
            values[name] += getattr(diagnostics, name)
        if diagnostics.physical_snapshot_id:
            candidate_identity_sets[platform]["physical_snapshot_ids"].add(
                diagnostics.physical_snapshot_id
            )
        if diagnostics.physical_reachability_algorithm_id:
            candidate_identity_sets[platform][
                "physical_reachability_algorithm_ids"
            ].add(diagnostics.physical_reachability_algorithm_id)
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
            for name in _CANDIDATE_COUNT_FIELDS
        }
        for platform in sorted(allocation)
    }
    terminal_identity_sets = {
        platform: {
            "physical_snapshot_ids": set(),
            "physical_reachability_algorithm_ids": set(),
        }
        for platform in sorted(allocation)
    }
    remaining_counts: list[int] = []
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
        if audit.candidate_diagnostics.physical_snapshot_id:
            terminal_identity_sets[platform]["physical_snapshot_ids"].add(
                audit.candidate_diagnostics.physical_snapshot_id
            )
        if audit.candidate_diagnostics.physical_reachability_algorithm_id:
            terminal_identity_sets[platform][
                "physical_reachability_algorithm_ids"
            ].add(
                audit.candidate_diagnostics.physical_reachability_algorithm_id
            )
        if audit.remaining_coverable_detail_cell_count is not None:
            remaining_counts.append(
                audit.remaining_coverable_detail_cell_count
            )
        if audit.reason in _PHYSICAL_EXHAUSTION_REASONS:
            candidate_by_platform[platform]["physical_exhaustion_count"] += 1
        if audit.reason is TerminalReason.PLANNER_EXHAUSTED:
            candidate_by_platform[platform]["planner_exhausted_count"] += 1
    start_mean = float(start_coverage.astype(np.float64).mean())
    end_mean = float(end_coverage.astype(np.float64).mean())
    record: dict[str, object] = {
        "schema_version": LEGACY_TRAINING_UPDATE_METRICS_SCHEMA,
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
        "candidate": {
            "by_platform": candidate_by_platform,
            "identity_by_platform": _materialize_identity_sets(
                candidate_identity_sets
            ),
        },
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
            "candidate_identity_by_platform": _materialize_identity_sets(
                terminal_identity_sets
            ),
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


def _reward_v4_ppo_strata(
    metrics: PPOUpdateMetrics,
) -> dict[tuple[str, TaskScaleBucket], StratumPPOUpdateMetrics]:
    platform_by_id = {
        identifier: platform
        for platform, identifier in PLATFORM_ID_BY_TYPE.items()
    }
    bucket_by_id = {
        identifier: bucket
        for bucket, identifier in SCALE_BUCKET_ID_BY_BUCKET.items()
    }
    result: dict[tuple[str, TaskScaleBucket], StratumPPOUpdateMetrics] = {}
    for item in metrics.strata:
        if not isinstance(item, StratumPPOUpdateMetrics):
            raise TrainingMetricsError("Reward V4 PPO stratum metric is invalid")
        try:
            key = (
                platform_by_id[item.platform_id],
                bucket_by_id[item.scale_bucket_id],
            )
        except KeyError as error:
            raise TrainingMetricsError(
                "Reward V4 PPO stratum identity is invalid"
            ) from error
        if key in result:
            raise TrainingMetricsError("Reward V4 PPO stratum is duplicated")
        result[key] = item
    return result


def _reward_v4_stratum_record(
    audits: Sequence[RewardV4MacroAudit],
    loss: StratumPPOUpdateMetrics,
) -> dict[str, object]:
    values = tuple(audits)
    if not values or loss.sample_count != len(values):
        raise TrainingMetricsError("Reward V4 stratum sample count differs")
    stage_values = {audit.reward_stage for audit in values}
    weight_values = {audit.reward_weights for audit in values}
    if len(stage_values) != 1 or len(weight_values) != 1:
        raise TrainingMetricsError("Reward V4 stratum reward state drifts")
    stage = next(iter(stage_values))
    weights = next(iter(weight_values))
    component_names = (
        "coverage",
        "success",
        "terminal_gap",
        "priority",
        "path",
        "total",
    )
    component_means = {
        name: math.fsum(
            float(getattr(audit.reward_components, name)) for audit in values
        )
        / len(values)
        for name in component_names
    }
    advantages = np.asarray(
        [audit.advantage for audit in values], dtype=np.float64
    )
    normalized_paths = np.asarray(
        [
            audit.cumulative_path_after_m / audit.task_scale_m
            for audit in values
        ],
        dtype=np.float64,
    )
    if not np.isfinite(advantages).all() or not np.isfinite(normalized_paths).all():
        raise TrainingMetricsError("Reward V4 stratum metric is non-finite")
    return {
        "platform": values[0].platform_type,
        "scale_bucket": values[0].scale_bucket.value,
        "sample_count": len(values),
        "reward_stage": stage.value,
        "reward_weights": {
            "coverage": weights.coverage,
            "success": weights.success,
            "terminal_gap": weights.terminal_gap,
            "priority": weights.priority,
            "path": weights.path,
        },
        "reward_components_mean": component_means,
        "advantage": {
            "mean": float(advantages.mean(dtype=np.float64)),
            "std": float(advantages.std(dtype=np.float64)),
            "min": float(advantages.min()),
            "max": float(advantages.max()),
        },
        "result": {
            "terminal_count": sum(audit.done for audit in values),
            "success_first_crossing_count": sum(
                audit.success_first_crossing for audit in values
            ),
            "mean_coverage_after": math.fsum(
                audit.coverage_after for audit in values
            )
            / len(values),
            "mean_priority_after": math.fsum(
                audit.priority_after for audit in values
            )
            / len(values),
            "mean_normalized_path_after": float(
                normalized_paths.mean(dtype=np.float64)
            ),
        },
        "loss": {
            "logical_weight": loss.logical_weight,
            "total_loss": loss.total_loss,
            "policy_loss": loss.policy_loss,
            "value_loss": loss.value_loss,
            "frontier_entropy": loss.frontier_entropy,
            "theta_entropy": loss.theta_entropy,
            "approx_kl": loss.approx_kl,
        },
    }


def _reward_v4_stratum_key(
    platform: str, bucket: TaskScaleBucket
) -> str:
    return f"{platform}/{bucket.value}"


def _ppo_record(metrics: PPOUpdateMetrics) -> dict[str, object]:
    return {
        "total_loss": metrics.total_loss,
        "policy_loss": metrics.policy_loss,
        "value_loss": metrics.value_loss,
        "frontier_entropy": metrics.frontier_entropy,
        "theta_entropy": metrics.theta_entropy,
        "approx_kl": metrics.approx_kl,
        "gradient_norm": metrics.gradient_norm,
        "clipped_gradient_norm": metrics.clipped_gradient_norm,
        "parameter_change_l2": metrics.parameter_change_l2,
        "optimizer_steps": metrics.optimizer_steps,
        "epochs_completed": metrics.epochs_completed,
        "target_kl_early_stopped": metrics.target_kl_early_stopped,
    }


class TrainingMetricsJournal:
    """Single-writer, fsync-backed formal training update journal."""

    def __init__(
        self,
        path: Path,
        *,
        resume_global_step: int,
        checkpoint_record: Mapping[str, object] | None = None,
        checkpoint_record_sha256: str | None = None,
    ) -> None:
        if not isinstance(path, Path) or not path.is_absolute():
            raise TrainingMetricsError("metrics journal path must be absolute")
        if type(resume_global_step) is not int or resume_global_step < 0:
            raise TrainingMetricsError("metrics resume global step is invalid")
        if (checkpoint_record is None) != (checkpoint_record_sha256 is None):
            raise TrainingMetricsError(
                "metrics checkpoint record identity is incomplete"
            )
        expected_record: dict[str, object] | None = None
        if checkpoint_record is not None:
            expected_record = dict(checkpoint_record)
            _validate_record(expected_record)
            if expected_record["global_step"] != resume_global_step:
                raise TrainingMetricsError(
                    "metrics checkpoint record step differs"
                )
            if (
                checkpoint_record_sha256
                != training_metrics_record_sha256(expected_record)
            ):
                raise TrainingMetricsError(
                    "metrics checkpoint record hash differs"
                )
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
        row_schemas = {row["schema_version"] for row in rows}
        if len(row_schemas) > 1:
            raise TrainingMetricsError("metrics journal schema changes within a run")
        expected_schema = (
            expected_record.get("schema_version")
            if expected_record is not None
            else None
        )
        self._schema_version = (
            next(iter(row_schemas)) if row_schemas else expected_schema
        )
        if (
            expected_schema is not None
            and self._schema_version != expected_schema
        ):
            raise TrainingMetricsError("metrics journal schema differs from checkpoint")
        if rows:
            steps = [row["global_step"] for row in rows]
            if any(right != left + 1 for left, right in zip(steps, steps[1:])):
                raise TrainingMetricsError("metrics global steps are not contiguous")
            if steps[-1] > resume_global_step:
                raise TrainingMetricsError("metrics journal is ahead of checkpoint")
            self._last_step = int(steps[-1])
        else:
            self._last_step = 0 if expected_record is not None else resume_global_step
        if self._last_step == resume_global_step:
            if expected_record is not None and rows and (
                training_metrics_record_sha256(rows[-1])
                != checkpoint_record_sha256
            ):
                raise TrainingMetricsError(
                    "metrics journal conflicts with checkpoint record"
                )
        elif (
            expected_record is not None
            and self._last_step == resume_global_step - 1
        ):
            self.mirror_checkpoint_record(
                expected_record,
                expected_sha256=checkpoint_record_sha256,
            )
        elif expected_record is not None:
            raise TrainingMetricsError(
                "metrics journal is more than one update behind checkpoint"
            )
        else:
            raise TrainingMetricsError("metrics journal does not reach checkpoint")

    @property
    def last_global_step(self) -> int:
        return self._last_step

    def append(self, record: Mapping[str, object]) -> None:
        if not isinstance(record, Mapping):
            raise TrainingMetricsError("metrics record must be a mapping")
        payload = dict(record)
        _validate_record(payload)
        if (
            self._schema_version is not None
            and payload["schema_version"] != self._schema_version
        ):
            raise TrainingMetricsError("metrics journal schema changes within a run")
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
        self._schema_version = str(payload["schema_version"])

    def mirror_checkpoint_record(
        self,
        record: Mapping[str, object],
        *,
        expected_sha256: str,
    ) -> None:
        """Append the checkpoint-owned next record, or verify its exact mirror."""
        if not isinstance(record, Mapping):
            raise TrainingMetricsError("metrics record must be a mapping")
        payload = dict(record)
        actual_sha256 = training_metrics_record_sha256(payload)
        if expected_sha256 != actual_sha256:
            raise TrainingMetricsError("metrics checkpoint record hash differs")
        step = payload["global_step"]
        if step == self._last_step:
            rows = self._read_rows()
            if not rows or training_metrics_record_sha256(rows[-1]) != actual_sha256:
                raise TrainingMetricsError(
                    "metrics journal conflicts with checkpoint record"
                )
            return
        if step != self._last_step + 1:
            raise TrainingMetricsError(
                "metrics checkpoint record is not the next global step"
            )
        self.append(payload)

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
    schema = record.get("schema_version")
    if schema == TRAINING_UPDATE_METRICS_SCHEMA:
        _validate_reward_v4_record(record, require_runtime_diagnostics=True)
        return
    if schema == REWARD_V4_TRAINING_UPDATE_METRICS_SCHEMA_V6:
        _validate_reward_v4_record(record, require_runtime_diagnostics=False)
        return
    if schema != LEGACY_TRAINING_UPDATE_METRICS_SCHEMA:
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
    aggregate_fields = set(_CANDIDATE_COUNT_FIELDS) | {
        "no_candidate_termination_count",
        "physical_exhaustion_count",
        "planner_exhausted_count",
    }
    for values in candidate_by_platform.values():
        _validate_diagnostic_counts(values, aggregate_fields)
    _validate_identity_groups(
        candidate.get("identity_by_platform"), set(candidate_by_platform)
    )
    terminal = record.get("terminal")
    terminal_by_platform = (
        terminal.get("candidate_diagnostics_by_platform")
        if isinstance(terminal, Mapping)
        else None
    )
    if not isinstance(terminal_by_platform, Mapping) or not terminal_by_platform:
        raise TrainingMetricsError("metrics terminal diagnostics are missing")
    for values in terminal_by_platform.values():
        _validate_diagnostic_counts(values, set(_CANDIDATE_COUNT_FIELDS))
    _validate_identity_groups(
        terminal.get("candidate_identity_by_platform"),
        set(terminal_by_platform),
    )
    _validate_finite_json(record)


def _validate_reward_v4_record(
    record: dict[str, object], *, require_runtime_diagnostics: bool
) -> None:
    expected_fields = {
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
        "invalid_tasks",
    }
    if require_runtime_diagnostics:
        expected_fields.add("runtime_diagnostics")
    if set(record) != expected_fields:
        raise TrainingMetricsError("Reward V4 metrics record structure is invalid")
    if type(record.get("global_step")) is not int or record["global_step"] <= 0:
        raise TrainingMetricsError("Reward V4 metrics global step is invalid")
    if require_runtime_diagnostics:
        runtime = RewardV4RuntimeDiagnostics.from_dict(
            record.get("runtime_diagnostics")
        )
        if runtime.update_id != record["global_step"]:
            raise TrainingMetricsError(
                "Reward V4 runtime diagnostics update differs"
            )
    if (
        not isinstance(record.get("timestamp_utc"), str)
        or not str(record["timestamp_utc"]).endswith("Z")
        or record.get("curriculum_phase")
        not in {"GROUND_R1", "GROUND_R2_HOPPER_R1", "THREE_PLATFORM_R2"}
        or record.get("actions_per_worker")
        != DEFAULT_REWARD_CONFIG.macro_actions_per_worker
    ):
        raise TrainingMetricsError("Reward V4 metrics identity is invalid")
    invalid_tasks = record.get("invalid_tasks")
    if (
        not isinstance(invalid_tasks, Mapping)
        or set(invalid_tasks)
        != {"invalid_start_task_count", "reason_counts"}
        or type(invalid_tasks.get("invalid_start_task_count")) is not int
        or int(invalid_tasks["invalid_start_task_count"]) < 0
        or not isinstance(invalid_tasks.get("reason_counts"), Mapping)
        or any(
            not isinstance(reason, str)
            or not reason
            or type(count) is not int
            or count <= 0
            for reason, count in invalid_tasks["reason_counts"].items()
        )
    ):
        raise TrainingMetricsError("Reward V4 invalid-task metrics are invalid")
    allocation = record.get("platform_allocation")
    if not isinstance(allocation, Mapping) or not allocation:
        raise TrainingMetricsError("Reward V4 metrics allocation is invalid")
    macros = record.get("macro_transitions")
    transition_count = record.get("transition_count")
    if (
        not isinstance(macros, list)
        or type(transition_count) is not int
        or transition_count != len(macros)
        or not macros
    ):
        raise TrainingMetricsError("Reward V4 macro metrics are incomplete")
    macro_fields = {
        "worker_index",
        "slot_index",
        "transition_id",
        "transition_sha256",
        "payload_sha256",
        "platform",
        "scale_bucket",
        "reward_stage",
        "reward_weights",
        "terminal_class",
        "done",
        "success_first_crossing",
        "coverage",
        "priority",
        "path",
        "reward_components",
        "advantage",
    }
    if any(not isinstance(row, Mapping) or set(row) != macro_fields for row in macros):
        raise TrainingMetricsError("Reward V4 macro metric structure is invalid")
    transition_ids = [row["transition_id"] for row in macros]
    payload_ids = [row["payload_sha256"] for row in macros]
    if (
        len(set(transition_ids)) != len(transition_ids)
        or len(set(payload_ids)) != len(payload_ids)
        or any(
            not isinstance(row["transition_id"], str)
            or not row["transition_id"]
            for row in macros
        )
    ):
        raise TrainingMetricsError("Reward V4 macro identity is invalid")
    for row in macros:
        _require_sha256(row["transition_sha256"], "Reward V4 transition")
        _require_sha256(row["payload_sha256"], "Reward V4 payload")
    strata = record.get("strata")
    expected_stratum_count = (
        8 if record["curriculum_phase"] == "GROUND_R1" else 12
    )
    if not isinstance(strata, Mapping) or len(strata) != expected_stratum_count:
        raise TrainingMetricsError("Reward V4 stratum metrics are incomplete")
    if sum(
        int(value.get("sample_count", -1))
        for value in strata.values()
        if isinstance(value, Mapping)
    ) != transition_count:
        raise TrainingMetricsError("Reward V4 stratum sample counts differ")
    if not isinstance(record.get("ppo"), Mapping) or not isinstance(
        record.get("timing"), Mapping
    ):
        raise TrainingMetricsError("Reward V4 update metrics are incomplete")
    if not isinstance(record.get("curriculum_events"), list) or (
        record.get("checkpoint_decision") is not None
        and not isinstance(record.get("checkpoint_decision"), Mapping)
    ):
        raise TrainingMetricsError("Reward V4 curriculum metrics are invalid")
    if "candidate" in record or "oracle" in record:
        raise TrainingMetricsError("Reward V4 metrics contain forbidden diagnostics")
    _validate_finite_json(record)


def training_metrics_record_sha256(record: Mapping[str, object]) -> str:
    """Hash one validated metrics record without JSONL framing."""
    if not isinstance(record, Mapping):
        raise TrainingMetricsError("metrics record must be a mapping")
    payload = dict(record)
    _validate_record(payload)
    try:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise TrainingMetricsError(
            "metrics record is not JSON-compatible"
        ) from error
    return hashlib.sha256(encoded).hexdigest()


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


def _validate_identity_groups(
    value: object, expected_platforms: set[object]
) -> None:
    expected_fields = {
        "physical_snapshot_ids",
        "physical_reachability_algorithm_ids",
    }
    if not isinstance(value, Mapping) or set(value) != expected_platforms:
        raise TrainingMetricsError("metrics candidate identities are invalid")
    for fields in value.values():
        if not isinstance(fields, Mapping) or set(fields) != expected_fields:
            raise TrainingMetricsError("metrics candidate identities are invalid")
        for identities in fields.values():
            if (
                not isinstance(identities, list)
                or any(
                    not isinstance(identity, str) or not identity
                    for identity in identities
                )
                or identities != sorted(set(identities))
            ):
                raise TrainingMetricsError("metrics candidate identities are invalid")


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


def _finite_ratio(value: object) -> bool:
    return _finite_nonnegative(value) and float(value) <= 1.0


def _require_sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise TrainingMetricsError(f"{name} identity is invalid")
    return value


__all__ = [
    "LEGACY_TRAINING_UPDATE_METRICS_SCHEMA",
    "REWARD_V4_TRAINING_UPDATE_METRICS_SCHEMA_V6",
    "RewardV4RuntimeDiagnostics",
    "RuntimeWorkerIdentity",
    "RewardV4MacroAudit",
    "TRAINING_UPDATE_METRICS_SCHEMA",
    "TrainingMetricsError",
    "TrainingMetricsJournal",
    "build_reward_v4_update_record",
    "build_training_update_record",
    "training_metrics_record_sha256",
]

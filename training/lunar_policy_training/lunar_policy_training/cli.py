"""Current formal and development-smoke training entry point."""

from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import random
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from types import FrameType
from typing import TypeVar

import numpy as np
import torch
from lunar_planner_training_bridge import PlanningOutcome, TrainingPlanRequest

from .capability_freeze import (
    CapabilityFreezeError,
    FrozenCapabilityBundle,
    FrozenCapabilityEnvironmentFactory,
)
from .project_capability import load_project_formal_capability
from .polar_data.formal_cache import (
    FormalCache,
    FormalCacheError,
    FormalCacheIdentity,
    current_v3_identity,
    load_formal_cache,
    prepare_formal_training_cache,
)
from .polar_data.multires_scene import GENERATOR_SHA256
from .budget import (
    BudgetExceededError,
    CalibrationMeasurement,
    HorizonCalibrationMeasurement,
    ROLLOUT_HORIZON_TRANSITIONS_PER_WORKER,
    TRAINING_ROLLOUT_UPDATE_UPPER_BOUND_GPU_SECONDS,
    TrainingBudget,
    calibrate_runtime,
    extend_budget_manifest,
    select_qualified_rollout_horizon,
)
from .checkpoint import (
    FORMAL_ENVIRONMENT_STATE_SCHEMA_VERSION,
    OBSERVATION_CONTRACT_VERSION,
    RunIdentity,
    TrainingCheckpointV6,
    build_training_checkpoint,
    config_sha256,
    load_checkpoint,
    load_checkpoint_for_resume,
    migrate_checkpoint_source_commit,
    restore_training_state,
    save_checkpoint_atomic,
)
from .config import (
    PPOConfig,
    ROLLOUT_HORIZON_CANDIDATES,
    ResolvedTrainingConfig,
    load_training_config,
    resolve_training_config,
    with_rollout_horizon,
)
from .environment.parallel_pool import (
    ParallelActions,
    ParallelEnvironmentWorker,
    ParallelEnvPool,
    ParallelPoolError,
    joint_worker_allocation,
)
from .environment.candidate_builder import CandidateDiagnostics
from .environment.formal_builder import (
    FormalEnvironmentAssembly,
    FormalEnvironmentBuilder,
    _formal_scheduled_entries,
)
from .environment.formal_episode_state import FormalWorkerState
from .environment.macro_step import PlannerTransition, PolicyAction, TerminalAudit
from .environment.v3_environment import PreparedPlanRequest, create_v3_environment
from .training_semantics import training_semantics_sha256
from .formal_preflight import (
    FormalPreflightError,
    _request_signature,
    run_formal_preflight,
)
from .sensor_performance import (
    SensorPerformanceError,
    current_host_identity,
    load_sensor_performance_report,
    require_clean_sensor_source,
    sensor_source_commit,
    validate_sensor_performance_report,
)
from .curriculum import (
    CurriculumSchedule,
    FORMAL_SEED,
    PLATFORMS,
    REWARD_CALIBRATION_SEEDS,
    next_joint_evaluation_gpu_seconds,
    resume_worker_episode_states,
)
from .evaluation.release_gate import (
    GateResult,
    evaluate_release_gate,
    load_gate_rules,
)
from .evaluation.report import (
    EvaluationReport,
    FORMAL_EVALUATION_WATCHDOG_SECONDS,
    FormalEvaluationBatch,
    evaluate_formal_policy,
    evaluate_proxy_policy,
    report_sha256,
    write_report,
)
from .policy.cross_attention import CrossAttentionPolicy, sample_action
from .policy.action_semantics import apply_goal_theta
from .policy.observation import ObservationIdentity, PolicyBatch
from .proxy_scenario import proxy_environment_factory, proxy_observation
from .ppo.collector import (
    CollectedRollout,
    CollectorConfig,
    EnvStep,
    collect_rollout as collect_ppo_rollout,
)
from .ppo.checkpoint import _semantic_sha256
from .ppo.rollout import RolloutBatch
from .ppo.trainer import PPOTrainer, PPOUpdateMetrics
from .reward import compute_transition_reward, reward_weights_sha256
from .training_metrics import (
    TrainingMetricsJournal,
    build_training_update_record,
)


_Rollout = TypeVar("_Rollout")


class ArtifactRootError(ValueError):
    """Runtime artifacts must use an explicit absolute path outside Git."""


class PreflightError(ValueError):
    """A formal command failed before artifact or accelerator access."""


def _build_formal_resume_equivalence_evidence(
    *,
    checkpoint_relative_path: str,
    checkpoint_sha256: str,
    checkpoint_roundtrip: bool,
    rollout_exact: bool,
    uninterrupted: object,
    resumed: object,
    uninterrupted_observation_sha256: str,
    resumed_observation_sha256: str,
    uninterrupted_candidate_sha256: str,
    resumed_candidate_sha256: str,
    uninterrupted_request_sha256: str,
    resumed_request_sha256: str,
) -> dict[str, object]:
    """Fail closed unless update two is exact across every resumed boundary."""
    if (
        not isinstance(checkpoint_relative_path, str)
        or not checkpoint_relative_path
        or Path(checkpoint_relative_path).is_absolute()
        or ".." in Path(checkpoint_relative_path).parts
        or checkpoint_roundtrip is not True
        or rollout_exact is not True
        or not isinstance(checkpoint_sha256, str)
        or len(checkpoint_sha256) != 64
    ):
        raise PreflightError("formal resume checkpoint roundtrip is invalid")
    if any(
        getattr(value, "schema_version", None) != "lunar-ppo-checkpoint/v6"
        for value in (uninterrupted, resumed)
    ):
        raise PreflightError("formal resume checkpoint schema differs")

    comparisons = {
        "model": (
            _semantic_sha256(getattr(uninterrupted, "model_state")),
            _semantic_sha256(getattr(resumed, "model_state")),
        ),
        "optimizer": (
            _semantic_sha256(getattr(uninterrupted, "optimizer_state")),
            _semantic_sha256(getattr(resumed, "optimizer_state")),
        ),
        "rng": (
            _semantic_sha256(getattr(uninterrupted, "rng_state")),
            _semantic_sha256(getattr(resumed, "rng_state")),
        ),
        "environment_state": (
            _semantic_sha256(getattr(uninterrupted, "environment_state")),
            _semantic_sha256(getattr(resumed, "environment_state")),
        ),
        "observation": (
            uninterrupted_observation_sha256,
            resumed_observation_sha256,
        ),
        "candidate": (
            uninterrupted_candidate_sha256,
            resumed_candidate_sha256,
        ),
        "first_request": (
            uninterrupted_request_sha256,
            resumed_request_sha256,
        ),
    }
    for name, (left, right) in comparisons.items():
        if (
            not isinstance(left, str)
            or len(left) != 64
            or not isinstance(right, str)
            or len(right) != 64
            or left != right
        ):
            raise PreflightError(f"formal resume {name} differs at update two")

    body: dict[str, object] = {
        "checkpoint_schema": "lunar-ppo-checkpoint/v6",
        "checkpoint_relative_path": checkpoint_relative_path,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_roundtrip": True,
        "rollout_exact": True,
        "model_exact": True,
        "optimizer_exact": True,
        "rng_exact": True,
        "environment_state_exact": True,
        "observation_exact": True,
        "candidate_exact": True,
        "first_request_exact": True,
        "uninterrupted_update": 2,
        "resumed_update": 2,
    }
    body["evidence_sha256"] = hashlib.sha256(
        json.dumps(
            {**body, "comparison_sha256s": comparisons},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return body


def _collector_config_for_run(
    config: ResolvedTrainingConfig,
    *,
    horizon: int | None = None,
) -> CollectorConfig:
    """Sample formal PPO actions; keep bounded development smoke repeatable."""
    if not isinstance(config, ResolvedTrainingConfig):
        raise ValueError("rollout collection requires resolved training config")
    return CollectorConfig(
        horizon=config.ppo.rollout_horizon if horizon is None else horizon,
        deterministic=config.run_kind != "formal",
    )


@dataclass(slots=True)
class SignalStopFlag:
    """Signal-safe state: handlers only record the requested stop."""

    requested: bool = False
    signal_number: int | None = None

    def _handle(self, signal_number: int, frame: FrameType | None) -> None:
        self.requested = True
        self.signal_number = signal_number

    @contextlib.contextmanager
    def installed(self) -> Iterator["SignalStopFlag"]:
        previous = {
            signal_number: signal.getsignal(signal_number)
            for signal_number in (signal.SIGINT, signal.SIGTERM)
        }
        try:
            for signal_number in previous:
                signal.signal(signal_number, self._handle)
            yield self
        finally:
            for signal_number, handler in previous.items():
                signal.signal(signal_number, handler)


@dataclass(frozen=True, slots=True)
class TrainingLoopState:
    global_step: int
    rollout_discarded: bool
    stop_signal: int | None
    latest_checkpoint_gpu_seconds: float
    candidate_checkpoint_gpu_seconds: float


@dataclass(frozen=True, slots=True)
class CudaSmokeEvidence:
    device_name: str
    interrupted_global_step: int
    resumed_global_step: int
    interrupted_consumed_gpu_seconds: float
    resumed_consumed_gpu_seconds: float
    platform_allocation: dict[str, int]
    signal_observed_at_update_boundary: bool


@dataclass(frozen=True, slots=True)
class _RunEvidence:
    global_step: int
    consumed_gpu_seconds: float
    platform_allocation: dict[str, int]
    signal_observed_at_update_boundary: bool


@dataclass(frozen=True, slots=True)
class _CollectedTrainingRollout:
    collected: CollectedRollout
    raw_rewards: np.ndarray
    start_coverage: np.ndarray
    end_coverage: np.ndarray
    success_first_crossings: tuple[bool, ...]
    planning_outcomes: tuple[PlanningOutcome, ...]
    candidate_diagnostics: tuple[CandidateDiagnostics, ...]
    no_candidate_terminations: tuple[bool, ...]
    terminal_audits: tuple[TerminalAudit | None, ...]
    collect_wall_seconds: float
    worker_wait_seconds: float


@dataclass(frozen=True, slots=True)
class _CompletedTrainingUpdate:
    ppo_metrics: PPOUpdateMetrics
    update_wall_seconds: float


@dataclass(frozen=True, slots=True)
class CalibratedRunState:
    config: ResolvedTrainingConfig
    budget: TrainingBudget
    allocation: dict[str, int]
    selected_workers: int
    micro_batch_size: int
    rollout_horizon: int
    reward_hash: str
    scenario_schedule_id: str
    formal_seed: int
    reward_calibration_seeds: tuple[int, int, int]
    calibration_end_gpu_seconds: float
    run_identity: RunIdentity
    cache_manifest_path: Path | None
    cache_manifest_sha256: str | None
    sensor_performance_sha256: str | None


class _ParallelPoolVectorEnv:
    """Thin production adapter from the shared pool to the PPO collector."""

    def __init__(self, pool: ParallelEnvPool, *, policy_version: int) -> None:
        if not isinstance(pool, ParallelEnvPool):
            raise ValueError("parallel rollout adapter requires ParallelEnvPool")
        self._pool = pool
        self.env_count = pool.worker_count
        self._current = None
        self.policy_versions: list[int] = []
        self.planning_outcomes: list[object] = []
        self.reason_codes: list[str] = []
        self.candidate_diagnostics: list[CandidateDiagnostics] = []
        self.no_candidate_terminations: list[bool] = []
        self.terminal_audits: list[TerminalAudit | None] = []
        self.raw_rewards: list[float] = []
        self.success_first_crossings: list[bool] = []
        self.worker_wait_seconds = 0.0
        self._normalization_state: Mapping[str, object] | None = None
        self.set_policy_version(policy_version)

    def use_normalization_state(
        self, normalization_state: Mapping[str, object]
    ) -> None:
        if not isinstance(normalization_state, Mapping):
            raise ValueError("normalization state must be a mapping")
        self._normalization_state = normalization_state

    def set_policy_version(self, policy_version: int) -> None:
        if type(policy_version) is not int or policy_version < 0:
            raise ValueError("policy version must be a non-negative integer")
        self._policy_version = policy_version
        self.policy_versions.clear()
        self.planning_outcomes.clear()
        self.reason_codes.clear()
        self.candidate_diagnostics.clear()
        self.no_candidate_terminations.clear()
        self.terminal_audits.clear()
        self.raw_rewards.clear()
        self.success_first_crossings.clear()
        self.worker_wait_seconds = 0.0

    @property
    def current_observations(self) -> PolicyBatch:
        if self._current is None:
            raise ValueError("parallel rollout adapter is not initialized")
        return self._current.observations

    def advance_policy_version(self, policy_version: int) -> PolicyBatch:
        """Start a new on-policy batch without replacing active episodes."""
        if self._current is None:
            raise ValueError(
                "parallel rollout adapter must be initialized before policy advance"
            )
        self.set_policy_version(policy_version)
        return self._current.observations

    def reset(self) -> PolicyBatch:
        if self._current is None:
            self._current = self._pool.reset()
        return self._current.observations

    def step(
        self, candidate_indices: np.ndarray, thetas: np.ndarray
    ) -> EnvStep:
        started = time.perf_counter()
        try:
            step = self._pool.step(
                ParallelActions(
                    candidate_indices=torch.from_numpy(candidate_indices),
                    thetas=torch.from_numpy(thetas),
                ),
                policy_version=self._policy_version,
            )
        finally:
            self.worker_wait_seconds += time.perf_counter() - started
        self._current = step
        self.policy_versions.append(self._policy_version)
        self.planning_outcomes.extend(step.planning_outcomes)
        self.reason_codes.extend(step.reason_codes)
        self.candidate_diagnostics.extend(step.candidate_diagnostics)
        self.terminal_audits.extend(step.terminal_audits)
        self.raw_rewards.extend(float(value) for value in step.rewards.tolist())
        self.success_first_crossings.extend(
            bool(value) for value in step.success_first_crossings.tolist()
        )
        rewards = step.rewards.numpy().astype(np.float32, copy=True)
        if self._normalization_state is not None:
            mean = float(self._normalization_state["reward_mean"])
            variance = float(self._normalization_state["reward_var"])
            if not math.isfinite(mean) or not math.isfinite(variance) or variance <= 0.0:
                raise ValueError("live reward normalization state is invalid")
            rewards = np.asarray(
                (rewards - np.float32(mean))
                / np.float32(math.sqrt(variance)),
                dtype=np.float32,
            )
        return EnvStep(
            observations=step.observations,
            rewards=rewards,
            dones=step.dones.numpy().astype(np.bool_, copy=True),
        )

    def prepare_decision_boundaries(self) -> EnvStep:
        """Refresh producers and resolve no-action rows before policy forward."""
        started = time.perf_counter()
        try:
            step = self._pool.prepare_decision_boundaries(
                policy_version=self._policy_version,
            )
        finally:
            self.worker_wait_seconds += time.perf_counter() - started
        self._current = step
        self.no_candidate_terminations.extend(
            step.no_candidate_terminations
        )
        self.terminal_audits.extend(step.terminal_audits)
        return EnvStep(
            observations=step.observations,
            rewards=np.zeros((self.env_count,), dtype=np.float32),
            dones=step.dones.numpy().astype(np.bool_, copy=True),
        )

class ResumablePPOTrainer:
    """Task 1 PPO core plus an injectable Task 4-compatible reward boundary."""

    def __init__(
        self,
        policy: CrossAttentionPolicy,
        *,
        reward_fn: Callable[[PlannerTransition], float],
        ppo_config: PPOConfig,
        device: torch.device | str,
    ) -> None:
        if not callable(reward_fn):
            raise ValueError("reward function must be callable")
        self._reward_fn = reward_fn
        if not isinstance(ppo_config, PPOConfig):
            raise ValueError("resumable trainer requires typed PPO config")
        self._ppo = PPOTrainer(policy, config=ppo_config, device=device)
        self.normalization: dict[str, object] = {
            "reward_mean": 0.0,
            "reward_var": 1.0,
        }

    @property
    def policy(self) -> CrossAttentionPolicy:
        return self._ppo.policy

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self._ppo.optimizer

    def reward_transition(self, transition: PlannerTransition) -> float:
        if not isinstance(transition, PlannerTransition):
            raise ValueError("reward input must be PlannerTransition")
        reward = self._reward_fn(transition)
        if (
            not isinstance(reward, (int, float))
            or isinstance(reward, bool)
            or not math.isfinite(float(reward))
        ):
            raise ValueError("reward function must return a finite scalar")
        return float(reward)

    def update(
        self, rollout: RolloutBatch, *, micro_batch_size: int | None = None
    ):
        return self._ppo.update(
            rollout, micro_batch_size=micro_batch_size
        )


class TrainingBoundaryLoop:
    """Run complete optimizer updates and checkpoint only at their boundaries."""

    def __init__(
        self,
        *,
        budget: TrainingBudget,
        stop_flag: SignalStopFlag,
        checkpoint_interval_seconds: int,
        candidate_checkpoint_interval_seconds: int,
        curriculum_phase: str,
        phase_end_gpu_seconds: float | None = None,
        pause_after_gpu_seconds: float | None = None,
        initial_global_step: int = 0,
        initial_latest_checkpoint_gpu_seconds: float = 0.0,
        initial_candidate_checkpoint_gpu_seconds: float = 0.0,
        clock: Callable[[], float] = time.monotonic,
        synchronize_device: Callable[[], None] | None = None,
    ) -> None:
        if not isinstance(budget, TrainingBudget):
            raise ValueError("boundary loop requires TrainingBudget")
        if not isinstance(stop_flag, SignalStopFlag):
            raise ValueError("boundary loop requires SignalStopFlag")
        if type(checkpoint_interval_seconds) is not int or checkpoint_interval_seconds != 1800:
            raise ValueError("latest checkpoint interval must be 1800 seconds")
        if (
            type(candidate_checkpoint_interval_seconds) is not int
            or candidate_checkpoint_interval_seconds != 3600
        ):
            raise ValueError("candidate checkpoint interval must be 3600 seconds")
        if not isinstance(curriculum_phase, str) or not curriculum_phase:
            raise ValueError("curriculum phase must be non-empty")
        if type(initial_global_step) is not int or initial_global_step < 0:
            raise ValueError("initial global step must be non-negative")
        if phase_end_gpu_seconds is not None and (
            not isinstance(phase_end_gpu_seconds, (int, float))
            or isinstance(phase_end_gpu_seconds, bool)
            or not math.isfinite(float(phase_end_gpu_seconds))
            or not budget.consumed_gpu_seconds
            <= float(phase_end_gpu_seconds)
            <= budget.total_gpu_seconds
        ):
            raise ValueError("curriculum phase end GPU seconds are invalid")
        for name, marker in (
            ("latest", initial_latest_checkpoint_gpu_seconds),
            ("candidate", initial_candidate_checkpoint_gpu_seconds),
        ):
            if (
                not isinstance(marker, (int, float))
                or isinstance(marker, bool)
                or not math.isfinite(float(marker))
                or marker < 0.0
                or marker > budget.consumed_gpu_seconds
            ):
                raise ValueError(f"initial {name} checkpoint marker is invalid")
        if not callable(clock):
            raise ValueError("boundary loop clock must be callable")
        if synchronize_device is not None and not callable(synchronize_device):
            raise ValueError("device synchronizer must be callable")
        self._budget = budget
        self._stop_flag = stop_flag
        self._checkpoint_interval_seconds = checkpoint_interval_seconds
        self._candidate_checkpoint_interval_seconds = (
            candidate_checkpoint_interval_seconds
        )
        self._curriculum_phase = curriculum_phase
        self._phase_end_gpu_seconds = (
            None
            if phase_end_gpu_seconds is None
            else float(phase_end_gpu_seconds)
        )
        if pause_after_gpu_seconds is not None and (
            curriculum_phase != "joint"
            or not isinstance(pause_after_gpu_seconds, (int, float))
            or isinstance(pause_after_gpu_seconds, bool)
            or not math.isfinite(float(pause_after_gpu_seconds))
            or not 0.0
            <= float(pause_after_gpu_seconds)
            <= budget.total_gpu_seconds
        ):
            raise ValueError("joint evaluation pause boundary is invalid")
        self._pause_after_gpu_seconds = (
            None
            if pause_after_gpu_seconds is None
            else float(pause_after_gpu_seconds)
        )
        self._global_step = initial_global_step
        self._latest_checkpoint_gpu_seconds = float(
            initial_latest_checkpoint_gpu_seconds
        )
        self._candidate_checkpoint_gpu_seconds = float(
            initial_candidate_checkpoint_gpu_seconds
        )
        self._clock = clock
        self._synchronize_device = synchronize_device or (lambda: None)

    def run(
        self,
        *,
        collect_rollout: Callable[[], _Rollout],
        update_rollout: Callable[[_Rollout], object],
        save_checkpoint: Callable[[str, TrainingLoopState], None],
        max_updates: int,
        record_update: (
            Callable[[_Rollout, object, TrainingLoopState], None] | None
        ) = None,
    ) -> TrainingLoopState:
        if not all(
            callable(value)
            for value in (collect_rollout, update_rollout, save_checkpoint)
        ):
            raise ValueError("boundary loop callbacks must be callable")
        if type(max_updates) is not int or max_updates <= 0:
            raise ValueError("max updates must be a positive integer")
        if record_update is not None and not callable(record_update):
            raise ValueError("update recorder must be callable")
        updates = 0
        while updates < max_updates:
            if (
                self._pause_after_gpu_seconds is not None
                and self._budget.consumed_gpu_seconds
                >= self._pause_after_gpu_seconds
            ):
                self._latest_checkpoint_gpu_seconds = (
                    self._budget.consumed_gpu_seconds
                )
                state = self._state(rollout_discarded=False)
                save_checkpoint("latest", state)
                return state
            if (
                self._phase_end_gpu_seconds is not None
                and self._phase_end_gpu_seconds
                - self._budget.consumed_gpu_seconds
                <= TRAINING_ROLLOUT_UPDATE_UPPER_BOUND_GPU_SECONDS
            ):
                self._latest_checkpoint_gpu_seconds = (
                    self._budget.consumed_gpu_seconds
                )
                state = self._state(rollout_discarded=False)
                save_checkpoint("latest", state)
                return state
            interval_start = self._clock()
            try:
                self._budget.begin_gpu_interval(
                    monotonic_seconds=interval_start,
                    upper_bound_gpu_seconds=(
                        TRAINING_ROLLOUT_UPDATE_UPPER_BOUND_GPU_SECONDS
                    ),
                )
            except BudgetExceededError:
                self._latest_checkpoint_gpu_seconds = (
                    self._budget.consumed_gpu_seconds
                )
                state = self._state(rollout_discarded=False)
                save_checkpoint("latest", state)
                return state
            collection_error: BaseException | None = None
            update_error: BaseException | None = None
            budget_error: BudgetExceededError | None = None
            rollout = None
            update_result = None
            update_completed = False
            try:
                rollout = collect_rollout()
                self._synchronize_device()
            except BaseException as error:
                collection_error = error
            if collection_error is None and not self._stop_flag.requested:
                try:
                    update_result = update_rollout(rollout)
                    self._synchronize_device()
                    update_completed = True
                except BaseException as error:
                    update_error = error
            try:
                interval_end = self._clock()
                self._budget.end_gpu_interval(monotonic_seconds=interval_end)
            except BudgetExceededError as error:
                budget_error = error
            if collection_error is not None:
                raise collection_error
            if self._stop_flag.requested and not update_completed:
                self._latest_checkpoint_gpu_seconds = (
                    self._budget.consumed_gpu_seconds
                )
                state = self._state(rollout_discarded=True)
                save_checkpoint("latest", state)
                return state
            if update_error is not None:
                raise update_error
            if update_completed:
                self._global_step += 1
                updates += 1
                if record_update is not None:
                    record_update(
                        rollout,
                        update_result,
                        self._state(rollout_discarded=False),
                    )
            if budget_error is not None:
                self._latest_checkpoint_gpu_seconds = (
                    self._budget.consumed_gpu_seconds
                )
                state = self._state(rollout_discarded=False)
                save_checkpoint("latest", state)
                return state
            now = self._budget.consumed_gpu_seconds
            latest_saved = False
            if (
                now - self._latest_checkpoint_gpu_seconds
                >= self._checkpoint_interval_seconds
            ):
                self._latest_checkpoint_gpu_seconds = now
                save_checkpoint("latest", self._state(rollout_discarded=False))
                latest_saved = True
            if (
                self._curriculum_phase == "joint"
                and now - self._candidate_checkpoint_gpu_seconds
                >= self._candidate_checkpoint_interval_seconds
            ):
                self._candidate_checkpoint_gpu_seconds = now
                save_checkpoint("candidate", self._state(rollout_discarded=False))
            if self._stop_flag.requested:
                state = self._state(rollout_discarded=False)
                if not latest_saved:
                    self._latest_checkpoint_gpu_seconds = now
                    state = self._state(rollout_discarded=False)
                    save_checkpoint("latest", state)
                return state
        self._latest_checkpoint_gpu_seconds = self._budget.consumed_gpu_seconds
        state = self._state(rollout_discarded=False)
        save_checkpoint("latest", state)
        return state

    def _state(self, *, rollout_discarded: bool) -> TrainingLoopState:
        return TrainingLoopState(
            global_step=self._global_step,
            rollout_discarded=rollout_discarded,
            stop_signal=self._stop_flag.signal_number,
            latest_checkpoint_gpu_seconds=self._latest_checkpoint_gpu_seconds,
            candidate_checkpoint_gpu_seconds=(
                self._candidate_checkpoint_gpu_seconds
            ),
        )


def validate_artifact_root(
    path: str | Path, *, repository_root: str | Path
) -> Path:
    """Return a normalized outside-repository absolute artifact root."""
    target = Path(path)
    repository = Path(repository_root)
    if not target.is_absolute():
        raise ArtifactRootError("artifact root must be an explicit absolute path")
    if not repository.is_absolute():
        raise ArtifactRootError("repository root must be absolute")
    lexical_target = Path(os.path.abspath(target))
    resolved_target = target.resolve(strict=False)
    lexical_repository = Path(os.path.abspath(repository))
    resolved_repository = repository.resolve(strict=True)
    if _is_within(lexical_target, lexical_repository) or _is_within(
        resolved_target, resolved_repository
    ):
        raise ArtifactRootError("artifact root must remain outside the repository")
    return resolved_target


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _freeze_formal_environment_manifest(
    path: Path,
    *,
    cache_manifest_path: Path,
    cache_manifest_sha256: str,
    scenario_schedule_id: str,
    sensor_performance_sha256: str,
) -> None:
    resolved_cache = cache_manifest_path.resolve(strict=True)
    digests = (cache_manifest_sha256, sensor_performance_sha256)
    if any(
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in digests
    ):
        raise ArtifactRootError("formal environment digest is invalid")
    if not isinstance(scenario_schedule_id, str) or not scenario_schedule_id:
        raise ArtifactRootError("formal scenario schedule identity is invalid")
    frozen = {
        "schema_version": "lunar-formal-run-environment/v1",
        "cache_manifest_path": str(resolved_cache),
        "cache_manifest_sha256": cache_manifest_sha256,
        "scenario_schedule_id": scenario_schedule_id,
        "sensor_performance_sha256": sensor_performance_sha256,
    }
    payload = _read_run_manifest(path)
    existing = payload.get("formal_environment")
    if existing is not None and existing != frozen:
        raise ArtifactRootError("formal environment identity cannot drift")
    payload["formal_environment"] = frozen
    _write_manifest_payload(path, payload)


def _freeze_task4_manifest(
    path: Path,
    *,
    schedule: CurriculumSchedule,
    reward_hash: str,
    reward_seed_results: tuple[dict[str, object], ...],
    proxy: bool = True,
    scenario_schedule_id: str | None = None,
) -> None:
    """Freeze reward, scenario schedule, curriculum and the formal seed."""
    if not isinstance(schedule, CurriculumSchedule):
        raise ArtifactRootError("Task 4 curriculum schedule is invalid")
    if len(reward_hash) != 64 or any(
        character not in "0123456789abcdef" for character in reward_hash
    ):
        raise ArtifactRootError("Task 4 reward hash is invalid")
    if (
        len(reward_seed_results) != 3
        or tuple(result.get("seed") for result in reward_seed_results)
        != REWARD_CALIBRATION_SEEDS
    ):
        raise ArtifactRootError("three reward calibration seed results are required")
    if type(proxy) is not bool:
        raise ArtifactRootError("calibration proxy flag must be boolean")
    if proxy:
        resolved_schedule_id = schedule.scenario_schedule_id
        if scenario_schedule_id not in (None, resolved_schedule_id):
            raise ArtifactRootError("proxy calibration schedule identity differs")
        for result in reward_seed_results:
            score = result.get("minimum_platform_score")
            digest = result.get("report_sha256")
            if (
                not isinstance(score, (int, float))
                or isinstance(score, bool)
                or not math.isfinite(float(score))
                or not 0.0 <= float(score) <= 1.0
                or not isinstance(digest, str)
                or len(digest) != 64
            ):
                raise ArtifactRootError("reward calibration seed result is invalid")
    else:
        resolved_schedule_id = scenario_schedule_id
        if not isinstance(resolved_schedule_id, str) or not resolved_schedule_id:
            raise ArtifactRootError("formal calibration schedule identity is missing")
        for result in reward_seed_results:
            rewards = result.get("platform_mean_rewards")
            digest = result.get("rollout_sha256")
            if (
                not isinstance(rewards, dict)
                or set(rewards) != {"WHEELED", "LEGGED", "HOPPER"}
                or any(
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not math.isfinite(float(value))
                    for value in rewards.values()
                )
                or not isinstance(digest, str)
                or len(digest) != 64
            ):
                raise ArtifactRootError("formal reward calibration result is invalid")
    payload = _read_run_manifest(path)
    calibration_end = payload.get("consumed_gpu_seconds")
    if (
        not isinstance(calibration_end, (int, float))
        or isinstance(calibration_end, bool)
        or not math.isfinite(float(calibration_end))
        or float(calibration_end) < 0.0
    ):
        raise ArtifactRootError("calibration GPU budget state is invalid")
    frozen = {
        "schema_version": "lunar-policy-calibration/v2",
        "proxy": proxy,
        "reward_hash": reward_hash,
        "reward_selection": "single-approved-shared-weight-set/v1",
        "reward_calibration_seeds": list(REWARD_CALIBRATION_SEEDS),
        "reward_seed_results": [dict(result) for result in reward_seed_results],
        "scenario_schedule_id": resolved_schedule_id,
        "formal_seed": FORMAL_SEED,
        "calibration_end_gpu_seconds": float(calibration_end),
        "curriculum": {
            "calibration_limit_s": schedule.calibration_limit_s,
            "platform_warmup_order": list(schedule.platform_warmup_order),
            "platform_warmup_limit_s": schedule.platform_warmup_limit_s,
            "joint_minimum_s": schedule.joint_minimum_s,
            "joint_worker_allocation": schedule.joint_worker_allocation,
            "evaluation_interval_s": schedule.evaluation_interval_s,
            "total_gpu_limit_s": schedule.total_gpu_limit_s,
        },
    }
    existing = payload.get("task4_calibration")
    if existing is not None and existing != frozen:
        raise ArtifactRootError("Task 4 calibrated run identity cannot drift")
    payload["task4_calibration"] = frozen
    _write_manifest_payload(path, payload)


def _load_calibrated_run_state(root: Path) -> CalibratedRunState:
    manifest = _read_run_manifest(root / "run-manifest.json")
    frozen_config = manifest.get("frozen_config")
    runtime = manifest.get("runtime_calibration")
    task4 = manifest.get("task4_calibration")
    try:
        run_identity = RunIdentity.from_mapping(manifest.get("run_identity"))
    except Exception as error:
        raise ArtifactRootError("run manifest identity is missing or invalid") from error
    if not isinstance(frozen_config, dict) or not isinstance(runtime, dict):
        raise ArtifactRootError("calibrated runtime state is incomplete")
    if not isinstance(task4, dict):
        raise ArtifactRootError("Task 4 calibration is missing; run calibrate first")
    schedule = CurriculumSchedule()
    config = resolve_training_config(frozen_config)
    if config.run_kind != run_identity.run_kind:
        raise ArtifactRootError("run manifest kind differs from frozen config")
    formal_environment = manifest.get("formal_environment")
    cache_manifest_path: Path | None = None
    cache_manifest_sha256: str | None = None
    sensor_performance_sha256: str | None = None
    if config.run_kind == "formal":
        required_environment = {
            "schema_version",
            "cache_manifest_path",
            "cache_manifest_sha256",
            "scenario_schedule_id",
            "sensor_performance_sha256",
        }
        if (
            not isinstance(formal_environment, dict)
            or set(formal_environment) != required_environment
            or formal_environment.get("schema_version")
            != "lunar-formal-run-environment/v1"
        ):
            raise ArtifactRootError("formal run environment identity is invalid")
        cache_path_value = formal_environment["cache_manifest_path"]
        scenario_value = formal_environment["scenario_schedule_id"]
        if (
            not isinstance(cache_path_value, str)
            or not cache_path_value
            or not isinstance(scenario_value, str)
            or not scenario_value
        ):
            raise ArtifactRootError("formal run environment identity is invalid")
        cache_manifest_path = Path(cache_path_value)
        cache_manifest_sha256 = formal_environment["cache_manifest_sha256"]
        sensor_performance_sha256 = formal_environment[
            "sensor_performance_sha256"
        ]
        if (
            not cache_manifest_path.is_absolute()
            or not isinstance(cache_manifest_sha256, str)
            or len(cache_manifest_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in cache_manifest_sha256
            )
            or not isinstance(sensor_performance_sha256, str)
            or len(sensor_performance_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in sensor_performance_sha256
            )
        ):
            raise ArtifactRootError("formal run environment identity is invalid")
        expected_proxy = False
        expected_schedule_id = scenario_value
    else:
        if formal_environment is not None:
            raise ArtifactRootError("development calibration cannot bind formal cache")
        expected_proxy = True
        expected_schedule_id = schedule.scenario_schedule_id
    expected_static = {
        "schema_version": "lunar-policy-calibration/v2",
        "proxy": expected_proxy,
        "reward_hash": reward_weights_sha256(),
        "reward_selection": "single-approved-shared-weight-set/v1",
        "reward_calibration_seeds": list(REWARD_CALIBRATION_SEEDS),
        "scenario_schedule_id": expected_schedule_id,
        "formal_seed": FORMAL_SEED,
        "curriculum": {
            "calibration_limit_s": schedule.calibration_limit_s,
            "platform_warmup_order": list(schedule.platform_warmup_order),
            "platform_warmup_limit_s": schedule.platform_warmup_limit_s,
            "joint_minimum_s": schedule.joint_minimum_s,
            "joint_worker_allocation": schedule.joint_worker_allocation,
            "evaluation_interval_s": schedule.evaluation_interval_s,
            "total_gpu_limit_s": schedule.total_gpu_limit_s,
        },
    }
    for name, value in expected_static.items():
        if task4.get(name) != value:
            raise ArtifactRootError("Task 4 calibrated run identity is invalid")
    reward_results = task4.get("reward_seed_results")
    if (
        not isinstance(reward_results, list)
        or tuple(result.get("seed") for result in reward_results if isinstance(result, dict))
        != REWARD_CALIBRATION_SEEDS
    ):
        raise ArtifactRootError("Task 4 calibrated run identity is invalid")
    calibration_end = task4.get("calibration_end_gpu_seconds")
    if (
        not isinstance(calibration_end, (int, float))
        or isinstance(calibration_end, bool)
        or not math.isfinite(float(calibration_end))
        or not 0.0 <= float(calibration_end) <= schedule.calibration_limit_s
    ):
        raise ArtifactRootError("Task 4 calibration end is invalid")
    selected_workers = runtime.get("selected_workers")
    micro_batch_size = runtime.get("selected_micro_batch")
    selected_rollout_horizon = runtime.get("selected_rollout_horizon")
    if (
        type(selected_workers) is not int
        or type(micro_batch_size) is not int
        or type(selected_rollout_horizon) is not int
        or selected_rollout_horizon != config.ppo.rollout_horizon
    ):
        raise ArtifactRootError("runtime calibration selection is invalid")
    horizon_candidates = runtime.get("rollout_horizon_candidates")
    horizon_work = runtime.get("horizon_transitions_per_worker")
    horizon_measurements = runtime.get("horizon_measurements")
    if config.run_kind == "formal":
        if (
            horizon_candidates != list(ROLLOUT_HORIZON_CANDIDATES)
            or horizon_work != ROLLOUT_HORIZON_TRANSITIONS_PER_WORKER
            or not isinstance(horizon_measurements, list)
            or len(horizon_measurements) != len(ROLLOUT_HORIZON_CANDIDATES)
        ):
            raise ArtifactRootError(
                "formal rollout horizon calibration evidence is incomplete"
            )
        try:
            parsed_horizons = tuple(
                HorizonCalibrationMeasurement(**value)
                for value in horizon_measurements
                if isinstance(value, dict)
            )
        except Exception as error:
            raise ArtifactRootError(
                "formal rollout horizon measurement is invalid"
            ) from error
        if len(parsed_horizons) != len(ROLLOUT_HORIZON_CANDIDATES):
            raise ArtifactRootError(
                "formal rollout horizon measurement is invalid"
            )
        expected_total = (
            selected_workers * ROLLOUT_HORIZON_TRANSITIONS_PER_WORKER
        )
        if any(
            measurement.workers != selected_workers
            or measurement.micro_batch != micro_batch_size
            or measurement.rollout_horizon != candidate
            or measurement.total_transitions != expected_total
            or measurement.completed_updates
            > ROLLOUT_HORIZON_TRANSITIONS_PER_WORKER // candidate
            for candidate, measurement in zip(
                ROLLOUT_HORIZON_CANDIDATES,
                parsed_horizons,
            )
        ):
            raise ArtifactRootError(
                "formal rollout horizon probes did not use equal work"
            )
        qualified = tuple(
            measurement
            for measurement in parsed_horizons
            if not measurement.oom
            and measurement.ipc_failures == 0
            and measurement.planner_timeouts == 0
            and measurement.peak_gpu_memory_fraction
            <= config.parallel.gpu_memory_fraction_max
            and measurement.advantages_finite
            and measurement.returns_finite
            and measurement.update_boundary_continuity_verified
            and measurement.resume_verified
            and measurement.throughput_transitions_per_second > 0.0
            and measurement.completed_updates
            == ROLLOUT_HORIZON_TRANSITIONS_PER_WORKER
            // measurement.rollout_horizon
        )
        if not qualified:
            raise ArtifactRootError(
                "formal rollout horizon calibration has no qualified candidate"
            )
        expected_selection = select_qualified_rollout_horizon(qualified)
        if selected_rollout_horizon != expected_selection:
            raise ArtifactRootError(
                "formal rollout horizon selection differs from evidence"
            )
    elif (
        horizon_candidates != []
        or horizon_work != 0
        or horizon_measurements != []
    ):
        raise ArtifactRootError(
            "development calibration cannot contain formal horizon evidence"
        )
    consumed = manifest.get("consumed_gpu_seconds")
    budget = TrainingBudget(
        total_gpu_seconds=manifest.get("total_gpu_budget_seconds"),
        consumed_gpu_seconds=consumed,
        budget_extension_blocks=manifest.get("budget_extension_blocks"),
    )
    return CalibratedRunState(
        config=config,
        budget=budget,
        allocation=_allocation_for_workers(selected_workers),
        selected_workers=selected_workers,
        micro_batch_size=micro_batch_size,
        rollout_horizon=selected_rollout_horizon,
        reward_hash=task4["reward_hash"],
        scenario_schedule_id=task4["scenario_schedule_id"],
        formal_seed=task4["formal_seed"],
        reward_calibration_seeds=tuple(task4["reward_calibration_seeds"]),
        calibration_end_gpu_seconds=float(calibration_end),
        run_identity=run_identity,
        cache_manifest_path=cache_manifest_path,
        cache_manifest_sha256=cache_manifest_sha256,
        sensor_performance_sha256=sensor_performance_sha256,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lunar-policy-training")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_data = subparsers.add_parser("prepare-data")
    prepare_data.add_argument("--source-lock", required=True)
    prepare_data.add_argument("--split-manifest", required=True)
    prepare_data.add_argument("--cache-root", required=True)
    prepare_data.add_argument(
        "--materialization",
        required=True,
        choices=("preflight", "full"),
    )
    prepare_data.add_argument(
        "--preflight-scenario-limit",
        type=_positive_scenario_count,
    )

    calibrate = subparsers.add_parser("calibrate")
    calibrate.add_argument("--config", required=True)
    calibrate.add_argument("--artifact-root", required=True)
    calibrate.add_argument("--cache-manifest", required=True)
    calibrate.add_argument("--sensor-performance-report", required=True)

    train = subparsers.add_parser("train")
    train.add_argument("--config", required=True)
    train.add_argument("--artifact-root", required=True)
    train.add_argument("--sensor-performance-report")

    resume = subparsers.add_parser("resume")
    resume.add_argument("--artifact-root", required=True)
    resume.add_argument("--checkpoint", required=True)
    resume.add_argument("--sensor-performance-report")

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--gate", required=True)
    evaluate.add_argument("--artifact-root", required=True)
    evaluate.add_argument("--sensor-performance-report")

    formal_preflight = subparsers.add_parser("formal-preflight")
    formal_preflight.add_argument("--config", required=True)
    formal_preflight.add_argument("--cache-manifest", required=True)
    formal_preflight.add_argument("--artifact-root", required=True)
    formal_preflight.add_argument("--calibration-root", required=True)
    formal_preflight.add_argument("--sensor-performance-report", required=True)

    extend_budget = subparsers.add_parser("extend-budget")
    extend_budget.add_argument("--artifact-root", required=True)
    extend_budget.add_argument("--blocks", required=True, type=_positive_block_count)
    return parser


def _positive_block_count(value: str) -> int:
    try:
        blocks = int(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "budget extension blocks must be a positive integer"
        ) from error
    if str(blocks) != value or blocks <= 0:
        raise argparse.ArgumentTypeError(
            "budget extension blocks must be a positive integer"
        )
    return blocks


def _positive_scenario_count(value: str) -> int:
    try:
        count = int(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "preflight scenario limit must be a positive integer"
        ) from error
    if str(count) != value or count <= 0:
        raise argparse.ArgumentTypeError(
            "preflight scenario limit must be a positive integer"
        )
    return count


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    repository_root = Path(__file__).resolve().parents[3]
    if arguments.command == "prepare-data":
        if (
            arguments.materialization == "full"
            and arguments.preflight_scenario_limit is not None
        ):
            raise PreflightError("full materialization rejects a scenario limit")
        if (
            arguments.materialization == "preflight"
            and arguments.preflight_scenario_limit is None
        ):
            raise PreflightError(
                "preflight materialization requires --preflight-scenario-limit"
            )
        capability_bundle = _formal_capability_preflight(repository_root)
        try:
            manifest = prepare_formal_training_cache(
                source_lock_path=Path(arguments.source_lock),
                split_manifest_path=Path(arguments.split_manifest),
                cache_root=Path(arguments.cache_root),
                materialization=arguments.materialization,
                preflight_scenario_limit=arguments.preflight_scenario_limit,
                capability_bundle=capability_bundle,
                repository_root=repository_root,
            )
        except FormalCacheError as error:
            raise PreflightError(f"formal cache preparation failed: {error}") from error
        print(
            json.dumps(
                {
                    "cache_manifest": str(
                        Path(arguments.cache_root) / "cache-manifest.json"
                    ),
                    "cache_manifest_sha256": manifest["cache_manifest_sha256"],
                    "materialization": manifest["materialization"],
                    "scene_count": manifest["scene_count"],
                },
                sort_keys=True,
            )
        )
    elif arguments.command == "calibrate":
        requested_config = load_training_config(Path(arguments.config))
        if requested_config.run_kind != "formal":
            raise PreflightError("public calibrate requires the formal config")
        capability_bundle = _formal_capability_preflight(repository_root)
        sensor_report_sha256 = _formal_sensor_performance_preflight(
            arguments.sensor_performance_report,
            capability_bundle=capability_bundle,
            repository_root=repository_root,
        )
        formal_cache, formal_assembly = _build_formal_environment(
            Path(arguments.cache_manifest),
            capability_bundle=capability_bundle,
            repository_root=repository_root,
            split="train",
        )
        _calibrate_training_run(
            config_path=Path(arguments.config),
            artifact_root=Path(arguments.artifact_root),
            repository_root=repository_root,
            capability_bundle=capability_bundle,
            formal_cache=formal_cache,
            formal_environment_assembly=formal_assembly,
            cache_manifest_path=Path(arguments.cache_manifest),
            sensor_performance_sha256=sensor_report_sha256,
        )
    elif arguments.command == "train":
        requested_config = load_training_config(Path(arguments.config))
        if requested_config.run_kind != "formal":
            raise PreflightError(
                "public train is formal; use the development-smoke helper"
            )
        capability_bundle = _formal_capability_preflight(repository_root)
        sensor_performance_sha256 = _formal_sensor_performance_preflight(
            arguments.sensor_performance_report,
            capability_bundle=capability_bundle,
            repository_root=repository_root,
        )
        if not Path(arguments.artifact_root).is_dir():
            raise ArtifactRootError("run calibrate before public train")
        formal_assembly = _formal_environment_from_calibrated_root(
            Path(arguments.artifact_root),
            capability_bundle=capability_bundle,
            repository_root=repository_root,
            sensor_performance_sha256=sensor_performance_sha256,
            split="train",
        )
        formal_batches = _formal_evaluation_batches_from_calibrated_root(
            Path(arguments.artifact_root),
            capability_bundle=capability_bundle,
            repository_root=repository_root,
            sensor_performance_sha256=sensor_performance_sha256,
        )
        _start_training_run(
            config_path=Path(arguments.config),
            artifact_root=Path(arguments.artifact_root),
            repository_root=repository_root,
            max_updates=None,
            interrupt_first_update=False,
            capability_bundle=capability_bundle,
            formal_environment_factory=formal_assembly.factory,
            formal_observation_template=formal_assembly.observation_template,
            formal_evaluation_batches=formal_batches,
            formal_gate_path=(
                repository_root / "training/configs/candidate_gate_v1.yaml"
            ),
        )
    elif arguments.command == "resume":
        capability_bundle = _formal_capability_preflight(repository_root)
        sensor_performance_sha256 = _formal_sensor_performance_preflight(
            arguments.sensor_performance_report,
            capability_bundle=capability_bundle,
            repository_root=repository_root,
        )
        formal_assembly = _formal_environment_from_calibrated_root(
            Path(arguments.artifact_root),
            capability_bundle=capability_bundle,
            repository_root=repository_root,
            sensor_performance_sha256=sensor_performance_sha256,
            split="train",
        )
        formal_batches = _formal_evaluation_batches_from_calibrated_root(
            Path(arguments.artifact_root),
            capability_bundle=capability_bundle,
            repository_root=repository_root,
            sensor_performance_sha256=sensor_performance_sha256,
        )
        _resume_training_run(
            artifact_root=Path(arguments.artifact_root),
            checkpoint_path=Path(arguments.checkpoint),
            repository_root=repository_root,
            max_updates=None,
            capability_bundle=capability_bundle,
            formal_environment_factory=formal_assembly.factory,
            formal_observation_template=formal_assembly.observation_template,
            formal_evaluation_batches=formal_batches,
            formal_gate_path=(
                repository_root / "training/configs/candidate_gate_v1.yaml"
            ),
        )
    elif arguments.command == "evaluate":
        capability_bundle = _formal_capability_preflight(repository_root)
        sensor_performance_sha256 = _formal_sensor_performance_preflight(
            arguments.sensor_performance_report,
            capability_bundle=capability_bundle,
            repository_root=repository_root,
        )
        formal_batches = _formal_evaluation_batches_from_calibrated_root(
            Path(arguments.artifact_root),
            capability_bundle=capability_bundle,
            repository_root=repository_root,
            sensor_performance_sha256=sensor_performance_sha256,
        )
        _evaluate_checkpoint(
            checkpoint_path=Path(arguments.checkpoint),
            gate_path=Path(arguments.gate),
            artifact_root=Path(arguments.artifact_root),
            repository_root=repository_root,
            capability_bundle=capability_bundle,
            formal_batches=formal_batches,
        )
    elif arguments.command == "formal-preflight":
        requested_config = load_training_config(Path(arguments.config))
        if requested_config.run_kind != "formal":
            raise PreflightError("formal-preflight requires the formal config")
        capability_bundle = _formal_capability_preflight(repository_root)
        sensor_performance_sha256 = _formal_sensor_performance_preflight(
            arguments.sensor_performance_report,
            capability_bundle=capability_bundle,
            repository_root=repository_root,
        )
        cache, train_assembly = _build_formal_environment(
            Path(arguments.cache_manifest),
            capability_bundle=capability_bundle,
            repository_root=repository_root,
            split="train",
        )
        assemblies = {"train": train_assembly}
        for split in ("validation", "test", "holdout"):
            _, assemblies[split] = _build_formal_environment(
                Path(arguments.cache_manifest),
                capability_bundle=capability_bundle,
                repository_root=repository_root,
                split=split,
            )
        evaluation_batches = _formal_evaluation_batches(cache, assemblies)
        calibrated = _validated_formal_preflight_calibration(
            calibration_root=Path(arguments.calibration_root),
            requested_config=requested_config,
            cache=cache,
            train_assembly=train_assembly,
            cache_manifest_path=Path(arguments.cache_manifest),
            sensor_performance_sha256=sensor_performance_sha256,
            repository_root=repository_root,
        )
        preflight_root = validate_artifact_root(
            Path(arguments.artifact_root), repository_root=repository_root
        )
        if (preflight_root / "checkpoints").exists():
            raise PreflightError(
                "formal-preflight artifact root cannot contain checkpoints"
            )
        try:
            resume_equivalence = _formal_resume_equivalence_check(
                config=calibrated.config,
                assembly=train_assembly,
                run_identity=_formal_run_identity(cache.identity),
                source_commit=_source_commit(repository_root),
                artifact_root=preflight_root,
                selected_workers=calibrated.selected_workers,
                micro_batch_size=calibrated.micro_batch_size,
            )
            report, report_path = run_formal_preflight(
                cache=cache,
                assemblies=assemblies,
                evaluation_batches=evaluation_batches,
                run_identity=_formal_run_identity(cache.identity),
                source_commit=_source_commit(repository_root),
                sensor_performance_sha256=sensor_performance_sha256,
                artifact_root=preflight_root,
                worker_candidates=requested_config.parallel.worker_candidates,
                selected_workers=calibrated.selected_workers,
                selected_micro_batch=calibrated.micro_batch_size,
                selected_rollout_horizon=calibrated.rollout_horizon,
                resume_equivalence=resume_equivalence,
            )
        except FormalPreflightError as error:
            raise PreflightError(f"formal preflight failed: {error}") from error
        print(
            json.dumps(
                {
                    "formal_preflight_report": str(report_path),
                    "selected_workers": report.payload["selected_workers"],
                    "selected_micro_batch": report.payload[
                        "selected_micro_batch"
                    ],
                    "training_started": False,
                },
                sort_keys=True,
            )
        )
    elif arguments.command == "extend-budget":
        root = validate_artifact_root(
            Path(arguments.artifact_root), repository_root=repository_root
        )
        if not root.is_dir():
            raise ArtifactRootError("budget extension artifact root is missing")
        extend_budget_manifest(
            root / "run-manifest.json", blocks=arguments.blocks
        )
    return 0


def _formal_capability_preflight(
    repository_root: Path,
) -> FrozenCapabilityBundle:
    try:
        return load_project_formal_capability(repository_root)
    except CapabilityFreezeError as error:
        raise PreflightError(f"project formal capability is invalid: {error}") from error


_CACHE_SAFE_V3_RUNTIME_REPAIR_PATHS = frozenset(
    {
        "ros2_ws/src/lunar_planner_training_bridge/include/"
        "lunar_planner_training_bridge/visibility.hpp",
        "ros2_ws/src/lunar_planner_training_bridge/src/visibility.cpp",
        "ros2_ws/src/lunar_planner_training_bridge/test/"
        "visibility_benchmark.cpp",
    }
)
_V3_SOURCE_PREFIXES = (
    "ros2_ws/src/lunar_navigation_msgs/",
    "ros2_ws/src/lunar_planning_msgs/",
    "ros2_ws/src/lunar_planner_core/",
    "ros2_ws/src/lunar_planner_training_bridge/",
)


def _cache_accepts_runtime_only_v3_repair(
    repository_root: Path,
    *,
    cached_commit: str,
    current_commit: str,
) -> bool:
    """Allow an old static cache only for the audited visibility runtime repair."""
    if cached_commit == current_commit:
        return False
    try:
        if not _commit_is_ancestor(
            repository_root, cached_commit, current_commit
        ):
            return False
        changed_paths = _commit_changed_paths(
            repository_root, cached_commit, current_commit
        )
    except RuntimeError:
        return False
    v3_changes = {
        path
        for path in changed_paths
        if path.startswith(_V3_SOURCE_PREFIXES)
    }
    return bool(v3_changes) and v3_changes.issubset(
        _CACHE_SAFE_V3_RUNTIME_REPAIR_PATHS
    )


def _build_formal_environment(
    cache_manifest_path: Path,
    *,
    capability_bundle: FrozenCapabilityBundle,
    repository_root: Path,
    split: str,
) -> tuple[FormalCache, FormalEnvironmentAssembly]:
    """Validate every current identity before constructing one formal factory."""
    try:
        cache = load_formal_cache(cache_manifest_path, require_full=True)
        identity = cache.identity
        current_v3_commit, current_v3_sha256 = current_v3_identity(repository_root)
        expected = {
            "generator_sha256": GENERATOR_SHA256,
            "capability_sha256": capability_bundle.bundle_sha256,
            "reward_sha256": reward_weights_sha256(),
            "training_semantics_sha256": training_semantics_sha256(),
        }
        for name, value in expected.items():
            if getattr(identity, name) != value:
                raise FormalCacheError(
                    f"cache identity {name} differs from the current project"
                )
        v3_identity_matches = (
            identity.v3_source_commit == current_v3_commit
            and identity.v3_sha256 == current_v3_sha256
        )
        runtime_only_repair = (
            identity.v3_source_commit != current_v3_commit
            and identity.v3_sha256 != current_v3_sha256
            and _cache_accepts_runtime_only_v3_repair(
                repository_root,
                cached_commit=identity.v3_source_commit,
                current_commit=current_v3_commit,
            )
        )
        if not v3_identity_matches and not runtime_only_repair:
            raise FormalCacheError(
                "cache identity v3 differs from the current project"
            )
        assembly = FormalEnvironmentBuilder(
            cache_manifest_path=cache_manifest_path,
            capability_bundle=capability_bundle,
            split=split,
        ).build()
    except (FormalCacheError, ValueError) as error:
        raise PreflightError(f"formal environment is invalid: {error}") from error
    return cache, assembly


def _formal_environment_from_calibrated_root(
    artifact_root: Path,
    *,
    capability_bundle: FrozenCapabilityBundle,
    repository_root: Path,
    sensor_performance_sha256: str,
    split: str,
) -> FormalEnvironmentAssembly:
    root = validate_artifact_root(
        artifact_root, repository_root=repository_root
    )
    if not root.is_dir():
        raise ArtifactRootError("formal calibrated run root is missing")
    calibrated = _load_calibrated_run_state(root)
    if calibrated.run_identity.run_kind != "formal":
        raise PreflightError("formal command rejects a proxy calibrated root")
    if calibrated.sensor_performance_sha256 != sensor_performance_sha256:
        raise PreflightError("formal sensor performance identity differs from calibration")
    if calibrated.cache_manifest_path is None:
        raise PreflightError("formal cache manifest identity is missing")
    cache, assembly = _build_formal_environment(
        calibrated.cache_manifest_path,
        capability_bundle=capability_bundle,
        repository_root=repository_root,
        split=split,
    )
    if cache.manifest["cache_manifest_sha256"] != calibrated.cache_manifest_sha256:
        raise PreflightError("formal cache manifest differs from calibration")
    if _formal_run_identity(cache.identity) != calibrated.run_identity:
        raise PreflightError("formal cache run identity differs from calibration")
    if split == "train" and (
        assembly.scenario_schedule_id != calibrated.scenario_schedule_id
    ):
        raise PreflightError("formal training scenario schedule differs from calibration")
    return assembly


def _validated_formal_preflight_calibration(
    *,
    calibration_root: Path,
    requested_config: ResolvedTrainingConfig,
    cache: FormalCache,
    train_assembly: FormalEnvironmentAssembly,
    cache_manifest_path: Path,
    sensor_performance_sha256: str,
    repository_root: Path,
) -> CalibratedRunState:
    """Bind preflight evidence to the exact frozen calibration selection."""
    try:
        require_clean_sensor_source(repository_root)
    except SensorPerformanceError as error:
        raise PreflightError(
            f"formal preflight requires a clean source tree: {error}"
        ) from error
    root = validate_artifact_root(
        calibration_root, repository_root=repository_root
    )
    calibrated = _load_calibrated_run_state(root)
    manifest = _read_run_manifest(root / "run-manifest.json")
    expected_config = with_rollout_horizon(
        requested_config, calibrated.rollout_horizon
    )
    expected_cache_path = cache_manifest_path.resolve(strict=True)
    if calibrated.config != expected_config:
        raise PreflightError("formal preflight config differs from calibration")
    if (
        manifest.get("source_commit") != _source_commit(repository_root)
        or manifest.get("global_step") != 0
        or (root / "checkpoints").exists()
    ):
        raise PreflightError(
            "formal preflight requires current unstarted calibration evidence"
        )
    if (
        calibrated.run_identity != _formal_run_identity(cache.identity)
        or calibrated.cache_manifest_path is None
        or calibrated.cache_manifest_path.resolve(strict=True)
        != expected_cache_path
        or calibrated.cache_manifest_sha256
        != cache.manifest["cache_manifest_sha256"]
        or calibrated.sensor_performance_sha256
        != sensor_performance_sha256
        or calibrated.scenario_schedule_id
        != train_assembly.scenario_schedule_id
        or calibrated.selected_workers
        not in requested_config.parallel.worker_candidates
    ):
        raise PreflightError(
            "formal preflight environment differs from calibration"
        )
    return calibrated


def _formal_evaluation_batches(
    cache: FormalCache,
    assemblies: Mapping[str, FormalEnvironmentAssembly],
) -> tuple[FormalEvaluationBatch, ...]:
    """Bind cache-order scene seeds to the three non-training factories."""
    expected_splits = ("validation", "test", "holdout")
    assembly_splits = set(assemblies)
    non_training_splits = set(expected_splits)
    if assembly_splits not in (
        non_training_splits,
        non_training_splits | {"train"},
    ):
        raise PreflightError(
            "formal evaluation requires validation, test, and holdout assemblies"
        )
    try:
        document = json.loads(
            (cache.root / "scenario-manifest.json").read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PreflightError("formal evaluation scenario manifest is invalid") from error
    scenarios = document.get("scenarios") if isinstance(document, Mapping) else None
    if not isinstance(scenarios, list) or any(
        not isinstance(item, Mapping) for item in scenarios
    ):
        raise PreflightError("formal evaluation scenario inventory is invalid")
    by_id = {str(item.get("scene_id")): item for item in scenarios}
    if len(by_id) != len(scenarios):
        raise PreflightError("formal evaluation scenario IDs are not unique")
    batches: list[FormalEvaluationBatch] = []
    for split in expected_splits:
        common = cache.manifest.get("exact_common_evaluation", {}).get(
            "splits", {}
        ).get(split)
        if not isinstance(common, Mapping):
            raise PreflightError("formal exact-common evaluation inventory is invalid")
        common_ids = common.get("scene_ids")
        common_schedule_id = common.get("scenario_schedule_id")
        if not isinstance(common_ids, list) or not isinstance(
            common_schedule_id, str
        ):
            raise PreflightError("formal exact-common evaluation inventory is invalid")
        selected = set(common_ids)
        entries = _formal_scheduled_entries(
            [
                entry
                for entry in cache.manifest["scenes"]
                if entry.get("scene_id") in selected
            ],
            scenario_schedule_id=common_schedule_id,
        )
        seeds: list[int] = []
        for entry in entries:
            scenario = by_id.get(str(entry.get("scene_id")))
            if scenario is None or scenario.get("split") != split:
                raise PreflightError(
                    "formal evaluation cache order differs from scenario manifest"
                )
            scenario_seed = scenario.get("scenario_seed")
            if type(scenario_seed) is int:
                seeds.append(scenario_seed)
                continue
            scene_seed = scenario.get("scene_seed")
            if (
                split != "holdout"
                or not isinstance(scene_seed, str)
                or len(scene_seed) != 64
            ):
                raise PreflightError("formal evaluation scenario seed is invalid")
            try:
                seeds.append(int(scene_seed[:16], 16))
            except ValueError as error:
                raise PreflightError(
                    "formal evaluation holdout seed is invalid"
                ) from error
        if not seeds:
            raise PreflightError(f"formal evaluation split {split} is empty")
        assembly = assemblies[split]
        batches.append(
            FormalEvaluationBatch(
                split=split,
                factory=assembly.factory,
                observation_template=assembly.observation_template,
                scenario_seeds=tuple(seeds),
            )
        )
    return tuple(batches)


def _formal_evaluation_batches_from_calibrated_root(
    artifact_root: Path,
    *,
    capability_bundle: FrozenCapabilityBundle,
    repository_root: Path,
    sensor_performance_sha256: str,
) -> tuple[FormalEvaluationBatch, ...]:
    """Construct all frozen non-training batches for manual or periodic gates."""
    calibrated = _load_calibrated_run_state(artifact_root)
    if calibrated.cache_manifest_path is None:
        raise PreflightError("formal evaluation cache manifest is missing")
    assemblies = {
        split: _formal_environment_from_calibrated_root(
            artifact_root,
            capability_bundle=capability_bundle,
            repository_root=repository_root,
            sensor_performance_sha256=sensor_performance_sha256,
            split=split,
        )
        for split in ("validation", "test", "holdout")
    }
    formal_cache = load_formal_cache(
        calibrated.cache_manifest_path, require_full=True
    )
    return _formal_evaluation_batches(formal_cache, assemblies)


def _formal_sensor_performance_preflight(
    report_path: str | None,
    *,
    capability_bundle: FrozenCapabilityBundle,
    repository_root: str | Path,
) -> str:
    """Require host- and source-bound Release evidence before formal artifacts."""
    if report_path is None:
        raise PreflightError("formal sensor performance report is required")
    try:
        report = load_sensor_performance_report(Path(report_path))
        return validate_sensor_performance_report(
            report,
            expected_host=current_host_identity(),
            expected_source_commit=sensor_source_commit(repository_root),
            expected_capability_sha256=capability_bundle.bundle_sha256,
            expected_training_semantics_sha256=training_semantics_sha256(),
        )
    except SensorPerformanceError as error:
        raise PreflightError(
            f"formal sensor performance report is invalid: {error}"
        ) from error


def _validate_formal_bundle_identity(
    bundle: FrozenCapabilityBundle | None,
    run_identity: RunIdentity,
) -> None:
    if bundle is None or not bundle.formal_eligible:
        raise PreflightError("formal capability bundle is required")
    if run_identity.run_kind != "formal":
        raise PreflightError("formal capability bundle cannot enter development-smoke")
    if run_identity.capability_sha256 != bundle.bundle_sha256:
        raise PreflightError("formal capability bundle hash differs from run identity")


def run_cuda_interrupt_resume_smoke(
    *,
    config_path: str | Path,
    artifact_root: str | Path,
    repository_root: str | Path,
) -> CudaSmokeEvidence:
    """Run a short real-CUDA pause/resume without claiming formal training."""
    device_name = _validated_cuda_device()
    interrupted = _start_training_run(
        config_path=Path(config_path),
        artifact_root=Path(artifact_root),
        repository_root=Path(repository_root),
        max_updates=2,
        interrupt_first_update=True,
        capability_bundle=None,
    )
    resumed = _resume_training_run(
        artifact_root=Path(artifact_root),
        checkpoint_path=(Path(artifact_root) / "checkpoints/latest.pt"),
        repository_root=Path(repository_root),
        max_updates=1,
        capability_bundle=None,
    )
    return CudaSmokeEvidence(
        device_name=device_name,
        interrupted_global_step=interrupted.global_step,
        resumed_global_step=resumed.global_step,
        interrupted_consumed_gpu_seconds=interrupted.consumed_gpu_seconds,
        resumed_consumed_gpu_seconds=resumed.consumed_gpu_seconds,
        platform_allocation=resumed.platform_allocation,
        signal_observed_at_update_boundary=(
            interrupted.signal_observed_at_update_boundary
        ),
    )


def _calibrate_training_run(
    *,
    config_path: Path,
    artifact_root: Path,
    repository_root: Path,
    capability_bundle: FrozenCapabilityBundle | None = None,
    formal_cache: FormalCache | None = None,
    formal_environment_assembly: FormalEnvironmentAssembly | None = None,
    cache_manifest_path: Path | None = None,
    sensor_performance_sha256: str | None = None,
    calibration_micro_batch_candidates: tuple[int, ...] | None = None,
) -> CalibratedRunState:
    """Freeze runtime and reward calibration against one explicit environment."""
    config = load_training_config(config_path)
    source_commit = _source_commit(repository_root)
    formal = config.run_kind == "formal"
    if formal:
        if (
            not isinstance(capability_bundle, FrozenCapabilityBundle)
            or not isinstance(formal_cache, FormalCache)
            or not formal_cache.formal_eligible
            or not isinstance(
                formal_environment_assembly, FormalEnvironmentAssembly
            )
            or cache_manifest_path is None
            or not isinstance(sensor_performance_sha256, str)
            or len(sensor_performance_sha256) != 64
        ):
            raise PreflightError("formal calibration inputs are incomplete")
        if formal_environment_assembly.factory.bundle is not capability_bundle:
            raise PreflightError("formal calibration capability bundle is not shared")
        if (
            formal_environment_assembly.cache_manifest_sha256
            != formal_cache.manifest["cache_manifest_sha256"]
        ):
            raise PreflightError("formal calibration cache identity differs")
        run_identity = _formal_run_identity(formal_cache.identity)
    else:
        if any(
            value is not None
            for value in (
                capability_bundle,
                formal_cache,
                formal_environment_assembly,
                cache_manifest_path,
                sensor_performance_sha256,
            )
        ):
            raise PreflightError("development calibration rejects formal inputs")
        run_identity = _development_run_identity(source_commit)
    root = validate_artifact_root(
        artifact_root, repository_root=repository_root
    )
    if root.exists():
        raise ArtifactRootError("calibration artifact root already exists")
    if not root.parent.is_dir():
        raise ArtifactRootError("artifact root parent directory is missing")
    root.mkdir()
    budget = TrainingBudget(
        total_gpu_seconds=float(config.total_gpu_budget_seconds)
    )
    workload = _CudaPlannerCalibrationWorkload(
        config.ppo,
        environment_factory=(
            formal_environment_assembly.factory if formal else None
        ),
        observation_template=(
            formal_environment_assembly.observation_template if formal else None
        ),
    )
    try:
        calibration = calibrate_runtime(
            config=config,
            workload=workload,
            horizon_workload=(
                workload.measure_rollout_horizon if formal else None
            ),
            budget=budget,
            manifest_path=root / "run-manifest.json",
            micro_batch_candidates=(
                (1, 2, 4)
                if calibration_micro_batch_candidates is None
                else calibration_micro_batch_candidates
            ),
        )
    finally:
        workload.close()
    config = with_rollout_horizon(
        config,
        calibration.selected_rollout_horizon,
    )
    schedule = CurriculumSchedule()
    if formal:
        assert formal_environment_assembly is not None
        assert formal_cache is not None
        assert cache_manifest_path is not None
        assert sensor_performance_sha256 is not None
        reward_results = list(
            _formal_reward_seed_results(
                factory=formal_environment_assembly.factory,
                observation_template=formal_environment_assembly.observation_template,
                selected_workers=calibration.selected_workers,
                budget=budget,
            )
        )
        _freeze_formal_environment_manifest(
            root / "run-manifest.json",
            cache_manifest_path=cache_manifest_path,
            cache_manifest_sha256=str(
                formal_cache.manifest["cache_manifest_sha256"]
            ),
            scenario_schedule_id=formal_environment_assembly.scenario_schedule_id,
            sensor_performance_sha256=sensor_performance_sha256,
        )
    else:
        reward_results = []
        for seed in REWARD_CALIBRATION_SEEDS:
            _seed_everything(seed)
            start = time.monotonic()
            budget.begin_gpu_interval(
                monotonic_seconds=start,
                upper_bound_gpu_seconds=TRAINING_ROLLOUT_UPDATE_UPPER_BOUND_GPU_SECONDS,
            )
            try:
                report = evaluate_proxy_policy(
                    CrossAttentionPolicy(),
                    device="cuda",
                    checkpoint_sha256=hashlib.sha256(
                        f"reward-calibration-{seed}".encode("utf-8")
                    ).hexdigest(),
                    schedule=schedule,
                    run_identity=run_identity,
                )
                torch.cuda.synchronize()
            except BaseException:
                budget.end_gpu_interval(monotonic_seconds=time.monotonic())
                raise
            else:
                budget.end_gpu_interval(monotonic_seconds=time.monotonic())
            ppo = report.method("ppo_policy").per_platform
            reward_results.append(
                {
                    "seed": seed,
                    "minimum_platform_score": min(
                        metrics.success_coverage_rate for metrics in ppo.values()
                    ),
                    "report_sha256": report_sha256(report),
                }
            )
    allocation = _allocation_for_workers(calibration.selected_workers)
    _update_run_manifest(
        root / "run-manifest.json",
        source_commit=source_commit,
        config_hash=config_sha256(config.as_frozen_dict()),
        run_identity=run_identity,
        global_step=0,
        consumed_gpu_seconds=budget.consumed_gpu_seconds,
        platform_allocation=allocation,
    )
    _freeze_task4_manifest(
        root / "run-manifest.json",
        schedule=schedule,
        reward_hash=reward_weights_sha256(),
        reward_seed_results=tuple(reward_results),
        proxy=not formal,
        scenario_schedule_id=(
            formal_environment_assembly.scenario_schedule_id
            if formal_environment_assembly is not None
            else None
        ),
    )
    return _load_calibrated_run_state(root)


def _formal_reward_seed_results(
    *,
    factory: FrozenCapabilityEnvironmentFactory,
    observation_template: PolicyBatch,
    selected_workers: int,
    budget: TrainingBudget,
) -> tuple[dict[str, object], ...]:
    """Measure the approved reward on real formal transitions for three seeds."""
    allocation = _allocation_for_workers(selected_workers)
    results: list[dict[str, object]] = []
    for seed in REWARD_CALIBRATION_SEEDS:
        _seed_everything(seed)
        start = time.monotonic()
        budget.begin_gpu_interval(
            monotonic_seconds=start,
            upper_bound_gpu_seconds=TRAINING_ROLLOUT_UPDATE_UPPER_BOUND_GPU_SECONDS,
        )
        pool: ParallelEnvPool | None = None
        try:
            pool = ParallelEnvPool(
                allocation=allocation,
                observation_template=observation_template,
                environment_factory=factory,
                reward_fn=compute_transition_reward,
                worker_timeout_seconds=60.0,
            )
            environment = _ParallelPoolVectorEnv(pool, policy_version=seed)
            collected = collect_ppo_rollout(
                environment,
                CrossAttentionPolicy(),
                CollectorConfig(horizon=1, deterministic=True),
                device="cuda",
            )
            torch.cuda.synchronize()
            platform_rewards: dict[str, float] = {}
            offset = 0
            for platform, count in allocation.items():
                values = collected.rewards[:, offset : offset + count]
                platform_rewards[platform] = float(values.mean())
                offset += count
            evidence = {
                "seed": seed,
                "rewards": collected.rewards.tolist(),
                "dones": collected.dones.tolist(),
                "planning_outcomes": [
                    int(outcome) for outcome in environment.planning_outcomes
                ],
                "reason_codes": list(environment.reason_codes),
            }
            digest = hashlib.sha256(
                json.dumps(
                    evidence,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest()
            results.append(
                {
                    "seed": seed,
                    "platform_mean_rewards": platform_rewards,
                    "rollout_sha256": digest,
                }
            )
        finally:
            if pool is not None:
                pool.close()
            budget.end_gpu_interval(monotonic_seconds=time.monotonic())
    return tuple(results)


def _write_development_evaluation_artifacts(
    *,
    root: Path,
    checkpoint_path: Path,
    report: EvaluationReport,
    gate_result: GateResult,
) -> str:
    """Persist proxy metrics under names that cannot imply formal release."""
    if (
        not isinstance(report, EvaluationReport)
        or not report.proxy
        or report.run_identity.run_kind != "development-smoke"
        or not isinstance(gate_result, GateResult)
        or gate_result.passed
        or gate_result.formal_candidate_eligible
    ):
        raise PreflightError(
            "development evaluation artifacts require ineligible proxy result"
        )
    evaluation_dir = root / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    digest = write_report(evaluation_dir / "development-report.json", report)
    result_payload = {
        "schema_version": "lunar-policy-development-evaluation-result/v1",
        "report_sha256": digest,
        "run_kind": report.run_identity.run_kind,
        "proxy": report.proxy,
        "formal_candidate_eligible": False,
        "release_gate_passed": False,
        "failed_rules": list(gate_result.failed_rules),
    }
    (evaluation_dir / "development-result.json").write_text(
        json.dumps(result_payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest_path = root / "run-manifest.json"
    manifest = _read_run_manifest(manifest_path)
    manifest["last_development_evaluation"] = {
        "checkpoint": str(checkpoint_path),
        "report_sha256": digest,
        "run_kind": report.run_identity.run_kind,
        "proxy": report.proxy,
        "formal_candidate_eligible": False,
    }
    _write_manifest_payload(manifest_path, manifest)
    return digest


def _write_formal_evaluation_artifacts(
    *,
    root: Path,
    checkpoint_path: Path,
    report: EvaluationReport,
    gate_result: GateResult,
    started_gpu_seconds: float | None = None,
    completed_gpu_seconds: float | None = None,
) -> str:
    """Persist an explicitly non-proxy release evaluation and its gate result."""
    if (
        not isinstance(report, EvaluationReport)
        or report.proxy
        or report.run_identity.run_kind != "formal"
        or not isinstance(gate_result, GateResult)
        or gate_result.proxy
        or gate_result.run_kind != "formal"
        or not gate_result.formal_candidate_eligible
    ):
        raise PreflightError(
            "formal evaluation artifacts require an eligible non-proxy result"
        )
    if (started_gpu_seconds is None) != (completed_gpu_seconds is None):
        raise PreflightError("formal evaluation GPU interval is incomplete")
    if started_gpu_seconds is not None:
        if (
            not isinstance(started_gpu_seconds, (int, float))
            or isinstance(started_gpu_seconds, bool)
            or not math.isfinite(float(started_gpu_seconds))
            or not isinstance(completed_gpu_seconds, (int, float))
            or isinstance(completed_gpu_seconds, bool)
            or not math.isfinite(float(completed_gpu_seconds))
            or float(completed_gpu_seconds) < float(started_gpu_seconds)
        ):
            raise PreflightError("formal evaluation GPU interval is invalid")
    evaluation_dir = root / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    result_payload = {
        "schema_version": "lunar-policy-formal-evaluation-result/v1",
        "run_kind": gate_result.run_kind,
        "proxy": gate_result.proxy,
        "formal_candidate_eligible": gate_result.formal_candidate_eligible,
        "release_gate_passed": gate_result.passed,
        "failed_rules": list(gate_result.failed_rules),
    }
    if started_gpu_seconds is not None:
        result_payload.update(
            {
                "started_gpu_seconds": float(started_gpu_seconds),
                "completed_gpu_seconds": float(completed_gpu_seconds),
            }
        )
    candidate_name = checkpoint_path.stem
    if checkpoint_path.name.startswith("candidate-step-"):
        immutable_dir = evaluation_dir / candidate_name
        if immutable_dir.exists() or immutable_dir.is_symlink():
            raise PreflightError("formal candidate evaluation already exists")
        temporary_dir = Path(
            tempfile.mkdtemp(
                dir=evaluation_dir,
                prefix=f".{candidate_name}.",
            )
        )
        try:
            immutable_digest = write_report(
                temporary_dir / "formal-report.json", report
            )
            immutable_result = {
                **result_payload,
                "report_sha256": immutable_digest,
            }
            (temporary_dir / "formal-result.json").write_text(
                json.dumps(immutable_result, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            os.rename(temporary_dir, immutable_dir)
        except Exception:
            for child in temporary_dir.iterdir():
                child.unlink()
            temporary_dir.rmdir()
            raise
    digest = write_report(evaluation_dir / "formal-report.json", report)
    result_payload["report_sha256"] = digest
    (evaluation_dir / "formal-result.json").write_text(
        json.dumps(result_payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest_path = root / "run-manifest.json"
    manifest = _read_run_manifest(manifest_path)
    manifest["last_evaluation"] = {
        "checkpoint": str(checkpoint_path),
        "report_sha256": digest,
        "run_kind": gate_result.run_kind,
        "proxy": gate_result.proxy,
        "formal_candidate_eligible": gate_result.formal_candidate_eligible,
        "release_gate_passed": gate_result.passed,
    }
    if started_gpu_seconds is not None:
        manifest["last_evaluation"].update(
            {
                "started_gpu_seconds": float(started_gpu_seconds),
                "completed_gpu_seconds": float(completed_gpu_seconds),
            }
        )
    _write_manifest_payload(manifest_path, manifest)
    return digest


def _evaluate_checkpoint(
    *,
    checkpoint_path: Path,
    gate_path: Path,
    artifact_root: Path,
    repository_root: Path,
    capability_bundle: FrozenCapabilityBundle | None = None,
    formal_batches: tuple[FormalEvaluationBatch, ...] | None = None,
    shared_budget: TrainingBudget | None = None,
    evaluation_started_gpu_seconds: float | None = None,
    calibrated_state: CalibratedRunState | None = None,
):
    """Evaluate one explicit checkpoint using its calibrated environment kind."""
    root = validate_artifact_root(
        artifact_root, repository_root=repository_root
    )
    calibrated = (
        _load_calibrated_run_state(root)
        if calibrated_state is None
        else calibrated_state
    )
    if not isinstance(calibrated, CalibratedRunState):
        raise PreflightError("formal evaluation calibrated state is invalid")
    budget = calibrated.budget if shared_budget is None else shared_budget
    if not isinstance(budget, TrainingBudget):
        raise PreflightError("formal evaluation shared budget is invalid")
    if shared_budget is not None and budget is not calibrated.budget:
        raise PreflightError(
            "formal evaluation must share the calibrated training budget"
        )
    if evaluation_started_gpu_seconds is None:
        evaluation_started = budget.consumed_gpu_seconds
    elif (
        shared_budget is None
        or not isinstance(evaluation_started_gpu_seconds, (int, float))
        or isinstance(evaluation_started_gpu_seconds, bool)
        or not math.isfinite(float(evaluation_started_gpu_seconds))
        or float(evaluation_started_gpu_seconds) != budget.consumed_gpu_seconds
    ):
        raise PreflightError("formal evaluation start differs from shared budget")
    else:
        evaluation_started = float(evaluation_started_gpu_seconds)
    if calibrated.config.run_kind == "formal":
        _validate_formal_bundle_identity(capability_bundle, calibrated.run_identity)
        if (
            not isinstance(formal_batches, tuple)
            or len(formal_batches) != 3
            or any(
                not isinstance(batch, FormalEvaluationBatch)
                for batch in formal_batches
            )
        ):
            raise PreflightError("formal evaluation environments are required")
    else:
        if capability_bundle is not None or formal_batches is not None:
            raise PreflightError(
                "development-smoke evaluation rejects formal environment inputs"
            )
    checkpoint_target = checkpoint_path.resolve(strict=True)
    authoritative_parent = (root / "checkpoints").resolve(strict=True)
    if checkpoint_target.parent != authoritative_parent:
        raise ArtifactRootError(
            "evaluate checkpoint must be an explicit file under checkpoints/"
        )
    checkpoint = load_checkpoint_for_resume(
        checkpoint_target,
        expected_contract_version=OBSERVATION_CONTRACT_VERSION,
        expected_config_hash=config_sha256(calibrated.config.as_frozen_dict()),
        expected_source_commit=_source_commit(repository_root),
        expected_run_identity=calibrated.run_identity,
        expected_worker_allocation=calibrated.allocation,
        expected_micro_batch_size=calibrated.micro_batch_size,
        expected_budget_extension_blocks=(
            calibrated.budget.budget_extension_blocks
        ),
        expected_total_gpu_budget_seconds=(
            calibrated.budget.total_gpu_seconds
        ),
    )
    policy = CrossAttentionPolicy()
    policy.load_state_dict(dict(checkpoint.model_state), strict=True)

    def persist_budget_position() -> None:
        current_manifest = _read_run_manifest(root / "run-manifest.json")
        current_step = current_manifest.get("global_step")
        current_allocation = current_manifest.get("platform_allocation")
        if type(current_step) is not int or not isinstance(
            current_allocation, dict
        ):
            raise ArtifactRootError("run manifest training position is invalid")
        _update_run_manifest(
            root / "run-manifest.json",
            source_commit=checkpoint.source_commit,
            config_hash=checkpoint.config_hash,
            run_identity=calibrated.run_identity,
            global_step=current_step,
            consumed_gpu_seconds=budget.consumed_gpu_seconds,
            platform_allocation={
                str(name): int(count)
                for name, count in current_allocation.items()
            },
        )

    start = time.monotonic()
    budget.begin_gpu_interval(
        monotonic_seconds=start,
        upper_bound_gpu_seconds=(
            FORMAL_EVALUATION_WATCHDOG_SECONDS
            if calibrated.config.run_kind == "formal"
            else TRAINING_ROLLOUT_UPDATE_UPPER_BOUND_GPU_SECONDS
        ),
    )
    try:
        if calibrated.config.run_kind == "formal":
            report = evaluate_formal_policy(
                policy,
                device="cuda",
                checkpoint_sha256=checkpoint.payload_sha256,
                run_identity=calibrated.run_identity,
                batches=formal_batches,
            )
        else:
            report = evaluate_proxy_policy(
                policy,
                device="cuda",
                checkpoint_sha256=checkpoint.payload_sha256,
                schedule=CurriculumSchedule(),
                run_identity=calibrated.run_identity,
            )
        torch.cuda.synchronize()
    except BaseException:
        try:
            budget.end_gpu_interval(monotonic_seconds=time.monotonic())
        finally:
            persist_budget_position()
        raise
    else:
        try:
            budget.end_gpu_interval(monotonic_seconds=time.monotonic())
        finally:
            persist_budget_position()
    evaluation_completed = budget.consumed_gpu_seconds
    gate_result = evaluate_release_gate(report, load_gate_rules(gate_path))
    if calibrated.config.run_kind == "formal":
        _write_formal_evaluation_artifacts(
            root=root,
            checkpoint_path=checkpoint_target,
            report=report,
            gate_result=gate_result,
            started_gpu_seconds=evaluation_started,
            completed_gpu_seconds=evaluation_completed,
        )
    else:
        _write_development_evaluation_artifacts(
            root=root,
            checkpoint_path=checkpoint_target,
            report=report,
            gate_result=gate_result,
        )
    return report, gate_result


def _run_curriculum_training(
    *,
    calibrated: CalibratedRunState,
    artifact_root: Path,
    repository_root: Path,
    source_commit: str,
    initial_global_step: int,
    max_updates: int | None,
    interrupt_first_update: bool,
    restore_checkpoint,
    capability_bundle: FrozenCapabilityBundle | None = None,
    formal_environment_factory: FrozenCapabilityEnvironmentFactory | None = None,
    formal_observation_template: PolicyBatch | None = None,
    evaluate_candidate: (
        Callable[[Path, TrainingBudget, float], GateResult] | None
    ) = None,
) -> _RunEvidence:
    """Run one smoke override or advance across four active-GPU phases."""
    if calibrated.config.run_kind == "formal":
        _validate_formal_bundle_identity(capability_bundle, calibrated.run_identity)
        if not isinstance(
            formal_environment_factory, FrozenCapabilityEnvironmentFactory
        ) or formal_environment_factory.scenario_schedule_id != (
            calibrated.scenario_schedule_id
        ):
            raise PreflightError(
                "formal training factory differs from calibrated scenario schedule"
            )
        if (
            formal_environment_factory is not None
            and formal_environment_factory.bundle is not capability_bundle
        ):
            raise PreflightError(
                "formal environment factory must bind the validated bundle object"
            )
        rollout_factory = formal_environment_factory
    else:
        if capability_bundle is not None or formal_environment_factory is not None:
            raise PreflightError("development-smoke cannot consume formal capability")
        rollout_factory = proxy_environment_factory
    schedule = CurriculumSchedule()
    checkpoint = restore_checkpoint
    global_step = initial_global_step
    while True:
        phase = schedule.phase_for(
            consumed_gpu_s=calibrated.budget.consumed_gpu_seconds,
            calibration_end_gpu_s=calibrated.calibration_end_gpu_seconds,
        )
        allocation = schedule.worker_allocation(
            phase, selected_workers=calibrated.selected_workers
        )
        phase_end = schedule.phase_end_gpu_s(
            phase,
            calibration_end_gpu_s=calibrated.calibration_end_gpu_seconds,
        )
        if phase == "joint":
            phase_end = calibrated.budget.total_gpu_seconds
        evaluation_due_gpu_seconds: float | None = None
        if phase == "joint" and evaluate_candidate is not None:
            manifest = _read_run_manifest(artifact_root / "run-manifest.json")
            last_evaluation = manifest.get("last_evaluation")
            last_started = (
                last_evaluation.get("started_gpu_seconds")
                if isinstance(last_evaluation, dict)
                else None
            )
            evaluation_due_gpu_seconds = next_joint_evaluation_gpu_seconds(
                schedule=schedule,
                calibration_end_gpu_seconds=(
                    calibrated.calibration_end_gpu_seconds
                ),
                last_evaluation_started_gpu_seconds=last_started,
            )
        evidence = _run_updates(
            config=calibrated.config,
            artifact_root=artifact_root,
            repository_root=repository_root,
            source_commit=source_commit,
            budget=calibrated.budget,
            allocation=allocation,
            micro_batch_size=calibrated.micro_batch_size,
            initial_global_step=global_step,
            max_updates=max_updates if max_updates is not None else 2**63 - 1,
            interrupt_first_update=interrupt_first_update,
            restore_checkpoint=checkpoint,
            curriculum_phase=phase,
            phase_end_gpu_seconds=phase_end,
            pause_after_gpu_seconds=evaluation_due_gpu_seconds,
            rollout_environment_factory=rollout_factory,
            rollout_observation_template=formal_observation_template,
            rollout_reward_fn=compute_transition_reward,
            run_identity=calibrated.run_identity,
        )
        if (
            max_updates is not None
            or calibrated.budget.exhausted
            or evidence.signal_observed_at_update_boundary
        ):
            return evidence
        latest = load_checkpoint(
            _checkpoint_target(
                artifact_root, kind="latest", global_step=evidence.global_step
            )
        )
        next_phase = schedule.phase_for(
            consumed_gpu_s=calibrated.budget.consumed_gpu_seconds,
            calibration_end_gpu_s=calibrated.calibration_end_gpu_seconds,
        )
        if (
            phase == "joint"
            and evaluate_candidate is not None
            and evaluation_due_gpu_seconds is not None
            and calibrated.budget.consumed_gpu_seconds
            >= evaluation_due_gpu_seconds
        ):
            evaluation_started = calibrated.budget.consumed_gpu_seconds
            gate_result = evaluate_candidate(
                _latest_candidate_checkpoint(artifact_root),
                calibrated.budget,
                evaluation_started,
            )
            if not isinstance(gate_result, GateResult):
                raise PreflightError(
                    "automatic candidate evaluation returned invalid gate result"
                )
            if gate_result.passed or calibrated.budget.exhausted:
                return evidence
            checkpoint = latest
            global_step = latest.global_step
            interrupt_first_update = False
            continue
        if next_phase == phase:
            return evidence
        checkpoint = latest
        global_step = latest.global_step
        interrupt_first_update = False


def _start_training_run(
    *,
    config_path: Path,
    artifact_root: Path,
    repository_root: Path,
    max_updates: int | None,
    interrupt_first_update: bool,
    capability_bundle: FrozenCapabilityBundle | None = None,
    formal_environment_factory: FrozenCapabilityEnvironmentFactory | None = None,
    formal_observation_template: PolicyBatch | None = None,
    formal_evaluation_batches: tuple[FormalEvaluationBatch, ...] | None = None,
    formal_gate_path: Path | None = None,
) -> _RunEvidence:
    requested_config = load_training_config(config_path)
    if (formal_evaluation_batches is None) != (formal_gate_path is None):
        raise PreflightError("automatic formal evaluation inputs are incomplete")
    if requested_config.run_kind == "formal":
        if max_updates is not None:
            raise PreflightError("formal train cannot bound updates")
        if capability_bundle is None:
            raise PreflightError("formal capability bundle is required")
    else:
        if capability_bundle is not None or formal_evaluation_batches is not None:
            raise PreflightError("development-smoke rejects formal capability bundle")
        if type(max_updates) is not int or not 1 <= max_updates <= 2:
            raise PreflightError("development-smoke permits only one or two updates")
    root = validate_artifact_root(
        artifact_root, repository_root=repository_root
    )
    if not root.exists():
        if requested_config.run_kind == "formal":
            raise PreflightError("formal train requires a frozen calibrated run")
        _calibrate_training_run(
            config_path=config_path,
            artifact_root=root,
            repository_root=repository_root,
        )
    calibrated = _load_calibrated_run_state(root)
    config = calibrated.config
    expected_config = (
        with_rollout_horizon(requested_config, calibrated.rollout_horizon)
        if requested_config.run_kind == "formal"
        else requested_config
    )
    if expected_config.as_frozen_dict() != config.as_frozen_dict():
        raise ArtifactRootError("train config differs from calibrated run")
    if config.run_kind == "formal":
        _validate_formal_bundle_identity(capability_bundle, calibrated.run_identity)
    source_commit = _source_commit(repository_root)
    _seed_everything(calibrated.formal_seed)

    def evaluate_candidate(
        checkpoint: Path,
        budget: TrainingBudget,
        started_gpu_seconds: float,
    ) -> GateResult:
        if formal_evaluation_batches is None or formal_gate_path is None:
            raise PreflightError("automatic formal evaluation inputs are missing")
        _, result = _evaluate_checkpoint(
            checkpoint_path=checkpoint,
            gate_path=formal_gate_path,
            artifact_root=root,
            repository_root=repository_root,
            capability_bundle=capability_bundle,
            formal_batches=formal_evaluation_batches,
            shared_budget=budget,
            evaluation_started_gpu_seconds=started_gpu_seconds,
            calibrated_state=calibrated,
        )
        return result

    return _run_curriculum_training(
        calibrated=calibrated,
        artifact_root=root,
        repository_root=repository_root,
        source_commit=source_commit,
        initial_global_step=0,
        max_updates=max_updates,
        interrupt_first_update=interrupt_first_update,
        restore_checkpoint=None,
        capability_bundle=capability_bundle,
        formal_environment_factory=formal_environment_factory,
        formal_observation_template=formal_observation_template,
        evaluate_candidate=(
            evaluate_candidate
            if formal_evaluation_batches is not None
            and formal_gate_path is not None
            else None
        ),
    )


def _resume_training_run(
    *,
    artifact_root: Path,
    checkpoint_path: Path | None = None,
    repository_root: Path,
    max_updates: int | None,
    capability_bundle: FrozenCapabilityBundle | None = None,
    formal_environment_factory: FrozenCapabilityEnvironmentFactory | None = None,
    formal_observation_template: PolicyBatch | None = None,
    formal_evaluation_batches: tuple[FormalEvaluationBatch, ...] | None = None,
    formal_gate_path: Path | None = None,
) -> _RunEvidence:
    root = validate_artifact_root(
        artifact_root, repository_root=repository_root
    )
    calibrated = _load_calibrated_run_state(root)
    manifest = _read_run_manifest(root / "run-manifest.json")
    frozen_config = manifest.get("frozen_config")
    if not isinstance(frozen_config, dict):
        raise ArtifactRootError("run manifest frozen config is missing")
    config = resolve_training_config(frozen_config)
    if (formal_evaluation_batches is None) != (formal_gate_path is None):
        raise PreflightError("automatic formal evaluation inputs are incomplete")
    if config.run_kind == "formal":
        if max_updates is not None:
            raise PreflightError("formal resume cannot bound updates")
        _validate_formal_bundle_identity(capability_bundle, calibrated.run_identity)
    else:
        if capability_bundle is not None or formal_evaluation_batches is not None:
            raise PreflightError("development-smoke rejects formal capability bundle")
        if type(max_updates) is not int or not 1 <= max_updates <= 2:
            raise PreflightError("development-smoke permits only one or two updates")
    source_commit = _source_commit(repository_root)
    allocation_raw = manifest.get("platform_allocation")
    if not isinstance(allocation_raw, dict):
        raise ArtifactRootError("run manifest platform allocation is missing")
    allocation = {name: int(value) for name, value in allocation_raw.items()}
    runtime_calibration = manifest.get("runtime_calibration")
    if not isinstance(runtime_calibration, dict):
        raise ArtifactRootError("run manifest runtime calibration is missing")
    micro_batch_size = runtime_calibration.get("selected_micro_batch")
    if type(micro_batch_size) is not int or micro_batch_size <= 0:
        raise ArtifactRootError("run manifest micro-batch is invalid")
    target = (
        checkpoint_path.resolve(strict=True)
        if checkpoint_path is not None
        else _checkpoint_target(root, kind="latest", global_step=0).resolve(
            strict=True
        )
    )
    authoritative_parent = (root / "checkpoints").resolve(strict=True)
    if target.parent != authoritative_parent:
        raise ArtifactRootError("resume checkpoint must be under checkpoints/")
    checkpoint = load_checkpoint_for_resume(
        target,
        expected_contract_version=OBSERVATION_CONTRACT_VERSION,
        expected_config_hash=config_sha256(config.as_frozen_dict()),
        expected_source_commit=source_commit,
        expected_run_identity=calibrated.run_identity,
        expected_worker_allocation=allocation,
        expected_micro_batch_size=micro_batch_size,
        expected_budget_extension_blocks=(
            calibrated.budget.budget_extension_blocks
        ),
        expected_total_gpu_budget_seconds=(
            calibrated.budget.total_gpu_seconds
        ),
    )
    if manifest.get("global_step") != checkpoint.global_step:
        raise ArtifactRootError("run manifest global step differs from latest checkpoint")
    manifest_budget = manifest.get("consumed_gpu_seconds")
    if (
        not isinstance(manifest_budget, (int, float))
        or float(manifest_budget) < checkpoint.consumed_gpu_seconds
    ):
        raise ArtifactRootError("run manifest budget differs from latest checkpoint")
    expected_budget_state = "exhausted" if calibrated.budget.exhausted else "active"
    if manifest.get("budget_state") != expected_budget_state:
        raise ArtifactRootError("run manifest budget state differs from checkpoint")
    phases = (
        "warmup_wheeled",
        "warmup_legged",
        "warmup_hopper",
        "joint",
    )
    derived_phase = CurriculumSchedule().phase_for(
        consumed_gpu_s=float(manifest_budget),
        calibration_end_gpu_s=calibrated.calibration_end_gpu_seconds,
    )
    if checkpoint.curriculum_phase not in phases or (
        checkpoint.curriculum_phase != derived_phase
        and phases.index(derived_phase) - phases.index(checkpoint.curriculum_phase)
        != 1
    ):
        raise ArtifactRootError("checkpoint curriculum phase differs from GPU budget")

    def evaluate_candidate(
        candidate: Path,
        budget: TrainingBudget,
        started_gpu_seconds: float,
    ) -> GateResult:
        if formal_evaluation_batches is None or formal_gate_path is None:
            raise PreflightError("automatic formal evaluation inputs are missing")
        _, result = _evaluate_checkpoint(
            checkpoint_path=candidate,
            gate_path=formal_gate_path,
            artifact_root=root,
            repository_root=repository_root,
            capability_bundle=capability_bundle,
            formal_batches=formal_evaluation_batches,
            shared_budget=budget,
            evaluation_started_gpu_seconds=started_gpu_seconds,
            calibrated_state=calibrated,
        )
        return result

    return _run_curriculum_training(
        calibrated=calibrated,
        artifact_root=root,
        repository_root=repository_root,
        source_commit=source_commit,
        initial_global_step=checkpoint.global_step,
        max_updates=max_updates,
        interrupt_first_update=False,
        restore_checkpoint=checkpoint,
        capability_bundle=capability_bundle,
        formal_environment_factory=formal_environment_factory,
        formal_observation_template=formal_observation_template,
        evaluate_candidate=(
            evaluate_candidate
            if formal_evaluation_batches is not None
            and formal_gate_path is not None
            else None
        ),
    )


def _parallel_pool_startup_timeout_seconds(
    initial_episode_states: tuple[Mapping[str, object], ...] | None,
) -> float:
    """Give active-episode reconstruction headroom without loosening steps."""
    return 600.0 if initial_episode_states is not None else 60.0


def _run_updates(
    *,
    config: ResolvedTrainingConfig,
    artifact_root: Path,
    repository_root: Path,
    source_commit: str,
    budget: TrainingBudget,
    allocation: dict[str, int],
    micro_batch_size: int,
    initial_global_step: int,
    max_updates: int,
    interrupt_first_update: bool,
    restore_checkpoint,
    curriculum_phase: str = "joint",
    phase_end_gpu_seconds: float | None = None,
    pause_after_gpu_seconds: float | None = None,
    rollout_environment_factory: Callable[
        [int, str], ParallelEnvironmentWorker
    ] | None = None,
    rollout_observation_template: PolicyBatch | None = None,
    rollout_reward_fn: Callable[[PlannerTransition], float] | None = None,
    run_identity: RunIdentity,
) -> _RunEvidence:
    if type(max_updates) is not int or max_updates <= 0:
        raise ValueError("max updates must be a positive integer")
    if not isinstance(run_identity, RunIdentity) or (
        run_identity.run_kind != config.run_kind
    ):
        raise PreflightError("training run identity differs from configuration")
    if config.run_kind == "formal":
        if not isinstance(
            rollout_environment_factory, FrozenCapabilityEnvironmentFactory
        ):
            raise PreflightError(
                "formal environment factory must bind the capability bundle"
            )
        if (
            rollout_environment_factory.bundle.bundle_sha256
            != run_identity.capability_sha256
        ):
            raise PreflightError("formal worker capability identity mismatch")
        if not isinstance(rollout_observation_template, PolicyBatch):
            raise PreflightError("formal observation template is required")
        environment_factory = rollout_environment_factory
        observation_template = rollout_observation_template
    else:
        if rollout_environment_factory not in (None, proxy_environment_factory):
            raise PreflightError("development-smoke must remain proxy-only")
        environment_factory = proxy_environment_factory
        observation_template = proxy_observation(0, "WHEELED", step=0)
    _validated_cuda_device()
    reward_fn = rollout_reward_fn or compute_transition_reward
    policy = CrossAttentionPolicy()
    trainer = ResumablePPOTrainer(
        policy,
        reward_fn=reward_fn,
        ppo_config=config.ppo,
        device="cuda",
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        trainer.optimizer, lr_lambda=lambda step: 1.0
    )
    stop_flag = SignalStopFlag()
    sent_interrupt = False
    rollout_policy_version = initial_global_step
    initial_episode_states: tuple[Mapping[str, object], ...] | None = None
    if config.run_kind == "formal" and restore_checkpoint is not None:
        environment_state = restore_checkpoint.environment_state
        if (
            environment_state["scenario_schedule_id"]
            != environment_factory.scenario_schedule_id
        ):
            raise PreflightError("checkpoint formal scenario schedule mismatch")
        initial_episode_states = resume_worker_episode_states(
            restore_checkpoint,
            target_phase=curriculum_phase,
            target_allocation=allocation,
        )
    pool = ParallelEnvPool(
        allocation=allocation,
        observation_template=observation_template,
        environment_factory=environment_factory,
        reward_fn=reward_fn,
        worker_timeout_seconds=60.0,
        worker_startup_timeout_seconds=(
            _parallel_pool_startup_timeout_seconds(initial_episode_states)
        ),
        initial_episode_states=initial_episode_states,
    )
    environment = _ParallelPoolVectorEnv(
        pool, policy_version=rollout_policy_version
    )
    metrics_journal = TrainingMetricsJournal(
        artifact_root / "metrics" / "train.jsonl",
        resume_global_step=initial_global_step,
    )
    worker_platforms = tuple(
        platform
        for platform, count in allocation.items()
        for _ in range(count)
    )
    environment.use_normalization_state(trainer.normalization)
    environment.reset()
    if restore_checkpoint is not None:
        # Process creation and shared-buffer setup are outside the durable RNG
        # boundary.  Restore last so resumed update N+1 starts at exactly the
        # same Python/NumPy/Torch RNG state as uninterrupted update N+1.
        restore_training_state(
            restore_checkpoint,
            trainer.policy,
            trainer.optimizer,
            scheduler,
            normalization_state=trainer.normalization,
        )

    def collect_rollout() -> _CollectedTrainingRollout:
        environment.set_policy_version(rollout_policy_version)
        started = time.perf_counter()
        collected = collect_ppo_rollout(
            environment,
            trainer.policy,
            _collector_config_for_run(config),
            device="cuda",
        )
        collect_wall_seconds = time.perf_counter() - started
        horizon = config.ppo.rollout_horizon
        worker_count = pool.worker_count
        raw_rewards = np.asarray(
            environment.raw_rewards, dtype=np.float32
        ).reshape(horizon, worker_count)
        start_coverage = (
            collected.rollout.pose_features.reshape(
                horizon, worker_count, -1
            )[0, :, 4]
            .astype(np.float32, copy=True)
        )
        end_coverage = (
            environment.current_observations.pose_features[:, 4]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32, copy=True)
        )
        return _CollectedTrainingRollout(
            collected=collected,
            raw_rewards=raw_rewards,
            start_coverage=start_coverage,
            end_coverage=end_coverage,
            success_first_crossings=tuple(
                environment.success_first_crossings
            ),
            planning_outcomes=tuple(environment.planning_outcomes),
            candidate_diagnostics=tuple(environment.candidate_diagnostics),
            no_candidate_terminations=tuple(
                environment.no_candidate_terminations
            ),
            terminal_audits=tuple(environment.terminal_audits),
            collect_wall_seconds=collect_wall_seconds,
            worker_wait_seconds=environment.worker_wait_seconds,
        )

    def update_rollout(
        rollout: _CollectedTrainingRollout,
    ) -> _CompletedTrainingUpdate:
        nonlocal sent_interrupt, rollout_policy_version
        timer: threading.Timer | None = None
        if interrupt_first_update and not sent_interrupt:
            sent_interrupt = True
            timer = threading.Timer(
                0.01, lambda: os.kill(os.getpid(), signal.SIGTERM)
            )
            timer.start()
        try:
            started = time.perf_counter()
            metrics = trainer.update(
                rollout.collected.rollout,
                micro_batch_size=micro_batch_size,
            )
            scheduler.step()
            rollout_policy_version += 1
            environment.advance_policy_version(rollout_policy_version)
            return _CompletedTrainingUpdate(
                ppo_metrics=metrics,
                update_wall_seconds=time.perf_counter() - started,
            )
        finally:
            if timer is not None:
                timer.join()

    def record_update(
        rollout: _CollectedTrainingRollout,
        result: object,
        state: TrainingLoopState,
    ) -> None:
        if not isinstance(result, _CompletedTrainingUpdate):
            raise RuntimeError("completed training update metrics are missing")
        metrics_journal.append(
            build_training_update_record(
                global_step=state.global_step,
                timestamp_utc=datetime.now(timezone.utc)
                .isoformat()
                .replace("+00:00", "Z"),
                curriculum_phase=curriculum_phase,
                platform_allocation=allocation,
                worker_platforms=worker_platforms,
                rollout_horizon=config.ppo.rollout_horizon,
                raw_rewards=rollout.raw_rewards,
                dones=rollout.collected.dones,
                start_coverage=rollout.start_coverage,
                end_coverage=rollout.end_coverage,
                success_first_crossings=(
                    rollout.success_first_crossings
                ),
                planning_outcomes=rollout.planning_outcomes,
                candidate_diagnostics=rollout.candidate_diagnostics,
                no_candidate_terminations=(
                    rollout.no_candidate_terminations
                ),
                terminal_audits=rollout.terminal_audits,
                ppo_metrics=result.ppo_metrics,
                collect_wall_seconds=rollout.collect_wall_seconds,
                update_wall_seconds=result.update_wall_seconds,
                worker_wait_seconds=rollout.worker_wait_seconds,
            )
        )

    def save(kind: str, state: TrainingLoopState) -> None:
        checkpoint = build_training_checkpoint(
            model=trainer.policy,
            optimizer=trainer.optimizer,
            scheduler=scheduler,
            global_step=state.global_step,
            curriculum_phase=curriculum_phase,
            normalization=trainer.normalization,
            frozen_config=config.as_frozen_dict(),
            run_identity=run_identity,
            source_commit=source_commit,
            consumed_gpu_seconds=budget.consumed_gpu_seconds,
            budget_extension_blocks=budget.budget_extension_blocks,
            total_gpu_budget_seconds=budget.total_gpu_seconds,
            worker_allocation=allocation,
            micro_batch_size=micro_batch_size,
            latest_checkpoint_gpu_seconds=(
                state.latest_checkpoint_gpu_seconds
            ),
            candidate_checkpoint_gpu_seconds=(
                state.candidate_checkpoint_gpu_seconds
            ),
            environment_state=(
                {
                    "schema_version": FORMAL_ENVIRONMENT_STATE_SCHEMA_VERSION,
                    "scenario_schedule_id": environment_factory.scenario_schedule_id,
                    "worker_episode_states": list(
                        pool.snapshot_episode_states(
                            policy_version=rollout_policy_version
                        )
                    ),
                }
                if config.run_kind == "formal"
                else {}
            ),
        )
        target = _checkpoint_target(
            artifact_root, kind=kind, global_step=state.global_step
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        overwrite = kind == "latest"
        save_checkpoint_atomic(target, checkpoint, overwrite=overwrite)
        _update_run_manifest(
            artifact_root / "run-manifest.json",
            source_commit=source_commit,
            config_hash=checkpoint.config_hash,
            run_identity=run_identity,
            global_step=state.global_step,
            consumed_gpu_seconds=budget.consumed_gpu_seconds,
            platform_allocation=allocation,
            training_metrics=metrics_journal.checkpoint_summary(
                artifact_root=artifact_root
            ),
        )

    loop = TrainingBoundaryLoop(
        budget=budget,
        stop_flag=stop_flag,
        checkpoint_interval_seconds=config.checkpoint_interval_seconds,
        candidate_checkpoint_interval_seconds=(
            config.candidate_checkpoint_interval_seconds
        ),
        curriculum_phase=curriculum_phase,
        phase_end_gpu_seconds=phase_end_gpu_seconds,
        pause_after_gpu_seconds=pause_after_gpu_seconds,
        initial_global_step=initial_global_step,
        initial_latest_checkpoint_gpu_seconds=(
            restore_checkpoint.latest_checkpoint_gpu_seconds
            if restore_checkpoint is not None
            else 0.0
        ),
        initial_candidate_checkpoint_gpu_seconds=(
            restore_checkpoint.candidate_checkpoint_gpu_seconds
            if restore_checkpoint is not None
            else 0.0
        ),
        synchronize_device=torch.cuda.synchronize,
    )
    try:
        with stop_flag.installed():
            state = loop.run(
                collect_rollout=collect_rollout,
                update_rollout=update_rollout,
                record_update=record_update,
                save_checkpoint=save,
                max_updates=max_updates,
            )
    finally:
        pool.close()
    loaded = load_checkpoint(
        _checkpoint_target(artifact_root, kind="latest", global_step=state.global_step)
    )
    if loaded.global_step != state.global_step:
        raise RuntimeError("latest checkpoint missed the completed update boundary")
    return _RunEvidence(
        global_step=state.global_step,
        consumed_gpu_seconds=budget.consumed_gpu_seconds,
        platform_allocation=allocation,
        signal_observed_at_update_boundary=(
            state.stop_signal == signal.SIGTERM
            and not state.rollout_discarded
            and state.global_step > initial_global_step
        ),
    )


def _policy_batch_digest(batch: PolicyBatch) -> str:
    digest = hashlib.sha256()
    for name in (
        "prior_channels",
        "coverage_summary",
        "local_crop",
        "frontier_features",
        "pose_features",
        "candidate_mask",
        "platform_context",
    ):
        values = getattr(batch, name).detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(values.dtype.str.encode("ascii"))
        digest.update(str(values.shape).encode("ascii"))
        digest.update(values.tobytes(order="C"))
    digest.update(repr(batch.observation_identities).encode("utf-8"))
    return digest.hexdigest()


def _candidate_batch_digest(batch: PolicyBatch) -> str:
    return _semantic_sha256(
        {
            "frontier_features": batch.frontier_features,
            "candidate_mask": batch.candidate_mask,
        }
    )


def _rollout_batch_field_digests(batch: RolloutBatch) -> dict[str, str]:
    if not isinstance(batch, RolloutBatch):
        raise PreflightError("formal resume rollout proof is invalid")
    output: dict[str, str] = {}
    for name in (
        "prior_channels",
        "coverage_summary",
        "local_crop",
        "frontier_features",
        "pose_features",
        "candidate_mask",
        "platform_context",
        "selected_frontier_indices",
        "selected_thetas",
        "old_log_prob_total",
        "old_values",
        "advantages",
        "returns",
    ):
        digest = hashlib.sha256()
        values = np.ascontiguousarray(getattr(batch, name))
        digest.update(name.encode("ascii"))
        digest.update(str(values.dtype).encode("ascii"))
        digest.update(repr(values.shape).encode("ascii"))
        digest.update(values.view(np.uint8).tobytes())
        output[name] = digest.hexdigest()
    return output


def _first_formal_request_digest(
    factory: FrozenCapabilityEnvironmentFactory,
    states: tuple[Mapping[str, object], ...],
) -> str:
    """Rebuild one active worker per platform and hash its next real request."""
    signatures: dict[str, str] = {}
    for raw in states:
        state = FormalWorkerState.from_dict(raw)
        if state.platform_type in signatures:
            continue
        worker = factory.restore_for_episode(
            worker_index=state.worker_index,
            platform_type=state.platform_type,
            episode_cursor=state.episode_cursor,
            platform_worker_index=state.platform_worker_index,
            platform_worker_count=state.platform_worker_count,
            state=raw,
        )
        observation = worker.environment.current_observation
        candidates = observation.candidate_mask[0].nonzero().flatten()
        if candidates.numel() == 0:
            raise PreflightError(
                "formal resume request proof reached a terminal observation"
            )
        action = PolicyAction(int(candidates[0].item()), 0.0)
        prepared = worker.episode.build_request(
            action,
            observation.observation_identities[0],
        )
        signatures[state.platform_type] = _request_signature(prepared.request)
    if set(signatures) != set(PLATFORMS):
        raise PreflightError("formal resume request proof missed a platform")
    return hashlib.sha256(
        json.dumps(signatures, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def _formal_resume_equivalence_check(
    *,
    config: ResolvedTrainingConfig,
    assembly: FormalEnvironmentAssembly,
    run_identity: RunIdentity,
    source_commit: str,
    artifact_root: Path,
    selected_workers: int,
    micro_batch_size: int,
) -> dict[str, object]:
    """Run update one, persist V6, then compare update two across resume."""
    _validated_cuda_device()
    if selected_workers not in (18, 24):
        raise PreflightError("formal resume proof requires 18 or 24 workers")
    allocation = {
        platform: selected_workers // len(PLATFORMS) for platform in PLATFORMS
    }
    resume_root = artifact_root / "resume-equivalence"
    resume_root.mkdir(parents=True, exist_ok=False)
    checkpoint_path = resume_root / "update-1.pt"
    checkpoint_relative_path = checkpoint_path.relative_to(artifact_root).as_posix()

    def runtime(
        *,
        policy_version: int,
        initial_states: tuple[Mapping[str, object], ...] | None = None,
    ):
        policy = CrossAttentionPolicy()
        trainer = ResumablePPOTrainer(
            policy,
            reward_fn=compute_transition_reward,
            ppo_config=config.ppo,
            device="cuda",
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            trainer.optimizer, lr_lambda=lambda _step: 1.0
        )
        pool = ParallelEnvPool(
            allocation=allocation,
            observation_template=assembly.observation_template,
            environment_factory=assembly.factory,
            reward_fn=compute_transition_reward,
            worker_timeout_seconds=120.0,
            initial_episode_states=initial_states,
        )
        environment = _ParallelPoolVectorEnv(pool, policy_version=policy_version)
        environment.use_normalization_state(trainer.normalization)
        environment.reset()
        return trainer, scheduler, pool, environment

    def update_once(
        trainer, scheduler, environment, *, policy_version: int
    ) -> dict[str, str]:
        environment.set_policy_version(policy_version)
        collected = collect_ppo_rollout(
            environment,
            trainer.policy,
            _collector_config_for_run(config, horizon=1),
            device="cuda",
        )
        rollout_sha256s = _rollout_batch_field_digests(collected.rollout)
        rollout_sha256s["step_rewards"] = _semantic_sha256(
            torch.from_numpy(collected.rewards)
        )
        rollout_sha256s["step_dones"] = _semantic_sha256(
            torch.from_numpy(collected.dones)
        )
        rollout_sha256s["final_observation"] = _policy_batch_digest(
            environment.current_observations
        )
        trainer.update(collected.rollout, micro_batch_size=micro_batch_size)
        scheduler.step()
        environment.advance_policy_version(policy_version + 1)
        torch.cuda.synchronize()
        return rollout_sha256s

    def environment_state(
        pool: ParallelEnvPool, *, policy_version: int
    ) -> tuple[dict[str, object], tuple[Mapping[str, object], ...]]:
        states = pool.snapshot_episode_states(policy_version=policy_version)
        return (
            {
                "schema_version": FORMAL_ENVIRONMENT_STATE_SCHEMA_VERSION,
                "scenario_schedule_id": assembly.scenario_schedule_id,
                "worker_episode_states": list(states),
            },
            states,
        )

    def checkpoint_for(
        trainer,
        scheduler,
        *,
        global_step: int,
        state: Mapping[str, object],
    ):
        return build_training_checkpoint(
            model=trainer.policy,
            optimizer=trainer.optimizer,
            scheduler=scheduler,
            global_step=global_step,
            curriculum_phase="preflight_resume_equivalence",
            normalization=trainer.normalization,
            frozen_config=config.as_frozen_dict(),
            run_identity=run_identity,
            source_commit=source_commit,
            consumed_gpu_seconds=0.0,
            budget_extension_blocks=0,
            total_gpu_budget_seconds=float(config.total_gpu_budget_seconds),
            worker_allocation=allocation,
            micro_batch_size=micro_batch_size,
            latest_checkpoint_gpu_seconds=0.0,
            candidate_checkpoint_gpu_seconds=0.0,
            environment_state=state,
        )

    _seed_everything(FORMAL_SEED)
    trainer = scheduler = pool = environment = None
    resumed_pool = None
    try:
        trainer, scheduler, pool, environment = runtime(policy_version=0)
        update_once(trainer, scheduler, environment, policy_version=0)
        update_one_state, _ = environment_state(
            pool, policy_version=1
        )
        update_one = checkpoint_for(
            trainer, scheduler, global_step=1, state=update_one_state
        )
        save_checkpoint_atomic(checkpoint_path, update_one, overwrite=False)
        loaded = load_checkpoint_for_resume(
            checkpoint_path,
            expected_contract_version=OBSERVATION_CONTRACT_VERSION,
            expected_config_hash=config_sha256(config.as_frozen_dict()),
            expected_source_commit=source_commit,
            expected_run_identity=run_identity,
            expected_worker_allocation=allocation,
            expected_micro_batch_size=micro_batch_size,
            expected_budget_extension_blocks=0,
            expected_total_gpu_budget_seconds=float(
                config.total_gpu_budget_seconds
            ),
        )
        checkpoint_roundtrip = (
            loaded.payload_sha256 == update_one.payload_sha256
            and loaded.environment_state == update_one.environment_state
        )

        uninterrupted_pre = checkpoint_for(
            trainer,
            scheduler,
            global_step=1,
            state=update_one_state,
        )
        for name in (
            "model_state",
            "optimizer_state",
            "scheduler_state",
            "normalization",
            "rng_state",
            "environment_state",
        ):
            if _semantic_sha256(getattr(uninterrupted_pre, name)) != (
                _semantic_sha256(getattr(loaded, name))
            ):
                raise PreflightError(
                    f"formal resume uninterrupted pre-update {name} differs"
                )
        uninterrupted_pre_observation_sha256 = _policy_batch_digest(
            environment.current_observations
        )

        uninterrupted_rollout_sha256 = update_once(
            trainer, scheduler, environment, policy_version=1
        )
        uninterrupted_state, uninterrupted_states = environment_state(
            pool, policy_version=2
        )
        uninterrupted_observation = environment.current_observations
        uninterrupted_request = _first_formal_request_digest(
            assembly.factory, uninterrupted_states
        )
        uninterrupted = checkpoint_for(
            trainer,
            scheduler,
            global_step=2,
            state=uninterrupted_state,
        )
        uninterrupted_observation_sha256 = _policy_batch_digest(
            uninterrupted_observation
        )
        uninterrupted_candidate_sha256 = _candidate_batch_digest(
            uninterrupted_observation
        )
        pool.close()
        pool = None

        _seed_everything(FORMAL_SEED + 999)
        resumed_trainer, resumed_scheduler, resumed_pool, resumed_environment = (
            runtime(
                policy_version=1,
                initial_states=tuple(
                    loaded.environment_state["worker_episode_states"]
                ),
            )
        )
        restore_training_state(
            loaded,
            resumed_trainer.policy,
            resumed_trainer.optimizer,
            resumed_scheduler,
            normalization_state=resumed_trainer.normalization,
        )
        resumed_pre_state, _ = environment_state(
            resumed_pool, policy_version=1
        )
        resumed_pre = checkpoint_for(
            resumed_trainer,
            resumed_scheduler,
            global_step=1,
            state=resumed_pre_state,
        )
        for name in (
            "model_state",
            "optimizer_state",
            "scheduler_state",
            "normalization",
            "rng_state",
            "environment_state",
        ):
            if _semantic_sha256(getattr(resumed_pre, name)) != (
                _semantic_sha256(getattr(loaded, name))
            ):
                raise PreflightError(
                    f"formal resume restored pre-update {name} differs"
                )
        if _policy_batch_digest(
            resumed_environment.current_observations
        ) != uninterrupted_pre_observation_sha256:
            raise PreflightError(
                "formal resume restored pre-update observation differs"
            )
        resumed_rollout_sha256 = update_once(
            resumed_trainer,
            resumed_scheduler,
            resumed_environment,
            policy_version=1,
        )
        if resumed_rollout_sha256 != uninterrupted_rollout_sha256:
            differing_fields = tuple(
                name
                for name in uninterrupted_rollout_sha256
                if uninterrupted_rollout_sha256[name]
                != resumed_rollout_sha256.get(name)
            )
            raise PreflightError(
                "formal resume rollout differs at update two: "
                + ",".join(differing_fields)
            )
        resumed_state, resumed_states = environment_state(
            resumed_pool, policy_version=2
        )
        resumed_observation = resumed_environment.current_observations
        resumed_request = _first_formal_request_digest(
            assembly.factory, resumed_states
        )
        resumed = checkpoint_for(
            resumed_trainer,
            resumed_scheduler,
            global_step=2,
            state=resumed_state,
        )
        evidence = _build_formal_resume_equivalence_evidence(
            checkpoint_relative_path=checkpoint_relative_path,
            checkpoint_sha256=loaded.payload_sha256,
            checkpoint_roundtrip=checkpoint_roundtrip,
            rollout_exact=True,
            uninterrupted=uninterrupted,
            resumed=resumed,
            uninterrupted_observation_sha256=(
                uninterrupted_observation_sha256
            ),
            resumed_observation_sha256=_policy_batch_digest(
                resumed_observation
            ),
            uninterrupted_candidate_sha256=(
                uninterrupted_candidate_sha256
            ),
            resumed_candidate_sha256=_candidate_batch_digest(
                resumed_observation
            ),
            uninterrupted_request_sha256=uninterrupted_request,
            resumed_request_sha256=resumed_request,
        )
    finally:
        if pool is not None:
            pool.close()
        if resumed_pool is not None:
            resumed_pool.close()
    return evidence


class _CudaPlannerCalibrationWorkload:
    """Same real PlannerBridge + policy workload at 18 and 24 workers."""

    _RUNTIME_PROBE_HORIZON = 4

    def __init__(
        self,
        ppo_config: PPOConfig,
        *,
        environment_factory: FrozenCapabilityEnvironmentFactory | None = None,
        observation_template: PolicyBatch | None = None,
    ) -> None:
        _validated_cuda_device()
        if (environment_factory is None) != (observation_template is None):
            raise PreflightError(
                "formal calibration factory and observation template are inseparable"
            )
        if environment_factory is not None and not isinstance(
            environment_factory, FrozenCapabilityEnvironmentFactory
        ):
            raise PreflightError("formal calibration factory is invalid")
        if observation_template is not None and not isinstance(
            observation_template, PolicyBatch
        ):
            raise PreflightError("formal calibration observation template is invalid")
        self._environment_factory = environment_factory
        self._observation_template = observation_template
        self._formal = environment_factory is not None
        self._ppo_config = ppo_config
        self._pool: ParallelEnvPool | None = None
        self._environment: _ParallelPoolVectorEnv | None = None
        self._workers: int | None = None
        self._policy_version = 0
        self._trainer: ResumablePPOTrainer | None = ResumablePPOTrainer(
            CrossAttentionPolicy(),
            reward_fn=compute_transition_reward,
            ppo_config=ppo_config,
            device="cuda",
        )

    def __call__(self, workers: int, micro_batch: int) -> CalibrationMeasurement:
        self._ensure_pool(workers)
        assert self._pool is not None
        assert self._environment is not None
        assert self._trainer is not None
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        wall_start = time.perf_counter()
        planner_timeouts = 0
        ipc_failures = 0
        optimizer_steps = 0
        oom = False
        gpu_seconds = 0.0
        try:
            self._policy_version += 1
            self._environment.set_policy_version(self._policy_version)
            start_event.record()
            collected = collect_ppo_rollout(
                self._environment,
                self._trainer.policy,
                CollectorConfig(
                    horizon=self._RUNTIME_PROBE_HORIZON,
                    deterministic=True,
                ),
                device="cuda",
            )
            metrics = self._trainer.update(
                collected.rollout, micro_batch_size=micro_batch
            )
            optimizer_steps = metrics.optimizer_steps
            end_event.record()
            torch.cuda.synchronize()
            gpu_seconds = max(start_event.elapsed_time(end_event) / 1000.0, 0.0)
            planner_timeouts = sum(
                outcome == PlanningOutcome.RESOURCE_EXHAUSTED
                and "TIMEOUT" in reason.upper()
                for outcome, reason in zip(
                    self._environment.planning_outcomes,
                    self._environment.reason_codes,
                )
            )
        except torch.cuda.OutOfMemoryError:
            oom = True
            torch.cuda.synchronize()
            gpu_seconds = max(time.perf_counter() - wall_start, 0.0)
        except ParallelPoolError:
            ipc_failures = 1
            torch.cuda.synchronize()
            gpu_seconds = max(time.perf_counter() - wall_start, 0.0)
        wall_seconds = max(time.perf_counter() - wall_start, 1.0e-9)
        total_memory = torch.cuda.get_device_properties(0).total_memory
        peak_fraction = torch.cuda.max_memory_allocated() / total_memory
        samples = workers * self._RUNTIME_PROBE_HORIZON
        return CalibrationMeasurement(
            workers=workers,
            micro_batch=micro_batch,
            throughput_samples_per_second=samples / wall_seconds,
            peak_gpu_memory_fraction=float(peak_fraction),
            planner_timeouts=planner_timeouts,
            oom=oom,
            gpu_seconds=gpu_seconds,
            ipc_failures=ipc_failures,
            optimizer_steps=optimizer_steps,
        )

    def measure_rollout_horizon(
        self,
        workers: int,
        micro_batch: int,
        rollout_horizon: int,
        transitions_per_worker: int,
    ) -> HorizonCalibrationMeasurement:
        """Compare one horizon using equal transitions and a fresh seeded run."""
        if not self._formal:
            raise PreflightError(
                "rollout horizon calibration requires the formal environment"
            )
        if transitions_per_worker % rollout_horizon:
            raise PreflightError("rollout horizon probe work is not divisible")
        assert self._environment_factory is not None
        assert self._observation_template is not None
        self.close()
        _seed_everything(FORMAL_SEED)
        self._trainer = None
        torch.cuda.empty_cache()
        selected_ppo_config = replace(
            self._ppo_config,
            rollout_horizon=rollout_horizon,
        )
        trainer = ResumablePPOTrainer(
            CrossAttentionPolicy(),
            reward_fn=compute_transition_reward,
            ppo_config=selected_ppo_config,
            device="cuda",
        )
        allocation = _allocation_for_workers(workers)
        pool: ParallelEnvPool | None = None
        resumed_pool: ParallelEnvPool | None = None
        environment: _ParallelPoolVectorEnv | None = None
        planner_timeouts = 0
        ipc_failures = 0
        completed_updates = 0
        update_walls: list[float] = []
        approximate_kls: list[float] = []
        value_losses: list[float] = []
        advantages_finite = True
        returns_finite = True
        continuity_verified = True
        resume_verified = False
        resume_digest: str | None = None
        oom = False
        worker_wait_seconds = 0.0
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        wall_start = time.perf_counter()
        gpu_seconds = 0.0
        failure_reason: str | None = None
        try:
            pool = ParallelEnvPool(
                allocation=allocation,
                observation_template=self._observation_template,
                environment_factory=self._environment_factory,
                reward_fn=compute_transition_reward,
                worker_timeout_seconds=60.0,
            )
            environment = _ParallelPoolVectorEnv(pool, policy_version=0)
            environment.reset()
            start_event.record()
            for policy_version in range(
                transitions_per_worker // rollout_horizon
            ):
                environment.set_policy_version(policy_version)
                collected = collect_ppo_rollout(
                    environment,
                    trainer.policy,
                    CollectorConfig(
                        horizon=rollout_horizon,
                        deterministic=True,
                    ),
                    device="cuda",
                )
                worker_wait_seconds += environment.worker_wait_seconds
                planner_timeouts += sum(
                    outcome == PlanningOutcome.RESOURCE_EXHAUSTED
                    and "TIMEOUT" in reason.upper()
                    for outcome, reason in zip(
                        environment.planning_outcomes,
                        environment.reason_codes,
                    )
                )
                advantages_finite &= bool(
                    np.isfinite(collected.rollout.advantages).all()
                )
                returns_finite &= bool(
                    np.isfinite(collected.rollout.returns).all()
                )
                boundary_digest = _policy_batch_digest(
                    environment.current_observations
                )
                update_started = time.perf_counter()
                metrics = trainer.update(
                    collected.rollout,
                    micro_batch_size=micro_batch,
                )
                update_walls.append(time.perf_counter() - update_started)
                approximate_kls.append(metrics.approx_kl)
                value_losses.append(metrics.value_loss)
                completed_updates += 1
                advanced = environment.advance_policy_version(
                    policy_version + 1
                )
                continuity_verified &= (
                    boundary_digest == _policy_batch_digest(advanced)
                )
            states = pool.snapshot_episode_states(
                policy_version=completed_updates
            )
            active_digest = _policy_batch_digest(
                environment.current_observations
            )
            resume_digest = hashlib.sha256(
                (
                    json.dumps(
                        states,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                    + active_digest
                ).encode("utf-8")
            ).hexdigest()
            pool.close()
            pool = None
            resumed_pool = ParallelEnvPool(
                allocation=allocation,
                observation_template=self._observation_template,
                environment_factory=self._environment_factory,
                reward_fn=compute_transition_reward,
                worker_timeout_seconds=60.0,
                initial_episode_states=states,
            )
            restored = resumed_pool.reset().observations
            resume_verified = active_digest == _policy_batch_digest(restored)
            end_event.record()
            torch.cuda.synchronize()
            gpu_seconds = max(
                start_event.elapsed_time(end_event) / 1000.0,
                0.0,
            )
        except torch.cuda.OutOfMemoryError:
            oom = True
            failure_reason = "CUDA_OUT_OF_MEMORY"
            torch.cuda.synchronize()
            gpu_seconds = max(time.perf_counter() - wall_start, 0.0)
        except ParallelPoolError as error:
            ipc_failures = 1
            failure_reason = str(error)
            torch.cuda.synchronize()
            gpu_seconds = max(time.perf_counter() - wall_start, 0.0)
        finally:
            if pool is not None:
                pool.close()
            if resumed_pool is not None:
                resumed_pool.close()
        wall_seconds = max(time.perf_counter() - wall_start, 1.0e-9)
        completed_transitions = (
            completed_updates * workers * rollout_horizon
        )
        total_memory = torch.cuda.get_device_properties(0).total_memory
        peak_fraction = torch.cuda.max_memory_allocated() / total_memory
        return HorizonCalibrationMeasurement(
            workers=workers,
            micro_batch=micro_batch,
            rollout_horizon=rollout_horizon,
            total_transitions=workers * transitions_per_worker,
            completed_updates=completed_updates,
            throughput_transitions_per_second=(
                completed_transitions / wall_seconds
            ),
            mean_update_wall_seconds=(
                sum(update_walls) / len(update_walls)
                if update_walls
                else wall_seconds
            ),
            peak_gpu_memory_fraction=float(peak_fraction),
            worker_wait_ratio=min(worker_wait_seconds / wall_seconds, 1.0),
            planner_timeouts=planner_timeouts,
            oom=oom,
            gpu_seconds=gpu_seconds,
            ipc_failures=ipc_failures,
            approximate_kl=(
                sum(approximate_kls) / len(approximate_kls)
                if approximate_kls
                else 0.0
            ),
            value_loss=(
                sum(value_losses) / len(value_losses)
                if value_losses
                else 0.0
            ),
            advantages_finite=advantages_finite,
            returns_finite=returns_finite,
            update_boundary_continuity_verified=continuity_verified,
            resume_digest=resume_digest,
            resume_verified=resume_verified,
            failure_reason=failure_reason,
        )

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool = None
            self._environment = None
            self._workers = None

    def _ensure_pool(self, workers: int) -> None:
        if self._workers == workers:
            return
        self.close()
        allocation = _allocation_for_workers(workers)
        self._pool = ParallelEnvPool(
            allocation=allocation,
            observation_template=(
                self._observation_template
                if self._observation_template is not None
                else proxy_observation(0, "WHEELED", step=0)
            ),
            environment_factory=(
                self._environment_factory
                if self._environment_factory is not None
                else proxy_environment_factory
            ),
            reward_fn=compute_transition_reward,
            worker_timeout_seconds=60.0,
        )
        self._environment = _ParallelPoolVectorEnv(
            self._pool, policy_version=self._policy_version
        )
        self._workers = workers


def _calibration_observation(
    worker_index: int, platform_type: str
) -> PolicyBatch:
    platform_index = {"WHEELED": 0, "LEGGED": 1, "HOPPER": 2}[platform_type]
    platform_context = torch.zeros((1, 3), dtype=torch.float32)
    platform_context[0, platform_index] = 1.0
    pose_features = torch.zeros((1, 5), dtype=torch.float32)
    pose_features[0, 0] = float(worker_index)
    frontier_features = torch.zeros((1, 2, 22), dtype=torch.float32)
    frontier_features[..., 15] = 1.0
    return PolicyBatch(
        prior_channels=torch.zeros((1, 7, 8, 8), dtype=torch.float32),
        coverage_summary=torch.zeros((1, 8, 8, 8), dtype=torch.float32),
        local_crop=torch.zeros((1, 8, 8, 8), dtype=torch.float32),
        frontier_features=frontier_features,
        pose_features=pose_features,
        candidate_mask=torch.tensor([[True, False]], dtype=torch.bool),
        platform_context=platform_context,
        observation_identities=(
            ObservationIdentity(
                episode_id=f"calibration-{worker_index}",
                mission_revision=1,
                map_snapshot_id="calibration-map",
                robot_state_id=f"calibration-state-{worker_index}",
                state_time_ns=1_000,
                execution_state=(
                    "GROUND_HOLD"
                    if platform_type == "HOPPER"
                    else "DECISION_BOUNDARY"
                ),
                candidate_set_id="calibration-candidates",
            ),
        ),
    )


def _calibration_environment_factory(
    worker_index: int, platform_type: str
) -> ParallelEnvironmentWorker:
    observation = _calibration_observation(worker_index, platform_type)
    request = TrainingPlanRequest()
    request.request_id = f"task3-calibration-{worker_index}"

    def build_request(action, identity):
        request.state_time.nanoseconds_since_epoch = identity.state_time_ns
        apply_goal_theta(request.goal, platform_type, action.theta_rad)
        return PreparedPlanRequest(request=request, identity=identity)

    environment = create_v3_environment(
        platform_type=platform_type,
        request_builder=build_request,
        initial_observation=observation,
        observation_provider=lambda: observation,
        committed_hop_executor=(lambda: None) if platform_type == "HOPPER" else None,
    )
    return ParallelEnvironmentWorker(
        environment=environment,
        initial_observation=observation,
    )


def _calibration_reward(transition) -> float:
    return float(transition.mission_observed_delta)


def _proxy_policy_batch(sample_count: int, *, device: str) -> PolicyBatch:
    platform_context = torch.zeros(
        (sample_count, 3), dtype=torch.float32, device=device
    )
    platform_context[
        torch.arange(sample_count, device=device),
        torch.arange(sample_count, device=device) % 3,
    ] = 1.0
    frontier_features = torch.zeros(
        (sample_count, 2, 22), dtype=torch.float32, device=device
    )
    frontier_features[..., 15] = 1.0
    return PolicyBatch(
        prior_channels=torch.zeros(
            (sample_count, 7, 8, 8), dtype=torch.float32, device=device
        ),
        coverage_summary=torch.zeros(
            (sample_count, 8, 8, 8), dtype=torch.float32, device=device
        ),
        local_crop=torch.zeros(
            (sample_count, 8, 8, 8), dtype=torch.float32, device=device
        ),
        frontier_features=frontier_features,
        pose_features=torch.zeros(
            (sample_count, 6), dtype=torch.float32, device=device
        ),
        candidate_mask=torch.ones(
            (sample_count, 2), dtype=torch.bool, device=device
        ),
        platform_context=platform_context,
    )


def _proxy_rollout(
    policy: CrossAttentionPolicy, *, sample_count: int
) -> RolloutBatch:
    batch = _proxy_policy_batch(sample_count, device="cuda")
    with torch.no_grad():
        output = policy(batch)
        action = sample_action(
            output,
            batch.candidate_mask,
            batch.platform_context,
            deterministic=True,
        )
    advantages = torch.ones((sample_count,), dtype=torch.float32, device="cuda")
    advantages[1::2] = -1.0

    def numpy(value: torch.Tensor, dtype) -> np.ndarray:
        return value.detach().cpu().numpy().astype(dtype, copy=True)

    old_values = numpy(output.value, np.float32)
    return RolloutBatch(
        prior_channels=numpy(batch.prior_channels, np.float32),
        coverage_summary=numpy(batch.coverage_summary, np.float32),
        local_crop=numpy(batch.local_crop, np.float32),
        frontier_features=numpy(batch.frontier_features, np.float32),
        pose_features=numpy(batch.pose_features, np.float32),
        candidate_mask=numpy(batch.candidate_mask, np.bool_),
        platform_context=numpy(batch.platform_context, np.float32),
        selected_frontier_indices=numpy(
            action.selected_frontier_index, np.int64
        ),
        selected_thetas=numpy(action.selected_theta, np.float32),
        old_log_prob_total=numpy(action.log_prob_total, np.float32),
        old_values=old_values,
        advantages=numpy(advantages, np.float32),
        returns=old_values + np.float32(0.1),
    )


def _allocation_for_workers(workers: int) -> dict[str, int]:
    if workers == 24:
        return joint_worker_allocation(24)
    if workers == 18:
        return {"WHEELED": 6, "LEGGED": 6, "HOPPER": 6}
    raise ValueError("runtime calibration only supports 18 or 24 workers")


def _checkpoint_target(
    artifact_root: Path, *, kind: str, global_step: int
) -> Path:
    if type(global_step) is not int or global_step < 0:
        raise ValueError("checkpoint global step must be non-negative")
    if kind == "latest":
        name = "latest.pt"
    elif kind == "candidate":
        name = f"candidate-step-{global_step}.pt"
    else:
        raise ValueError("unknown checkpoint kind")
    return artifact_root / "checkpoints" / name


def _latest_candidate_checkpoint(artifact_root: Path) -> Path:
    """Return the regular candidate file with the greatest numeric step."""
    checkpoints = artifact_root / "checkpoints"
    if not checkpoints.is_dir() or checkpoints.is_symlink():
        raise ArtifactRootError("candidate checkpoint directory is missing")
    candidates: list[tuple[int, Path]] = []
    for path in checkpoints.iterdir():
        prefix = "candidate-step-"
        suffix = ".pt"
        if (
            not path.name.startswith(prefix)
            or not path.name.endswith(suffix)
            or path.is_symlink()
            or not path.is_file()
        ):
            continue
        step_text = path.name[len(prefix) : -len(suffix)]
        if step_text.isdigit():
            candidates.append((int(step_text), path))
    if not candidates:
        raise ArtifactRootError("no immutable candidate checkpoint exists")
    return max(candidates, key=lambda value: value[0])[1]


def _validated_cuda_device() -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for the training smoke")
    name = torch.cuda.get_device_name(0)
    if name != "NVIDIA GeForce RTX 4080 SUPER":
        raise RuntimeError(
            "training smoke requires NVIDIA GeForce RTX 4080 SUPER"
        )
    return name


def _seed_everything(seed: int) -> None:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("exact CUDA replay requires CUBLAS_WORKSPACE_CONFIG=:4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _source_commit(repository_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    commit = completed.stdout.strip()
    if len(commit) != 40 or any(
        character not in "0123456789abcdef" for character in commit
    ):
        raise RuntimeError("repository HEAD is not a full source commit")
    return commit


def _commit_is_ancestor(
    repository_root: Path, ancestor: str, descendant: str
) -> bool:
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode not in (0, 1):
        raise RuntimeError("source migration ancestry check failed")
    return completed.returncode == 0


def _commit_changed_paths(
    repository_root: Path, ancestor: str, descendant: str
) -> tuple[str, ...]:
    completed = subprocess.run(
        ["git", "diff", "--name-only", f"{ancestor}..{descendant}"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return tuple(line for line in completed.stdout.splitlines() if line)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy_regular_file_exclusive(source: Path, target: Path) -> None:
    if target.exists() or target.is_symlink():
        if not target.is_file() or _file_sha256(target) != _file_sha256(source):
            raise ArtifactRootError("source checkpoint backup already differs")
        return
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with source.open("rb") as input_stream, os.fdopen(
            descriptor, "wb"
        ) as output_stream:
            descriptor = -1
            shutil.copyfileobj(input_stream, output_stream, length=1024 * 1024)
            output_stream.flush()
            os.fsync(output_stream.fileno())
        directory_fd = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        if descriptor >= 0:
            os.close(descriptor)
        if target.exists() and target.is_file():
            target.unlink()
        raise


def _validated_source_migration_sensor_reports(
    *,
    previous_report_path: Path,
    current_report_path: Path,
    repository_root: Path,
    expected_old_source_commit: str,
    run_identity: RunIdentity,
) -> tuple[str, str]:
    try:
        previous = load_sensor_performance_report(previous_report_path)
        current = load_sensor_performance_report(current_report_path)
        expected_host = current_host_identity()
        previous_sha256 = validate_sensor_performance_report(
            previous,
            expected_host=expected_host,
            expected_source_commit=sensor_source_commit(
                repository_root,
                revision=expected_old_source_commit,
            ),
            expected_capability_sha256=run_identity.capability_sha256,
            expected_training_semantics_sha256=(
                run_identity.training_semantics_sha256
            ),
        )
        current_sha256 = validate_sensor_performance_report(
            current,
            expected_host=expected_host,
            expected_source_commit=sensor_source_commit(repository_root),
            expected_capability_sha256=run_identity.capability_sha256,
            expected_training_semantics_sha256=(
                run_identity.training_semantics_sha256
            ),
        )
    except SensorPerformanceError as error:
        raise PreflightError(
            f"source migration sensor evidence is invalid: {error}"
        ) from error
    return previous_sha256, current_sha256


def _prepare_source_migrated_resume_checkpoint(
    *,
    artifact_root: Path,
    checkpoint_path: Path,
    repository_root: Path,
    expected_old_source_commit: str,
    expected_global_step: int,
    previous_sensor_performance_report: Path,
    current_sensor_performance_report: Path,
) -> Path:
    """Create an immutable, audited resume copy for an in-line code repair."""
    root = validate_artifact_root(
        artifact_root, repository_root=repository_root
    )
    if type(expected_global_step) is not int or expected_global_step < 0:
        raise PreflightError("source migration expected step is invalid")
    checkpoint_target = checkpoint_path.resolve(strict=True)
    checkpoints = (root / "checkpoints").resolve(strict=True)
    if (
        checkpoint_target.parent != checkpoints
        or checkpoint_target.name != "latest.pt"
    ):
        raise ArtifactRootError(
            "source migration requires the authoritative latest checkpoint"
        )
    checkpoint = load_checkpoint(checkpoint_target)
    if (
        not isinstance(checkpoint, TrainingCheckpointV6)
        or checkpoint.run_identity.run_kind != "formal"
        or checkpoint.source_commit != expected_old_source_commit
        or checkpoint.global_step != expected_global_step
    ):
        raise PreflightError("source migration checkpoint identity differs")
    new_source_commit = _source_commit(repository_root)
    if not _commit_is_ancestor(
        repository_root, expected_old_source_commit, new_source_commit
    ):
        raise PreflightError("source migration is not a descendant repair")
    changed_paths = _commit_changed_paths(
        repository_root,
        expected_old_source_commit,
        new_source_commit,
    )
    manifest_path = root / "run-manifest.json"
    manifest = _read_run_manifest(manifest_path)
    try:
        manifest_identity = RunIdentity.from_mapping(manifest.get("run_identity"))
    except Exception as error:
        raise ArtifactRootError(
            "source migration manifest identity is invalid"
        ) from error
    if (
        manifest.get("source_commit") != expected_old_source_commit
        or manifest.get("global_step") != checkpoint.global_step
        or manifest.get("config_hash") != checkpoint.config_hash
        or manifest_identity != checkpoint.run_identity
        or manifest.get("consumed_gpu_seconds")
        != checkpoint.consumed_gpu_seconds
        or manifest.get("platform_allocation")
        != checkpoint.worker_allocation
    ):
        raise ArtifactRootError("source migration manifest differs from checkpoint")
    formal_environment = manifest.get("formal_environment")
    if not isinstance(formal_environment, dict):
        raise ArtifactRootError("source migration formal environment is missing")
    previous_sensor_sha256, current_sensor_sha256 = (
        _validated_source_migration_sensor_reports(
            previous_report_path=previous_sensor_performance_report,
            current_report_path=current_sensor_performance_report,
            repository_root=repository_root,
            expected_old_source_commit=expected_old_source_commit,
            run_identity=checkpoint.run_identity,
        )
    )
    if (
        formal_environment.get("sensor_performance_sha256")
        != previous_sensor_sha256
    ):
        raise ArtifactRootError(
            "source migration previous sensor evidence differs from manifest"
        )
    backup_path = checkpoints / (
        f"source-checkpoint-step-{checkpoint.global_step}-"
        f"{expected_old_source_commit[:12]}.pt"
    )
    _copy_regular_file_exclusive(checkpoint_target, backup_path)
    migrated = migrate_checkpoint_source_commit(
        checkpoint,
        expected_source_commit=expected_old_source_commit,
        new_source_commit=new_source_commit,
    )
    migrated_path = checkpoints / (
        f"resume-step-{checkpoint.global_step}-{new_source_commit[:12]}.pt"
    )
    if migrated_path.exists() or migrated_path.is_symlink():
        existing_migrated = load_checkpoint(migrated_path)
        if (
            not isinstance(existing_migrated, TrainingCheckpointV6)
            or existing_migrated.payload_sha256 != migrated.payload_sha256
        ):
            raise ArtifactRootError("existing source-migrated checkpoint differs")
    else:
        save_checkpoint_atomic(migrated_path, migrated, overwrite=False)
    verified = load_checkpoint_for_resume(
        migrated_path,
        expected_contract_version=OBSERVATION_CONTRACT_VERSION,
        expected_config_hash=checkpoint.config_hash,
        expected_source_commit=new_source_commit,
        expected_run_identity=checkpoint.run_identity,
        expected_worker_allocation=checkpoint.worker_allocation,
        expected_micro_batch_size=checkpoint.micro_batch_size,
        expected_budget_extension_blocks=checkpoint.budget_extension_blocks,
        expected_total_gpu_budget_seconds=checkpoint.total_gpu_budget_seconds,
    )
    migrations = manifest.get("source_migrations", [])
    if not isinstance(migrations, list):
        raise ArtifactRootError("source migration manifest history is invalid")
    migrations.append(
        {
            "schema_version": "lunar-training-source-migration/v1",
            "created_at_utc": datetime.now(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            "from_source_commit": expected_old_source_commit,
            "to_source_commit": new_source_commit,
            "global_step": checkpoint.global_step,
            "consumed_gpu_seconds": checkpoint.consumed_gpu_seconds,
            "original_checkpoint": str(checkpoint_target),
            "original_backup": str(backup_path),
            "migrated_checkpoint": str(migrated_path),
            "original_file_sha256": _file_sha256(checkpoint_target),
            "original_payload_sha256": checkpoint.payload_sha256,
            "migrated_payload_sha256": verified.payload_sha256,
            "previous_sensor_performance_report": str(
                previous_sensor_performance_report.resolve(strict=True)
            ),
            "previous_sensor_performance_sha256": previous_sensor_sha256,
            "current_sensor_performance_report": str(
                current_sensor_performance_report.resolve(strict=True)
            ),
            "current_sensor_performance_sha256": current_sensor_sha256,
            "changed_paths": list(changed_paths),
        }
    )
    manifest["source_commit"] = new_source_commit
    formal_environment["sensor_performance_sha256"] = current_sensor_sha256
    manifest["formal_environment"] = formal_environment
    manifest["source_migrations"] = migrations
    _write_manifest_payload(manifest_path, manifest)
    return migrated_path


def _development_run_identity(source_commit: str) -> RunIdentity:
    """Identity for bounded proxy smoke; it can never label a formal candidate."""
    def digest(label: str) -> str:
        return hashlib.sha256(label.encode("utf-8")).hexdigest()

    return RunIdentity(
        run_kind="development-smoke",
        data_sha256=digest("development-smoke/proxy-data/v1"),
        split_sha256=digest("development-smoke/proxy-split/v1"),
        generator_sha256=digest("development-smoke/proxy-generator/v1"),
        capability_sha256=digest(
            f"development-smoke/proxy-capability/v1:{source_commit}"
        ),
        reward_sha256=reward_weights_sha256(),
        v3_sha256=digest(f"lunar-planner-v3-source:{source_commit}"),
        training_semantics_sha256=training_semantics_sha256(),
    )


def _formal_run_identity(identity: FormalCacheIdentity) -> RunIdentity:
    """Project one verified full-cache identity into the resumable run identity."""
    if not isinstance(identity, FormalCacheIdentity):
        raise PreflightError("formal run identity requires a verified cache identity")
    return RunIdentity(
        run_kind="formal",
        data_sha256=identity.source_lock_file_sha256,
        split_sha256=identity.split_sha256,
        generator_sha256=identity.generator_sha256,
        capability_sha256=identity.capability_sha256,
        reward_sha256=identity.reward_sha256,
        v3_sha256=identity.v3_sha256,
        training_semantics_sha256=identity.training_semantics_sha256,
    )


def _read_run_manifest(path: Path) -> dict[str, object]:
    if not path.is_file() or path.is_symlink():
        raise ArtifactRootError("run manifest is missing or unsafe")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ArtifactRootError("run manifest is invalid") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != "lunar-training-run/v1":
        raise ArtifactRootError("run manifest schema is invalid")
    return payload


def _update_run_manifest(
    path: Path,
    *,
    source_commit: str,
    config_hash: str,
    run_identity: RunIdentity,
    global_step: int,
    consumed_gpu_seconds: float,
    platform_allocation: dict[str, int],
    training_metrics: Mapping[str, object] | None = None,
) -> None:
    payload = _read_run_manifest(path)
    if "runtime_calibration" not in payload:
        raise ArtifactRootError("run manifest calibration is missing")
    if not isinstance(run_identity, RunIdentity):
        raise ArtifactRootError("run manifest identity must use RunIdentity")
    existing_identity = payload.get("run_identity")
    if existing_identity is not None:
        try:
            existing = RunIdentity.from_mapping(existing_identity)
        except Exception as error:
            raise ArtifactRootError("run manifest identity is invalid") from error
        if existing != run_identity:
            raise ArtifactRootError("run manifest identity cannot drift")
    try:
        budget = TrainingBudget(
            total_gpu_seconds=payload.get("total_gpu_budget_seconds"),
            consumed_gpu_seconds=consumed_gpu_seconds,
            budget_extension_blocks=payload.get("budget_extension_blocks"),
        )
    except (BudgetExceededError, ValueError) as error:
        raise ArtifactRootError("run manifest budget identity is invalid") from error
    if training_metrics is not None:
        metrics = dict(training_metrics)
        if (
            metrics.get("schema_version")
            != "lunar-training-metrics-summary/v1"
            or metrics.get("path") != "metrics/train.jsonl"
            or metrics.get("last_global_step") != global_step
            or not isinstance(metrics.get("sha256"), str)
            or len(metrics["sha256"]) != 64
            or any(
                character not in "0123456789abcdef"
                for character in metrics["sha256"]
            )
        ):
            raise ArtifactRootError("run manifest training metrics are invalid")
    payload.update(
        {
            "source_commit": source_commit,
            "config_hash": config_hash,
            "run_identity": run_identity.to_dict(),
            "global_step": global_step,
            "consumed_gpu_seconds": consumed_gpu_seconds,
            "budget_state": (
                "exhausted" if budget.exhausted else "active"
            ),
            "platform_allocation": dict(platform_allocation),
        }
    )
    if training_metrics is not None:
        payload["training_metrics"] = dict(training_metrics)
    _write_manifest_payload(path, payload)


def _write_manifest_payload(path: Path, payload: dict[str, object]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(
                payload,
                stream,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception as error:
        if temporary.exists():
            temporary.unlink()
        raise ArtifactRootError("run manifest update failed") from error


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ArtifactRootError",
    "CudaSmokeEvidence",
    "PreflightError",
    "ResumablePPOTrainer",
    "SignalStopFlag",
    "TrainingBoundaryLoop",
    "TrainingLoopState",
    "build_parser",
    "main",
    "run_cuda_interrupt_resume_smoke",
    "validate_artifact_root",
]

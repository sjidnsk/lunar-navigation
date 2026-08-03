"""Volume 3 training entry point; Task 4 extends this parser later."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import signal
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from types import FrameType
from typing import TypeVar

import numpy as np
import torch
from lunar_planner_training_bridge import TrainingPlanRequest

from .budget import (
    CalibrationMeasurement,
    TrainingBudget,
    calibrate_runtime,
)
from .checkpoint import (
    build_training_checkpoint,
    config_sha256,
    load_checkpoint,
    load_checkpoint_for_resume,
    restore_training_state,
    save_checkpoint_atomic,
)
from .config import (
    ResolvedTrainingConfig,
    load_training_config,
    resolve_training_config,
)
from .environment.parallel_pool import (
    ParallelActions,
    ParallelEnvironmentWorker,
    ParallelEnvPool,
    ParallelPoolError,
    joint_worker_allocation,
)
from .environment.macro_step import PlannerTransition
from .environment.v3_environment import create_v3_environment
from .policy.cross_attention import CrossAttentionPolicy, sample_action
from .policy.observation import PolicyBatch
from .ppo.rollout import RolloutBatch
from .ppo.trainer import PPOTrainer


_Rollout = TypeVar("_Rollout")


class ArtifactRootError(ValueError):
    """Runtime artifacts must use an explicit absolute path outside Git."""


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


class ResumablePPOTrainer:
    """Task 1 PPO core plus an injectable Task 4-compatible reward boundary."""

    def __init__(
        self,
        policy: CrossAttentionPolicy,
        *,
        reward_fn: Callable[[PlannerTransition], float],
        device: torch.device | str,
    ) -> None:
        if not callable(reward_fn):
            raise ValueError("reward function must be callable")
        self._reward_fn = reward_fn
        self._ppo = PPOTrainer(policy, device=device)

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

    def update(self, rollout: RolloutBatch):
        return self._ppo.update(rollout)


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
        initial_global_step: int = 0,
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
        self._global_step = initial_global_step
        self._clock = clock
        self._synchronize_device = synchronize_device or (lambda: None)

    def run(
        self,
        *,
        collect_rollout: Callable[[], _Rollout],
        update_rollout: Callable[[_Rollout], None],
        save_checkpoint: Callable[[str, TrainingLoopState], None],
        max_updates: int,
    ) -> TrainingLoopState:
        if not all(
            callable(value)
            for value in (collect_rollout, update_rollout, save_checkpoint)
        ):
            raise ValueError("boundary loop callbacks must be callable")
        if type(max_updates) is not int or max_updates <= 0:
            raise ValueError("max updates must be a positive integer")
        last_latest = self._clock()
        last_candidate = last_latest
        updates = 0
        while updates < max_updates:
            rollout = collect_rollout()
            if self._stop_flag.requested:
                state = self._state(rollout_discarded=True)
                save_checkpoint("latest", state)
                return state
            interval_start = self._clock()
            self._budget.begin_gpu_interval(monotonic_seconds=interval_start)
            update_error: BaseException | None = None
            try:
                update_rollout(rollout)
                self._synchronize_device()
            except BaseException as error:
                update_error = error
            finally:
                interval_end = self._clock()
                self._budget.end_gpu_interval(monotonic_seconds=interval_end)
            if update_error is not None:
                raise update_error
            self._global_step += 1
            updates += 1
            now = self._clock()
            latest_saved = False
            if now - last_latest >= self._checkpoint_interval_seconds:
                save_checkpoint("latest", self._state(rollout_discarded=False))
                last_latest = now
                latest_saved = True
            if (
                self._curriculum_phase == "joint"
                and now - last_candidate
                >= self._candidate_checkpoint_interval_seconds
            ):
                save_checkpoint("candidate", self._state(rollout_discarded=False))
                last_candidate = now
            if self._stop_flag.requested:
                state = self._state(rollout_discarded=False)
                if not latest_saved:
                    save_checkpoint("latest", state)
                return state
        state = self._state(rollout_discarded=False)
        save_checkpoint("latest", state)
        return state

    def _state(self, *, rollout_discarded: bool) -> TrainingLoopState:
        return TrainingLoopState(
            global_step=self._global_step,
            rollout_discarded=rollout_discarded,
            stop_signal=self._stop_flag.signal_number,
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lunar-policy-training")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train")
    train.add_argument("--config", required=True)
    train.add_argument("--artifact-root", required=True)
    train.add_argument("--max-updates", type=int, default=1)

    resume = subparsers.add_parser("resume")
    resume.add_argument("--artifact-root", required=True)
    resume.add_argument("--max-updates", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    repository_root = Path(__file__).resolve().parents[3]
    if arguments.command == "train":
        _start_training_run(
            config_path=Path(arguments.config),
            artifact_root=Path(arguments.artifact_root),
            repository_root=repository_root,
            max_updates=arguments.max_updates,
            interrupt_first_update=False,
        )
    elif arguments.command == "resume":
        _resume_training_run(
            artifact_root=Path(arguments.artifact_root),
            repository_root=repository_root,
            max_updates=arguments.max_updates,
        )
    return 0


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
        max_updates=100,
        interrupt_first_update=True,
    )
    resumed = _resume_training_run(
        artifact_root=Path(artifact_root),
        repository_root=Path(repository_root),
        max_updates=1,
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


def _start_training_run(
    *,
    config_path: Path,
    artifact_root: Path,
    repository_root: Path,
    max_updates: int,
    interrupt_first_update: bool,
) -> _RunEvidence:
    root = validate_artifact_root(
        artifact_root, repository_root=repository_root
    )
    if root.exists():
        raise ArtifactRootError("new training artifact root already exists")
    if not root.parent.is_dir():
        raise ArtifactRootError("artifact root parent directory is missing")
    root.mkdir()
    config = load_training_config(config_path)
    source_commit = _source_commit(repository_root)
    _seed_everything(config.formal_training_seeds[0])
    budget = TrainingBudget(
        total_gpu_seconds=float(config.total_gpu_budget_seconds)
    )
    workload = _CudaPlannerCalibrationWorkload()
    try:
        calibration = calibrate_runtime(
            config=config,
            workload=workload,
            budget=budget,
            manifest_path=root / "run-manifest.json",
            micro_batch_candidates=(1, 2, 4),
        )
    finally:
        workload.close()
    allocation = _allocation_for_workers(calibration.selected_workers)
    _update_run_manifest(
        root / "run-manifest.json",
        source_commit=source_commit,
        config_hash=config_sha256(config.as_frozen_dict()),
        global_step=0,
        consumed_gpu_seconds=budget.consumed_gpu_seconds,
        platform_allocation=allocation,
    )
    return _run_updates(
        config=config,
        artifact_root=root,
        repository_root=repository_root,
        source_commit=source_commit,
        budget=budget,
        allocation=allocation,
        initial_global_step=0,
        max_updates=max_updates,
        interrupt_first_update=interrupt_first_update,
        restore_checkpoint=None,
    )


def _resume_training_run(
    *,
    artifact_root: Path,
    repository_root: Path,
    max_updates: int,
) -> _RunEvidence:
    root = validate_artifact_root(
        artifact_root, repository_root=repository_root
    )
    manifest = _read_run_manifest(root / "run-manifest.json")
    frozen_config = manifest.get("frozen_config")
    if not isinstance(frozen_config, dict):
        raise ArtifactRootError("run manifest frozen config is missing")
    config = resolve_training_config(frozen_config)
    source_commit = _source_commit(repository_root)
    checkpoint = load_checkpoint_for_resume(
        root / "latest.pt",
        expected_contract_version="ObservationContractV1",
        expected_config_hash=config_sha256(config.as_frozen_dict()),
        expected_source_commit=source_commit,
    )
    if manifest.get("global_step") != checkpoint.global_step:
        raise ArtifactRootError("run manifest global step differs from latest checkpoint")
    manifest_budget = manifest.get("consumed_gpu_seconds")
    if (
        not isinstance(manifest_budget, (int, float))
        or float(manifest_budget) != checkpoint.consumed_gpu_seconds
    ):
        raise ArtifactRootError("run manifest budget differs from latest checkpoint")
    allocation_raw = manifest.get("platform_allocation")
    if not isinstance(allocation_raw, dict):
        raise ArtifactRootError("run manifest platform allocation is missing")
    allocation = {name: int(value) for name, value in allocation_raw.items()}
    budget = TrainingBudget.from_checkpoint(checkpoint)
    return _run_updates(
        config=config,
        artifact_root=root,
        repository_root=repository_root,
        source_commit=source_commit,
        budget=budget,
        allocation=allocation,
        initial_global_step=checkpoint.global_step,
        max_updates=max_updates,
        interrupt_first_update=False,
        restore_checkpoint=checkpoint,
    )


def _run_updates(
    *,
    config: ResolvedTrainingConfig,
    artifact_root: Path,
    repository_root: Path,
    source_commit: str,
    budget: TrainingBudget,
    allocation: dict[str, int],
    initial_global_step: int,
    max_updates: int,
    interrupt_first_update: bool,
    restore_checkpoint,
) -> _RunEvidence:
    if type(max_updates) is not int or max_updates <= 0:
        raise ValueError("max updates must be a positive integer")
    _validated_cuda_device()
    policy = CrossAttentionPolicy()
    trainer = ResumablePPOTrainer(
        policy,
        reward_fn=_calibration_reward,
        device="cuda",
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        trainer.optimizer, lr_lambda=lambda step: 1.0
    )
    if restore_checkpoint is not None:
        restore_training_state(
            restore_checkpoint,
            trainer.policy,
            trainer.optimizer,
            scheduler,
        )
    stop_flag = SignalStopFlag()
    sample_count = sum(allocation.values())
    sent_interrupt = False

    def collect_rollout() -> RolloutBatch:
        return _proxy_rollout(trainer.policy, sample_count=sample_count)

    def update_rollout(rollout: RolloutBatch) -> None:
        nonlocal sent_interrupt
        timer: threading.Timer | None = None
        if interrupt_first_update and not sent_interrupt:
            sent_interrupt = True
            timer = threading.Timer(
                0.01, lambda: os.kill(os.getpid(), signal.SIGTERM)
            )
            timer.start()
        try:
            trainer.update(rollout)
            scheduler.step()
        finally:
            if timer is not None:
                timer.join()

    normalization = {"reward_mean": 0.0, "reward_var": 1.0}

    def save(kind: str, state: TrainingLoopState) -> None:
        checkpoint = build_training_checkpoint(
            model=trainer.policy,
            optimizer=trainer.optimizer,
            scheduler=scheduler,
            global_step=state.global_step,
            curriculum_phase="joint",
            normalization=normalization,
            frozen_config=config.as_frozen_dict(),
            source_commit=source_commit,
            consumed_gpu_seconds=budget.consumed_gpu_seconds,
        )
        if kind == "latest":
            target = artifact_root / "latest.pt"
            overwrite = True
        elif kind == "candidate":
            target = artifact_root / f"candidate-step-{state.global_step}.pt"
            overwrite = False
        else:
            raise ValueError("unknown checkpoint kind")
        save_checkpoint_atomic(target, checkpoint, overwrite=overwrite)
        _update_run_manifest(
            artifact_root / "run-manifest.json",
            source_commit=source_commit,
            config_hash=checkpoint.config_hash,
            global_step=state.global_step,
            consumed_gpu_seconds=budget.consumed_gpu_seconds,
            platform_allocation=allocation,
        )

    loop = TrainingBoundaryLoop(
        budget=budget,
        stop_flag=stop_flag,
        checkpoint_interval_seconds=config.checkpoint_interval_seconds,
        candidate_checkpoint_interval_seconds=(
            config.candidate_checkpoint_interval_seconds
        ),
        curriculum_phase="joint",
        initial_global_step=initial_global_step,
        synchronize_device=torch.cuda.synchronize,
    )
    with stop_flag.installed():
        state = loop.run(
            collect_rollout=collect_rollout,
            update_rollout=update_rollout,
            save_checkpoint=save,
            max_updates=max_updates,
        )
    loaded = load_checkpoint(artifact_root / "latest.pt")
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


class _CudaPlannerCalibrationWorkload:
    """Same real PlannerBridge + policy workload at 18 and 24 workers."""

    def __init__(self) -> None:
        _validated_cuda_device()
        self._pool: ParallelEnvPool | None = None
        self._workers: int | None = None
        self._policy_version = 0
        self._policy = CrossAttentionPolicy().cuda().eval()

    def __call__(self, workers: int, micro_batch: int) -> CalibrationMeasurement:
        self._ensure_pool(workers)
        assert self._pool is not None
        actions = ParallelActions(
            candidate_indices=torch.zeros((workers,), dtype=torch.int64),
            thetas=torch.zeros((workers,), dtype=torch.float32),
        )
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        wall_start = time.perf_counter()
        planner_timeouts = 0
        oom = False
        gpu_seconds = 0.0
        try:
            for _ in range(micro_batch):
                self._policy_version += 1
                self._pool.step(actions, policy_version=self._policy_version)
            batch = _proxy_policy_batch(workers * micro_batch, device="cuda")
            start_event.record()
            with torch.no_grad():
                self._policy(batch)
            end_event.record()
            torch.cuda.synchronize()
            gpu_seconds = max(start_event.elapsed_time(end_event) / 1000.0, 0.0)
        except torch.cuda.OutOfMemoryError:
            oom = True
            torch.cuda.synchronize()
            gpu_seconds = max(time.perf_counter() - wall_start, 0.0)
        except ParallelPoolError as error:
            if "timed out" not in str(error):
                raise
            planner_timeouts = 1
        wall_seconds = max(time.perf_counter() - wall_start, 1.0e-9)
        total_memory = torch.cuda.get_device_properties(0).total_memory
        peak_fraction = torch.cuda.max_memory_allocated() / total_memory
        samples = workers * micro_batch
        return CalibrationMeasurement(
            workers=workers,
            micro_batch=micro_batch,
            throughput_samples_per_second=samples / wall_seconds,
            peak_gpu_memory_fraction=float(peak_fraction),
            planner_timeouts=planner_timeouts,
            oom=oom,
            gpu_seconds=gpu_seconds,
        )

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool = None
            self._workers = None

    def _ensure_pool(self, workers: int) -> None:
        if self._workers == workers:
            return
        self.close()
        allocation = _allocation_for_workers(workers)
        self._pool = ParallelEnvPool(
            allocation=allocation,
            observation_template=_calibration_observation(0, "WHEELED"),
            environment_factory=_calibration_environment_factory,
            reward_fn=_calibration_reward,
            worker_timeout_seconds=60.0,
        )
        self._pool.reset()
        self._workers = workers


def _calibration_observation(
    worker_index: int, platform_type: str
) -> PolicyBatch:
    platform_index = {"WHEELED": 0, "LEGGED": 1, "HOPPER": 2}[platform_type]
    platform_context = torch.zeros((1, 3), dtype=torch.float32)
    platform_context[0, platform_index] = 1.0
    pose_features = torch.zeros((1, 6), dtype=torch.float32)
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
    )


def _calibration_environment_factory(
    worker_index: int, platform_type: str
) -> ParallelEnvironmentWorker:
    observation = _calibration_observation(worker_index, platform_type)
    request = TrainingPlanRequest()
    request.request_id = f"task3-calibration-{worker_index}"
    environment = create_v3_environment(
        platform_type=platform_type,
        request_builder=lambda action: request,
        initial_observation=observation,
        committed_hop_executor=(lambda: None) if platform_type == "HOPPER" else None,
    )
    return ParallelEnvironmentWorker(
        environment=environment,
        initial_observation=observation,
    )


def _calibration_reward(transition) -> float:
    return float(transition.coverage_delta)


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
        action = sample_action(output, batch.candidate_mask, deterministic=True)
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
    global_step: int,
    consumed_gpu_seconds: float,
    platform_allocation: dict[str, int],
) -> None:
    payload = _read_run_manifest(path)
    if "runtime_calibration" not in payload:
        raise ArtifactRootError("run manifest calibration is missing")
    payload.update(
        {
            "source_commit": source_commit,
            "config_hash": config_hash,
            "global_step": global_step,
            "consumed_gpu_seconds": consumed_gpu_seconds,
            "platform_allocation": dict(platform_allocation),
        }
    )
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
    "ResumablePPOTrainer",
    "SignalStopFlag",
    "TrainingBoundaryLoop",
    "TrainingLoopState",
    "build_parser",
    "main",
    "run_cuda_interrupt_resume_smoke",
    "validate_artifact_root",
]

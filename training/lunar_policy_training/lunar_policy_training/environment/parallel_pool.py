"""Synchronous on-policy workers backed by shared Torch double buffers."""

from __future__ import annotations

import math
import multiprocessing as mp
import os
import queue
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import torch

from ..config import PLATFORMS
from ..policy.observation import PolicyBatch, validate_policy_batch
from .macro_step import PlannerTransition, PolicyAction


_OBSERVATION_FIELDS = (
    "prior_channels",
    "coverage_summary",
    "local_crop",
    "frontier_features",
    "pose_features",
    "candidate_mask",
    "platform_context",
)


class ParallelPoolError(RuntimeError):
    """A worker or shared rollout invariant failed; training must stop."""


@dataclass(frozen=True, slots=True)
class ParallelEnvironmentWorker:
    """An environment created inside one worker plus its episode observation."""

    environment: object
    initial_observation: PolicyBatch


@dataclass(slots=True)
class ParallelActions:
    candidate_indices: torch.Tensor
    thetas: torch.Tensor


@dataclass(frozen=True, slots=True)
class ParallelRolloutStep:
    observations: PolicyBatch
    rewards: torch.Tensor
    dones: torch.Tensor
    policy_versions: torch.Tensor
    buffer_index: int


class ParallelEnvPool:
    """Own one spawned process/environment/bridge per synchronous worker."""

    def __init__(
        self,
        *,
        allocation: Mapping[str, int],
        observation_template: PolicyBatch,
        environment_factory: Callable[[int, str], ParallelEnvironmentWorker],
        reward_fn: Callable[[PlannerTransition], float],
        worker_timeout_seconds: float = 30.0,
    ) -> None:
        self._platforms = _expanded_platforms(allocation)
        self.worker_count = len(self._platforms)
        _validate_template(observation_template)
        if not callable(environment_factory):
            raise ParallelPoolError("environment factory must be callable")
        if not callable(reward_fn):
            raise ParallelPoolError("reward function must be callable")
        if (
            not isinstance(worker_timeout_seconds, (int, float))
            or isinstance(worker_timeout_seconds, bool)
            or not math.isfinite(float(worker_timeout_seconds))
            or worker_timeout_seconds <= 0.0
        ):
            raise ParallelPoolError("worker timeout must be finite and positive")
        self._environment_factory = environment_factory
        self._reward_fn = reward_fn
        self._worker_timeout_seconds = float(worker_timeout_seconds)
        self.rollout_discarded = False
        self.training_stopped = False
        self._closed = False
        self._reset = False
        self._buffer_index = 0
        self.worker_pids: list[int] = []
        self.worker_thread_limits: list[tuple[str, str]] = []

        self.shared_observation_buffers = _shared_observation_double_buffer(
            observation_template, self.worker_count
        )
        self.shared_action_buffers = tuple(
            {
                "candidate_indices": torch.empty(
                    (self.worker_count,), dtype=torch.int64
                ).share_memory_(),
                "thetas": torch.empty(
                    (self.worker_count,), dtype=torch.float32
                ).share_memory_(),
            }
            for _ in range(2)
        )
        self._shared_rewards = tuple(
            torch.empty((self.worker_count,), dtype=torch.float32).share_memory_()
            for _ in range(2)
        )
        self._shared_dones = tuple(
            torch.empty((self.worker_count,), dtype=torch.bool).share_memory_()
            for _ in range(2)
        )
        self._shared_policy_versions = tuple(
            torch.full((self.worker_count,), -1, dtype=torch.int64).share_memory_()
            for _ in range(2)
        )
        use_pinned_memory = torch.cuda.is_available()
        try:
            self._staging_observations = tuple(
                {
                    name: torch.empty_like(
                        tensor, device="cpu", pin_memory=use_pinned_memory
                    )
                    for name, tensor in shared.items()
                }
                for shared in self.shared_observation_buffers
            )
            self._staging_rewards = tuple(
                torch.empty_like(
                    tensor, device="cpu", pin_memory=use_pinned_memory
                )
                for tensor in self._shared_rewards
            )
            self._staging_dones = tuple(
                torch.empty_like(
                    tensor, device="cpu", pin_memory=use_pinned_memory
                )
                for tensor in self._shared_dones
            )
            self._staging_policy_versions = tuple(
                torch.empty_like(
                    tensor, device="cpu", pin_memory=use_pinned_memory
                )
                for tensor in self._shared_policy_versions
            )
        except Exception as error:
            raise ParallelPoolError("page-locked staging allocation failed") from error
        self.staging_buffers_are_pinned = use_pinned_memory and all(
            tensor.is_pinned()
            for buffer in self._staging_observations
            for tensor in buffer.values()
        )

        context = mp.get_context("spawn")
        self._result_queue = context.Queue()
        self._command_queues = [context.Queue() for _ in self._platforms]
        self._processes: list[mp.Process] = []
        try:
            for worker_index, platform_type in enumerate(self._platforms):
                process = context.Process(
                    target=_worker_main,
                    name=f"lunar-env-{worker_index}-{platform_type.lower()}",
                    args=(
                        worker_index,
                        platform_type,
                        self._environment_factory,
                        self._reward_fn,
                        self._command_queues[worker_index],
                        self._result_queue,
                        self.shared_observation_buffers,
                        self.shared_action_buffers,
                        self._shared_rewards,
                        self._shared_dones,
                        self._shared_policy_versions,
                    ),
                )
                process.start()
                self._processes.append(process)
            self._await_ready()
        except Exception as error:
            self.rollout_discarded = True
            self.training_stopped = True
            self._stop_workers()
            if isinstance(error, ParallelPoolError):
                raise
            raise ParallelPoolError("worker startup failed") from error

    def __enter__(self) -> "ParallelEnvPool":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def reset(self) -> ParallelRolloutStep:
        if self._closed or self.training_stopped:
            raise ParallelPoolError("parallel pool is stopped")
        if self._reset:
            raise ParallelPoolError("parallel pool reset is only valid once")
        self._reset = True
        self._buffer_index = 0
        self._shared_rewards[0].zero_()
        self._shared_dones[0].zero_()
        self._shared_policy_versions[0].fill_(-1)
        return self._stage_buffer(0)

    def step(
        self, actions: ParallelActions, *, policy_version: int
    ) -> ParallelRolloutStep:
        if self._closed or self.training_stopped:
            raise ParallelPoolError("parallel pool is stopped")
        if not self._reset:
            return self._fail_closed("parallel pool must be reset before step")
        try:
            _validate_actions(actions, self.worker_count)
            if type(policy_version) is not int or policy_version < 0:
                raise ParallelPoolError(
                    "policy version must be a non-negative integer"
                )
            target_buffer = 1 - self._buffer_index
            self.shared_action_buffers[target_buffer]["candidate_indices"].copy_(
                actions.candidate_indices
            )
            self.shared_action_buffers[target_buffer]["thetas"].copy_(actions.thetas)
            for command_queue in self._command_queues:
                command_queue.put(("step", target_buffer, policy_version))
            self._await_step(target_buffer, policy_version)
            self.validate_policy_versions(
                self._shared_policy_versions[target_buffer],
                expected_policy_version=policy_version,
            )
            self._buffer_index = target_buffer
            return self._stage_buffer(target_buffer)
        except ParallelPoolError as error:
            if not self.training_stopped:
                return self._fail_closed(str(error))
            raise
        except Exception as error:
            return self._fail_closed("shared rollout operation failed", cause=error)

    def validate_policy_versions(
        self,
        policy_versions: torch.Tensor,
        *,
        expected_policy_version: int,
    ) -> None:
        try:
            if (
                not isinstance(policy_versions, torch.Tensor)
                or policy_versions.dtype != torch.int64
                or policy_versions.ndim != 1
                or not bool(
                    torch.eq(policy_versions, expected_policy_version).all()
                )
            ):
                raise ParallelPoolError(
                    "rollout contains a mixed or stale policy version"
                )
            if policy_versions.shape != (self.worker_count,):
                raise ParallelPoolError("rollout policy version count is invalid")
        except ParallelPoolError as error:
            if not self.training_stopped:
                self._fail_closed(str(error))
            raise
        except Exception as error:
            self._fail_closed("rollout policy version validation failed", cause=error)

    def close(self) -> None:
        if self._closed:
            return
        self._stop_workers()

    def _await_ready(self) -> None:
        ready: dict[int, tuple[int, str, str]] = {}
        deadline = time.monotonic() + self._worker_timeout_seconds
        while len(ready) < self.worker_count:
            message = self._next_result(deadline)
            kind, worker_index, *values = message
            if kind == "error":
                raise ParallelPoolError(
                    f"worker {worker_index} startup failed: {values[0]}"
                )
            if kind != "ready" or worker_index in ready or len(values) != 3:
                raise ParallelPoolError("worker startup protocol failed")
            ready[worker_index] = (values[0], values[1], values[2])
        self.worker_pids = [ready[index][0] for index in range(self.worker_count)]
        self.worker_thread_limits = [
            (ready[index][1], ready[index][2])
            for index in range(self.worker_count)
        ]

    def _await_step(self, buffer_index: int, policy_version: int) -> None:
        completed: set[int] = set()
        deadline = time.monotonic() + self._worker_timeout_seconds
        while len(completed) < self.worker_count:
            message = self._next_result(deadline)
            kind, worker_index, *values = message
            if kind == "error":
                raise ParallelPoolError(
                    f"worker {worker_index} failed: {values[0]}"
                )
            if (
                kind != "step"
                or worker_index in completed
                or values != [buffer_index, policy_version]
            ):
                raise ParallelPoolError("worker step protocol failed")
            completed.add(worker_index)

    def _next_result(self, deadline: float) -> tuple[object, ...]:
        while True:
            dead = [
                index
                for index, process in enumerate(self._processes)
                if not process.is_alive() and process.exitcode is not None
            ]
            if dead:
                raise ParallelPoolError(
                    f"worker process exited unexpectedly: {dead}"
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise ParallelPoolError("worker response timed out")
            try:
                message = self._result_queue.get(timeout=min(remaining, 0.1))
            except queue.Empty:
                continue
            except Exception as error:
                raise ParallelPoolError("worker result queue failed") from error
            if not isinstance(message, tuple) or len(message) < 2:
                raise ParallelPoolError("worker returned malformed control data")
            return message

    def _stage_buffer(self, buffer_index: int) -> ParallelRolloutStep:
        try:
            for name, shared in self.shared_observation_buffers[buffer_index].items():
                self._staging_observations[buffer_index][name].copy_(shared)
            self._staging_rewards[buffer_index].copy_(
                self._shared_rewards[buffer_index]
            )
            self._staging_dones[buffer_index].copy_(self._shared_dones[buffer_index])
            self._staging_policy_versions[buffer_index].copy_(
                self._shared_policy_versions[buffer_index]
            )
            observations = PolicyBatch(
                **self._staging_observations[buffer_index]
            )
            validate_policy_batch(observations)
            if not bool(torch.isfinite(self._staging_rewards[buffer_index]).all()):
                raise ParallelPoolError("worker rewards must be finite")
            return ParallelRolloutStep(
                observations=observations,
                rewards=self._staging_rewards[buffer_index],
                dones=self._staging_dones[buffer_index],
                policy_versions=self._staging_policy_versions[buffer_index],
                buffer_index=buffer_index,
            )
        except ParallelPoolError:
            raise
        except Exception as error:
            raise ParallelPoolError("shared observation staging failed") from error

    def _fail_closed(
        self, message: str, *, cause: Exception | None = None
    ) -> ParallelRolloutStep:
        self.rollout_discarded = True
        self.training_stopped = True
        self._stop_workers()
        if cause is None:
            raise ParallelPoolError(message)
        raise ParallelPoolError(message) from cause

    def _stop_workers(self) -> None:
        if self._closed:
            return
        self._closed = True
        for index, process in enumerate(self._processes):
            if process.is_alive():
                try:
                    self._command_queues[index].put(("stop",))
                except Exception:
                    pass
        for process in self._processes:
            process.join(timeout=0.5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)


def joint_worker_allocation(total_workers: int) -> dict[str, int]:
    """Return the frozen eight-per-platform joint allocation."""
    if type(total_workers) is not int or total_workers != 24:
        raise ValueError("joint training requires exactly 24 workers")
    return {platform: 8 for platform in PLATFORMS}


def _worker_main(
    worker_index: int,
    platform_type: str,
    environment_factory: Callable[[int, str], ParallelEnvironmentWorker],
    reward_fn: Callable[[PlannerTransition], float],
    command_queue,
    result_queue,
    observation_buffers: tuple[dict[str, torch.Tensor], ...],
    action_buffers: tuple[dict[str, torch.Tensor], ...],
    reward_buffers: tuple[torch.Tensor, ...],
    done_buffers: tuple[torch.Tensor, ...],
    policy_version_buffers: tuple[torch.Tensor, ...],
) -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    try:
        worker = environment_factory(worker_index, platform_type)
        if not isinstance(worker, ParallelEnvironmentWorker):
            raise ParallelPoolError(
                "environment factory must return ParallelEnvironmentWorker"
            )
        _write_observation(
            observation_buffers[0], worker_index, worker.initial_observation
        )
        result_queue.put(
            (
                "ready",
                worker_index,
                os.getpid(),
                os.environ["OMP_NUM_THREADS"],
                os.environ["MKL_NUM_THREADS"],
            )
        )
        while True:
            command = command_queue.get()
            if command == ("stop",):
                return
            if (
                not isinstance(command, tuple)
                or len(command) != 3
                or command[0] != "step"
            ):
                raise ParallelPoolError("worker command protocol failed")
            _, buffer_index, policy_version = command
            action = PolicyAction(
                frontier_index=int(
                    action_buffers[buffer_index]["candidate_indices"][worker_index]
                ),
                theta_rad=float(action_buffers[buffer_index]["thetas"][worker_index]),
            )
            transition = worker.environment.step(action)
            if not isinstance(transition, PlannerTransition):
                raise ParallelPoolError(
                    "worker environment must return PlannerTransition"
                )
            reward = reward_fn(transition)
            if (
                not isinstance(reward, (int, float))
                or isinstance(reward, bool)
                or not math.isfinite(float(reward))
            ):
                raise ParallelPoolError("worker reward must be finite")
            _write_observation(
                observation_buffers[buffer_index],
                worker_index,
                transition.next_observation,
            )
            reward_buffers[buffer_index][worker_index] = float(reward)
            done_buffers[buffer_index][worker_index] = transition.terminated
            policy_version_buffers[buffer_index][worker_index] = policy_version
            result_queue.put(
                ("step", worker_index, buffer_index, policy_version)
            )
    except BaseException as error:
        try:
            result_queue.put(("error", worker_index, repr(error)))
        except BaseException:
            pass


def _shared_observation_double_buffer(
    template: PolicyBatch, worker_count: int
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    return tuple(
        {
            name: torch.empty(
                (worker_count, *getattr(template, name).shape[1:]),
                dtype=getattr(template, name).dtype,
            ).share_memory_()
            for name in _OBSERVATION_FIELDS
        }
        for _ in range(2)
    )


def _write_observation(
    buffer: dict[str, torch.Tensor],
    worker_index: int,
    observation: PolicyBatch,
) -> None:
    validate_policy_batch(observation)
    if observation.prior_channels.shape[0] != 1:
        raise ParallelPoolError("worker observation batch size must be one")
    for name in _OBSERVATION_FIELDS:
        source = getattr(observation, name)
        destination = buffer[name][worker_index]
        if source.device.type != "cpu" or source.shape[1:] != destination.shape:
            raise ParallelPoolError(
                f"worker observation shape/device mismatch for {name}"
            )
        destination.copy_(source[0])


def _expanded_platforms(allocation: Mapping[str, int]) -> list[str]:
    if not isinstance(allocation, Mapping) or not allocation:
        raise ParallelPoolError("worker allocation must be a non-empty mapping")
    unknown = set(allocation) - set(PLATFORMS)
    if unknown:
        raise ParallelPoolError("worker allocation contains unknown platform")
    platforms: list[str] = []
    for platform in PLATFORMS:
        count = allocation.get(platform, 0)
        if type(count) is not int or count < 0:
            raise ParallelPoolError("worker counts must be non-negative integers")
        platforms.extend([platform] * count)
    if not platforms:
        raise ParallelPoolError("worker allocation must create at least one worker")
    return platforms


def _validate_template(template: PolicyBatch) -> None:
    if not isinstance(template, PolicyBatch):
        raise ParallelPoolError("observation template must use PolicyBatch")
    try:
        validate_policy_batch(template)
    except ValueError as error:
        raise ParallelPoolError("observation template is invalid") from error
    if template.prior_channels.shape[0] != 1:
        raise ParallelPoolError("observation template batch size must be one")
    if template.prior_channels.device.type != "cpu":
        raise ParallelPoolError("observation template must be on CPU")


def _validate_actions(actions: ParallelActions, worker_count: int) -> None:
    if not isinstance(actions, ParallelActions):
        raise ParallelPoolError("parallel actions must use ParallelActions")
    if (
        not isinstance(actions.candidate_indices, torch.Tensor)
        or actions.candidate_indices.dtype != torch.int64
        or actions.candidate_indices.device.type != "cpu"
        or actions.candidate_indices.shape != (worker_count,)
    ):
        raise ParallelPoolError("candidate indices must be CPU int64 [workers]")
    if (
        not isinstance(actions.thetas, torch.Tensor)
        or actions.thetas.dtype != torch.float32
        or actions.thetas.device.type != "cpu"
        or actions.thetas.shape != (worker_count,)
    ):
        raise ParallelPoolError("action thetas must be CPU float32 [workers]")
    if not bool(torch.isfinite(actions.thetas).all()):
        raise ParallelPoolError("action thetas must be finite")


__all__ = [
    "ParallelActions",
    "ParallelEnvironmentWorker",
    "ParallelEnvPool",
    "ParallelPoolError",
    "ParallelRolloutStep",
    "joint_worker_allocation",
]

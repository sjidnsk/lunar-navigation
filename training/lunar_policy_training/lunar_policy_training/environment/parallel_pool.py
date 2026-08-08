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
from lunar_planner_training_bridge import PlanningOutcome
from lunar_model_contract import ObservationContractV3

from ..capability_freeze import FrozenPlatformCapability, ScenarioIdentity
from ..config import PLATFORMS
from ..policy.observation import (
    ObservationIdentity,
    PolicyBatch,
    validate_policy_batch,
)
from .macro_step import ExecutionEvents, PlannerTransition, PolicyAction


_OBSERVATION_FIELDS = ObservationContractV3.input_names


class ParallelPoolError(RuntimeError):
    """A worker or shared rollout invariant failed; training must stop."""


@dataclass(frozen=True, slots=True)
class ParallelEnvironmentWorker:
    """An environment created inside one worker plus its episode observation."""

    environment: object
    initial_observation: PolicyBatch


EnvironmentFactory = Callable[[int, str], ParallelEnvironmentWorker]
CapabilityEnvironmentBuilder = Callable[
    [int, str, FrozenPlatformCapability, ScenarioIdentity],
    ParallelEnvironmentWorker,
]


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
    policy_decisions_consumed: torch.Tensor
    buffer_index: int
    planning_outcomes: tuple[PlanningOutcome, ...] = ()
    reason_codes: tuple[str, ...] = ()
    execution_events: tuple[ExecutionEvents, ...] = ()


class ParallelEnvPool:
    """Own one spawned process/environment/bridge per synchronous worker."""

    def __init__(
        self,
        *,
        allocation: Mapping[str, int],
        observation_template: PolicyBatch,
        environment_factory: EnvironmentFactory,
        reward_fn: Callable[[PlannerTransition], float],
        worker_timeout_seconds: float = 30.0,
        auto_reset: bool = True,
        initial_episode_cursors: tuple[int, ...] | None = None,
    ) -> None:
        self._platforms = _expanded_platforms(allocation)
        self.worker_count = len(self._platforms)
        local_counts = {platform: 0 for platform in PLATFORMS}
        platform_worker_indices: list[int] = []
        for platform in self._platforms:
            platform_worker_indices.append(local_counts[platform])
            local_counts[platform] += 1
        self._platform_worker_indices = tuple(platform_worker_indices)
        self._platform_worker_counts = tuple(
            local_counts[platform] for platform in self._platforms
        )
        _validate_template(observation_template)
        if not callable(environment_factory):
            raise ParallelPoolError("environment factory must be callable")
        if not callable(reward_fn):
            raise ParallelPoolError("reward function must be callable")
        if type(auto_reset) is not bool:
            raise ParallelPoolError("auto_reset must be boolean")
        if (
            not isinstance(worker_timeout_seconds, (int, float))
            or isinstance(worker_timeout_seconds, bool)
            or not math.isfinite(float(worker_timeout_seconds))
            or worker_timeout_seconds <= 0.0
        ):
            raise ParallelPoolError("worker timeout must be finite and positive")
        if initial_episode_cursors is None:
            initial_episode_cursors = (0,) * self.worker_count
        if (
            not isinstance(initial_episode_cursors, tuple)
            or len(initial_episode_cursors) != self.worker_count
            or any(
                type(cursor) is not int or cursor < 0
                for cursor in initial_episode_cursors
            )
        ):
            raise ParallelPoolError(
                "initial episode cursors must contain one non-negative integer per worker"
            )
        self._environment_factory = environment_factory
        self._reward_fn = reward_fn
        self._auto_reset = auto_reset
        self._worker_timeout_seconds = float(worker_timeout_seconds)
        self.rollout_discarded = False
        self.training_stopped = False
        self._closed = False
        self._reset = False
        self._buffer_index = 0
        self._episode_cursors = initial_episode_cursors
        self.worker_pids: list[int] = []
        self.worker_thread_limits: list[tuple[str, str]] = []
        self._buffer_identities: list[tuple[ObservationIdentity, ...] | None] = [
            None,
            None,
        ]

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
                        self._auto_reset,
                        self._command_queues[worker_index],
                        self._result_queue,
                        self.shared_observation_buffers,
                        self.shared_action_buffers,
                        self._shared_rewards,
                        self._shared_dones,
                        self._shared_policy_versions,
                        initial_episode_cursors[worker_index],
                        self._platform_worker_indices[worker_index],
                        self._platform_worker_counts[worker_index],
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

    @property
    def episode_cursors(self) -> tuple[int, ...]:
        """Return the exact episode ordinal currently owned by every worker."""
        return self._episode_cursors

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
            source_identities = self._buffer_identities[self._buffer_index]
            if source_identities is None:
                raise ParallelPoolError("current observation identities are missing")
            for worker_index, command_queue in enumerate(self._command_queues):
                command_queue.put(
                    (
                        "step",
                        target_buffer,
                        policy_version,
                        source_identities[worker_index],
                    )
                )
            (
                planning_outcomes,
                reason_codes,
                execution_events,
                identities,
                policy_decisions_consumed,
            ) = self._await_step(
                target_buffer, policy_version
            )
            self._buffer_identities[target_buffer] = identities
            self.validate_policy_versions(
                self._shared_policy_versions[target_buffer],
                expected_policy_version=policy_version,
            )
            self._buffer_index = target_buffer
            return self._stage_buffer(
                target_buffer,
                planning_outcomes=planning_outcomes,
                reason_codes=reason_codes,
                execution_events=execution_events,
                policy_decisions_consumed=policy_decisions_consumed,
            )
        except ParallelPoolError as error:
            if not self.training_stopped:
                return self._fail_closed(str(error))
            raise
        except Exception as error:
            return self._fail_closed("shared rollout operation failed", cause=error)

    def prepare_decision_boundaries(
        self, *, policy_version: int
    ) -> ParallelRolloutStep:
        """Refresh every worker and resolve no-action task boundaries."""
        if self._closed or self.training_stopped:
            raise ParallelPoolError("parallel pool is stopped")
        if not self._reset:
            return self._fail_closed("parallel pool must be reset before resolution")
        try:
            if type(policy_version) is not int or policy_version < 0:
                raise ParallelPoolError(
                    "policy version must be a non-negative integer"
                )
            target_buffer = 1 - self._buffer_index
            for command_queue in self._command_queues:
                command_queue.put(
                    ("prepare_decision_boundary", target_buffer, policy_version)
                )
            identities = self._await_resolution(target_buffer, policy_version)
            self._buffer_identities[target_buffer] = identities
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
            return self._fail_closed(
                "decision-boundary preparation failed", cause=error
            )

    def reset_terminated_workers(
        self,
        worker_indices: tuple[int, ...],
        *,
        policy_version: int,
    ) -> ParallelRolloutStep:
        """Explicitly reset selected preserved-terminal evaluation workers."""
        if self._closed or self.training_stopped:
            raise ParallelPoolError("parallel pool is stopped")
        if not self._reset:
            return self._fail_closed("parallel pool must be reset before worker reset")
        try:
            if self._auto_reset:
                raise ParallelPoolError(
                    "explicit worker reset requires auto_reset=False"
                )
            if type(policy_version) is not int or policy_version < 0:
                raise ParallelPoolError(
                    "policy version must be a non-negative integer"
                )
            if (
                not isinstance(worker_indices, tuple)
                or not worker_indices
                or any(type(index) is not int for index in worker_indices)
                or len(set(worker_indices)) != len(worker_indices)
                or any(
                    index < 0 or index >= self.worker_count
                    for index in worker_indices
                )
            ):
                raise ParallelPoolError("worker reset indices are invalid")
            target_buffer = 1 - self._buffer_index
            for name in _OBSERVATION_FIELDS:
                self.shared_observation_buffers[target_buffer][name].copy_(
                    self.shared_observation_buffers[self._buffer_index][name]
                )
            self._shared_rewards[target_buffer].copy_(
                self._shared_rewards[self._buffer_index]
            )
            self._shared_dones[target_buffer].copy_(
                self._shared_dones[self._buffer_index]
            )
            self._shared_policy_versions[target_buffer].copy_(
                self._shared_policy_versions[self._buffer_index]
            )
            current_identities = self._buffer_identities[self._buffer_index]
            if current_identities is None:
                raise ParallelPoolError("current observation identities are missing")
            for worker_index in worker_indices:
                self._command_queues[worker_index].put(
                    ("reset_terminated", target_buffer, policy_version)
                )
            reset_rows = self._await_worker_resets(
                worker_indices,
                target_buffer,
                policy_version,
            )
            identities = list(current_identities)
            cursors = list(self._episode_cursors)
            for worker_index, (identity, cursor) in reset_rows.items():
                identities[worker_index] = identity
                cursors[worker_index] = cursor
            self._episode_cursors = tuple(cursors)
            self._buffer_identities[target_buffer] = tuple(identities)
            self._buffer_index = target_buffer
            return self._stage_buffer(target_buffer)
        except ParallelPoolError as error:
            if not self.training_stopped:
                return self._fail_closed(str(error))
            raise
        except Exception as error:
            return self._fail_closed("explicit worker reset failed", cause=error)

    def rollover_all_workers(self, *, policy_version: int) -> ParallelRolloutStep:
        """Replace every episode together at one completed optimizer boundary."""
        if self._closed or self.training_stopped:
            raise ParallelPoolError("parallel pool is stopped")
        if not self._reset:
            return self._fail_closed("parallel pool must be reset before rollover")
        try:
            if type(policy_version) is not int or policy_version < 0:
                raise ParallelPoolError(
                    "policy version must be a non-negative integer"
                )
            current_identities = self._buffer_identities[self._buffer_index]
            if current_identities is None:
                raise ParallelPoolError("current observation identities are missing")
            safe_states = {"DECISION_BOUNDARY", "GROUND_HOLD", "LANDED_HOLD"}
            if any(
                identity.execution_state not in safe_states
                for identity in current_identities
            ):
                raise ParallelPoolError(
                    "episode rollover requires completed decision-boundary observations"
                )
            target_buffer = 1 - self._buffer_index
            for command_queue in self._command_queues:
                command_queue.put(("rollover", target_buffer, policy_version))
            identities = self._await_rollovers(target_buffer, policy_version)
            self._buffer_identities[target_buffer] = identities
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
            return self._fail_closed("episode rollover failed", cause=error)

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
        ready: dict[int, tuple[int, str, str, int, ObservationIdentity]] = {}
        deadline = time.monotonic() + self._worker_timeout_seconds
        while len(ready) < self.worker_count:
            message = self._next_result(deadline)
            kind, worker_index, *values = message
            if kind == "error":
                raise ParallelPoolError(
                    f"worker {worker_index} startup failed: {values[0]}"
                )
            if kind != "ready" or worker_index in ready or len(values) != 5:
                raise ParallelPoolError("worker startup protocol failed")
            if (
                type(values[3]) is not int
                or values[3] < 0
                or not isinstance(values[4], ObservationIdentity)
            ):
                raise ParallelPoolError("worker observation identity is invalid")
            ready[worker_index] = (
                values[0], values[1], values[2], values[3], values[4]
            )
        self.worker_pids = [ready[index][0] for index in range(self.worker_count)]
        self.worker_thread_limits = [
            (ready[index][1], ready[index][2])
            for index in range(self.worker_count)
        ]
        self._episode_cursors = tuple(
            ready[index][3] for index in range(self.worker_count)
        )
        self._buffer_identities[0] = tuple(
            ready[index][4] for index in range(self.worker_count)
        )

    def _await_step(
        self, buffer_index: int, policy_version: int
    ) -> tuple[
        tuple[PlanningOutcome, ...],
        tuple[str, ...],
        tuple[ExecutionEvents, ...],
        tuple[ObservationIdentity, ...],
        torch.Tensor,
    ]:
        completed: set[int] = set()
        metadata: dict[
            int,
            tuple[
                PlanningOutcome,
                str,
                ExecutionEvents,
                ObservationIdentity,
                int,
                int,
            ],
        ] = {}
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
                or len(values) != 8
                or values[:2] != [buffer_index, policy_version]
            ):
                raise ParallelPoolError("worker step protocol failed")
            (
                outcome_value,
                reason_code,
                execution_events,
                identity,
                policy_decisions_consumed,
                episode_cursor,
            ) = values[2:]
            if (
                type(outcome_value) is not int
                or not isinstance(reason_code, str)
                or not isinstance(execution_events, ExecutionEvents)
                or not isinstance(identity, ObservationIdentity)
                or policy_decisions_consumed not in (0, 1)
                or type(episode_cursor) is not int
                or episode_cursor < 0
            ):
                raise ParallelPoolError("worker transition metadata is invalid")
            try:
                outcome = PlanningOutcome(outcome_value)
            except (TypeError, ValueError) as error:
                raise ParallelPoolError(
                    "worker planning outcome is invalid"
                ) from error
            metadata[worker_index] = (
                outcome,
                reason_code,
                execution_events,
                identity,
                policy_decisions_consumed,
                episode_cursor,
            )
            completed.add(worker_index)
        self._episode_cursors = tuple(
            metadata[index][5] for index in range(self.worker_count)
        )
        return (
            tuple(metadata[index][0] for index in range(self.worker_count)),
            tuple(metadata[index][1] for index in range(self.worker_count)),
            tuple(metadata[index][2] for index in range(self.worker_count)),
            tuple(metadata[index][3] for index in range(self.worker_count)),
            torch.tensor(
                [metadata[index][4] for index in range(self.worker_count)],
                dtype=torch.int64,
            ),
        )

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

    def _await_resolution(
        self, buffer_index: int, policy_version: int
    ) -> tuple[ObservationIdentity, ...]:
        completed: set[int] = set()
        identities: dict[int, ObservationIdentity] = {}
        cursors: dict[int, int] = {}
        deadline = time.monotonic() + self._worker_timeout_seconds
        while len(completed) < self.worker_count:
            message = self._next_result(deadline)
            kind, worker_index, *values = message
            if kind == "error":
                raise ParallelPoolError(
                    f"worker {worker_index} failed: {values[0]}"
                )
            if (
                kind != "resolved"
                or worker_index in completed
                or len(values) != 4
                or values[:2] != [buffer_index, policy_version]
                or not isinstance(values[2], ObservationIdentity)
                or type(values[3]) is not int
                or values[3] < 0
            ):
                raise ParallelPoolError("worker resolution protocol failed")
            identities[worker_index] = values[2]
            cursors[worker_index] = values[3]
            completed.add(worker_index)
        self._episode_cursors = tuple(
            cursors[index] for index in range(self.worker_count)
        )
        return tuple(identities[index] for index in range(self.worker_count))

    def _await_worker_resets(
        self,
        worker_indices: tuple[int, ...],
        buffer_index: int,
        policy_version: int,
    ) -> dict[int, tuple[ObservationIdentity, int]]:
        pending = set(worker_indices)
        rows: dict[int, tuple[ObservationIdentity, int]] = {}
        deadline = time.monotonic() + self._worker_timeout_seconds
        while pending:
            message = self._next_result(deadline)
            kind, worker_index, *values = message
            if kind == "error":
                raise ParallelPoolError(
                    f"worker {worker_index} failed: {values[0]}"
                )
            if (
                kind != "worker_reset"
                or worker_index not in pending
                or len(values) != 4
                or values[:2] != [buffer_index, policy_version]
                or not isinstance(values[2], ObservationIdentity)
                or type(values[3]) is not int
                or values[3] < 0
            ):
                raise ParallelPoolError("worker reset protocol failed")
            rows[worker_index] = (values[2], values[3])
            pending.remove(worker_index)
        return rows

    def _await_rollovers(
        self, buffer_index: int, policy_version: int
    ) -> tuple[ObservationIdentity, ...]:
        completed: set[int] = set()
        identities: dict[int, ObservationIdentity] = {}
        cursors: dict[int, int] = {}
        deadline = time.monotonic() + self._worker_timeout_seconds
        while len(completed) < self.worker_count:
            message = self._next_result(deadline)
            kind, worker_index, *values = message
            if kind == "error":
                raise ParallelPoolError(
                    f"worker {worker_index} failed: {values[0]}"
                )
            if (
                kind != "rollover"
                or worker_index in completed
                or len(values) != 4
                or values[:2] != [buffer_index, policy_version]
                or not isinstance(values[2], ObservationIdentity)
                or type(values[3]) is not int
                or values[3] < 0
            ):
                raise ParallelPoolError("worker rollover protocol failed")
            identities[worker_index] = values[2]
            cursors[worker_index] = values[3]
            completed.add(worker_index)
        self._episode_cursors = tuple(
            cursors[index] for index in range(self.worker_count)
        )
        return tuple(identities[index] for index in range(self.worker_count))

    def _stage_buffer(
        self,
        buffer_index: int,
        *,
        planning_outcomes: tuple[PlanningOutcome, ...] = (),
        reason_codes: tuple[str, ...] = (),
        execution_events: tuple[ExecutionEvents, ...] = (),
        policy_decisions_consumed: torch.Tensor | None = None,
    ) -> ParallelRolloutStep:
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
                **self._staging_observations[buffer_index],
                observation_identities=self._buffer_identities[buffer_index],
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
                policy_decisions_consumed=(
                    torch.zeros((self.worker_count,), dtype=torch.int64)
                    if policy_decisions_consumed is None
                    else policy_decisions_consumed
                ),
                planning_outcomes=planning_outcomes,
                reason_codes=reason_codes,
                execution_events=execution_events,
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


def _create_environment_for_episode(
    environment_factory: EnvironmentFactory,
    worker_index: int,
    platform_type: str,
    episode_cursor: int,
    platform_worker_index: int,
    platform_worker_count: int,
) -> ParallelEnvironmentWorker:
    create_for_episode = getattr(environment_factory, "create_for_episode", None)
    if callable(create_for_episode):
        worker = create_for_episode(
            worker_index,
            platform_type,
            episode_cursor,
            platform_worker_index=platform_worker_index,
            platform_worker_count=platform_worker_count,
        )
    else:
        worker = environment_factory(worker_index, platform_type)
    if not isinstance(worker, ParallelEnvironmentWorker):
        raise ParallelPoolError(
            "environment factory must return ParallelEnvironmentWorker"
        )
    return worker


def _worker_main(
    worker_index: int,
    platform_type: str,
    environment_factory: EnvironmentFactory,
    reward_fn: Callable[[PlannerTransition], float],
    auto_reset: bool,
    command_queue,
    result_queue,
    observation_buffers: tuple[dict[str, torch.Tensor], ...],
    action_buffers: tuple[dict[str, torch.Tensor], ...],
    reward_buffers: tuple[torch.Tensor, ...],
    done_buffers: tuple[torch.Tensor, ...],
    policy_version_buffers: tuple[torch.Tensor, ...],
    initial_episode_cursor: int,
    platform_worker_index: int,
    platform_worker_count: int,
) -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    try:
        if type(initial_episode_cursor) is not int or initial_episode_cursor < 0:
            raise ParallelPoolError("worker episode cursor is invalid")
        episode_cursor = initial_episode_cursor
        worker = _create_environment_for_episode(
            environment_factory,
            worker_index,
            platform_type,
            episode_cursor,
            platform_worker_index,
            platform_worker_count,
        )
        current_observation = _environment_current_observation(worker)
        _write_observation(observation_buffers[0], worker_index, current_observation)
        result_queue.put(
            (
                "ready",
                worker_index,
                os.getpid(),
                os.environ["OMP_NUM_THREADS"],
                os.environ["MKL_NUM_THREADS"],
                episode_cursor,
                current_observation.observation_identities[0],
            )
        )
        terminal_transition: PlannerTransition | None = None
        no_action_terminal: str | None = None
        while True:
            command = command_queue.get()
            if command == ("stop",):
                return
            if not isinstance(command, tuple) or not command:
                raise ParallelPoolError("worker command protocol failed")
            if command[0] == "rollover":
                if len(command) != 3:
                    raise ParallelPoolError("worker rollover command failed")
                _, buffer_index, policy_version = command
                episode_cursor += 1
                worker = _create_environment_for_episode(
                    environment_factory,
                    worker_index,
                    platform_type,
                    episode_cursor,
                    platform_worker_index,
                    platform_worker_count,
                )
                terminal_transition = None
                no_action_terminal = None
                current_observation = _environment_current_observation(worker)
                _write_observation(
                    observation_buffers[buffer_index],
                    worker_index,
                    current_observation,
                )
                reward_buffers[buffer_index][worker_index] = 0.0
                done_buffers[buffer_index][worker_index] = False
                policy_version_buffers[buffer_index][worker_index] = policy_version
                result_queue.put(
                    (
                        "rollover",
                        worker_index,
                        buffer_index,
                        policy_version,
                        current_observation.observation_identities[0],
                        episode_cursor,
                    )
                )
                continue
            if command[0] == "reset_terminated":
                if len(command) != 3 or auto_reset:
                    raise ParallelPoolError("worker reset command failed")
                _, buffer_index, policy_version = command
                if terminal_transition is None and no_action_terminal is None:
                    raise ParallelPoolError(
                        "only a terminated worker can be explicitly reset"
                    )
                episode_cursor += 1
                worker = _create_environment_for_episode(
                    environment_factory,
                    worker_index,
                    platform_type,
                    episode_cursor,
                    platform_worker_index,
                    platform_worker_count,
                )
                terminal_transition = None
                no_action_terminal = None
                current_observation = _environment_current_observation(worker)
                _write_observation(
                    observation_buffers[buffer_index],
                    worker_index,
                    current_observation,
                )
                reward_buffers[buffer_index][worker_index] = 0.0
                done_buffers[buffer_index][worker_index] = False
                policy_version_buffers[buffer_index][worker_index] = policy_version
                result_queue.put(
                    (
                        "worker_reset",
                        worker_index,
                        buffer_index,
                        policy_version,
                        current_observation.observation_identities[0],
                        episode_cursor,
                    )
                )
                continue
            if command[0] == "prepare_decision_boundary":
                if len(command) != 3:
                    raise ParallelPoolError("worker preparation command failed")
                _, buffer_index, policy_version = command
                if terminal_transition is not None or no_action_terminal is not None:
                    raise ParallelPoolError(
                        "terminated worker requires reset before preparation"
                    )
                boundary = worker.environment.refresh_decision_boundary()
                if (
                    boundary.transition is not None
                    or boundary.policy_decisions_consumed != 0
                ):
                    raise ParallelPoolError(
                        "boundary preparation must not create an action transition"
                    )
                terminal_boundary = boundary.execution_state == "NO_CANDIDATES"
                if boundary.execution_state not in {
                    "DECISION_READY",
                    "NO_CANDIDATES",
                }:
                    raise ParallelPoolError(
                        "worker returned invalid decision-boundary state"
                    )
                if terminal_boundary:
                    if auto_reset:
                        episode_cursor += 1
                        worker = _create_environment_for_episode(
                            environment_factory,
                            worker_index,
                            platform_type,
                            episode_cursor,
                            platform_worker_index,
                            platform_worker_count,
                        )
                    else:
                        no_action_terminal = boundary.execution_state
                current_observation = _environment_current_observation(worker)
                _write_observation(
                    observation_buffers[buffer_index],
                    worker_index,
                    current_observation,
                )
                reward_buffers[buffer_index][worker_index] = 0.0
                done_buffers[buffer_index][worker_index] = terminal_boundary
                policy_version_buffers[buffer_index][worker_index] = policy_version
                result_queue.put(
                    (
                        "resolved",
                        worker_index,
                        buffer_index,
                        policy_version,
                        current_observation.observation_identities[0],
                        episode_cursor,
                    )
                )
                continue
            if (
                len(command) != 4
                or command[0] != "step"
                or not isinstance(command[3], ObservationIdentity)
            ):
                raise ParallelPoolError("worker command protocol failed")
            _, buffer_index, policy_version, expected_identity = command
            action = PolicyAction(
                frontier_index=int(
                    action_buffers[buffer_index]["candidate_indices"][worker_index]
                ),
                theta_rad=float(action_buffers[buffer_index]["thetas"][worker_index]),
            )
            if terminal_transition is None and no_action_terminal is None:
                boundary = worker.environment.advance_prepared_action(
                    action,
                    expected_identity=expected_identity,
                )
                transition = boundary.transition
                if not isinstance(transition, PlannerTransition):
                    raise ParallelPoolError(
                        "worker environment must return PlannerTransition"
                    )
                if boundary.policy_decisions_consumed != 1:
                    raise ParallelPoolError(
                        "worker action must consume exactly one policy decision"
                    )
                reward = reward_fn(transition)
                policy_decisions_consumed = boundary.policy_decisions_consumed
            else:
                raise ParallelPoolError(
                    "terminated worker requires reset before another action"
                )
            if (
                not isinstance(reward, (int, float))
                or isinstance(reward, bool)
                or not math.isfinite(float(reward))
            ):
                raise ParallelPoolError("worker reward must be finite")
            next_observation = transition.next_observation
            if transition.terminated and auto_reset:
                episode_cursor += 1
                reset_worker = _create_environment_for_episode(
                    environment_factory,
                    worker_index,
                    platform_type,
                    episode_cursor,
                    platform_worker_index,
                    platform_worker_count,
                )
                worker = reset_worker
                next_observation = _environment_current_observation(worker)
            elif transition.terminated:
                terminal_transition = transition
            current_observation = next_observation
            _write_observation(
                observation_buffers[buffer_index],
                worker_index,
                next_observation,
            )
            reward_buffers[buffer_index][worker_index] = float(reward)
            done_buffers[buffer_index][worker_index] = transition.terminated
            policy_version_buffers[buffer_index][worker_index] = policy_version
            result_queue.put(
                (
                    "step",
                    worker_index,
                    buffer_index,
                    policy_version,
                    int(transition.planning_outcome),
                    transition.reason_code,
                    transition.execution_events,
                    next_observation.observation_identities[0],
                    policy_decisions_consumed,
                    episode_cursor,
                )
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
    if observation.observation_identities is None:
        raise ParallelPoolError(
            "worker observation requires a producer-owned identity"
        )
    for name in _OBSERVATION_FIELDS:
        source = getattr(observation, name)
        destination = buffer[name][worker_index]
        if source.device.type != "cpu" or source.shape[1:] != destination.shape:
            raise ParallelPoolError(
                f"worker observation shape/device mismatch for {name}"
            )
        destination.copy_(source[0])


def _environment_current_observation(
    worker: ParallelEnvironmentWorker,
) -> PolicyBatch:
    observation = getattr(worker.environment, "current_observation", None)
    if not isinstance(observation, PolicyBatch):
        raise ParallelPoolError(
            "worker environment must expose its private current observation"
        )
    return observation


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
    if template.observation_identities is None:
        raise ParallelPoolError(
            "observation template requires a producer-owned identity"
        )
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
    "CapabilityEnvironmentBuilder",
    "EnvironmentFactory",
    "ParallelActions",
    "ParallelEnvironmentWorker",
    "ParallelEnvPool",
    "ParallelPoolError",
    "ParallelRolloutStep",
    "joint_worker_allocation",
]

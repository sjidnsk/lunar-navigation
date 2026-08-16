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
from lunar_model_contract import ObservationContractV4

from ..capability_freeze import FrozenPlatformCapability, ScenarioIdentity
from ..config import PLATFORMS, WORKER_CANDIDATES
from ..policy.observation import (
    ObservationIdentity,
    PolicyBatch,
    validate_policy_batch,
)
from ..reward import (
    DEFAULT_REWARD_WEIGHTS,
    RewardComponentsV4,
    RewardInputsV4,
    compute_reward_components,
)
from ..reward_contract import (
    RewardStage,
    RewardTerminalClass,
    RewardWeightsV4,
)
from .macro_step import (
    ExecutionEvents,
    PlannerTransition,
    PolicyAction,
    TerminalAudit,
    TerminalReason,
)
from .formal_episode_state import FormalWorkerState
from .candidate_builder import CandidateDiagnostics


_OBSERVATION_FIELDS = ObservationContractV4.input_names


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
    success_first_crossings: torch.Tensor
    buffer_index: int
    planning_outcomes: tuple[PlanningOutcome, ...] = ()
    reason_codes: tuple[str, ...] = ()
    execution_events: tuple[ExecutionEvents, ...] = ()
    candidate_diagnostics: tuple[CandidateDiagnostics, ...] = ()
    no_candidate_terminations: tuple[bool, ...] = ()
    terminal_audits: tuple[TerminalAudit | None, ...] = ()


@dataclass(frozen=True, slots=True)
class InvalidTaskAudit:
    """One pre-policy task rejection that must never enter PPO tensors."""

    worker_index: int
    episode_id: str
    platform_type: str
    reason_code: str
    physical_snapshot_id: str
    post_worker_state: Mapping[str, object]

    def __post_init__(self) -> None:
        if type(self.worker_index) is not int or self.worker_index < 0:
            raise ValueError("invalid-task worker index is invalid")
        if not isinstance(self.episode_id, str) or not self.episode_id:
            raise ValueError("invalid-task episode identity is invalid")
        expected_reason = {
            "WHEELED": "WHEEL_START_NOT_SAFE",
            "LEGGED": "LEGGED_START_NOT_SAFE",
        }.get(self.platform_type)
        if self.reason_code not in {
            expected_reason,
            "INITIAL_OBSERVATION_ALREADY_SUCCESSFUL",
        }:
            raise ValueError("invalid-task reason is invalid")
        if (
            not isinstance(self.physical_snapshot_id, str)
            or len(self.physical_snapshot_id) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.physical_snapshot_id
            )
        ):
            raise ValueError("invalid-task physical snapshot is invalid")
        if not isinstance(self.post_worker_state, Mapping):
            raise TypeError("invalid-task worker state is invalid")


@dataclass(frozen=True, slots=True)
class PreparedWorkerBoundary:
    """One worker's stable pre-action boundary or no-action task terminal."""

    worker_index: int
    observation: PolicyBatch
    policy_version: int
    worker_state: Mapping[str, object]
    candidate_diagnostics: CandidateDiagnostics
    no_candidate_termination: bool
    terminal_audit: TerminalAudit | None
    buffer_index: int
    invalid_task_audit: InvalidTaskAudit | None = None

    @property
    def actionable(self) -> bool:
        return self.terminal_audit is None and self.invalid_task_audit is None


@dataclass(frozen=True, slots=True)
class CompletedWorkerTransition:
    """One complete macro action at a recoverable post-action boundary."""

    worker_index: int
    observation: PolicyBatch
    policy_version: int
    worker_state: Mapping[str, object]
    reward_components: RewardComponentsV4
    reward_inputs: RewardInputsV4
    terminal_class: RewardTerminalClass
    done: bool
    planning_outcome: PlanningOutcome
    reason_code: str
    execution_events: ExecutionEvents
    candidate_diagnostics: CandidateDiagnostics
    terminal_audit: TerminalAudit | None
    policy_decisions_consumed: int
    success_first_crossing: bool
    buffer_index: int


@dataclass(frozen=True, slots=True)
class RejectedWorkerAction:
    """One pre-motion planner rejection that must not enter PPO tensors."""

    worker_index: int
    observation: PolicyBatch
    policy_version: int
    worker_state: Mapping[str, object]
    reason_code: str
    candidate_diagnostics: CandidateDiagnostics
    buffer_index: int


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
        worker_startup_timeout_seconds: float | None = None,
        auto_reset: bool = True,
        initial_episode_cursors: tuple[int, ...] | None = None,
        initial_episode_states: (
            tuple[Mapping[str, object] | None, ...] | None
        ) = None,
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
        if worker_startup_timeout_seconds is None:
            worker_startup_timeout_seconds = float(worker_timeout_seconds)
        elif (
            not isinstance(worker_startup_timeout_seconds, (int, float))
            or isinstance(worker_startup_timeout_seconds, bool)
            or not math.isfinite(float(worker_startup_timeout_seconds))
            or worker_startup_timeout_seconds <= 0.0
        ):
            raise ParallelPoolError(
                "worker startup timeout must be finite and positive"
            )
        parsed_initial_states: (
            tuple[FormalWorkerState | None, ...] | None
        ) = None
        if initial_episode_states is not None:
            if (
                not isinstance(initial_episode_states, tuple)
                or len(initial_episode_states) != self.worker_count
            ):
                raise ParallelPoolError(
                    "initial episode states must contain one mapping per worker"
                )
            try:
                parsed_initial_states = tuple(
                    None
                    if value is None
                    else FormalWorkerState.from_dict(value)
                    for value in initial_episode_states
                )
            except ValueError as error:
                raise ParallelPoolError("initial episode state is invalid") from error
            for index, state in enumerate(parsed_initial_states):
                if state is None:
                    continue
                if (
                    state.worker_index != index
                    or state.platform_type != self._platforms[index]
                    or state.platform_worker_index
                    != self._platform_worker_indices[index]
                    or state.platform_worker_count
                    != self._platform_worker_counts[index]
                ):
                    raise ParallelPoolError(
                        "initial episode state worker identity differs"
                    )
            restored_cursors = tuple(
                0 if state is None else state.episode_cursor
                for state in parsed_initial_states
            )
            if initial_episode_cursors is None:
                initial_episode_cursors = restored_cursors
            elif initial_episode_cursors != restored_cursors:
                raise ParallelPoolError(
                    "initial episode cursors differ from active states"
                )
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
        inventory_provider = getattr(
            environment_factory, "task_inventory_count", None
        )
        if inventory_provider is None:
            self._invalid_task_inventory_counts = (None,) * self.worker_count
        elif callable(inventory_provider):
            inventory_counts = tuple(
                inventory_provider(platform) for platform in self._platforms
            )
            if any(
                type(count) is not int or count <= 0
                for count in inventory_counts
            ):
                raise ParallelPoolError("formal task inventory is invalid")
            self._invalid_task_inventory_counts = inventory_counts
        else:
            raise ParallelPoolError("formal task inventory provider is invalid")
        self._reward_fn = reward_fn
        self._auto_reset = auto_reset
        self._worker_timeout_seconds = float(worker_timeout_seconds)
        self._worker_startup_timeout_seconds = float(
            worker_startup_timeout_seconds
        )
        self.rollout_discarded = False
        self.training_stopped = False
        self._closed = False
        self._reset = False
        self._buffer_index = 0
        self._episode_cursors = initial_episode_cursors
        self._initial_episode_states = (
            None
            if parsed_initial_states is None
            else tuple(
                None if state is None else state.to_dict()
                for state in parsed_initial_states
            )
        )
        self.worker_pids: list[int] = []
        self.worker_thread_limits: list[tuple[str, str]] = []
        self._buffer_identities: list[tuple[ObservationIdentity, ...] | None] = [
            None,
            None,
        ]
        self._async_started = False
        self._async_policy_version: int | None = None
        self._async_in_flight: dict[int, str] = {}
        self._async_buffer_indices = [0] * self.worker_count
        self._async_identities: list[ObservationIdentity] = []
        self._async_terminal_workers = {
            index
            for index, state in enumerate(parsed_initial_states or ())
            if state is not None and state.terminal_reason is not None
        }
        self._async_consecutive_invalid_tasks = [0] * self.worker_count

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
                        (
                            None
                            if self._initial_episode_states is None
                            else self._initial_episode_states[worker_index]
                        ),
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

    @property
    def terminal_worker_indices(self) -> tuple[int, ...]:
        """Return workers that require a reset before another macro action."""
        return tuple(sorted(self._async_terminal_workers))

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
        identities = self._buffer_identities[0]
        if identities is None:
            raise ParallelPoolError("initial observation identities are missing")
        self._async_started = False
        self._async_policy_version = None
        self._async_in_flight.clear()
        self._async_buffer_indices = [0] * self.worker_count
        self._async_identities = list(identities)
        return self._stage_buffer(0)

    def snapshot_episode_states(
        self, *, policy_version: int
    ) -> tuple[dict[str, object], ...]:
        """Capture every active formal worker at one stable decision boundary."""
        if self._closed or self.training_stopped:
            raise ParallelPoolError("parallel pool is stopped")
        if not self._reset:
            raise ParallelPoolError(
                "parallel pool must be reset before episode snapshot"
            )
        if type(policy_version) is not int or policy_version < 0:
            raise ParallelPoolError(
                "policy version must be a non-negative integer"
            )
        if self._async_started:
            if self._async_in_flight:
                raise ParallelPoolError(
                    "episode snapshot cannot contain in-flight workers"
                )
            identities = tuple(self._async_identities)
        else:
            identities = self._buffer_identities[self._buffer_index]
            if identities is None:
                raise ParallelPoolError(
                    "current observation identities are missing"
                )
        if any(
            identity.execution_state
            not in {"DECISION_BOUNDARY", "GROUND_HOLD", "LANDED_HOLD"}
            for identity in identities
        ):
            raise ParallelPoolError(
                "episode snapshot requires stable observation identities"
            )
        try:
            for index, command_queue in enumerate(self._command_queues):
                command_queue.put(
                    (
                        "snapshot_episode_state",
                        policy_version,
                        identities[index],
                    )
                )
            return self._await_episode_states(policy_version)
        except ParallelPoolError as error:
            if not self.training_stopped:
                self._fail_closed(str(error))
            raise
        except Exception as error:
            return self._fail_closed(
                "episode snapshot failed", cause=error
            )

    def prepare_workers(
        self,
        worker_indices: tuple[int, ...],
        *,
        policy_version: int,
        reset: bool = False,
    ) -> None:
        """Dispatch independent decision-boundary preparation commands."""
        if self._closed or self.training_stopped:
            raise ParallelPoolError("parallel pool is stopped")
        try:
            self._require_async_mode(policy_version)
            indices = _validate_worker_indices(
                worker_indices, worker_count=self.worker_count
            )
            if type(reset) is not bool:
                raise ParallelPoolError("worker reset flag must be boolean")
            for worker_index in indices:
                if worker_index in self._async_in_flight:
                    raise ParallelPoolError("worker already has an in-flight command")
                if reset != (worker_index in self._async_terminal_workers):
                    raise ParallelPoolError(
                        "worker reset flag differs from preserved terminal state"
                    )
            for worker_index in indices:
                target_buffer = 1 - self._async_buffer_indices[worker_index]
                self._command_queues[worker_index].put(
                    (
                        "async_prepare",
                        target_buffer,
                        policy_version,
                        reset,
                    )
                )
                self._async_in_flight[worker_index] = "async_prepared"
        except ParallelPoolError as error:
            if not self.training_stopped:
                self._fail_closed(str(error))
            raise
        except Exception as error:
            self._fail_closed("worker preparation dispatch failed", cause=error)

    def submit_actions(
        self,
        actions: Mapping[int, PolicyAction],
        *,
        policy_version: int,
        reward_stage: RewardStage,
        reward_weights: RewardWeightsV4 = DEFAULT_REWARD_WEIGHTS,
    ) -> None:
        """Dispatch actions only to the ready workers named by the caller."""
        if self._closed or self.training_stopped:
            raise ParallelPoolError("parallel pool is stopped")
        try:
            self._require_async_mode(policy_version)
            if not isinstance(actions, Mapping) or not actions:
                raise ParallelPoolError("asynchronous worker actions are invalid")
            if not isinstance(reward_stage, RewardStage):
                raise ParallelPoolError("asynchronous reward stage is invalid")
            if not isinstance(reward_weights, RewardWeightsV4):
                raise ParallelPoolError("asynchronous reward weights are invalid")
            indices = _validate_worker_indices(
                tuple(actions), worker_count=self.worker_count
            )
            for worker_index in indices:
                if worker_index in self._async_in_flight:
                    raise ParallelPoolError("worker already has an in-flight command")
                action = actions[worker_index]
                if not isinstance(action, PolicyAction):
                    raise ParallelPoolError("asynchronous worker action is invalid")
            for worker_index in indices:
                action = actions[worker_index]
                target_buffer = 1 - self._async_buffer_indices[worker_index]
                self.shared_action_buffers[target_buffer]["candidate_indices"][
                    worker_index
                ] = action.frontier_index
                self.shared_action_buffers[target_buffer]["thetas"][
                    worker_index
                ] = action.theta_rad
                self._command_queues[worker_index].put(
                    (
                        "async_step",
                        target_buffer,
                        policy_version,
                        self._async_identities[worker_index],
                        reward_stage.value,
                        reward_weights,
                    )
                )
                self._async_in_flight[worker_index] = "async_step"
        except ParallelPoolError as error:
            if not self.training_stopped:
                self._fail_closed(str(error))
            raise
        except Exception as error:
            self._fail_closed("worker action dispatch failed", cause=error)

    def await_completed_workers(
        self,
        *,
        timeout_seconds: float | None = None,
    ) -> tuple[
        PreparedWorkerBoundary
        | CompletedWorkerTransition
        | RejectedWorkerAction,
        ...,
    ]:
        """Return as soon as at least one independently dispatched worker completes."""
        if self._closed or self.training_stopped:
            raise ParallelPoolError("parallel pool is stopped")
        try:
            if not self._async_in_flight:
                raise ParallelPoolError("no asynchronous worker command is in flight")
            timeout = (
                self._worker_timeout_seconds
                if timeout_seconds is None
                else timeout_seconds
            )
            if (
                not isinstance(timeout, (int, float))
                or isinstance(timeout, bool)
                or not math.isfinite(float(timeout))
                or timeout <= 0.0
            ):
                raise ParallelPoolError("asynchronous wait timeout is invalid")
            deadline = time.monotonic() + float(timeout)
            messages = [self._next_result(deadline)]
            while True:
                try:
                    message = self._result_queue.get_nowait()
                except queue.Empty:
                    break
                if not isinstance(message, tuple) or len(message) < 2:
                    raise ParallelPoolError(
                        "worker returned malformed asynchronous data"
                    )
                messages.append(message)
            return tuple(self._consume_async_result(message) for message in messages)
        except ParallelPoolError as error:
            if not self.training_stopped:
                self._fail_closed(str(error))
            raise
        except Exception as error:
            self._fail_closed("asynchronous worker wait failed", cause=error)

    def complete_policy_update(self, *, policy_version: int) -> None:
        """Release the fixed policy version only after every worker slot is committed."""
        if type(policy_version) is not int or policy_version < 0:
            raise ParallelPoolError("policy version must be a non-negative integer")
        if self._async_policy_version != policy_version:
            raise ParallelPoolError("asynchronous policy version differs")
        if self._async_in_flight:
            raise ParallelPoolError("cannot complete update with in-flight workers")
        self._async_policy_version = None

    def current_worker_observations(
        self, worker_indices: tuple[int, ...]
    ) -> tuple[PolicyBatch, ...]:
        """Clone stable post-action observations without refreshing workers."""
        if self._closed or self.training_stopped:
            raise ParallelPoolError("parallel pool is stopped")
        indices = _validate_worker_indices(
            worker_indices, worker_count=self.worker_count
        )
        if any(worker in self._async_in_flight for worker in indices):
            raise ParallelPoolError("worker observation is still in flight")
        if not self._reset:
            raise ParallelPoolError("parallel pool must be reset before bootstrap")
        return tuple(
            self._stage_worker_observation(
                self._async_buffer_indices[worker],
                worker,
                self._async_identities[worker],
            )
            for worker in indices
        )

    def _require_async_mode(self, policy_version: int) -> None:
        if not self._reset:
            raise ParallelPoolError("parallel pool must be reset before collection")
        if self._auto_reset:
            raise ParallelPoolError(
                "independent collection requires explicit worker reset"
            )
        if type(policy_version) is not int or policy_version < 0:
            raise ParallelPoolError("policy version must be a non-negative integer")
        if not self._async_started:
            identities = self._buffer_identities[self._buffer_index]
            if identities is None:
                raise ParallelPoolError("current observation identities are missing")
            self._async_identities = list(identities)
            self._async_buffer_indices = [self._buffer_index] * self.worker_count
            self._async_started = True
        if self._async_policy_version is None:
            self._async_policy_version = policy_version
        elif self._async_policy_version != policy_version:
            raise ParallelPoolError("asynchronous rollout mixed policy versions")

    def _consume_async_result(
        self, message: tuple[object, ...]
    ) -> (
        PreparedWorkerBoundary
        | CompletedWorkerTransition
        | RejectedWorkerAction
    ):
        kind, worker_index, *values = message
        if kind == "error":
            raise ParallelPoolError(f"worker {worker_index} failed: {values[0]}")
        if type(worker_index) is not int or worker_index not in self._async_in_flight:
            raise ParallelPoolError("asynchronous worker identity is invalid")
        expected_kind = self._async_in_flight[worker_index]
        if kind != expected_kind and not (
            expected_kind == "async_step" and kind == "async_rejected"
        ):
            raise ParallelPoolError("asynchronous worker result kind differs")
        if kind == "async_prepared":
            result = self._parse_async_prepared(worker_index, values)
        elif kind == "async_step":
            result = self._parse_async_step(worker_index, values)
        elif kind == "async_rejected":
            result = self._parse_async_rejected(worker_index, values)
        else:
            raise ParallelPoolError("asynchronous worker result kind is invalid")
        del self._async_in_flight[worker_index]
        if isinstance(result, PreparedWorkerBoundary):
            if result.invalid_task_audit is None:
                self._async_consecutive_invalid_tasks[worker_index] = 0
            else:
                count = self._async_consecutive_invalid_tasks[worker_index] + 1
                self._async_consecutive_invalid_tasks[worker_index] = count
                inventory_count = self._invalid_task_inventory_counts[
                    worker_index
                ]
                if inventory_count is not None and count >= inventory_count:
                    raise ParallelPoolError(
                        "FORMAL_INVALID_TASK_INVENTORY_EXHAUSTED"
                    )
        self._async_buffer_indices[worker_index] = result.buffer_index
        identity = result.observation.observation_identities
        if identity is None or len(identity) != 1:
            raise ParallelPoolError("asynchronous observation identity is missing")
        self._async_identities[worker_index] = identity[0]
        if (
            isinstance(result, CompletedWorkerTransition) and result.done
        ) or (
            isinstance(result, PreparedWorkerBoundary) and not result.actionable
        ):
            self._async_terminal_workers.add(worker_index)
        else:
            self._async_terminal_workers.discard(worker_index)
        return result

    def _parse_async_rejected(
        self, worker_index: int, values: list[object]
    ) -> RejectedWorkerAction:
        if (
            len(values) != 7
            or type(values[0]) is not int
            or values[1] != self._async_policy_version
            or not isinstance(values[2], str)
            or not values[2]
            or not isinstance(values[3], CandidateDiagnostics)
            or not isinstance(values[4], ObservationIdentity)
            or type(values[5]) is not int
            or values[5] < 0
            or not isinstance(values[6], Mapping)
        ):
            raise ParallelPoolError(
                "asynchronous rejected action result is invalid"
            )
        buffer_index = int(values[0])
        state = self._validated_async_state(
            values[6],
            worker_index=worker_index,
            episode_cursor=int(values[5]),
            identity=values[4],
        )
        self._set_async_cursor(worker_index, int(values[5]))
        return RejectedWorkerAction(
            worker_index=worker_index,
            observation=self._stage_worker_observation(
                buffer_index, worker_index, values[4]
            ),
            policy_version=int(values[1]),
            worker_state=state,
            reason_code=values[2],
            candidate_diagnostics=values[3],
            buffer_index=buffer_index,
        )

    def _parse_async_prepared(
        self, worker_index: int, values: list[object]
    ) -> PreparedWorkerBoundary:
        if (
            len(values) != 10
            or type(values[0]) is not int
            or values[1] != self._async_policy_version
            or not isinstance(values[2], CandidateDiagnostics)
            or type(values[3]) is not bool
            or type(values[4]) is not bool
            or (values[5] is not None and not isinstance(values[5], TerminalAudit))
            or (
                values[6] is not None
                and not isinstance(values[6], InvalidTaskAudit)
            )
            or values[3] != (values[5] is not None)
            or (values[4] and not values[3])
            or (values[5] is not None and values[6] is not None)
            or not isinstance(values[7], ObservationIdentity)
            or type(values[8]) is not int
            or values[8] < 0
            or not isinstance(values[9], Mapping)
        ):
            raise ParallelPoolError("asynchronous preparation result is invalid")
        buffer_index = int(values[0])
        state = self._validated_async_state(
            values[9],
            worker_index=worker_index,
            episode_cursor=int(values[8]),
            identity=values[7],
        )
        invalid_task_audit = values[6]
        if invalid_task_audit is not None and (
            invalid_task_audit.worker_index != worker_index
            or invalid_task_audit.episode_id != values[7].episode_id
            or invalid_task_audit.platform_type != self._platforms[worker_index]
            or invalid_task_audit.physical_snapshot_id
            != values[2].physical_snapshot_id
            or dict(invalid_task_audit.post_worker_state) != dict(state)
        ):
            raise ParallelPoolError("invalid-task audit identity differs")
        self._set_async_cursor(worker_index, int(values[8]))
        return PreparedWorkerBoundary(
            worker_index=worker_index,
            observation=self._stage_worker_observation(
                buffer_index, worker_index, values[7]
            ),
            policy_version=int(values[1]),
            worker_state=state,
            candidate_diagnostics=values[2],
            no_candidate_termination=bool(values[4]),
            terminal_audit=values[5],
            buffer_index=buffer_index,
            invalid_task_audit=invalid_task_audit,
        )

    def _parse_async_step(
        self, worker_index: int, values: list[object]
    ) -> CompletedWorkerTransition:
        if (
            len(values) != 15
            or type(values[0]) is not int
            or values[1] != self._async_policy_version
            or type(values[2]) is not int
            or not isinstance(values[3], str)
            or not isinstance(values[4], ExecutionEvents)
            or not isinstance(values[5], CandidateDiagnostics)
            or (values[6] is not None and not isinstance(values[6], TerminalAudit))
            or not isinstance(values[7], ObservationIdentity)
            or values[8] != 1
            or type(values[9]) is not bool
            or type(values[10]) is not int
            or values[10] < 0
            or not isinstance(values[11], RewardComponentsV4)
            or not isinstance(values[12], RewardInputsV4)
            or not isinstance(values[13], RewardTerminalClass)
            or not isinstance(values[14], Mapping)
        ):
            raise ParallelPoolError("asynchronous step result is invalid")
        try:
            outcome = PlanningOutcome(values[2])
        except (TypeError, ValueError) as error:
            raise ParallelPoolError("asynchronous planning outcome is invalid") from error
        buffer_index = int(values[0])
        done = bool(self._shared_dones[buffer_index][worker_index])
        if done != (values[6] is not None):
            raise ParallelPoolError("asynchronous terminal metadata disagrees")
        reward = float(self._shared_rewards[buffer_index][worker_index])
        if not math.isfinite(reward) or not math.isclose(
            reward,
            values[11].total,
            rel_tol=1.0e-6,
            abs_tol=1.0e-6,
        ):
            raise ParallelPoolError("asynchronous reward components disagree")
        state = self._validated_async_state(
            values[14],
            worker_index=worker_index,
            episode_cursor=int(values[10]),
            identity=values[7],
        )
        self._set_async_cursor(worker_index, int(values[10]))
        return CompletedWorkerTransition(
            worker_index=worker_index,
            observation=self._stage_worker_observation(
                buffer_index, worker_index, values[7]
            ),
            policy_version=int(values[1]),
            worker_state=state,
            reward_components=values[11],
            reward_inputs=values[12],
            terminal_class=values[13],
            done=done,
            planning_outcome=outcome,
            reason_code=values[3],
            execution_events=values[4],
            candidate_diagnostics=values[5],
            terminal_audit=values[6],
            policy_decisions_consumed=int(values[8]),
            success_first_crossing=bool(values[9]),
            buffer_index=buffer_index,
        )

    def _validated_async_state(
        self,
        value: object,
        *,
        worker_index: int,
        episode_cursor: int,
        identity: ObservationIdentity,
    ) -> dict[str, object]:
        try:
            state = FormalWorkerState.from_dict(value)
        except ValueError as error:
            raise ParallelPoolError("asynchronous worker state is invalid") from error
        if (
            state.worker_index != worker_index
            or state.episode_cursor != episode_cursor
            or state.observation_identity != identity
        ):
            raise ParallelPoolError("asynchronous worker state identity differs")
        return state.to_dict()

    def _set_async_cursor(self, worker_index: int, cursor: int) -> None:
        cursors = list(self._episode_cursors)
        cursors[worker_index] = cursor
        self._episode_cursors = tuple(cursors)

    def _stage_worker_observation(
        self,
        buffer_index: int,
        worker_index: int,
        identity: ObservationIdentity,
    ) -> PolicyBatch:
        if buffer_index not in (0, 1):
            raise ParallelPoolError("asynchronous buffer index is invalid")
        observation = PolicyBatch(
            **{
                name: tensor[worker_index].detach().clone().unsqueeze(0)
                for name, tensor in self.shared_observation_buffers[
                    buffer_index
                ].items()
            },
            observation_identities=(identity,),
        )
        validate_policy_batch(observation)
        return observation

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
                candidate_diagnostics,
                terminal_audits,
                identities,
                policy_decisions_consumed,
                success_first_crossings,
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
                candidate_diagnostics=candidate_diagnostics,
                terminal_audits=terminal_audits,
                policy_decisions_consumed=policy_decisions_consumed,
                success_first_crossings=success_first_crossings,
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
            (
                identities,
                candidate_diagnostics,
                no_candidate_terminations,
                terminal_audits,
            ) = self._await_resolution(target_buffer, policy_version)
            self._buffer_identities[target_buffer] = identities
            self.validate_policy_versions(
                self._shared_policy_versions[target_buffer],
                expected_policy_version=policy_version,
            )
            self._buffer_index = target_buffer
            return self._stage_buffer(
                target_buffer,
                candidate_diagnostics=candidate_diagnostics,
                no_candidate_terminations=no_candidate_terminations,
                terminal_audits=terminal_audits,
            )
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
        deadline = time.monotonic() + self._worker_startup_timeout_seconds
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
        tuple[CandidateDiagnostics, ...],
        tuple[TerminalAudit | None, ...],
        tuple[ObservationIdentity, ...],
        torch.Tensor,
        torch.Tensor,
    ]:
        completed: set[int] = set()
        metadata: dict[
            int,
            tuple[
                PlanningOutcome,
                str,
                ExecutionEvents,
                CandidateDiagnostics,
                TerminalAudit | None,
                ObservationIdentity,
                int,
                bool,
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
                or len(values) != 11
                or values[:2] != [buffer_index, policy_version]
            ):
                raise ParallelPoolError("worker step protocol failed")
            (
                outcome_value,
                reason_code,
                execution_events,
                candidate_diagnostics,
                terminal_audit,
                identity,
                policy_decisions_consumed,
                success_first_crossing,
                episode_cursor,
            ) = values[2:]
            if (
                type(outcome_value) is not int
                or not isinstance(reason_code, str)
                or not isinstance(execution_events, ExecutionEvents)
                or not isinstance(candidate_diagnostics, CandidateDiagnostics)
                or (
                    terminal_audit is not None
                    and not isinstance(terminal_audit, TerminalAudit)
                )
                or not isinstance(identity, ObservationIdentity)
                or policy_decisions_consumed not in (0, 1)
                or type(success_first_crossing) is not bool
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
                candidate_diagnostics,
                terminal_audit,
                identity,
                policy_decisions_consumed,
                success_first_crossing,
                episode_cursor,
            )
            completed.add(worker_index)
        self._episode_cursors = tuple(
            metadata[index][8] for index in range(self.worker_count)
        )
        return (
            tuple(metadata[index][0] for index in range(self.worker_count)),
            tuple(metadata[index][1] for index in range(self.worker_count)),
            tuple(metadata[index][2] for index in range(self.worker_count)),
            tuple(metadata[index][3] for index in range(self.worker_count)),
            tuple(metadata[index][4] for index in range(self.worker_count)),
            tuple(metadata[index][5] for index in range(self.worker_count)),
            torch.tensor(
                [metadata[index][6] for index in range(self.worker_count)],
                dtype=torch.int64,
            ),
            torch.tensor(
                [metadata[index][7] for index in range(self.worker_count)],
                dtype=torch.bool,
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
    ) -> tuple[
        tuple[ObservationIdentity, ...],
        tuple[CandidateDiagnostics, ...],
        tuple[bool, ...],
        tuple[TerminalAudit | None, ...],
    ]:
        completed: set[int] = set()
        identities: dict[int, ObservationIdentity] = {}
        diagnostics: dict[int, CandidateDiagnostics] = {}
        no_candidate_terminations: dict[int, bool] = {}
        terminal_audits: dict[int, TerminalAudit | None] = {}
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
                or len(values) != 8
                or values[:2] != [buffer_index, policy_version]
                or not isinstance(values[2], CandidateDiagnostics)
                or type(values[3]) is not bool
                or type(values[4]) is not bool
                or (
                    values[5] is not None
                    and not isinstance(values[5], TerminalAudit)
                )
                or values[3] != (values[5] is not None)
                or (values[4] and not values[3])
                or not isinstance(values[6], ObservationIdentity)
                or type(values[7]) is not int
                or values[7] < 0
            ):
                raise ParallelPoolError("worker resolution protocol failed")
            diagnostics[worker_index] = values[2]
            no_candidate_terminations[worker_index] = values[4]
            terminal_audits[worker_index] = values[5]
            identities[worker_index] = values[6]
            cursors[worker_index] = values[7]
            completed.add(worker_index)
        self._episode_cursors = tuple(
            cursors[index] for index in range(self.worker_count)
        )
        return (
            tuple(identities[index] for index in range(self.worker_count)),
            tuple(diagnostics[index] for index in range(self.worker_count)),
            tuple(
                no_candidate_terminations[index]
                for index in range(self.worker_count)
            ),
            tuple(
                terminal_audits[index] for index in range(self.worker_count)
            ),
        )

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

    def _await_episode_states(
        self, policy_version: int
    ) -> tuple[dict[str, object], ...]:
        states: dict[int, dict[str, object]] = {}
        deadline = time.monotonic() + self._worker_timeout_seconds
        while len(states) < self.worker_count:
            message = self._next_result(deadline)
            kind, worker_index, *values = message
            if kind == "error":
                raise ParallelPoolError(
                    f"worker {worker_index} failed: {values[0]}"
                )
            if (
                kind != "episode_state"
                or worker_index in states
                or len(values) != 2
                or values[0] != policy_version
            ):
                raise ParallelPoolError("worker episode snapshot protocol failed")
            try:
                parsed = FormalWorkerState.from_dict(values[1])
            except ValueError as error:
                raise ParallelPoolError(
                    "worker episode snapshot state is invalid"
                ) from error
            if (
                parsed.worker_index != worker_index
                or parsed.episode_cursor != self._episode_cursors[worker_index]
            ):
                raise ParallelPoolError(
                    "worker episode snapshot identity differs"
                )
            states[worker_index] = parsed.to_dict()
        return tuple(states[index] for index in range(self.worker_count))

    def _stage_buffer(
        self,
        buffer_index: int,
        *,
        planning_outcomes: tuple[PlanningOutcome, ...] = (),
        reason_codes: tuple[str, ...] = (),
        execution_events: tuple[ExecutionEvents, ...] = (),
        candidate_diagnostics: tuple[CandidateDiagnostics, ...] = (),
        no_candidate_terminations: tuple[bool, ...] = (),
        terminal_audits: tuple[TerminalAudit | None, ...] = (),
        policy_decisions_consumed: torch.Tensor | None = None,
        success_first_crossings: torch.Tensor | None = None,
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
                success_first_crossings=(
                    torch.zeros((self.worker_count,), dtype=torch.bool)
                    if success_first_crossings is None
                    else success_first_crossings
                ),
                planning_outcomes=planning_outcomes,
                reason_codes=reason_codes,
                execution_events=execution_events,
                candidate_diagnostics=candidate_diagnostics,
                no_candidate_terminations=no_candidate_terminations,
                terminal_audits=terminal_audits,
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
    """Balance one calibrated worker tier equally across all platforms."""
    if type(total_workers) is not int or total_workers not in WORKER_CANDIDATES:
        raise ValueError("joint training requires a calibrated worker tier")
    per_platform = total_workers // len(PLATFORMS)
    return {platform: per_platform for platform in PLATFORMS}


def _create_environment_for_episode(
    environment_factory: EnvironmentFactory,
    worker_index: int,
    platform_type: str,
    episode_cursor: int,
    platform_worker_index: int,
    platform_worker_count: int,
    restore_state: Mapping[str, object] | None = None,
) -> ParallelEnvironmentWorker:
    restore_for_episode = getattr(environment_factory, "restore_for_episode", None)
    create_for_episode = getattr(environment_factory, "create_for_episode", None)
    if restore_state is not None:
        if not callable(restore_for_episode):
            raise ParallelPoolError(
                "environment factory does not support active episode restore"
            )
        worker = restore_for_episode(
            worker_index=worker_index,
            platform_type=platform_type,
            episode_cursor=episode_cursor,
            platform_worker_index=platform_worker_index,
            platform_worker_count=platform_worker_count,
            state=restore_state,
        )
    elif callable(create_for_episode):
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


def _current_candidate_diagnostics(
    worker: ParallelEnvironmentWorker,
) -> CandidateDiagnostics:
    getter = getattr(worker, "current_candidate_diagnostics", None)
    if getter is None:
        return CandidateDiagnostics()
    if not callable(getter):
        raise ParallelPoolError("worker candidate diagnostics accessor is invalid")
    diagnostics = getter()
    if not isinstance(diagnostics, CandidateDiagnostics):
        raise ParallelPoolError("worker candidate diagnostics are invalid")
    return diagnostics


def _snapshot_formal_worker_state(
    worker: ParallelEnvironmentWorker,
    *,
    worker_index: int,
    platform_type: str,
    episode_cursor: int,
    identity: ObservationIdentity,
) -> dict[str, object]:
    snapshot = getattr(worker, "snapshot_episode_state", None)
    if not callable(snapshot):
        raise ParallelPoolError(
            "worker does not support stable episode snapshots"
        )
    try:
        state = FormalWorkerState.from_dict(snapshot())
    except ValueError as error:
        raise ParallelPoolError("worker episode snapshot state is invalid") from error
    if (
        state.worker_index != worker_index
        or state.platform_type != platform_type
        or state.episode_cursor != episode_cursor
        or state.observation_identity != identity
    ):
        raise ParallelPoolError("worker episode snapshot identity differs")
    return state.to_dict()


def _terminal_audit(
    *,
    terminated: bool,
    terminal_reason: TerminalReason | None,
    remaining_coverable_detail_cell_count: int | None,
    candidate_diagnostics: CandidateDiagnostics,
) -> TerminalAudit | None:
    if type(terminated) is not bool:
        raise ParallelPoolError("worker terminal flag is invalid")
    if not terminated:
        if terminal_reason is not None:
            raise ParallelPoolError("nonterminal worker has a terminal reason")
        return None
    if not isinstance(terminal_reason, TerminalReason):
        raise ParallelPoolError("terminated worker has no terminal reason")
    try:
        return TerminalAudit(
            reason=terminal_reason,
            candidate_diagnostics=candidate_diagnostics,
            remaining_coverable_detail_cell_count=(
                remaining_coverable_detail_cell_count
            ),
        )
    except ValueError as error:
        raise ParallelPoolError("worker terminal audit is invalid") from error


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
    initial_episode_state: Mapping[str, object] | None,
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
            initial_episode_state,
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
        restored_terminal = (
            initial_episode_state is not None
            and FormalWorkerState.from_dict(initial_episode_state).terminal_reason
            is not None
        )
        no_action_terminal: str | None = (
            "RESTORED_TERMINAL" if restored_terminal else None
        )
        while True:
            command = command_queue.get()
            if command == ("stop",):
                return
            if not isinstance(command, tuple) or not command:
                raise ParallelPoolError("worker command protocol failed")
            if command[0] == "async_prepare":
                if (
                    len(command) != 4
                    or type(command[1]) is not int
                    or type(command[2]) is not int
                    or command[2] < 0
                    or type(command[3]) is not bool
                ):
                    raise ParallelPoolError(
                        "asynchronous worker preparation command failed"
                    )
                _, buffer_index, policy_version, reset_worker = command
                if reset_worker:
                    if terminal_transition is None and no_action_terminal is None:
                        raise ParallelPoolError(
                            "only a terminated worker can be reset"
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
                elif terminal_transition is not None or no_action_terminal is not None:
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
                terminal_boundary = boundary.terminal_reason is not None
                invalid_task = boundary.task_resample_required
                no_candidate_termination = (
                    boundary.execution_state == "NO_CANDIDATES"
                )
                if boundary.execution_state not in {
                    "DECISION_READY",
                    "NO_CANDIDATES",
                    "TERMINATED",
                    "TASK_RESAMPLE_REQUIRED",
                }:
                    raise ParallelPoolError(
                        "worker returned invalid decision-boundary state"
                    )
                candidate_diagnostics = _current_candidate_diagnostics(worker)
                terminal_audit = _terminal_audit(
                    terminated=terminal_boundary,
                    terminal_reason=boundary.terminal_reason,
                    remaining_coverable_detail_cell_count=(
                        boundary.remaining_coverable_detail_cell_count
                    ),
                    candidate_diagnostics=candidate_diagnostics,
                )
                if invalid_task != (
                    boundary.execution_state == "TASK_RESAMPLE_REQUIRED"
                ) or invalid_task != isinstance(
                    boundary.task_resample_reason, str
                ):
                    raise ParallelPoolError(
                        "worker task-resample boundary is invalid"
                    )
                if terminal_boundary or invalid_task:
                    no_action_terminal = boundary.execution_state
                current_observation = _environment_current_observation(worker)
                identity = current_observation.observation_identities[0]
                state = _snapshot_formal_worker_state(
                    worker,
                    worker_index=worker_index,
                    platform_type=platform_type,
                    episode_cursor=episode_cursor,
                    identity=identity,
                )
                invalid_task_audit = (
                    InvalidTaskAudit(
                        worker_index=worker_index,
                        episode_id=identity.episode_id,
                        platform_type=platform_type,
                        reason_code=str(boundary.task_resample_reason),
                        physical_snapshot_id=(
                            candidate_diagnostics.physical_snapshot_id
                        ),
                        post_worker_state=state,
                    )
                    if invalid_task
                    else None
                )
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
                        "async_prepared",
                        worker_index,
                        buffer_index,
                        policy_version,
                        candidate_diagnostics,
                        terminal_boundary,
                        no_candidate_termination,
                        terminal_audit,
                        invalid_task_audit,
                        identity,
                        episode_cursor,
                        state,
                    )
                )
                continue
            if command[0] == "async_step":
                if (
                    len(command) != 6
                    or type(command[1]) is not int
                    or type(command[2]) is not int
                    or command[2] < 0
                    or not isinstance(command[3], ObservationIdentity)
                    or not isinstance(command[5], RewardWeightsV4)
                ):
                    raise ParallelPoolError(
                        "asynchronous worker step command failed"
                    )
                (
                    _,
                    buffer_index,
                    policy_version,
                    expected_identity,
                    reward_stage_value,
                    reward_weights,
                ) = command
                try:
                    reward_stage = RewardStage(reward_stage_value)
                except (TypeError, ValueError) as error:
                    raise ParallelPoolError(
                        "asynchronous worker reward stage is invalid"
                    ) from error
                if terminal_transition is not None or no_action_terminal is not None:
                    raise ParallelPoolError(
                        "terminated worker requires reset before another action"
                    )
                action = PolicyAction(
                    frontier_index=int(
                        action_buffers[buffer_index]["candidate_indices"][
                            worker_index
                        ]
                    ),
                    theta_rad=float(
                        action_buffers[buffer_index]["thetas"][worker_index]
                    ),
                )
                candidate_diagnostics = _current_candidate_diagnostics(worker)
                boundary = worker.environment.advance_prepared_action(
                    action,
                    expected_identity=expected_identity,
                )
                transition = boundary.transition
                if boundary.reselection_required:
                    if (
                        transition is not None
                        or boundary.policy_decisions_consumed != 1
                        or not isinstance(boundary.reselection_reason, str)
                        or not boundary.reselection_reason
                        or boundary.terminal_reason is not None
                        or boundary.task_resample_required
                    ):
                        raise ParallelPoolError(
                            "worker reselection boundary is invalid"
                        )
                    current_observation = _environment_current_observation(
                        worker
                    )
                    current_diagnostics = _current_candidate_diagnostics(
                        worker
                    )
                    identity = current_observation.observation_identities[0]
                    state = _snapshot_formal_worker_state(
                        worker,
                        worker_index=worker_index,
                        platform_type=platform_type,
                        episode_cursor=episode_cursor,
                        identity=identity,
                    )
                    _write_observation(
                        observation_buffers[buffer_index],
                        worker_index,
                        current_observation,
                    )
                    reward_buffers[buffer_index][worker_index] = 0.0
                    done_buffers[buffer_index][worker_index] = False
                    policy_version_buffers[buffer_index][worker_index] = (
                        policy_version
                    )
                    result_queue.put(
                        (
                            "async_rejected",
                            worker_index,
                            buffer_index,
                            policy_version,
                            boundary.reselection_reason,
                            current_diagnostics,
                            identity,
                            episode_cursor,
                            state,
                        )
                    )
                    continue
                if not isinstance(transition, PlannerTransition):
                    raise ParallelPoolError(
                        "worker environment must return PlannerTransition"
                    )
                if boundary.policy_decisions_consumed != 1:
                    raise ParallelPoolError(
                        "worker action must consume exactly one policy decision"
                    )
                reward_inputs = RewardInputsV4.from_transition(
                    transition, platform_type=platform_type
                )
                components = compute_reward_components(
                    reward_inputs,
                    reward_stage,
                    reward_weights,
                )
                current_diagnostics = _current_candidate_diagnostics(worker)
                terminal_audit = _terminal_audit(
                    terminated=transition.terminated,
                    terminal_reason=transition.terminal_reason,
                    remaining_coverable_detail_cell_count=(
                        transition.remaining_coverable_detail_cell_count
                    ),
                    candidate_diagnostics=(
                        current_diagnostics
                        if transition.terminated
                        else candidate_diagnostics
                    ),
                )
                current_observation = transition.next_observation
                if transition.terminated:
                    terminal_transition = transition
                identity = current_observation.observation_identities[0]
                state = _snapshot_formal_worker_state(
                    worker,
                    worker_index=worker_index,
                    platform_type=platform_type,
                    episode_cursor=episode_cursor,
                    identity=identity,
                )
                _write_observation(
                    observation_buffers[buffer_index],
                    worker_index,
                    current_observation,
                )
                reward_buffers[buffer_index][worker_index] = components.total
                done_buffers[buffer_index][worker_index] = transition.terminated
                policy_version_buffers[buffer_index][worker_index] = policy_version
                result_queue.put(
                    (
                        "async_step",
                        worker_index,
                        buffer_index,
                        policy_version,
                        int(transition.planning_outcome),
                        transition.reason_code,
                        transition.execution_events,
                        candidate_diagnostics,
                        terminal_audit,
                        identity,
                        boundary.policy_decisions_consumed,
                        transition.success_first_crossing,
                        episode_cursor,
                        components,
                        reward_inputs,
                        transition.reward_terminal_class,
                        state,
                    )
                )
                continue
            if command[0] == "snapshot_episode_state":
                if (
                    len(command) != 3
                    or type(command[1]) is not int
                    or command[1] < 0
                    or not isinstance(command[2], ObservationIdentity)
                ):
                    raise ParallelPoolError(
                        "worker episode snapshot command failed"
                    )
                _, policy_version, expected_identity = command
                current_observation = _environment_current_observation(worker)
                if (
                    current_observation.observation_identities[0]
                    != expected_identity
                ):
                    raise ParallelPoolError(
                        "worker episode snapshot observation is stale"
                    )
                snapshot = getattr(worker, "snapshot_episode_state", None)
                if not callable(snapshot):
                    raise ParallelPoolError(
                        "worker does not support active episode snapshots"
                    )
                state = FormalWorkerState.from_dict(snapshot())
                if (
                    state.worker_index != worker_index
                    or state.platform_type != platform_type
                    or state.episode_cursor != episode_cursor
                    or state.observation_identity != expected_identity
                ):
                    raise ParallelPoolError(
                        "worker episode snapshot identity differs"
                    )
                result_queue.put(
                    (
                        "episode_state",
                        worker_index,
                        policy_version,
                        state.to_dict(),
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
                terminal_boundary = boundary.terminal_reason is not None
                no_candidate_termination = (
                    boundary.execution_state == "NO_CANDIDATES"
                )
                if boundary.execution_state not in {
                    "DECISION_READY",
                    "NO_CANDIDATES",
                    "TERMINATED",
                }:
                    raise ParallelPoolError(
                        "worker returned invalid decision-boundary state"
                    )
                candidate_diagnostics = _current_candidate_diagnostics(worker)
                terminal_audit = _terminal_audit(
                    terminated=terminal_boundary,
                    terminal_reason=boundary.terminal_reason,
                    remaining_coverable_detail_cell_count=(
                        boundary.remaining_coverable_detail_cell_count
                    ),
                    candidate_diagnostics=candidate_diagnostics,
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
                        candidate_diagnostics,
                        terminal_boundary,
                        no_candidate_termination,
                        terminal_audit,
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
                candidate_diagnostics = _current_candidate_diagnostics(worker)
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
                current_diagnostics = _current_candidate_diagnostics(worker)
                terminal_audit = _terminal_audit(
                    terminated=transition.terminated,
                    terminal_reason=transition.terminal_reason,
                    remaining_coverable_detail_cell_count=(
                        transition.remaining_coverable_detail_cell_count
                    ),
                    candidate_diagnostics=(
                        current_diagnostics
                        if transition.terminated
                        else candidate_diagnostics
                    ),
                )
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
                    candidate_diagnostics,
                    terminal_audit,
                    next_observation.observation_identities[0],
                    policy_decisions_consumed,
                    transition.success_first_crossing,
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


def _validate_worker_indices(
    worker_indices: tuple[int, ...], *, worker_count: int
) -> tuple[int, ...]:
    if (
        not isinstance(worker_indices, tuple)
        or not worker_indices
        or any(type(index) is not int for index in worker_indices)
        or len(set(worker_indices)) != len(worker_indices)
        or any(index < 0 or index >= worker_count for index in worker_indices)
    ):
        raise ParallelPoolError("asynchronous worker indices are invalid")
    return worker_indices


__all__ = [
    "CapabilityEnvironmentBuilder",
    "CompletedWorkerTransition",
    "EnvironmentFactory",
    "InvalidTaskAudit",
    "ParallelActions",
    "ParallelEnvironmentWorker",
    "ParallelEnvPool",
    "ParallelPoolError",
    "ParallelRolloutStep",
    "PreparedWorkerBoundary",
    "RejectedWorkerAction",
    "joint_worker_allocation",
]

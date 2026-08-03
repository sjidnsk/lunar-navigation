"""Stage 4 真正 spawn 的 8-env rollout collector。"""

from __future__ import annotations
from typing import TYPE_CHECKING

import json
import math
import multiprocessing as mp
import os
import pickle
import sys
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from multiprocessing.connection import Connection
from types import MappingProxyType
from typing import Any, Final

import numpy as np
import torch
from torch import nn

if TYPE_CHECKING:
    from typing import Any as EnvAction
    from typing import Any as StepResult
from ..policy.cross_attention import PolicyForwardOutput, batch_policy_observations, sample_action
from ..policy.observation import PolicyObservation
from .rollout import ROLLOUT_ENV_COUNT, ROLLOUT_STEPS_PER_ENV, CandidateSnapshot, RolloutBatch, RolloutBuffer, RolloutContractError, RolloutTransition
from .trainer import policy_state_sha256
if TYPE_CHECKING:
    from typing import Any as ArtifactStore
if TYPE_CHECKING:
    from typing import Any as CellXY


COLLECTOR_SCHEMA_VERSION: Final = "stage4_spawn_collector/v2"
VECTOR_STATE_SCHEMA_VERSION: Final = "stage4_spawn_vector_worker_state/v1"
_MAX_DIAGNOSTIC_RESETS_PER_ENV: Final = 1024
PLANNER_FAILURE_REASONS: Final = (
    "endpoint_physical_unsafe",
    "endpoint_unknown_buffer_unsafe",
    "path_physical_unsafe",
    "path_unknown_buffer_unsafe",
    "planner_no_path",
)


def _process_is_alive(process: mp.Process) -> bool:
    try:
        return bool(process.is_alive())
    except BaseException:
        return False


def _process_exitcode(process: mp.Process) -> int | None:
    try:
        return process.exitcode
    except BaseException:
        return None


class WorkerProcessError(RuntimeError):
    """Spawn worker exception、EOF、timeout 或协议漂移。"""


@dataclass(frozen=True, slots=True)
class _SpawnCleanupIssue:
    worker_index: int | None
    resource: str
    operation: str
    error_type: str
    message: str

    def render(self) -> str:
        worker = "none" if self.worker_index is None else str(self.worker_index)
        return (
            f"{self.resource}[worker={worker}].{self.operation}="
            f"{self.error_type}: {self.message}"
        )


class SpawnCleanupError(WorkerProcessError):
    """Spawn vector 无法确认全部 worker 与 IPC/进程 handle 已清理。"""

    def __init__(
        self,
        *,
        alive_workers: Sequence[tuple[int, int | None, int | None]],
        open_child_endpoints: Sequence[int],
        open_parent_endpoints: Sequence[int],
        open_process_handles: Sequence[int],
        issues: Sequence[_SpawnCleanupIssue],
    ) -> None:
        self.alive_workers = tuple(alive_workers)
        self.open_child_endpoints = tuple(open_child_endpoints)
        self.open_parent_endpoints = tuple(open_parent_endpoints)
        self.open_process_handles = tuple(open_process_handles)
        self.issues = tuple(issues)
        alive = ", ".join(
            f"worker={worker},pid={pid},exitcode={exitcode}"
            for worker, pid, exitcode in self.alive_workers
        )
        rendered_issues = ", ".join(issue.render() for issue in self.issues)
        super().__init__(
            "SpawnVectorEnv cleanup incomplete: "
            f"alive_workers=[{alive}]; "
            f"open_child_endpoints={list(self.open_child_endpoints)}; "
            f"open_parent_endpoints={list(self.open_parent_endpoints)}; "
            f"open_process_handles={list(self.open_process_handles)}; "
            f"errors=[{rendered_issues}]"
        )


class CollectorError(RuntimeError):
    """Rollout collection 未满足固定 Stage 4 合同。"""


@dataclass(frozen=True, slots=True)
class SpawnEnvSpec:
    """可 pickle 的 top-level factory 与 JSON kwargs。"""

    factory: Callable[..., object]
    kwargs: Mapping[str, object]

    def __post_init__(self) -> None:
        if not callable(self.factory):
            raise ValueError("spawn env factory must be callable")
        try:
            kwargs = json.loads(
                json.dumps(
                    self.kwargs,
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                )
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("spawn env kwargs must be finite JSON") from exc
        if not isinstance(kwargs, dict):
            raise ValueError("spawn env kwargs must be a JSON object")
        object.__setattr__(self, "kwargs", MappingProxyType(kwargs))
        try:
            pickle.dumps(self)
        except Exception as exc:
            raise ValueError("spawn env spec must be serializable") from exc

    def __reduce__(self) -> tuple[object, tuple[object, dict[str, object]]]:
        return type(self), (self.factory, dict(self.kwargs))


@dataclass(frozen=True, slots=True)
class WorkerObservation:
    observation: PolicyObservation
    frontier_cells: tuple[CellXY, ...]
    top_m_selection_summary: Mapping[str, object]
    needs_policy: bool
    reset_trainable: bool = False
    fake_logprob_created: bool = False
    reset_coverage_rate: float = 0.0
    reset_reason: str = "none"
    reset_done: bool = False
    reset_diagnostics: Mapping[str, object] | None = None


@dataclass(frozen=True, slots=True)
class WorkerStep:
    result: StepResult
    next_observation: WorkerObservation


@dataclass(frozen=True, slots=True)
class CollectorContract:
    config_sha256: str
    lineage: Mapping[str, object]
    frontier_extractor_version: str
    observation_schema_version: str
    action_space_version: str
    reward_version: str
    planner_version: str

    def __post_init__(self) -> None:
        if not _is_sha256(self.config_sha256):
            raise ValueError("collector config hash is invalid")
        lineage = _copy_json_mapping(self.lineage, "collector lineage")
        object.__setattr__(self, "lineage", MappingProxyType(lineage))
        for name in (
            "frontier_extractor_version",
            "observation_schema_version",
            "action_space_version",
            "reward_version",
            "planner_version",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"collector {name} is invalid")


@dataclass(frozen=True, slots=True)
class CollectorAudit:
    schema_version: str
    worker_pids: tuple[int, ...]
    worker_start_methods: tuple[str, ...]
    per_env_trainable_counts: tuple[int, ...]
    diagnostic_reset_counts: tuple[int, ...]
    trainable_transition_count: int
    inference_pids: tuple[int, ...]
    inference_batch_count: int
    policy_device: str
    policy_state_sha256: str
    snapshot_sha256: tuple[str, ...]
    terminal_transition_count: int
    reset_diagnostics: tuple[Mapping[str, object], ...] = ()
    planner_failure_counts: Mapping[str, int] = field(
        default_factory=lambda: {
            reason: 0 for reason in PLANNER_FAILURE_REASONS
        }
    )


@dataclass(frozen=True, slots=True)
class CollectionResult:
    batch: RolloutBatch
    audit: CollectorAudit
    vector_env_states: tuple[dict[str, object], ...]


@dataclass(frozen=True, slots=True)
class CollectorPreflightResult:
    schema_version: str
    transitions: tuple[RolloutTransition, ...]
    trainable_transition_count: int
    worker_pids: tuple[int, ...]
    worker_start_methods: tuple[str, ...]
    diagnostic_reset_counts: tuple[int, ...]
    inference_pids: tuple[int, ...]
    inference_batch_count: int
    policy_device: str
    policy_state_sha256: str
    snapshot_sha256: tuple[str, ...]
    terminal_transition_count: int
    timings_seconds: Mapping[str, float]
    reset_diagnostics: tuple[Mapping[str, object], ...] = ()
    planner_failure_counts: Mapping[str, int] = field(
        default_factory=lambda: {
            reason: 0 for reason in PLANNER_FAILURE_REASONS
        }
    )


class SpawnVectorEnv:
    """八个显式 spawn worker；环境、planner、sensor 全在 child。"""

    def __init__(
        self,
        specs: Sequence[SpawnEnvSpec],
        *,
        timeout_seconds: float = 60.0,
    ) -> None:
        if len(specs) != ROLLOUT_ENV_COUNT or any(
            not isinstance(spec, SpawnEnvSpec) for spec in specs
        ):
            raise ValueError("Stage 4 vector env requires exactly 8 spawn specs")
        if (
            not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(float(timeout_seconds))
            or timeout_seconds <= 0.0
        ):
            raise ValueError("worker timeout must be finite and positive")
        self.timeout_seconds = float(timeout_seconds)
        self._context = mp.get_context("spawn")
        self._connections: list[Connection] = []
        self._child_connections: list[Connection] = []
        self._processes: list[mp.Process] = []
        self._worker_pids: tuple[int, ...] = ()
        self._worker_start_methods: tuple[str, ...] = ()
        self._worker_alive_after_close: tuple[bool, ...] | None = None
        self._worker_exitcodes_after_close: tuple[int | None, ...] | None = None
        self._worker_cleanup_issues_after_close: tuple[_SpawnCleanupIssue, ...] = ()
        self._closed_child_endpoint_indices: set[int] = set()
        self._closed_parent_endpoint_indices: set[int] = set()
        self._closed_process_handle_indices: set[int] = set()
        self._closed = False
        self._broken = False
        try:
            for worker_index, spec in enumerate(specs):
                parent, child = self._context.Pipe(duplex=True)
                self._connections.append(parent)
                self._child_connections.append(child)
                process = self._context.Process(
                    target=_worker_main,
                    args=(worker_index, child, spec),
                    name=f"stage4-spawn-env-{worker_index}",
                    daemon=False,
                )
                self._processes.append(process)
                process.start()
                child.close()
                self._closed_child_endpoint_indices.add(worker_index)
            ready = tuple(
                self._receive(
                    worker_index,
                    timeout_seconds=max(30.0, self.timeout_seconds),
                )
                for worker_index in range(ROLLOUT_ENV_COUNT)
            )
            if any(
                not isinstance(value, dict)
                or set(value) != {"kind", "worker_index", "pid", "start_method"}
                or value["kind"] != "ready"
                or value["worker_index"] != index
                or value["start_method"] != "spawn"
                for index, value in enumerate(ready)
            ):
                raise WorkerProcessError("spawn worker ready protocol mismatch")
            self._worker_pids = tuple(int(value["pid"]) for value in ready)
            self._worker_start_methods = tuple(
                str(value["start_method"]) for value in ready
            )
            if len(set(self._worker_pids)) != ROLLOUT_ENV_COUNT:
                raise WorkerProcessError("spawn workers do not have distinct PIDs")
        except BaseException:
            try:
                self.close()
            except BaseException:
                pass
            raise

    @property
    def worker_pids(self) -> tuple[int, ...]:
        return self._worker_pids

    @property
    def worker_start_methods(self) -> tuple[str, ...]:
        return self._worker_start_methods

    @property
    def worker_alive(self) -> tuple[bool, ...]:
        if self._worker_alive_after_close is not None:
            return self._worker_alive_after_close
        return tuple(process.is_alive() for process in self._processes)

    @property
    def worker_exitcodes(self) -> tuple[int | None, ...]:
        if self._worker_exitcodes_after_close is not None:
            return self._worker_exitcodes_after_close
        return tuple(process.exitcode for process in self._processes)

    @property
    def closed(self) -> bool:
        return self._closed

    def reset(
        self,
        worker_indices: Sequence[int] | None = None,
    ) -> dict[int, WorkerObservation]:
        indices = self._validated_indices(worker_indices)
        values = self._round_trip(
            {index: ("reset", None) for index in indices}
        )
        if any(not isinstance(value, WorkerObservation) for value in values.values()):
            self._fail("worker reset response type mismatch")
        return values  # type: ignore[return-value]

    def step(self, actions: Mapping[int, EnvAction]) -> dict[int, WorkerStep]:
        if not isinstance(actions, Mapping) or not actions:
            raise ValueError("vector step requires worker actions")
        indices = self._validated_indices(tuple(actions))
        if any(not isinstance(actions[index], EnvAction) for index in indices):
            raise ValueError("vector step actions must be EnvAction values")
        values = self._round_trip(
            {index: ("step", actions[index]) for index in indices}
        )
        if any(not isinstance(value, WorkerStep) for value in values.values()):
            self._fail("worker step response type mismatch")
        return values  # type: ignore[return-value]

    def current_observations(self) -> dict[int, WorkerObservation]:
        """读取已恢复 worker 的当前 policy observation，不触发 reset。"""

        values = self._round_trip(
            {
                index: ("current_observation", None)
                for index in range(ROLLOUT_ENV_COUNT)
            }
        )
        if any(not isinstance(value, WorkerObservation) for value in values.values()):
            self._fail("worker current observation response type mismatch")
        return values  # type: ignore[return-value]

    def capture_states(self) -> tuple[dict[str, object], ...]:
        values = self._round_trip(
            {index: ("capture_state", None) for index in range(ROLLOUT_ENV_COUNT)}
        )
        ordered: list[dict[str, object]] = []
        for index in range(ROLLOUT_ENV_COUNT):
            value = values[index]
            if not isinstance(value, dict) or set(value) != {
                "episode_state",
                "sampler_state",
            }:
                self._fail(f"worker {index} state response mismatch")
            ordered.append(_copy_json_mapping(value, "worker vector state"))
        return tuple(ordered)

    def restore_states(
        self,
        states: Sequence[Mapping[str, object]],
    ) -> None:
        if len(states) != ROLLOUT_ENV_COUNT:
            raise ValueError("vector restore requires exactly 8 worker states")
        payloads = {
            index: (
                "restore_state",
                _copy_json_mapping(states[index], "worker vector state"),
            )
            for index in range(ROLLOUT_ENV_COUNT)
        }
        values = self._round_trip(payloads)
        if any(value != "restored" for value in values.values()):
            self._fail("worker restore acknowledgement mismatch")

    def close(self) -> None:
        if self._closed:
            return
        primary_exception = sys.exception()
        issues: list[_SpawnCleanupIssue] = []
        fatal_issues: list[_SpawnCleanupIssue] = []

        def record_issue(
            worker_index: int | None,
            resource: str,
            operation: str,
            exc: BaseException,
            *,
            fatal: bool = True,
        ) -> None:
            message = str(exc).replace("\r", "\\r").replace("\n", "\\n")
            issue = _SpawnCleanupIssue(
                worker_index=worker_index,
                resource=resource,
                operation=operation,
                error_type=type(exc).__name__,
                message=message or "<empty>",
            )
            if issue not in issues:
                issues.append(issue)
            if fatal and issue not in fatal_issues:
                fatal_issues.append(issue)

        def close_endpoints(
            connections: Sequence[Connection],
            closed_indices: set[int],
            resource: str,
        ) -> None:
            for worker_index, connection in enumerate(connections):
                if worker_index in closed_indices:
                    continue
                try:
                    connection.close()
                    if getattr(connection, "closed", True) is False:
                        raise RuntimeError("endpoint remained open after close")
                except BaseException as exc:
                    record_issue(worker_index, resource, "close", exc)
                else:
                    closed_indices.add(worker_index)

        def process_alive(
            worker_index: int,
            process: mp.Process,
            operation: str,
        ) -> bool:
            if worker_index in self._closed_process_handle_indices:
                return False
            try:
                return bool(process.is_alive())
            except BaseException as exc:
                record_issue(worker_index, "process", operation, exc)
                return True

        def process_pid(worker_index: int, process: mp.Process) -> int | None:
            if worker_index < len(self._worker_pids):
                return self._worker_pids[worker_index]
            try:
                value = process.pid
            except BaseException as exc:
                record_issue(worker_index, "process", "pid", exc)
                return None
            return int(value) if isinstance(value, int) else None

        def process_exitcode(
            worker_index: int,
            process: mp.Process,
        ) -> int | None:
            if (
                worker_index in self._closed_process_handle_indices
                and self._worker_exitcodes_after_close is not None
                and worker_index < len(self._worker_exitcodes_after_close)
            ):
                return self._worker_exitcodes_after_close[worker_index]
            try:
                return process.exitcode
            except BaseException as exc:
                record_issue(worker_index, "process", "exitcode", exc)
                return None

        close_endpoints(
            self._child_connections,
            self._closed_child_endpoint_indices,
            "child_endpoint",
        )

        for worker_index, process in enumerate(self._processes):
            if worker_index in self._closed_process_handle_indices:
                continue
            connection = (
                self._connections[worker_index]
                if worker_index < len(self._connections)
                else None
            )
            if (
                connection is not None
                and worker_index not in self._closed_parent_endpoint_indices
                and process_alive(worker_index, process, "before-graceful-send")
            ):
                try:
                    connection.send(("close", None))
                except BaseException as exc:
                    record_issue(
                        worker_index,
                        "parent_endpoint",
                        "send-close",
                        exc,
                        fatal=False,
                    )

        for worker_index, process in enumerate(self._processes):
            if worker_index in self._closed_process_handle_indices:
                continue
            connection = (
                self._connections[worker_index]
                if worker_index < len(self._connections)
                else None
            )
            if (
                connection is not None
                and worker_index not in self._closed_parent_endpoint_indices
                and process_alive(worker_index, process, "before-graceful-receive")
            ):
                try:
                    if connection.poll(min(max(self.timeout_seconds, 0.1), 1.0)):
                        connection.recv()
                except BaseException as exc:
                    record_issue(
                        worker_index,
                        "parent_endpoint",
                        "receive-close",
                        exc,
                        fatal=False,
                    )
            try:
                process.join(timeout=5.0)
            except BaseException as exc:
                record_issue(worker_index, "process", "graceful-join", exc)

        terminate_attempted: list[int] = []
        for worker_index, process in enumerate(self._processes):
            if worker_index in self._closed_process_handle_indices:
                continue
            if process_alive(worker_index, process, "before-terminate"):
                terminate_attempted.append(worker_index)
                try:
                    process.terminate()
                except BaseException as exc:
                    record_issue(worker_index, "process", "terminate", exc)
        for worker_index in terminate_attempted:
            try:
                self._processes[worker_index].join(timeout=5.0)
            except BaseException as exc:
                record_issue(worker_index, "process", "terminate-join", exc)

        kill_attempted: list[int] = []
        for worker_index, process in enumerate(self._processes):
            if worker_index in self._closed_process_handle_indices:
                continue
            if process_alive(worker_index, process, "before-kill"):
                kill_attempted.append(worker_index)
                kill = getattr(process, "kill", None)
                if callable(kill):
                    try:
                        kill()
                    except BaseException as exc:
                        record_issue(worker_index, "process", "kill", exc)
                else:
                    record_issue(
                        worker_index,
                        "process",
                        "kill",
                        RuntimeError("kill is unavailable"),
                    )
        for worker_index in kill_attempted:
            try:
                self._processes[worker_index].join(timeout=5.0)
            except BaseException as exc:
                record_issue(worker_index, "process", "kill-join", exc)

        close_endpoints(
            self._connections,
            self._closed_parent_endpoint_indices,
            "parent_endpoint",
        )

        alive_values: list[bool] = []
        exitcodes: list[int | None] = []
        alive_workers: list[tuple[int, int | None, int | None]] = []
        for worker_index, process in enumerate(self._processes):
            alive = process_alive(worker_index, process, "final-is-alive")
            exitcode = process_exitcode(worker_index, process)
            alive_values.append(alive)
            exitcodes.append(exitcode)
            if alive:
                alive_workers.append(
                    (worker_index, process_pid(worker_index, process), exitcode)
                )
                continue
            if worker_index in self._closed_process_handle_indices:
                continue
            close_process = getattr(process, "close", None)
            if callable(close_process):
                try:
                    close_process()
                except BaseException as exc:
                    record_issue(worker_index, "process_handle", "close", exc)
                else:
                    self._closed_process_handle_indices.add(worker_index)
            else:
                self._closed_process_handle_indices.add(worker_index)

        self._worker_alive_after_close = tuple(alive_values)
        self._worker_exitcodes_after_close = tuple(exitcodes)
        self._worker_cleanup_issues_after_close = tuple(issues)
        open_child_endpoints = tuple(
            index
            for index in range(len(self._child_connections))
            if index not in self._closed_child_endpoint_indices
        )
        open_parent_endpoints = tuple(
            index
            for index in range(len(self._connections))
            if index not in self._closed_parent_endpoint_indices
        )
        open_process_handles = tuple(
            index
            for index in range(len(self._processes))
            if index not in self._closed_process_handle_indices
        )
        resources_closed = not (
            alive_workers
            or open_child_endpoints
            or open_parent_endpoints
            or open_process_handles
        )
        if fatal_issues or not resources_closed:
            cleanup_error = SpawnCleanupError(
                alive_workers=alive_workers,
                open_child_endpoints=open_child_endpoints,
                open_parent_endpoints=open_parent_endpoints,
                open_process_handles=open_process_handles,
                issues=issues,
            )
            if primary_exception is not None:
                self._closed = resources_closed
                note = str(cleanup_error)
                if note not in tuple(getattr(primary_exception, "__notes__", ())):
                    primary_exception.add_note(note)
                return
            self._closed = False
            raise cleanup_error
        self._closed = True

    def __enter__(self) -> "SpawnVectorEnv":
        if self._closed:
            raise WorkerProcessError("vector env is already closed")
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _round_trip(
        self,
        commands: Mapping[int, tuple[str, object]],
    ) -> dict[int, object]:
        if self._closed or self._broken:
            raise WorkerProcessError("vector env is closed or broken")
        for worker_index, command in commands.items():
            try:
                self._connections[worker_index].send(command)
            except (BrokenPipeError, EOFError, OSError) as exc:
                self._fail(
                    f"worker {worker_index} command send failed",
                    cause=exc,
                )
        values: dict[int, object] = {}
        for worker_index in commands:
            values[worker_index] = self._receive(
                worker_index,
                timeout_seconds=self.timeout_seconds,
            )
        return values

    def _receive(self, worker_index: int, *, timeout_seconds: float) -> object:
        connection = self._connections[worker_index]
        process = self._processes[worker_index]
        try:
            if not connection.poll(timeout_seconds):
                self._fail(
                    f"worker {worker_index} timed out after {timeout_seconds:g}s"
                )
            response = connection.recv()
        except EOFError as exc:
            self._fail(
                f"worker {worker_index} reached EOF (exitcode={process.exitcode})",
                cause=exc,
            )
        except OSError as exc:
            self._fail(f"worker {worker_index} receive failed", cause=exc)
        if (
            isinstance(response, dict)
            and response.get("kind") == "remote_error"
        ):
            self._fail(
                f"worker {worker_index} remote exception: "
                f"{response.get('error_type')}: {response.get('message')}\n"
                f"{response.get('traceback')}"
            )
        return response

    def _validated_indices(
        self,
        values: Sequence[int] | None,
    ) -> tuple[int, ...]:
        indices = (
            tuple(range(ROLLOUT_ENV_COUNT))
            if values is None
            else tuple(values)
        )
        if (
            not indices
            or len(set(indices)) != len(indices)
            or any(
                type(index) is not int
                or not 0 <= index < ROLLOUT_ENV_COUNT
                for index in indices
            )
        ):
            raise ValueError("worker index set is invalid")
        return indices

    def _fail(
        self,
        message: str,
        *,
        cause: BaseException | None = None,
    ) -> None:
        self._broken = True
        if cause is None:
            raise WorkerProcessError(message)
        raise WorkerProcessError(message) from cause


class RolloutCollector:
    """主进程批量 policy inference，worker 只执行环境语义。"""

    def __init__(
        self,
        *,
        policy: nn.Module,
        vector_env: SpawnVectorEnv,
        device: torch.device | str,
        contract: CollectorContract,
        require_dual_scan: bool = False,
    ) -> None:
        if not isinstance(policy, nn.Module):
            raise CollectorError("collector policy must be a torch module")
        if not isinstance(vector_env, SpawnVectorEnv):
            raise CollectorError("collector requires SpawnVectorEnv")
        if not isinstance(contract, CollectorContract):
            raise CollectorError("collector contract is invalid")
        if type(require_dual_scan) is not bool:
            raise CollectorError("collector reset diagnostics profile is invalid")
        self.policy = policy
        self.vector_env = vector_env
        self.device = torch.device(device)
        self.contract = contract
        self.require_dual_scan = require_dual_scan
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise CollectorError("CUDA was requested but is unavailable; no CPU fallback")
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        try:
            model_device = next(policy.parameters()).device
        except StopIteration as exc:
            raise CollectorError("collector policy has no trainable parameters") from exc
        if model_device != self.device:
            raise CollectorError("collector policy and requested device differ")
        if any(parameter.dtype != torch.float32 for parameter in policy.parameters()):
            raise CollectorError("collector policy parameters must be FP32")
        self._inference_pids: set[int] = set()
        self._inference_batch_count = 0

    def collect(
        self,
        *,
        continue_from_current_state: bool = False,
        resource_guard: Callable[[str], None] | None = None,
    ) -> CollectionResult:
        if type(continue_from_current_state) is not bool:
            raise CollectorError("continue_from_current_state must be boolean")
        if resource_guard is not None and not callable(resource_guard):
            raise CollectorError("resource_guard must be callable")

        def poll(boundary: str) -> None:
            if resource_guard is not None:
                resource_guard(boundary)

        self._reset_inference_audit()
        self.policy.eval()
        initial_policy_hash = policy_state_sha256(self.policy)
        buffer = RolloutBuffer()
        counts = [0] * ROLLOUT_ENV_COUNT
        diagnostic_resets = [0] * ROLLOUT_ENV_COUNT
        reset_audits: list[dict[str, object]] = []
        planner_failure_counts = _empty_planner_failure_counts()
        current: dict[int, WorkerObservation] = {}
        if continue_from_current_state:
            poll("rollout:before-current-observations")
            current.update(self.vector_env.current_observations())
            poll("rollout:after-current-observations")
        else:
            poll("rollout:before-reset")
            first_resets = self.vector_env.reset()
            poll("rollout:after-reset")
            for env_index, envelope in first_resets.items():
                current[env_index] = self._accept_reset(
                    env_index,
                    envelope,
                    diagnostic_resets,
                    reset_audits,
                )
        pending = [
            index
            for index, envelope in current.items()
            if not envelope.needs_policy
        ]
        self._reset_until_trainable(
            current,
            pending,
            diagnostic_resets,
            reset_audits,
            resource_guard=resource_guard,
        )

        last_done = [False] * ROLLOUT_ENV_COUNT
        snapshot_hashes: list[str] = []
        terminal_count = 0
        while any(count < ROLLOUT_STEPS_PER_ENV for count in counts):
            active = tuple(
                index
                for index, count in enumerate(counts)
                if count < ROLLOUT_STEPS_PER_ENV
            )
            if any(not current[index].needs_policy for index in active):
                raise CollectorError("active env lacks a trainable policy observation")
            snapshots = tuple(
                CandidateSnapshot.capture(
                    observation=current[index].observation,
                    frontier_cells=current[index].frontier_cells,
                    top_m_selection_summary=current[
                        index
                    ].top_m_selection_summary,
                )
                for index in active
            )
            poll("rollout:before-inference")
            output, action_sample = self._infer_actions(
                tuple(current[index].observation for index in active)
            )
            poll("rollout:after-inference")
            actions = {
                env_index: EnvAction(
                    candidate_index=int(
                        action_sample.selected_frontier_index[row].item()
                    ),
                    target_theta=float(action_sample.selected_theta[row].item()),
                )
                for row, env_index in enumerate(active)
            }
            poll("rollout:before-step")
            step_values = self.vector_env.step(actions)
            poll("rollout:after-step")
            terminal_to_reset: list[int] = []
            for row, env_index in enumerate(active):
                worker_step = step_values[env_index]
                result = worker_step.result
                if not result.trainable:
                    raise CollectorError("sampled action produced non-trainable transition")
                if result.done != result.terminal:
                    raise CollectorError("Stage 4 treats every done as terminal")
                if result.done and result.bootstrap_value != 0.0:
                    raise CollectorError("terminal transition attempted to bootstrap")
                if not result.done and not worker_step.next_observation.needs_policy:
                    raise CollectorError(
                        "next-empty sampled transition was not closed as terminal"
                    )
                _accumulate_planner_failure(
                    result,
                    planner_failure_counts,
                )
                transition = self._transition_from_step(
                    snapshot=snapshots[row],
                    policy_hash=initial_policy_hash,
                    selected_index=actions[env_index].candidate_index,
                    selected_theta=actions[env_index].target_theta,
                    old_log_prob_frontier=float(
                        action_sample.log_prob_frontier[row].item()
                    ),
                    old_log_prob_theta=float(
                        action_sample.log_prob_theta[row].item()
                    ),
                    old_log_prob_total=float(
                        action_sample.log_prob_total[row].item()
                    ),
                    old_value=float(output.value[row].item()),
                    result=result,
                )
                buffer.add(env_index, transition)
                counts[env_index] += 1
                snapshot_hashes.append(transition.candidate_snapshot_sha256)
                current[env_index] = worker_step.next_observation
                last_done[env_index] = result.done
                if result.done:
                    terminal_count += 1
                    if counts[env_index] < ROLLOUT_STEPS_PER_ENV:
                        terminal_to_reset.append(env_index)
            self._reset_until_trainable(
                current,
                terminal_to_reset,
                diagnostic_resets,
                reset_audits,
                resource_guard=resource_guard,
            )

        last_values = np.zeros((ROLLOUT_ENV_COUNT,), dtype=np.float32)
        nonterminal = tuple(
            index for index, done in enumerate(last_done) if not done
        )
        if nonterminal:
            poll("rollout:before-bootstrap-inference")
            output = self._forward(
                tuple(current[index].observation for index in nonterminal)
            )
            poll("rollout:after-bootstrap-inference")
            last_values[np.asarray(nonterminal, dtype=np.int64)] = (
                output.value.detach().cpu().numpy().astype(np.float32, copy=False)
            )
        poll("rollout:before-state-capture")
        vector_states = self.vector_env.capture_states()
        poll("rollout:after-state-capture")
        if policy_state_sha256(self.policy) != initial_policy_hash:
            raise CollectorError("policy changed during rollout collection")
        try:
            batch = buffer.finalize(last_values)
        except RolloutContractError as exc:
            raise CollectorError("collected rollout failed buffer finalization") from exc
        audit = CollectorAudit(
            schema_version=COLLECTOR_SCHEMA_VERSION,
            worker_pids=self.vector_env.worker_pids,
            worker_start_methods=self.vector_env.worker_start_methods,
            per_env_trainable_counts=tuple(counts),
            diagnostic_reset_counts=tuple(diagnostic_resets),
            trainable_transition_count=sum(counts),
            inference_pids=tuple(sorted(self._inference_pids)),
            inference_batch_count=self._inference_batch_count,
            policy_device=str(self.device),
            policy_state_sha256=initial_policy_hash,
            snapshot_sha256=tuple(snapshot_hashes),
            terminal_transition_count=terminal_count,
            reset_diagnostics=tuple(reset_audits),
            planner_failure_counts=dict(planner_failure_counts),
        )
        return CollectionResult(
            batch=batch,
            audit=audit,
            vector_env_states=vector_states,
        )

    def preflight_one_step(self) -> CollectorPreflightResult:
        """以真实 collector 合同让全部 worker 各执行一条 transition。"""

        self._reset_inference_audit()
        self.policy.eval()
        initial_policy_hash = policy_state_sha256(self.policy)
        diagnostic_resets = [0] * ROLLOUT_ENV_COUNT
        reset_audits: list[dict[str, object]] = []
        planner_failure_counts = _empty_planner_failure_counts()

        reset_started = time.perf_counter()
        current = self.vector_env.reset()
        for env_index, envelope in tuple(current.items()):
            current[env_index] = self._accept_reset(
                env_index,
                envelope,
                diagnostic_resets,
                reset_audits,
            )
        pending = tuple(
            index for index, envelope in current.items() if not envelope.needs_policy
        )
        self._reset_until_trainable(
            current,
            pending,
            diagnostic_resets,
            reset_audits,
        )
        reset_seconds = time.perf_counter() - reset_started
        if set(current) != set(range(ROLLOUT_ENV_COUNT)) or any(
            not current[index].needs_policy for index in range(ROLLOUT_ENV_COUNT)
        ):
            raise CollectorError("preflight reset did not produce eight policy rows")

        snapshots = tuple(
            CandidateSnapshot.capture(
                observation=current[index].observation,
                frontier_cells=current[index].frontier_cells,
                top_m_selection_summary=current[index].top_m_selection_summary,
            )
            for index in range(ROLLOUT_ENV_COUNT)
        )
        inference_started = time.perf_counter()
        output = self._forward(
            tuple(current[index].observation for index in range(ROLLOUT_ENV_COUNT))
        )
        policy_batch = batch_policy_observations(
            tuple(current[index].observation for index in range(ROLLOUT_ENV_COUNT)),
            device=self.device,
        )
        with torch.no_grad():
            action_sample = sample_action(
                output,
                policy_batch.candidate_mask,
                deterministic=True,
            )
        inference_seconds = time.perf_counter() - inference_started
        actions = {
            index: EnvAction(
                candidate_index=int(action_sample.selected_frontier_index[index].item()),
                target_theta=float(action_sample.selected_theta[index].item()),
            )
            for index in range(ROLLOUT_ENV_COUNT)
        }

        step_started = time.perf_counter()
        step_values = self.vector_env.step(actions)
        step_seconds = time.perf_counter() - step_started
        transitions: list[RolloutTransition] = []
        terminal_count = 0
        for index in range(ROLLOUT_ENV_COUNT):
            worker_step = step_values[index]
            result = worker_step.result
            if (
                not result.trainable
                or result.done != result.terminal
                or (result.done and result.bootstrap_value != 0.0)
                or (not result.done and not worker_step.next_observation.needs_policy)
            ):
                raise CollectorError("preflight worker transition contract drifted")
            _accumulate_planner_failure(
                result,
                planner_failure_counts,
            )
            transition = self._transition_from_step(
                snapshot=snapshots[index],
                policy_hash=initial_policy_hash,
                selected_index=actions[index].candidate_index,
                selected_theta=actions[index].target_theta,
                old_log_prob_frontier=float(
                    action_sample.log_prob_frontier[index].item()
                ),
                old_log_prob_theta=float(action_sample.log_prob_theta[index].item()),
                old_log_prob_total=float(action_sample.log_prob_total[index].item()),
                old_value=float(output.value[index].item()),
                result=result,
            )
            transitions.append(transition)
            terminal_count += int(result.done)
        if policy_state_sha256(self.policy) != initial_policy_hash:
            raise CollectorError("policy changed during one-step preflight")
        timings = MappingProxyType(
            {
                "reset": float(reset_seconds),
                "inference": float(inference_seconds),
                "step": float(step_seconds),
            }
        )
        if any(not math.isfinite(value) or value < 0.0 for value in timings.values()):
            raise CollectorError("preflight timing is invalid")
        return CollectorPreflightResult(
            schema_version="stage6_collector_one_step_preflight/v1",
            transitions=tuple(transitions),
            trainable_transition_count=len(transitions),
            worker_pids=self.vector_env.worker_pids,
            worker_start_methods=self.vector_env.worker_start_methods,
            diagnostic_reset_counts=tuple(diagnostic_resets),
            inference_pids=tuple(sorted(self._inference_pids)),
            inference_batch_count=self._inference_batch_count,
            policy_device=str(self.device),
            policy_state_sha256=initial_policy_hash,
            snapshot_sha256=tuple(snapshot.sha256 for snapshot in snapshots),
            terminal_transition_count=terminal_count,
            timings_seconds=timings,
            reset_diagnostics=tuple(reset_audits),
            planner_failure_counts=dict(planner_failure_counts),
        )

    def _accept_reset(
        self,
        env_index: int,
        envelope: WorkerObservation,
        diagnostic_resets: list[int],
        reset_audits: list[dict[str, object]],
    ) -> WorkerObservation:
        if envelope.reset_trainable or envelope.fake_logprob_created:
            raise CollectorError(
                f"worker {env_index} reset created trainable/fake policy data"
            )
        if not envelope.needs_policy:
            diagnostic_resets[env_index] += 1
            if diagnostic_resets[env_index] > _MAX_DIAGNOSTIC_RESETS_PER_ENV:
                raise CollectorError("diagnostic reset retry limit exceeded")
        try:
            reset_payload = validate_reset_diagnostics_payload(
                envelope.reset_diagnostics,
                require_dual_scan=self.require_dual_scan,
            )
        except ValueError as exc:
            raise CollectorError("worker reset diagnostics drifted") from exc
        reset_audits.append(
            {
                "worker_index": env_index,
                "reset_index": len(reset_audits),
                **reset_payload,
            }
        )
        return envelope

    def _reset_until_trainable(
        self,
        current: dict[int, WorkerObservation],
        indices: Sequence[int],
        diagnostic_resets: list[int],
        reset_audits: list[dict[str, object]],
        *,
        resource_guard: Callable[[str], None] | None = None,
    ) -> None:
        pending = tuple(indices)
        while pending:
            if resource_guard is not None:
                resource_guard("rollout:before-diagnostic-reset")
            reset_values = self.vector_env.reset(pending)
            if resource_guard is not None:
                resource_guard("rollout:after-diagnostic-reset")
            next_pending: list[int] = []
            for env_index in pending:
                envelope = self._accept_reset(
                    env_index,
                    reset_values[env_index],
                    diagnostic_resets,
                    reset_audits,
                )
                current[env_index] = envelope
                if not envelope.needs_policy:
                    next_pending.append(env_index)
            pending = tuple(next_pending)

    def _infer_actions(
        self,
        observations: tuple[PolicyObservation, ...],
    ) -> tuple[PolicyForwardOutput, Any]:
        policy_batch = batch_policy_observations(observations, device=self.device)
        output = self._forward_batch(policy_batch)
        with torch.no_grad():
            action = sample_action(
                output,
                policy_batch.candidate_mask,
                deterministic=False,
            )
        return output, action

    def _forward(
        self,
        observations: tuple[PolicyObservation, ...],
    ) -> PolicyForwardOutput:
        policy_batch = batch_policy_observations(observations, device=self.device)
        return self._forward_batch(policy_batch)

    def _forward_batch(self, policy_batch: Any) -> PolicyForwardOutput:
        if os.getpid() in self.vector_env.worker_pids:
            raise CollectorError("policy inference attempted inside a worker")
        with torch.no_grad():
            output = self.policy(policy_batch)
        if not isinstance(output, PolicyForwardOutput):
            raise CollectorError("policy returned an invalid forward output")
        self._inference_pids.add(os.getpid())
        self._inference_batch_count += 1
        return output

    def _reset_inference_audit(self) -> None:
        self._inference_pids.clear()
        self._inference_batch_count = 0

    def _transition_from_step(
        self,
        *,
        snapshot: CandidateSnapshot,
        policy_hash: str,
        selected_index: int,
        selected_theta: float,
        old_log_prob_frontier: float,
        old_log_prob_theta: float,
        old_log_prob_total: float,
        old_value: float,
        result: StepResult,
    ) -> RolloutTransition:
        valid_indices = tuple(
            int(value) for value in np.flatnonzero(snapshot.candidate_mask)
        )
        try:
            selected_rank = valid_indices.index(selected_index)
        except ValueError as exc:
            raise CollectorError("sampled frontier index is not valid in snapshot") from exc
        selected_cell = snapshot.frontier_cells[selected_rank]
        diagnostics = result.diagnostics
        sensor = diagnostics.sensor
        execution = diagnostics.execution
        if sensor is None:
            observation_sample_count = 0
            ray_count_per_sample = 0
            ray_cell_visit_count = 0
            sensor_summary: dict[str, object] = {}
        else:
            observation_sample_count = sensor.sample_count
            if (
                observation_sample_count <= 0
                or sensor.ray_count % observation_sample_count != 0
            ):
                raise CollectorError("sensor ray count is not per-sample auditable")
            ray_count_per_sample = sensor.ray_count // observation_sample_count
            ray_cell_visit_count = sensor.cell_visit_count
            sensor_summary = asdict(sensor)
        execution_summary = asdict(execution) if execution is not None else {}
        execution_summary["sensor"] = sensor_summary
        path_length = float(diagnostics.path_length_m)
        gain_per_meter = (
            result.coverage_gain_cells / path_length if path_length > 0.0 else 0.0
        )
        return RolloutTransition.create(
            snapshot=snapshot,
            policy_state_sha256=policy_hash,
            config_sha256=self.contract.config_sha256,
            lineage=self.contract.lineage,
            selected_frontier_index=selected_index,
            selected_cell_xy=selected_cell,
            selected_theta=selected_theta,
            old_log_prob_frontier=old_log_prob_frontier,
            old_log_prob_theta=old_log_prob_theta,
            old_log_prob_total=old_log_prob_total,
            old_value=old_value,
            reward=result.reward,
            done=result.done,
            done_reason=result.reason,
            planned_path_cells=diagnostics.planned_path_cells,
            path_length_m=path_length,
            path_observation_step_m=diagnostics.path_observation_step_m,
            coverage_gain_cells=result.coverage_gain_cells,
            coverage_gain_per_meter=gain_per_meter,
            observation_sample_count=observation_sample_count,
            ray_count_per_sample=ray_count_per_sample,
            ray_cell_visit_count=ray_cell_visit_count,
            newly_observed_cell_count=diagnostics.newly_observed_cell_count,
            planner_validation_summary=diagnostics.planner,
            execution_observation_summary=execution_summary,
            frontier_extractor_version=self.contract.frontier_extractor_version,
            observation_schema_version=self.contract.observation_schema_version,
            action_space_version=self.contract.action_space_version,
            reward_version=self.contract.reward_version,
            planner_version=self.contract.planner_version,
        )


def build_lunar_exploration_env(*, stage1_config: Mapping[str, object]) -> object:
    """Spawn-safe production factory；延迟导入，worker 内构造 env/planner/sensor。"""

    if TYPE_CHECKING:
        from typing import Any as Stage1Config
    if TYPE_CHECKING:
        from typing import Any as LunarExplorationEnv

    return LunarExplorationEnv(Stage1Config.model_validate(dict(stage1_config)))


def lunar_env_specs(
    stage1_config: Mapping[str, object],
) -> tuple[SpawnEnvSpec, ...]:
    payload = _copy_json_mapping(stage1_config, "Stage 1 config")
    return tuple(
        SpawnEnvSpec(
            factory=build_lunar_exploration_env,
            kwargs={"stage1_config": payload},
        )
        for _ in range(ROLLOUT_ENV_COUNT)
    )


def _worker_main(
    worker_index: int,
    connection: Connection,
    spec: SpawnEnvSpec,
) -> None:
    env: object | None = None
    try:
        env = spec.factory(**dict(spec.kwargs))
        connection.send(
            {
                "kind": "ready",
                "worker_index": worker_index,
                "pid": os.getpid(),
                "start_method": mp.get_start_method(),
            }
        )
        while True:
            command, payload = connection.recv()
            if command == "close":
                connection.send("closed")
                break
            try:
                if command == "reset":
                    observation = env.reset()  # type: ignore[attr-defined]
                    connection.send(
                        _worker_observation(env, observation, is_reset=True)
                    )
                elif command == "step":
                    result = env.step(payload)  # type: ignore[attr-defined]
                    if not isinstance(result, StepResult):
                        raise TypeError("worker env step must return StepResult")
                    connection.send(
                        WorkerStep(
                            result=result,
                            next_observation=_worker_observation(
                                env,
                                result.observation,
                                is_reset=False,
                            ),
                        )
                    )
                elif command == "current_observation":
                    current_observation = env.current_observation  # type: ignore[attr-defined]
                    if current_observation is None:
                        raise RuntimeError("worker current observation is unavailable")
                    connection.send(
                        _worker_observation(
                            env,
                            current_observation,
                            is_reset=False,
                        )
                    )
                elif command == "capture_state":
                    episode_state = env.export_episode_state()  # type: ignore[attr-defined]
                    sampler_state = (
                        env.export_sampler_state()  # type: ignore[attr-defined]
                        if hasattr(env, "export_sampler_state")
                        else {}
                    )
                    connection.send(
                        {
                            "episode_state": episode_state,
                            "sampler_state": sampler_state,
                        }
                    )
                elif command == "restore_state":
                    if not isinstance(payload, dict) or set(payload) != {
                        "episode_state",
                        "sampler_state",
                    }:
                        raise ValueError("worker restore payload mismatch")
                    env.import_episode_state(payload["episode_state"])  # type: ignore[attr-defined]
                    if hasattr(env, "import_sampler_state"):
                        env.import_sampler_state(payload["sampler_state"])  # type: ignore[attr-defined]
                    elif payload["sampler_state"] != {}:
                        raise ValueError("worker env has no sampler state importer")
                    connection.send("restored")
                else:
                    raise ValueError(f"unknown worker command: {command}")
            except BaseException as exc:
                connection.send(
                    {
                        "kind": "remote_error",
                        "worker_index": worker_index,
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
                break
    except (EOFError, BrokenPipeError, OSError):
        pass
    except BaseException as exc:
        try:
            connection.send(
                {
                    "kind": "remote_error",
                    "worker_index": worker_index,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
        except (EOFError, BrokenPipeError, OSError):
            pass
    finally:
        if env is not None and hasattr(env, "close"):
            try:
                env.close()  # type: ignore[attr-defined]
            except Exception:
                pass
        connection.close()


def _empty_planner_failure_counts() -> dict[str, int]:
    return {reason: 0 for reason in PLANNER_FAILURE_REASONS}


def _sensor_diagnostics_payload(value: object) -> dict[str, object]:
    if not hasattr(value, "__dataclass_fields__"):
        raise TypeError("reset sensor diagnostics are invalid")
    raw = asdict(value)  # type: ignore[arg-type]
    try:
        copied = json.loads(
            json.dumps(
                raw,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
        )
    except (TypeError, ValueError) as exc:
        raise TypeError("reset sensor diagnostics are not finite JSON") from exc
    if not isinstance(copied, dict):
        raise TypeError("reset sensor diagnostics mapping is invalid")
    return copied


def validate_reset_diagnostics_payload(
    value: object,
    *,
    require_dual_scan: bool,
) -> dict[str, object]:
    if type(require_dual_scan) is not bool or not isinstance(value, Mapping):
        raise ValueError("reset diagnostics payload is invalid")
    if (
        set(value)
        != {
            "schema_version",
            "scan_order",
            "local_safety_sensor",
            "exploration_sensor",
        }
        or value.get("schema_version")
        != "stage6_reset_scan_diagnostics/v1"
    ):
        raise ValueError("reset diagnostics field set drifted")
    order = value.get("scan_order")
    expected_order = ["reset_local_safety", "reset_exploration"]
    if (
        not isinstance(order, list)
        or (
            order != expected_order
            and (require_dual_scan or order != ["reset_exploration"])
        )
    ):
        raise ValueError("reset scan order drifted")
    sensor_fields = {
        "sample_count",
        "ray_count",
        "cell_visit_count",
        "unique_visible_cell_count",
        "duplicate_cell_visits",
        "sample_sources",
        "sample_headings",
    }
    sensors: dict[str, dict[str, object]] = {}
    for name in ("local_safety_sensor", "exploration_sensor"):
        sensor = value.get(name)
        if (
            not isinstance(sensor, Mapping)
            or set(sensor) != sensor_fields
            or any(
                type(sensor.get(field_name)) is not int
                or int(sensor[field_name]) < 0
                for field_name in (
                    "sample_count",
                    "ray_count",
                    "cell_visit_count",
                    "unique_visible_cell_count",
                    "duplicate_cell_visits",
                )
            )
            or not isinstance(sensor.get("sample_sources"), list)
            or any(
                not isinstance(source, str) or not source
                for source in sensor["sample_sources"]
            )
            or not isinstance(sensor.get("sample_headings"), list)
            or any(
                not isinstance(heading, (int, float))
                or not math.isfinite(float(heading))
                for heading in sensor["sample_headings"]
            )
        ):
            raise ValueError("reset sensor diagnostics drifted")
        sensors[name] = dict(sensor)
    if order == expected_order and (
        sensors["local_safety_sensor"]["sample_sources"]
        != ["reset_local_safety"]
        or sensors["exploration_sensor"]["sample_sources"] != ["reset"]
    ):
        raise ValueError("reset sensor source order drifted")
    return {
        "schema_version": "stage6_reset_scan_diagnostics/v1",
        "scan_order": list(order),
        "local_safety_sensor": sensors["local_safety_sensor"],
        "exploration_sensor": sensors["exploration_sensor"],
    }


def _reset_diagnostics_payload(value: object) -> dict[str, object]:
    local = getattr(value, "local_safety_sensor", None)
    exploration = getattr(value, "exploration_sensor", None)
    order = getattr(value, "scan_order", None)
    return validate_reset_diagnostics_payload(
        {
            "schema_version": "stage6_reset_scan_diagnostics/v1",
            "scan_order": list(order) if isinstance(order, tuple) else order,
            "local_safety_sensor": _sensor_diagnostics_payload(local),
            "exploration_sensor": _sensor_diagnostics_payload(exploration),
        },
        require_dual_scan=False,
    )


def _accumulate_planner_failure(
    result: StepResult,
    counts: dict[str, int],
) -> None:
    reason = result.diagnostics.planner.get("failure_reason")
    if reason == "none":
        return
    if not isinstance(reason, str) or reason not in PLANNER_FAILURE_REASONS:
        raise CollectorError("planner failure reason is unknown")
    counts[reason] += 1


def _worker_observation(
    env: object,
    observation: object,
    *,
    is_reset: bool,
) -> WorkerObservation:
    if not isinstance(observation, PolicyObservation):
        raise TypeError("worker env reset/step observation must be PolicyObservation")
    action_set = env.current_action_set  # type: ignore[attr-defined]
    cells = tuple(action_set.cells)
    summary = dict(action_set.diagnostics or {})
    summary["candidate_count"] = len(cells)
    needs_policy = env.needs_policy  # type: ignore[attr-defined]
    if type(needs_policy) is not bool:
        raise TypeError("worker env needs_policy must be boolean")
    reset_trainable = False
    fake_logprob_created = False
    reset_payload = None
    if is_reset:
        reset_diagnostics = env.last_reset_diagnostics  # type: ignore[attr-defined]
        if reset_diagnostics is None:
            raise RuntimeError("worker env reset diagnostics are missing")
        reset_trainable = reset_diagnostics.trainable
        fake_logprob_created = reset_diagnostics.fake_logprob_created
        reset_coverage_rate = float(reset_diagnostics.coverage_rate)
        reset_reason = str(reset_diagnostics.reason)
        reset_done = not needs_policy
        reset_payload = _reset_diagnostics_payload(reset_diagnostics)
    else:
        reset_coverage_rate = 0.0
        reset_reason = "none"
        reset_done = False
    return WorkerObservation(
        observation=observation,
        frontier_cells=cells,
        top_m_selection_summary=summary,
        needs_policy=needs_policy,
        reset_trainable=reset_trainable,
        fake_logprob_created=fake_logprob_created,
        reset_coverage_rate=reset_coverage_rate,
        reset_reason=reset_reason,
        reset_done=reset_done,
        reset_diagnostics=reset_payload,
    )


def _copy_json_mapping(
    value: Mapping[str, object],
    name: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    try:
        copied = json.loads(
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain finite JSON") from exc
    if not isinstance(copied, dict):
        raise ValueError(f"{name} must be a JSON object")
    ArtifactStore.canonical_json_bytes(copied)
    return copied


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


__all__ = [
    "COLLECTOR_SCHEMA_VERSION",
    "CollectionResult",
    "CollectorAudit",
    "CollectorContract",
    "CollectorError",
    "CollectorPreflightResult",
    "PLANNER_FAILURE_REASONS",
    "RolloutCollector",
    "SpawnEnvSpec",
    "SpawnVectorEnv",
    "VECTOR_STATE_SCHEMA_VERSION",
    "WorkerObservation",
    "WorkerProcessError",
    "WorkerStep",
    "build_lunar_exploration_env",
    "lunar_env_specs",
    "validate_reset_diagnostics_payload",
]

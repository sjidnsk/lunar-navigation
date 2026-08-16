"""Bounded, priority-ordered single-flight construction of task artifacts."""

from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass
from enum import IntEnum
import heapq
import itertools
import multiprocessing as mp
import os
from pathlib import Path
import queue
import threading
import time
from types import MappingProxyType
from typing import Callable, Mapping, Sequence

from .task_cache import (
    PlatformTaskArtifact,
    PlatformTaskKey,
    PlatformTaskPayload,
    TaskCacheStore,
    TaskCommonArtifact,
    TaskCommonKey,
    TaskCommonPayload,
)


class TaskCacheCoordinatorError(RuntimeError):
    """One bounded cache request failed or violated its authority."""


class TaskBuildPriority(IntEnum):
    CURRENT_GROUND = 0
    NEXT_GROUND = 1
    CURRENT_CURRICULUM = 2
    FUTURE_HOPPER = 3


def priority_for_scheduled_task(
    *,
    stage: object,
    platform_type: str,
    episode_offset: int,
) -> TaskBuildPriority:
    """Classify work without changing its deterministic schedule identity."""
    stage_value = getattr(stage, "value", stage)
    if platform_type not in {"WHEELED", "LEGGED", "HOPPER"}:
        raise ValueError("scheduled task platform is invalid")
    if type(episode_offset) is not int or episode_offset < 0:
        raise ValueError("scheduled task episode offset is invalid")
    if stage_value == "GROUND_R1":
        if platform_type == "HOPPER":
            return TaskBuildPriority.FUTURE_HOPPER
        if episode_offset == 0:
            return TaskBuildPriority.CURRENT_GROUND
        return TaskBuildPriority.NEXT_GROUND
    if stage_value in {"GROUND_R2_HOPPER_R1", "THREE_PLATFORM_R2"}:
        return TaskBuildPriority.CURRENT_CURRICULUM
    raise ValueError("scheduled task curriculum stage is invalid")


@dataclass(frozen=True, slots=True)
class TaskBuildEstimate:
    in_flight_bytes: int
    native_calls: int

    def __post_init__(self) -> None:
        if (
            type(self.in_flight_bytes) is not int
            or self.in_flight_bytes <= 0
            or type(self.native_calls) is not int
            or self.native_calls < 0
        ):
            raise ValueError("task build estimate is invalid")


@dataclass(frozen=True, slots=True)
class TaskBuildRequest:
    key: PlatformTaskKey
    priority: TaskBuildPriority
    estimate: TaskBuildEstimate
    checkpoint_reference_count: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.key, PlatformTaskKey):
            raise TypeError("task build request key is invalid")
        if not isinstance(self.priority, TaskBuildPriority):
            raise TypeError("task build request priority is invalid")
        if not isinstance(self.estimate, TaskBuildEstimate):
            raise TypeError("task build request estimate is invalid")
        if (
            type(self.checkpoint_reference_count) is not int
            or self.checkpoint_reference_count < 0
        ):
            raise ValueError("task checkpoint reference count is invalid")


@dataclass(frozen=True, slots=True)
class TaskBuildProduct:
    payload: PlatformTaskPayload
    native_calls: int

    def __post_init__(self) -> None:
        if not isinstance(self.payload, PlatformTaskPayload):
            raise TypeError("task build product payload is invalid")
        if type(self.native_calls) is not int or self.native_calls < 0:
            raise ValueError("task build native call count is invalid")


@dataclass(frozen=True, slots=True)
class TaskArtifactReference:
    key: PlatformTaskKey
    artifact_sha256: str
    cache_root: Path

    @classmethod
    def from_artifact(
        cls,
        artifact: PlatformTaskArtifact,
        cache_root: Path,
    ) -> "TaskArtifactReference":
        if not isinstance(artifact, PlatformTaskArtifact):
            raise TypeError("task artifact reference source is invalid")
        return cls(
            key=artifact.key,
            artifact_sha256=artifact.artifact_sha256,
            cache_root=cache_root,
        )

    def load(self) -> PlatformTaskArtifact:
        artifact = TaskCacheStore(self.cache_root).load_platform(self.key)
        if artifact is None or artifact.artifact_sha256 != self.artifact_sha256:
            raise TaskCacheCoordinatorError(
                "task artifact reference no longer resolves exactly"
            )
        return artifact

    def to_dict(self) -> dict[str, object]:
        return {
            "key": self.key.to_dict(),
            "artifact_sha256": self.artifact_sha256,
            "cache_root": str(self.cache_root),
        }

    @classmethod
    def from_dict(cls, value: object) -> "TaskArtifactReference":
        if not isinstance(value, Mapping) or set(value) != {
            "key",
            "artifact_sha256",
            "cache_root",
        }:
            raise TaskCacheCoordinatorError(
                "task artifact reference schema differs"
            )
        try:
            key = PlatformTaskKey(**value["key"])
            root = Path(value["cache_root"])
        except (TypeError, ValueError) as error:
            raise TaskCacheCoordinatorError(
                "task artifact reference payload differs"
            ) from error
        digest = value["artifact_sha256"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or not root.is_absolute()
        ):
            raise TaskCacheCoordinatorError(
                "task artifact reference authority differs"
            )
        return cls(key=key, artifact_sha256=digest, cache_root=root)


@dataclass(slots=True)
class _PendingBuild:
    request: TaskBuildRequest
    future: Future[TaskArtifactReference]
    sequence: int
    enqueued_at: float
    running: bool = False


class TaskCacheCoordinator:
    """Own the only bounded builder executor used by training workers."""

    def __init__(
        self,
        *,
        store: TaskCacheStore,
        builder: Callable[
            [PlatformTaskKey],
            PlatformTaskPayload | PlatformTaskArtifact | TaskBuildProduct,
        ],
        estimator: Callable[[PlatformTaskKey], TaskBuildEstimate],
        max_builder_workers: int,
        max_in_flight_bytes: int,
        max_native_calls: int,
        contextual_request_registrar: Callable[
            [Mapping[str, object]], None
        ]
        | None = None,
    ) -> None:
        if not isinstance(store, TaskCacheStore):
            raise TypeError("task cache coordinator requires TaskCacheStore")
        if not callable(builder) or not callable(estimator):
            raise TypeError("task cache coordinator callbacks are invalid")
        for value, name in (
            (max_builder_workers, "builder workers"),
            (max_in_flight_bytes, "in-flight bytes"),
            (max_native_calls, "native calls"),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"task cache {name} limit is invalid")
        self.store = store
        self.builder = builder
        self.estimator = estimator
        self.max_builder_workers = max_builder_workers
        self.max_in_flight_bytes = max_in_flight_bytes
        self.max_native_calls = max_native_calls
        if contextual_request_registrar is not None and not callable(
            contextual_request_registrar
        ):
            raise TypeError("task cache contextual registrar is invalid")
        self._contextual_request_registrar = contextual_request_registrar
        self._condition = threading.Condition()
        self._pending_heap: list[tuple[int, int, str]] = []
        self._jobs: dict[str, _PendingBuild] = {}
        self._attempts: dict[str, int] = {}
        self._common_futures: dict[str, Future[TaskCommonArtifact]] = {}
        self._sequence = itertools.count()
        self._active_builders = 0
        self._in_flight_bytes = 0
        self._in_flight_native_calls = 0
        self._closed = False
        self._metrics: dict[str, object] = {
            "cache_hit_count": 0,
            "cache_miss_count": 0,
            "build_count": 0,
            "single_flight_join_count": 0,
            "retry_count": 0,
            "failure_count": 0,
            "cancelled_count": 0,
            "peak_builder_count": 0,
            "peak_in_flight_bytes": 0,
            "peak_in_flight_native_calls": 0,
            "actual_native_call_count": 0,
            "artifact_bytes": 0,
            "checkpoint_reference_count": 0,
            "common_cache_hit_count": 0,
            "common_build_count": 0,
            "common_single_flight_join_count": 0,
            "priority_wait_seconds": {
                priority.name: 0.0 for priority in TaskBuildPriority
            },
        }
        from concurrent.futures import ThreadPoolExecutor

        self._executor = ThreadPoolExecutor(
            max_workers=max_builder_workers,
            thread_name_prefix="lunar-task-builder",
        )
        self._scheduler = threading.Thread(
            target=self._schedule,
            name="lunar-task-cache-scheduler",
            daemon=True,
        )
        self._scheduler.start()
        context = mp.get_context("spawn")
        self._ipc_request_queue = context.Queue()
        self._ipc_responses: dict[int, object] = {}
        self._ipc_next_client = itertools.count()
        self._ipc_thread: threading.Thread | None = None
        self._context_manager = (
            context.Manager()
            if contextual_request_registrar is not None
            else None
        )
        self._context_responses = (
            self._context_manager.dict()
            if self._context_manager is not None
            else None
        )

    def request(
        self,
        key: PlatformTaskKey,
        priority: TaskBuildPriority,
        *,
        timeout: float | None = None,
    ) -> TaskArtifactReference:
        request = TaskBuildRequest(
            key=key,
            priority=priority,
            estimate=self.estimator(key),
        )
        reference, future = self._request_or_future(request)
        if reference is not None:
            return reference
        assert future is not None
        try:
            return future.result(timeout=timeout)
        except TimeoutError as error:
            raise TaskCacheCoordinatorError(
                "task cache request timed out"
            ) from error

    def prefetch(self, requests: Sequence[TaskBuildRequest]) -> None:
        if not isinstance(requests, Sequence):
            raise TypeError("task cache prefetch requests are invalid")
        for request in requests:
            if not isinstance(request, TaskBuildRequest):
                raise TypeError("task cache prefetch request is invalid")
            self._request_or_future(request)

    def request_common(
        self,
        key: TaskCommonKey,
        builder: Callable[[], TaskCommonPayload],
    ) -> TaskCommonArtifact:
        """Resolve one shared crop inside a bounded platform builder."""
        if not isinstance(key, TaskCommonKey) or not callable(builder):
            raise TypeError("task common single-flight request is invalid")
        existing = self.store.load_common(key)
        if existing is not None:
            with self._condition:
                self._metrics["common_cache_hit_count"] += 1
            return existing
        identity = key.sha256()
        producer = False
        with self._condition:
            if self._closed:
                raise TaskCacheCoordinatorError(
                    "task cache coordinator is closed"
                )
            future = self._common_futures.get(identity)
            if future is None:
                future = Future()
                future.set_running_or_notify_cancel()
                self._common_futures[identity] = future
                producer = True
            else:
                self._metrics["common_single_flight_join_count"] += 1
        if not producer:
            return future.result()
        try:
            payload = builder()
            artifact = self.store.commit_common(key, payload)
            with self._condition:
                self._metrics["common_build_count"] += 1
            future.set_result(artifact)
            return artifact
        except Exception as error:
            future.set_exception(error)
            raise
        finally:
            with self._condition:
                if self._common_futures.get(identity) is future:
                    self._common_futures.pop(identity, None)
                self._condition.notify_all()

    def cancel(self, key: PlatformTaskKey) -> bool:
        if not isinstance(key, PlatformTaskKey):
            raise TypeError("task cache cancellation key is invalid")
        identity = key.sha256()
        with self._condition:
            job = self._jobs.get(identity)
            if job is None or job.running:
                return False
            cancelled = job.future.cancel()
            if cancelled:
                self._jobs.pop(identity, None)
                self._metrics["cancelled_count"] += 1
                self._condition.notify_all()
            return cancelled

    def wait_for_idle(self, timeout: float | None = None) -> None:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while self._jobs or self._active_builders:
                remaining = (
                    None if deadline is None else deadline - time.monotonic()
                )
                if remaining is not None and remaining <= 0.0:
                    raise TaskCacheCoordinatorError(
                        "task cache coordinator did not become idle"
                    )
                self._condition.wait(timeout=remaining)

    def create_client(self) -> "TaskCacheClient":
        with self._condition:
            if self._closed:
                raise TaskCacheCoordinatorError(
                    "task cache coordinator is closed"
                )
            client_id = next(self._ipc_next_client)
            context = mp.get_context("spawn")
            response_queue = context.Queue()
            self._ipc_responses[client_id] = response_queue
            if self._ipc_thread is None:
                self._ipc_thread = threading.Thread(
                    target=self._serve_ipc,
                    name="lunar-task-cache-ipc",
                    daemon=True,
                )
                self._ipc_thread.start()
        return TaskCacheClient(
            cache_root=self.store.root,
            client_id=client_id,
            request_queue=self._ipc_request_queue,
            response_queue=response_queue,
            context_responses=self._context_responses,
        )

    def metrics(self) -> Mapping[str, object]:
        with self._condition:
            values = dict(self._metrics)
            values["priority_wait_seconds"] = dict(
                self._metrics["priority_wait_seconds"]
            )
            return MappingProxyType(values)

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            for identity, job in tuple(self._jobs.items()):
                if not job.running and job.future.cancel():
                    self._jobs.pop(identity, None)
                    self._metrics["cancelled_count"] += 1
            self._condition.notify_all()
        self._scheduler.join()
        self._executor.shutdown(wait=True, cancel_futures=False)
        if self._ipc_thread is not None:
            self._ipc_request_queue.put(None)
            self._ipc_thread.join()
        if self._context_manager is not None:
            self._context_manager.shutdown()

    def __enter__(self) -> "TaskCacheCoordinator":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _request_or_future(
        self,
        request: TaskBuildRequest,
    ) -> tuple[
        TaskArtifactReference | None,
        Future[TaskArtifactReference] | None,
    ]:
        artifact = self.store.load_platform(request.key)
        if artifact is not None:
            with self._condition:
                self._metrics["cache_hit_count"] += 1
                self._metrics["checkpoint_reference_count"] += (
                    request.checkpoint_reference_count
                )
            return (
                TaskArtifactReference.from_artifact(artifact, self.store.root),
                None,
            )
        estimate = request.estimate
        if (
            estimate.in_flight_bytes > self.max_in_flight_bytes
            or estimate.native_calls > self.max_native_calls
        ):
            raise TaskCacheCoordinatorError(
                "task build estimate exceeds coordinator capacity"
            )
        identity = request.key.sha256()
        with self._condition:
            if self._closed:
                raise TaskCacheCoordinatorError(
                    "task cache coordinator is closed"
                )
            self._metrics["cache_miss_count"] += 1
            self._metrics["checkpoint_reference_count"] += (
                request.checkpoint_reference_count
            )
            existing = self._jobs.get(identity)
            if existing is not None:
                self._metrics["single_flight_join_count"] += 1
                if (
                    not existing.running
                    and request.priority < existing.request.priority
                ):
                    existing.request = TaskBuildRequest(
                        key=existing.request.key,
                        priority=request.priority,
                        estimate=existing.request.estimate,
                        checkpoint_reference_count=(
                            existing.request.checkpoint_reference_count
                        ),
                    )
                    heapq.heappush(
                        self._pending_heap,
                        (
                            int(request.priority),
                            existing.sequence,
                            identity,
                        ),
                    )
                    self._condition.notify_all()
                return None, existing.future
            attempt = self._attempts.get(identity, 0)
            if attempt:
                self._metrics["retry_count"] += 1
            self._attempts[identity] = attempt + 1
            sequence = next(self._sequence)
            job = _PendingBuild(
                request=request,
                future=Future(),
                sequence=sequence,
                enqueued_at=time.monotonic(),
            )
            self._jobs[identity] = job
            heapq.heappush(
                self._pending_heap,
                (int(request.priority), sequence, identity),
            )
            self._condition.notify_all()
            return None, job.future

    def _schedule(self) -> None:
        while True:
            with self._condition:
                job: _PendingBuild | None = None
                identity = ""
                while job is None:
                    if self._closed and not self._jobs:
                        return
                    while self._pending_heap:
                        priority, sequence, candidate = heapq.heappop(
                            self._pending_heap
                        )
                        current = self._jobs.get(candidate)
                        if (
                            current is None
                            or current.running
                            or current.future.cancelled()
                            or current.sequence != sequence
                            or int(current.request.priority) != priority
                        ):
                            continue
                        estimate = current.request.estimate
                        if (
                            self._active_builders
                            >= self.max_builder_workers
                            or self._in_flight_bytes
                            + estimate.in_flight_bytes
                            > self.max_in_flight_bytes
                            or self._in_flight_native_calls
                            + estimate.native_calls
                            > self.max_native_calls
                        ):
                            heapq.heappush(
                                self._pending_heap,
                                (priority, sequence, candidate),
                            )
                            break
                        job = current
                        identity = candidate
                        break
                    if job is None:
                        self._condition.wait()
                if not job.future.set_running_or_notify_cancel():
                    self._jobs.pop(identity, None)
                    self._condition.notify_all()
                    continue
                job.running = True
                estimate = job.request.estimate
                self._active_builders += 1
                self._in_flight_bytes += estimate.in_flight_bytes
                self._in_flight_native_calls += estimate.native_calls
                self._metrics["peak_builder_count"] = max(
                    self._metrics["peak_builder_count"],
                    self._active_builders,
                )
                self._metrics["peak_in_flight_bytes"] = max(
                    self._metrics["peak_in_flight_bytes"],
                    self._in_flight_bytes,
                )
                self._metrics["peak_in_flight_native_calls"] = max(
                    self._metrics["peak_in_flight_native_calls"],
                    self._in_flight_native_calls,
                )
                waits = self._metrics["priority_wait_seconds"]
                waits[job.request.priority.name] += (
                    time.monotonic() - job.enqueued_at
                )
            running = self._executor.submit(self._build, job.request)
            running.add_done_callback(
                lambda completed, build=job, key=identity: self._finish(
                    key, build, completed
                )
            )

    def _build(self, request: TaskBuildRequest) -> TaskArtifactReference:
        existing = self.store.load_platform(request.key)
        if existing is not None:
            return TaskArtifactReference.from_artifact(
                existing, self.store.root
            )
        product = self.builder(request.key)
        native_calls = 0
        if isinstance(product, TaskBuildProduct):
            payload = product.payload
            native_calls = product.native_calls
            artifact = self.store.commit_platform(request.key, payload)
        elif isinstance(product, PlatformTaskPayload):
            artifact = self.store.commit_platform(request.key, product)
        elif isinstance(product, PlatformTaskArtifact):
            artifact = product
        else:
            raise TaskCacheCoordinatorError(
                "task builder returned an unsupported product"
            )
        if artifact.key != request.key:
            raise TaskCacheCoordinatorError(
                "task builder artifact key differs"
            )
        if native_calls > request.estimate.native_calls:
            raise TaskCacheCoordinatorError(
                "task builder exceeded its native-call estimate"
            )
        loaded = self.store.load_platform(request.key)
        if loaded is None or loaded.artifact_sha256 != artifact.artifact_sha256:
            raise TaskCacheCoordinatorError(
                "task builder did not atomically publish its artifact"
            )
        artifact_bytes = sum(
            path.stat().st_size
            for path in loaded.root.rglob("*")
            if path.is_file() and not path.is_symlink()
        )
        with self._condition:
            self._metrics["build_count"] += 1
            self._metrics["actual_native_call_count"] += native_calls
            self._metrics["artifact_bytes"] += artifact_bytes
        return TaskArtifactReference.from_artifact(loaded, self.store.root)

    def _finish(
        self,
        identity: str,
        job: _PendingBuild,
        completed: Future[TaskArtifactReference],
    ) -> None:
        error = completed.exception()
        if error is None:
            job.future.set_result(completed.result())
        else:
            with self._condition:
                self._metrics["failure_count"] += 1
            job.future.set_exception(error)
        with self._condition:
            estimate = job.request.estimate
            self._active_builders -= 1
            self._in_flight_bytes -= estimate.in_flight_bytes
            self._in_flight_native_calls -= estimate.native_calls
            if self._jobs.get(identity) is job:
                self._jobs.pop(identity, None)
            self._condition.notify_all()

    def _serve_ipc(self) -> None:
        while True:
            message = self._ipc_request_queue.get()
            if message is None:
                return
            threading.Thread(
                target=self._respond_ipc,
                args=(message,),
                daemon=True,
            ).start()

    def _respond_ipc(self, message: object) -> None:
        if not isinstance(message, Mapping):
            return
        client_id = message.get("client_id")
        response = self._ipc_responses.get(client_id)
        if response is None:
            return
        request_id = message.get("request_id")
        context_token = message.get("context_token")
        contextual = isinstance(context_token, str) and bool(context_token)
        try:
            if contextual:
                if (
                    self._contextual_request_registrar is None
                    or self._context_responses is None
                    or not isinstance(message.get("context"), Mapping)
                ):
                    raise TaskCacheCoordinatorError(
                        "task cache contextual request authority is unavailable"
                    )
                self._contextual_request_registrar(message["context"])
            key = PlatformTaskKey(**message["key"])
            priority = TaskBuildPriority(message["priority"])
            reference = self.request(key, priority)
            value = {
                "request_id": request_id,
                "ok": True,
                "reference": reference.to_dict(),
            }
        except Exception as error:  # IPC must return a finite failure packet.
            value = {
                "request_id": request_id,
                "ok": False,
                "error": f"{type(error).__name__}: {error}",
            }
        if contextual:
            assert self._context_responses is not None
            self._context_responses[context_token] = value
        else:
            response.put(value)


@dataclass(slots=True)
class TaskCacheClient:
    """Picklable worker-side endpoint; it never builds a task inline."""

    cache_root: Path
    client_id: int
    request_queue: object
    response_queue: object
    context_responses: object | None = None
    _next_request_id: int = 0

    def get(
        self,
        key: PlatformTaskKey,
        priority: TaskBuildPriority,
        *,
        timeout: float | None = None,
    ) -> TaskArtifactReference:
        if not isinstance(key, PlatformTaskKey) or not isinstance(
            priority, TaskBuildPriority
        ):
            raise TypeError("task cache client request is invalid")
        request_id = self._next_request_id
        self._next_request_id += 1
        self.request_queue.put(
            {
                "client_id": self.client_id,
                "request_id": request_id,
                "key": key.to_dict(),
                "priority": int(priority),
            }
        )
        try:
            response = self.response_queue.get(timeout=timeout)
        except queue.Empty as error:
            raise TaskCacheCoordinatorError(
                "task cache IPC request timed out"
            ) from error
        if (
            not isinstance(response, Mapping)
            or response.get("request_id") != request_id
        ):
            raise TaskCacheCoordinatorError(
                "task cache IPC response identity differs"
            )
        if response.get("ok") is not True:
            raise TaskCacheCoordinatorError(
                f"task cache IPC build failed: {response.get('error')}"
            )
        reference = TaskArtifactReference.from_dict(response.get("reference"))
        if reference.key != key or reference.cache_root != self.cache_root:
            raise TaskCacheCoordinatorError(
                "task cache IPC artifact authority differs"
            )
        return reference

    def get_contextual(
        self,
        key: PlatformTaskKey,
        priority: TaskBuildPriority,
        *,
        context: Mapping[str, object],
        timeout: float | None = None,
    ) -> TaskArtifactReference:
        """Register immutable build context before one bounded IPC request."""
        if not isinstance(key, PlatformTaskKey) or not isinstance(
            priority, TaskBuildPriority
        ):
            raise TypeError("task cache contextual request is invalid")
        if not isinstance(context, Mapping) or self.context_responses is None:
            raise TypeError("task cache contextual authority is invalid")
        request_id = self._next_request_id
        self._next_request_id += 1
        token = (
            f"{os.getpid()}-{time.monotonic_ns()}-{self.client_id}-"
            f"{request_id}"
        )
        self.request_queue.put(
            {
                "client_id": self.client_id,
                "request_id": request_id,
                "context_token": token,
                "context": dict(context),
                "key": key.to_dict(),
                "priority": int(priority),
            }
        )
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            try:
                response = self.context_responses.pop(token)
                break
            except KeyError:
                if deadline is not None and time.monotonic() >= deadline:
                    raise TaskCacheCoordinatorError(
                        "task cache contextual IPC request timed out"
                    )
                time.sleep(0.005)
        if (
            not isinstance(response, Mapping)
            or response.get("request_id") != request_id
        ):
            raise TaskCacheCoordinatorError(
                "task cache contextual IPC response identity differs"
            )
        if response.get("ok") is not True:
            raise TaskCacheCoordinatorError(
                f"task cache contextual build failed: {response.get('error')}"
            )
        reference = TaskArtifactReference.from_dict(response.get("reference"))
        if reference.key != key or reference.cache_root != self.cache_root:
            raise TaskCacheCoordinatorError(
                "task cache contextual artifact authority differs"
            )
        return reference


__all__ = [
    "TaskArtifactReference",
    "TaskBuildEstimate",
    "TaskBuildPriority",
    "TaskBuildProduct",
    "TaskBuildRequest",
    "TaskCacheClient",
    "TaskCacheCoordinator",
    "TaskCacheCoordinatorError",
    "priority_for_scheduled_task",
]

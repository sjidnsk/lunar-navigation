"""One cumulative GPU-time budget shared by every training phase."""

from __future__ import annotations

import math
import json
import os
import tempfile
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from .config import (
    FORMAL_ROLLOUT_HORIZON,
    FORMAL_WORKER_RESPONSE_TIMEOUT_SECONDS,
)


if TYPE_CHECKING:
    from .config import ResolvedTrainingConfig


INITIAL_GPU_BUDGET_SECONDS = 86400.0
BUDGET_EXTENSION_BLOCK_SECONDS = 21600.0
# Backward-compatible name for the immutable initial limit. Runtime code must
# use TrainingBudget.total_gpu_seconds for an explicitly extended run.
TOTAL_GPU_BUDGET_SECONDS = INITIAL_GPU_BUDGET_SECONDS
# A legacy measured calibration probe may perform four macro steps across two
# bounded pool-step phases at the formal 1200-second worker timeout:
# 4 * 2 * 1200 = 9600 seconds. The remaining 120 seconds covers policy
# inference, backward/update, and CUDA synchronization.
CALIBRATION_PROBE_UPPER_BOUND_GPU_SECONDS = (
    8.0 * FORMAL_WORKER_RESPONSE_TIMEOUT_SECONDS + 120.0
)
# A horizon-12 update can legally wait for 12 sequential 1200-second macro
# actions. The remaining 600 seconds covers the optimizer and synchronization.
TRAINING_ROLLOUT_UPDATE_UPPER_BOUND_GPU_SECONDS = (
    float(FORMAL_ROLLOUT_HORIZON) * FORMAL_WORKER_RESPONSE_TIMEOUT_SECONDS + 600.0
)
ROLLOUT_HORIZON_TRANSITIONS_PER_WORKER = 64
ROLLOUT_HORIZON_THROUGHPUT_NEAR_TIE_RATIO = 0.99


class BudgetError(ValueError):
    """GPU budget state is malformed or used out of order."""


class BudgetExceededError(BudgetError):
    """A requested GPU interval exceeds the remaining cumulative budget."""


class CalibrationError(BudgetError):
    """Runtime calibration could not produce a safe frozen selection."""


@dataclass(frozen=True, slots=True)
class CalibrationMeasurement:
    workers: int
    micro_batch: int
    throughput_samples_per_second: float
    peak_gpu_memory_fraction: float
    planner_timeouts: int
    oom: bool
    gpu_seconds: float
    ipc_failures: int = 0
    optimizer_steps: int = 4

    def __post_init__(self) -> None:
        if type(self.workers) is not int or self.workers <= 0:
            raise CalibrationError("calibration workers must be positive")
        if type(self.micro_batch) is not int or self.micro_batch <= 0:
            raise CalibrationError("calibration micro-batch must be positive")
        if (
            not isinstance(self.throughput_samples_per_second, (int, float))
            or isinstance(self.throughput_samples_per_second, bool)
            or not math.isfinite(float(self.throughput_samples_per_second))
            or self.throughput_samples_per_second < 0.0
        ):
            raise CalibrationError("calibration throughput must be finite")
        if (
            not isinstance(self.peak_gpu_memory_fraction, (int, float))
            or isinstance(self.peak_gpu_memory_fraction, bool)
            or not math.isfinite(float(self.peak_gpu_memory_fraction))
            or not 0.0 <= self.peak_gpu_memory_fraction <= 1.0
        ):
            raise CalibrationError("calibration memory fraction is invalid")
        if type(self.planner_timeouts) is not int or self.planner_timeouts < 0:
            raise CalibrationError("planner timeout count is invalid")
        if type(self.oom) is not bool:
            raise CalibrationError("calibration OOM flag must be boolean")
        if type(self.ipc_failures) is not int or self.ipc_failures < 0:
            raise CalibrationError("calibration IPC failure count is invalid")
        if type(self.optimizer_steps) is not int or self.optimizer_steps < 0:
            raise CalibrationError("calibration optimizer step count is invalid")
        _finite_nonnegative(self.gpu_seconds, "calibration GPU seconds")


@dataclass(frozen=True, slots=True)
class HorizonCalibrationMeasurement:
    """One equal-work rollout/update probe for a candidate batch boundary."""

    workers: int
    micro_batch: int
    rollout_horizon: int
    total_transitions: int
    completed_updates: int
    throughput_transitions_per_second: float
    mean_update_wall_seconds: float
    peak_gpu_memory_fraction: float
    worker_wait_ratio: float
    planner_timeouts: int
    oom: bool
    gpu_seconds: float
    ipc_failures: int
    approximate_kl: float
    value_loss: float
    advantages_finite: bool
    returns_finite: bool
    update_boundary_continuity_verified: bool
    resume_digest: str | None
    resume_verified: bool
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("workers", self.workers),
            ("micro-batch", self.micro_batch),
            ("rollout horizon", self.rollout_horizon),
            ("total transitions", self.total_transitions),
        ):
            if type(value) is not int or value <= 0:
                raise CalibrationError(f"horizon calibration {name} must be positive")
        if type(self.completed_updates) is not int or self.completed_updates < 0:
            raise CalibrationError(
                "horizon calibration completed updates must be non-negative"
            )
        _finite_nonnegative(
            self.throughput_transitions_per_second,
            "horizon calibration throughput",
        )
        _finite_positive(
            self.mean_update_wall_seconds,
            "horizon calibration update wall time",
        )
        memory = _finite_nonnegative(
            self.peak_gpu_memory_fraction,
            "horizon calibration memory fraction",
        )
        wait_ratio = _finite_nonnegative(
            self.worker_wait_ratio,
            "horizon calibration worker wait ratio",
        )
        if memory > 1.0 or wait_ratio > 1.0:
            raise CalibrationError("horizon calibration fraction exceeds one")
        for name, value in (
            ("planner timeout count", self.planner_timeouts),
            ("IPC failure count", self.ipc_failures),
        ):
            if type(value) is not int or value < 0:
                raise CalibrationError(f"horizon calibration {name} is invalid")
        for name, value in (
            ("OOM flag", self.oom),
            ("advantage finite flag", self.advantages_finite),
            ("return finite flag", self.returns_finite),
            (
                "update-boundary continuity flag",
                self.update_boundary_continuity_verified,
            ),
            ("resume verification flag", self.resume_verified),
        ):
            if type(value) is not bool:
                raise CalibrationError(f"horizon calibration {name} must be boolean")
        _finite_nonnegative(self.gpu_seconds, "horizon calibration GPU seconds")
        _finite_number(self.approximate_kl, "horizon calibration approximate KL")
        _finite_nonnegative(self.value_loss, "horizon calibration value loss")
        if self.resume_digest is not None and not _is_sha256(self.resume_digest):
            raise CalibrationError("horizon calibration resume digest is invalid")
        if self.resume_verified and self.resume_digest is None:
            raise CalibrationError(
                "verified horizon calibration resume requires a digest"
            )
        if self.failure_reason is not None and (
            not isinstance(self.failure_reason, str) or not self.failure_reason
        ):
            raise CalibrationError(
                "horizon calibration failure reason must be non-empty"
            )


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    selected_workers: int
    selected_micro_batch: int
    selected_rollout_horizon: int
    compared_workers: tuple[int, ...]
    measurements: tuple[CalibrationMeasurement, ...]
    horizon_measurements: tuple[HorizonCalibrationMeasurement, ...]


def freeze_fixed_runtime_selection(
    *,
    config: "ResolvedTrainingConfig",
    budget: TrainingBudget,
    manifest_path: str | Path,
    selected_micro_batch: int = 4,
) -> CalibrationResult:
    """Freeze the explicit 24-worker runtime without benchmarking worker tiers."""
    from .config import ResolvedTrainingConfig

    if not isinstance(config, ResolvedTrainingConfig):
        raise CalibrationError("fixed runtime requires resolved training config")
    if config.run_kind != "formal":
        raise CalibrationError("fixed runtime selection is formal-only")
    if (
        config.parallel.worker_candidates != (24,)
        or config.parallel.preferred_workers != 24
        or config.ppo.rollout_horizon != FORMAL_ROLLOUT_HORIZON
        or dict(config.parallel.joint_workers)
        != {"WHEELED": 8, "LEGGED": 8, "HOPPER": 8}
    ):
        raise CalibrationError(
            "fixed runtime selection requires 24 workers and formal horizon"
        )
    if not isinstance(budget, TrainingBudget):
        raise CalibrationError("fixed runtime requires the run TrainingBudget")
    if type(selected_micro_batch) is not int or selected_micro_batch <= 0:
        raise CalibrationError("fixed runtime micro-batch must be positive")
    target = Path(manifest_path)
    if not target.is_absolute():
        raise CalibrationError("run manifest path must be absolute")
    if target.exists():
        raise CalibrationError("runtime calibration manifest is already frozen")
    if not target.parent.is_dir():
        raise CalibrationError("run manifest parent directory is missing")
    result = CalibrationResult(
        selected_workers=24,
        selected_micro_batch=selected_micro_batch,
        selected_rollout_horizon=config.ppo.rollout_horizon,
        compared_workers=(24,),
        measurements=(),
        horizon_measurements=(),
    )
    payload = {
        "schema_version": "lunar-training-run/v1",
        "frozen_config": config.as_frozen_dict(),
        "runtime_calibration": {
            "selection_mode": "operator-fixed/v1",
            "selected_workers": result.selected_workers,
            "selected_micro_batch": result.selected_micro_batch,
            "selected_rollout_horizon": result.selected_rollout_horizon,
            "compared_workers": [24],
            "measurements": [],
            "rollout_horizon_candidates": [],
            "horizon_transitions_per_worker": 0,
            "horizon_measurements": [],
        },
        "consumed_gpu_seconds": budget.consumed_gpu_seconds,
        "budget_extension_blocks": budget.budget_extension_blocks,
        "total_gpu_budget_seconds": budget.total_gpu_seconds,
    }
    _write_json_atomic_new(target, payload)
    return result


def select_qualified_rollout_horizon(
    measurements: Iterable[HorizonCalibrationMeasurement],
) -> int:
    """Prefer a shorter update when throughput is within one percent of best."""
    candidates = tuple(measurements)
    if not candidates or any(
        not isinstance(value, HorizonCalibrationMeasurement)
        or value.throughput_transitions_per_second <= 0.0
        for value in candidates
    ):
        raise CalibrationError(
            "rollout horizon selection requires qualified measurements"
        )
    fastest = max(
        value.throughput_transitions_per_second for value in candidates
    )
    near_ties = tuple(
        value
        for value in candidates
        if value.throughput_transitions_per_second
        >= fastest * ROLLOUT_HORIZON_THROUGHPUT_NEAR_TIE_RATIO
    )
    return min(
        near_ties,
        key=lambda value: (
            value.mean_update_wall_seconds,
            value.rollout_horizon,
            -value.throughput_transitions_per_second,
        ),
    ).rollout_horizon


@dataclass(slots=True)
class TrainingBudget:
    total_gpu_seconds: float = TOTAL_GPU_BUDGET_SECONDS
    consumed_gpu_seconds: float = 0.0
    budget_extension_blocks: int = 0
    _active_since: float | None = field(default=None, init=False, repr=False)
    _active_upper_bound: float | None = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if (
            type(self.budget_extension_blocks) is not int
            or self.budget_extension_blocks < 0
        ):
            raise BudgetError("budget extension blocks must be a non-negative integer")
        self.total_gpu_seconds = _finite_nonnegative(
            self.total_gpu_seconds, "total GPU seconds"
        )
        self.consumed_gpu_seconds = _finite_nonnegative(
            self.consumed_gpu_seconds, "consumed GPU seconds"
        )
        expected_total = (
            INITIAL_GPU_BUDGET_SECONDS
            + self.budget_extension_blocks * BUDGET_EXTENSION_BLOCK_SECONDS
        )
        if self.total_gpu_seconds != expected_total:
            raise BudgetError("total GPU budget does not match extension blocks")
        if self.consumed_gpu_seconds > self.total_gpu_seconds:
            raise BudgetExceededError("consumed GPU seconds exceed total budget")

    @property
    def remaining_gpu_seconds(self) -> float:
        return self.total_gpu_seconds - self.consumed_gpu_seconds

    @property
    def interval_active(self) -> bool:
        return self._active_since is not None

    @property
    def exhausted(self) -> bool:
        return self.consumed_gpu_seconds >= self.total_gpu_seconds

    def consume(self, gpu_seconds: float) -> None:
        interval = _finite_nonnegative(gpu_seconds, "GPU interval")
        if interval > self.remaining_gpu_seconds:
            self.consumed_gpu_seconds = self.total_gpu_seconds
            raise BudgetExceededError(
                "GPU interval exceeds remaining cumulative budget"
            )
        self.consumed_gpu_seconds += interval

    def extend_by_blocks(self, blocks: int) -> None:
        """Add explicit six-hour blocks without changing consumed GPU time."""
        if type(blocks) is not int or blocks <= 0:
            raise BudgetError("budget extension blocks must be a positive integer")
        self.budget_extension_blocks += blocks
        self.total_gpu_seconds = (
            INITIAL_GPU_BUDGET_SECONDS
            + self.budget_extension_blocks * BUDGET_EXTENSION_BLOCK_SECONDS
        )

    def begin_gpu_interval(
        self,
        *,
        monotonic_seconds: float,
        upper_bound_gpu_seconds: float,
    ) -> None:
        timestamp = _finite_nonnegative(monotonic_seconds, "GPU interval start")
        upper_bound = _finite_positive(
            upper_bound_gpu_seconds, "bounded unit upper bound"
        )
        if self._active_since is not None:
            raise BudgetError("GPU interval is already active")
        if self.remaining_gpu_seconds < upper_bound:
            self.consumed_gpu_seconds = self.total_gpu_seconds
            raise BudgetExceededError(
                "remaining GPU budget is below the bounded unit upper bound"
            )
        self._active_since = timestamp
        self._active_upper_bound = upper_bound

    def end_gpu_interval(
        self,
        *,
        monotonic_seconds: float,
        measured_gpu_seconds: float | None = None,
    ) -> None:
        timestamp = _finite_nonnegative(monotonic_seconds, "GPU interval end")
        if self._active_since is None:
            raise BudgetError("GPU interval is not active")
        if timestamp < self._active_since:
            raise BudgetError("GPU interval end precedes its start")
        interval = timestamp - self._active_since
        upper_bound = self._active_upper_bound
        self._active_since = None
        self._active_upper_bound = None
        actual = (
            interval if measured_gpu_seconds is None else measured_gpu_seconds
        )
        actual = _finite_nonnegative(actual, "GPU interval")
        if upper_bound is None:
            raise BudgetError("bounded unit upper bound is missing")
        if actual > upper_bound:
            self.consumed_gpu_seconds = self.total_gpu_seconds
            raise BudgetExceededError(
                "bounded unit exceeded its reserved upper bound"
            )
        self.consume(actual)

    @classmethod
    def from_checkpoint(cls, checkpoint: object) -> "TrainingBudget":
        consumed = getattr(checkpoint, "consumed_gpu_seconds", None)
        total = getattr(checkpoint, "total_gpu_budget_seconds", None)
        blocks = getattr(checkpoint, "budget_extension_blocks", None)
        return cls(
            total_gpu_seconds=total,
            consumed_gpu_seconds=consumed,
            budget_extension_blocks=blocks,
        )


def extend_budget_manifest(
    manifest_path: str | Path, *, blocks: int
) -> TrainingBudget:
    """Atomically add explicit six-hour blocks to one existing run manifest."""
    if type(blocks) is not int or blocks <= 0:
        raise BudgetError("budget extension blocks must be a positive integer")
    target = Path(manifest_path)
    if target.is_symlink() or not target.is_file():
        raise BudgetError("run budget manifest is missing or unsafe")
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise BudgetError("run budget manifest is invalid") from error
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != "lunar-training-run/v1"
    ):
        raise BudgetError("run budget manifest schema is invalid")
    try:
        budget = TrainingBudget(
            total_gpu_seconds=payload.get("total_gpu_budget_seconds"),
            consumed_gpu_seconds=payload.get("consumed_gpu_seconds"),
            budget_extension_blocks=payload.get("budget_extension_blocks"),
        )
    except BudgetError as error:
        raise BudgetError("run budget identity is invalid") from error
    budget.extend_by_blocks(blocks)
    payload["budget_extension_blocks"] = budget.budget_extension_blocks
    payload["total_gpu_budget_seconds"] = budget.total_gpu_seconds
    payload["budget_state"] = (
        "exhausted" if budget.exhausted else "active"
    )
    _write_json_atomic_replace(target, payload)
    return budget


def calibrate_runtime(
    *,
    config: "ResolvedTrainingConfig",
    workload: Callable[[int, int], CalibrationMeasurement],
    horizon_workload: Callable[
        [int, int, int, int], HorizonCalibrationMeasurement
    ]
    | None = None,
    budget: TrainingBudget,
    manifest_path: str | Path,
    micro_batch_candidates: Iterable[int] = (1, 2, 4, 8, 16, 32),
    horizon_candidates: Iterable[int] = (16, 32, 64),
    horizon_transitions_per_worker: int = (
        ROLLOUT_HORIZON_TRANSITIONS_PER_WORKER
    ),
    clock: Callable[[], float] = time.monotonic,
) -> CalibrationResult:
    """Measure both worker candidates and freeze the safe fastest selection."""
    from .config import (
        ROLLOUT_HORIZON_CANDIDATES,
        ResolvedTrainingConfig,
        with_rollout_horizon,
    )

    if not isinstance(config, ResolvedTrainingConfig):
        raise CalibrationError("calibration requires resolved training config")
    if not isinstance(budget, TrainingBudget):
        raise CalibrationError("calibration requires the run TrainingBudget")
    if not callable(workload):
        raise CalibrationError("calibration workload must be callable")
    if horizon_workload is not None and not callable(horizon_workload):
        raise CalibrationError("horizon calibration workload must be callable")
    if not callable(clock):
        raise CalibrationError("calibration clock must be callable")
    target = Path(manifest_path)
    if not target.is_absolute():
        raise CalibrationError("run manifest path must be absolute")
    if target.exists():
        raise CalibrationError("runtime calibration manifest is already frozen")
    if not target.parent.is_dir():
        raise CalibrationError("run manifest parent directory is missing")
    micro_batches = tuple(micro_batch_candidates)
    if (
        not micro_batches
        or any(type(value) is not int or value <= 0 for value in micro_batches)
        or any(
            right <= left
            for left, right in zip(micro_batches, micro_batches[1:])
        )
    ):
        raise CalibrationError(
            "micro-batch candidates must be strictly increasing positive integers"
        )
    horizons = tuple(horizon_candidates)
    if horizon_workload is not None and horizons != ROLLOUT_HORIZON_CANDIDATES:
        raise CalibrationError("formal horizon calibration must compare 16, 32, 64")
    if (
        type(horizon_transitions_per_worker) is not int
        or horizon_transitions_per_worker <= 0
        or any(horizon_transitions_per_worker % value for value in horizons)
    ):
        raise CalibrationError(
            "horizon calibration work must divide equally across every candidate"
        )

    measurements: list[CalibrationMeasurement] = []
    best: dict[int, CalibrationMeasurement] = {}
    worker_safe: dict[int, bool] = {}
    for workers in config.parallel.worker_candidates:
        safe = True
        for micro_batch in micro_batches:
            budget.begin_gpu_interval(
                monotonic_seconds=clock(),
                upper_bound_gpu_seconds=(
                    CALIBRATION_PROBE_UPPER_BOUND_GPU_SECONDS
                ),
            )
            try:
                measurement = workload(workers, micro_batch)
                if not isinstance(measurement, CalibrationMeasurement):
                    raise CalibrationError(
                        "calibration workload must return CalibrationMeasurement"
                    )
                if (
                    measurement.workers != workers
                    or measurement.micro_batch != micro_batch
                ):
                    raise CalibrationError(
                        "calibration workload mislabeled a measurement"
                    )
            except BaseException:
                budget.end_gpu_interval(monotonic_seconds=clock())
                raise
            else:
                budget.end_gpu_interval(
                    monotonic_seconds=clock(),
                    measured_gpu_seconds=measurement.gpu_seconds,
                )
            measurements.append(measurement)
            if measurement.ipc_failures:
                raise CalibrationError(
                    "runtime calibration encountered an IPC failure"
                )
            if (
                not measurement.oom
                and measurement.optimizer_steps
                != config.ppo.epochs_per_update
            ):
                raise CalibrationError(
                    "runtime calibration did not complete the frozen PPO epochs"
                )
            if measurement.oom or measurement.planner_timeouts:
                safe = False
                break
            if (
                measurement.peak_gpu_memory_fraction
                > config.parallel.gpu_memory_fraction_max
            ):
                break
            best[workers] = measurement
        worker_safe[workers] = safe and workers in best

    compared = tuple(config.parallel.worker_candidates)
    if compared != (18, 24, 30):
        raise CalibrationError(
            "runtime calibration must compare 18, 24, and 30 workers"
        )
    if not worker_safe.get(18, False):
        raise CalibrationError("18-worker control did not produce a safe result")
    selected_workers = max(
        (workers for workers in compared if worker_safe.get(workers, False)),
        key=lambda workers: (
            best[workers].throughput_samples_per_second,
            workers,
        ),
    )
    selected_micro_batch = best[selected_workers].micro_batch
    horizon_measurements: list[HorizonCalibrationMeasurement] = []
    selected_rollout_horizon = config.ppo.rollout_horizon
    if horizon_workload is not None:
        qualified_horizons: list[HorizonCalibrationMeasurement] = []
        expected_total = selected_workers * horizon_transitions_per_worker
        for rollout_horizon in horizons:
            budget.begin_gpu_interval(
                monotonic_seconds=clock(),
                upper_bound_gpu_seconds=(
                    CALIBRATION_PROBE_UPPER_BOUND_GPU_SECONDS
                ),
            )
            try:
                measurement = horizon_workload(
                    selected_workers,
                    selected_micro_batch,
                    rollout_horizon,
                    horizon_transitions_per_worker,
                )
                if not isinstance(measurement, HorizonCalibrationMeasurement):
                    raise CalibrationError(
                        "horizon workload must return HorizonCalibrationMeasurement"
                    )
                if (
                    measurement.workers != selected_workers
                    or measurement.micro_batch != selected_micro_batch
                    or measurement.rollout_horizon != rollout_horizon
                    or measurement.total_transitions != expected_total
                ):
                    raise CalibrationError(
                        "horizon calibration workload mislabeled unequal work"
                    )
            except BaseException:
                budget.end_gpu_interval(monotonic_seconds=clock())
                raise
            else:
                budget.end_gpu_interval(
                    monotonic_seconds=clock(),
                    measured_gpu_seconds=measurement.gpu_seconds,
                )
            horizon_measurements.append(measurement)
            if (
                not measurement.oom
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
                == horizon_transitions_per_worker // rollout_horizon
            ):
                qualified_horizons.append(measurement)
        if not qualified_horizons:
            raise CalibrationError("no rollout horizon passed semantic calibration")
        selected_rollout_horizon = select_qualified_rollout_horizon(
            qualified_horizons
        )
    frozen_config = with_rollout_horizon(config, selected_rollout_horizon)
    result = CalibrationResult(
        selected_workers=selected_workers,
        selected_micro_batch=selected_micro_batch,
        selected_rollout_horizon=selected_rollout_horizon,
        compared_workers=compared,
        measurements=tuple(measurements),
        horizon_measurements=tuple(horizon_measurements),
    )
    payload = {
        "schema_version": "lunar-training-run/v1",
        "frozen_config": frozen_config.as_frozen_dict(),
        "runtime_calibration": {
            "selected_workers": result.selected_workers,
            "selected_micro_batch": result.selected_micro_batch,
            "selected_rollout_horizon": result.selected_rollout_horizon,
            "compared_workers": list(result.compared_workers),
            "measurements": [asdict(value) for value in result.measurements],
            "rollout_horizon_candidates": (
                list(horizons) if horizon_workload is not None else []
            ),
            "horizon_transitions_per_worker": (
                horizon_transitions_per_worker
                if horizon_workload is not None
                else 0
            ),
            "horizon_measurements": [
                asdict(value) for value in result.horizon_measurements
            ],
        },
        "consumed_gpu_seconds": budget.consumed_gpu_seconds,
        "budget_extension_blocks": budget.budget_extension_blocks,
        "total_gpu_budget_seconds": budget.total_gpu_seconds,
    }
    _write_json_atomic_new(target, payload)
    return result


def _write_json_atomic_new(path: Path, payload: dict[str, object]) -> None:
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
        if path.exists():
            raise CalibrationError(
                "runtime calibration manifest is already frozen"
            )
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception as error:
        if temporary.exists():
            temporary.unlink()
        if isinstance(error, CalibrationError):
            raise
        raise CalibrationError("runtime calibration manifest write failed") from error


def _write_json_atomic_replace(path: Path, payload: dict[str, object]) -> None:
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
        raise BudgetError("run budget manifest update failed") from error


def _finite_nonnegative(value: object, name: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise BudgetError(f"{name} must be finite and non-negative")
    return float(value)


def _finite_number(value: object, name: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise BudgetError(f"{name} must be finite")
    return float(value)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite_positive(value: object, name: str) -> float:
    result = _finite_nonnegative(value, name)
    if result <= 0.0:
        raise BudgetError(f"{name} must be positive")
    return result


__all__ = [
    "TOTAL_GPU_BUDGET_SECONDS",
    "INITIAL_GPU_BUDGET_SECONDS",
    "BUDGET_EXTENSION_BLOCK_SECONDS",
    "CALIBRATION_PROBE_UPPER_BOUND_GPU_SECONDS",
    "TRAINING_ROLLOUT_UPDATE_UPPER_BOUND_GPU_SECONDS",
    "BudgetError",
    "BudgetExceededError",
    "CalibrationError",
    "CalibrationMeasurement",
    "HorizonCalibrationMeasurement",
    "CalibrationResult",
    "ROLLOUT_HORIZON_TRANSITIONS_PER_WORKER",
    "ROLLOUT_HORIZON_THROUGHPUT_NEAR_TIE_RATIO",
    "TrainingBudget",
    "calibrate_runtime",
    "freeze_fixed_runtime_selection",
    "select_qualified_rollout_horizon",
    "extend_budget_manifest",
]

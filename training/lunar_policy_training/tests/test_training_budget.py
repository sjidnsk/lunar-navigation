from __future__ import annotations

import pathlib
import sys
import json

import pytest


PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "model_contract"))

from lunar_policy_training.budget import (  # noqa: E402
    BudgetExceededError,
    CalibrationError,
    CalibrationMeasurement,
    TrainingBudget,
    calibrate_runtime,
)
from lunar_policy_training.config import load_training_config  # noqa: E402


def test_training_budget_uses_fixed_total_and_rejects_overspend() -> None:
    """Would fail if a phase could silently exceed the shared 86400-second cap."""
    budget = TrainingBudget(total_gpu_seconds=86400.0)

    budget.consume(86399.0)

    assert budget.remaining_gpu_seconds == 1.0
    with pytest.raises(BudgetExceededError, match="remaining"):
        budget.consume(1.001)
    assert budget.remaining_gpu_seconds == 0.0


def test_active_gpu_intervals_exclude_paused_wall_clock() -> None:
    """Would fail if time between active intervals consumed the GPU budget."""
    budget = TrainingBudget()

    budget.begin_gpu_interval(monotonic_seconds=10.0)
    budget.end_gpu_interval(monotonic_seconds=13.5)
    budget.begin_gpu_interval(monotonic_seconds=1013.5)
    budget.end_gpu_interval(monotonic_seconds=1014.0)

    assert budget.consumed_gpu_seconds == 4.0


class _CalibrationProbe:
    def __init__(self, bad_24: str | None = None) -> None:
        self.calls: list[tuple[int, int]] = []
        self.bad_24 = bad_24

    def __call__(self, workers: int, micro_batch: int) -> CalibrationMeasurement:
        self.calls.append((workers, micro_batch))
        peak = {1: 0.40, 2: 0.80, 4: 0.95}[micro_batch]
        throughput = (100.0 if workers == 18 else 120.0) + micro_batch
        oom = workers == 24 and self.bad_24 == "oom" and micro_batch == 2
        planner_timeouts = (
            1
            if workers == 24 and self.bad_24 == "timeout" and micro_batch == 2
            else 0
        )
        if workers == 24 and self.bad_24 == "throughput":
            throughput = 50.0 + micro_batch
        return CalibrationMeasurement(
            workers=workers,
            micro_batch=micro_batch,
            throughput_samples_per_second=throughput,
            peak_gpu_memory_fraction=peak,
            planner_timeouts=planner_timeouts,
            oom=oom,
            gpu_seconds=1.0,
            ipc_failures=0,
        )


def test_calibration_really_compares_18_and_24_and_freezes_manifest(
    tmp_path: pathlib.Path,
) -> None:
    """Would fail if preferred=24 skipped the required 18-worker control."""
    config = load_training_config(
        REPOSITORY_ROOT / "training/configs/rtx4080_super_v3_joint.yaml"
    )
    budget = TrainingBudget()
    probe = _CalibrationProbe()
    manifest = tmp_path / "run-manifest.json"

    result = calibrate_runtime(
        config=config,
        workload=probe,
        budget=budget,
        manifest_path=manifest,
        micro_batch_candidates=(1, 2, 4),
    )

    assert probe.calls == [
        (18, 1),
        (18, 2),
        (18, 4),
        (24, 1),
        (24, 2),
        (24, 4),
    ]
    assert result.selected_workers == 24
    assert result.selected_micro_batch == 2
    assert result.compared_workers == (18, 24)
    assert budget.consumed_gpu_seconds == 6.0
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["runtime_calibration"]["selected_workers"] == 24
    assert payload["runtime_calibration"]["selected_micro_batch"] == 2
    with pytest.raises(CalibrationError, match="frozen"):
        calibrate_runtime(
            config=config,
            workload=probe,
            budget=budget,
            manifest_path=manifest,
            micro_batch_candidates=(1, 2, 4),
        )


@pytest.mark.parametrize("bad_24", ["oom", "timeout", "throughput"])
def test_calibration_falls_back_to_18_when_24_is_not_safe_or_faster(
    tmp_path: pathlib.Path, bad_24: str
) -> None:
    """Would fail if preferred workers overrode OOM, timeout, or throughput gates."""
    config = load_training_config(
        REPOSITORY_ROOT / "training/configs/rtx4080_super_v3_joint.yaml"
    )

    result = calibrate_runtime(
        config=config,
        workload=_CalibrationProbe(bad_24),
        budget=TrainingBudget(),
        manifest_path=tmp_path / f"{bad_24}.json",
        micro_batch_candidates=(1, 2, 4),
    )

    assert result.selected_workers == 18


def test_calibration_rejects_ipc_failure_instead_of_labeling_planner_timeout(
    tmp_path: pathlib.Path,
) -> None:
    """Would fail if a dead/late worker were treated as a normal planner transition."""
    config = load_training_config(
        REPOSITORY_ROOT / "training/configs/rtx4080_super_v3_joint.yaml"
    )

    def ipc_failure(workers: int, micro_batch: int) -> CalibrationMeasurement:
        return CalibrationMeasurement(
            workers=workers,
            micro_batch=micro_batch,
            throughput_samples_per_second=0.0,
            peak_gpu_memory_fraction=0.0,
            planner_timeouts=0,
            oom=False,
            gpu_seconds=0.25,
            ipc_failures=1,
        )

    budget = TrainingBudget()
    with pytest.raises(CalibrationError, match="IPC"):
        calibrate_runtime(
            config=config,
            workload=ipc_failure,
            budget=budget,
            manifest_path=tmp_path / "ipc.json",
            micro_batch_candidates=(1,),
        )

    assert budget.consumed_gpu_seconds == 0.25


def test_budget_refuses_to_start_another_bounded_unit_when_empty() -> None:
    """Would fail if a new inference/update unit started after the shared cap."""
    budget = TrainingBudget()
    budget.consume(86400.0)

    with pytest.raises(BudgetExceededError, match="remaining"):
        budget.begin_gpu_interval(monotonic_seconds=5.0)

    assert budget.interval_active is False


def test_calibration_unexpected_exit_still_settles_shared_budget(
    tmp_path: pathlib.Path,
) -> None:
    """Would fail if a failed calibration probe left active GPU time uncharged."""
    config = load_training_config(
        REPOSITORY_ROOT / "training/configs/rtx4080_super_v3_joint.yaml"
    )
    timestamps = iter((20.0, 23.0))
    budget = TrainingBudget()

    with pytest.raises(RuntimeError, match="probe crashed"):
        calibrate_runtime(
            config=config,
            workload=lambda workers, micro_batch: (_ for _ in ()).throw(
                RuntimeError("probe crashed")
            ),
            budget=budget,
            manifest_path=tmp_path / "failed.json",
            micro_batch_candidates=(1,),
            clock=lambda: next(timestamps),
        )

    assert budget.consumed_gpu_seconds == 3.0
    assert budget.interval_active is False

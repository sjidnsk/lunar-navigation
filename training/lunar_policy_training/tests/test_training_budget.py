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
    BUDGET_EXTENSION_BLOCK_SECONDS,
    INITIAL_GPU_BUDGET_SECONDS,
    BudgetError,
    BudgetExceededError,
    CalibrationError,
    HorizonCalibrationMeasurement,
    CalibrationMeasurement,
    TrainingBudget,
    calibrate_runtime,
    extend_budget_manifest,
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

    budget.begin_gpu_interval(
        monotonic_seconds=10.0, upper_bound_gpu_seconds=10.0
    )
    budget.end_gpu_interval(monotonic_seconds=13.5)
    budget.begin_gpu_interval(
        monotonic_seconds=1013.5, upper_bound_gpu_seconds=10.0
    )
    budget.end_gpu_interval(monotonic_seconds=1014.0)

    assert budget.consumed_gpu_seconds == 4.0


def test_bounded_interval_saturates_when_actual_exceeds_reserve() -> None:
    budget = TrainingBudget()
    budget.begin_gpu_interval(
        monotonic_seconds=0.0, upper_bound_gpu_seconds=600.0
    )

    with pytest.raises(BudgetExceededError, match="reserved upper bound"):
        budget.end_gpu_interval(monotonic_seconds=600.001)

    assert budget.exhausted is True
    assert budget.consumed_gpu_seconds == 86400.0
    assert budget.interval_active is False


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


class _HorizonProbe:
    def __init__(self) -> None:
        self.calls: list[tuple[int, int, int, int]] = []

    def __call__(
        self,
        workers: int,
        micro_batch: int,
        rollout_horizon: int,
        transitions_per_worker: int,
    ) -> HorizonCalibrationMeasurement:
        self.calls.append(
            (workers, micro_batch, rollout_horizon, transitions_per_worker)
        )
        throughput = {16: 90.0, 32: 120.0, 64: 120.0}[rollout_horizon]
        update_wall = {16: 2.0, 32: 3.0, 64: 5.0}[rollout_horizon]
        return HorizonCalibrationMeasurement(
            workers=workers,
            micro_batch=micro_batch,
            rollout_horizon=rollout_horizon,
            total_transitions=workers * transitions_per_worker,
            completed_updates=transitions_per_worker // rollout_horizon,
            throughput_transitions_per_second=throughput,
            mean_update_wall_seconds=update_wall,
            peak_gpu_memory_fraction=0.5,
            worker_wait_ratio=0.25,
            planner_timeouts=0,
            oom=False,
            gpu_seconds=1.0,
            ipc_failures=0,
            approximate_kl=0.01,
            value_loss=0.2,
            advantages_finite=True,
            returns_finite=True,
            update_boundary_continuity_verified=True,
            resume_digest="a" * 64,
            resume_verified=True,
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
    assert payload["budget_extension_blocks"] == 0
    assert payload["total_gpu_budget_seconds"] == 86400
    with pytest.raises(CalibrationError, match="frozen"):
        calibrate_runtime(
            config=config,
            workload=probe,
            budget=budget,
            manifest_path=manifest,
            micro_batch_candidates=(1, 2, 4),
        )


def test_rollout_horizon_calibration_uses_equal_work_and_freezes_selection(
    tmp_path: pathlib.Path,
) -> None:
    """Would fail if horizon changed episode semantics or compared unequal work."""
    config = load_training_config(
        REPOSITORY_ROOT / "training/configs/rtx4080_super_v3_joint.yaml"
    )
    horizon_probe = _HorizonProbe()

    result = calibrate_runtime(
        config=config,
        workload=_CalibrationProbe(),
        horizon_workload=horizon_probe,
        budget=TrainingBudget(),
        manifest_path=tmp_path / "run-manifest.json",
        micro_batch_candidates=(1, 2, 4),
    )

    assert horizon_probe.calls == [
        (24, 2, 16, 64),
        (24, 2, 32, 64),
        (24, 2, 64, 64),
    ]
    assert result.selected_rollout_horizon == 32
    assert {item.total_transitions for item in result.horizon_measurements} == {
        24 * 64
    }
    payload = json.loads(
        (tmp_path / "run-manifest.json").read_text(encoding="utf-8")
    )
    assert payload["frozen_config"]["ppo"]["rollout_horizon"] == 32
    assert payload["runtime_calibration"]["rollout_horizon_candidates"] == [
        16,
        32,
        64,
    ]
    assert payload["runtime_calibration"]["selected_rollout_horizon"] == 32
    assert len(payload["runtime_calibration"]["horizon_measurements"]) == 3


def test_rollout_horizon_selection_rejects_fast_but_semantically_invalid_candidate(
    tmp_path: pathlib.Path,
) -> None:
    config = load_training_config(
        REPOSITORY_ROOT / "training/configs/rtx4080_super_v3_joint.yaml"
    )

    def probe(
        workers: int,
        micro_batch: int,
        rollout_horizon: int,
        transitions_per_worker: int,
    ) -> HorizonCalibrationMeasurement:
        valid = rollout_horizon != 64
        return HorizonCalibrationMeasurement(
            workers=workers,
            micro_batch=micro_batch,
            rollout_horizon=rollout_horizon,
            total_transitions=workers * transitions_per_worker,
            completed_updates=transitions_per_worker // rollout_horizon,
            throughput_transitions_per_second=float(rollout_horizon * 10),
            mean_update_wall_seconds=float(rollout_horizon),
            peak_gpu_memory_fraction=0.5,
            worker_wait_ratio=0.25,
            planner_timeouts=0,
            oom=False,
            gpu_seconds=1.0,
            ipc_failures=0,
            approximate_kl=0.01,
            value_loss=0.2,
            advantages_finite=True,
            returns_finite=True,
            update_boundary_continuity_verified=valid,
            resume_digest="b" * 64,
            resume_verified=valid,
        )

    result = calibrate_runtime(
        config=config,
        workload=_CalibrationProbe(),
        horizon_workload=probe,
        budget=TrainingBudget(),
        manifest_path=tmp_path / "run-manifest.json",
        micro_batch_candidates=(1, 2, 4),
    )

    assert result.selected_rollout_horizon == 32


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
        budget.begin_gpu_interval(
            monotonic_seconds=5.0, upper_bound_gpu_seconds=1.0
        )

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


def test_calibration_does_not_start_probe_below_bounded_unit_reserve(
    tmp_path: pathlib.Path,
) -> None:
    """Would fail if the final partial budget still launched an 18/24 probe."""
    config = load_training_config(
        REPOSITORY_ROOT / "training/configs/rtx4080_super_v3_joint.yaml"
    )
    calls: list[tuple[int, int]] = []
    budget = TrainingBudget(
        consumed_gpu_seconds=(
            86400.0 - 600.0 + 1.0
        )
    )

    with pytest.raises(BudgetExceededError, match="bounded unit"):
        calibrate_runtime(
            config=config,
            workload=lambda workers, micro_batch: calls.append(
                (workers, micro_batch)
            ),
            budget=budget,
            manifest_path=tmp_path / "must-not-start.json",
            micro_batch_candidates=(1,),
        )

    assert calls == []
    assert budget.consumed_gpu_seconds == 86400.0
    assert budget.interval_active is False
    assert not (tmp_path / "must-not-start.json").exists()


def _write_budget_manifest(
    path: pathlib.Path,
    *,
    blocks: object = 0,
    total: object = 86400,
    consumed: object = 123.5,
) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "lunar-training-run/v1",
                "budget_extension_blocks": blocks,
                "total_gpu_budget_seconds": total,
                "consumed_gpu_seconds": consumed,
            }
        ),
        encoding="utf-8",
    )


@pytest.mark.parametrize("blocks", [0, -1, 1.5, True, "1"])
def test_budget_extension_rejects_non_positive_integer_blocks(
    tmp_path: pathlib.Path, blocks: object
) -> None:
    """Would fail if an implicit, fractional, or decreasing extension were accepted."""
    manifest = tmp_path / "run-manifest.json"
    _write_budget_manifest(manifest)

    with pytest.raises(BudgetError, match="positive integer"):
        extend_budget_manifest(manifest, blocks=blocks)

    assert json.loads(manifest.read_text(encoding="utf-8"))[
        "total_gpu_budget_seconds"
    ] == 86400


def test_budget_extension_accumulates_six_hour_blocks_without_resetting_consumed(
    tmp_path: pathlib.Path,
) -> None:
    """Would fail if repeated extension reset time or replaced prior blocks."""
    manifest = tmp_path / "run-manifest.json"
    _write_budget_manifest(manifest)

    first = extend_budget_manifest(manifest, blocks=2)
    second = extend_budget_manifest(manifest, blocks=3)

    assert INITIAL_GPU_BUDGET_SECONDS == 86400
    assert BUDGET_EXTENSION_BLOCK_SECONDS == 21600
    assert first.budget_extension_blocks == 2
    assert first.total_gpu_seconds == 86400 + 2 * 21600
    assert second.budget_extension_blocks == 5
    assert second.total_gpu_seconds == 86400 + 5 * 21600
    assert second.consumed_gpu_seconds == 123.5
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["budget_extension_blocks"] == 5
    assert payload["total_gpu_budget_seconds"] == 194400
    assert payload["consumed_gpu_seconds"] == 123.5
    assert list(tmp_path.iterdir()) == [manifest]


@pytest.mark.parametrize(
    ("blocks", "total", "consumed"),
    [(1, 86400, 1.0), (0, 86401, 1.0), (0, 86400, 86401.0)],
)
def test_budget_extension_rejects_stale_or_malformed_manifest_identity(
    tmp_path: pathlib.Path,
    blocks: object,
    total: object,
    consumed: object,
) -> None:
    """Would fail if extension repaired a stale identity instead of failing closed."""
    manifest = tmp_path / "run-manifest.json"
    _write_budget_manifest(
        manifest, blocks=blocks, total=total, consumed=consumed
    )

    with pytest.raises(BudgetError, match="budget"):
        extend_budget_manifest(manifest, blocks=1)

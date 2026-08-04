from __future__ import annotations

import pathlib
import os
import signal
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
import yaml

import lunar_policy_training.cli as cli_module


PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "model_contract"))

from lunar_policy_training.budget import TrainingBudget  # noqa: E402
from lunar_policy_training.cli import (  # noqa: E402
    ArtifactRootError,
    SignalStopFlag,
    TrainingBoundaryLoop,
    _checkpoint_target,
    _freeze_task4_manifest,
    _load_calibrated_run_state,
    _run_curriculum_training,
    _update_run_manifest,
    build_parser,
    validate_artifact_root,
)
from lunar_policy_training.curriculum import CurriculumSchedule
from lunar_policy_training.reward import reward_weights_sha256


def test_task_four_cli_registers_calibrate_train_resume_and_evaluate() -> None:
    """Would fail if the approved calibration/evaluation handoff were missing."""
    parser = build_parser()

    calibrate = parser.parse_args(
        [
            "calibrate",
            "--config",
            "training/configs/rtx4080_super_smoke.yaml",
            "--artifact-root",
            "/tmp/lunar-task4",
        ]
    )

    train = parser.parse_args(
        [
            "train",
            "--config",
            "training/configs/rtx4080_super_smoke.yaml",
            "--artifact-root",
            "/tmp/lunar-task3",
        ]
    )
    resume = parser.parse_args(
        [
            "resume",
            "--artifact-root",
            "/tmp/lunar-task4",
            "--checkpoint",
            "/tmp/lunar-task4/checkpoints/latest.pt",
        ]
    )
    evaluate = parser.parse_args(
        [
            "evaluate",
            "--checkpoint",
            "/tmp/lunar-task4/latest.pt",
            "--gate",
            "training/configs/candidate_gate_v1.yaml",
            "--artifact-root",
            "/tmp/lunar-task4",
        ]
    )

    assert calibrate.command == "calibrate"
    assert train.command == "train"
    assert train.max_updates is None
    assert resume.command == "resume"
    assert resume.max_updates is None
    assert evaluate.command == "evaluate"


def test_task_five_authoritative_checkpoint_paths_are_versioned_under_directory(
    tmp_path: pathlib.Path,
) -> None:
    """Would fail if Task 5 train/resume used the legacy root-level checkpoint."""
    assert _checkpoint_target(tmp_path, kind="latest", global_step=12) == (
        tmp_path / "checkpoints/latest.pt"
    )
    assert _checkpoint_target(tmp_path, kind="candidate", global_step=12) == (
        tmp_path / "checkpoints/candidate-step-12.pt"
    )


def test_artifact_root_must_be_explicit_absolute_and_outside_repository(
    tmp_path: pathlib.Path,
) -> None:
    """Would fail if checkpoints or logs could be written into the Git tree."""
    with pytest.raises(ArtifactRootError, match="absolute"):
        validate_artifact_root(
            pathlib.Path("relative/run"), repository_root=REPOSITORY_ROOT
        )
    with pytest.raises(ArtifactRootError, match="outside"):
        validate_artifact_root(
            REPOSITORY_ROOT / "training-output", repository_root=REPOSITORY_ROOT
        )

    assert validate_artifact_root(
        tmp_path / "run", repository_root=REPOSITORY_ROOT
    ) == tmp_path / "run"


def _write_calibrated_manifest(root: pathlib.Path, *, consumed: float = 12.5) -> None:
    root.mkdir()
    config = yaml.safe_load(
        (REPOSITORY_ROOT / "training/configs/rtx4080_super_smoke.yaml").read_text(
            encoding="utf-8"
        )
    )
    (root / "run-manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "lunar-training-run/v1",
                "frozen_config": config,
                "runtime_calibration": {
                    "selected_workers": 24,
                    "selected_micro_batch": 2,
                    "compared_workers": [18, 24],
                    "measurements": [],
                },
                "consumed_gpu_seconds": consumed,
            }
        ),
        encoding="utf-8",
    )
    _freeze_task4_manifest(
        root / "run-manifest.json",
        schedule=CurriculumSchedule(),
        reward_hash=reward_weights_sha256(),
        reward_seed_results=(
            {"seed": 4081, "minimum_platform_score": 0.5, "report_sha256": "1" * 64},
            {"seed": 4082, "minimum_platform_score": 0.5, "report_sha256": "2" * 64},
            {"seed": 4083, "minimum_platform_score": 0.5, "report_sha256": "3" * 64},
        ),
    )


def test_calibration_freezes_reward_schedule_and_one_shared_budget(
    tmp_path: pathlib.Path,
) -> None:
    """Would fail if train/evaluate could create a fresh Task 4 run identity."""
    root = tmp_path / "calibrated"
    _write_calibrated_manifest(root, consumed=123.5)

    state = _load_calibrated_run_state(root)

    assert state.budget.consumed_gpu_seconds == 123.5
    assert state.reward_hash == reward_weights_sha256()
    assert state.scenario_schedule_id == CurriculumSchedule().scenario_schedule_id
    assert state.formal_seed == 4080
    assert state.reward_calibration_seeds == (4081, 4082, 4083)
    assert state.allocation == {"WHEELED": 8, "LEGGED": 8, "HOPPER": 8}


def test_train_consumes_existing_calibrated_root_without_recalibration(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Would fail if public train rejected or recalibrated an existing run root."""
    root = tmp_path / "calibrated"
    _write_calibrated_manifest(root)
    sentinel = object()
    monkeypatch.setattr(cli_module, "_run_updates", lambda **kwargs: sentinel)

    result = cli_module._start_training_run(
        config_path=REPOSITORY_ROOT / "training/configs/rtx4080_super_smoke.yaml",
        artifact_root=root,
        repository_root=REPOSITORY_ROOT,
        max_updates=1,
        interrupt_first_update=False,
    )

    assert result is sentinel
    payload = json.loads((root / "run-manifest.json").read_text(encoding="utf-8"))
    assert payload["task4_calibration"]["scenario_schedule_id"] == (
        CurriculumSchedule().scenario_schedule_id
    )


def test_sigterm_during_update_saves_only_after_complete_update_boundary() -> None:
    """Would fail if the signal handler checkpointed inside a PPO update."""
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    events: list[str] = []
    flag = SignalStopFlag()
    budget = TrainingBudget()
    loop = TrainingBoundaryLoop(
        budget=budget,
        stop_flag=flag,
        checkpoint_interval_seconds=1800,
        candidate_checkpoint_interval_seconds=3600,
        curriculum_phase="joint",
    )

    def update(batch: torch.Tensor) -> None:
        events.append("update-start")
        optimizer.zero_grad(set_to_none=True)
        model(batch).sum().backward()
        optimizer.step()
        os.kill(os.getpid(), signal.SIGTERM)
        events.append("update-finished")

    def save(kind: str, state) -> None:
        events.append(f"save-{kind}-step-{state.global_step}")

    with flag.installed():
        result = loop.run(
            collect_rollout=lambda: torch.ones((1, 2)),
            update_rollout=update,
            save_checkpoint=save,
            max_updates=5,
        )

    assert events == [
        "update-start",
        "update-finished",
        "save-latest-step-1",
    ]
    assert result.global_step == 1
    assert result.rollout_discarded is False
    assert result.stop_signal == signal.SIGTERM
    assert budget.consumed_gpu_seconds > 0.0


def test_signal_after_collection_discards_unfinished_rollout_without_update() -> None:
    """Would fail if a signal-boundary partial rollout entered an optimizer step."""
    events: list[str] = []
    flag = SignalStopFlag()
    loop = TrainingBoundaryLoop(
        budget=TrainingBudget(),
        stop_flag=flag,
        checkpoint_interval_seconds=1800,
        candidate_checkpoint_interval_seconds=3600,
        curriculum_phase="joint",
    )

    def collect() -> object:
        events.append("rollout-collected")
        os.kill(os.getpid(), signal.SIGINT)
        return object()

    with flag.installed():
        result = loop.run(
            collect_rollout=collect,
            update_rollout=lambda rollout: events.append("updated"),
            save_checkpoint=lambda kind, state: events.append(
                f"save-{kind}-step-{state.global_step}"
            ),
            max_updates=3,
        )

    assert events == ["rollout-collected", "save-latest-step-0"]
    assert result.global_step == 0
    assert result.rollout_discarded is True
    assert result.stop_signal == signal.SIGINT


def test_rollout_and_failed_update_both_settle_the_same_active_gpu_budget() -> None:
    """Would fail if policy inference or an exceptional PPO update escaped accounting."""
    timestamps = iter((10.0, 15.0))
    budget = TrainingBudget()
    loop = TrainingBoundaryLoop(
        budget=budget,
        stop_flag=SignalStopFlag(),
        checkpoint_interval_seconds=1800,
        candidate_checkpoint_interval_seconds=3600,
        curriculum_phase="joint",
        clock=lambda: next(timestamps),
    )

    with pytest.raises(RuntimeError, match="update failed"):
        loop.run(
            collect_rollout=lambda: object(),
            update_rollout=lambda rollout: (_ for _ in ()).throw(
                RuntimeError("update failed")
            ),
            save_checkpoint=lambda kind, state: None,
            max_updates=1,
        )

    assert budget.consumed_gpu_seconds == 5.0
    assert budget.interval_active is False


def test_resume_continues_latest_and_candidate_rhythms_from_active_gpu_markers() -> None:
    """Would fail if pause/resume restarted the 30/60-minute checkpoint clocks."""
    timestamps = iter((0.0, 1.0))
    budget = TrainingBudget(consumed_gpu_seconds=3599.0)
    saves: list[tuple[str, object]] = []
    loop = TrainingBoundaryLoop(
        budget=budget,
        stop_flag=SignalStopFlag(),
        checkpoint_interval_seconds=1800,
        candidate_checkpoint_interval_seconds=3600,
        curriculum_phase="joint",
        initial_latest_checkpoint_gpu_seconds=1800.0,
        initial_candidate_checkpoint_gpu_seconds=0.0,
        clock=lambda: next(timestamps),
    )

    state = loop.run(
        collect_rollout=lambda: object(),
        update_rollout=lambda rollout: None,
        save_checkpoint=lambda kind, saved_state: saves.append((kind, saved_state)),
        max_updates=1,
    )

    assert [kind for kind, _ in saves][:2] == ["latest", "candidate"]
    assert state.latest_checkpoint_gpu_seconds == 3600.0
    assert state.candidate_checkpoint_gpu_seconds == 3600.0


def test_training_does_not_start_callbacks_below_bounded_unit_reserve() -> None:
    """Would fail if the final partial budget launched another rollout/update."""
    budget = TrainingBudget(consumed_gpu_seconds=86400.0 - 600.0 + 1.0)
    events: list[str] = []
    loop = TrainingBoundaryLoop(
        budget=budget,
        stop_flag=SignalStopFlag(),
        checkpoint_interval_seconds=1800,
        candidate_checkpoint_interval_seconds=3600,
        curriculum_phase="joint",
        initial_global_step=41,
        initial_latest_checkpoint_gpu_seconds=84000.0,
        initial_candidate_checkpoint_gpu_seconds=82800.0,
        clock=lambda: 10.0,
    )

    state = loop.run(
        collect_rollout=lambda: events.append("collect"),
        update_rollout=lambda rollout: events.append("update"),
        save_checkpoint=lambda kind, saved_state: events.append(
            f"save-{kind}-step-{saved_state.global_step}"
        ),
        max_updates=1,
    )

    assert events == ["save-latest-step-41"]
    assert state.global_step == 41
    assert state.rollout_discarded is False
    assert state.latest_checkpoint_gpu_seconds == 86400.0
    assert budget.consumed_gpu_seconds == 86400.0
    assert budget.interval_active is False


def test_curriculum_phase_returns_before_next_update_could_cross_boundary() -> None:
    """Would fail if a warmup could consume time reserved for its next phase."""
    budget = TrainingBudget(consumed_gpu_seconds=6599.0)
    events: list[str] = []
    loop = TrainingBoundaryLoop(
        budget=budget,
        stop_flag=SignalStopFlag(),
        checkpoint_interval_seconds=1800,
        candidate_checkpoint_interval_seconds=3600,
        curriculum_phase="warmup_wheeled",
        phase_end_gpu_seconds=7199.0,
        clock=lambda: 10.0,
    )

    state = loop.run(
        collect_rollout=lambda: events.append("collect"),
        update_rollout=lambda rollout: events.append("update"),
        save_checkpoint=lambda kind, saved: events.append(
            f"save-{kind}-step-{saved.global_step}"
        ),
        max_updates=10,
    )

    assert events == ["save-latest-step-0"]
    assert state.global_step == 0
    assert budget.consumed_gpu_seconds == 6599.0


def test_formal_curriculum_invocation_advances_all_phases_without_restart(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Would fail if an unbounded formal invocation stalled at a phase edge."""
    root = tmp_path / "calibrated"
    _write_calibrated_manifest(root)
    calibrated = _load_calibrated_run_state(root)
    phases: list[tuple[str, dict[str, int]]] = []

    def run_phase(**kwargs):
        phase = kwargs["curriculum_phase"]
        allocation = kwargs["allocation"]
        budget = kwargs["budget"]
        phase_end = kwargs["phase_end_gpu_seconds"]
        phases.append((phase, allocation))
        if phase == "joint":
            budget.consume(budget.remaining_gpu_seconds)
        else:
            budget.consume(
                max(0.0, phase_end - budget.consumed_gpu_seconds - 600.0)
            )
        return SimpleNamespace(
            global_step=len(phases),
            consumed_gpu_seconds=budget.consumed_gpu_seconds,
            platform_allocation=allocation,
            signal_observed_at_update_boundary=False,
        )

    monkeypatch.setattr(cli_module, "_run_updates", run_phase)
    monkeypatch.setattr(
        cli_module,
        "load_checkpoint",
        lambda path: SimpleNamespace(
            global_step=len(phases), curriculum_phase=phases[-1][0]
        ),
    )

    result = _run_curriculum_training(
        calibrated=calibrated,
        artifact_root=root,
        repository_root=REPOSITORY_ROOT,
        source_commit="a" * 40,
        initial_global_step=0,
        max_updates=None,
        interrupt_first_update=False,
        restore_checkpoint=None,
    )

    assert phases == [
        ("warmup_wheeled", {"WHEELED": 24}),
        ("warmup_legged", {"LEGGED": 24}),
        ("warmup_hopper", {"HOPPER": 24}),
        ("joint", {"WHEELED": 8, "LEGGED": 8, "HOPPER": 8}),
    ]
    assert result.global_step == 4
    assert calibrated.budget.exhausted is True


def test_run_manifest_persists_exhausted_budget_terminal_state(
    tmp_path: pathlib.Path,
) -> None:
    manifest = tmp_path / "run-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "lunar-training-run/v1",
                "runtime_calibration": {"selected_workers": 18},
            }
        ),
        encoding="utf-8",
    )

    _update_run_manifest(
        manifest,
        source_commit="a" * 40,
        config_hash="b" * 64,
        global_step=41,
        consumed_gpu_seconds=86400.0,
        platform_allocation={"WHEELED": 6, "LEGGED": 6, "HOPPER": 6},
    )

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["budget_state"] == "exhausted"
    assert payload["consumed_gpu_seconds"] == 86400.0


def test_training_overrun_saves_terminal_latest_after_complete_update() -> None:
    timestamps = iter((0.0, 600.001))
    events: list[str] = []
    budget = TrainingBudget()
    loop = TrainingBoundaryLoop(
        budget=budget,
        stop_flag=SignalStopFlag(),
        checkpoint_interval_seconds=1800,
        candidate_checkpoint_interval_seconds=3600,
        curriculum_phase="joint",
        initial_global_step=7,
        clock=lambda: next(timestamps),
    )

    state = loop.run(
        collect_rollout=lambda: events.append("collect") or object(),
        update_rollout=lambda rollout: events.append("update"),
        save_checkpoint=lambda kind, saved_state: events.append(
            f"save-{kind}-step-{saved_state.global_step}"
        ),
        max_updates=2,
    )

    assert events == ["collect", "update", "save-latest-step-8"]
    assert state.global_step == 8
    assert state.rollout_discarded is False
    assert state.latest_checkpoint_gpu_seconds == 86400.0
    assert budget.exhausted is True


def _training_fingerprint() -> dict[str, object]:
    return {
        "schema_version": "lunar-platform-fingerprint/v1",
        "profile": "train_amd64_rtx4080_super",
        "captured_at_utc": "2026-08-03T00:00:00Z",
        "os": {"name": "Ubuntu", "version_id": "22.04"},
        "architecture": "amd64",
        "ros": {"available": True, "distro": "humble"},
        "gpu": {
            "available": True,
            "model": "NVIDIA GeForce RTX 4080 SUPER",
        },
        "cuda": {"available": True, "release": "13.2"},
        "readiness": {"ready": True, "errors": []},
    }


def test_lock_training_stack_is_deterministic_and_portable(
    tmp_path: pathlib.Path,
) -> None:
    """Would fail if constraints embedded host paths or changed for one input."""
    fingerprint = tmp_path / "fingerprint.json"
    fingerprint.write_text(
        json.dumps(_training_fingerprint()), encoding="utf-8"
    )
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    script = REPOSITORY_ROOT / "training/tools/lock_training_stack.py"

    for output in (first, second):
        subprocess.run(
            [
                sys.executable,
                str(script),
                "--fingerprint",
                str(fingerprint),
                "--output",
                str(output),
            ],
            cwd=REPOSITORY_ROOT,
            check=True,
        )

    contents = first.read_text(encoding="utf-8")
    assert contents == second.read_text(encoding="utf-8")
    assert "numpy==" in contents
    assert "PyYAML==" in contents
    assert f"torch=={torch.__version__}" in contents
    assert "/home/" not in contents
    assert "/mnt/" not in contents
    assert str(fingerprint) not in contents


def test_lock_training_stack_rejects_unready_fingerprint(
    tmp_path: pathlib.Path,
) -> None:
    """Would fail if an unqualified host could generate approved constraints."""
    payload = _training_fingerprint()
    payload["readiness"] = {"ready": False, "errors": ["gpu mismatch"]}
    fingerprint = tmp_path / "unready.json"
    fingerprint.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "constraints.txt"

    completed = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "training/tools/lock_training_stack.py"),
            "--fingerprint",
            str(fingerprint),
            "--output",
            str(output),
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "ready" in completed.stderr.lower()
    assert not output.exists()

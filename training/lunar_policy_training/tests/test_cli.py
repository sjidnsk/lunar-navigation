from __future__ import annotations

import pathlib
import os
import signal
import json
import subprocess
import sys

import pytest
import torch


PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "model_contract"))

from lunar_policy_training.budget import TrainingBudget  # noqa: E402
from lunar_policy_training.cli import (  # noqa: E402
    ArtifactRootError,
    SignalStopFlag,
    TrainingBoundaryLoop,
    build_parser,
    validate_artifact_root,
)


def test_task_three_cli_registers_only_train_and_resume() -> None:
    """Would fail if Task 4 commands leaked into the Task 3 public surface."""
    parser = build_parser()

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
        ["resume", "--artifact-root", "/tmp/lunar-task3"]
    )

    assert train.command == "train"
    assert resume.command == "resume"


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

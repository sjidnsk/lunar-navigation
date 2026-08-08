from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
WATCHER_PATH = REPOSITORY_ROOT / "training/tools/watch_training.py"


def _load_watcher():
    if not WATCHER_PATH.is_file():
        return None
    spec = importlib.util.spec_from_file_location("watch_training", WATCHER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_inputs(root: Path) -> dict[str, Path]:
    manifest = root / "run-manifest.json"
    checkpoint = root / "latest.pt"
    pause = root / "pause.marker"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "lunar-training-run/v1",
                "global_step": 7,
                "consumed_gpu_seconds": 12.5,
                "total_gpu_budget_seconds": 86400,
                "metrics": {"policy_loss": 0.25, "success_rate": 0.5},
            }
        ),
        encoding="utf-8",
    )
    checkpoint.write_bytes(b"inert-checkpoint-metadata")
    os.utime(manifest, (1_000.0, 1_000.0))
    os.utime(checkpoint, (1_000.0, 1_000.0))
    return {
        "manifest": manifest,
        "checkpoint": checkpoint,
        "pause": pause,
    }


def _inspect(watcher, paths: dict[str, Path], **overrides):
    arguments = {
        "pid": 1234,
        "expected_cmdline_token": "lunar-policy-training",
        "manifest_path": paths["manifest"],
        "checkpoint_path": paths["checkpoint"],
        "pause_marker_path": paths["pause"],
        "repository_root": REPOSITORY_ROOT,
        "disk_path": paths["manifest"].parent,
        "stale_after_seconds": 300.0,
        "minimum_disk_free_bytes": 1024,
        "clock": lambda: 1_100.0,
        "process_probe": lambda pid: watcher.ProcessEvidence(
            running=True,
            owned=True,
            cmdline=("python", "-m", "lunar-policy-training"),
            reason="owned-command",
        ),
        "capability_probe": lambda path: {
            "formal_eligible": True,
            "reason": "project-formal-capability",
        },
        "gpu_probe": lambda: watcher.GpuEvidence(
            ok=True,
            name="NVIDIA GeForce RTX 4080 SUPER",
            memory_used_mib=1024,
            memory_total_mib=16376,
            reason="query-ok",
        ),
        "disk_probe": lambda path: 10_000,
    }
    arguments.update(overrides)
    return watcher.inspect_training(**arguments)


def test_healthy_inspection_is_deterministic_and_read_only(tmp_path: Path) -> None:
    """Would fail if inspection mutated inputs or emitted unstable evidence."""
    watcher = _load_watcher()
    assert watcher is not None, "watch_training.py is not implemented"
    paths = _write_inputs(tmp_path)
    before = {
        name: (path.read_bytes(), path.stat().st_mtime_ns)
        for name, path in paths.items()
        if path.exists()
    }

    first = _inspect(watcher, paths)
    second = _inspect(watcher, paths)

    assert first == second
    assert first["state"] == "healthy"
    assert first["pid"] == {
        "value": 1234,
        "running": True,
        "owned": True,
        "reason": "owned-command",
    }
    assert first["checkpoint"]["age_seconds"] == 100.0
    assert first["manifest"]["metrics"] == {
        "policy_loss": 0.25,
        "success_rate": 0.5,
    }
    assert first["gpu"]["name"] == "NVIDIA GeForce RTX 4080 SUPER"
    after = {
        name: (path.read_bytes(), path.stat().st_mtime_ns)
        for name, path in paths.items()
        if path.exists()
    }
    assert after == before


def test_pause_marker_precedes_capability_and_live_probes(tmp_path: Path) -> None:
    """Would fail if automation acted past an explicit operator pause."""
    watcher = _load_watcher()
    assert watcher is not None, "watch_training.py is not implemented"
    paths = _write_inputs(tmp_path)
    paths["pause"].write_text("paused\n", encoding="utf-8")

    result = _inspect(
        watcher,
        paths,
        capability_probe=watcher._capability_evidence,
        process_probe=lambda pid: (_ for _ in ()).throw(
            AssertionError("paused inspection must not probe the process")
        ),
        gpu_probe=lambda: (_ for _ in ()).throw(
            AssertionError("paused inspection must not query the GPU")
        ),
    )

    assert result["state"] == "paused"
    assert result["pause_marker"]["present"] is True


def test_invalid_project_capability_blocks_before_runtime_probes(tmp_path: Path) -> None:
    """Would fail if watcher called an unresolved project capability healthy."""
    watcher = _load_watcher()
    assert watcher is not None, "watch_training.py is not implemented"
    paths = _write_inputs(tmp_path)

    result = _inspect(
        watcher,
        paths,
        capability_probe=lambda _root: {
            "formal_eligible": False,
            "reason": "invalid",
        },
        process_probe=lambda pid: (_ for _ in ()).throw(
            AssertionError("capability gate must precede process probes")
        ),
        gpu_probe=lambda: (_ for _ in ()).throw(
            AssertionError("capability gate must precede GPU probes")
        ),
    )

    assert result["state"] == "blocked-capability"
    assert result["capability"] == {
        "formal_eligible": False,
        "reason": "invalid",
    }


def test_missing_project_capability_files_remain_blocked(
    tmp_path: Path,
) -> None:
    """Would fail if watcher trusted a repository path without canonical files."""
    watcher = _load_watcher()
    assert watcher is not None, "watch_training.py is not implemented"
    paths = _write_inputs(tmp_path)

    result = _inspect(
        watcher,
        paths,
        repository_root=tmp_path,
        capability_probe=watcher._capability_evidence,
        process_probe=lambda pid: (_ for _ in ()).throw(
            AssertionError("invalid capability must precede process probes")
        ),
        gpu_probe=lambda: (_ for _ in ()).throw(
            AssertionError("invalid capability must precede GPU probes")
        ),
    )

    assert result["state"] == "blocked-capability"
    assert result["capability"] == {
        "formal_eligible": False,
        "reason": "invalid",
    }


@pytest.mark.parametrize(
    ("process", "expected_reason"),
    [
        ((False, False, (), "pid-not-running"), "pid-not-running"),
        ((True, False, ("python", "other-job"), "cmdline-mismatch"), "cmdline-mismatch"),
    ],
)
def test_pid_mismatch_or_absence_is_stopped(
    tmp_path: Path,
    process: tuple[bool, bool, tuple[str, ...], str],
    expected_reason: str,
) -> None:
    """Would fail if an unrelated or vanished PID were treated as the trainer."""
    watcher = _load_watcher()
    assert watcher is not None, "watch_training.py is not implemented"
    paths = _write_inputs(tmp_path)

    result = _inspect(
        watcher,
        paths,
        process_probe=lambda pid: watcher.ProcessEvidence(*process),
    )

    assert result["state"] == "stopped"
    assert result["pid"]["reason"] == expected_reason


def test_old_checkpoint_is_stale(tmp_path: Path) -> None:
    """Would fail if checkpoint age beyond the finite threshold looked healthy."""
    watcher = _load_watcher()
    assert watcher is not None, "watch_training.py is not implemented"
    paths = _write_inputs(tmp_path)

    result = _inspect(watcher, paths, clock=lambda: 1_301.0)

    assert result["state"] == "stale"
    assert result["checkpoint"]["age_seconds"] == 301.0


@pytest.mark.parametrize(
    ("mutation", "overrides", "expected_reason"),
    [
        (
            lambda paths: paths["manifest"].write_text(
                '{"global_step": 1, "metrics": {"loss": NaN}}',
                encoding="utf-8",
            ),
            {},
            "manifest-invalid",
        ),
        (
            lambda paths: None,
            {"gpu_probe": lambda: None},
            "gpu-query-failed",
        ),
        (
            lambda paths: None,
            {"disk_probe": lambda path: 100},
            "disk-free-below-threshold",
        ),
    ],
)
def test_invalid_runtime_evidence_is_error(
    tmp_path: Path,
    mutation,
    overrides: dict[str, object],
    expected_reason: str,
) -> None:
    """Would fail if malformed metrics or failed resource probes were guessed safe."""
    watcher = _load_watcher()
    assert watcher is not None, "watch_training.py is not implemented"
    paths = _write_inputs(tmp_path)
    mutation(paths)
    if overrides.get("gpu_probe") is not None:
        overrides["gpu_probe"] = lambda: watcher.GpuEvidence(
            ok=False,
            name=None,
            memory_used_mib=None,
            memory_total_mib=None,
            reason="gpu-query-failed",
        )

    result = _inspect(watcher, paths, **overrides)

    assert result["state"] == "error"
    assert expected_reason in result["reasons"]


def test_symlinked_required_file_is_error(tmp_path: Path) -> None:
    """Would fail if watcher followed a mutable symlink for authoritative state."""
    watcher = _load_watcher()
    assert watcher is not None, "watch_training.py is not implemented"
    paths = _write_inputs(tmp_path)
    real_manifest = tmp_path / "real-manifest.json"
    paths["manifest"].rename(real_manifest)
    paths["manifest"].symlink_to(real_manifest)

    result = _inspect(watcher, paths)

    assert result["state"] == "error"
    assert "manifest-unsafe" in result["reasons"]


def test_once_cli_emits_exactly_one_sorted_json_line(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Would fail if --once slept, emitted prose, or performed another inspection."""
    watcher = _load_watcher()
    assert watcher is not None, "watch_training.py is not implemented"
    calls: list[str] = []
    monkeypatch.setattr(
        watcher,
        "inspect_training",
        lambda **kwargs: calls.append("inspect") or {"z": 1, "state": "healthy"},
    )

    assert watcher.main(
        [
            "--once",
            "--pid",
            "1234",
            "--expected-cmdline-token",
            "lunar-policy-training",
            "--manifest",
            "/tmp/run-manifest.json",
            "--checkpoint",
            "/tmp/latest.pt",
            "--pause-marker",
            "/tmp/pause.marker",
            "--disk-path",
            "/tmp",
        ]
    ) == 0

    assert calls == ["inspect"]
    assert capsys.readouterr().out == '{"state":"healthy","z":1}\n'


def test_default_loop_waits_exactly_600_seconds_between_inspections() -> None:
    """Would fail if the automatic supervision cadence drifted from ten minutes."""
    watcher = _load_watcher()
    assert watcher is not None, "watch_training.py is not implemented"
    intervals: list[float] = []

    class StopLoop(Exception):
        pass

    def stop_after_first_interval(seconds: float) -> None:
        intervals.append(seconds)
        raise StopLoop

    with pytest.raises(StopLoop):
        watcher.run_watcher(
            inspect=lambda: {"state": "healthy"},
            emit=lambda result: None,
            once=False,
            sleep=stop_after_first_interval,
        )

    assert intervals == [600]

#!/usr/bin/env python3
"""Read-only JSON watcher for an explicitly identified training process."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from typing import NamedTuple


PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "lunar_policy_training"
sys.path.insert(0, str(PACKAGE_ROOT))

from lunar_policy_training.capability_freeze import (  # noqa: E402
    CapabilityFreezeError,
    load_frozen_capability_bundle,
)


DEFAULT_INTERVAL_SECONDS = 600
DEFAULT_STALE_AFTER_SECONDS = 2_400.0
DEFAULT_MINIMUM_DISK_FREE_BYTES = 10 * 1024**3


class ProcessEvidence(NamedTuple):
    running: bool
    owned: bool
    cmdline: tuple[str, ...]
    reason: str


class GpuEvidence(NamedTuple):
    ok: bool
    name: str | None
    memory_used_mib: int | None
    memory_total_mib: int | None
    reason: str


def _safe_regular_file(path: Path, *, allow_missing: bool = False) -> str:
    if path.is_symlink():
        return "unsafe"
    if path.is_file():
        return "regular"
    if allow_missing and not path.exists():
        return "missing"
    return "missing"


def _load_json(path: Path) -> object:
    def reject_constant(value: str) -> object:
        raise ValueError(f"non-finite JSON constant: {value}")

    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_constant,
    )


def _capability_evidence(path: Path) -> dict[str, object]:
    state = _safe_regular_file(path, allow_missing=True)
    if state == "missing":
        return {"formal_eligible": False, "reason": "missing"}
    if state != "regular":
        return {"formal_eligible": False, "reason": "unsafe"}
    try:
        bundle = load_frozen_capability_bundle(path, run_kind="formal")
    except CapabilityFreezeError:
        return {"formal_eligible": False, "reason": "invalid"}
    return {
        "formal_eligible": bundle.formal_eligible,
        "reason": "formal-lock-present",
    }


def _manifest_evidence(path: Path, *, now: float) -> dict[str, object]:
    if _safe_regular_file(path) != "regular":
        raise ValueError("manifest-unsafe")
    try:
        document = _load_json(path)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("manifest-invalid") from error
    if not isinstance(document, Mapping):
        raise ValueError("manifest-invalid")
    step = document.get("global_step")
    consumed = document.get("consumed_gpu_seconds")
    total = document.get("total_gpu_budget_seconds")
    metrics = document.get("metrics", {})
    if type(step) is not int or step < 0:
        raise ValueError("manifest-invalid")
    if not _finite_nonnegative(consumed) or not _finite_nonnegative(total):
        raise ValueError("manifest-invalid")
    if float(consumed) > float(total) or not isinstance(metrics, Mapping):
        raise ValueError("manifest-invalid")
    clean_metrics: dict[str, float] = {}
    for name, value in metrics.items():
        if not isinstance(name, str) or not name or not _finite_number(value):
            raise ValueError("manifest-invalid")
        clean_metrics[name] = float(value)
    return {
        "age_seconds": max(0.0, now - path.stat().st_mtime),
        "global_step": step,
        "consumed_gpu_seconds": float(consumed),
        "total_gpu_budget_seconds": float(total),
        "metrics": dict(sorted(clean_metrics.items())),
    }


def _checkpoint_evidence(path: Path, *, now: float) -> dict[str, object]:
    if _safe_regular_file(path) != "regular":
        raise ValueError("checkpoint-unsafe")
    stat = path.stat()
    return {
        "age_seconds": max(0.0, now - stat.st_mtime),
        "mtime_ns": stat.st_mtime_ns,
        "size_bytes": stat.st_size,
    }


def _finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _finite_nonnegative(value: object) -> bool:
    return _finite_number(value) and float(value) >= 0.0


def probe_process(pid: int, expected_cmdline_token: str) -> ProcessEvidence:
    if type(pid) is not int or pid <= 0:
        return ProcessEvidence(False, False, (), "pid-invalid")
    cmdline_path = Path("/proc") / str(pid) / "cmdline"
    if cmdline_path.is_symlink() or not cmdline_path.is_file():
        return ProcessEvidence(False, False, (), "pid-not-running")
    try:
        raw = cmdline_path.read_bytes()
        cmdline = tuple(
            field.decode("utf-8", errors="replace")
            for field in raw.split(b"\0")
            if field
        )
    except OSError:
        return ProcessEvidence(False, False, (), "pid-not-readable")
    owned = bool(cmdline) and any(
        expected_cmdline_token in field for field in cmdline
    )
    return ProcessEvidence(
        True,
        owned,
        cmdline,
        "owned-command" if owned else "cmdline-mismatch",
    )


def probe_gpu() -> GpuEvidence:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        rows = [row.strip() for row in completed.stdout.splitlines() if row.strip()]
        if len(rows) != 1:
            raise ValueError("expected exactly one GPU")
        name, used, total = (field.strip() for field in rows[0].split(",", 2))
        used_mib, total_mib = int(used), int(total)
        if not name or used_mib < 0 or total_mib <= 0 or used_mib > total_mib:
            raise ValueError("invalid GPU values")
        return GpuEvidence(True, name, used_mib, total_mib, "query-ok")
    except (OSError, subprocess.SubprocessError, ValueError):
        return GpuEvidence(False, None, None, None, "gpu-query-failed")


def inspect_training(
    *,
    pid: int,
    expected_cmdline_token: str,
    manifest_path: str | Path,
    checkpoint_path: str | Path,
    pause_marker_path: str | Path,
    capability_lock_path: str | Path,
    disk_path: str | Path,
    stale_after_seconds: float = DEFAULT_STALE_AFTER_SECONDS,
    minimum_disk_free_bytes: int = DEFAULT_MINIMUM_DISK_FREE_BYTES,
    clock: Callable[[], float] = time.time,
    process_probe: Callable[[int], ProcessEvidence] | None = None,
    capability_probe: Callable[[Path], dict[str, object]] = _capability_evidence,
    gpu_probe: Callable[[], GpuEvidence] = probe_gpu,
    disk_probe: Callable[[Path], int] | None = None,
) -> dict[str, object]:
    """Inspect explicit runtime inputs without writing or signaling anything."""
    if not _finite_number(stale_after_seconds) or stale_after_seconds <= 0:
        raise ValueError("stale threshold must be finite and positive")
    if type(minimum_disk_free_bytes) is not int or minimum_disk_free_bytes < 0:
        raise ValueError("minimum disk free bytes must be non-negative")
    pause = Path(pause_marker_path)
    pause_state = _safe_regular_file(pause, allow_missing=True)
    if pause_state == "regular":
        return {
            "state": "paused",
            "pause_marker": {"present": True, "reason": "explicit-marker"},
            "reasons": ["explicit-pause"],
        }
    if pause_state == "unsafe":
        return {
            "state": "error",
            "pause_marker": {"present": False, "reason": "unsafe"},
            "reasons": ["pause-marker-unsafe"],
        }

    capability = capability_probe(Path(capability_lock_path))
    if not capability["formal_eligible"]:
        return {
            "state": "blocked-capability",
            "pause_marker": {"present": False, "reason": "missing"},
            "capability": capability,
            "reasons": [f"capability-{capability['reason']}"],
        }

    now = float(clock())
    if not math.isfinite(now):
        raise ValueError("watcher clock must be finite")
    result: dict[str, object] = {
        "state": "error",
        "pause_marker": {"present": False, "reason": "missing"},
        "capability": capability,
        "reasons": [],
    }
    reasons: list[str] = []
    try:
        result["manifest"] = _manifest_evidence(Path(manifest_path), now=now)
    except ValueError as error:
        reasons.append(str(error))
    try:
        result["checkpoint"] = _checkpoint_evidence(
            Path(checkpoint_path), now=now
        )
    except ValueError as error:
        reasons.append(str(error))
    if reasons:
        result["reasons"] = reasons
        return result

    probe = process_probe or (
        lambda candidate_pid: probe_process(candidate_pid, expected_cmdline_token)
    )
    try:
        process = probe(pid)
    except Exception:
        process = ProcessEvidence(False, False, (), "pid-probe-failed")
    result["pid"] = {
        "value": pid,
        "running": process.running,
        "owned": process.owned,
        "reason": process.reason,
    }
    if not process.running or not process.owned:
        result["state"] = "stopped"
        result["reasons"] = [process.reason]
        return result

    try:
        gpu = gpu_probe()
    except Exception:
        gpu = GpuEvidence(False, None, None, None, "gpu-query-failed")
    if not isinstance(gpu, GpuEvidence):
        gpu = GpuEvidence(False, None, None, None, "gpu-query-failed")
    result["gpu"] = {
        "ok": gpu.ok,
        "name": gpu.name,
        "memory_used_mib": gpu.memory_used_mib,
        "memory_total_mib": gpu.memory_total_mib,
        "reason": gpu.reason,
    }
    if not gpu.ok:
        reasons.append(gpu.reason)

    try:
        free_bytes = (
            disk_probe(Path(disk_path))
            if disk_probe is not None
            else shutil.disk_usage(Path(disk_path)).free
        )
    except (OSError, ValueError):
        free_bytes = -1
    result["disk"] = {
        "free_bytes": free_bytes,
        "minimum_free_bytes": minimum_disk_free_bytes,
    }
    if type(free_bytes) is not int or free_bytes < minimum_disk_free_bytes:
        reasons.append(
            "disk-probe-failed" if free_bytes < 0 else "disk-free-below-threshold"
        )
    if reasons:
        result["reasons"] = reasons
        return result

    checkpoint = result["checkpoint"]
    assert isinstance(checkpoint, Mapping)
    if float(checkpoint["age_seconds"]) > float(stale_after_seconds):
        result["state"] = "stale"
        result["reasons"] = ["checkpoint-stale"]
    else:
        result["state"] = "healthy"
        result["reasons"] = []
    return result


def run_watcher(
    *,
    inspect: Callable[[], dict[str, object]],
    emit: Callable[[dict[str, object]], None],
    once: bool,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    while True:
        emit(inspect())
        if once:
            return
        sleep(DEFAULT_INTERVAL_SECONDS)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="watch-training")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--pid", required=True, type=int)
    parser.add_argument("--expected-cmdline-token", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--pause-marker", required=True)
    parser.add_argument("--capability-lock", required=True)
    parser.add_argument("--disk-path", required=True)
    parser.add_argument(
        "--stale-after-seconds",
        type=float,
        default=DEFAULT_STALE_AFTER_SECONDS,
    )
    parser.add_argument(
        "--minimum-disk-free-bytes",
        type=int,
        default=DEFAULT_MINIMUM_DISK_FREE_BYTES,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    inspect = lambda: inspect_training(
        pid=arguments.pid,
        expected_cmdline_token=arguments.expected_cmdline_token,
        manifest_path=arguments.manifest,
        checkpoint_path=arguments.checkpoint,
        pause_marker_path=arguments.pause_marker,
        capability_lock_path=arguments.capability_lock,
        disk_path=arguments.disk_path,
        stale_after_seconds=arguments.stale_after_seconds,
        minimum_disk_free_bytes=arguments.minimum_disk_free_bytes,
    )
    run_watcher(
        inspect=inspect,
        emit=lambda result: print(
            json.dumps(result, sort_keys=True, separators=(",", ":")),
            flush=True,
        ),
        once=arguments.once,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

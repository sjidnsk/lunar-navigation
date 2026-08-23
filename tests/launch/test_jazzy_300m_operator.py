"""Contract tests for the 300 m Jazzy exploration operator entry point."""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SHELL_ENTRY = ROOT / "scripts" / "run_jazzy_300m_exploration_sim.sh"
OPERATOR = ROOT / "scripts" / "jazzy_300m_operator.py"


def _module():
    spec = importlib.util.spec_from_file_location("jazzy_300m_operator", OPERATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_successful_results(output_dir: Path, *, csv_coverage: str = "0.625"):
    output_dir.mkdir()
    summary = {
        "success": True,
        "terminal_state": "COMPLETED",
        "reason_code": "COMPLETED_NO_REACHABLE_FRONTIER",
        "coverage_ratio": 0.625,
        "status_coverage_ratio": 0.625,
        "distance_m": 12.5,
        "completed_goal_count": 3,
        "wall_elapsed_s": 40.0,
        "sim_elapsed_s": 800.0,
        "planner": {
            "global": {"call_count": 4, "total_elapsed_ms": 12.0},
            "local": {"call_count": 4, "total_elapsed_ms": 24.0},
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    with (output_dir / "coverage.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["coverage_ratio", "state"])
        writer.writeheader()
        writer.writerow({"coverage_ratio": csv_coverage, "state": "COMPLETED"})
    (output_dir / "trajectory.csv").write_text(
        "x_m,y_m,distance_m\n0,0,0\n1,0,1\n", encoding="utf-8"
    )
    return summary


def test_shell_entry_is_strict_and_sources_explicit_overridable_overlay() -> None:
    source = SHELL_ENTRY.read_text(encoding="utf-8")
    assert "set -euo pipefail" in source
    assert "/opt/ros/jazzy/setup.bash" in source
    assert "LUNAR_JAZZY_OVERLAY" in source
    assert "exploration_jazzy_build/closed_loop/install/setup.bash" in source
    assert "set +u\nsource \"$JAZZY_SETUP\"\nsource \"$LUNAR_JAZZY_OVERLAY\"\nset -u" in source
    assert 'exec python3 "$SCRIPT_DIR/jazzy_300m_operator.py" "$@"' in source


def test_operator_contract_uses_isolated_domain_external_runs_and_exact_group() -> None:
    source = OPERATOR.read_text(encoding="utf-8")
    assert "ROS_LOCALHOST_ONLY" in source
    assert "ROS2CLI_NO_DAEMON" in source
    assert '"node", "list", "--no-daemon"' in source
    assert "/home/kai/CodexDownloads/lunar_navigation/exploration_jazzy_runs" in source
    assert "start_new_session=True" in source
    assert "start_rviz:=true" in source
    assert "start_rviz:=false" in source
    assert "SIGINT" in source and "SIGTERM" in source
    for forbidden in ("rm -rf", "rmdir -p", "pkill", "killall", "SIGKILL"):
        assert forbidden not in source


def test_validate_results_accepts_only_exact_completed_artifacts(tmp_path: Path) -> None:
    operator = _module()
    output_dir = tmp_path / "results"
    expected = _write_successful_results(output_dir)

    validated = operator.validate_results(output_dir)

    assert validated == expected


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("success", False),
        ("terminal_state", "TIMEOUT"),
        ("reason_code", "WALL_TIMEOUT"),
        ("status_coverage_ratio", 0.5),
        ("distance_m", 0.0),
        ("completed_goal_count", 0),
    ],
)
def test_validate_results_rejects_interrupt_guard_and_incomplete_evidence(
    tmp_path: Path, field: str, value: object
) -> None:
    operator = _module()
    output_dir = tmp_path / "results"
    summary = _write_successful_results(output_dir)
    summary[field] = value
    (output_dir / "summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )

    with pytest.raises(operator.AcceptanceError):
        operator.validate_results(output_dir)


@pytest.mark.parametrize("planner", ["global", "local"])
def test_validate_results_requires_each_planner_and_exact_csv_coverage(
    tmp_path: Path, planner: str
) -> None:
    operator = _module()
    output_dir = tmp_path / "results"
    summary = _write_successful_results(output_dir, csv_coverage="0.5")
    with pytest.raises(operator.AcceptanceError, match="coverage"):
        operator.validate_results(output_dir)

    with (output_dir / "coverage.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["coverage_ratio", "state"])
        writer.writeheader()
        writer.writerow({"coverage_ratio": "0.625", "state": "COMPLETED"})
    summary["planner"][planner]["call_count"] = 0
    (output_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(operator.AcceptanceError, match=planner):
        operator.validate_results(output_dir)


def test_exact_group_teardown_escalates_for_surviving_child() -> None:
    operator = _module()
    program = "\n".join(
        [
            "import os, signal, sys, time",
            "child = os.fork()",
            "if child == 0:",
            "    signal.signal(signal.SIGINT, signal.SIG_IGN)",
            "    while True: time.sleep(0.05)",
            "print(child, flush=True)",
            "signal.signal(signal.SIGINT, lambda *_: sys.exit(0))",
            "while True: time.sleep(0.05)",
        ]
    )
    process = subprocess.Popen(
        [sys.executable, "-c", program],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        start_new_session=True,
    )
    group = operator.fallback_new_session_group(process)
    try:
        group = operator.capture_exact_process_group(process)
        assert process.stdout is not None
        child_pid = int(process.stdout.readline().strip())
        result = operator.terminate_exact_process_group(
            process, group, interrupt_timeout=0.2, terminate_timeout=2.0
        )
        assert result.sigterm_sent
        deadline = time.monotonic() + 2.0
        while Path(f"/proc/{child_pid}").exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not Path(f"/proc/{child_pid}").exists()
    finally:
        operator.terminate_exact_process_group(
            process, group, interrupt_timeout=0.2, terminate_timeout=2.0
        )
        if process.stdout is not None:
            process.stdout.close()

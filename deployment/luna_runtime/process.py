from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Protocol

from .commands import RuntimeRefusal
from .config import RuntimeConfig, render_planner_params
from .state import RuntimePaths, atomic_write_json


class ManagedChild(Protocol):
    pid: int
    fingerprint: str

    def terminate(self) -> None: ...


class ProcessRunner(Protocol):
    def popen(self, command: tuple[str, ...]) -> ManagedChild: ...

    def run(self, command: tuple[str, ...]) -> None: ...

    def lifecycle_state(self, node_name: str) -> str: ...

    def process_matches(self, pid: int, fingerprint: str) -> bool: ...

    def terminate(self, pid: int, fingerprint: str) -> None: ...


def _state_path(paths: RuntimePaths) -> Path:
    return paths.data / "runtime-state.json"


def _read_state(paths: RuntimePaths) -> dict[str, object] | None:
    try:
        payload = json.loads(_state_path(paths).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def start_runtime(config: RuntimeConfig, paths: RuntimePaths, runner: ProcessRunner) -> dict[str, object]:
    if config.policy["mode"] != "fallback":
        raise RuntimeRefusal("POLICY_RUNTIME_UNBOUND")
    params = paths.data / "generated" / "planner-params.yaml"
    render_planner_params(config, params)
    if config.input_adapters["mode"] == "task3_adapted":
        command = (
            "ros2", "launch", "luna_t3_map_adapter", "task3_live_integration.launch.py",
            f"params_file:={params}",
            f"task3_config_file:={config.input_adapters['task3_config_file']}",
        )
    else:
        command = (
            "ros2", "launch", "lunar_planner_ros", "lunar_planner.launch.py",
            f"params_file:={params}",
        )
    child = runner.popen(command)
    try:
        if runner.lifecycle_state("/lunar_planner") != "unconfigured":
            raise RuntimeRefusal("LIFECYCLE_NOT_UNCONFIGURED")
        runner.run(("ros2", "lifecycle", "set", "/lunar_planner", "configure"))
        if runner.lifecycle_state("/lunar_planner") != "inactive":
            raise RuntimeRefusal("LIFECYCLE_CONFIGURE_FAILED")
        runner.run(("ros2", "lifecycle", "set", "/lunar_planner", "activate"))
        if runner.lifecycle_state("/lunar_planner") != "active":
            raise RuntimeRefusal("LIFECYCLE_ACTIVATE_FAILED")
    except Exception:
        child.terminate()
        raise
    state = {
        "pid": child.pid,
        "fingerprint": child.fingerprint,
        "running": True,
        "lifecycle_state": "active",
        "model_binding": "fallback",
    }
    atomic_write_json(_state_path(paths), state)
    return state


def read_runtime_status(paths: RuntimePaths, runner: ProcessRunner) -> dict[str, object]:
    state = _read_state(paths)
    if not state:
        return {"running": False, "reason": "NOT_STARTED"}
    pid = state.get("pid")
    fingerprint = state.get("fingerprint")
    if not isinstance(pid, int) or not isinstance(fingerprint, str) or not runner.process_matches(pid, fingerprint):
        return {"running": False, "reason": "STALE_OR_STOPPED_PID"}
    return dict(state)


def stop_runtime(paths: RuntimePaths, runner: ProcessRunner) -> dict[str, object]:
    state = _read_state(paths)
    if not state:
        return {"running": False, "reason": "NOT_STARTED"}
    pid = state.get("pid")
    fingerprint = state.get("fingerprint")
    if isinstance(pid, int) and isinstance(fingerprint, str) and runner.process_matches(pid, fingerprint):
        # The runner owns the only child it may terminate; unrelated PIDs are never signalled.
        runner.terminate(pid, fingerprint)
    try:
        _state_path(paths).unlink()
    except FileNotFoundError:
        pass
    return {"running": False, "reason": "STOPPED"}


def tail_log(paths: RuntimePaths) -> Path:
    return paths.data / "log"

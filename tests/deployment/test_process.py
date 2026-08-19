from __future__ import annotations

import json
from pathlib import Path

import pytest

from deployment.luna_runtime.commands import RuntimeRefusal
from deployment.luna_runtime.config import load_runtime_config
from deployment.luna_runtime.process import read_runtime_status, start_runtime, stop_runtime
from deployment.luna_runtime.state import resolve_runtime_paths


class FakeChild:
    pid = 4242
    fingerprint = "luna-child-4242"
    terminated = False

    def terminate(self) -> None:
        self.terminated = True


class FakeRunner:
    def __init__(self, lifecycle: list[str]) -> None:
        self.lifecycle = iter(lifecycle)
        self.commands: list[tuple[str, ...]] = []
        self.child = FakeChild()

    def popen(self, command: tuple[str, ...]) -> FakeChild:
        self.commands.append(command)
        return self.child

    def run(self, command: tuple[str, ...]) -> None:
        self.commands.append(command)

    def lifecycle_state(self, _node_name: str) -> str:
        return next(self.lifecycle)

    def process_matches(self, pid: int, fingerprint: str) -> bool:
        return pid == self.child.pid and fingerprint == self.child.fingerprint

    def terminate(self, pid: int, fingerprint: str) -> None:
        if self.process_matches(pid, fingerprint):
            self.child.terminate()


def _config(tmp_path: Path, *, policy_mode: str = "fallback"):
    root = Path(__file__).resolve().parents[2]
    text = (root / "deployment/config/runtime.default.yaml").read_text(encoding="utf-8")
    text = text.replace("mode: fallback", f"mode: {policy_mode}")
    if policy_mode != "fallback":
        text = text.replace("model_id: null", "model_id: demo-v4")
    path = tmp_path / "runtime.yaml"
    path.write_text(text, encoding="utf-8")
    return load_runtime_config(path)


def test_start_configures_then_activates_lifecycle_and_records_verified_pid(tmp_path: Path) -> None:
    runner = FakeRunner(["unconfigured", "inactive", "active"])
    paths = resolve_runtime_paths("dev", home=tmp_path)

    state = start_runtime(_config(tmp_path), paths, runner)

    assert runner.commands[-2:] == [
        ("ros2", "lifecycle", "set", "/lunar_planner", "configure"),
        ("ros2", "lifecycle", "set", "/lunar_planner", "activate"),
    ]
    assert state["lifecycle_state"] == "active"
    assert read_runtime_status(paths, runner)["running"] is True


def test_nonfallback_policy_is_refused_before_launch(tmp_path: Path) -> None:
    runner = FakeRunner([])
    with pytest.raises(RuntimeRefusal, match="POLICY_RUNTIME_UNBOUND"):
        start_runtime(_config(tmp_path, policy_mode="onnx"), resolve_runtime_paths("dev", home=tmp_path), runner)
    assert runner.commands == []


def test_stop_is_idempotent_and_rejects_stale_pid(tmp_path: Path) -> None:
    paths = resolve_runtime_paths("dev", home=tmp_path)
    paths.data.mkdir(parents=True)
    (paths.data / "runtime-state.json").write_text(
        json.dumps({"pid": 9999, "fingerprint": "different", "lifecycle_state": "active"}),
        encoding="utf-8",
    )
    runner = FakeRunner([])

    assert stop_runtime(paths, runner)["running"] is False
    assert stop_runtime(paths, runner)["running"] is False
    assert runner.child.terminated is False

"""Behavioral contracts for the split Orin Humble operator scripts."""

from __future__ import annotations

import ast
import os
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "orin"
LAUNCH = ROOT / "launch" / "exploration_navigation.launch.py"


def _launch_arguments() -> set[str]:
    tree = ast.parse(LAUNCH.read_text(encoding="utf-8"))
    result: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "DeclareLaunchArgument":
            continue
        if node.args and isinstance(node.args[0], ast.Constant):
            result.add(str(node.args[0].value))
    return result


def _fake_environment(
    tmp_path: Path, *, emulate_ament_trace_setup: bool = False
) -> tuple[dict[str, str], Path]:
    bin_dir = tmp_path / "bin"
    log_dir = tmp_path / "logs"
    bin_dir.mkdir()
    log_dir.mkdir()
    for name in ("colcon", "ros2", "rsync", "ssh"):
        executable = bin_dir / name
        executable.write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "log=\"${LUNAR_ORIN_TEST_LOG_DIR}/$(basename \"$0\").log\"\n"
            "printf 'PWD=%s\\n' \"$PWD\" > \"$log\"\n"
            "printf '%s\\n' \"$@\" >> \"$log\"\n",
            encoding="utf-8",
        )
        executable.chmod(0o755)

    bash_env = tmp_path / "bash_env"
    if emulate_ament_trace_setup:
        bash_env.write_text(
            "source() {\n"
            "  if [[ \"$-\" == *u* && -z \"${AMENT_TRACE_SETUP_FILES+x}\" ]]; then\n"
            "    printf '%s\\n' 'AMENT_TRACE_SETUP_FILES: unbound variable' >&2\n"
            "    return 1\n"
            "  fi\n"
            "}\n",
            encoding="utf-8",
        )
    else:
        bash_env.write_text("source() { :; }\n", encoding="utf-8")
    environment = os.environ.copy()
    environment.pop("AMENT_TRACE_SETUP_FILES", None)
    environment["BASH_ENV"] = str(bash_env)
    environment["LUNAR_ORIN_TEST_LOG_DIR"] = str(log_dir)
    environment["PATH"] = f"{bin_dir}{os.pathsep}{environment['PATH']}"
    return environment, log_dir


def _run(
    tmp_path: Path,
    script_name: str,
    *arguments: str,
    emulate_ament_trace_setup: bool = False,
) -> tuple[list[str], Path]:
    script = SCRIPTS / script_name
    assert script.is_file(), f"missing operator script: {script.relative_to(ROOT)}"
    environment, log_dir = _fake_environment(
        tmp_path, emulate_ament_trace_setup=emulate_ament_trace_setup
    )
    result = subprocess.run(
        [str(script), *arguments],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.splitlines(), log_dir


def _command(log_dir: Path, name: str) -> list[str]:
    return (log_dir / f"{name}.log").read_text(encoding="utf-8").splitlines()


def test_formal_launch_exposes_independent_navigation_and_exploration_selection() -> None:
    """Catch removal of either switch that the separate operator scripts require."""
    assert {"start_navigation", "start_exploration"} <= _launch_arguments()


def test_navigation_start_script_launches_only_navigation(tmp_path: Path) -> None:
    """Catch a navigation command that also starts the exploration node."""
    _run(tmp_path, "start_navigation.sh", "legged", "false")
    command = _command(tmp_path / "logs", "ros2")

    assert command[1:4] == [
        "launch",
        "lunar_pure_exploration_ros",
        "exploration_navigation.launch.py",
    ]
    assert "stack_mode:=incremental_v2" in command
    assert "platform_type:=legged" in command
    assert "use_sim_time:=false" in command
    assert "start_navigation:=true" in command
    assert "start_exploration:=false" in command


def test_rviz_goal_bridge_start_script_launches_the_incremental_bridge(
    tmp_path: Path,
) -> None:
    """Catch accidentally launching the legacy PlanMotion RViz bridge."""
    _run(tmp_path, "start_rviz_goal_bridge.sh")
    command = _command(tmp_path / "logs", "ros2")

    assert command[1:4] == [
        "launch",
        "lunar_incremental_navigation_ros",
        "incremental_rviz_goal_bridge.launch.py",
    ]


def test_exploration_start_script_launches_only_exploration(tmp_path: Path) -> None:
    """Catch an exploration command that also recreates the navigation node."""
    _run(tmp_path, "start_exploration.sh", "wheel", "true")
    command = _command(tmp_path / "logs", "ros2")

    assert command[1:4] == [
        "launch",
        "lunar_pure_exploration_ros",
        "exploration_navigation.launch.py",
    ]
    assert "stack_mode:=incremental_v2" in command
    assert "platform_type:=wheel" in command
    assert "use_sim_time:=true" in command
    assert "start_navigation:=false" in command
    assert "start_exploration:=true" in command


def test_build_script_uses_the_fixed_workspace_outputs(tmp_path: Path) -> None:
    """Catch temporary or alternate build/install/log locations in the Orin build path."""
    _run(tmp_path, "build.sh")
    command = _command(tmp_path / "logs", "colcon")

    assert command[0] == f"PWD={ROOT / 'ros2_ws'}"
    assert command[1:4] == ["build", "--merge-install", "--packages-up-to"]
    assert command[4:6] == [
        "lunar_incremental_navigation_ros",
        "lunar_pure_exploration_ros",
    ]
    assert "--build-base" not in command
    assert "--install-base" not in command
    assert "--log-base" not in command
    assert "mktemp" not in command


@pytest.mark.parametrize(
    ("script_name", "arguments"),
    [
        ("build.sh", ()),
        ("start_navigation.sh", ("wheel", "false")),
        ("start_rviz_goal_bridge.sh", ()),
        ("start_exploration.sh", ("wheel", "false")),
        ("publish_navigation_goal.sh", ("0.0", "0.0")),
        ("publish_exploration_task.sh", ("task", "0", "0", "1", "1")),
    ],
)
def test_ros_setup_is_loaded_before_enabling_nounset(
    tmp_path: Path, script_name: str, arguments: tuple[str, ...]
) -> None:
    """Catch any Humble setup source that runs while nounset is active."""
    _run(
        tmp_path,
        script_name,
        *arguments,
        emulate_ament_trace_setup=True,
    )


def test_fixed_workspace_outputs_are_gitignored() -> None:
    """Catch a fixed Orin build directory becoming an accidental source artifact."""
    for path in ("ros2_ws/build", "ros2_ws/install", "ros2_ws/log"):
        result = subprocess.run(
            ["git", "check-ignore", "-q", "--no-index", f"{path}/artifact"],
            cwd=ROOT,
            check=False,
        )
        assert result.returncode == 0, path


def test_navigation_goal_script_sends_navigate_to_pose_goal(tmp_path: Path) -> None:
    """Catch publishing to the legacy action or dropping an explicit terminal yaw."""
    _run(tmp_path, "publish_navigation_goal.sh", "12.5", "-4.0", "1.57")
    command = _command(tmp_path / "logs", "ros2")
    payload = "\n".join(command)

    assert command[1:5] == ["action", "send_goal", "--feedback", "/Car/T4/navigation/navigate_to_pose"]
    assert "lunar_planning_msgs/action/NavigateToPose" in command
    assert "target_x_m: 12.5" in payload
    assert "target_y_m: -4.0" in payload
    assert "has_target_yaw: true" in payload
    assert "target_yaw_rad: 1.57" in payload


def test_exploration_task_script_publishes_a_map_rectangle(tmp_path: Path) -> None:
    """Catch a task command that uses the wrong topic, type, frame, or boundary shape."""
    _run(tmp_path, "publish_exploration_task.sh", "area-7", "0", "1", "2", "3")
    command = _command(tmp_path / "logs", "ros2")
    payload = "\n".join(command)

    assert command[1:5] == ["topic", "pub", "--once", "/Car/T4/exploration/task"]
    assert "lunar_pure_exploration_msgs/msg/PureExplorationTask" in command
    assert "frame_id: map" in payload
    assert "task_id: area-7" in payload
    assert "command: 1" in payload
    assert "{x: 0, y: 1, z: 0.0}" in payload
    assert "{x: 2, y: 3, z: 0.0}" in payload


def test_deploy_script_syncs_without_requesting_remote_deletion(tmp_path: Path) -> None:
    """Catch a deployment command that deletes remote state or creates a temporary bundle."""
    _run(tmp_path, "deploy.sh", "orin@example", "/opt/lunar")
    ssh = _command(tmp_path / "logs", "ssh")
    rsync = _command(tmp_path / "logs", "rsync")

    assert ssh[1:] == ["orin@example", "mkdir", "-p", "/opt/lunar"]
    assert "--delete" not in rsync
    assert "orin@example:/opt/lunar/" == rsync[-1]

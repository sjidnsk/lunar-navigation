"""Static contract for the isolated 300 m Jazzy exploration composition."""

from __future__ import annotations

import ast
import fcntl
import importlib.util
import json
import math
import os
import select
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
LAUNCH = ROOT / "launch" / "jazzy_300m_exploration_sim.launch.py"
RVIZ = ROOT / "rviz" / "jazzy_300m_exploration_sim.rviz"
CMAKE = ROOT / "ros2_ws" / "src" / "lunar_pure_exploration_sim" / "CMakeLists.txt"
LIVE_OPT_IN = "LUNAR_RUN_LIVE_JAZZY_SMOKE"
LIVE_ROOT = Path(
    "/home/kai/CodexDownloads/lunar_navigation/exploration_jazzy_runs/live-smoke"
)

EXPECTED_EXECUTABLES = {
    "simulation_node",
    "lunar_pure_planner_node",
    "pure_exploration_node",
    "lunar_pure_wheeled_controller_node.py",
    "run_coordinator",
    "run_recorder",
    "simulation_hud_node",
    "rviz2",
}

EXACT_TOPICS = {
    "/Car/T3/mapping/global_overview",
    "/Car/T3/mapping/grid_map",
    "/Car/T3/localization/odometry",
    "/Car/T4/plan_motion",
    "/Car/T4/planning/diagnostics",
    "/Car/T4/exploration/task",
    "/Car/T4/exploration/status",
    "/Car/T4/exploration/current_goal",
    "/Car/T4/exploration/frontiers",
    "/Car/T4/exploration/diagnostics",
    "/Car/T4/execution/motion_reference",
    "/Car/T4/execution/cancel",
    "/Car/T4/simulation/sensor_fov",
    "/Car/T4/simulation/vehicle_markers",
    "/Car/T4/simulation/local_map_markers",
    "/Car/T4/simulation/actual_path",
    "/Car/T4/simulation/planned_path",
    "/Car/T4/simulation/hud",
    "/Car/T4/simulation/sim_elapsed",
    "/Car/T5/Car_Cmd_Vel",
    "/tf",
}

CAPACITIES = {
    "maximum_position_probes": 8192,
    "maximum_candidate_views": 4096,
    "maximum_collision_work_units": 4194304,
    "maximum_visibility_work_units": 4096,
    "maximum_task_raster_cells": 1048576,
    "maximum_guidance_grid_cells": 1048576,
    "maximum_guidance_work_units": 8388608,
    "maximum_approach_candidates": 4096,
    "maximum_path_preview_poses": 4096,
    "maximum_executable_path_points": 4096,
    "maximum_failure_entries": 2048,
    "maximum_failure_patch_cells_per_entry": 512,
    "maximum_failure_total_patch_cells": 262144,
}


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _keyword_strings(source: str, function_name: str, keyword: str) -> list[str]:
    tree = ast.parse(source)
    values: list[str] = []
    for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
        if not isinstance(call.func, ast.Name) or call.func.id != function_name:
            continue
        for item in call.keywords:
            if item.arg == keyword and isinstance(item.value, ast.Constant):
                if isinstance(item.value.value, str):
                    values.append(item.value.value)
    return values


def _launch_module():
    spec = importlib.util.spec_from_file_location("jazzy_300m_launch", LAUNCH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextmanager
def _temporary_directory():
    root = Path(tempfile.mkdtemp(prefix="lunar-launch-contract-"))
    try:
        yield root
    finally:
        git_marker = root / ".git"
        if git_marker.is_file():
            git_marker.unlink()
        elif git_marker.is_dir():
            git_marker.rmdir()
        root.rmdir()


def _context(output_dir: str):
    from launch import LaunchContext

    context = LaunchContext()
    context.launch_configurations.update(
        output_dir=output_dir,
        seed="20260824",
        speed_multiplier="20.0",
        start_rviz="false",
    )
    return context


def _action_parameters(action, context) -> dict[str, object]:
    from launch_ros.utilities import evaluate_parameters

    result: dict[str, object] = {}
    for item in evaluate_parameters(context, action._Node__parameters):
        if isinstance(item, dict):
            result.update(item)
    return result


def test_static_launch_declares_required_arguments_and_absolute_output_guard() -> None:
    source = _source(LAUNCH)
    assert 'DeclareLaunchArgument("seed", default_value="20260824")' in source
    assert 'DeclareLaunchArgument("speed_multiplier", default_value="20.0")' in source
    assert 'DeclareLaunchArgument("start_rviz", default_value="true")' in source
    assert 'DeclareLaunchArgument("output_dir")' in source
    assert "Path(output_dir).is_absolute()" in source
    assert "output_dir must be an absolute external path" in source


def test_static_launch_starts_exact_graph_and_conditionally_starts_rviz() -> None:
    source = _source(LAUNCH)
    assert set(_keyword_strings(source, "Node", "executable")) == EXPECTED_EXECUTABLES
    assert "IfCondition(start_rviz)" in source
    assert "LifecycleNode" not in source


def test_static_launch_freezes_interfaces_controller_override_and_capacities() -> None:
    source = _source(LAUNCH)
    for topic in EXACT_TOPICS:
        assert topic in source
    assert '"reference_topic": "/Car/T4/execution/motion_reference"' in source
    assert '"controller_command_topic": "/Car/T5/Car_Cmd_Vel"' in source
    assert '"platform_type": "wheel"' in source
    assert '"speed_multiplier": ParameterValue(speed_multiplier, value_type=float)' in source
    for name, value in CAPACITIES.items():
        assert f'"{name}": {value}' in source
    assert "/lunar_demo/" not in source


def test_launch_composition_exposes_exact_actions_and_runtime_parameters(monkeypatch) -> None:
    module = _launch_module()
    monkeypatch.setattr(
        module,
        "get_package_share_directory",
        lambda package: f"/opt/ros/jazzy/share/{package}",
    )
    with _temporary_directory() as root:
        context = _context(str(root / "run"))
        actions = module._compose(context, rviz_config="/tmp/demo.rviz")

    assert [action.node_executable for action in actions] == [
        "simulation_node",
        "lunar_pure_planner_node",
        "pure_exploration_node",
        "lunar_pure_wheeled_controller_node.py",
        "run_coordinator",
        "run_recorder",
        "simulation_hud_node",
        "rviz2",
    ]
    by_executable = {action.node_executable: action for action in actions}
    simulation = _action_parameters(by_executable["simulation_node"], context)
    controller = _action_parameters(
        by_executable["lunar_pure_wheeled_controller_node.py"], context
    )
    coordinator = _action_parameters(by_executable["run_coordinator"], context)
    explorer = _action_parameters(by_executable["pure_exploration_node"], context)
    planner = _action_parameters(by_executable["lunar_pure_planner_node"], context)
    assert simulation["command_topic"] == "/Car/T5/Car_Cmd_Vel"
    assert controller == {
        "reference_topic": "/Car/T4/execution/motion_reference",
        "odometry_topic": "/Car/T3/localization/odometry",
        "command_topic": "/Car/T5/Car_Cmd_Vel",
        "execution_cancel_topic": "/Car/T4/execution/cancel",
    }
    assert coordinator["controller_command_topic"] == "/Car/T5/Car_Cmd_Vel"
    assert planner["wheel_planner_mode"] == "grid_traversability_v1"
    assert planner["rolling_surface_enabled"] is False
    for name, value in CAPACITIES.items():
        assert explorer[name] == value
    assert actions[-1].condition is not None


def test_output_validation_accepts_external_and_rejects_relative_and_repo() -> None:
    module = _launch_module()
    with _temporary_directory() as root:
        accepted = module._validate_output_dir(str(root / "run"), [ROOT])
        assert accepted == (root / "run").resolve(strict=False)
    with pytest.raises(RuntimeError, match="absolute external path"):
        module._validate_output_dir("relative", [ROOT])
    with pytest.raises(RuntimeError, match="outside repositories"):
        module._validate_output_dir(str(ROOT / "run-output"), [ROOT])


@pytest.mark.parametrize("git_marker_kind", ["file", "directory"])
def test_output_validation_rejects_any_git_ancestor(git_marker_kind: str) -> None:
    module = _launch_module()
    with _temporary_directory() as root:
        marker = root / ".git"
        if git_marker_kind == "file":
            marker.write_text("gitdir: elsewhere\n", encoding="utf-8")
        else:
            marker.mkdir()
        with pytest.raises(RuntimeError, match="Git repository or worktree"):
            module._validate_output_dir(str(root / "nested" / "run"), [ROOT])


def test_static_rviz_uses_only_standard_displays_for_approved_topics() -> None:
    source = _source(RVIZ)
    assert "Fixed Frame: map" in source
    assert "grid_map" not in source.lower()
    assert "rviz_grid_map_plugins" not in source
    for display_class, topic in (
        ("rviz_default_plugins/Map", "/Car/T3/mapping/global_overview"),
        ("rviz_default_plugins/Path", "/Car/T4/simulation/actual_path"),
        ("rviz_default_plugins/Path", "/Car/T4/simulation/planned_path"),
        ("rviz_default_plugins/MarkerArray", "/Car/T4/exploration/frontiers"),
        ("rviz_default_plugins/Pose", "/Car/T4/exploration/current_goal"),
        ("rviz_default_plugins/MarkerArray", "/Car/T4/simulation/vehicle_markers"),
        ("rviz_default_plugins/Marker", "/Car/T4/simulation/sensor_fov"),
        ("rviz_default_plugins/MarkerArray", "/Car/T4/simulation/local_map_markers"),
        ("rviz_default_plugins/MarkerArray", "/Car/T4/simulation/hud"),
    ):
        assert display_class in source
        assert f"Topic: {topic}" in source
    assert "/lunar_demo/" not in source


def test_static_sim_package_installs_launch_and_rviz_assets() -> None:
    source = _source(CMAKE)
    assert '"${LUNAR_PURE_SIM_LAUNCH_SOURCE_DIR}/jazzy_300m_exploration_sim.launch.py"' in source
    assert 'DESTINATION share/${PROJECT_NAME}/launch' in source
    assert '"${LUNAR_PURE_SIM_RVIZ_SOURCE_DIR}/jazzy_300m_exploration_sim.rviz"' in source
    assert 'DESTINATION share/${PROJECT_NAME}/rviz' in source


def _cli(command: list[str], env: dict[str, str], timeout: float = 5.0):
    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )


@contextmanager
def _empty_live_domain():
    """Reserve a derived candidate and reject every nonempty ROS graph."""
    first = 20 + ((os.getpid() * 37 + time.time_ns()) % 180)
    for offset in range(180):
        candidate = 20 + ((first - 20 + offset) % 180)
        lock_path = Path(f"/tmp/lunar_jazzy_live_domain_{candidate}.lock")
        lock = lock_path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock.close()
            continue
        env = os.environ.copy()
        env.update(
            {
                "ROS_DOMAIN_ID": str(candidate),
                "ROS_LOCALHOST_ONLY": "1",
                "ROS2CLI_NO_DAEMON": "1",
            }
        )
        try:
            probe = _cli(
                ["ros2", "node", "list", "--no-daemon"], env, timeout=3.0
            )
        except subprocess.TimeoutExpired:
            lock.close()
            continue
        if probe.returncode != 0 or probe.stdout.strip():
            lock.close()
            continue
        try:
            yield env
        finally:
            lock.close()
        return
    pytest.fail("no empty isolated ROS_DOMAIN_ID candidate was available")


def _wait_for_empty_live_graph(env: dict[str, str], timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    last = ""
    while time.monotonic() < deadline:
        try:
            probe = _cli(
                ["ros2", "node", "list", "--no-daemon"], env, timeout=2.0
            )
        except subprocess.TimeoutExpired:
            last = "node-list probe timed out"
            continue
        last = f"rc={probe.returncode}\n{probe.stdout}{probe.stderr}"
        if probe.returncode == 0 and not probe.stdout.strip():
            return
        time.sleep(0.1)
    pytest.fail(f"isolated live-smoke graph did not drain:\n{last}")


@dataclass(frozen=True)
class _ProcessIdentity:
    pid: int
    start_time_ticks: int
    pgid: int
    session_id: int


@dataclass
class _ExactProcessGroup:
    pgid: int
    session_id: int
    leader: _ProcessIdentity | None
    members: set[_ProcessIdentity] = field(default_factory=set)


@dataclass(frozen=True)
class _GroupTeardownResult:
    sigterm_sent: bool
    recorded_members: tuple[_ProcessIdentity, ...]


def _read_process_identity(pid: int) -> _ProcessIdentity | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None
    command_end = stat.rfind(")")
    if command_end < 0:
        return None
    fields = stat[command_end + 2 :].split()
    if len(fields) < 20:
        return None
    try:
        return _ProcessIdentity(
            pid=pid,
            start_time_ticks=int(fields[19]),
            pgid=int(fields[2]),
            session_id=int(fields[3]),
        )
    except ValueError:
        return None


def _capture_exact_process_group(process: subprocess.Popen[bytes]) -> _ExactProcessGroup:
    leader = _read_process_identity(process.pid)
    if leader is None:
        raise RuntimeError("launch leader exited before process identity capture")
    pgid = os.getpgid(process.pid)
    if leader.pgid != pgid or leader.session_id != pgid:
        raise RuntimeError("launch did not create the expected new session/process group")
    group = _ExactProcessGroup(
        pgid=pgid,
        session_id=leader.session_id,
        leader=leader,
        members={leader},
    )
    _observe_exact_group_members(group)
    return group


def _fallback_new_session_group(
    process: subprocess.Popen[bytes],
) -> _ExactProcessGroup:
    """Purely describe the exact new session promised by start_new_session."""
    return _ExactProcessGroup(
        pgid=process.pid,
        session_id=process.pid,
        leader=None,
    )


def _observe_exact_group_members(group: _ExactProcessGroup) -> None:
    for entry in Path("/proc").iterdir():
        if not entry.name.isdecimal():
            continue
        identity = _read_process_identity(int(entry.name))
        if (
            identity is not None
            and identity.pgid == group.pgid
            and identity.session_id == group.session_id
        ):
            group.members.add(identity)


def _identity_is_alive(identity: _ProcessIdentity) -> bool:
    current = _read_process_identity(identity.pid)
    return current == identity


def _remaining_exact_group_members(
    group: _ExactProcessGroup,
) -> set[_ProcessIdentity]:
    _observe_exact_group_members(group)
    return {identity for identity in group.members if _identity_is_alive(identity)}


def _wait_for_exact_group_exit(
    process: subprocess.Popen[bytes],
    group: _ExactProcessGroup,
    timeout: float,
) -> set[_ProcessIdentity]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        process.poll()
        remaining = _remaining_exact_group_members(group)
        if not remaining:
            return set()
        time.sleep(0.02)
    process.poll()
    return _remaining_exact_group_members(group)


def _terminate_exact_process_group(
    process: subprocess.Popen[bytes],
    group: _ExactProcessGroup,
    *,
    interrupt_timeout: float = 10.0,
    terminate_timeout: float = 5.0,
) -> _GroupTeardownResult:
    _observe_exact_group_members(group)
    initial_members = _remaining_exact_group_members(group)
    if not initial_members:
        process.poll()
        return _GroupTeardownResult(
            sigterm_sent=False,
            recorded_members=tuple(sorted(group.members, key=lambda item: item.pid)),
        )
    try:
        os.killpg(group.pgid, signal.SIGINT)
    except ProcessLookupError:
        pass
    remaining = _wait_for_exact_group_exit(process, group, interrupt_timeout)
    sigterm_sent = bool(remaining)
    if remaining:
        try:
            os.killpg(group.pgid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        remaining = _wait_for_exact_group_exit(process, group, terminate_timeout)
    if remaining:
        details = ", ".join(
            f"pid={item.pid}/start={item.start_time_ticks}"
            for item in sorted(remaining, key=lambda item: item.pid)
        )
        pytest.fail(f"exact launch process group survived SIGINT and SIGTERM: {details}")
    if group.leader is not None and _identity_is_alive(group.leader):
        pytest.fail("launch leader identity remained after process-group teardown")
    return _GroupTeardownResult(
        sigterm_sent=sigterm_sent,
        recorded_members=tuple(sorted(group.members, key=lambda item: item.pid)),
    )


def _stop_exact_launch_group(
    process: subprocess.Popen[bytes],
    env: dict[str, str],
    group: _ExactProcessGroup,
) -> _GroupTeardownResult:
    result = _terminate_exact_process_group(process, group)
    _wait_for_empty_live_graph(env)
    return result


def _diagnostic_fields(message) -> dict[str, str]:
    if len(message.status) != 1:
        return {}
    return {item.key: item.value for item in message.status[0].values}


def test_exact_group_teardown_escalates_after_leader_exits() -> None:
    """A launch child that ignores SIGINT must not outlive its leader."""
    program = "\n".join(
        [
            "import os, signal, sys, time",
            "child = os.fork()",
            "if child == 0:",
            "    signal.signal(signal.SIGINT, signal.SIG_IGN)",
            "    while True: time.sleep(0.1)",
            "print(child, flush=True)",
            "signal.signal(signal.SIGINT, lambda *_: sys.exit(0))",
            "while True: time.sleep(0.1)",
        ]
    )
    process = subprocess.Popen(
        [sys.executable, "-c", program],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
        text=True,
    )
    group = _fallback_new_session_group(process)
    child_pid: int | None = None
    try:
        group = _capture_exact_process_group(process)
        assert process.stdout is not None
        ready, _, _ = select.select([process.stdout], [], [], 2.0)
        if not ready:
            raise RuntimeError("synthetic child PID was not reported")
        child_pid = int(process.stdout.readline().strip())
        result = _terminate_exact_process_group(
            process, group, interrupt_timeout=0.2, terminate_timeout=2.0
        )
        assert result.sigterm_sent
        deadline = time.monotonic() + 2.0
        while Path(f"/proc/{child_pid}").exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not Path(f"/proc/{child_pid}").exists()
    finally:
        _terminate_exact_process_group(
            process, group, interrupt_timeout=0.2, terminate_timeout=2.0
        )
        if process.stdout is not None:
            process.stdout.close()
        if process.stderr is not None:
            process.stderr.close()


def test_fallback_group_cleans_child_when_leader_exits_before_capture() -> None:
    """The Popen-to-identity-capture exception window remains bounded."""
    program = "\n".join(
        [
            "import os, signal, sys, time",
            "child = os.fork()",
            "if child == 0:",
            "    signal.signal(signal.SIGINT, signal.SIG_IGN)",
            "    while True: time.sleep(0.1)",
            "sys.exit(0)",
        ]
    )
    process = subprocess.Popen(
        [sys.executable, "-c", program],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    group = _fallback_new_session_group(process)
    try:
        process.wait(timeout=2.0)
        with pytest.raises(RuntimeError, match="identity capture"):
            _capture_exact_process_group(process)
        result = _terminate_exact_process_group(
            process, group, interrupt_timeout=0.2, terminate_timeout=2.0
        )
        assert result.sigterm_sent
        assert not _remaining_exact_group_members(group)
    finally:
        _terminate_exact_process_group(
            process, group, interrupt_timeout=0.2, terminate_timeout=2.0
        )


@pytest.mark.skipif(
    os.environ.get(LIVE_OPT_IN) != "1",
    reason=f"set {LIVE_OPT_IN}=1 to run the isolated Jazzy live smoke",
)
def test_live_closed_loop_reaches_real_motion_and_planner_timing() -> None:
    """Bounded proof of the real installed Jazzy closed loop, never completion."""
    with _empty_live_domain() as env:
        run_dir = LIVE_ROOT / f"run-{uuid.uuid4().hex}"
        output_dir = run_dir / "results"
        ros_log_dir = run_dir / "ros-logs"
        run_dir.mkdir(parents=True)
        output_dir.mkdir()
        ros_log_dir.mkdir()
        env = env.copy()
        env["ROS_LOG_DIR"] = str(ros_log_dir)

        # ROS_DOMAIN_ID must be frozen before importing/initializing rclpy.
        os.environ.update(
            {
                "ROS_DOMAIN_ID": env["ROS_DOMAIN_ID"],
                "ROS_LOCALHOST_ONLY": "1",
                "ROS2CLI_NO_DAEMON": "1",
            }
        )
        import rclpy
        from diagnostic_msgs.msg import DiagnosticArray
        from geometry_msgs.msg import PoseStamped, Twist
        from grid_map_msgs.msg import GridMap
        from lunar_planning_msgs.action import PlanMotion
        from lunar_planning_msgs.msg import MotionReference
        from lunar_pure_exploration_msgs.msg import (
            PureExplorationStatus,
            PureExplorationTask,
        )
        from nav_msgs.msg import OccupancyGrid, Odometry
        from rclpy.action import ActionClient
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from tf2_msgs.msg import TFMessage
        from visualization_msgs.msg import Marker, MarkerArray

        evidence: dict[str, object] = {
            "domain_id": env["ROS_DOMAIN_ID"],
            "task_ids": [],
            "task_message_count": 0,
            "states": [],
            "status_trace": [],
            "streams": {"global": 0, "local": 0, "odometry": 0, "tf": 0},
            "action_ready": False,
            "reference_points": 0,
            "nonzero_command": False,
            "displacement_m": 0.0,
            "coverage_baseline": None,
            "coverage_maximum": None,
            "planner_diagnostics": [],
            "current_goals": [],
            "candidate_markers": [],
            "odom_arrival_monotonic_s": [],
        }
        reliable = QoSProfile(depth=20, reliability=ReliabilityPolicy.RELIABLE)
        latched = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        rclpy.init(args=None)
        node = rclpy.create_node(f"jazzy_live_probe_{uuid.uuid4().hex[:8]}")
        action = ActionClient(node, PlanMotion, "/Car/T4/plan_motion")
        first_xy: tuple[float, float] | None = None
        tf_edges: set[tuple[str, str]] = set()

        def on_global(_message: OccupancyGrid) -> None:
            evidence["streams"]["global"] += 1

        def on_local(_message: GridMap) -> None:
            evidence["streams"]["local"] += 1

        def on_odometry(message: Odometry) -> None:
            nonlocal first_xy
            evidence["streams"]["odometry"] += 1
            evidence["odom_arrival_monotonic_s"].append(time.monotonic())
            xy = (message.pose.pose.position.x, message.pose.pose.position.y)
            if first_xy is None:
                first_xy = xy
            evidence["displacement_m"] = max(
                float(evidence["displacement_m"]),
                math.hypot(xy[0] - first_xy[0], xy[1] - first_xy[1]),
            )

        def on_tf(message: TFMessage) -> None:
            evidence["streams"]["tf"] += 1
            for transform in message.transforms:
                tf_edges.add(
                    (transform.header.frame_id, transform.child_frame_id)
                )

        def on_task(message: PureExplorationTask) -> None:
            evidence["task_message_count"] += 1
            if message.task_id not in evidence["task_ids"]:
                evidence["task_ids"].append(message.task_id)

        def on_status(message: PureExplorationStatus) -> None:
            state = int(message.state)
            if state not in evidence["states"]:
                evidence["states"].append(state)
            evidence["status_trace"].append(
                {
                    "state": state,
                    "reason": message.reason_code,
                    "coverage": message.coverage_ratio,
                    "task_id": message.task_id,
                }
            )
            if message.task_id == "jazzy-300m-20260824":
                if evidence["coverage_baseline"] is None:
                    evidence["coverage_baseline"] = message.coverage_ratio
                current = evidence["coverage_maximum"]
                evidence["coverage_maximum"] = (
                    message.coverage_ratio
                    if current is None
                    else max(float(current), message.coverage_ratio)
                )

        def on_reference(message: MotionReference) -> None:
            points = max(len(message.path_preview.poses), len(message.trajectory.points))
            evidence["reference_points"] = max(
                int(evidence["reference_points"]), points
            )

        def on_goal(message: PoseStamped) -> None:
            orientation = message.pose.orientation
            yaw = math.atan2(
                2.0 * (orientation.w * orientation.z + orientation.x * orientation.y),
                1.0 - 2.0 * (orientation.y**2 + orientation.z**2),
            )
            goal = [message.pose.position.x, message.pose.position.y, yaw]
            if goal not in evidence["current_goals"]:
                evidence["current_goals"].append(goal)

        def on_frontiers(message: MarkerArray) -> None:
            candidates = []
            for marker in message.markers:
                if marker.ns != "candidates" or marker.action != Marker.ADD:
                    continue
                orientation = marker.pose.orientation
                yaw = math.atan2(
                    2.0 * (orientation.w * orientation.z),
                    1.0 - 2.0 * orientation.z**2,
                )
                candidates.append(
                    [marker.pose.position.x, marker.pose.position.y, yaw]
                )
            if candidates:
                evidence["candidate_markers"] = candidates

        def on_command(message: Twist) -> None:
            if abs(message.linear.x) > 1.0e-6 or abs(message.angular.z) > 1.0e-6:
                evidence["nonzero_command"] = True

        def on_diagnostics(message: DiagnosticArray) -> None:
            fields = _diagnostic_fields(message)
            if fields:
                evidence["planner_diagnostics"].append(fields)

        subscriptions = [
            node.create_subscription(OccupancyGrid, "/Car/T3/mapping/global_overview", on_global, latched),
            node.create_subscription(GridMap, "/Car/T3/mapping/grid_map", on_local, reliable),
            node.create_subscription(Odometry, "/Car/T3/localization/odometry", on_odometry, reliable),
            node.create_subscription(TFMessage, "/tf", on_tf, reliable),
            node.create_subscription(PureExplorationTask, "/Car/T4/exploration/task", on_task, latched),
            node.create_subscription(PureExplorationStatus, "/Car/T4/exploration/status", on_status, latched),
            node.create_subscription(PoseStamped, "/Car/T4/exploration/current_goal", on_goal, latched),
            node.create_subscription(MarkerArray, "/Car/T4/exploration/frontiers", on_frontiers, reliable),
            node.create_subscription(MotionReference, "/Car/T4/execution/motion_reference", on_reference, reliable),
            node.create_subscription(Twist, "/Car/T5/Car_Cmd_Vel", on_command, reliable),
            node.create_subscription(DiagnosticArray, "/Car/T4/planning/diagnostics", on_diagnostics, reliable),
        ]
        assert subscriptions

        launch_log_path = run_dir / "launch.log"
        probe_path = run_dir / "probe.json"
        launch_log = launch_log_path.open("wb")
        process = subprocess.Popen(
            [
                "ros2",
                "launch",
                "lunar_pure_exploration_sim",
                "jazzy_300m_exploration_sim.launch.py",
                "start_rviz:=false",
                "seed:=20260824",
                "speed_multiplier:=20.0",
                f"output_dir:={output_dir}",
            ],
            stdin=subprocess.DEVNULL,
            stdout=launch_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env,
        )
        launch_group = _fallback_new_session_group(process)
        failure: str | None = None
        try:
            launch_group = _capture_exact_process_group(process)
            assert launch_group.leader is not None
            evidence["launch_pgid"] = launch_group.pgid
            evidence["launch_leader"] = {
                "pid": launch_group.leader.pid,
                "start_time_ticks": launch_group.leader.start_time_ticks,
            }
            deadline = time.monotonic() + 120.0
            while time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=0.05)
                evidence["action_ready"] = action.server_is_ready()
                diagnostics = evidence["planner_diagnostics"]
                has_global_local_timing = any(
                    int(item.get("global_call_count", "0")) > 0
                    and int(item.get("local_call_count", "0")) > 0
                    and float(item.get("global_elapsed_ms", "0")) > 0.0
                    and float(item.get("local_elapsed_ms", "0")) > 0.0
                    for item in diagnostics
                )
                grid_v1_active = any(
                    item.get("grid_v1_active") == "true" for item in diagnostics
                )
                observed_transitions = {
                    (item["state"], item["reason"])
                    for item in evidence["status_trace"]
                }
                coverage_increased = (
                    evidence["coverage_baseline"] is not None
                    and evidence["coverage_maximum"] is not None
                    and float(evidence["coverage_maximum"])
                    > float(evidence["coverage_baseline"])
                )
                if (
                    all(int(value) > 0 for value in evidence["streams"].values())
                    and {("map", "odom"), ("odom", "base_link")} <= tf_edges
                    and evidence["action_ready"]
                    and evidence["task_ids"] == ["jazzy-300m-20260824"]
                    and evidence["task_message_count"] == 1
                    and {
                        (
                            PureExplorationStatus.SELECTING_FRONTIER,
                            "SELECTING_FRONTIER",
                        ),
                        (PureExplorationStatus.PLANNING, "PLANNING"),
                        (PureExplorationStatus.EXECUTING, "EXECUTING"),
                    }
                    <= observed_transitions
                    and int(evidence["reference_points"]) > 0
                    and evidence["nonzero_command"]
                    and float(evidence["displacement_m"]) >= 0.2
                    and coverage_increased
                    and has_global_local_timing
                    and grid_v1_active
                ):
                    break
                if process.poll() is not None:
                    failure = f"launch exited early with rc={process.returncode}"
                    break
                terminal_states = {
                    PureExplorationStatus.COMPLETED,
                    PureExplorationStatus.ERROR,
                }
                if set(evidence["states"]) & terminal_states:
                    failure = "exploration reached a terminal state before real motion"
                    break
            else:
                failure = "120 second smoke deadline expired"

            arrivals = evidence["odom_arrival_monotonic_s"]
            gaps = [right - left for left, right in zip(arrivals, arrivals[1:])]
            evidence["odom_sample_count"] = len(arrivals)
            evidence["odom_mean_gap_s"] = (
                sum(gaps) / len(gaps) if gaps else None
            )
            evidence["tf_edges"] = sorted([list(edge) for edge in tf_edges])
            probe_path.write_text(
                json.dumps(evidence, indent=2, sort_keys=True), encoding="utf-8"
            )
            if failure is not None:
                pytest.fail(f"{failure}; evidence={probe_path}; log={launch_log_path}")
            assert len(arrivals) >= 20, evidence
            assert evidence["odom_mean_gap_s"] is not None
            assert float(evidence["odom_mean_gap_s"]) <= 0.1, evidence
        finally:
            try:
                action.destroy()
                node.destroy_node()
                rclpy.shutdown()
            finally:
                try:
                    teardown = _stop_exact_launch_group(process, env, launch_group)
                    evidence["teardown"] = {
                        "sigterm_sent": teardown.sigterm_sent,
                        "recorded_members": [
                            {
                                "pid": member.pid,
                                "start_time_ticks": member.start_time_ticks,
                            }
                            for member in teardown.recorded_members
                        ],
                        "exact_members_gone": True,
                        "ros_graph_empty": True,
                    }
                    probe_path.write_text(
                        json.dumps(evidence, indent=2, sort_keys=True),
                        encoding="utf-8",
                    )
                finally:
                    launch_log.close()

"""Installed launch and fail-closed configuration checks for the pure planner."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import fcntl

import pytest

APPROVED_SUBSCRIPTIONS = {
    "/Car/T3/localization/odometry",
    "/Car/T3/mapping/global_overview",
    "/Car/T3/mapping/grid_map",
    "/tf",
}
FRAMEWORK_SUBSCRIPTIONS = {"/parameter_events"}
ACTION_SERVICES = {
    "/Car/T4/plan_motion/_action/cancel_goal",
    "/Car/T4/plan_motion/_action/get_result",
    "/Car/T4/plan_motion/_action/send_goal",
}
# Keep allocator locks until pytest exits so a recently stopped DDS participant
# cannot collide with the next parameter case while discovery state drains.
_PROCESS_DOMAIN_LOCKS: list[object] = []


def test_domain_reservation_respects_existing_ci_domain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ROS_DOMAIN_ID", "137")
    with _reserved_domain() as domain_id:
        assert domain_id == "137"


def test_domain_reservation_holds_exclusive_allocator_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ROS_DOMAIN_ID", raising=False)
    with _reserved_domain() as first:
        with _reserved_domain() as second:
            assert second != first


def test_domain_reservation_does_not_reuse_a_recent_process_domain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ROS_DOMAIN_ID", raising=False)
    with _reserved_domain() as first:
        pass
    with _reserved_domain() as second:
        assert second != first


@pytest.mark.parametrize(
    "failed_probe",
    ["node", "service", "topic"],
    ids=["node-list", "hidden-action-services", "diagnostic-topic"],
)
def test_graph_drain_treats_cli_probe_failure_as_cleanup_failure(
    monkeypatch: pytest.MonkeyPatch, failed_probe: str
) -> None:
    calls = 0

    def fake_run(
        command: list[str],
        timeout: float = 5.0,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        del timeout, env
        nonlocal calls
        calls += 1
        probe = command[1]
        failed = probe == failed_probe
        return subprocess.CompletedProcess(
            command,
            1 if failed else 0,
            stdout=(
                "Publisher count: 0\n"
                if probe == "topic" and not failed
                else ""
            ),
            stderr=("synthetic CLI failure" if failed else ""),
        )

    monkeypatch.setattr(__name__ + "._run", fake_run)
    monkeypatch.setattr(time, "sleep", lambda _: None)

    with pytest.raises(pytest.fail.Exception, match="did not drain"):
        _wait_for_empty_graph({}, timeout=0.01)
    assert calls > 0


def test_graph_drain_checks_diagnostic_topic_and_publisher_separately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    topic_list_calls = 0
    topic_info_calls = 0

    def fake_run(
        command: list[str],
        timeout: float = 5.0,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        del timeout, env
        nonlocal topic_list_calls, topic_info_calls
        stdout = ""
        if command[1:3] == ["topic", "list"]:
            topic_list_calls += 1
            if topic_list_calls == 1:
                stdout = "/Car/T4/planning/diagnostics\n"
        elif command[1:3] == ["topic", "info"]:
            topic_info_calls += 1
            stdout = "Publisher count: 0\n"
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(__name__ + "._run", fake_run)
    monkeypatch.setattr(time, "sleep", lambda _: None)

    _wait_for_empty_graph({}, timeout=0.1)
    assert topic_list_calls == 2
    assert topic_info_calls == 1


@contextmanager
def _reserved_domain() -> Iterator[str]:
    configured = os.environ.get("ROS_DOMAIN_ID")
    if configured is not None:
        yield configured
        return

    for candidate in range(20, 200):
        lock_file = open(  # noqa: SIM115 - lifetime is the context's lock.
            f"/tmp/lunar_pure_planner_ros_domain_{candidate}.lock",
            "a+",
            encoding="utf-8",
        )
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock_file.close()
            continue
        _PROCESS_DOMAIN_LOCKS.append(lock_file)
        yield str(candidate)
        return
    raise RuntimeError("no isolated ROS_DOMAIN_ID is available")


def _run(
    command: list[str],
    timeout: float = 5.0,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env or os.environ.copy(),
    )


def _wait_for_ready_graph(env: dict[str, str], timeout: float = 8.0) -> str:
    deadline = time.monotonic() + timeout
    last = ""
    while time.monotonic() < deadline:
        node = _run(
            ["ros2", "node", "info", "--no-daemon", "/pure_planner"],
            timeout=2.0,
            env=env,
        )
        services = _run(
            [
                "ros2",
                "service",
                "list",
                "--no-daemon",
                "--include-hidden-services",
            ],
            timeout=2.0,
            env=env,
        )
        diagnostics = _run(
            [
                "ros2",
                "topic",
                "info",
                "--no-daemon",
                "/Car/T4/planning/diagnostics",
            ],
            timeout=2.0,
            env=env,
        )
        service_names = set(services.stdout.splitlines())
        last = node.stdout + node.stderr + services.stdout + diagnostics.stdout
        if (
            node.returncode == 0
            and services.returncode == 0
            and diagnostics.returncode == 0
            and _business_subscription_topics(node.stdout)
            == APPROVED_SUBSCRIPTIONS
            and ACTION_SERVICES <= service_names
            and "/Car/T4/plan_motion: lunar_planning_msgs/action/PlanMotion"
            in node.stdout
            and "Publisher count: 1" in diagnostics.stdout
        ):
            return node.stdout
        time.sleep(0.1)
    pytest.fail(f"pure planner node did not become ready:\n{last}")


def _wait_for_empty_graph(env: dict[str, str], timeout: float = 8.0) -> None:
    deadline = time.monotonic() + timeout
    last = ""
    while time.monotonic() < deadline:
        nodes = _run(
            ["ros2", "node", "list", "--no-daemon"], timeout=2.0, env=env
        )
        services = _run(
            [
                "ros2",
                "service",
                "list",
                "--no-daemon",
                "--include-hidden-services",
            ],
            timeout=2.0,
            env=env,
        )
        topics = _run(
            [
                "ros2",
                "topic",
                "list",
                "--no-daemon",
            ],
            timeout=2.0,
            env=env,
        )
        node_names = set(nodes.stdout.splitlines())
        service_names = set(services.stdout.splitlines())
        topic_names = set(topics.stdout.splitlines())
        node_gone = "/pure_planner" not in node_names
        action_gone = ACTION_SERVICES.isdisjoint(service_names)
        diagnostic_topic_gone = (
            "/Car/T4/planning/diagnostics" not in topic_names
        )
        diagnostic_publisher_gone = diagnostic_topic_gone
        diagnostic_info: subprocess.CompletedProcess[str] | None = None
        diagnostic_info_ok = True
        if topics.returncode == 0 and not diagnostic_topic_gone:
            diagnostic_info = _run(
                [
                    "ros2",
                    "topic",
                    "info",
                    "--no-daemon",
                    "/Car/T4/planning/diagnostics",
                ],
                timeout=2.0,
                env=env,
            )
            diagnostic_info_ok = diagnostic_info.returncode == 0
            diagnostic_publisher_gone = (
                diagnostic_info_ok
                and "Publisher count: 0" in diagnostic_info.stdout
            )
        probes_ok = (
            nodes.returncode == 0
            and services.returncode == 0
            and topics.returncode == 0
            and diagnostic_info_ok
        )
        if (
            probes_ok
            and node_gone
            and action_gone
            and diagnostic_topic_gone
            and diagnostic_publisher_gone
        ):
            return
        diagnostic_info_output = (
            "diagnostic publisher info: not probed because topic is absent\n"
            if diagnostic_info is None
            else (
                f"diagnostic publisher info (rc={diagnostic_info.returncode}):\n"
                f"{diagnostic_info.stdout}{diagnostic_info.stderr}"
            )
        )
        last = (
            f"nodes (rc={nodes.returncode}):\n{nodes.stdout}{nodes.stderr}"
            f"services (rc={services.returncode}):\n"
            f"{services.stdout}{services.stderr}"
            f"topics (rc={topics.returncode}):\n{topics.stdout}{topics.stderr}"
            f"{diagnostic_info_output}"
        )
        time.sleep(0.1)
    pytest.fail(f"pure planner ROS graph did not drain:\n{last}")


def _node_subscription_topics(node_info: str) -> set[str]:
    topics: set[str] = set()
    in_subscribers = False
    for line in node_info.splitlines():
        if line == "  Subscribers:":
            in_subscribers = True
            continue
        if in_subscribers and line.startswith("  ") and not line.startswith("    "):
            break
        if in_subscribers and line.startswith("    "):
            topics.add(line.strip().split(": ", maxsplit=1)[0])
    return topics


def _business_subscription_topics(node_info: str) -> set[str]:
    return _node_subscription_topics(node_info) - FRAMEWORK_SUBSCRIPTIONS


def _stop(
    process: subprocess.Popen[str], env: dict[str, str], timeout: float = 8.0
) -> str:
    timed_out = False
    try:
        os.killpg(process.pid, signal.SIGINT)
    except ProcessLookupError:
        pass
    if process.poll() is None:
        try:
            process.wait(timeout=4.0)
        except subprocess.TimeoutExpired:
            timed_out = True
    # ros2 launch can exit before its child. The child retains the launch
    # process group, so always make one final bounded group cleanup attempt.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    if process.poll() is None:
        try:
            process.wait(timeout=4.0)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=2.0)
    output, _ = process.communicate(timeout=1.0)
    if process.stdout is not None and not process.stdout.closed:
        process.stdout.close()
    _wait_for_empty_graph(env, timeout=timeout)
    if timed_out:
        pytest.fail(f"pure planner launch did not exit within cleanup budget:\n{output}")
    return output


def test_fixed_domain_runs_all_platforms_and_mismatch_without_graph_leaks() -> None:
    with _reserved_domain() as domain_id:
        env = os.environ.copy()
        env.update({"ROS_DOMAIN_ID": domain_id, "ROS2CLI_NO_DAEMON": "1"})
        for platform in ("wheel", "legged", "hopper"):
            process = subprocess.Popen(
                [
                    "ros2",
                    "launch",
                    "lunar_pure_planner_ros",
                    "pure_planner.launch.py",
                    f"platform_type:={platform}",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
                env=env,
            )
            try:
                info = _wait_for_ready_graph(env)
                raw_subscriptions = _node_subscription_topics(info)
                assert raw_subscriptions == (
                    APPROVED_SUBSCRIPTIONS | FRAMEWORK_SUBSCRIPTIONS
                )
                business_subscriptions = _business_subscription_topics(info)
                assert len(business_subscriptions) == 4
                assert business_subscriptions == APPROVED_SUBSCRIPTIONS
                assert (
                    "/Car/T4/plan_motion: lunar_planning_msgs/action/PlanMotion"
                    in info
                )
                services = _run(
                    [
                        "ros2",
                        "service",
                        "list",
                        "--no-daemon",
                        "--include-hidden-services",
                    ],
                    env=env,
                )
                assert services.returncode == 0
                assert ACTION_SERVICES <= set(services.stdout.splitlines())
                diagnostics = _run(
                    [
                        "ros2",
                        "topic",
                        "info",
                        "--no-daemon",
                        "/Car/T4/planning/diagnostics",
                    ],
                    env=env,
                )
                assert diagnostics.returncode == 0
                assert "Publisher count: 1" in diagnostics.stdout
            finally:
                _stop(process, env)

        mismatch = _run(
            [
                "ros2",
                "run",
                "lunar_pure_planner_ros",
                "lunar_pure_planner_node",
                "--ros-args",
                "-p",
                "platform_type:=legged",
                "-p",
                "platform_config:=wheel.yaml",
            ],
            timeout=5.0,
            env=env,
        )
        assert mismatch.returncode != 0
        assert "PLANNER_ERROR" in mismatch.stdout + mismatch.stderr
        _wait_for_empty_graph(env)


def test_installed_production_closure_excludes_test_seams() -> None:
    prefix = _run(
        ["ros2", "pkg", "prefix", "lunar_pure_planner_ros"], timeout=3.0
    )
    assert prefix.returncode == 0, prefix.stdout + prefix.stderr
    install_prefix = Path(prefix.stdout.strip())
    include_root = install_prefix / "include" / "lunar_pure_planner_ros"
    installed_headers = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(include_root.rglob("*.hpp"))
    )
    library = install_prefix / "lib" / "liblunar_pure_planner_ros.so"
    symbols = _run(
        ["nm", "-D", "-C", "--defined-only", str(library)], timeout=5.0
    )
    assert symbols.returncode == 0, symbols.stdout + symbols.stderr
    object_strings = _run(["strings", "-a", str(library)], timeout=5.0)
    assert object_strings.returncode == 0, (
        object_strings.stdout + object_strings.stderr
    )
    for token in (
        "ServerExecutionHooks",
        "PurePlanMotionServerTestFactory",
        "pure_plan_motion_server_test_seam",
        "before_terminal_primitive",
        "after_terminal_primitive",
        "before_cancel_response_return",
    ):
        assert token not in installed_headers
        assert token not in symbols.stdout
        assert token not in object_strings.stdout
    assert not (include_root / "pure_plan_motion_server_test_seam.hpp").exists()
    assert not (include_root / "action_execution_state.hpp").exists()
    assert not (include_root / "accepted_goal_finalizer.hpp").exists()

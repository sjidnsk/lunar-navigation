import os
import subprocess
import time
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "start_p4_vehicle_interface.sh"


def _write_executable(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def _test_environment(
    tmp_path: Path, *, duplicate: bool = False, setup_reads_unset_variable: bool = False
):
    p4_root = tmp_path / "P4"
    ros_setup = tmp_path / "ros_setup.bash"
    vehicle_setup = p4_root / "vehicle_interface/install/setup.bash"
    node = p4_root / "vehicle_interface/install/lunar_car_ctrl/lib/lunar_car_ctrl/lunar_car_node"
    capture = tmp_path / "capture.txt"
    fake_bin = tmp_path / "bin"

    ros_setup.parent.mkdir(parents=True, exist_ok=True)
    ros_setup.write_text(
        ("printf '%s' \"$AMENT_TRACE_SETUP_FILES\" >/dev/null\n" if setup_reads_unset_variable else "")
        + "export TEST_ROS_SETUP=loaded\n",
        encoding="utf-8",
    )
    vehicle_setup.parent.mkdir(parents=True, exist_ok=True)
    vehicle_setup.write_text("export TEST_VEHICLE_SETUP=loaded\n", encoding="utf-8")
    _write_executable(
        node,
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$ROS_DOMAIN_ID|$RMW_IMPLEMENTATION|$ROS_LOCALHOST_ONLY\" > \"$P4_TEST_CAPTURE\"\n"
        "printf '%s\\n' \"$@\" >> \"$P4_TEST_CAPTURE\"\n"
        "sleep 30\n",
    )
    _write_executable(
        fake_bin / "pgrep",
        "#!/usr/bin/env bash\n"
        + ("printf '%s\\n' '4321 existing lunar_car_node'\nexit 0\n" if duplicate else "exit 1\n"),
    )

    env = os.environ.copy()
    env.pop("AMENT_TRACE_SETUP_FILES", None)
    env.update(
        {
            "P4_ROOT": str(p4_root),
            "ROS_SETUP": str(ros_setup),
            "P4_TEST_CAPTURE": str(capture),
            "P4_STARTUP_WAIT_SECONDS": "0.2",
            "PATH": f"{fake_bin}:{env['PATH']}",
        }
    )
    return env, p4_root, node, capture


def _stop_started_process(p4_root: Path) -> None:
    pid_file = p4_root / "run/lunar_car_ctrl.pid"
    if not pid_file.exists():
        return
    pid = int(pid_file.read_text(encoding="utf-8").strip())
    try:
        os.kill(pid, 15)
    except ProcessLookupError:
        pass
    for _ in range(20):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.05)


def test_starts_p4_copy_with_safe_defaults(tmp_path):
    env, p4_root, _, capture = _test_environment(tmp_path)
    try:
        result = subprocess.run(
            ["bash", str(SCRIPT)], env=env, text=True, capture_output=True, timeout=5
        )
        assert result.returncode == 0, result.stderr
        assert (p4_root / "run/lunar_car_ctrl.pid").is_file()
        values = capture.read_text(encoding="utf-8").splitlines()
        assert values[0] == "10|rmw_cyclonedds_cpp|0"
        assert "tcp_host:=192.168.10.23" in values
        assert "tcp_port:=6668" in values
        assert "big_endian:=false" in values
        assert "max_linear_speed:=0.2" in values
        assert "control_mode:=park" in values
    finally:
        _stop_started_process(p4_root)


def test_refuses_to_start_when_an_interface_is_already_running(tmp_path):
    env, p4_root, _, capture = _test_environment(tmp_path, duplicate=True)
    result = subprocess.run(
        ["bash", str(SCRIPT)], env=env, text=True, capture_output=True, timeout=5
    )
    assert result.returncode != 0
    assert "已经存在" in result.stderr
    assert not capture.exists()
    assert not (p4_root / "run/lunar_car_ctrl.pid").exists()


def test_reports_missing_installed_node(tmp_path):
    env, _, node, _ = _test_environment(tmp_path)
    node.unlink()
    result = subprocess.run(
        ["bash", str(SCRIPT)], env=env, text=True, capture_output=True, timeout=5
    )
    assert result.returncode != 0
    assert "没有找到接口程序" in result.stderr


def test_ros_setup_can_read_optional_unset_variables(tmp_path):
    env, p4_root, _, _ = _test_environment(
        tmp_path, setup_reads_unset_variable=True
    )
    try:
        result = subprocess.run(
            ["bash", str(SCRIPT)], env=env, text=True, capture_output=True, timeout=5
        )
        assert result.returncode == 0, result.stderr
    finally:
        _stop_started_process(p4_root)

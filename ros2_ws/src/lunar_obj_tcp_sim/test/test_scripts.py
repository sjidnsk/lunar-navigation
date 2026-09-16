from pathlib import Path
import os
import subprocess


ROOT = Path(__file__).resolve().parents[4]
SCRIPTS = ROOT / "scripts" / "simulation"


def run_script(name, *args, env=None):
    return subprocess.run(
        [str(SCRIPTS / name), *args],
        cwd=ROOT,
        env={**os.environ, **(env or {})},
        text=True,
        capture_output=True,
        check=False,
    )


def test_prepare_help_does_not_require_ros_environment():
    result = run_script("prepare_obj_map.sh", "--help", env={"ROS_DISTRO": ""})
    assert result.returncode == 0, result.stderr
    assert "--auto-center" in result.stdout
    assert "--preset" in result.stdout
    assert "--exclude-file" in result.stdout


def test_run_dry_run_exposes_runtime_overrides_without_connecting(tmp_path):
    result = run_script(
        "run_obj_tcp_sim.sh",
        "--host", "10.0.0.7",
        "--port", "7002",
        "--map", str(tmp_path),
        "--mode", "nav",
        "--rviz",
        "--dry-run",
        env={"ROS_DISTRO": "jazzy", "ROS_DOMAIN_ID": ""},
    )
    assert result.returncode == 0, result.stderr
    assert "ROS_DOMAIN_ID=74" in result.stdout
    assert "python3 -m lunar_obj_tcp_sim.operator" in result.stdout
    assert "host:=10.0.0.7" in result.stdout
    assert "port:=7002" in result.stdout
    assert "mode:=nav" in result.stdout
    assert "start_rviz:=true" in result.stdout
    assert "/lunar_sim/cmd_vel" in result.stdout


def test_run_rejects_unknown_mode_before_ros_setup(tmp_path):
    result = run_script(
        "run_obj_tcp_sim.sh", "--map", str(tmp_path), "--mode", "invalid", "--dry-run"
    )
    assert result.returncode == 2
    assert "mode must be explore or nav" in result.stderr


def test_build_places_colcon_log_base_before_build_verb(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    recorder = bin_dir / "colcon"
    recorder.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\"\n", encoding="utf-8")
    recorder.chmod(0o755)
    result = run_script(
        "build.sh",
        env={
            "ROS_DISTRO": "jazzy",
            "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
            "LUNAR_OBJ_TCP_SIM_BUILD_BASE": str(tmp_path / "artifacts"),
        },
    )
    assert result.returncode == 0, result.stderr
    assert "-DCMAKE_BUILD_TYPE=Release" in result.stdout
    assert result.stdout.splitlines()[:3] == [
        "--log-base", str(tmp_path / "artifacts" / "log"), "build"
    ]


def test_run_config_alone_preserves_file_values(tmp_path):
    map_dir = tmp_path / "configured map"
    map_dir.mkdir()
    config = tmp_path / "simulation.yaml"
    source = (ROOT / "ros2_ws/src/lunar_obj_tcp_sim/config/simulation.yaml").read_text()
    source = source.replace("127.0.0.1", "10.8.0.4")
    source = source.replace('map_directory: ""', f'map_directory: "{map_dir}"')
    source = source.replace("mode: explore", "mode: nav")
    source = source.replace("prefix: /lunar_sim", "prefix: /configured")
    config.write_text(source, encoding="utf-8")

    result = run_script("run_obj_tcp_sim.sh", "--config", str(config), "--dry-run")

    assert result.returncode == 0, result.stderr
    assert "host:=" not in result.stdout
    assert "port:=" not in result.stdout
    assert "map_directory:=" not in result.stdout
    assert "mode:=" not in result.stdout
    assert "command_topic=/configured/cmd_vel" in result.stdout


def test_no_rviz_closes_both_views_regardless_of_option_order(tmp_path):
    result = run_script(
        "run_obj_tcp_sim.sh", "--map", str(tmp_path), "--no-rviz", "--local-rviz",
        "--dry-run",
    )
    assert result.returncode == 0, result.stderr
    assert "start_rviz:=false" in result.stdout
    assert "start_local_rviz:=false" in result.stdout


def test_build_rejects_an_inherited_different_ros_distribution(tmp_path):
    result = run_script(
        "build.sh",
        env={"ROS_DISTRO": "humble", "LUNAR_ROS_DISTRO": "jazzy",
             "LUNAR_OBJ_TCP_SIM_BUILD_BASE": str(tmp_path)},
    )
    assert result.returncode == 2
    assert "mixed ROS distributions" in result.stderr


def test_run_rejects_an_inherited_different_ros_distribution(tmp_path):
    result = run_script(
        "run_obj_tcp_sim.sh", "--map", str(tmp_path),
        env={"ROS_DISTRO": "humble", "LUNAR_ROS_DISTRO": "jazzy"},
    )
    assert result.returncode == 2
    assert "mixed ROS distributions" in result.stderr

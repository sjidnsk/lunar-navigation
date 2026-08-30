"""Contract checks for the copy-paste operator entry points."""

import json
import os
import subprocess
from pathlib import Path


BUNDLE_ROOT = Path(__file__).resolve().parents[1]


def test_start_all_starts_planner_and_wheeled_controller() -> None:
    script = (BUNDLE_ROOT / "scripts/start_all.sh").read_text(encoding="utf-8")

    assert "lunar_pure_planner_ros pure_planner.launch.py" in script
    assert "lunar_pure_wheeled_controller pure_wheeled_controller.launch.py" in script
    assert 'source "${bundle_root}/ros2_ws/install/setup.bash"' in script


def test_start_all_passes_grid_v1_mode_to_the_real_planner_invocation(
    tmp_path: Path,
) -> None:
    captured = tmp_path / "ros2-argv.jsonl"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_ros2 = fake_bin / "ros2"
    fake_ros2.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "with open(os.environ['LUNAR_CAPTURED_ROS2_ARGV'], 'a', encoding='utf-8') as stream:\n"
        "    stream.write(json.dumps(sys.argv[1:]) + '\\n')\n",
        encoding="utf-8",
    )
    fake_ros2.chmod(0o755)
    bash_env = tmp_path / "bash-env"
    bash_env.write_text(
        "source() {\n"
        "  if [ \"$1\" = /opt/ros/humble/setup.bash ] || "
        "[[ \"$1\" = */ros2_ws/install/setup.bash ]]; then return 0; fi\n"
        "  builtin source \"$@\"\n"
        "}\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env["LUNAR_CAPTURED_ROS2_ARGV"] = str(captured)
    env["BASH_ENV"] = str(bash_env)

    completed = subprocess.run(
        [BUNDLE_ROOT / "scripts/start_all.sh"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        timeout=10.0,
    )

    assert completed.returncode == 0, completed.stderr
    argv = [json.loads(line) for line in captured.read_text(encoding="utf-8").splitlines()]
    assert [
        "launch",
        "lunar_pure_planner_ros",
        "pure_planner.launch.py",
        "platform_type:=wheel",
        "wheel_planner_mode:=grid_traversability_v1",
    ] in argv


def test_send_goal_prompts_for_planar_goal_and_can_enable_trusted_bridge() -> None:
    script_path = BUNDLE_ROOT / "scripts/send_goal.sh"
    assert script_path.exists(), "operator goal-entry script must exist"

    script = script_path.read_text(encoding="utf-8")
    assert "目标 x (odom, m): " in script
    assert "目标 y (odom, m): " in script
    assert "/Car/T4/plan_motion" in script
    assert "lunar_planning_msgs/action/PlanMotion" in script
    assert "trusted_bridge_once" in script
    assert 'source "${bundle_root}/ros2_ws/install/setup.bash"' in script

"""Contract checks for the copy-paste operator entry points."""

from pathlib import Path


BUNDLE_ROOT = Path(__file__).resolve().parents[1]


def test_start_all_starts_planner_and_wheeled_controller() -> None:
    script = (BUNDLE_ROOT / "scripts/start_all.sh").read_text(encoding="utf-8")

    assert "lunar_pure_planner_ros pure_planner.launch.py" in script
    assert "lunar_pure_wheeled_controller pure_wheeled_controller.launch.py" in script
    assert 'source "${bundle_root}/ros2_ws/install/setup.bash"' in script


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

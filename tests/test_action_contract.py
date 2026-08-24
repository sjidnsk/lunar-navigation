"""Contract checks for the shared PlanMotion action."""

import hashlib
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ACTION_PATH = REPOSITORY_ROOT / "ros2_ws/src/lunar_planning_msgs/action/PlanMotion.action"
BASELINE_ACTION_SHA256 = (
    "5bbc8bfe85d422a798ff17e44b605b0da043795a4eecaf8a6db8c22853641f2e"
)


def test_plan_motion_goal_has_exact_environment_contract() -> None:
    action_text = ACTION_PATH.read_text(encoding="utf-8")
    goal = action_text.split("---", 1)[0].splitlines()
    assert "uint8 LUNAR_SURFACE=1" in goal
    assert "uint8 LAVA_TUBE=2" in goal
    assert "uint8 environment_mode" in goal
    assert goal.index("uint8 environment_mode") < goal.index("string request_id")

    remaining = action_text
    for line in (
        "uint8 LUNAR_SURFACE=1",
        "uint8 LAVA_TUBE=2",
        "uint8 environment_mode",
    ):
        remaining = remaining.replace(f"{line}\n", "")
    assert hashlib.sha256(remaining.encode("utf-8")).hexdigest() == BASELINE_ACTION_SHA256

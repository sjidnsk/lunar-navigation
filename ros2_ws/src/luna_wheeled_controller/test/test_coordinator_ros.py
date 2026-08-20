"""ROS action-message adaptation tests for the task execution coordinator."""

from __future__ import annotations

from lunar_planning_msgs.msg import GoalRegion

from luna_wheeled_controller.coordinator_node_support import make_ros_plan_goal
from luna_wheeled_controller.coordinator import ActiveMission, ExecutionGoal, make_plan_request


def test_ros_plan_goal_preserves_external_goal_and_active_mission_identity() -> None:
    region = GoalRegion()
    region.goal_id = "frontier-17"
    request = make_plan_request(
        ActiveMission(mission_id="mission-a", revision=7),
        ExecutionGoal(region=region),
        request_id="wheel-1",
    )

    action_goal = make_ros_plan_goal(request)

    assert action_goal.request_id == "wheel-1"
    assert action_goal.mission_id == "mission-a"
    assert action_goal.mission_revision == 7
    assert action_goal.goal.goal_id == "frontier-17"
    assert not action_goal.replace_active_request

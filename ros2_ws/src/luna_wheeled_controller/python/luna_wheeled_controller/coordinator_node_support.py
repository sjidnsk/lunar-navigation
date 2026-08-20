"""ROS-message adaptation helpers for the task execution coordinator."""

from __future__ import annotations

from lunar_planning_msgs.action import PlanMotion

from .coordinator import PlanRequest


def make_ros_plan_goal(request: PlanRequest) -> PlanMotion.Goal:
    goal = PlanMotion.Goal()
    goal.request_id = request.request_id
    goal.mission_id = request.mission_id
    goal.mission_revision = request.mission_revision
    goal.goal = request.region
    goal.replace_active_request = False
    return goal

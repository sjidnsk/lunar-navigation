from __future__ import annotations

import math

from lunar_navigation_msgs.msg import MotionExecutionFeedback

from lunar_exploration_policy.coordinator import (
    ExecutionFeedback,
    PlannerGoal,
)
from lunar_exploration_policy.identity import DecisionIdentity
from lunar_exploration_policy.ros_runtime import (
    feedback_from_ros,
    map_pose_from_odom,
    planner_goal_to_ros,
)


def _identity() -> DecisionIdentity:
    return DecisionIdentity(1, "map", "robot", 3, "DECISION_BOUNDARY", "candidates")


def test_planner_goal_to_ros_preserves_frozen_identity_and_theta() -> None:
    goal = PlannerGoal(
        "request", "mission", 1, _identity(), 4,
        (1.0, 2.0, 3.0), 0.2, math.pi / 4.0, math.pi / 24.0,
    )

    message = planner_goal_to_ros(goal, stamp_ns=7)

    assert message.request_id == "request"
    assert message.mission_id == "mission"
    assert message.mission_revision == 1
    assert message.replace_active_request is False
    assert message.goal.goal_id.endswith("/4")
    assert message.goal.goal_type == message.goal.POINT
    assert (message.goal.point.x, message.goal.point.y, message.goal.point.z) == (1.0, 2.0, 3.0)
    assert message.goal.position_tolerance_m == 0.2
    assert message.goal.has_yaw_constraint is True
    assert message.goal.yaw_rad == math.pi / 4.0


def test_hopper_planner_goal_has_no_yaw_constraint() -> None:
    goal = PlannerGoal(
        "request", "mission", 1, _identity(), 4,
        (1.0, 2.0, 3.0), 0.0, None, 0.0,
    )

    message = planner_goal_to_ros(goal, stamp_ns=7)

    assert message.goal.has_yaw_constraint is False
    assert message.goal.position_tolerance_m == 0.0


def test_feedback_mapping_is_explicit() -> None:
    message = MotionExecutionFeedback()
    message.sequence = 9
    message.platform_type = message.LEGGED
    message.plan_id = "plan"
    message.state = message.SEGMENT_COMPLETE

    assert feedback_from_ros(message) == ExecutionFeedback(
        9, "LEGGED", "plan", "SEGMENT_COMPLETE"
    )


def test_planar_map_from_odom_composition() -> None:
    pose = map_pose_from_odom(
        odom_xyz=(2.0, 0.0, 0.5),
        odom_yaw_rad=math.pi / 2.0,
        map_from_odom_xyz=(10.0, 20.0, 1.0),
        map_from_odom_yaw_rad=math.pi / 2.0,
    )

    assert math.isclose(pose.x_m, 10.0, abs_tol=1e-12)
    assert math.isclose(pose.y_m, 22.0, abs_tol=1e-12)
    assert math.isclose(pose.elevation_m, 1.5, abs_tol=1e-12)
    assert math.isclose(pose.yaw_rad, -math.pi, abs_tol=1e-12)


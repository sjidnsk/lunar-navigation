"""ROS 消息与平台无关闭环类型之间的显式转换。"""

from __future__ import annotations

import math

from lunar_navigation_msgs.msg import MotionExecutionFeedback
from lunar_planning_msgs.action import PlanMotion

from lunar_policy_training.environment.observation_builder import Pose2

from .coordinator import ExecutionFeedback, PlannerGoal


_PLATFORMS = {
    MotionExecutionFeedback.WHEELED: "WHEELED",
    MotionExecutionFeedback.LEGGED: "LEGGED",
    MotionExecutionFeedback.HOPPER: "HOPPER",
}
_EXECUTION_STATES = {
    MotionExecutionFeedback.IDLE: "IDLE",
    MotionExecutionFeedback.ACCEPTED: "ACCEPTED",
    MotionExecutionFeedback.EXECUTING: "EXECUTING",
    MotionExecutionFeedback.SEGMENT_COMPLETE: "SEGMENT_COMPLETE",
    MotionExecutionFeedback.LANDED_HOLD: "LANDED_HOLD",
    MotionExecutionFeedback.FAILED: "FAILED",
    MotionExecutionFeedback.CANCELED: "CANCELED",
}


def _stamp(message, stamp_ns: int) -> None:
    if type(stamp_ns) is not int or stamp_ns <= 0:
        raise ValueError("stamp_ns must be a positive integer")
    message.sec = stamp_ns // 1_000_000_000
    message.nanosec = stamp_ns % 1_000_000_000


def planner_goal_to_ros(goal: PlannerGoal, *, stamp_ns: int):
    if not isinstance(goal, PlannerGoal):
        raise TypeError("goal must use PlannerGoal")
    message = PlanMotion.Goal()
    message.request_id = goal.request_id
    message.mission_id = goal.mission_id
    message.mission_revision = goal.mission_revision
    message.replace_active_request = False
    _stamp(message.goal.header.stamp, stamp_ns)
    message.goal.header.frame_id = "map"
    message.goal.goal_id = f"interface-v1/frontier/{goal.frontier_index}"
    message.goal.goal_type = message.goal.POINT
    message.goal.point.x, message.goal.point.y, message.goal.point.z = goal.target_xyz
    message.goal.position_tolerance_m = goal.tolerance_m
    message.goal.has_yaw_constraint = goal.theta_rad is not None
    if goal.theta_rad is not None:
        message.goal.yaw_rad = goal.theta_rad
        message.goal.yaw_tolerance_rad = goal.yaw_tolerance_rad
    return message


def feedback_from_ros(message: MotionExecutionFeedback) -> ExecutionFeedback:
    if not isinstance(message, MotionExecutionFeedback):
        raise TypeError("message must use MotionExecutionFeedback")
    try:
        platform = _PLATFORMS[message.platform_type]
        state = _EXECUTION_STATES[message.state]
    except KeyError as error:
        raise ValueError("execution feedback enum is invalid") from error
    return ExecutionFeedback(int(message.sequence), platform, message.plan_id, state)


def _normalize_angle(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def map_pose_from_odom(
    *,
    odom_xyz: tuple[float, float, float],
    odom_yaw_rad: float,
    map_from_odom_xyz: tuple[float, float, float],
    map_from_odom_yaw_rad: float,
) -> Pose2:
    values = (*odom_xyz, odom_yaw_rad, *map_from_odom_xyz, map_from_odom_yaw_rad)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("pose transform must be finite")
    cosine = math.cos(map_from_odom_yaw_rad)
    sine = math.sin(map_from_odom_yaw_rad)
    x_m = map_from_odom_xyz[0] + cosine * odom_xyz[0] - sine * odom_xyz[1]
    y_m = map_from_odom_xyz[1] + sine * odom_xyz[0] + cosine * odom_xyz[1]
    return Pose2(
        x_m=x_m,
        y_m=y_m,
        yaw_rad=_normalize_angle(map_from_odom_yaw_rad + odom_yaw_rad),
        frame_id="map",
        elevation_m=map_from_odom_xyz[2] + odom_xyz[2],
    )


__all__ = ["feedback_from_ros", "map_pose_from_odom", "planner_goal_to_ros"]

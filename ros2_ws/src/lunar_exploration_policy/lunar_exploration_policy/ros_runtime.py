"""ROS 消息与平台无关闭环类型之间的显式转换。"""

from __future__ import annotations

import math

from lunar_navigation_msgs.msg import MotionExecutionFeedback
from lunar_planning_msgs.action import PlanMotion
from lunar_planning_msgs.msg import MotionReference

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


def feedback_from_ros(
    message: MotionExecutionFeedback,
    *,
    expected_frame: str,
    now_ns: int,
    maximum_age_ns: int,
) -> ExecutionFeedback:
    if not isinstance(message, MotionExecutionFeedback):
        raise TypeError("message must use MotionExecutionFeedback")
    try:
        platform = _PLATFORMS[message.platform_type]
        state = _EXECUTION_STATES[message.state]
    except KeyError as error:
        raise ValueError("execution feedback enum is invalid") from error
    if not message.plan_id or not message.segment_id:
        raise ValueError("execution feedback identity is incomplete")
    stamp_ns = (
        int(message.header.stamp.sec) * 1_000_000_000
        + int(message.header.stamp.nanosec)
    )
    if stamp_ns <= 0:
        raise ValueError("execution feedback stamp is invalid")
    if message.header.frame_id != expected_frame:
        raise ValueError("execution feedback frame is invalid")
    if (
        type(now_ns) is not int
        or type(maximum_age_ns) is not int
        or maximum_age_ns <= 0
        or stamp_ns > now_ns
        or now_ns - stamp_ns > maximum_age_ns
    ):
        raise ValueError("execution feedback is stale or future-dated")
    return ExecutionFeedback(
        int(message.sequence),
        platform,
        message.plan_id,
        state,
        message.segment_id,
        stamp_ns,
    )


def reference_segment_id(
    reference: MotionReference, platform_type: str
) -> str:
    """提取外部执行反馈必须回传的唯一 segment 身份。"""
    if not isinstance(reference, MotionReference):
        raise TypeError("reference must use MotionReference")
    expected = {
        "WHEELED": MotionReference.WHEELED,
        "LEGGED": MotionReference.LEGGED,
        "HOPPER": MotionReference.HOPPER,
    }
    try:
        expected_type = expected[platform_type]
    except KeyError as error:
        raise ValueError("platform_type is invalid") from error
    if reference.platform_type != expected_type or not reference.plan_id:
        raise ValueError("motion reference identity is invalid")
    if platform_type != "HOPPER":
        return reference.plan_id
    if len(reference.hops) != 1 or not reference.hops[0].segment_id:
        raise ValueError("hopper reference requires one identified hop segment")
    return reference.hops[0].segment_id


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


__all__ = [
    "feedback_from_ros",
    "map_pose_from_odom",
    "planner_goal_to_ros",
    "reference_segment_id",
]

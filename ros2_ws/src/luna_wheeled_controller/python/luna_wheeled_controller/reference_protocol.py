"""Convert certified ROS references into fail-closed WHEELED control input."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, isfinite

from lunar_planning_msgs.msg import MotionReference


@dataclass(frozen=True)
class ParsedReference:
    plan_id: str
    path_xy_yaw: tuple[tuple[float, float, float], ...]
    reason: str | None


@dataclass(frozen=True)
class FeedbackIdentity:
    plan_id: str
    segment_id: str


def _yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float | None:
    if not all(isfinite(value) for value in (x, y, z, w)):
        return None
    norm_squared = x * x + y * y + z * z + w * w
    if norm_squared <= 0.0:
        return None
    return atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def parse_wheeled_reference(reference: MotionReference) -> ParsedReference:
    if reference.platform_type != MotionReference.WHEELED or not reference.plan_id:
        return ParsedReference("", (), "INVALID_REFERENCE")
    path: list[tuple[float, float, float]] = []
    for stamped in reference.path_preview.poses:
        pose = stamped.pose
        yaw = _yaw_from_quaternion(
            pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w
        )
        if yaw is None or not all(isfinite(value) for value in (pose.position.x, pose.position.y)):
            return ParsedReference("", (), "INVALID_REFERENCE")
        path.append((pose.position.x, pose.position.y, yaw))
    if not path:
        return ParsedReference("", (), "INVALID_REFERENCE")
    return ParsedReference(reference.plan_id, tuple(path), None)


def make_feedback_identity(plan_id: str) -> FeedbackIdentity:
    if not plan_id:
        raise ValueError("plan_id must be nonempty")
    return FeedbackIdentity(plan_id=plan_id, segment_id=plan_id)

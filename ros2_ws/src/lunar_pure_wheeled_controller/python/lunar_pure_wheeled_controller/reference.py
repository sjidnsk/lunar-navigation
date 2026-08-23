"""Validation and extraction of a wheeled motion reference."""

from dataclasses import dataclass
import math

from lunar_planning_msgs.msg import MotionReference


@dataclass(frozen=True)
class ParsedReference:
    """Trackable data extracted from a valid motion reference."""

    plan_id: str
    path_xy_yaw: tuple[tuple[float, float, float], ...]
    reason: str | None


def parse_reference(reference: MotionReference) -> ParsedReference:
    """Return a finite XY-yaw path, or the reason the reference is unusable."""
    if reference.platform_type != MotionReference.WHEELED or not reference.plan_id:
        return ParsedReference("", (), "INVALID_REFERENCE")

    path = []
    for pose_stamped in reference.path_preview.poses:
        pose = pose_stamped.pose
        x = float(pose.position.x)
        y = float(pose.position.y)
        qx = float(pose.orientation.x)
        qy = float(pose.orientation.y)
        qz = float(pose.orientation.z)
        qw = float(pose.orientation.w)
        if not all(math.isfinite(value) for value in (x, y, qx, qy, qz, qw)):
            return ParsedReference("", (), "INVALID_REFERENCE")
        quaternion_norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
        if not math.isfinite(quaternion_norm) or quaternion_norm == 0.0:
            return ParsedReference("", (), "INVALID_REFERENCE")
        qx /= quaternion_norm
        qy /= quaternion_norm
        qz /= quaternion_norm
        qw /= quaternion_norm
        yaw = math.atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz),
        )
        if not math.isfinite(yaw):
            return ParsedReference("", (), "INVALID_REFERENCE")
        path.append((x, y, yaw))

    if not path:
        return ParsedReference("", (), "INVALID_REFERENCE")
    return ParsedReference(reference.plan_id, tuple(path), None)

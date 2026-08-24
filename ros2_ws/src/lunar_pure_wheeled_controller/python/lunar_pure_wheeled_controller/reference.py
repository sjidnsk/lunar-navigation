"""Validation and extraction of a wheeled path."""

from dataclasses import dataclass
import math

from nav_msgs.msg import Path


@dataclass(frozen=True)
class ParsedReference:
    """Trackable data extracted from a valid path."""

    path_xy_yaw: tuple[tuple[float, float, float], ...]
    reason: str | None


def parse_path(path: Path) -> ParsedReference:
    """Return a finite XY-yaw path, or the reason the path is unusable."""
    points = []
    for pose_stamped in path.poses:
        pose = pose_stamped.pose
        x = float(pose.position.x)
        y = float(pose.position.y)
        qx = float(pose.orientation.x)
        qy = float(pose.orientation.y)
        qz = float(pose.orientation.z)
        qw = float(pose.orientation.w)
        if not all(math.isfinite(value) for value in (x, y, qx, qy, qz, qw)):
            return ParsedReference((), "INVALID_PATH")
        quaternion_norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
        if not math.isfinite(quaternion_norm) or quaternion_norm == 0.0:
            return ParsedReference((), "INVALID_PATH")
        qx /= quaternion_norm
        qy /= quaternion_norm
        qz /= quaternion_norm
        qw /= quaternion_norm
        yaw = math.atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz),
        )
        if not math.isfinite(yaw):
            return ParsedReference((), "INVALID_PATH")
        points.append((x, y, yaw))

    if not points:
        return ParsedReference((), "INVALID_PATH")
    return ParsedReference(tuple(points), None)

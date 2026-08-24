"""Validation and extraction of a wheeled motion reference."""

from dataclasses import dataclass
import math

from lunar_planning_msgs.msg import MotionReference


@dataclass(frozen=True)
class TrajectorySample:
    """One executable map-frame wheel trajectory sample."""

    x_m: float
    y_m: float
    yaw_rad: float
    signed_speed_mps: float
    yaw_rate_radps: float


@dataclass(frozen=True)
class ParsedReference:
    """Trackable data extracted from a valid motion reference."""

    plan_id: str
    path_xy_yaw: tuple[tuple[float, float, float], ...]
    trajectory_samples: tuple[TrajectorySample, ...]
    reason: str | None


def _yaw_from_quaternion(qx: float, qy: float, qz: float, qw: float) -> float | None:
    if not all(math.isfinite(value) for value in (qx, qy, qz, qw)):
        return None
    quaternion_norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if not math.isfinite(quaternion_norm) or quaternion_norm == 0.0:
        return None
    qx /= quaternion_norm
    qy /= quaternion_norm
    qz /= quaternion_norm
    qw /= quaternion_norm
    yaw = math.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )
    return yaw if math.isfinite(yaw) else None


def _invalid_reference() -> ParsedReference:
    return ParsedReference("", (), (), "INVALID_REFERENCE")


def parse_reference(reference: MotionReference) -> ParsedReference:
    """Return executable trajectory data or a compatible finite XY-yaw path."""
    if reference.platform_type != MotionReference.WHEELED or not reference.plan_id:
        return _invalid_reference()

    if reference.trajectory.points:
        if len(reference.trajectory.points) < 2:
            return _invalid_reference()
        samples = []
        path = []
        for point in reference.trajectory.points:
            if len(point.transforms) != 1 or len(point.velocities) != 1:
                return _invalid_reference()
            transform = point.transforms[0]
            velocity = point.velocities[0]
            x = float(transform.translation.x)
            y = float(transform.translation.y)
            qx = float(transform.rotation.x)
            qy = float(transform.rotation.y)
            qz = float(transform.rotation.z)
            qw = float(transform.rotation.w)
            linear_x = float(velocity.linear.x)
            linear_y = float(velocity.linear.y)
            yaw_rate = float(velocity.angular.z)
            if not all(math.isfinite(value) for value in (
                x, y, qx, qy, qz, qw, linear_x, linear_y, yaw_rate,
            )):
                return _invalid_reference()
            yaw = _yaw_from_quaternion(qx, qy, qz, qw)
            if yaw is None:
                return _invalid_reference()
            signed_speed = linear_x * math.cos(yaw) + linear_y * math.sin(yaw)
            if not math.isfinite(signed_speed):
                return _invalid_reference()
            samples.append(TrajectorySample(x, y, yaw, signed_speed, yaw_rate))
            path.append((x, y, yaw))
        return ParsedReference(reference.plan_id, tuple(path), tuple(samples), None)

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
            return _invalid_reference()
        yaw = _yaw_from_quaternion(qx, qy, qz, qw)
        if yaw is None:
            return _invalid_reference()
        path.append((x, y, yaw))

    if not path:
        return _invalid_reference()
    return ParsedReference(reference.plan_id, tuple(path), (), None)

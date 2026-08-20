"""Validation-only status classification for direct Task3 body odometry."""

from __future__ import annotations

from dataclasses import dataclass
import math

from lunar_navigation_msgs.msg import LocalizationStatus
from nav_msgs.msg import Odometry


@dataclass(frozen=True)
class LocalizationStatusPolicy:
    max_age_s: float
    valid_position_variance_max: float
    degraded_position_variance_max: float


def _stamp_ns(message: Odometry) -> int:
    return int(message.header.stamp.sec) * 1_000_000_000 + int(message.header.stamp.nanosec)


def _all_finite(values: object) -> bool:
    return all(math.isfinite(float(value)) for value in values)  # type: ignore[arg-type]


def classify_task3_odometry(
    source: Odometry,
    *,
    now_ns: int,
    policy: LocalizationStatusPolicy,
) -> int:
    """Return a status without changing the source pose, frame, or TF chain."""

    if (
        now_ns <= 0
        or policy.max_age_s < 0.0
        or policy.valid_position_variance_max < 0.0
        or policy.degraded_position_variance_max < policy.valid_position_variance_max
        or source.header.frame_id != "odom"
        or source.child_frame_id != "base_link"
    ):
        return LocalizationStatus.INVALID
    stamp_ns = _stamp_ns(source)
    if stamp_ns <= 0 or now_ns - stamp_ns > int(policy.max_age_s * 1_000_000_000):
        return LocalizationStatus.INVALID
    pose = source.pose.pose
    twist = source.twist.twist
    quaternion = pose.orientation
    orientation_norm = math.sqrt(
        quaternion.x * quaternion.x
        + quaternion.y * quaternion.y
        + quaternion.z * quaternion.z
        + quaternion.w * quaternion.w
    )
    scalar_values = (
        pose.position.x, pose.position.y, pose.position.z,
        quaternion.x, quaternion.y, quaternion.z, quaternion.w,
        twist.linear.x, twist.linear.y, twist.linear.z,
        twist.angular.x, twist.angular.y, twist.angular.z,
    )
    if (
        not _all_finite(scalar_values)
        or not _all_finite(source.pose.covariance)
        or not _all_finite(source.twist.covariance)
        or not math.isfinite(orientation_norm)
        or not math.isclose(orientation_norm, 1.0, rel_tol=0.0, abs_tol=1.0e-3)
    ):
        return LocalizationStatus.INVALID
    position_variance = max(
        float(source.pose.covariance[0]),
        float(source.pose.covariance[7]),
        float(source.pose.covariance[14]),
    )
    if position_variance < 0.0 or position_variance > policy.degraded_position_variance_max:
        return LocalizationStatus.INVALID
    if position_variance > policy.valid_position_variance_max:
        return LocalizationStatus.DEGRADED
    return LocalizationStatus.VALID


def make_localization_status(
    source: Odometry,
    *,
    now_ns: int,
    policy: LocalizationStatusPolicy,
) -> LocalizationStatus:
    """Create the companion status while retaining the Task3 source timestamp."""

    result = LocalizationStatus()
    result.header.stamp = source.header.stamp
    result.header.frame_id = "odom"
    result.status = classify_task3_odometry(source, now_ns=now_ns, policy=policy)
    return result

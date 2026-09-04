"""Small planar geometry helpers shared by controller input adapters."""

import math


def yaw_from_quaternion(qx: float, qy: float, qz: float, qw: float) -> float | None:
    """Return finite planar yaw for a finite, nonzero quaternion."""
    try:
        values = (float(qx), float(qy), float(qz), float(qw))
    except (TypeError, ValueError, OverflowError):
        return None
    if not all(math.isfinite(value) for value in values):
        return None

    norm = math.sqrt(sum(value * value for value in values))
    if not math.isfinite(norm) or norm == 0.0:
        return None

    qx, qy, qz, qw = (value / norm for value in values)
    yaw = math.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )
    return yaw if math.isfinite(yaw) else None

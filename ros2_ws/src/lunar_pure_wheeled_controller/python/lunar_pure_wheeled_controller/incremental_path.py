"""Validation boundary for incremental navigator local paths."""

from dataclasses import dataclass
import math

from nav_msgs.msg import Path

from .geometry import yaw_from_quaternion


@dataclass(frozen=True)
class ParsedIncrementalPath:
    """One validated local path or an explicit clear request."""

    path_xy_yaw: tuple[tuple[float, float, float], ...]
    clear: bool
    reason: str | None


def _invalid_path() -> ParsedIncrementalPath:
    return ParsedIncrementalPath((), False, "INVALID_PATH")


def parse_incremental_path(path: Path) -> ParsedIncrementalPath:
    """Validate a map-frame Path while treating an empty path as a clear."""
    if not path.poses:
        return ParsedIncrementalPath((), True, None)
    if path.header.frame_id != "map":
        return _invalid_path()

    samples: list[tuple[float, float, float]] = []
    for stamped_pose in path.poses:
        pose = stamped_pose.pose
        try:
            x_m = float(pose.position.x)
            y_m = float(pose.position.y)
        except (TypeError, ValueError, OverflowError):
            return _invalid_path()
        yaw_rad = yaw_from_quaternion(
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        )
        if not math.isfinite(x_m) or not math.isfinite(y_m) or yaw_rad is None:
            return _invalid_path()
        samples.append((x_m, y_m, yaw_rad))

    return ParsedIncrementalPath(tuple(samples), False, None)

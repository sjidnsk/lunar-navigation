"""Explicit provisional-message conversions for the external adapter."""

from __future__ import annotations

import math
from typing import Any, Callable, Final


class ConversionError(ValueError):
    """One external message violates the stable internal input contract."""


def _field(message: Any, name: str, path: str | None = None) -> Any:
    if not hasattr(message, name):
        raise ConversionError(f"missing field: {path or name}")
    return getattr(message, name)


def _nonempty(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConversionError(f"{path} must be non-empty")
    return value


def _copy_header(message: Any) -> Any:
    header = _field(message, "header")
    frame_id = _nonempty(
        _field(header, "frame_id", "header.frame_id"), "frame_id"
    )
    stamp = _field(header, "stamp", "header.stamp")
    sec = _field(stamp, "sec", "header.stamp.sec")
    nanosec = _field(stamp, "nanosec", "header.stamp.nanosec")
    if (
        isinstance(sec, bool)
        or not isinstance(sec, int)
        or isinstance(nanosec, bool)
        or not isinstance(nanosec, int)
        or sec < 0
        or not 0 <= nanosec < 1_000_000_000
        or (sec == 0 and nanosec == 0)
    ):
        raise ConversionError("header.stamp must be a valid non-zero ROS time")
    copied_stamp = type(stamp)()
    copied_stamp.sec = sec
    copied_stamp.nanosec = nanosec
    copied_header = type(header)()
    copied_header.stamp = copied_stamp
    copied_header.frame_id = frame_id
    return copied_header


def _finite(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConversionError(f"{path} must be finite")
    converted = float(value)
    if not math.isfinite(converted):
        raise ConversionError(f"{path} must be finite")
    return converted


def convert_localization_status(message: Any) -> Any:
    """Validate and explicitly copy LocalizationStatus v1."""
    header = _copy_header(message)
    if header.frame_id != "odom":
        raise ConversionError("LocalizationStatus frame_id must be odom")
    status = _field(message, "status")
    if (
        isinstance(status, bool)
        or not isinstance(status, int)
        or status not in range(5)
    ):
        raise ConversionError("status is outside UNKNOWN..RELOCALIZING")
    converted = type(message)()
    converted.header = header
    converted.status = status
    return converted


def _copy_point(point: Any, path: str) -> Any:
    converted = type(point)()
    converted.x = _finite(_field(point, "x", f"{path}.x"), f"{path}.x")
    converted.y = _finite(_field(point, "y", f"{path}.y"), f"{path}.y")
    converted.z = _finite(_field(point, "z", f"{path}.z"), f"{path}.z")
    return converted


def _copy_science_region(region: Any, index: int) -> Any:
    path = f"science_regions[{index}]"
    region_id = _nonempty(
        _field(region, "region_id", f"{path}.region_id"),
        f"{path}.region_id",
    )
    objective_id = _nonempty(
        _field(region, "objective_id", f"{path}.objective_id"),
        f"{path}.objective_id",
    )
    priority = _finite(
        _field(region, "priority", f"{path}.priority"), f"{path}.priority"
    )
    if not 0.0 < priority <= 1.0:
        raise ConversionError(f"{path}.priority must be in (0, 1]")
    boundary = _field(region, "boundary", f"{path}.boundary")
    points = list(_field(boundary, "points", f"{path}.boundary.points"))
    if not 3 <= len(points) <= 256:
        raise ConversionError(f"{path}.boundary must contain 3..256 points")
    copied_points = [
        _copy_point(point, f"{path}.boundary.points[{point_index}]")
        for point_index, point in enumerate(points)
    ]
    xy = {(float(point.x), float(point.y)) for point in copied_points}
    if len(xy) != len(copied_points):
        raise ConversionError(f"{path}.boundary contains duplicate vertices")
    copied_boundary = type(boundary)()
    copied_boundary.points = copied_points
    converted = type(region)()
    converted.region_id = region_id
    converted.objective_id = objective_id
    converted.boundary = copied_boundary
    converted.priority = priority
    return converted


def convert_exploration_task(message: Any) -> Any:
    """Validate and explicitly copy ExplorationTask v1."""
    header = _copy_header(message)
    if header.frame_id != "map":
        raise ConversionError("ExplorationTask frame_id must be map")
    mission_id = _nonempty(_field(message, "mission_id"), "mission_id")
    revision = _field(message, "revision")
    if (
        isinstance(revision, bool)
        or not isinstance(revision, int)
        or revision <= 0
    ):
        raise ConversionError("revision must be a positive integer")
    desired_state = _field(message, "desired_state")
    if (
        isinstance(desired_state, bool)
        or not isinstance(desired_state, int)
        or desired_state not in {1, 2, 3}
    ):
        raise ConversionError("desired_state is outside ACTIVE..CANCELED")
    roi = (
        _finite(_field(message, "roi_min_x_m"), "ROI roi_min_x_m"),
        _finite(_field(message, "roi_min_y_m"), "ROI roi_min_y_m"),
        _finite(_field(message, "roi_max_x_m"), "ROI roi_max_x_m"),
        _finite(_field(message, "roi_max_y_m"), "ROI roi_max_y_m"),
    )
    if roi[0] >= roi[2] or roi[1] >= roi[3]:
        raise ConversionError("ROI minimums must be less than maximums")
    regions = list(_field(message, "science_regions"))
    if len(regions) > 64:
        raise ConversionError("science_regions exceeds 64 entries")
    copied_regions = [
        _copy_science_region(region, index)
        for index, region in enumerate(regions)
    ]
    ids = [region.region_id for region in copied_regions]
    if len(set(ids)) != len(ids):
        raise ConversionError(
            "science_regions region_id values must be unique"
        )

    converted = type(message)()
    converted.header = header
    converted.mission_id = mission_id
    converted.revision = revision
    converted.desired_state = desired_state
    converted.roi_min_x_m = roi[0]
    converted.roi_min_y_m = roi[1]
    converted.roi_max_x_m = roi[2]
    converted.roi_max_y_m = roi[3]
    converted.science_regions = copied_regions
    return converted


def convert_motion_execution_feedback(message: Any) -> Any:
    """Validate and explicitly copy MotionExecutionFeedback v1."""
    header = _copy_header(message)
    sequence = _field(message, "sequence")
    if (
        isinstance(sequence, bool)
        or not isinstance(sequence, int)
        or sequence < 1
    ):
        raise ConversionError("sequence must start at 1")
    platform_type = _field(message, "platform_type")
    if (
        isinstance(platform_type, bool)
        or not isinstance(platform_type, int)
        or platform_type not in {1, 2, 3}
    ):
        raise ConversionError("platform_type is outside WHEELED..HOPPER")
    plan_id = _nonempty(_field(message, "plan_id"), "plan_id")
    segment_id = _nonempty(_field(message, "segment_id"), "segment_id")
    state = _field(message, "state")
    if (
        isinstance(state, bool)
        or not isinstance(state, int)
        or state not in range(7)
    ):
        raise ConversionError("state is outside IDLE..CANCELED")
    if state == 4 and platform_type != 3:
        raise ConversionError("LANDED_HOLD is valid only for HOPPER")
    reason_code = _field(message, "reason_code")
    if not isinstance(reason_code, str):
        raise ConversionError("reason_code must be a string")
    if state in {5, 6} and not reason_code.strip():
        raise ConversionError("FAILED or CANCELED requires reason_code")

    converted = type(message)()
    converted.header = header
    converted.sequence = sequence
    converted.platform_type = platform_type
    converted.plan_id = plan_id
    converted.segment_id = segment_id
    converted.state = state
    converted.reason_code = reason_code
    return converted


Converter = Callable[[Any], Any]
CONVERTERS: Final[dict[str, Converter]] = {
    "localization_status_v1": convert_localization_status,
    "exploration_task_v1": convert_exploration_task,
    "motion_execution_feedback_v1": convert_motion_execution_feedback,
}

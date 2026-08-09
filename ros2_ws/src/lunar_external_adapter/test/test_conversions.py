from __future__ import annotations

import math

import pytest
from builtin_interfaces.msg import Time
from geometry_msgs.msg import Point32, Polygon
from std_msgs.msg import Header

from lunar_navigation_msgs.msg import (
    ExplorationTask,
    LocalizationStatus,
    MotionExecutionFeedback,
    ScienceTargetRegion,
)
from lunar_external_adapter.conversions import (
    ConversionError,
    convert_exploration_task,
    convert_localization_status,
    convert_motion_execution_feedback,
)


def header(frame_id: str) -> Header:
    return Header(stamp=Time(sec=10, nanosec=20), frame_id=frame_id)


def science_region() -> ScienceTargetRegion:
    return ScienceTargetRegion(
        region_id="region-1",
        objective_id="objective-1",
        boundary=Polygon(
            points=[
                Point32(x=0.0, y=0.0, z=0.0),
                Point32(x=1.0, y=0.0, z=0.0),
                Point32(x=0.0, y=1.0, z=0.0),
            ]
        ),
        priority=0.8,
    )


def test_localization_conversion_explicitly_copies_valid_message() -> None:
    source = LocalizationStatus(
        header=header("odom"),
        status=LocalizationStatus.DEGRADED,
    )

    converted = convert_localization_status(source)

    assert converted == source
    assert converted is not source
    assert converted.header is not source.header


def test_exploration_conversion_validates_roi_and_copies_regions() -> None:
    source = ExplorationTask(
        header=header("map"),
        mission_id="mission-1",
        revision=3,
        desired_state=ExplorationTask.ACTIVE,
        roi_min_x_m=-10.0,
        roi_min_y_m=-20.0,
        roi_max_x_m=30.0,
        roi_max_y_m=40.0,
        science_regions=[science_region()],
    )

    converted = convert_exploration_task(source)

    assert converted == source
    assert converted is not source
    assert converted.science_regions[0] is not source.science_regions[0]


def test_invalid_feedback_is_not_republished() -> None:
    message = MotionExecutionFeedback(
        header=header("base_link"),
        sequence=1,
        platform_type=MotionExecutionFeedback.WHEELED,
        plan_id="",
        segment_id="segment-1",
        state=MotionExecutionFeedback.EXECUTING,
        reason_code="",
    )

    with pytest.raises(ConversionError, match="plan_id"):
        convert_motion_execution_feedback(message)


@pytest.mark.parametrize(
    "message, expected",
    [
        (
            LocalizationStatus(
                header=header("map"),
                status=LocalizationStatus.VALID,
            ),
            "frame_id",
        ),
        (
            LocalizationStatus(header=header("odom"), status=99),
            "status",
        ),
        (
            ExplorationTask(
                header=header("map"),
                mission_id="mission-1",
                revision=1,
                desired_state=ExplorationTask.ACTIVE,
                roi_min_x_m=0.0,
                roi_min_y_m=0.0,
                roi_max_x_m=math.nan,
                roi_max_y_m=1.0,
            ),
            "ROI",
        ),
        (
            MotionExecutionFeedback(
                header=header("base_link"),
                sequence=0,
                platform_type=MotionExecutionFeedback.LEGGED,
                plan_id="plan-1",
                segment_id="segment-1",
                state=MotionExecutionFeedback.EXECUTING,
            ),
            "sequence",
        ),
        (
            MotionExecutionFeedback(
                header=header("base_link"),
                sequence=1,
                platform_type=MotionExecutionFeedback.WHEELED,
                plan_id="plan-1",
                segment_id="segment-1",
                state=MotionExecutionFeedback.LANDED_HOLD,
            ),
            "LANDED_HOLD",
        ),
    ],
)
def test_semantically_invalid_messages_are_rejected(
    message, expected: str
) -> None:
    converter = {
        LocalizationStatus: convert_localization_status,
        ExplorationTask: convert_exploration_task,
        MotionExecutionFeedback: convert_motion_execution_feedback,
    }[type(message)]

    with pytest.raises(ConversionError, match=expected):
        converter(message)

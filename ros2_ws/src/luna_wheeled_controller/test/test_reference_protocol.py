"""Reference parsing and feedback identity contract tests."""

from __future__ import annotations

from geometry_msgs.msg import PoseStamped
from lunar_planning_msgs.msg import MotionReference

from luna_wheeled_controller.reference_protocol import (
    make_feedback_identity,
    parse_wheeled_reference,
)


def _reference(*, platform: int = MotionReference.WHEELED, plan_id: str = "wheel-7") -> MotionReference:
    reference = MotionReference()
    reference.platform_type = platform
    reference.plan_id = plan_id
    start = PoseStamped()
    start.pose.orientation.w = 1.0
    finish = PoseStamped()
    finish.pose.orientation.w = 1.0
    finish.pose.position.x = 1.0
    reference.path_preview.poses = [start, finish]
    return reference


def test_valid_wheeled_reference_keeps_plan_identity_and_path() -> None:
    parsed = parse_wheeled_reference(_reference())

    assert parsed.reason is None
    assert parsed.plan_id == "wheel-7"
    assert parsed.path_xy_yaw == ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))


def test_non_wheeled_reference_is_invalid_not_executable() -> None:
    parsed = parse_wheeled_reference(_reference(platform=MotionReference.LEGGED))

    assert parsed.reason == "INVALID_REFERENCE"
    assert parsed.path_xy_yaw == ()


def test_feedback_identity_uses_plan_id_as_ground_segment_id() -> None:
    identity = make_feedback_identity("wheel-7")

    assert identity.plan_id == "wheel-7"
    assert identity.segment_id == "wheel-7"

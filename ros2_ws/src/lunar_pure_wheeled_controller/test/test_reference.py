import math

from geometry_msgs.msg import PoseStamped
from lunar_planning_msgs.msg import MotionReference
from nav_msgs.msg import Path

from lunar_pure_wheeled_controller.reference import parse_reference


def make_reference(*, platform: int = MotionReference.WHEELED) -> MotionReference:
    reference = MotionReference()
    reference.plan_id = "wheel-plan"
    reference.platform_type = platform
    reference.path_preview = Path()
    for x in (0.0, 2.0):
        pose = PoseStamped()
        pose.pose.position.x = x
        pose.pose.orientation.w = 1.0
        reference.path_preview.poses.append(pose)
    return reference


def test_wheeled_reference_yields_xy_yaw_path() -> None:
    parsed = parse_reference(make_reference(platform=MotionReference.WHEELED))

    assert parsed.reason is None
    assert parsed.plan_id == "wheel-plan"
    assert parsed.path_xy_yaw == ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0))


def test_non_wheeled_or_empty_reference_is_not_trackable() -> None:
    assert parse_reference(make_reference(platform=MotionReference.LEGGED)).reason == "INVALID_REFERENCE"
    assert parse_reference(MotionReference()).reason == "INVALID_REFERENCE"


def test_empty_path_is_not_trackable() -> None:
    reference = make_reference()
    reference.path_preview.poses.clear()

    assert parse_reference(reference).reason == "INVALID_REFERENCE"


def test_non_finite_pose_is_not_trackable() -> None:
    reference = make_reference()
    reference.path_preview.poses[1].pose.position.x = math.nan

    assert parse_reference(reference).reason == "INVALID_REFERENCE"


def test_non_finite_quaternion_is_not_trackable() -> None:
    reference = make_reference()
    reference.path_preview.poses[1].pose.orientation.z = math.inf

    assert parse_reference(reference).reason == "INVALID_REFERENCE"


def test_zero_quaternion_is_not_trackable() -> None:
    reference = make_reference()
    reference.path_preview.poses[1].pose.orientation.w = 0.0

    assert parse_reference(reference).reason == "INVALID_REFERENCE"


def test_non_unit_quaternion_is_normalized_before_yaw_conversion() -> None:
    reference = make_reference()
    reference.path_preview.poses[1].pose.orientation.z = 0.5
    reference.path_preview.poses[1].pose.orientation.w = 0.5

    parsed = parse_reference(reference)

    assert parsed.reason is None
    assert math.isclose(parsed.path_xy_yaw[1][2], math.pi / 2.0, abs_tol=1e-12)

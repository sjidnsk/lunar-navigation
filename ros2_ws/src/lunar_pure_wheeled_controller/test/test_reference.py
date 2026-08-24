import math

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path

from lunar_pure_wheeled_controller.reference import parse_path


def make_path() -> Path:
    path = Path()
    path.header.frame_id = "map"
    for x in (0.0, 2.0):
        pose = PoseStamped()
        pose.pose.position.x = x
        pose.pose.orientation.w = 1.0
        path.poses.append(pose)
    return path


def test_path_yields_xy_yaw_path() -> None:
    parsed = parse_path(make_path())

    assert parsed.reason is None
    assert parsed.path_xy_yaw == ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0))


def test_empty_path_is_not_trackable() -> None:
    path = Path()

    assert parse_path(path).reason == "INVALID_PATH"


def test_non_finite_pose_is_not_trackable() -> None:
    path = make_path()
    path.poses[1].pose.position.x = math.nan

    assert parse_path(path).reason == "INVALID_PATH"


def test_non_finite_quaternion_is_not_trackable() -> None:
    path = make_path()
    path.poses[1].pose.orientation.z = math.inf

    assert parse_path(path).reason == "INVALID_PATH"


def test_zero_quaternion_is_not_trackable() -> None:
    path = make_path()
    path.poses[1].pose.orientation.w = 0.0

    assert parse_path(path).reason == "INVALID_PATH"


def test_non_unit_quaternion_is_normalized_before_yaw_conversion() -> None:
    path = make_path()
    path.poses[1].pose.orientation.z = 0.5
    path.poses[1].pose.orientation.w = 0.5

    parsed = parse_path(path)

    assert parsed.reason is None
    assert math.isclose(parsed.path_xy_yaw[1][2], math.pi / 2.0, abs_tol=1e-12)

import math

import pytest
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path

from lunar_pure_wheeled_controller.incremental_path import parse_incremental_path


def make_path(*, frame_id: str, points: list[tuple[float, float, float]]) -> Path:
    path = Path()
    path.header.frame_id = frame_id
    for x, y, yaw in points:
        pose = PoseStamped()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.orientation.z = math.sin(yaw / 2.0)
        pose.pose.orientation.w = math.cos(yaw / 2.0)
        path.poses.append(pose)
    return path


def test_nonempty_map_path_becomes_finite_xy_yaw_samples() -> None:
    path = make_path(
        frame_id="map",
        points=[(0.0, 0.0, 0.0), (1.0, 0.0, math.pi / 2.0)],
    )

    parsed = parse_incremental_path(path)

    assert parsed.reason is None
    assert not parsed.clear
    assert parsed.path_xy_yaw[0] == pytest.approx((0.0, 0.0, 0.0))
    assert parsed.path_xy_yaw[1] == pytest.approx((1.0, 0.0, math.pi / 2.0))


def test_empty_path_means_clear_even_without_a_header() -> None:
    parsed = parse_incremental_path(Path())

    assert parsed.clear
    assert parsed.reason is None
    assert parsed.path_xy_yaw == ()


def test_nonempty_path_with_wrong_frame_or_bad_quaternion_is_rejected() -> None:
    wrong_frame = make_path(frame_id="odom", points=[(0.0, 0.0, 0.0)])
    assert parse_incremental_path(wrong_frame).reason == "INVALID_PATH"

    malformed = make_path(frame_id="map", points=[(0.0, 0.0, 0.0)])
    malformed.poses[0].pose.orientation.w = 0.0
    assert parse_incremental_path(malformed).reason == "INVALID_PATH"

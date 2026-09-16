import math

import pytest
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry, Path
from tf2_msgs.msg import TFMessage

from lunar_pure_wheeled_controller.frame import (
    MapFromOdomUpdate,
    map_tracking_state,
    parse_map_from_odom,
)
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


def make_odometry(*, x: float, y: float, yaw: float) -> Odometry:
    odometry = Odometry()
    odometry.pose.pose.position.x = x
    odometry.pose.pose.position.y = y
    odometry.pose.pose.orientation.z = math.sin(yaw / 2.0)
    odometry.pose.pose.orientation.w = math.cos(yaw / 2.0)
    return odometry


def make_tf_message(
    *,
    parent: str,
    child: str,
    x: float = 0.0,
    y: float = 0.0,
    yaw: float = 0.0,
) -> TFMessage:
    message = TFMessage()
    transform = TransformStamped()
    transform.header.frame_id = parent
    transform.child_frame_id = child
    transform.transform.translation.x = x
    transform.transform.translation.y = y
    transform.transform.rotation.z = math.sin(yaw / 2.0)
    transform.transform.rotation.w = math.cos(yaw / 2.0)
    message.transforms.append(transform)
    return message


def test_direct_map_from_odom_transforms_position_and_yaw() -> None:
    update = parse_map_from_odom(
        make_tf_message(
            parent="map",
            child="odom",
            x=10.0,
            y=20.0,
            yaw=math.pi / 2.0,
        )
    )

    assert update.found and update.transform is not None
    state = map_tracking_state(make_odometry(x=2.0, y=0.0, yaw=0.0), update.transform)
    assert state is not None
    assert state.x_m == pytest.approx(10.0)
    assert state.y_m == pytest.approx(22.0)
    assert state.yaw_rad == pytest.approx(math.pi / 2.0)


def test_unrelated_or_invalid_tf_cannot_create_a_map_transform() -> None:
    assert parse_map_from_odom(TFMessage()) == MapFromOdomUpdate(False, None)

    invalid = make_tf_message(parent="map", child="odom")
    invalid.transforms[0].transform.rotation.w = 0.0
    assert parse_map_from_odom(invalid) == MapFromOdomUpdate(True, None)


def test_configured_world_frame_is_checked():
    path = make_path(frame_id="world", points=[(1., 2., 0.), (2., 2., 0.)])
    assert parse_incremental_path(path, "world").reason is None
    assert parse_incremental_path(path).reason == "INVALID_PATH"


@pytest.mark.parametrize('component', ['x', 'y'])
def test_nonfinite_body_translation_cannot_be_used_as_stopped_feedback(component):
    from lunar_pure_wheeled_controller.frame import MapFromOdom
    odometry = make_odometry(x=1., y=0., yaw=0.)
    setattr(odometry.twist.twist.linear, component, float('nan'))
    assert map_tracking_state(odometry, MapFromOdom(0., 0., 0.)) is None

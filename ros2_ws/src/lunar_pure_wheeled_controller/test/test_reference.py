import math

from geometry_msgs.msg import PoseStamped, Transform, Twist
from lunar_planning_msgs.msg import MotionReference
from nav_msgs.msg import Path
from trajectory_msgs.msg import MultiDOFJointTrajectoryPoint

from lunar_pure_wheeled_controller.reference import TrajectorySample, parse_reference


def make_path() -> Path:
    path = Path()
    path.header.frame_id = "map"
    for x in (0.0, 2.0):
        pose = PoseStamped()
        pose.pose.position.x = x
        pose.pose.orientation.w = 1.0
        path.poses.append(pose)
    return path


def append_trajectory_point(
    reference: MotionReference,
    pose_xy_yaw: tuple[float, float, float],
    map_velocity_xy: tuple[float, float],
    yaw_rate_radps: float,
) -> MultiDOFJointTrajectoryPoint:
    point = MultiDOFJointTrajectoryPoint()
    transform = Transform()
    transform.translation.x, transform.translation.y, yaw = pose_xy_yaw
    transform.rotation.z = math.sin(yaw / 2.0)
    transform.rotation.w = math.cos(yaw / 2.0)
    velocity = Twist()
    velocity.linear.x, velocity.linear.y = map_velocity_xy
    velocity.angular.z = yaw_rate_radps
    point.transforms.append(transform)
    point.velocities.append(velocity)
    reference.trajectory.points.append(point)
    return point


def make_trajectory_reference(
    samples: list[tuple[tuple[float, float, float], tuple[float, float], float]],
) -> MotionReference:
    reference = make_reference()
    for pose_xy_yaw, map_velocity_xy, yaw_rate_radps in samples:
        append_trajectory_point(
            reference, pose_xy_yaw, map_velocity_xy, yaw_rate_radps
        )
    return reference


def test_wheeled_reference_yields_xy_yaw_path() -> None:
    parsed = parse_reference(make_reference(platform=MotionReference.WHEELED))

    assert parsed.reason is None
    assert parsed.path_xy_yaw == ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0))
    assert parsed.trajectory_samples == ()


def test_trajectory_keeps_forward_and_reverse_signed_speed() -> None:
    parsed = parse_reference(make_trajectory_reference([
        ((0.0, 0.0, 0.0), (0.2, 0.0), 0.0),
        ((-0.2, 0.0, 0.0), (-0.2, 0.0), 0.0),
    ]))

    assert parsed.reason is None
    assert parsed.trajectory_samples[0].signed_speed_mps == 0.2
    assert parsed.trajectory_samples[1].signed_speed_mps == -0.2
    assert parsed.path_xy_yaw == ((0.0, 0.0, 0.0), (-0.2, 0.0, 0.0))


def test_trajectory_projects_map_velocity_and_preserves_yaw_rate() -> None:
    parsed = parse_reference(make_trajectory_reference([
        ((1.0, 2.0, math.pi / 2.0), (0.0, 0.3), -0.4),
    ]))

    assert parsed.reason is None
    sample = parsed.trajectory_samples[0]
    assert isinstance(sample, TrajectorySample)
    assert sample.x_m == 1.0
    assert sample.y_m == 2.0
    assert math.isclose(sample.yaw_rad, math.pi / 2.0, abs_tol=1e-12)
    assert sample.signed_speed_mps == 0.3
    assert sample.yaw_rate_radps == -0.4


def test_non_finite_trajectory_transform_is_not_trackable() -> None:
    reference = make_trajectory_reference([((0.0, 0.0, 0.0), (0.2, 0.0), 0.0)])
    reference.trajectory.points[0].transforms[0].translation.x = math.nan

    assert parse_reference(reference).reason == "INVALID_REFERENCE"


def test_non_finite_trajectory_velocity_is_not_trackable() -> None:
    reference = make_trajectory_reference([((0.0, 0.0, 0.0), (0.2, 0.0), 0.0)])
    reference.trajectory.points[0].velocities[0].angular.z = math.inf

    assert parse_reference(reference).reason == "INVALID_REFERENCE"


def test_trajectory_point_without_exactly_one_transform_is_not_trackable() -> None:
    reference = make_trajectory_reference([((0.0, 0.0, 0.0), (0.2, 0.0), 0.0)])
    reference.trajectory.points[0].transforms.append(Transform())

    assert parse_reference(reference).reason == "INVALID_REFERENCE"


def test_trajectory_point_without_a_transform_is_not_trackable() -> None:
    reference = make_trajectory_reference([((0.0, 0.0, 0.0), (0.2, 0.0), 0.0)])
    reference.trajectory.points[0].transforms.clear()

    assert parse_reference(reference).reason == "INVALID_REFERENCE"


def test_trajectory_point_without_exactly_one_velocity_is_not_trackable() -> None:
    reference = make_trajectory_reference([((0.0, 0.0, 0.0), (0.2, 0.0), 0.0)])
    reference.trajectory.points[0].velocities.clear()

    assert parse_reference(reference).reason == "INVALID_REFERENCE"


def test_trajectory_point_with_multiple_velocities_is_not_trackable() -> None:
    reference = make_trajectory_reference([((0.0, 0.0, 0.0), (0.2, 0.0), 0.0)])
    reference.trajectory.points[0].velocities.append(Twist())

    assert parse_reference(reference).reason == "INVALID_REFERENCE"


def test_trajectory_is_authoritative_over_a_malformed_path_preview() -> None:
    reference = make_trajectory_reference([((0.0, 0.0, 0.0), (0.2, 0.0), 0.0)])
    reference.path_preview.poses[0].pose.position.x = math.nan

    assert parse_reference(reference).reason is None


def test_malformed_trajectory_does_not_fall_back_to_path_preview() -> None:
    reference = make_trajectory_reference([((0.0, 0.0, 0.0), (0.2, 0.0), 0.0)])
    reference.trajectory.points[0].transforms.clear()

    assert parse_reference(reference).reason == "INVALID_REFERENCE"


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

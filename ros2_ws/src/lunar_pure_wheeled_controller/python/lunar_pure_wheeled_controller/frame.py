"""Direct planar map/odom frame adaptation for incremental path tracking."""

from dataclasses import dataclass
import math

from nav_msgs.msg import Odometry
from tf2_msgs.msg import TFMessage

from .geometry import yaw_from_quaternion
from .tracking import TrackingState


@dataclass(frozen=True)
class MapFromOdom:
    """A direct planar transform from odom coordinates into map coordinates."""

    x_m: float
    y_m: float
    yaw_rad: float


@dataclass(frozen=True)
class MapFromOdomUpdate:
    """Result of inspecting one TF message for the direct map/odom edge."""

    found: bool
    transform: MapFromOdom | None


def parse_map_from_odom(message: TFMessage) -> MapFromOdomUpdate:
    """Extract the last direct ``map <- odom`` transform in one TF message."""
    matching = [
        transform
        for transform in message.transforms
        if transform.header.frame_id == "map" and transform.child_frame_id == "odom"
    ]
    if not matching:
        return MapFromOdomUpdate(False, None)

    transform = matching[-1].transform
    try:
        x_m = float(transform.translation.x)
        y_m = float(transform.translation.y)
    except (TypeError, ValueError, OverflowError):
        return MapFromOdomUpdate(True, None)
    yaw_rad = yaw_from_quaternion(
        transform.rotation.x,
        transform.rotation.y,
        transform.rotation.z,
        transform.rotation.w,
    )
    if not math.isfinite(x_m) or not math.isfinite(y_m) or yaw_rad is None:
        return MapFromOdomUpdate(True, None)
    return MapFromOdomUpdate(True, MapFromOdom(x_m, y_m, yaw_rad))


def map_tracking_state(
    odometry: Odometry,
    transform: MapFromOdom,
) -> TrackingState | None:
    """Compose map-from-odom with odometry into a map-frame tracking state."""
    position = odometry.pose.pose.position
    orientation = odometry.pose.pose.orientation
    try:
        odometry_x_m = float(position.x)
        odometry_y_m = float(position.y)
        transform_x_m = float(transform.x_m)
        transform_y_m = float(transform.y_m)
        transform_yaw_rad = float(transform.yaw_rad)
        linear_mps = float(odometry.twist.twist.linear.x)
        angular_radps = float(odometry.twist.twist.angular.z)
    except (AttributeError, TypeError, ValueError, OverflowError):
        return None
    yaw_rad = yaw_from_quaternion(
        orientation.x,
        orientation.y,
        orientation.z,
        orientation.w,
    )
    if not all(math.isfinite(value) for value in (
        odometry_x_m,
        odometry_y_m,
        linear_mps,
        angular_radps,
        transform_x_m,
        transform_yaw_rad,
        transform_y_m,
    )) or yaw_rad is None:
        return None

    cos_yaw = math.cos(transform_yaw_rad)
    sin_yaw = math.sin(transform_yaw_rad)
    return TrackingState(
        transform_x_m + cos_yaw * odometry_x_m - sin_yaw * odometry_y_m,
        transform_y_m + sin_yaw * odometry_x_m + cos_yaw * odometry_y_m,
        math.atan2(
            math.sin(transform_yaw_rad + yaw_rad),
            math.cos(transform_yaw_rad + yaw_rad),
        ),
        linear_mps,
        angular_radps,
    )

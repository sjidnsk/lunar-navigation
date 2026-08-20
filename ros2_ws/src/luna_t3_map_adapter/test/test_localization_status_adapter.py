from __future__ import annotations

from math import nan
import time

import rclpy
from nav_msgs.msg import Odometry
from lunar_navigation_msgs.msg import LocalizationStatus

from luna_t3_map_adapter.localization_status_adapter import (
    LocalizationStatusPolicy,
    make_localization_status,
    classify_task3_odometry,
)
from luna_t3_map_adapter.localization_status_adapter_node import LocalizationStatusAdapterNode


def _odometry(*, stamp_ns: int = 2_000_000_000) -> Odometry:
    message = Odometry()
    message.header.frame_id = "odom"
    message.child_frame_id = "base_link"
    message.header.stamp.sec = stamp_ns // 1_000_000_000
    message.header.stamp.nanosec = stamp_ns % 1_000_000_000
    message.pose.pose.orientation.w = 1.0
    return message


def _policy() -> LocalizationStatusPolicy:
    return LocalizationStatusPolicy(
        max_age_s=1.0,
        valid_position_variance_max=0.10,
        degraded_position_variance_max=0.50,
    )


def test_valid_task3_body_odometry_produces_valid_status_without_pose_rewrite() -> None:
    message = _odometry()
    message.pose.covariance[0] = 0.01

    status = classify_task3_odometry(message, now_ns=2_500_000_000, policy=_policy())

    assert status == LocalizationStatus.VALID
    assert message.header.frame_id == "odom"
    assert message.child_frame_id == "base_link"


def test_large_but_bounded_covariance_is_degraded() -> None:
    message = _odometry()
    message.pose.covariance[0] = 0.20

    assert classify_task3_odometry(message, now_ns=2_500_000_000, policy=_policy()) == LocalizationStatus.DEGRADED


def test_camera_frame_stale_stamp_or_nonfinite_covariance_is_invalid() -> None:
    camera = _odometry()
    camera.child_frame_id = "car_stereo_left_optical_frame"
    stale = _odometry(stamp_ns=100)
    nonfinite = _odometry()
    nonfinite.pose.covariance[0] = nan

    assert classify_task3_odometry(camera, now_ns=2_500_000_000, policy=_policy()) == LocalizationStatus.INVALID
    assert classify_task3_odometry(stale, now_ns=2_500_000_000, policy=_policy()) == LocalizationStatus.INVALID
    assert classify_task3_odometry(nonfinite, now_ns=2_500_000_000, policy=_policy()) == LocalizationStatus.INVALID


def test_status_message_keeps_task3_stamp_and_uses_odom_frame() -> None:
    source = _odometry()
    source.pose.covariance[0] = 0.01

    status = make_localization_status(source, now_ns=2_500_000_000, policy=_policy())

    assert status.header.frame_id == "odom"
    assert status.header.stamp == source.header.stamp
    assert status.status == LocalizationStatus.VALID


def test_ros_adapter_publishes_status_without_odometry_or_tf_output() -> None:
    rclpy.init()
    adapter = LocalizationStatusAdapterNode()
    observer = rclpy.create_node("localization_status_adapter_test_observer")
    received: list[LocalizationStatus] = []
    observer.create_subscription(
        LocalizationStatus, "/localization/status", received.append, 10
    )
    try:
        source = _odometry(stamp_ns=adapter.get_clock().now().nanoseconds)
        source.pose.covariance[0] = 0.01
        adapter._on_odometry(source)
        deadline = time.monotonic() + 1.0
        while not received and time.monotonic() < deadline:
            rclpy.spin_once(adapter, timeout_sec=0.01)
            rclpy.spin_once(observer, timeout_sec=0.01)
        assert len(received) == 1
        assert received[0].status == LocalizationStatus.VALID
        assert received[0].header.stamp == source.header.stamp
    finally:
        observer.destroy_node()
        adapter.destroy_node()
        rclpy.shutdown()

"""Fail-closed controller-node behavior with real ROS messages."""

from __future__ import annotations

import time

import rclpy
from lunar_navigation_msgs.msg import MotionExecutionFeedback
from lunar_planning_msgs.msg import MotionReference

from luna_wheeled_controller.wheeled_controller_node import WheeledControllerNode


def test_invalid_platform_reference_publishes_failed_feedback_with_its_identity() -> None:
    rclpy.init()
    controller = WheeledControllerNode()
    observer = rclpy.create_node("wheeled_controller_feedback_observer")
    received: list[MotionExecutionFeedback] = []
    observer.create_subscription(
        MotionExecutionFeedback,
        "/execution/motion_feedback",
        received.append,
        10,
    )
    reference = MotionReference()
    reference.platform_type = MotionReference.LEGGED
    reference.plan_id = "leg-7"
    try:
        controller._on_reference(reference)
        deadline = time.monotonic() + 1.0
        while not received and time.monotonic() < deadline:
            rclpy.spin_once(controller, timeout_sec=0.01)
            rclpy.spin_once(observer, timeout_sec=0.01)
        assert len(received) == 1
        assert received[0].state == MotionExecutionFeedback.FAILED
        assert received[0].reason_code == "INVALID_REFERENCE"
        assert (received[0].plan_id, received[0].segment_id) == ("leg-7", "leg-7")
    finally:
        observer.destroy_node()
        controller.destroy_node()
        rclpy.shutdown()

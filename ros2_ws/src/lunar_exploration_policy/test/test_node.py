from __future__ import annotations

import os
from pathlib import Path

import pytest
import rclpy
from rclpy.parameter import Parameter
from lunar_planning_msgs.action import PlanMotion

from lunar_exploration_policy.node import (
    InterfaceV1PolicyNode,
    _timestamps_are_usable,
)


ROOT = Path(__file__).resolve().parents[4]
PROFILE = ROOT / "ros2_ws/src/lunar_navigation_config/config/platform_profiles/wheeled.yaml"
INTERFACE = ROOT / "ros2_ws/src/lunar_navigation_config/config/interface_profiles/default.yaml"


def _real_model_dir() -> Path:
    value = os.environ.get("LUNAR_INTERFACE_V1_MODEL_DIR")
    if not value:
        pytest.skip("real interface-v1 model directory was not supplied")
    return Path(value).resolve(strict=True)


def _configured_node(profile: Path) -> InterfaceV1PolicyNode:
    node = InterfaceV1PolicyNode()
    node.set_parameters(
        [
            Parameter("model_dir", value=str(_real_model_dir())),
            Parameter("platform_profile_file", value=str(profile)),
            Parameter("interface_profile_file", value=str(INTERFACE)),
        ]
    )
    node.trigger_configure()
    return node


def test_state_timestamps_must_follow_terminal_feedback_boundary() -> None:
    assert not _timestamps_are_usable(
        (100, 101, 102, 103, 104),
        now_ns=120,
        minimum_state_time_ns=100,
        maximum_age_ns=30,
        maximum_skew_ns=10,
    )
    assert _timestamps_are_usable(
        (101, 102, 103, 104, 105),
        now_ns=120,
        minimum_state_time_ns=100,
        maximum_age_ns=30,
        maximum_skew_ns=10,
    )


def test_state_timestamps_reject_stale_future_or_skewed_snapshots() -> None:
    common = {
        "now_ns": 1_000,
        "minimum_state_time_ns": 0,
        "maximum_age_ns": 100,
        "maximum_skew_ns": 10,
    }
    assert not _timestamps_are_usable((899, 995), **common)
    assert not _timestamps_are_usable((995, 1_001), **common)
    assert not _timestamps_are_usable((980, 995), **common)
    assert _timestamps_are_usable((990, 995), **common)


def test_policy_feedback_age_default_matches_planner_tracker() -> None:
    rclpy.init()
    node = InterfaceV1PolicyNode()
    try:
        assert node.get_parameter("input_max_age_s").value == 1.0
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_real_model_and_frozen_profile_configure_lifecycle_node() -> None:
    rclpy.init()
    node = _configured_node(PROFILE)
    try:
        assert node._coordinator is not None
        assert node._assembler is not None
        assert node._platform_type == "WHEELED"
        assert len(node._subscriptions) == 7
        assert node._reference_publisher is not None
    finally:
        node.trigger_cleanup()
        node.destroy_node()
        rclpy.shutdown()


def test_modified_profile_fails_closed(tmp_path: Path) -> None:
    modified = tmp_path / "platform_profile.yaml"
    modified.write_bytes(PROFILE.read_bytes() + b"\n# changed\n")
    rclpy.init()
    node = _configured_node(modified)
    try:
        assert node._coordinator is None
        assert not node._subscriptions
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_successful_plan_result_is_published_for_external_execution() -> None:
    class _Completed:
        def __init__(self, result):
            self.result = result

    class _Future:
        def __init__(self, result):
            self._result = result

        def result(self):
            return _Completed(self._result)

    class _Recorder:
        def __init__(self):
            self.messages = []

        def publish(self, message):
            self.messages.append(message)

    result = PlanMotion.Result()
    result.has_reference = True
    result.reference.plan_id = "plan-fed9-1"
    result.reference.platform_type = result.reference.WHEELED
    result.reason_code = "OK"

    rclpy.init()
    node = InterfaceV1PolicyNode()
    accepted = []
    recorder = _Recorder()
    node._enabled = True
    node._platform_type = "WHEELED"
    node._reference_publisher = recorder
    node._accept_planner_result = lambda value: accepted.append(value) or True
    try:
        node._on_action_result(_Future(result), "request-fed9-1")
        assert [message.plan_id for message in recorder.messages] == ["plan-fed9-1"]
        assert len(accepted) == 1
        assert accepted[0].has_reference is True
        assert accepted[0].segment_id == "plan-fed9-1"
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_late_plan_result_after_deactivate_is_not_published() -> None:
    class _Completed:
        def __init__(self, result):
            self.result = result

    class _Future:
        def __init__(self, result):
            self._result = result

        def result(self):
            return _Completed(self._result)

    class _Recorder:
        def __init__(self):
            self.messages = []

        def publish(self, message):
            self.messages.append(message)

    result = PlanMotion.Result()
    result.has_reference = True
    result.reference.plan_id = "late-plan"
    result.reference.platform_type = result.reference.WHEELED
    result.reason_code = "OK"

    rclpy.init()
    node = InterfaceV1PolicyNode()
    accepted = []
    recorder = _Recorder()
    node._enabled = False
    node._platform_type = "WHEELED"
    node._reference_publisher = recorder
    node._accept_planner_result = lambda value: accepted.append(value) or True
    try:
        node._on_action_result(_Future(result), "old-request")
        assert recorder.messages == []
        assert accepted == []
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_result_rejected_by_current_coordinator_is_not_published() -> None:
    class _Completed:
        def __init__(self, result):
            self.result = result

    class _Future:
        def __init__(self, result):
            self._result = result

        def result(self):
            return _Completed(self._result)

    class _Recorder:
        def __init__(self):
            self.messages = []

        def publish(self, message):
            self.messages.append(message)

    result = PlanMotion.Result()
    result.has_reference = True
    result.reference.plan_id = "foreign-plan"
    result.reference.platform_type = result.reference.WHEELED
    result.reason_code = "OK"

    rclpy.init()
    node = InterfaceV1PolicyNode()
    recorder = _Recorder()
    node._enabled = True
    node._platform_type = "WHEELED"
    node._reference_publisher = recorder
    node._accept_planner_result = lambda value: False
    try:
        node._on_action_result(_Future(result), "foreign-request")
        assert recorder.messages == []
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_reference_publish_failure_does_not_leave_executing_context() -> None:
    class _Completed:
        def __init__(self, result):
            self.result = result

    class _Future:
        def __init__(self, result):
            self._result = result

        def result(self):
            return _Completed(self._result)

    class _FailingPublisher:
        def publish(self, message):
            del message
            raise RuntimeError("transport closed")

    result = PlanMotion.Result()
    result.has_reference = True
    result.reference.plan_id = "plan-undelivered"
    result.reference.platform_type = result.reference.WHEELED
    result.reason_code = "OK"

    rclpy.init()
    node = InterfaceV1PolicyNode()
    accepted = []
    halted = []
    node._enabled = True
    node._platform_type = "WHEELED"
    node._reference_publisher = _FailingPublisher()
    node._accept_planner_result = lambda value: accepted.append(value) or True
    node._halt_current = halted.append
    try:
        node._on_action_result(_Future(result), "current-request")
        assert len(accepted) == 1
        assert halted == ["REFERENCE_PUBLISH_FAILED"]
    finally:
        node.destroy_node()
        rclpy.shutdown()

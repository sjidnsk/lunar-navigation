from __future__ import annotations

import os
from pathlib import Path

import pytest
import rclpy
from rclpy.parameter import Parameter

from lunar_exploration_policy.node import InterfaceV1PolicyNode


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
            Parameter("repository_root", value=str(ROOT)),
        ]
    )
    node.trigger_configure()
    return node


def test_real_model_and_frozen_profile_configure_lifecycle_node() -> None:
    rclpy.init()
    node = _configured_node(PROFILE)
    try:
        assert node._coordinator is not None
        assert node._assembler is not None
        assert node._platform_type == "WHEELED"
        assert len(node._subscriptions) == 7
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


"""Behavioral checks for externally owned ROS interface declarations."""

from __future__ import annotations

import subprocess
from pathlib import Path

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONFIG = REPOSITORY_ROOT / "ros2_ws/src/lunar_navigation_config/config/external_interfaces.yaml"


def test_external_interfaces_declare_owner_and_required_fields():
    """Changing the external localization contract must break this test."""
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))

    localization = config["topics"]["localization_status"]
    assert localization["owner"] == "external"
    assert localization["type"] == "lunar_navigation_msgs/msg/LocalizationStatus"
    assert localization["required_fields"] == ["header", "status"]


def test_parse_top_level_fields_ignores_nested_members_and_constants():
    """Treating a nested member or constant as a required top-level field is a bug."""
    from tools.check_external_interfaces import parse_top_level_fields

    interface = """std_msgs/Header header
uint8 UNKNOWN=0
uint8 status
geometry_msgs/Pose pose
  float64 x
"""

    assert parse_top_level_fields(interface) == {"header", "status", "pose"}


def test_check_interfaces_reports_missing_top_level_required_field(tmp_path):
    """Dropping a required interface field must produce a clear validation error."""
    from tools.check_external_interfaces import check_interfaces

    config = tmp_path / "interfaces.yaml"
    config.write_text(
        """topics:
  localization_status:
    type: lunar_navigation_msgs/msg/LocalizationStatus
    required_fields: [header, status]
""",
        encoding="utf-8",
    )

    def run(command: list[str]) -> subprocess.CompletedProcess[str]:
        if command[:3] == ["ros2", "pkg", "prefix"]:
            return subprocess.CompletedProcess(command, 0, stdout="/opt/ros/humble\n", stderr="")
        return subprocess.CompletedProcess(command, 0, stdout="std_msgs/Header header\nuint8 state\n", stderr="")

    errors = check_interfaces(config, run=run)

    assert errors == [
        "lunar_navigation_msgs/msg/LocalizationStatus: missing top-level fields: status"
    ]


def test_check_interfaces_reports_ros_command_failure(tmp_path):
    """A failed ROS interface lookup must not be mistaken for a visible interface."""
    from tools.check_external_interfaces import check_interfaces

    config = tmp_path / "interfaces.yaml"
    config.write_text(
        """topics:
  odometry:
    type: nav_msgs/msg/Odometry
    required_fields: [header]
""",
        encoding="utf-8",
    )

    def run(command: list[str]) -> subprocess.CompletedProcess[str]:
        if command[:3] == ["ros2", "pkg", "prefix"]:
            return subprocess.CompletedProcess(command, 0, stdout="/opt/ros/humble\n", stderr="")
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="unknown interface")

    errors = check_interfaces(config, run=run)

    assert errors == ["nav_msgs/msg/Odometry: ros2 interface show failed: unknown interface"]


def test_check_interfaces_records_successful_package_prefix(tmp_path):
    """Discarding a resolved package location would make an audit trail incomplete."""
    from tools.check_external_interfaces import check_interfaces

    config = tmp_path / "interfaces.yaml"
    config.write_text(
        """topics:
  odometry:
    type: nav_msgs/msg/Odometry
    required_fields: [header]
""",
        encoding="utf-8",
    )
    package_locations: dict[str, str] = {}

    def run(command: list[str]) -> subprocess.CompletedProcess[str]:
        if command[:3] == ["ros2", "pkg", "prefix"]:
            return subprocess.CompletedProcess(command, 0, stdout="/opt/ros/humble\n", stderr="")
        return subprocess.CompletedProcess(command, 0, stdout="std_msgs/Header header\n", stderr="")

    errors = check_interfaces(config, run=run, package_locations=package_locations)

    assert errors == []
    assert package_locations == {"nav_msgs": "/opt/ros/humble"}

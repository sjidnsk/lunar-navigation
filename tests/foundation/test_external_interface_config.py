"""Behavioral checks for externally owned ROS interface declarations."""

from __future__ import annotations

import copy
import subprocess
import xml.etree.ElementTree as element_tree
from pathlib import Path

import pytest
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONFIG = REPOSITORY_ROOT / "ros2_ws/src/lunar_navigation_config/config/external_interfaces.yaml"
PACKAGE_XML = REPOSITORY_ROOT / "ros2_ws/src/lunar_navigation_config/package.xml"

EXPECTED_DOCUMENT = {
    "schema_version": "lunar-external-interfaces/v1",
    "topics": {
        "map_global": {
            "name": "/environment/map_global",
            "type": "grid_map_msgs/msg/GridMap",
            "owner": "external",
            "frame": "map",
            "required_fields": [
                "header",
                "info",
                "layers",
                "basic_layers",
                "data",
                "outer_start_index",
                "inner_start_index",
            ],
        },
        "map_local": {
            "name": "/environment/map_local",
            "type": "grid_map_msgs/msg/GridMap",
            "owner": "external",
            "frame": "odom",
            "required_fields": [
                "header",
                "info",
                "layers",
                "basic_layers",
                "data",
                "outer_start_index",
                "inner_start_index",
            ],
        },
        "odometry": {
            "name": "/localization/odometry",
            "type": "nav_msgs/msg/Odometry",
            "owner": "external",
            "frame": "odom",
            "child_frame": "base_link",
            "required_fields": ["header", "child_frame_id", "pose", "twist"],
        },
        "localization_status": {
            "name": "/localization/status",
            "type": "lunar_navigation_msgs/msg/LocalizationStatus",
            "owner": "external",
            "frame": "odom",
            "required_fields": ["header", "status"],
        },
        "exploration_task": {
            "name": "/mission/exploration_task",
            "type": "lunar_navigation_msgs/msg/ExplorationTask",
            "owner": "external",
            "frame": "map",
            "required_fields": [
                "header",
                "mission_id",
                "revision",
                "desired_state",
                "science_regions",
            ],
        },
    },
    "tf": {
        "topic": "/tf",
        "type": "tf2_msgs/msg/TFMessage",
        "chain": ["map", "odom", "base_link"],
    },
    "required_grid_layers": [
        "elevation",
        "valid_mask",
        "obstacle",
        "obstacle_height",
        "observation_age_s",
        "observation_quality",
        "elevation_variance",
        "obstacle_variance",
        "observation_count",
        "forbidden",
    ],
    "static_inputs": {
        "observation_capability": {
            "owner": "external",
            "formats": ["yaml", "json"],
            "required_fields": ["sensor_range_m", "sensor_fov_deg"],
        },
        "platform_capability": {
            "owner": "external",
            "schema": "platform-control-capability-source/v1",
            "formats": ["yaml", "json", "urdf"],
            "required_fields": ["platform", "geometry_source"],
        },
    },
}


def complete_valid_config() -> dict[str, object]:
    """Return a hand-maintained complete contract fixture for checker tests."""
    return copy.deepcopy(EXPECTED_DOCUMENT)


def write_config(path: Path, document: dict[str, object]) -> Path:
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


def successful_ros_runner(required_fields_by_type: dict[str, list[str]], calls: list[list[str]]):
    """Provide complete CLI-shaped results without requiring a ROS installation."""

    def run(command: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        if command[:3] == ["ros2", "pkg", "prefix"]:
            return subprocess.CompletedProcess(command, 0, stdout="/opt/ros/humble\n", stderr="")
        fields = required_fields_by_type.get(command[-1], [])
        output = "\n".join(f"string {field}" for field in fields) + "\n"
        return subprocess.CompletedProcess(command, 0, stdout=output, stderr="")

    return run


def test_authoritative_config_matches_complete_external_contract():
    """Changing any declared external topic, frame, field, layer, or static input is a bug."""
    assert yaml.safe_load(CONFIG.read_text(encoding="utf-8")) == EXPECTED_DOCUMENT


def test_config_package_declares_exact_external_runtime_dependencies():
    """Removing or adding an external runtime dependency changes the package contract."""
    package = element_tree.parse(PACKAGE_XML).getroot()

    assert {node.text for node in package.findall("exec_depend")} == {
        "grid_map_msgs",
        "nav_msgs",
        "tf2_msgs",
        "lunar_navigation_msgs",
    }


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


@pytest.mark.parametrize(
    ("section", "expected_error"),
    [
        ("topics", "config error: missing required key: topics"),
        ("tf", "config error: missing required key: tf"),
        ("required_grid_layers", "config error: missing required key: required_grid_layers"),
        ("static_inputs", "config error: missing required key: static_inputs"),
    ],
)
def test_check_interfaces_rejects_missing_required_contract_sections_without_ros(
    tmp_path, section, expected_error
):
    """Omitting a contract section must fail before any ROS CLI lookup runs."""
    from tools.check_external_interfaces import check_interfaces

    document = complete_valid_config()
    document.pop(section)
    calls: list[list[str]] = []

    errors = check_interfaces(
        write_config(tmp_path / "interfaces.yaml", document),
        run=successful_ros_runner({}, calls),
    )

    assert expected_error in errors
    assert calls == []


@pytest.mark.parametrize(
    ("path", "value", "expected_error"),
    [
        (("schema_version",), "wrong/v1", "config error: schema_version must be 'lunar-external-interfaces/v1'"),
        (("topics", "map_global", "owner"), "internal", "config error: topics.map_global.owner must be 'external'"),
        (("topics", "map_global", "type"), "nav_msgs/msg/Path", "config error: topics.map_global.type must be 'grid_map_msgs/msg/GridMap'"),
        (("topics", "map_global", "frame"), "odom", "config error: topics.map_global.frame must be 'map'"),
        (("tf", "chain"), ["odom", "map", "base_link"], "config error: tf.chain must be ['map', 'odom', 'base_link']"),
        (("static_inputs", "platform_capability", "schema"), "wrong/v1", "config error: static_inputs.platform_capability.schema must be 'platform-control-capability-source/v1'"),
    ],
)
def test_check_interfaces_rejects_wrong_fixed_contract_values_without_ros(
    tmp_path, path, value, expected_error
):
    """Changing an owned contract value must be a configuration error, not a ROS result."""
    from tools.check_external_interfaces import check_interfaces

    document = complete_valid_config()
    target: object = document
    for key in path[:-1]:
        target = target[key]  # type: ignore[index]
    target[path[-1]] = value  # type: ignore[index]
    calls: list[list[str]] = []

    errors = check_interfaces(
        write_config(tmp_path / "interfaces.yaml", document),
        run=successful_ros_runner({}, calls),
    )

    assert expected_error in errors
    assert calls == []


@pytest.mark.parametrize(
    ("required_fields", "expected_error"),
    [
        ([], "config error: topics.map_global.required_fields must not be empty"),
        (["header", "header"], "config error: topics.map_global.required_fields must not contain duplicates"),
    ],
)
def test_check_interfaces_rejects_empty_or_duplicate_required_fields_without_ros(
    tmp_path, required_fields, expected_error
):
    """Required field lists must be populated and unambiguous before ROS lookups run."""
    from tools.check_external_interfaces import check_interfaces

    document = complete_valid_config()
    document["topics"]["map_global"]["required_fields"] = required_fields
    calls: list[list[str]] = []

    errors = check_interfaces(
        write_config(tmp_path / "interfaces.yaml", document),
        run=successful_ros_runner({}, calls),
    )

    assert expected_error in errors
    assert calls == []


def test_check_interfaces_reports_missing_top_level_required_field(tmp_path):
    """Dropping a required interface field must produce a clear validation error."""
    from tools.check_external_interfaces import check_interfaces

    document = complete_valid_config()
    fields_by_type = {
        topic["type"]: list(topic["required_fields"])
        for topic in document["topics"].values()
    }
    fields_by_type["tf2_msgs/msg/TFMessage"] = []
    fields_by_type["lunar_navigation_msgs/msg/LocalizationStatus"] = ["header"]
    calls: list[list[str]] = []

    errors = check_interfaces(
        write_config(tmp_path / "interfaces.yaml", document),
        run=successful_ros_runner(fields_by_type, calls),
    )

    assert errors == [
        "lunar_navigation_msgs/msg/LocalizationStatus: missing top-level fields: status"
    ]
    assert calls


def test_check_interfaces_reports_ros_command_failure(tmp_path):
    """A failed ROS interface lookup must not be mistaken for a visible interface."""
    from tools.check_external_interfaces import check_interfaces

    document = complete_valid_config()

    def run(command: list[str]) -> subprocess.CompletedProcess[str]:
        if command[:3] == ["ros2", "pkg", "prefix"]:
            return subprocess.CompletedProcess(command, 0, stdout="/opt/ros/humble\n", stderr="")
        if command[-1] == "nav_msgs/msg/Odometry":
            return subprocess.CompletedProcess(command, 1, stdout="", stderr="unknown interface")
        fields_by_type = {
            topic["type"]: list(topic["required_fields"])
            for topic in document["topics"].values()
        }
        fields_by_type["tf2_msgs/msg/TFMessage"] = []
        output = "\n".join(f"string {field}" for field in fields_by_type[command[-1]]) + "\n"
        return subprocess.CompletedProcess(command, 0, stdout=output, stderr="")

    errors = check_interfaces(write_config(tmp_path / "interfaces.yaml", document), run=run)

    assert errors == ["nav_msgs/msg/Odometry: ros2 interface show failed: unknown interface"]


def test_check_interfaces_records_successful_package_prefix(tmp_path):
    """Discarding a resolved package location would make an audit trail incomplete."""
    from tools.check_external_interfaces import check_interfaces

    document = complete_valid_config()
    fields_by_type = {
        topic["type"]: list(topic["required_fields"])
        for topic in document["topics"].values()
    }
    fields_by_type["tf2_msgs/msg/TFMessage"] = []
    calls: list[list[str]] = []
    package_locations: dict[str, str] = {}

    errors = check_interfaces(
        write_config(tmp_path / "interfaces.yaml", document),
        run=successful_ros_runner(fields_by_type, calls),
        package_locations=package_locations,
    )

    assert errors == []
    assert package_locations == {
        "grid_map_msgs": "/opt/ros/humble",
        "nav_msgs": "/opt/ros/humble",
        "tf2_msgs": "/opt/ros/humble",
        "lunar_navigation_msgs": "/opt/ros/humble",
    }

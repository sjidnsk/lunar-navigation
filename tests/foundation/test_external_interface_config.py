"""Behavioral checks for externally owned ROS interface declarations."""

from __future__ import annotations

import copy
import subprocess
import xml.etree.ElementTree as element_tree
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONFIG = REPOSITORY_ROOT / "ros2_ws/src/lunar_navigation_config/config/external_interfaces.yaml"
PACKAGE_XML = REPOSITORY_ROOT / "ros2_ws/src/lunar_navigation_config/package.xml"

EXPECTED_DOCUMENT = {
    "schema_version": "lunar-external-interfaces/v5",
    "interface_packages": {
        "lunar_navigation_msgs": {
            "schema_provider": "in_repository_provisional",
            "source_path": "ros2_ws/src/lunar_navigation_msgs",
            "upstream_status": "undefined",
            "replacement_policy": "atomic",
        }
    },
    "topics": {
        "map_global": {
            "name": "/environment/map_global",
            "type": "grid_map_msgs/msg/GridMap",
            "owner": "external",
            "frame": "map",
            "level_semantics": "selected_dyadic_global",
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
            "level_semantics": "l0_platform_window",
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
                "roi_min_x_m",
                "roi_min_y_m",
                "roi_max_x_m",
                "roi_max_y_m",
                "science_regions",
            ],
        },
        "motion_execution_feedback": {
            "name": "/execution/motion_feedback",
            "type": "lunar_navigation_msgs/msg/MotionExecutionFeedback",
            "owner": "external",
            "frame": "platform_base_frame",
            "qos": {
                "reliability": "reliable",
                "durability": "volatile",
                "depth": 10,
            },
            "required_fields": [
                "header",
                "sequence",
                "platform_type",
                "plan_id",
                "segment_id",
                "state",
                "reason_code",
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
    "map_pyramid": {
        "base_resolution_m": 0.2,
        "resolution_scale_per_level": 2,
        "maximum_level": 4,
        "maximum_cells": 1_048_576,
        "maximum_axis_cells": 4_096,
        "global_selection": "smallest_admissible_level",
        "local_level": 0,
        "aggregation_version": "lunar-conservative-grid-aggregation/v1",
    },
    "static_inputs": {
        "observation_capability": {
            "owner": "external",
            "formats": ["yaml", "json"],
            "required_fields": ["sensor_range_m", "sensor_fov_deg"],
        },
        "platform_capability": {
            "owner": "external",
            "schema": "platform-control-capability-source/v2",
            "formats": ["yaml", "json", "urdf"],
            "required_fields": ["platform", "geometry_source", "sources"],
        },
    },
}

VALID_LOCALIZATION_STATUS = """uint8 UNKNOWN=0
uint8 VALID=1
uint8 DEGRADED=2
uint8 INVALID=3
uint8 RELOCALIZING=4
std_msgs/Header header
uint8 status
"""
VALID_SCIENCE_TARGET_REGION = """string region_id
string objective_id
geometry_msgs/Polygon boundary
float64 priority
"""
VALID_EXPLORATION_TASK = """uint8 ACTIVE=1
uint8 PAUSED=2
uint8 CANCELED=3
std_msgs/Header header
string mission_id
uint64 revision
uint8 desired_state
float64 roi_min_x_m
float64 roi_min_y_m
float64 roi_max_x_m
float64 roi_max_y_m
lunar_navigation_msgs/ScienceTargetRegion[<=64] science_regions
"""
VALID_MOTION_EXECUTION_FEEDBACK = """uint8 WHEELED=1
uint8 LEGGED=2
uint8 HOPPER=3
uint8 IDLE=0
uint8 ACCEPTED=1
uint8 EXECUTING=2
uint8 SEGMENT_COMPLETE=3
uint8 LANDED_HOLD=4
uint8 FAILED=5
uint8 CANCELED=6
std_msgs/Header header
uint64 sequence
uint8 platform_type
string plan_id
string segment_id
uint8 state
string reason_code
"""
VALID_HOPPER_PROPELLANT_STATE = """std_msgs/Header header
string platform_id
string capability_version
float64 total_mass_kg
float64 remaining_usable_fuel_mass_kg
"""


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


def field_definition(*names: str) -> str:
    return "".join(f"string {name}\n" for name in names)


@dataclass
class SourceAwareRosRunner:
    lunar_prefix: Path
    interface_outputs: dict[str, str]

    def __call__(self, command: list[str]) -> subprocess.CompletedProcess[str]:
        if command[:3] == ["ros2", "pkg", "prefix"]:
            package = command[3]
            prefix = (
                self.lunar_prefix
                if package == "lunar_navigation_msgs"
                else Path("/opt/ros/humble")
            )
            return subprocess.CompletedProcess(command, 0, stdout=f"{prefix}\n", stderr="")
        if command[:3] == ["ros2", "interface", "show"]:
            output = self.interface_outputs[command[3]]
            return subprocess.CompletedProcess(command, 0, stdout=output, stderr="")
        raise AssertionError(f"unexpected command: {command!r}")


def source_aware_ros_runner(lunar_prefix: Path) -> SourceAwareRosRunner:
    document = complete_valid_config()
    outputs = {
        topic["type"]: field_definition(*topic["required_fields"])
        for topic in document["topics"].values()
    }
    outputs["tf2_msgs/msg/TFMessage"] = field_definition("transforms")
    outputs.update(
        {
            "lunar_navigation_msgs/msg/LocalizationStatus": VALID_LOCALIZATION_STATUS,
            "lunar_navigation_msgs/msg/ScienceTargetRegion": VALID_SCIENCE_TARGET_REGION,
            "lunar_navigation_msgs/msg/ExplorationTask": VALID_EXPLORATION_TASK,
            "lunar_navigation_msgs/msg/MotionExecutionFeedback": (
                VALID_MOTION_EXECUTION_FEEDBACK
            ),
            "lunar_navigation_msgs/msg/HopperPropellantState": (
                VALID_HOPPER_PROPELLANT_STATE
            ),
        }
    )
    return SourceAwareRosRunner(lunar_prefix.resolve(), outputs)


def make_package_provider(prefix: Path, package: str) -> Path:
    resolved = prefix.resolve()
    index = resolved / "share/ament_index/resource_index/packages"
    index.mkdir(parents=True)
    (index / package).write_text("", encoding="utf-8")
    return resolved


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


def test_parse_top_level_declarations_preserves_constants_fields_and_bounds():
    """Losing constants or bounded sequence syntax would hide provisional schema drift."""
    from tools.check_external_interfaces import parse_top_level_declarations

    interface = """uint8 ACTIVE=1 # state
std_msgs/Header header
lunar_navigation_msgs/ScienceTargetRegion[<=64] science_regions
  string nested
# top-level comment
"""

    assert parse_top_level_declarations(interface) == (
        "uint8 ACTIVE=1",
        "std_msgs/Header header",
        "lunar_navigation_msgs/ScienceTargetRegion[<=64] science_regions",
    )


@pytest.mark.parametrize(
    ("section", "expected_error"),
    [
        ("topics", "config error: missing required key: topics"),
        ("interface_packages", "config error: missing required key: interface_packages"),
        ("tf", "config error: missing required key: tf"),
        ("required_grid_layers", "config error: missing required key: required_grid_layers"),
        ("map_pyramid", "config error: missing required key: map_pyramid"),
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
        expected_lunar_navigation_prefix=tmp_path / "expected",
        run=successful_ros_runner({}, calls),
    )

    assert expected_error in errors
    assert calls == []


@pytest.mark.parametrize(
    ("path", "value", "expected_error"),
    [
        (("schema_version",), "wrong/v1", "config error: schema_version must be 'lunar-external-interfaces/v5'"),
        (("topics", "map_global", "owner"), "internal", "config error: topics.map_global.owner must be 'external'"),
        (("topics", "map_global", "type"), "nav_msgs/msg/Path", "config error: topics.map_global.type must be 'grid_map_msgs/msg/GridMap'"),
        (("topics", "map_global", "frame"), "odom", "config error: topics.map_global.frame must be 'map'"),
        (("topics", "map_global", "level_semantics"), "l0_platform_window", "config error: topics.map_global.level_semantics must be 'selected_dyadic_global'"),
        (("topics", "map_local", "level_semantics"), "selected_dyadic_global", "config error: topics.map_local.level_semantics must be 'l0_platform_window'"),
        (("map_pyramid", "base_resolution_m"), 0.25, "config error: map_pyramid.base_resolution_m must be 0.2"),
        (("map_pyramid", "resolution_scale_per_level"), 3, "config error: map_pyramid.resolution_scale_per_level must be 2"),
        (("map_pyramid", "maximum_level"), 5, "config error: map_pyramid.maximum_level must be 4"),
        (("map_pyramid", "maximum_cells"), 1_048_577, "config error: map_pyramid.maximum_cells must be 1048576"),
        (("map_pyramid", "maximum_axis_cells"), 4_097, "config error: map_pyramid.maximum_axis_cells must be 4096"),
        (("map_pyramid", "global_selection"), "coarsest_admissible_level", "config error: map_pyramid.global_selection must be 'smallest_admissible_level'"),
        (("map_pyramid", "local_level"), 1, "config error: map_pyramid.local_level must be 0"),
        (("map_pyramid", "aggregation_version"), "unsafe-average/v1", "config error: map_pyramid.aggregation_version must be 'lunar-conservative-grid-aggregation/v1'"),
        (("tf", "chain"), ["odom", "map", "base_link"], "config error: tf.chain must be ['map', 'odom', 'base_link']"),
        (("static_inputs", "platform_capability", "schema"), "wrong/v1", "config error: static_inputs.platform_capability.schema must be 'platform-control-capability-source/v2'"),
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
        expected_lunar_navigation_prefix=tmp_path / "expected",
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
        expected_lunar_navigation_prefix=tmp_path / "expected",
        run=successful_ros_runner({}, calls),
    )

    assert expected_error in errors
    assert calls == []


def test_check_interfaces_reports_missing_top_level_required_field(tmp_path):
    """Dropping a required interface field must produce a clear validation error."""
    from tools.check_external_interfaces import check_interfaces

    document = complete_valid_config()
    expected = make_package_provider(tmp_path / "expected", "lunar_navigation_msgs")
    run = source_aware_ros_runner(expected)
    run.interface_outputs["lunar_navigation_msgs/msg/LocalizationStatus"] = (
        VALID_LOCALIZATION_STATUS.replace("uint8 status\n", "")
    )

    errors = check_interfaces(
        write_config(tmp_path / "interfaces.yaml", document),
        expected_lunar_navigation_prefix=expected,
        run=run,
        ament_prefix_path=f"{expected}:/opt/ros/humble",
    )

    assert errors == [
        "lunar_navigation_msgs/msg/LocalizationStatus: declaration mismatch: "
        "expected 'uint8 status' at declaration 7, got end of declarations"
    ]


def test_check_interfaces_reports_ros_command_failure(tmp_path):
    """A failed ROS interface lookup must not be mistaken for a visible interface."""
    from tools.check_external_interfaces import check_interfaces

    document = complete_valid_config()

    expected = make_package_provider(tmp_path / "expected", "lunar_navigation_msgs")
    successful_run = source_aware_ros_runner(expected)

    def run(command: list[str]) -> subprocess.CompletedProcess[str]:
        if command[-1] == "nav_msgs/msg/Odometry":
            return subprocess.CompletedProcess(command, 1, stdout="", stderr="unknown interface")
        return successful_run(command)

    errors = check_interfaces(
        write_config(tmp_path / "interfaces.yaml", document),
        expected_lunar_navigation_prefix=expected,
        run=run,
        ament_prefix_path=f"{expected}:/opt/ros/humble",
    )

    assert errors == ["nav_msgs/msg/Odometry: ros2 interface show failed: unknown interface"]


def test_check_interfaces_records_successful_package_prefix(tmp_path):
    """Discarding a resolved package location would make an audit trail incomplete."""
    from tools.check_external_interfaces import check_interfaces

    document = complete_valid_config()
    expected = make_package_provider(tmp_path / "expected", "lunar_navigation_msgs")
    package_locations: dict[str, str] = {}

    errors = check_interfaces(
        write_config(tmp_path / "interfaces.yaml", document),
        expected_lunar_navigation_prefix=expected,
        run=source_aware_ros_runner(expected),
        ament_prefix_path=f"{expected}:/opt/ros/humble",
        package_locations=package_locations,
    )

    assert errors == []
    assert package_locations == {
        "grid_map_msgs": "/opt/ros/humble",
        "nav_msgs": "/opt/ros/humble",
        "tf2_msgs": "/opt/ros/humble",
        "lunar_navigation_msgs": str(expected),
    }


def test_rejects_provisional_package_from_wrong_prefix(tmp_path):
    from tools.check_external_interfaces import check_interfaces

    other = make_package_provider(tmp_path / "other", "lunar_navigation_msgs")
    errors = check_interfaces(
        write_config(tmp_path / "interfaces.yaml", complete_valid_config()),
        expected_lunar_navigation_prefix=tmp_path / "expected",
        run=source_aware_ros_runner(other),
        ament_prefix_path=str(other),
    )
    assert any("unexpected package prefix" in error for error in errors)


def test_rejects_duplicate_ament_provider(tmp_path):
    from tools.check_external_interfaces import check_interfaces

    expected = make_package_provider(tmp_path / "expected", "lunar_navigation_msgs")
    duplicate = make_package_provider(tmp_path / "duplicate", "lunar_navigation_msgs")
    errors = check_interfaces(
        write_config(tmp_path / "interfaces.yaml", complete_valid_config()),
        expected_lunar_navigation_prefix=expected,
        run=source_aware_ros_runner(expected),
        ament_prefix_path=f"{expected}:{duplicate}:/opt/ros/humble",
    )
    assert any("multiple ament providers" in error for error in errors)


def test_rejects_provisional_declaration_drift(tmp_path):
    from tools.check_external_interfaces import check_interfaces

    expected = make_package_provider(tmp_path / "expected", "lunar_navigation_msgs")
    run = source_aware_ros_runner(lunar_prefix=expected)
    run.interface_outputs["lunar_navigation_msgs/msg/ExplorationTask"] = (
        VALID_EXPLORATION_TASK.replace("[<=64]", "[]")
    )
    errors = check_interfaces(
        write_config(tmp_path / "interfaces.yaml", complete_valid_config()),
        expected_lunar_navigation_prefix=expected,
        run=run,
        ament_prefix_path=f"{expected}:/opt/ros/humble",
    )
    assert any("declaration mismatch" in error for error in errors)


def test_rejects_motion_execution_feedback_declaration_drift(tmp_path):
    from tools.check_external_interfaces import check_interfaces

    expected = make_package_provider(tmp_path / "expected", "lunar_navigation_msgs")
    run = source_aware_ros_runner(lunar_prefix=expected)
    run.interface_outputs[
        "lunar_navigation_msgs/msg/MotionExecutionFeedback"
    ] = VALID_MOTION_EXECUTION_FEEDBACK.replace(
        "uint64 sequence\n", "uint32 sequence\n"
    )
    errors = check_interfaces(
        write_config(tmp_path / "interfaces.yaml", complete_valid_config()),
        expected_lunar_navigation_prefix=expected,
        run=run,
        ament_prefix_path=f"{expected}:/opt/ros/humble",
    )
    assert any(
        "MotionExecutionFeedback" in error and "declaration mismatch" in error
        for error in errors
    )


def test_rejects_hopper_propellant_state_declaration_drift(tmp_path):
    """Changing hopper mass precision would silently change reachability."""
    from tools.check_external_interfaces import check_interfaces

    expected = make_package_provider(tmp_path / "expected", "lunar_navigation_msgs")
    run = source_aware_ros_runner(lunar_prefix=expected)
    run.interface_outputs[
        "lunar_navigation_msgs/msg/HopperPropellantState"
    ] = VALID_HOPPER_PROPELLANT_STATE.replace(
        "float64 total_mass_kg\n", "float32 total_mass_kg\n"
    )
    errors = check_interfaces(
        write_config(tmp_path / "interfaces.yaml", complete_valid_config()),
        expected_lunar_navigation_prefix=expected,
        run=run,
        ament_prefix_path=f"{expected}:/opt/ros/humble",
    )

    assert any(
        "HopperPropellantState" in error and "declaration mismatch" in error
        for error in errors
    )

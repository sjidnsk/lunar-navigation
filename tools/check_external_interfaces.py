"""Check that externally owned ROS interfaces are installed and compatible."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path

import yaml


CommandRunner = Callable[[list[str]], subprocess.CompletedProcess[str]]
_FIELD_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_SCHEMA_VERSION = "lunar-external-interfaces/v5"
_INTERFACE_PACKAGES = {
    "lunar_navigation_msgs": {
        "schema_provider": "in_repository_provisional",
        "source_path": "ros2_ws/src/lunar_navigation_msgs",
        "upstream_status": "undefined",
        "replacement_policy": "atomic",
    }
}
LOCALIZATION_STATUS_DECLARATIONS = (
    "uint8 UNKNOWN=0",
    "uint8 VALID=1",
    "uint8 DEGRADED=2",
    "uint8 INVALID=3",
    "uint8 RELOCALIZING=4",
    "std_msgs/Header header",
    "uint8 status",
)
SCIENCE_TARGET_REGION_DECLARATIONS = (
    "string region_id",
    "string objective_id",
    "geometry_msgs/Polygon boundary",
    "float64 priority",
)
EXPLORATION_TASK_DECLARATIONS = (
    "uint8 ACTIVE=1",
    "uint8 PAUSED=2",
    "uint8 CANCELED=3",
    "std_msgs/Header header",
    "string mission_id",
    "uint64 revision",
    "uint8 desired_state",
    "float64 roi_min_x_m",
    "float64 roi_min_y_m",
    "float64 roi_max_x_m",
    "float64 roi_max_y_m",
    "lunar_navigation_msgs/ScienceTargetRegion[<=64] science_regions",
)
MOTION_EXECUTION_FEEDBACK_DECLARATIONS = (
    "uint8 WHEELED=1",
    "uint8 LEGGED=2",
    "uint8 HOPPER=3",
    "uint8 IDLE=0",
    "uint8 ACCEPTED=1",
    "uint8 EXECUTING=2",
    "uint8 SEGMENT_COMPLETE=3",
    "uint8 LANDED_HOLD=4",
    "uint8 FAILED=5",
    "uint8 CANCELED=6",
    "std_msgs/Header header",
    "uint64 sequence",
    "uint8 platform_type",
    "string plan_id",
    "string segment_id",
    "uint8 state",
    "string reason_code",
)
HOPPER_PROPELLANT_STATE_DECLARATIONS = (
    "std_msgs/Header header",
    "string platform_id",
    "string capability_version",
    "float64 total_mass_kg",
    "float64 remaining_usable_fuel_mass_kg",
)
_PROVISIONAL_DECLARATIONS = {
    "lunar_navigation_msgs/msg/LocalizationStatus": LOCALIZATION_STATUS_DECLARATIONS,
    "lunar_navigation_msgs/msg/ScienceTargetRegion": SCIENCE_TARGET_REGION_DECLARATIONS,
    "lunar_navigation_msgs/msg/ExplorationTask": EXPLORATION_TASK_DECLARATIONS,
    "lunar_navigation_msgs/msg/MotionExecutionFeedback": (
        MOTION_EXECUTION_FEEDBACK_DECLARATIONS
    ),
    "lunar_navigation_msgs/msg/HopperPropellantState": (
        HOPPER_PROPELLANT_STATE_DECLARATIONS
    ),
}
_SYSTEM_PREFIX = Path("/opt/ros/humble").resolve()
_TOPICS = {
    "map_global": {
        "name": "/environment/map_global",
        "type": "grid_map_msgs/msg/GridMap",
        "owner": "external",
        "frame": "map",
        "level_semantics": "selected_configured_global",
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
}
_TF = {
    "topic": "/tf",
    "type": "tf2_msgs/msg/TFMessage",
    "chain": ["map", "odom", "base_link"],
}
_REQUIRED_GRID_LAYERS = [
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
]
_MAP_PYRAMID = {
    "base_resolution_m": 0.2,
    "resolution_scale_factors": [1, 2, 4, 8, 16, 20],
    "maximum_level": 5,
    "maximum_cells": 1_048_576,
    "maximum_axis_cells": 4_096,
    "target_axis_cells": 256,
    "global_selection": "smallest_admissible_level",
    "local_level": 0,
    "aggregation_version": "lunar-conservative-grid-aggregation/v1",
}
_STATIC_INPUTS = {
    "platform_profile": {
        "owner": "external",
        "schema": "lunar-platform-profile/v1",
        "formats": ["yaml"],
        "required_sections": ["platform", "observation", "assets", "sources"],
        "runtime_path": "/etc/lunar_navigation/platform_profile.yaml",
    },
}


def run_ros_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    """Run one ROS CLI command without raising when ROS is unavailable."""
    try:
        return subprocess.run(command, check=False, capture_output=True, text=True)
    except OSError as error:
        return subprocess.CompletedProcess(command, 127, stdout="", stderr=str(error))


def parse_top_level_fields(interface_definition: str) -> set[str]:
    """Extract declared top-level ROS message field names from CLI output."""
    fields: set[str] = set()
    for declaration in parse_top_level_declarations(interface_definition):
        tokens = declaration.split()
        name = tokens[1]
        if "=" not in name and _FIELD_NAME.fullmatch(name):
            fields.add(name)
    return fields


def parse_top_level_declarations(interface_definition: str) -> tuple[str, ...]:
    """Extract normalized, unindented ROS message constants and fields in order."""
    declarations: list[str] = []
    for raw_line in interface_definition.splitlines():
        if not raw_line or raw_line[0].isspace():
            continue
        declaration = raw_line.split("#", maxsplit=1)[0].strip()
        tokens = declaration.split()
        if len(tokens) != 2:
            continue
        name = tokens[1].split("=", maxsplit=1)[0]
        if _FIELD_NAME.fullmatch(name):
            declarations.append(declaration)
    return tuple(declarations)


def _command_failure(command: str, result: subprocess.CompletedProcess[str]) -> str:
    detail = result.stderr.strip() or result.stdout.strip() or f"exit code {result.returncode}"
    return f"{command} failed: {detail}"


def _missing_key(errors: list[str], path: str) -> None:
    errors.append(f"config error: missing required key: {path}")


def _unexpected_keys(errors: list[str], path: str, value: Mapping[str, object], allowed: set[str]) -> None:
    unexpected = sorted(set(value) - allowed)
    if unexpected:
        errors.append(f"config error: {path} has unexpected keys: {', '.join(unexpected)}")


def _validate_required_fields(
    errors: list[str], path: str, value: object, expected: list[str]
) -> None:
    if not isinstance(value, list):
        errors.append(f"config error: {path} must be a list")
        return
    if not all(isinstance(field, str) and field for field in value):
        errors.append(f"config error: {path} must contain non-empty strings")
        return
    if not value:
        errors.append(f"config error: {path} must not be empty")
        return
    if len(set(value)) != len(value):
        errors.append(f"config error: {path} must not contain duplicates")
        return
    if value != expected:
        errors.append(f"config error: {path} must be {expected!r}")


def _validate_fixed_mapping(
    errors: list[str], path: str, value: object, expected: Mapping[str, object]
) -> None:
    if not isinstance(value, Mapping):
        errors.append(f"config error: {path} must be a mapping")
        return

    _unexpected_keys(errors, path, value, set(expected))
    for key, expected_value in expected.items():
        field_path = f"{path}.{key}"
        if key not in value:
            _missing_key(errors, field_path)
            continue
        actual_value = value[key]
        if key == "required_fields":
            _validate_required_fields(errors, field_path, actual_value, expected_value)
        elif isinstance(expected_value, list):
            if not isinstance(actual_value, list):
                errors.append(f"config error: {field_path} must be a list")
            elif actual_value != expected_value:
                errors.append(f"config error: {field_path} must be {expected_value!r}")
        elif not isinstance(actual_value, type(expected_value)):
            errors.append(
                f"config error: {field_path} must be a {type(expected_value).__name__}"
            )
        elif actual_value != expected_value:
            errors.append(f"config error: {field_path} must be {expected_value!r}")


def validate_external_config(document: object) -> list[str]:
    """Return contract errors without invoking ROS commands."""
    if not isinstance(document, Mapping):
        return ["config error: document must be a mapping"]

    errors: list[str] = []
    required_top_level = {
        "schema_version",
        "interface_packages",
        "topics",
        "tf",
        "required_grid_layers",
        "map_pyramid",
        "static_inputs",
    }
    _unexpected_keys(errors, "document", document, required_top_level)
    for key in sorted(required_top_level):
        if key not in document:
            _missing_key(errors, key)

    schema_version = document.get("schema_version")
    if schema_version is not None:
        if not isinstance(schema_version, str):
            errors.append("config error: schema_version must be a string")
        elif schema_version != _SCHEMA_VERSION:
            errors.append(f"config error: schema_version must be {_SCHEMA_VERSION!r}")

    interface_packages = document.get("interface_packages")
    if interface_packages is not None:
        if not isinstance(interface_packages, Mapping):
            errors.append("config error: interface_packages must be a mapping")
        else:
            _unexpected_keys(
                errors, "interface_packages", interface_packages, set(_INTERFACE_PACKAGES)
            )
            for name, expected_package in _INTERFACE_PACKAGES.items():
                if name not in interface_packages:
                    _missing_key(errors, f"interface_packages.{name}")
                else:
                    _validate_fixed_mapping(
                        errors,
                        f"interface_packages.{name}",
                        interface_packages[name],
                        expected_package,
                    )

    topics = document.get("topics")
    if topics is not None:
        if not isinstance(topics, Mapping):
            errors.append("config error: topics must be a mapping")
        else:
            _unexpected_keys(errors, "topics", topics, set(_TOPICS))
            for name, expected_topic in _TOPICS.items():
                if name not in topics:
                    _missing_key(errors, f"topics.{name}")
                else:
                    _validate_fixed_mapping(errors, f"topics.{name}", topics[name], expected_topic)

    tf = document.get("tf")
    if tf is not None:
        _validate_fixed_mapping(errors, "tf", tf, _TF)

    grid_layers = document.get("required_grid_layers")
    if grid_layers is not None:
        if not isinstance(grid_layers, list):
            errors.append("config error: required_grid_layers must be a list")
        elif not all(isinstance(layer, str) and layer for layer in grid_layers):
            errors.append("config error: required_grid_layers must contain non-empty strings")
        elif len(set(grid_layers)) != len(grid_layers):
            errors.append("config error: required_grid_layers must not contain duplicates")
        elif grid_layers != _REQUIRED_GRID_LAYERS:
            errors.append(f"config error: required_grid_layers must be {_REQUIRED_GRID_LAYERS!r}")

    map_pyramid = document.get("map_pyramid")
    if map_pyramid is not None:
        _validate_fixed_mapping(errors, "map_pyramid", map_pyramid, _MAP_PYRAMID)

    static_inputs = document.get("static_inputs")
    if static_inputs is not None:
        if not isinstance(static_inputs, Mapping):
            errors.append("config error: static_inputs must be a mapping")
        else:
            _unexpected_keys(errors, "static_inputs", static_inputs, set(_STATIC_INPUTS))
            for name, expected_input in _STATIC_INPUTS.items():
                if name not in static_inputs:
                    _missing_key(errors, f"static_inputs.{name}")
                else:
                    _validate_fixed_mapping(
                        errors, f"static_inputs.{name}", static_inputs[name], expected_input
                    )
    return errors


def _interface_specs() -> list[tuple[str, tuple[str, ...]]]:
    specs = [
        (topic["type"], tuple(topic["required_fields"])) for topic in _TOPICS.values()
    ]
    specs.append(("lunar_navigation_msgs/msg/ScienceTargetRegion", ()))
    # Compatibility-only schema: generated and checked, but not an active
    # planner input topic in external_interfaces.yaml.
    specs.append(("lunar_navigation_msgs/msg/HopperPropellantState", ()))
    specs.append((_TF["type"], ()))
    return specs


def _package_for(interface_type: str) -> str:
    package, marker, _message_name = interface_type.partition("/msg/")
    if not marker or not package:
        raise ValueError(f"invalid ROS message type: {interface_type}")
    return package


def _ament_providers(ament_prefix_path: str, package: str) -> list[Path]:
    providers: list[Path] = []
    seen: set[Path] = set()
    for prefix_text in ament_prefix_path.split(os.pathsep):
        if not prefix_text:
            continue
        prefix = Path(prefix_text).resolve()
        marker = prefix / "share/ament_index/resource_index/packages" / package
        if marker.is_file() and prefix not in seen:
            seen.add(prefix)
            providers.append(prefix)
    return providers


def _validate_lunar_provider(
    expected_prefix: Path, ament_prefix_path: str
) -> list[str]:
    providers = _ament_providers(ament_prefix_path, "lunar_navigation_msgs")
    if not providers:
        return [
            "lunar_navigation_msgs: no ament provider found in AMENT_PREFIX_PATH"
        ]
    if len(providers) > 1:
        locations = ", ".join(str(provider) for provider in providers)
        return [
            "lunar_navigation_msgs: multiple ament providers in AMENT_PREFIX_PATH: "
            f"{locations}"
        ]
    if providers[0] != expected_prefix:
        return [
            "lunar_navigation_msgs: unexpected ament provider: "
            f"expected {expected_prefix}, got {providers[0]}"
        ]
    return []


def _declaration_mismatch(
    interface_type: str, expected: tuple[str, ...], actual: tuple[str, ...]
) -> str | None:
    for index in range(max(len(expected), len(actual))):
        expected_value = expected[index] if index < len(expected) else None
        actual_value = actual[index] if index < len(actual) else None
        if expected_value == actual_value:
            continue
        expected_text = repr(expected_value) if expected_value is not None else "end of declarations"
        actual_text = repr(actual_value) if actual_value is not None else "end of declarations"
        return (
            f"{interface_type}: declaration mismatch: expected {expected_text} "
            f"at declaration {index + 1}, got {actual_text}"
        )
    return None


def check_interfaces(
    config: Path,
    *,
    expected_lunar_navigation_prefix: Path,
    run: CommandRunner = run_ros_command,
    ament_prefix_path: str | None = None,
    package_locations: dict[str, str] | None = None,
) -> list[str]:
    """Return external ROS package, type, or field validation errors."""
    try:
        document = yaml.safe_load(config.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        return [f"config error: unable to load config: {error}"]

    config_errors = validate_external_config(document)
    if config_errors:
        return config_errors

    expected_lunar_prefix = expected_lunar_navigation_prefix.resolve()
    effective_ament_prefix_path = (
        os.environ.get("AMENT_PREFIX_PATH", "")
        if ament_prefix_path is None
        else ament_prefix_path
    )
    errors = _validate_lunar_provider(
        expected_lunar_prefix, effective_ament_prefix_path
    )
    checked_packages: set[str] = set()
    for interface_type, required_fields in _interface_specs():
        package = _package_for(interface_type)
        if package not in checked_packages:
            checked_packages.add(package)
            package_result = run(["ros2", "pkg", "prefix", package])
            if package_result.returncode != 0:
                errors.append(f"{package}: {_command_failure('ros2 pkg prefix', package_result)}")
            else:
                actual_prefix = Path(package_result.stdout.strip()).resolve()
                expected_prefix = (
                    expected_lunar_prefix
                    if package == "lunar_navigation_msgs"
                    else _SYSTEM_PREFIX
                )
                if package_locations is not None:
                    package_locations[package] = str(actual_prefix)
                if actual_prefix != expected_prefix:
                    errors.append(
                        f"{package}: unexpected package prefix: "
                        f"expected {expected_prefix}, got {actual_prefix}"
                    )

        interface_result = run(["ros2", "interface", "show", interface_type])
        if interface_result.returncode != 0:
            errors.append(
                f"{interface_type}: {_command_failure('ros2 interface show', interface_result)}"
            )
            continue

        expected_declarations = _PROVISIONAL_DECLARATIONS.get(interface_type)
        if expected_declarations is not None:
            actual_declarations = parse_top_level_declarations(interface_result.stdout)
            mismatch = _declaration_mismatch(
                interface_type, expected_declarations, actual_declarations
            )
            if mismatch is not None:
                errors.append(mismatch)
            continue

        available_fields = parse_top_level_fields(interface_result.stdout)
        missing_fields = [field for field in required_fields if field not in available_fields]
        if missing_fields:
            errors.append(
                f"{interface_type}: missing top-level fields: {', '.join(missing_fields)}"
            )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-lunar-navigation-prefix", type=Path, required=True)
    arguments = parser.parse_args()

    package_locations: dict[str, str] = {}
    errors = check_interfaces(
        arguments.config,
        expected_lunar_navigation_prefix=arguments.expected_lunar_navigation_prefix,
        package_locations=package_locations,
    )
    for package, location in sorted(package_locations.items()):
        print(f"external package: {package}: {location}")
    if errors:
        for error in errors:
            print(f"external interface error: {error}")
        return 1
    print("external interfaces: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

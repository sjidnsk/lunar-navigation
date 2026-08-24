"""Check the isolated pure-planner external ROS contract without touching legacy v5."""

from __future__ import annotations

import argparse
import os
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path

import yaml


CommandRunner = Callable[[list[str]], subprocess.CompletedProcess[str]]
SCHEMA_VERSION = "lunar-pure-planner-task3-inputs/v1"
CONFIG_RELATIVE_PATH = Path("config/external_interfaces.yaml")
CONTRACT = {
    "schema_version": SCHEMA_VERSION,
    "topics": {
        "global_overview": {
            "name": "/Car/T3/mapping/global_overview",
            "type": "nav_msgs/msg/OccupancyGrid",
            "owner": "external",
            "frame": "map",
            "required_fields": ["header", "info", "data"],
            "data_encoding": "int8",
        },
        "grid_map": {
            "name": "/Car/T3/mapping/grid_map",
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
            "required_layers": ["occupancy", "elevation"],
        },
        "odometry": {
            "name": "/Car/T3/localization/odometry",
            "type": "nav_msgs/msg/Odometry",
            "owner": "external",
            "frame": "odom",
            "child_frame": "base_link",
            "required_fields": ["header", "child_frame_id", "pose", "twist"],
        },
    },
    "tf": {
        "name": "/tf",
        "type": "tf2_msgs/msg/TFMessage",
        "owner": "external",
        "chain": ["map", "odom"],
        "required_fields": ["transforms"],
    },
    "action": {
        "name": "/Car/T4/plan_motion",
        "type": "lunar_planning_msgs/action/PlanMotion",
        "owner": "lunar_pure_planner_ros",
        "required_goal_fields": ["environment_mode"],
    },
    "diagnostics": {
        "name": "/Car/T4/planning/diagnostics",
        "type": "diagnostic_msgs/msg/DiagnosticArray",
        "owner": "lunar_pure_planner_ros",
        "required_fields": ["header", "status"],
        "required_keys": [
            "request_id",
            "platform_type",
            "environment_mode",
            "planning_outcome",
            "reason_code",
            "global_elapsed_ms",
            "global_call_count",
            "local_elapsed_ms",
            "local_call_count",
            "total_elapsed_ms",
        ],
    },
}


def run_ros_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    """Run one ROS CLI command without masking a missing ROS environment."""
    try:
        return subprocess.run(command, check=False, capture_output=True, text=True)
    except OSError as error:
        return subprocess.CompletedProcess(command, 127, "", str(error))


def _fields(definition: str) -> set[str]:
    fields: set[str] = set()
    for raw_line in definition.splitlines():
        if not raw_line or raw_line[0].isspace():
            continue
        tokens = raw_line.split("#", maxsplit=1)[0].split()
        if len(tokens) == 2 and "=" not in tokens[1]:
            fields.add(tokens[1])
    return fields


def _declarations(definition: str) -> set[str]:
    declarations: set[str] = set()
    for raw_line in definition.splitlines():
        if not raw_line or raw_line[0].isspace():
            continue
        declaration = raw_line.split("#", maxsplit=1)[0].strip()
        if declaration:
            declarations.add(declaration)
    return declarations


def _action_sections(definition: str) -> tuple[list[str], int]:
    """Split a ROS Action definition without merging its three scoped sections."""
    lines = definition.splitlines()
    separators = [
        index for index, raw_line in enumerate(lines) if raw_line.strip() == "---"
    ]
    if len(separators) != 2:
        return [], len(separators)
    first, second = separators
    return [
        "\n".join(lines[:first]),
        "\n".join(lines[first + 1:second]),
        "\n".join(lines[second + 1:]),
    ], len(separators)


def _validate_action_definition(
    interface_type: str,
    definition: str,
    action_contract: Mapping[str, object],
) -> list[str]:
    """Validate Goal, Result, and Feedback declarations in their own scopes."""
    sections, separator_count = _action_sections(definition)
    if not sections:
        return [
            f"{interface_type}: malformed Action definition: expected exactly 2 "
            "section separators for Goal/Result/Feedback, "
            f"got {separator_count}"
        ]

    section_names = ("Goal", "Result", "Feedback")
    section_fields = [_fields(section) for section in sections]
    section_declarations = [_declarations(section) for section in sections]
    errors: list[str] = []

    for section_index, section_name in enumerate(section_names):
        contract_key = f"required_{section_name.lower()}_fields"
        required_fields = action_contract.get(contract_key, [])
        for field in required_fields:
            if field in section_fields[section_index]:
                continue
            other_section = next(
                (
                    other_name
                    for other_index, other_name in enumerate(section_names)
                    if other_index != section_index
                    and field in section_fields[other_index]
                ),
                None,
            )
            if other_section is None:
                errors.append(
                    f"{interface_type}: {section_name} section missing top-level "
                    f"field: {field}"
                )
            else:
                errors.append(
                    f"{interface_type}: {section_name} field {field} is declared "
                    f"in {other_section} section"
                )

    for declaration in (
        "uint8 LUNAR_SURFACE=1",
        "uint8 LAVA_TUBE=2",
        "uint8 environment_mode",
    ):
        if declaration in section_declarations[0]:
            continue
        other_section = next(
            (
                section_names[index]
                for index in (1, 2)
                if declaration in section_declarations[index]
            ),
            None,
        )
        if other_section is None:
            errors.append(
                f"{interface_type}: Goal section missing declaration: {declaration}"
            )
        else:
            errors.append(
                f"{interface_type}: Goal declaration {declaration} is declared in "
                f"{other_section} section"
            )
    return errors


def validate_external_config(document: object) -> list[str]:
    """Return deterministic errors for any drift from the six-interface pure contract."""
    if not isinstance(document, Mapping):
        return ["config error: document must be a mapping"]
    if document == CONTRACT:
        return []
    if document.get("schema_version") != SCHEMA_VERSION:
        return [f"config error: schema_version must be {SCHEMA_VERSION!r}"]
    return ["config error: document must exactly match the pure-planner v1 contract"]


def _providers(ament_prefix_path: str, package: str) -> list[Path]:
    providers: list[Path] = []
    for raw_prefix in ament_prefix_path.split(os.pathsep):
        if not raw_prefix:
            continue
        prefix = Path(raw_prefix).resolve()
        marker = prefix / "share/ament_index/resource_index/packages" / package
        if marker.is_file() and prefix not in providers:
            providers.append(prefix)
    return providers


def _interface_specs() -> list[tuple[str, list[str]]]:
    return [
        (value["type"], value["required_fields"])
        for value in CONTRACT["topics"].values()
    ] + [
        (CONTRACT["tf"]["type"], CONTRACT["tf"]["required_fields"]),
        (CONTRACT["action"]["type"], CONTRACT["action"]["required_goal_fields"]),
        (CONTRACT["diagnostics"]["type"], CONTRACT["diagnostics"]["required_fields"]),
    ]


def _package_name(interface_type: str) -> str:
    package, _separator, _rest = interface_type.partition("/")
    return package


def _command_error(command: str, result: subprocess.CompletedProcess[str]) -> str:
    detail = result.stderr.strip() or result.stdout.strip() or f"exit code {result.returncode}"
    return f"{command} failed: {detail}"


def check_interfaces(
    config: Path,
    *,
    expected_lunar_planning_prefix: Path,
    run: CommandRunner = run_ros_command,
    ament_prefix_path: str | None = None,
    package_locations: dict[str, str] | None = None,
) -> list[str]:
    """Validate pure ROS types and ensure one rebuilt PlanMotion provider is selected."""
    try:
        document = yaml.safe_load(config.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        return [f"config error: unable to load config: {error}"]
    errors = validate_external_config(document)
    if errors:
        return errors

    expected_prefix = expected_lunar_planning_prefix.resolve()
    providers = _providers(
        os.environ.get("AMENT_PREFIX_PATH", "") if ament_prefix_path is None else ament_prefix_path,
        "lunar_planning_msgs",
    )
    if len(providers) != 1:
        rendered = ", ".join(str(provider) for provider in providers) or "none"
        errors.append(f"lunar_planning_msgs: expected one ament provider, got {rendered}")
    elif providers[0] != expected_prefix:
        errors.append(
            "lunar_planning_msgs: unexpected ament provider: "
            f"expected {expected_prefix}, got {providers[0]}"
        )

    checked_packages: set[str] = set()
    for interface_type, required_fields in _interface_specs():
        package = _package_name(interface_type)
        if package not in checked_packages:
            checked_packages.add(package)
            package_result = run(["ros2", "pkg", "prefix", package])
            if package_result.returncode != 0:
                errors.append(f"{package}: {_command_error('ros2 pkg prefix', package_result)}")
            else:
                actual_prefix = Path(package_result.stdout.strip()).resolve()
                if package_locations is not None:
                    package_locations[package] = str(actual_prefix)
                if package == "lunar_planning_msgs" and actual_prefix != expected_prefix:
                    errors.append(
                        "lunar_planning_msgs: unexpected package prefix: "
                        f"expected {expected_prefix}, got {actual_prefix}"
                    )

        interface_result = run(["ros2", "interface", "show", interface_type])
        if interface_result.returncode != 0:
            errors.append(f"{interface_type}: {_command_error('ros2 interface show', interface_result)}")
            continue
        if interface_type == CONTRACT["action"]["type"]:
            errors.extend(
                _validate_action_definition(
                    interface_type,
                    interface_result.stdout,
                    document["action"],
                )
            )
            continue
        missing = [field for field in required_fields if field not in _fields(interface_result.stdout)]
        if missing:
            errors.append(f"{interface_type}: missing top-level fields: {', '.join(missing)}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repository", nargs="?", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--expected-lunar-planning-prefix", type=Path)
    arguments = parser.parse_args()
    config = arguments.config or arguments.repository / CONFIG_RELATIVE_PATH
    try:
        document = yaml.safe_load(config.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        print(f"pure planner external interface error: unable to load config: {error}")
        return 1
    errors = validate_external_config(document)
    if not errors and arguments.expected_lunar_planning_prefix is not None:
        locations: dict[str, str] = {}
        errors = check_interfaces(
            config,
            expected_lunar_planning_prefix=arguments.expected_lunar_planning_prefix,
            package_locations=locations,
        )
        for package, prefix in sorted(locations.items()):
            print(f"pure planner external package: {package}: {prefix}")
    if errors:
        for error in errors:
            print(f"pure planner external interface error: {error}")
        return 1
    print("pure planner external interfaces: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


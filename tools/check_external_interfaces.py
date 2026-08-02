"""Check that externally owned ROS interfaces are installed and compatible."""

from __future__ import annotations

import argparse
import re
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path

import yaml


CommandRunner = Callable[[list[str]], subprocess.CompletedProcess[str]]
_FIELD_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


def run_ros_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    """Run one ROS CLI command without raising when ROS is unavailable."""
    try:
        return subprocess.run(command, check=False, capture_output=True, text=True)
    except OSError as error:
        return subprocess.CompletedProcess(command, 127, stdout="", stderr=str(error))


def parse_top_level_fields(interface_definition: str) -> set[str]:
    """Extract declared top-level ROS message field names from CLI output."""
    fields: set[str] = set()
    for raw_line in interface_definition.splitlines():
        if not raw_line or raw_line[0].isspace():
            continue
        declaration = raw_line.split("#", maxsplit=1)[0].strip()
        tokens = declaration.split()
        if len(tokens) != 2:
            continue
        name = tokens[1]
        if "=" not in name and _FIELD_NAME.fullmatch(name):
            fields.add(name)
    return fields


def _command_failure(command: str, result: subprocess.CompletedProcess[str]) -> str:
    detail = result.stderr.strip() or result.stdout.strip() or f"exit code {result.returncode}"
    return f"{command} failed: {detail}"


def _interface_specs(document: Mapping[str, object]) -> list[tuple[str, tuple[str, ...]]]:
    topics = document.get("topics", {})
    if not isinstance(topics, Mapping):
        raise ValueError("topics must be a mapping")

    specs: list[tuple[str, tuple[str, ...]]] = []
    for topic in topics.values():
        if not isinstance(topic, Mapping):
            raise ValueError("each topic must be a mapping")
        interface_type = topic.get("type")
        required_fields = topic.get("required_fields", [])
        if not isinstance(interface_type, str) or not isinstance(required_fields, list):
            raise ValueError("topic type and required_fields must be declared")
        specs.append((interface_type, tuple(str(field) for field in required_fields)))

    tf = document.get("tf")
    if tf is not None:
        if not isinstance(tf, Mapping) or not isinstance(tf.get("type"), str):
            raise ValueError("tf type must be declared")
        specs.append((tf["type"], ()))
    return specs


def _package_for(interface_type: str) -> str:
    package, marker, _message_name = interface_type.partition("/msg/")
    if not marker or not package:
        raise ValueError(f"invalid ROS message type: {interface_type}")
    return package


def check_interfaces(
    config: Path,
    run: CommandRunner = run_ros_command,
    package_locations: dict[str, str] | None = None,
) -> list[str]:
    """Return external ROS package, type, or field validation errors."""
    document = yaml.safe_load(config.read_text(encoding="utf-8"))
    if not isinstance(document, Mapping):
        raise ValueError("external interface config must be a mapping")

    errors: list[str] = []
    checked_packages: set[str] = set()
    for interface_type, required_fields in _interface_specs(document):
        package = _package_for(interface_type)
        if package not in checked_packages:
            checked_packages.add(package)
            package_result = run(["ros2", "pkg", "prefix", package])
            if package_result.returncode != 0:
                errors.append(f"{package}: {_command_failure('ros2 pkg prefix', package_result)}")
            elif package_locations is not None:
                package_locations[package] = package_result.stdout.strip()

        interface_result = run(["ros2", "interface", "show", interface_type])
        if interface_result.returncode != 0:
            errors.append(
                f"{interface_type}: {_command_failure('ros2 interface show', interface_result)}"
            )
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
    arguments = parser.parse_args()

    package_locations: dict[str, str] = {}
    errors = check_interfaces(arguments.config, package_locations=package_locations)
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

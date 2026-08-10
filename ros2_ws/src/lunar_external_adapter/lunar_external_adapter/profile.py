"""Strict loader for the fixed-path external ROS interface profile."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Callable, Final

import yaml


SCHEMA_VERSION: Final = "lunar-interface-profile/v1"
DEFAULT_RUNTIME_PATH: Final = Path(
    "/etc/lunar_navigation/interface_profile.yaml"
)
REQUIRED_CHANNELS: Final = frozenset(
    {
        "map_global",
        "map_local",
        "odometry",
        "localization_status",
        "exploration_task",
        "motion_execution_feedback",
        "tf",
    }
)
_TOPIC_PATTERN: Final = re.compile(r"^/[A-Za-z0-9_/]+$")
_TYPE_PATTERN: Final = re.compile(
    r"^[A-Za-z][A-Za-z0-9_]*/msg/[A-Za-z][A-Za-z0-9_]*$"
)


class InterfaceProfileError(ValueError):
    """The interface profile cannot define one unambiguous adapter graph."""


@dataclass(frozen=True)
class ChannelProfile:
    mode: str
    input_topic: str
    output_topic: str
    input_type: str
    output_type: str
    frame: str
    converter: str | None = None


@dataclass(frozen=True)
class InterfaceProfile:
    schema_version: str
    project_id: str
    channels: dict[str, ChannelProfile]


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(
        isinstance(key, str) for key in value
    ):
        raise InterfaceProfileError(f"{path} must be a string-keyed map")
    return value


def _exact_keys(
    value: dict[str, Any],
    *,
    required: set[str] | frozenset[str],
    optional: set[str] | frozenset[str] = frozenset(),
    path: str,
) -> None:
    missing = required - value.keys()
    unknown = value.keys() - required - optional
    if missing:
        raise InterfaceProfileError(
            f"{path} missing keys: {', '.join(sorted(missing))}"
        )
    if unknown:
        raise InterfaceProfileError(
            f"{path} unknown keys: {', '.join(sorted(unknown))}"
        )


def _nonempty_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InterfaceProfileError(f"{path} must be a non-empty string")
    return value


def _channel(name: str, value: Any) -> ChannelProfile:
    document = _mapping(value, f"channels.{name}")
    _exact_keys(
        document,
        required={
            "mode",
            "input_topic",
            "output_topic",
            "input_type",
            "output_type",
            "frame",
        },
        optional={"converter"},
        path=f"channels.{name}",
    )
    mode = _nonempty_string(document["mode"], f"channels.{name}.mode")
    if mode not in {"direct_remap", "converter"}:
        raise InterfaceProfileError(f"channels.{name}.mode is unsupported")
    input_topic = _nonempty_string(
        document["input_topic"], f"channels.{name}.input_topic"
    )
    output_topic = _nonempty_string(
        document["output_topic"], f"channels.{name}.output_topic"
    )
    if (
        not _TOPIC_PATTERN.fullmatch(input_topic)
        or not _TOPIC_PATTERN.fullmatch(output_topic)
    ):
        raise InterfaceProfileError(
            f"channels.{name} topics must be an absolute ROS topic"
        )
    input_type = _nonempty_string(
        document["input_type"], f"channels.{name}.input_type"
    )
    output_type = _nonempty_string(
        document["output_type"], f"channels.{name}.output_type"
    )
    if not _TYPE_PATTERN.fullmatch(input_type) or not _TYPE_PATTERN.fullmatch(
        output_type
    ):
        raise InterfaceProfileError(
            f"channels.{name} types must use package/msg/Type syntax"
        )
    frame = _nonempty_string(document["frame"], f"channels.{name}.frame")

    converter_value = document.get("converter")
    if mode == "direct_remap":
        if converter_value is not None:
            raise InterfaceProfileError(
                f"channels.{name}.converter is forbidden for direct_remap"
            )
        if input_type != output_type:
            raise InterfaceProfileError(
                f"channels.{name} direct_remap types must be identical"
            )
        converter = None
    else:
        converter = _nonempty_string(
            converter_value, f"channels.{name}.converter"
        )
        resolve_converter(converter)
        if input_topic == output_topic:
            raise InterfaceProfileError(
                f"channels.{name} converter input_topic and output_topic "
                "must differ"
            )
    return ChannelProfile(
        mode=mode,
        input_topic=input_topic,
        output_topic=output_topic,
        input_type=input_type,
        output_type=output_type,
        frame=frame,
        converter=converter,
    )


def resolve_converter(name: str) -> Callable[[Any], Any]:
    """Resolve only a compiled-in converter.

    Dynamic expressions are forbidden.
    """
    from .conversions import CONVERTERS

    try:
        return CONVERTERS[name]
    except KeyError as error:
        raise InterfaceProfileError(f"unknown converter: {name}") from error


def load_interface_profile(path: str | Path) -> InterfaceProfile:
    """Load and strictly validate one complete interface profile."""
    profile_path = Path(path)
    try:
        document = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise InterfaceProfileError(
            f"cannot read interface profile: {error}"
        ) from error
    root = _mapping(document, "profile")
    _exact_keys(
        root,
        required={"schema_version", "project_id", "channels"},
        path="profile",
    )
    schema_version = _nonempty_string(root["schema_version"], "schema_version")
    if schema_version != SCHEMA_VERSION:
        raise InterfaceProfileError(
            f"schema_version must equal {SCHEMA_VERSION}"
        )
    project_id = _nonempty_string(root["project_id"], "project_id")
    channel_documents = _mapping(root["channels"], "channels")
    _exact_keys(
        channel_documents,
        required=REQUIRED_CHANNELS,
        path="channels",
    )
    channels = {
        name: _channel(name, channel_documents[name])
        for name in sorted(REQUIRED_CHANNELS)
    }
    return InterfaceProfile(
        schema_version=schema_version,
        project_id=project_id,
        channels=channels,
    )


def direct_remappings(
    profile: InterfaceProfile,
) -> tuple[tuple[str, str], ...]:
    """Return stable-name to provider-name remaps without copying messages."""
    return tuple(
        (channel.output_topic, channel.input_topic)
        for channel in profile.channels.values()
        if channel.mode == "direct_remap"
    )

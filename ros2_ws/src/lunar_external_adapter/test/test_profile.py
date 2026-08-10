from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from rclpy.qos import DurabilityPolicy

from lunar_external_adapter.node import ExternalAdapter

from lunar_external_adapter.profile import (
    InterfaceProfileError,
    load_interface_profile,
    resolve_converter,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROFILE = (
    ROOT
    / "lunar_navigation_config"
    / "config"
    / "interface_profiles"
    / "default.yaml"
)
PROFILE_SCHEMA = (
    ROOT
    / "lunar_navigation_config"
    / "config"
    / "interface_profile.schema.json"
)


def write_mutated_profile(tmp_path: Path, mutate) -> Path:
    document = yaml.safe_load(DEFAULT_PROFILE.read_text(encoding="utf-8"))
    mutate(document)
    path = tmp_path / "interface_profile.yaml"
    path.write_text(
        yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
    )
    return path


def test_default_profile_uses_direct_remap_for_standard_types() -> None:
    profile = load_interface_profile(DEFAULT_PROFILE)

    assert profile.schema_version == "lunar-interface-profile/v1"
    assert set(profile.channels) == {
        "map_global",
        "map_local",
        "odometry",
        "localization_status",
        "exploration_task",
        "motion_execution_feedback",
        "tf",
    }
    assert profile.channels["map_global"].mode == "direct_remap"
    assert profile.channels["map_local"].mode == "direct_remap"
    assert profile.channels["odometry"].mode == "direct_remap"
    assert profile.channels["tf"].mode == "direct_remap"
    assert profile.channels["localization_status"].converter == (
        "localization_status_v1"
    )


def test_exploration_task_converter_preserves_transient_local_delivery() -> None:
    profile = load_interface_profile(DEFAULT_PROFILE)

    task_qos = ExternalAdapter._channel_qos(profile.qos, "exploration_task")
    feedback_qos = ExternalAdapter._channel_qos(
        profile.qos, "motion_execution_feedback"
    )

    assert task_qos.depth == 1
    assert task_qos.durability == DurabilityPolicy.TRANSIENT_LOCAL
    assert feedback_qos.durability == DurabilityPolicy.VOLATILE


def test_schema_and_default_profile_define_same_closed_channels() -> None:
    schema = json.loads(PROFILE_SCHEMA.read_text(encoding="utf-8"))
    channel_schema = schema["properties"]["channels"]
    document = yaml.safe_load(DEFAULT_PROFILE.read_text(encoding="utf-8"))

    assert channel_schema["additionalProperties"] is False
    assert set(channel_schema["required"]) == set(document["channels"])
    assert set(channel_schema["properties"]) == set(document["channels"])


def test_unknown_converter_is_rejected(tmp_path: Path) -> None:
    path = write_mutated_profile(
        tmp_path,
        lambda document: document["channels"]["localization_status"].update(
            converter="dynamic_eval"
        ),
    )

    with pytest.raises(InterfaceProfileError, match="unknown converter"):
        load_interface_profile(path)
    with pytest.raises(InterfaceProfileError, match="unknown converter"):
        resolve_converter("dynamic_eval")


@pytest.mark.parametrize(
    "mutate, expected",
    [
        (
            lambda document: document["channels"]["map_global"].update(
                input_topic="relative/map"
            ),
            "absolute ROS topic",
        ),
        (
            lambda document: document["channels"]["map_global"].update(
                output_type="nav_msgs/msg/Odometry"
            ),
            "direct_remap types",
        ),
        (
            lambda document: document["channels"][
                "localization_status"
            ].update(output_topic="/localization/status"),
            "converter input_topic and output_topic",
        ),
        (
            lambda document: document["channels"]["odometry"].update(frame=""),
            "frame",
        ),
        (
            lambda document: document["qos"].update(depth=0),
            "qos.depth",
        ),
    ],
)
def test_invalid_channel_contract_is_rejected(
    tmp_path: Path,
    mutate,
    expected: str,
) -> None:
    path = write_mutated_profile(tmp_path, mutate)

    with pytest.raises(InterfaceProfileError, match=expected):
        load_interface_profile(path)


def test_unknown_profile_keys_are_rejected(tmp_path: Path) -> None:
    path = write_mutated_profile(
        tmp_path,
        lambda document: document.update(dynamic_fields="forbidden"),
    )

    with pytest.raises(InterfaceProfileError, match="unknown keys"):
        load_interface_profile(path)

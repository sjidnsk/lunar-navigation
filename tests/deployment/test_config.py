from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from deployment.luna_runtime.config import (
    ConfigError,
    load_runtime_config,
    load_profile,
    render_planner_params,
    validate_host,
)
from deployment.luna_runtime.host import HostFacts


def test_deployment_observation_capability_is_ten_meter_ninety_degree_baseline() -> None:
    root = Path(__file__).resolve().parents[2]
    capability_path = root / "deployment" / "config" / "observation.yaml"

    with capability_path.open(encoding="utf-8") as stream:
        capability = yaml.safe_load(stream)

    assert capability == {"sensor_range_m": 10.0, "sensor_fov_deg": 90.0}


def _runtime_config_text(*, include_global_map: bool = True) -> str:
    global_map = "  map_global: /environment/map_global\n" if include_global_map else ""
    return f"""\
profile: ubuntu22-humble-amd64
interfaces:
{global_map}  map_local: /environment/map_local
  odometry: /localization/odometry
  localization_status: /localization/status
  tf: /tf
  exploration_task: /mission/exploration_task
  motion_feedback: /execution/motion_feedback
  plan_motion: /plan_motion
  diagnostics: /diagnostics
  certified_route_markers: /planning/certified_route_markers
  provisional_route_markers: /planning/provisional_route_markers
capabilities:
  platform_file: /tmp/platform.yaml
  observation_file: /tmp/observation.yaml
planner:
  enable_nav2_adapter: false
  snapshot_policy:
    global_map_max_age: 1.0
    local_map_max_age: 1.0
    odometry_max_age: 1.0
    localization_status_max_age: 1.0
    tf_max_age: 1.0
    max_pairwise_skew: 1.0
policy: {{mode: fallback, model_id: null}}
extensions: {{map_pipeline: false, path_tracking: false}}
input_adapters: {{mode: external_canonical, task3_config_file: null}}
controller:
  wheeled:
    enabled: false
    command_topic: /Car/T5/Car_Cmd_Vel
    odometry_topic: /Car/T3/semantic/current_pose
    feedback_topic: /execution/motion_feedback
    execution_goal_topic: /mission/execution_goal
    reference_topic: /execution/wheeled_reference
    control_rate_hz: 20.0
    lookahead_m: 1.0
    max_linear_mps: 0.2
    max_angular_radps: 0.5
    max_cross_track_error_m: 1.0
    goal_position_tolerance_m: 0.25
    goal_yaw_tolerance_rad: 0.35
    reference_max_age_s: 1.0
    odometry_max_age_s: 0.5
runtime: {{log_level: INFO}}
"""


def test_orin_profile_rejects_amd64_host() -> None:
    profile = load_profile("jetson-orin-r36")
    facts = HostFacts(
        os_id="ubuntu",
        os_version="22.04",
        architecture="x86_64",
        ros_distro="humble",
        l4t="R36.0.0",
        jetpack="6.0",
    )
    assert validate_host(profile, facts) == ("ARCHITECTURE_MISMATCH",)


def test_runtime_config_rejects_missing_required_interface(tmp_path: Path) -> None:
    config = tmp_path / "runtime.yaml"
    config.write_text(_runtime_config_text(include_global_map=False), encoding="utf-8")

    with pytest.raises(ConfigError, match="interfaces.map_global"):
        load_runtime_config(config)


def test_runtime_config_rejects_duplicate_endpoint(tmp_path: Path) -> None:
    config = tmp_path / "runtime.yaml"
    config.write_text(
        _runtime_config_text().replace("  diagnostics: /diagnostics", "  diagnostics: /plan_motion"),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="duplicate interface"):
        load_runtime_config(config)


def test_runtime_config_rejects_safety_disable_switch(tmp_path: Path) -> None:
    config = tmp_path / "runtime.yaml"
    config.write_text(
        _runtime_config_text().replace(
            "  enable_nav2_adapter: false", "  enable_nav2_adapter: false\n  disable_safety_projection: true"
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="safety switch"):
        load_runtime_config(config)


def test_valid_fallback_config_renders_planner_ros_parameters(tmp_path: Path) -> None:
    config_path = tmp_path / "runtime.yaml"
    config_path.write_text(_runtime_config_text(), encoding="utf-8")
    config = load_runtime_config(config_path)
    output = tmp_path / "planner.yaml"

    render_planner_params(config, output)

    text = output.read_text(encoding="utf-8")
    assert "/lunar_planner:" in text
    assert "interfaces.map_global: /environment/map_global" in text
    assert "global_map_max_age: 1.0" in text
    assert "enable_nav2_adapter: false" in text
    assert "platform_capability_file: /tmp/platform.yaml" in text
    assert "observation_capability_file: /tmp/observation.yaml" in text
    assert "capability_package" not in text


def test_task3_adapted_config_requires_absolute_adapter_config_path(tmp_path: Path) -> None:
    config = tmp_path / "runtime.yaml"
    config.write_text(
        _runtime_config_text().replace(
            "input_adapters: {mode: external_canonical, task3_config_file: null}",
            "input_adapters: {mode: task3_adapted, task3_config_file: relative.yaml}",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="input_adapters.task3_config_file"):
        load_runtime_config(config)


def test_explicit_disabled_wheeled_controller_is_retained_in_runtime_config(tmp_path: Path) -> None:
    path = tmp_path / "runtime.yaml"
    path.write_text(_runtime_config_text(), encoding="utf-8")

    config = load_runtime_config(path)

    assert config.controller["wheeled"]["enabled"] is False

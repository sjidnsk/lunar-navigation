from __future__ import annotations

from pathlib import Path

import pytest

from deployment.luna_runtime.config import (
    ConfigError,
    load_runtime_config,
    load_profile,
    render_planner_params,
    validate_host,
)
from deployment.luna_runtime.host import HostFacts


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

"""Static Task 14 contracts for the isolated pure-exploration launch surface."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import subprocess
import uuid

import pytest
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PURE_PLANNER_ROOT = REPOSITORY_ROOT
YAML_PATH = PURE_PLANNER_ROOT / "config" / "pure_exploration.yaml"
LAUNCH_PATH = PURE_PLANNER_ROOT / "launch" / "pure_exploration.launch.py"
VALIDATOR_PATH = PURE_PLANNER_ROOT / "tools" / "validate_pure_exploration_profile.py"
EXPLORATION_CMAKE_PATH = (
    PURE_PLANNER_ROOT / "ros2_ws" / "src" / "lunar_pure_exploration_ros" / "CMakeLists.txt"
)
PLANNER_CMAKE_PATH = (
    PURE_PLANNER_ROOT / "ros2_ws" / "src" / "lunar_pure_planner_ros" / "CMakeLists.txt"
)
LIVE_GRAPH_PROBE_PATH = Path(__file__).with_name("pure_exploration_live_graph_probe.py")

RESOURCE_LIMITS = (
    "maximum_position_probes",
    "maximum_candidate_views",
    "maximum_collision_work_units",
    "maximum_visibility_work_units",
    "maximum_path_preview_poses",
    "maximum_executable_path_points",
    "maximum_failure_entries",
    "maximum_failure_patch_cells_per_entry",
    "maximum_failure_total_patch_cells",
)
RESOURCE_ENV = {
    name: f"PURE_EXPLORATION_{name.upper()}" for name in RESOURCE_LIMITS
}
EXPECTED_TOPICS = {
    "global_map_topic": "/Car/T3/mapping/global_overview",
    "odometry_topic": "/Car/T3/localization/odometry",
    "tf_topic": "/tf",
    "task_topic": "/Car/T4/exploration/task",
    "planner_action": "/Car/T4/plan_motion",
    "planner_diagnostics_topic": "/Car/T4/planning/diagnostics",
    "motion_reference_topic": "/Car/T4/execution/motion_reference",
    "execution_cancel_topic": "/Car/T4/execution/cancel",
}
FAILURE_SLOTS = {
    "entries": ("entry", "maximum_entries"),
    "patch_per_entry": ("per_entry", "maximum_patch_cells_per_entry"),
    "total_patch": ("total", "maximum_total_patch_cells"),
}


def _docker(
    arguments: list[str], *, check: bool = True, timeout: float | None = None
) -> subprocess.CompletedProcess[str]:
    """Run one Docker command and preserve stdout/stderr for a failing test."""
    return subprocess.run(
        ["docker", *arguments],
        check=check,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _run_no_mount_live_graph() -> None:
    """Build and exercise the installed graph in an ephemeral, mount-free Humble container."""
    assert LIVE_GRAPH_PROBE_PATH.is_file()
    image = "osrf/ros:humble-desktop-full-jammy"
    if _docker(["image", "inspect", image], check=False).returncode != 0:
        _docker(["pull", image])

    container = f"task14-live-{uuid.uuid4().hex[:12]}"
    domain_id = str(20 + (uuid.uuid4().int % 200))
    try:
        _docker(["create", "--name", container, image, "sleep", "infinity"])
        mounts = json.loads(_docker(["inspect", "--format", "{{json .Mounts}}", container]).stdout)
        assert mounts == [], "Task14 live graph container must not mount the checkout"
        _docker(["start", container])
        _docker(["exec", container, "mkdir", "-p", "/workspace/ros2_ws/src"])
        _docker(["cp", str(PURE_PLANNER_ROOT), f"{container}:/workspace/pure_planner"])
        _docker(
            [
                "exec",
                container,
                "bash",
                "-lc",
                "set -eo pipefail\n"
                "export DEBIAN_FRONTEND=noninteractive\n"
                "apt-get update\n"
                "apt-get install -y --no-install-recommends ros-humble-grid-map-msgs\n"
                "source /opt/ros/humble/setup.bash\n"
                "cd /workspace\n"
                "colcon build --merge-install "
                "--base-paths pure_planner/ros2_ws/src "
                "--packages-up-to lunar_pure_exploration_ros "
                "--cmake-args -DBUILD_TESTING=OFF\n",
            ]
        )
        _docker(
            [
                "exec",
                "-e",
                f"ROS_DOMAIN_ID={domain_id}",
                "-e",
                "ROS_LOCALHOST_ONLY=1",
                container,
                "bash",
                "-lc",
                "set -eo pipefail\n"
                "source /opt/ros/humble/setup.bash\n"
                "source /workspace/install/setup.bash\n"
                "python3 /workspace/pure_planner/tests/launch/"
                "pure_exploration_live_graph_probe.py\n",
            ],
            timeout=90.0,
        )
    finally:
        _docker(["rm", "--force", container], check=False)


def _load_validator():
    """Load the isolated tool without requiring it to be an installed package."""
    spec = importlib.util.spec_from_file_location("pure_exploration_profile", VALIDATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _profile_fixture(tmp_path: Path) -> tuple[dict[str, str], Path]:
    """Build temporary test-only external evidence, never a repository profile."""
    limits = {
        "maximum_entries": 8,
        "maximum_patch_cells_per_entry": 256,
        "maximum_total_patch_cells": 1024,
    }
    environment = "test-only-humble"
    image = "test-only-image@sha256:fixture"
    profile = {
        "PURE_EXPLORATION_ORIN_VALIDATED": "1",
        **{
            RESOURCE_ENV[name]: str(value)
            for name, value in zip(
                RESOURCE_LIMITS, (64, 64, 4096, 4096, 64, 64, 8, 256, 1024)
            )
        },
        "PURE_EXPLORATION_VALIDATED_VISIBILITY_PRODUCT": str(64 * 4096),
        "PURE_EXPLORATION_VALIDATED_WHOLE_CYCLE_ELAPSED_MS": "12.5",
        "PURE_EXPLORATION_VALIDATED_WHOLE_CYCLE_PEAK_RSS_MIB": "64.5",
        "PURE_EXPLORATION_FAILURE_EVIDENCE_AUTHORITY_ENVIRONMENT": environment,
        "PURE_EXPLORATION_FAILURE_EVIDENCE_AUTHORITY_IMAGE": image,
    }
    for index, (slot, (kind, target)) in enumerate(FAILURE_SLOTS.items(), start=1):
        reached = limits[target]
        document = {
            "schema": "lunar-pure-exploration/failure-saturation-evidence/v1",
            "fixture_kind": kind,
            "configured_limits": limits,
            "target_limit_name": target,
            "reached_value": reached,
            "over_one_attempted_value": reached + 1,
            "over_one_rejected": True,
            "elapsed_ms": float(index),
            "peak_retained_rss_mib": float(index + 10),
            "authority": {"environment": environment, "image": image},
            "generated_at": "not-a-date-but-a-record",
        }
        evidence_path = tmp_path / f"{slot}.json"
        evidence_bytes = json.dumps(document, separators=(",", ":")).encode("utf-8")
        evidence_path.write_bytes(evidence_bytes)
        env_prefix = f"PURE_EXPLORATION_FAILURE_{slot.upper()}"
        profile[f"{env_prefix}_EVIDENCE_PATH"] = str(evidence_path)
        profile[f"{env_prefix}_EVIDENCE_SHA256"] = hashlib.sha256(evidence_bytes).hexdigest()
        profile[f"{env_prefix}_VALIDATED_LIMIT"] = str(reached)
        profile[f"{env_prefix}_SATURATION_ELAPSED_MS"] = str(float(index))
        profile[f"{env_prefix}_PEAK_RETAINED_RSS_MIB"] = str(float(index + 10))
        profile[f"{env_prefix}_OVER_LIMIT_REJECTED"] = "1"

    profile_path = tmp_path / "pure-exploration-profile.env"
    profile_path.write_text(
        "\n".join(f"export {key}={value}" for key, value in profile.items()) + "\n",
        encoding="utf-8",
    )
    return profile, profile_path


def _write_profile(path: Path, values: dict[str, str]) -> None:
    path.write_text(
        "\n".join(f"export {key}={value}" for key, value in values.items()) + "\n",
        encoding="utf-8",
    )


def _parameters() -> dict[str, object]:
    document = yaml.safe_load(YAML_PATH.read_text(encoding="utf-8"))
    return document["pure_exploration"]["ros__parameters"]


def test_shipped_yaml_is_only_the_approved_non_resource_defaults() -> None:
    """No tested or production resource capacity may masquerade as a default."""
    parameters = _parameters()
    assert parameters["global_occupied_threshold"] == 50
    assert parameters["maximum_task_raster_cells"] == 1048576
    assert parameters["sensor_range_m"] == 10.0
    assert parameters["sensor_fov_deg"] == 90.0
    assert parameters["yaw_offsets_deg"] == [-45.0, -22.5, 0.0, 22.5, 45.0]
    assert parameters["candidates_per_planning_batch"] == 16
    assert parameters["score_weights"] == {
        "information_gain": 0.60,
        "global_path_length": 0.30,
        "heading_change": 0.05,
        "revisit": 0.05,
    }
    assert math.isclose(sum(parameters["score_weights"].values()), 1.0)
    assert all(value >= 0 and math.isfinite(value) for value in parameters["score_weights"].values())
    assert parameters["stuck_timeout_s"] == 30.0
    assert parameters["minimum_progress_m"] == 0.2
    assert parameters["maximum_replans_per_candidate"] == 2
    assert parameters["goal_yaw_tolerance_deg"] == 11.25
    assert parameters["planner_result_timeout_s"] == 2.0
    assert all(
        math.isfinite(parameters[name]) and parameters[name] > 0
        for name in ("sensor_range_m", "sensor_fov_deg", "stuck_timeout_s", "minimum_progress_m", "goal_yaw_tolerance_deg", "planner_result_timeout_s")
    )
    assert {name: parameters[name] for name in EXPECTED_TOPICS} == EXPECTED_TOPICS
    assert all(value.startswith("/") for value in EXPECTED_TOPICS.values())
    assert not set(RESOURCE_LIMITS) & set(parameters)
    forbidden = (
        "footprint", "clearance", "platform_dimension", "frontier_length",
        "candidate_spacing", "standoff", "revisit_radius", "failure_radius",
        "planner_goal_tolerance",
    )
    text = YAML_PATH.read_text(encoding="utf-8")
    assert not any(token in text for token in forbidden)


def test_profile_validator_accepts_only_temporary_external_evidence(tmp_path: Path) -> None:
    """The isolated validator proves wiring, not an Orin capability declaration."""
    _, profile_path = _profile_fixture(tmp_path)
    result = _load_validator().validate_profile(profile_path, repository_root=REPOSITORY_ROOT)
    assert result["visibility_product"] == 64 * 4096
    assert set(result["failure_evidence"]) == set(FAILURE_SLOTS)


def test_profile_validator_rejects_a_profile_inside_the_current_repository() -> None:
    """The explicit repository root rejects a profile before any file read."""
    validator = _load_validator()
    with pytest.raises(validator.ProfileValidationError, match="profile path must be outside"):
        validator.validate_profile(
            REPOSITORY_ROOT / "pure-exploration-profile.env",
            repository_root=REPOSITORY_ROOT,
        )


def test_profile_validator_rejects_relative_profile_path() -> None:
    """A profile path cannot be made external by changing the caller CWD."""
    validator = _load_validator()
    with pytest.raises(validator.ProfileValidationError, match="profile path must be absolute"):
        validator.validate_profile(
            Path("pure-exploration-profile.env"), repository_root=REPOSITORY_ROOT
        )


def test_profile_validator_rejects_a_profile_inside_a_linked_worktree(
    tmp_path: Path,
) -> None:
    """An adjacent resolved worktree is still a repository, not external storage."""
    primary = tmp_path / "primary"
    linked = tmp_path / "linked"
    subprocess.run(["git", "init", str(primary)], check=True, capture_output=True, text=True)
    subprocess.run(["git", "-C", str(primary), "config", "user.email", "test@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(primary), "config", "user.name", "Task 14 test"], check=True)
    subprocess.run(["git", "-C", str(primary), "commit", "--allow-empty", "-m", "fixture"], check=True, capture_output=True, text=True)
    subprocess.run(["git", "-C", str(primary), "worktree", "add", "-b", "linked-profile", str(linked)], check=True, capture_output=True, text=True)
    profile, _ = _profile_fixture(tmp_path)
    linked_profile = linked / "pure-exploration-profile.env"
    _write_profile(linked_profile, profile)
    validator = _load_validator()
    with pytest.raises(validator.ProfileValidationError, match="Git repository/worktree"):
        validator.validate_profile(linked_profile, repository_root=REPOSITORY_ROOT)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("swap_measurements", "entries saturation elapsed"),
        ("stale_limit", "configured_limits"),
        ("bytes_without_digest", "sha256"),
        ("swap_path_only", "patch_per_entry evidence"),
    ],
)
def test_profile_validator_rejects_evidence_mutations(
    tmp_path: Path, mutation: str, message: str
) -> None:
    """Positive-looking stale, swapped, and unhashed evidence is never admission."""
    profile, profile_path = _profile_fixture(tmp_path)
    if mutation == "swap_measurements":
        profile["PURE_EXPLORATION_FAILURE_ENTRIES_SATURATION_ELAPSED_MS"] = profile[
            "PURE_EXPLORATION_FAILURE_PATCH_PER_ENTRY_SATURATION_ELAPSED_MS"
        ]
    elif mutation == "stale_limit":
        profile["PURE_EXPLORATION_MAXIMUM_FAILURE_TOTAL_PATCH_CELLS"] = "1023"
    elif mutation == "bytes_without_digest":
        path = Path(profile["PURE_EXPLORATION_FAILURE_ENTRIES_EVIDENCE_PATH"])
        path.write_bytes(path.read_bytes() + b" ")
    else:
        profile["PURE_EXPLORATION_FAILURE_PATCH_PER_ENTRY_EVIDENCE_PATH"] = profile[
            "PURE_EXPLORATION_FAILURE_ENTRIES_EVIDENCE_PATH"
        ]
    _write_profile(profile_path, profile)
    validator = _load_validator()
    with pytest.raises(validator.ProfileValidationError, match=re.escape(message)):
        validator.validate_profile(profile_path, repository_root=REPOSITORY_ROOT)


def test_profile_validator_treats_generated_at_as_record_only(tmp_path: Path) -> None:
    """A nonempty non-current timestamp is deliberately not a freshness gate."""
    profile, profile_path = _profile_fixture(tmp_path)
    path = Path(profile["PURE_EXPLORATION_FAILURE_ENTRIES_EVIDENCE_PATH"])
    document = json.loads(path.read_text(encoding="utf-8"))
    document["generated_at"] = "old-but-nonempty-record"
    evidence_bytes = json.dumps(document, separators=(",", ":")).encode("utf-8")
    path.write_bytes(evidence_bytes)
    profile["PURE_EXPLORATION_FAILURE_ENTRIES_EVIDENCE_SHA256"] = hashlib.sha256(evidence_bytes).hexdigest()
    _write_profile(profile_path, profile)
    _load_validator().validate_profile(profile_path, repository_root=REPOSITORY_ROOT)


def test_launch_contract_has_one_node_and_required_no_default_limits() -> None:
    """The launch surface owns required resource admission before the node starts."""
    tree = ast.parse(LAUNCH_PATH.read_text(encoding="utf-8"))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    names = [
        node.func.id if isinstance(node.func, ast.Name) else node.func.attr
        for node in calls
        if isinstance(node.func, (ast.Name, ast.Attribute))
    ]
    assert names.count("Node") == 1
    declarations = [
        node for node in calls
        if (isinstance(node.func, ast.Name) and node.func.id == "DeclareLaunchArgument")
    ]
    declared = {
        call.args[0].value: {keyword.arg: keyword.value for keyword in call.keywords}
        for call in declarations
        if call.args and isinstance(call.args[0], ast.Constant)
    }
    assert {"platform_selector", "platform_config", "exploration_config", "use_sim_time"} <= set(declared)
    source = LAUNCH_PATH.read_text(encoding="utf-8")
    assert "RESOURCE_LIMITS =" in source
    for name in RESOURCE_LIMITS:
        assert f'"{name}"' in source
    assert "DeclareLaunchArgument(name) for name in RESOURCE_LIMITS" in source
    assert "ERROR: pure exploration launch configuration requires" in source
    assert source.count("lunar_pure_exploration_ros") >= 2
    assert "lunar_pure_planner_ros" in source


def test_installed_share_ownership_is_unambiguous() -> None:
    """Explorer owns its default; planner retains only platform and planner config."""
    exploration_cmake = EXPLORATION_CMAKE_PATH.read_text(encoding="utf-8")
    planner_cmake = PLANNER_CMAKE_PATH.read_text(encoding="utf-8")
    assert "pure_exploration.yaml" in exploration_cmake
    assert "share/${PROJECT_NAME}/config" in exploration_cmake
    assert "pure_exploration.launch.py" in exploration_cmake
    assert 'DIRECTORY "${LUNAR_PURE_PLANNER_CONFIG_SOURCE_DIR}/"' in planner_cmake
    assert 'PATTERN "pure_exploration.yaml" EXCLUDE' in planner_cmake
    assert 'DIRECTORY "${LUNAR_PURE_PLANNER_LAUNCH_SOURCE_DIR}/"' in planner_cmake
    assert 'PATTERN "pure_exploration.launch.py" EXCLUDE' in planner_cmake


def test_installed_launch_has_complete_live_graph_in_no_mount_humble_container() -> None:
    """A real installed explorer exposes the complete Task14 graph without an Orin profile."""
    _run_no_mount_live_graph()

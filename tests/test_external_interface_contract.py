"""Contracts for the isolated pure-planner external ROS surface."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PURE_ROOT = REPOSITORY_ROOT
CONTRACT_PATH = PURE_ROOT / "config" / "external_interfaces.yaml"
PARAMETERS_PATH = PURE_ROOT / "config" / "pure_planner.yaml"
LAUNCH_PATH = PURE_ROOT / "launch" / "pure_planner.launch.py"

EXPECTED_INTERFACE_NAMES = {
    "global_overview": "/Car/T3/mapping/global_overview",
    "grid_map": "/Car/T3/mapping/grid_map",
    "odometry": "/Car/T3/localization/odometry",
    "tf": "/tf",
    "plan_motion": "/Car/T4/plan_motion",
    "wheeled_reference": "/Car/T4/planning/wheeled_reference",
    "wheeled_path": "/Car/T4/planning/wheeled_path",
    "wheeled_global_path": "/Car/T4/planning/wheeled_global_path",
    "wheeled_path_timing": "/Car/T4/planning/wheeled_path_timing",
    "diagnostics": "/Car/T4/planning/diagnostics",
}
FORBIDDEN_LEGACY_TOPICS = {
    "/environment/map_global",
    "/environment/map_local",
    "/localization/odometry",
    "/localization/status",
    "/mission/exploration_task",
    "/execution/motion_feedback",
}
FORBIDDEN_INPUT_KEY_FRAGMENTS = {
    "revision",
    "status",
    "/mission/",
    "feedback",
    "sqlite",
    "freshness",
    "covariance",
    "observation_age_s",
    "observation_quality",
    "elevation_variance",
    "obstacle_variance",
    "valid_mask",
    "obstacle",
    "obstacle_height",
    "forbidden",
}

PLAN_MOTION_DEFINITION = """\
# Goal declarations may be separated by comments and blank lines.
uint8 LUNAR_SURFACE=1
uint8 LAVA_TUBE=2

uint8 environment_mode
string request_id
---
# Result
uint8 planning_outcome
string reason_code
---

# Feedback
uint8 phase
float64 elapsed_s
"""


def load_contract() -> dict[str, object]:
    return yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))


def test_pure_contract_is_the_exact_task3_io_v1_surface() -> None:
    """The checked contract must match the reviewed Grid V1 ROS IO surface."""
    from tools.check_pure_planner_external_interfaces import CONTRACT

    contract = load_contract()
    assert contract["schema_version"] == "lunar-pure-planner-task3-io/v1"
    assert contract == CONTRACT
    assert contract["topics"] == {
        "global_overview": {
            "name": EXPECTED_INTERFACE_NAMES["global_overview"],
            "type": "nav_msgs/msg/OccupancyGrid",
            "owner": "external",
            "frame": "map",
            "required_fields": ["header", "info", "data"],
            "data_encoding": "int8",
        },
        "grid_map": {
            "name": EXPECTED_INTERFACE_NAMES["grid_map"],
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
            "name": EXPECTED_INTERFACE_NAMES["odometry"],
            "type": "nav_msgs/msg/Odometry",
            "owner": "external",
            "frame": "odom",
            "child_frame": "base_link",
            "required_fields": ["header", "child_frame_id", "pose", "twist"],
        },
    }
    assert contract["tf"] == {
        "name": EXPECTED_INTERFACE_NAMES["tf"],
        "type": "tf2_msgs/msg/TFMessage",
        "owner": "external",
        "chain": ["map", "odom"],
        "required_fields": ["transforms"],
    }
    assert contract["action"] == {
        "name": EXPECTED_INTERFACE_NAMES["plan_motion"],
        "type": "lunar_planning_msgs/action/PlanMotion",
        "owner": "lunar_pure_planner_ros",
        "required_goal_fields": ["environment_mode"],
    }
    assert contract["outputs"] == {
        "wheeled_reference": {
            "name": EXPECTED_INTERFACE_NAMES["wheeled_reference"],
            "type": "lunar_planning_msgs/msg/MotionReference",
            "owner": "lunar_pure_planner_ros",
            "frame": "map",
            "required_fields": [
                "header",
                "plan_id",
                "input_time",
                "platform_type",
                "path_preview",
                "trajectory",
            ],
        },
        "wheeled_path": {
            "name": EXPECTED_INTERFACE_NAMES["wheeled_path"],
            "type": "nav_msgs/msg/Path",
            "owner": "lunar_pure_planner_ros",
            "frame": "map",
            "required_fields": ["header", "poses"],
        },
        "wheeled_global_path": {
            "name": EXPECTED_INTERFACE_NAMES["wheeled_global_path"],
            "type": "nav_msgs/msg/Path",
            "owner": "lunar_pure_planner_ros",
            "frame": "map",
            "required_fields": ["header", "poses"],
        },
        "wheeled_path_timing": {
            "name": EXPECTED_INTERFACE_NAMES["wheeled_path_timing"],
            "type": "lunar_planning_msgs/msg/TimedPath",
            "owner": "lunar_pure_planner_ros",
            "frame": "map",
            "required_fields": ["path", "planning_time"],
        },
    }
    assert contract["diagnostics"] == CONTRACT["diagnostics"]


def test_checker_rejects_platform_capability_frame_as_odometry_child() -> None:
    """A platform capability reference frame must not replace upstream base_link."""
    from tools.check_pure_planner_external_interfaces import validate_external_config

    correct_contract = load_contract()
    correct_contract["topics"]["odometry"]["child_frame"] = "base_link"
    assert validate_external_config(correct_contract) == []

    mutated_contract = yaml.safe_load(yaml.safe_dump(correct_contract))
    mutated_contract["topics"]["odometry"]["child_frame"] = (
        "platform_base_frame"
    )
    assert validate_external_config(mutated_contract) == [
        "config error: document must exactly match the pure-planner v1 contract"
    ]


def test_node_parameters_and_launch_use_the_same_interface_names() -> None:
    """A contract-only topic rename must not leave the actual launch surface behind."""
    contract = load_contract()
    params = yaml.safe_load(PARAMETERS_PATH.read_text(encoding="utf-8"))["/**"][
        "ros__parameters"
    ]
    assert {
        "global_overview": params["global_map_topic"],
        "grid_map": params["local_map_topic"],
        "odometry": params["odometry_topic"],
        "tf": params["tf_topic"],
        "plan_motion": params["action_name"],
        "wheeled_reference": params["wheeled_reference_topic"],
        "wheeled_path": params["wheeled_path_topic"],
        "wheeled_global_path": params["wheeled_global_path_topic"],
        "wheeled_path_timing": params["wheeled_timed_path_topic"],
        "diagnostics": params["diagnostics_topic"],
    } == EXPECTED_INTERFACE_NAMES
    assert "pure_planner.yaml" in LAUNCH_PATH.read_text(encoding="utf-8")
    assert {
        name: contract["topics"][name]["name"]
        for name in ("global_overview", "grid_map", "odometry")
    } == {
        name: EXPECTED_INTERFACE_NAMES[name]
        for name in ("global_overview", "grid_map", "odometry")
    }


def test_pure_contract_and_node_parameters_reject_legacy_inputs() -> None:
    """Removed legacy admission inputs must not return as an optional pure setting."""
    contract = load_contract()
    input_names = {
        value["name"] for value in contract["topics"].values()
    } | {contract["tf"]["name"], contract["action"]["name"]}
    assert input_names.isdisjoint(FORBIDDEN_LEGACY_TOPICS)
    params = yaml.safe_load(PARAMETERS_PATH.read_text(encoding="utf-8"))["/**"][
        "ros__parameters"
    ]
    input_keys = set(contract["topics"]) | {"tf", "action"} | set(params)
    assert not [
        key
        for key in input_keys
        if any(fragment in key.lower() for fragment in FORBIDDEN_INPUT_KEY_FRAGMENTS)
    ]


class FakeRosRunner:
    """Minimal ROS CLI double with per-package prefix and interface responses."""

    def __init__(self, prefixes: dict[str, Path], definitions: dict[str, str]) -> None:
        self.prefixes = {package: prefix.resolve() for package, prefix in prefixes.items()}
        self.definitions = definitions

    def __call__(self, command: list[str]) -> subprocess.CompletedProcess[str]:
        if command[:3] == ["ros2", "pkg", "prefix"]:
            package = command[3]
            return subprocess.CompletedProcess(command, 0, f"{self.prefixes[package]}\n", "")
        if command[:3] == ["ros2", "interface", "show"]:
            interface_type = command[3]
            return subprocess.CompletedProcess(command, 0, self.definitions[interface_type], "")
        raise AssertionError(f"unexpected ROS command: {command}")


def make_provider(prefix: Path, package: str) -> Path:
    index = prefix / "share" / "ament_index" / "resource_index" / "packages"
    index.mkdir(parents=True)
    (index / package).write_text("", encoding="utf-8")
    return prefix.resolve()


def fake_ros_environment(tmp_path: Path) -> tuple[FakeRosRunner, Path, str]:
    packages = {
        package: tmp_path / package
        for package in {
            "nav_msgs",
            "grid_map_msgs",
            "tf2_msgs",
            "lunar_planning_msgs",
            "diagnostic_msgs",
        }
    }
    for package, prefix in packages.items():
        make_provider(prefix, package)
    definitions = {
        "nav_msgs/msg/OccupancyGrid": "std_msgs/Header header\nnav_msgs/MapMetaData info\nint8[] data\n",
        "grid_map_msgs/msg/GridMap": "std_msgs/Header header\ngrid_map_msgs/GridMapInfo info\nstring[] layers\nstring[] basic_layers\nstd_msgs/Float32MultiArray[] data\nuint16[] outer_start_index\nuint16[] inner_start_index\n",
        "nav_msgs/msg/Odometry": "std_msgs/Header header\nstring child_frame_id\ngeometry_msgs/PoseWithCovariance pose\ngeometry_msgs/TwistWithCovariance twist\n",
        "tf2_msgs/msg/TFMessage": "geometry_msgs/TransformStamped[] transforms\n",
        "lunar_planning_msgs/action/PlanMotion": PLAN_MOTION_DEFINITION,
        "lunar_planning_msgs/msg/MotionReference": (
            "std_msgs/Header header\nstring plan_id\nbuiltin_interfaces/Time input_time\n"
            "string platform_type\nnav_msgs/Path path_preview\n"
            "lunar_planning_msgs/TrajectoryPoint[] trajectory\n"
        ),
        "nav_msgs/msg/Path": "std_msgs/Header header\ngeometry_msgs/PoseStamped[] poses\n",
        "lunar_planning_msgs/msg/TimedPath": (
            "nav_msgs/Path path\nbuiltin_interfaces/Duration planning_time\n"
        ),
        "diagnostic_msgs/msg/DiagnosticArray": "std_msgs/Header header\ndiagnostic_msgs/DiagnosticStatus[] status\n",
    }
    return (
        FakeRosRunner(packages, definitions),
        packages["lunar_planning_msgs"],
        os.pathsep.join(str(prefix) for prefix in packages.values()),
    )


def test_checker_accepts_the_fixed_types_action_field_and_single_provider(tmp_path: Path) -> None:
    """A complete pure deployment must expose the approved types from one Action provider."""
    from tools.check_pure_planner_external_interfaces import check_interfaces

    runner, planning_prefix, ament_prefix_path = fake_ros_environment(tmp_path)
    package_locations: dict[str, str] = {}
    assert check_interfaces(
        CONTRACT_PATH,
        expected_lunar_planning_prefix=planning_prefix,
        run=runner,
        ament_prefix_path=ament_prefix_path,
        package_locations=package_locations,
    ) == []
    assert package_locations["lunar_planning_msgs"] == str(planning_prefix)


def test_checker_rejects_missing_environment_mode_and_wrong_action_provider(tmp_path: Path) -> None:
    """A stale shared Action or split provider must stop the cutover before launch."""
    from tools.check_pure_planner_external_interfaces import check_interfaces

    runner, planning_prefix, ament_prefix_path = fake_ros_environment(tmp_path)
    runner.definitions["lunar_planning_msgs/action/PlanMotion"] = (
        PLAN_MOTION_DEFINITION.replace("uint8 environment_mode\n", "", 1)
    )
    errors = check_interfaces(
        CONTRACT_PATH,
        expected_lunar_planning_prefix=tmp_path / "unexpected",
        run=runner,
        ament_prefix_path=ament_prefix_path,
    )
    assert any("unexpected package prefix" in error for error in errors)
    assert any("environment_mode" in error for error in errors)
    assert str(planning_prefix) in "\n".join(errors)


@pytest.mark.parametrize("target_section", ["Result", "Feedback"])
def test_checker_rejects_goal_field_moved_to_another_action_section(
    tmp_path: Path, target_section: str
) -> None:
    """A Goal field present elsewhere in the Action must not satisfy the Goal contract."""
    from tools.check_pure_planner_external_interfaces import check_interfaces

    runner, planning_prefix, ament_prefix_path = fake_ros_environment(tmp_path)
    sections = PLAN_MOTION_DEFINITION.replace(
        "uint8 environment_mode\n", "", 1
    ).split("---\n")
    target_index = {"Result": 1, "Feedback": 2}[target_section]
    sections[target_index] = "uint8 environment_mode\n" + sections[target_index]
    runner.definitions["lunar_planning_msgs/action/PlanMotion"] = "---\n".join(
        sections
    )

    errors = check_interfaces(
        CONTRACT_PATH,
        expected_lunar_planning_prefix=planning_prefix,
        run=runner,
        ament_prefix_path=ament_prefix_path,
    )
    assert any(
        f"Goal field environment_mode is declared in {target_section} section" in error
        for error in errors
    )


@pytest.mark.parametrize(
    ("declaration", "target_section"),
    [
        pytest.param("uint8 LUNAR_SURFACE=1", "Result", id="surface-in-result"),
        pytest.param("uint8 LAVA_TUBE=2", "Feedback", id="lava-tube-in-feedback"),
    ],
)
def test_checker_rejects_goal_constant_moved_to_another_action_section(
    tmp_path: Path, declaration: str, target_section: str
) -> None:
    """Goal enum declarations must not be accepted from Result or Feedback."""
    from tools.check_pure_planner_external_interfaces import check_interfaces

    runner, planning_prefix, ament_prefix_path = fake_ros_environment(tmp_path)
    sections = PLAN_MOTION_DEFINITION.replace(f"{declaration}\n", "", 1).split(
        "---\n"
    )
    target_index = {"Result": 1, "Feedback": 2}[target_section]
    sections[target_index] = f"{declaration}\n" + sections[target_index]
    runner.definitions["lunar_planning_msgs/action/PlanMotion"] = "---\n".join(
        sections
    )

    errors = check_interfaces(
        CONTRACT_PATH,
        expected_lunar_planning_prefix=planning_prefix,
        run=runner,
        ament_prefix_path=ament_prefix_path,
    )
    assert any(
        f"Goal declaration {declaration} is declared in {target_section} section"
        in error
        for error in errors
    )


def test_checker_preserves_the_exact_environment_mode_goal_declaration(
    tmp_path: Path,
) -> None:
    """A same-named Goal field with the wrong ROS type must not satisfy the contract."""
    from tools.check_pure_planner_external_interfaces import check_interfaces

    runner, planning_prefix, ament_prefix_path = fake_ros_environment(tmp_path)
    runner.definitions["lunar_planning_msgs/action/PlanMotion"] = (
        PLAN_MOTION_DEFINITION.replace(
            "uint8 environment_mode", "string environment_mode", 1
        )
    )
    errors = check_interfaces(
        CONTRACT_PATH,
        expected_lunar_planning_prefix=planning_prefix,
        run=runner,
        ament_prefix_path=ament_prefix_path,
    )
    assert any(
        "Goal section missing declaration: uint8 environment_mode" in error
        for error in errors
    )


@pytest.mark.parametrize(
    ("definition", "separator_count"),
    [
        pytest.param(
            PLAN_MOTION_DEFINITION.replace("---\n", "", 1),
            1,
            id="missing-separator",
        ),
        pytest.param(f"{PLAN_MOTION_DEFINITION}---\n", 3, id="extra-separator"),
    ],
)
def test_checker_rejects_malformed_action_section_separators(
    tmp_path: Path, definition: str, separator_count: int
) -> None:
    """An Action must contain exactly Goal, Result, and Feedback sections."""
    from tools.check_pure_planner_external_interfaces import check_interfaces

    runner, planning_prefix, ament_prefix_path = fake_ros_environment(tmp_path)
    runner.definitions["lunar_planning_msgs/action/PlanMotion"] = definition
    errors = check_interfaces(
        CONTRACT_PATH,
        expected_lunar_planning_prefix=planning_prefix,
        run=runner,
        ament_prefix_path=ament_prefix_path,
    )
    assert any(
        "malformed Action definition: expected exactly 2 section separators for "
        f"Goal/Result/Feedback, got {separator_count}" in error
        for error in errors
    )


@pytest.mark.parametrize(
    ("field", "source_section", "target_section"),
    [
        pytest.param(
            "planning_outcome", "Result", "Feedback", id="result-in-feedback"
        ),
        pytest.param("phase", "Feedback", "Result", id="feedback-in-result"),
    ],
)
def test_action_section_validator_checks_result_and_feedback_fields_when_defined(
    field: str, source_section: str, target_section: str
) -> None:
    """Future section-specific fields must be checked only in their declared section."""
    from tools.check_pure_planner_external_interfaces import _validate_action_definition

    action_contract = {
        "required_goal_fields": ["environment_mode"],
        "required_result_fields": ["planning_outcome"],
        "required_feedback_fields": ["phase"],
    }
    assert _validate_action_definition(
        "lunar_planning_msgs/action/PlanMotion",
        PLAN_MOTION_DEFINITION,
        action_contract,
    ) == []

    sections = PLAN_MOTION_DEFINITION.replace(f"uint8 {field}\n", "", 1).split(
        "---\n"
    )
    target_index = {"Result": 1, "Feedback": 2}[target_section]
    sections[target_index] = f"uint8 {field}\n" + sections[target_index]
    errors = _validate_action_definition(
        "lunar_planning_msgs/action/PlanMotion",
        "---\n".join(sections),
        action_contract,
    )
    assert any(
        f"{source_section} field {field} is declared in {target_section} section"
        in error
        for error in errors
    )

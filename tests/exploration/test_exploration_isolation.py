"""Compatibility contracts for the future pure-frontier exploration layer."""

import re
from pathlib import Path

import pytest
import yaml


PLANNER_DIAGNOSTIC_KEYS = [
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
]
EXPECTED_PLAN_MOTION_GOAL = [
    "uint8 LUNAR_SURFACE=1",
    "uint8 LAVA_TUBE=2",
    "uint8 environment_mode",
    "string request_id",
    "string mission_id",
    "uint64 mission_revision",
    "lunar_planning_msgs/GoalRegion goal",
    "bool replace_active_request",
]
PRODUCTION_SOURCE_SUFFIXES = {
    ".c",
    ".cc",
    ".cmake",
    ".cpp",
    ".h",
    ".hpp",
    ".py",
}
FORBIDDEN_EXPLORATION_DEPENDENCIES = (
    "lunar_policy_runtime",
    "lunar_policy_training",
    "lunar_pure_planner_core",
    "lunar_pure_planner_ros",
    "lunar_planner_core",
    "lunar_planner_ros",
    "luna_t3_map_adapter",
    "lunar_nav2_adapter",
    "lunar_nav2_plugin",
    "grid_map_adapter",
    "model_contract",
    "nav2_core",
    "pluginlib",
    "ppo",
)
PLATFORM_CONFIG_COPY_MARKERS = (
    "base_frame_id",
    "footprint_xy_m",
    "minimum_clearance_m",
)
ALLOWED_PLANNER_SHARE_LOOKUP = re.compile(
    r'ament_index_cpp::get_package_share_directory\(\s*"lunar_pure_planner_ros"\s*\)'
)
ALLOWED_PLANNER_CONFIG_PROVIDER = re.compile(
    r"<exec_depend>\s*lunar_pure_planner_ros\s*</exec_depend>"
)
APPROVED_PLANNER_RUNTIME_CONSUMERS = {
    "lunar_pure_exploration_ros",
    "lunar_pure_exploration_sim",
}
PACKAGE_TEST_DEPENDENCY = re.compile(
    r"<test_depend>.*?</test_depend>", re.DOTALL
)
CMAKE_IF_BUILD_TESTING = re.compile(
    r"^\s*if\s*\(\s*BUILD_TESTING\s*\)", re.IGNORECASE
)
CMAKE_IF = re.compile(r"^\s*if\s*\(", re.IGNORECASE)
CMAKE_ENDIF = re.compile(r"^\s*endif\b", re.IGNORECASE)


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def exploration_package_roots(source_root: Path) -> list[Path]:
    return sorted(
        path
        for path in source_root.glob("lunar_pure_exploration_*")
        if path.is_dir()
    )


def is_production_file(package_root: Path, source_path: Path) -> bool:
    if {"test", "tests"}.intersection(source_path.relative_to(package_root).parts):
        return False
    return (
        source_path.name in {"CMakeLists.txt", "package.xml"}
        or source_path.suffix in PRODUCTION_SOURCE_SUFFIXES
    )


def without_cmake_testing_blocks(contents: str) -> str:
    """Remove only complete if(BUILD_TESTING) ranges from a CMake source."""
    production_lines: list[str] = []
    testing_depth = 0
    for line in contents.splitlines(keepends=True):
        if testing_depth:
            if CMAKE_IF.match(line):
                testing_depth += 1
            elif CMAKE_ENDIF.match(line):
                testing_depth -= 1
            continue
        if CMAKE_IF_BUILD_TESTING.match(line):
            testing_depth = 1
            continue
        production_lines.append(line)
    return "".join(production_lines)


def production_contents(
    package_root: Path,
    source_path: Path,
    contents: str,
) -> str:
    """Retain production dependencies while admitting the reviewed config edge."""
    if source_path.name == "CMakeLists.txt":
        return without_cmake_testing_blocks(contents)
    if source_path == package_root / "package.xml":
        production_dependencies = PACKAGE_TEST_DEPENDENCY.sub("", contents)
        if package_root.name in APPROVED_PLANNER_RUNTIME_CONSUMERS:
            return ALLOWED_PLANNER_CONFIG_PROVIDER.sub("", production_dependencies)
        return production_dependencies
    if (
        package_root.name == "lunar_pure_exploration_ros"
        and source_path.suffix in PRODUCTION_SOURCE_SUFFIXES
    ):
        return ALLOWED_PLANNER_SHARE_LOOKUP.sub("", contents)
    return contents


def without_approved_global_goal_feasibility_edge(
    package_root: Path,
    source_path: Path,
    contents: str,
) -> str:
    """Remove only the reviewed explorer-to-planner feasibility dependency."""
    dependency = "lunar_pure_planner_core"
    if package_root.name != "lunar_pure_exploration_ros":
        return contents

    relative_path = source_path.relative_to(package_root).as_posix()
    if relative_path == "CMakeLists.txt":
        find_line = "find_package(lunar_pure_planner_core required)"
        dependency_line = "  lunar_pure_planner_core"
        if (
            contents.count(dependency) == 3
            and contents.splitlines().count(find_line) == 1
            and contents.splitlines().count(dependency_line) == 2
        ):
            return "\n".join(
                line
                for line in contents.splitlines()
                if line not in {find_line, dependency_line}
            )
        return contents

    if relative_path == "package.xml":
        approved_line = "  <depend>lunar_pure_planner_core</depend>"
        if contents.count(dependency) == 1 and approved_line in contents.splitlines():
            return contents.replace(approved_line, "", 1)
        return contents

    if relative_path == "src/exploration_node.cpp":
        approved_line = (
            '#include "lunar_pure_planner_core/global_goal_feasibility.hpp"'
        )
        if contents.count(dependency) == 1 and approved_line in contents.splitlines():
            return contents.replace(approved_line, "", 1)
    return contents


def scan_exploration_isolation(source_root: Path) -> tuple[list[str], list[str]]:
    config_copies: list[str] = []
    dependency_violations: list[str] = []
    for package_root in exploration_package_roots(source_root):
        for source_path in sorted(package_root.rglob("*")):
            if not source_path.is_file() or {"test", "tests"}.intersection(
                source_path.relative_to(package_root).parts
            ):
                continue

            contents = source_path.read_text(encoding="utf-8").lower()
            relative_path = source_path.relative_to(source_root).as_posix()
            if source_path.suffix in {".yaml", ".yml"}:
                copied_markers = [
                    marker
                    for marker in PLATFORM_CONFIG_COPY_MARKERS
                    if marker in contents
                ]
                if source_path.stem == "wheel" or copied_markers:
                    config_copies.append(
                        f"{relative_path}: wheel platform config {copied_markers}"
                    )

            if not is_production_file(package_root, source_path):
                continue
            contents = production_contents(
                package_root,
                source_path,
                contents,
            )
            contents = without_approved_global_goal_feasibility_edge(
                package_root,
                source_path,
                contents,
            )
            matched = [
                dependency
                for dependency in FORBIDDEN_EXPLORATION_DEPENDENCIES
                if (
                    re.search(r"\bppo\b", contents)
                    if dependency == "ppo"
                    else dependency in contents
                )
            ]
            if matched:
                dependency_violations.append(f"{relative_path}: {matched}")
    return config_copies, dependency_violations


def test_platform_config_has_one_source(repo_root: Path) -> None:
    source = repo_root / "config/wheel.yaml"
    assert source.is_file()
    assert not list((repo_root / "ros2_ws/src").glob("**/config/wheel.yaml"))


def test_wheel_geometry_contract(repo_root: Path) -> None:
    data = yaml.safe_load(
        (repo_root / "config/wheel.yaml")
        .read_text(encoding="utf-8")
    )

    def unique_value(node: object, key: str) -> list[object]:
        values: list[object] = []
        if isinstance(node, dict):
            values.extend(value for name, value in node.items() if name == key)
            for value in node.values():
                values.extend(unique_value(value, key))
        elif isinstance(node, list):
            for value in node:
                values.extend(unique_value(value, key))
        return values

    assert data["platform"] == "wheel"
    assert data["base_frame_id"] == "base_footprint"
    capability = data["capability"]
    assert capability["footprint_xy_m"] == [
        [0.591, 0.409],
        [0.591, -0.409],
        [-0.591, -0.409],
        [-0.591, 0.409],
    ]
    clearances = [
        value
        for value in unique_value(data, "minimum_clearance_m")
        if isinstance(value, (int, float))
    ]
    assert clearances == [0.2]


def test_plan_motion_freezes_environment_and_single_worker_request_contract(
    repo_root: Path,
) -> None:
    action_path = repo_root / "ros2_ws/src/lunar_planning_msgs/action/PlanMotion.action"
    goal_lines = action_path.read_text(encoding="utf-8").split("---", 1)[0].splitlines()
    assert goal_lines == EXPECTED_PLAN_MOTION_GOAL
    assert goal_lines.count("uint8 environment_mode") == 1
    assert goal_lines.count("bool replace_active_request") == 1


def test_planner_diagnostics_preserve_the_exploration_timing_subset(
    repo_root: Path,
) -> None:
    contract_path = repo_root / "config/external_interfaces.yaml"
    contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    assert contract["action"] == {
        "name": "/Car/T4/plan_motion",
        "type": "lunar_planning_msgs/action/PlanMotion",
        "owner": "lunar_pure_planner_ros",
        "required_goal_fields": ["environment_mode"],
    }
    diagnostics = contract["diagnostics"]
    expected_diagnostics = {
        "name": "/Car/T4/planning/diagnostics",
        "type": "diagnostic_msgs/msg/DiagnosticArray",
        "owner": "lunar_pure_planner_ros",
        "required_fields": ["header", "status"],
    }
    for key, value in expected_diagnostics.items():
        assert diagnostics[key] == value
    assert set(PLANNER_DIAGNOSTIC_KEYS) <= set(diagnostics["required_keys"])


def test_exploration_packages_cannot_copy_platform_config_or_link_algorithms(
    repo_root: Path,
) -> None:
    source_root = repo_root / "ros2_ws/src"
    assert exploration_package_roots(source_root)
    config_copies, dependency_violations = scan_exploration_isolation(source_root)

    assert not config_copies, (
        "exploration packages must consume, not copy, the planner-owned wheel config: "
        f"{config_copies}"
    )
    assert not dependency_violations, (
        "exploration packages must not link planner algorithms or policy code; "
        "the ROS adapter may only locate planner YAML through ament package share "
        "and an exec_depend config provider: "
        f"{dependency_violations}"
    )


def test_isolation_scanner_allows_only_reviewed_global_feasibility_edge(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "ros2_ws/src"
    package_root = source_root / "lunar_pure_exploration_ros"
    source_path = package_root / "src/exploration_node.cpp"
    source_path.parent.mkdir(parents=True)
    (package_root / "CMakeLists.txt").write_text(
        "find_package(lunar_pure_planner_core REQUIRED)\n"
        "ament_target_dependencies(node\n"
        "  lunar_pure_planner_core\n"
        ")\n"
        "ament_export_dependencies(\n"
        "  lunar_pure_planner_core\n"
        ")\n",
        encoding="utf-8",
    )
    (package_root / "package.xml").write_text(
        "  <depend>lunar_pure_planner_core</depend>\n",
        encoding="utf-8",
    )
    source_path.write_text(
        '#include "lunar_pure_planner_core/global_goal_feasibility.hpp"\n',
        encoding="utf-8",
    )

    assert scan_exploration_isolation(source_root) == ([], [])

    source_path.write_text(
        source_path.read_text(encoding="utf-8")
        + 'constexpr auto kUnexpected = "lunar_pure_planner_core";\n',
        encoding="utf-8",
    )
    assert scan_exploration_isolation(source_root) == (
        [],
        [
            "lunar_pure_exploration_ros/src/exploration_node.cpp: "
            "['lunar_pure_planner_core']"
        ],
    )


def test_isolation_scanner_allows_share_provider_and_rejects_mutations(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "pure_planner/ros2_ws/src"

    def write_file(relative_path: str, contents: str) -> None:
        path = source_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")

    write_file(
        "lunar_pure_exploration_ros/package.xml",
        "<exec_depend>lunar_pure_planner_ros</exec_depend>\n"
        "<test_depend>lunar_pure_planner_core</test_depend>\n"
        "<test_depend>lunar_pure_planner_ros</test_depend>",
    )
    write_file(
        "lunar_pure_exploration_ros/src/platform_config_loader.cpp",
        'ament_index_cpp::get_package_share_directory("lunar_pure_planner_ros");',
    )
    write_file(
        "lunar_pure_exploration_ros/CMakeLists.txt",
        "target_link_libraries(node lunar_pure_planner_core lunar_pure_planner_ros)\n"
        "if(BUILD_TESTING)\n"
        "  find_package(lunar_pure_planner_ros REQUIRED)\n"
        "  target_link_libraries(test_node lunar_pure_planner_core "
        "lunar_pure_planner_ros)\n"
        "endif()\n",
    )
    write_file(
        "lunar_pure_exploration_core/package.xml",
        "<depend>lunar_policy_runtime</depend>\n"
        "<depend>lunar_policy_training</depend>\n"
        "<depend>model_contract</depend>\n"
        "<build_depend>lunar_pure_planner_ros</build_depend>\n"
        "<build_export_depend>lunar_pure_planner_core</build_export_depend>\n"
        "<test_depend>lunar_pure_planner_core</test_depend>\n"
        "<test_depend>lunar_pure_planner_ros</test_depend>",
    )
    write_file(
        "lunar_pure_exploration_core/src/bad_loader.cpp",
        'ament_index_cpp::get_package_share_directory("lunar_pure_planner_ros");',
    )
    write_file(
        "lunar_pure_exploration_core/config/platform_copy.yaml",
        "footprint_xy_m: [[0.591, 0.409]]",
    )

    assert scan_exploration_isolation(source_root) == (
        [
            "lunar_pure_exploration_core/config/platform_copy.yaml: "
            "wheel platform config ['footprint_xy_m']"
        ],
        [
            "lunar_pure_exploration_core/package.xml: "
            "['lunar_policy_runtime', 'lunar_policy_training', "
            "'lunar_pure_planner_core', 'lunar_pure_planner_ros', "
            "'model_contract']",
            "lunar_pure_exploration_core/src/bad_loader.cpp: "
            "['lunar_pure_planner_ros']",
            "lunar_pure_exploration_ros/CMakeLists.txt: "
            "['lunar_pure_planner_core', 'lunar_pure_planner_ros']",
        ],
    )

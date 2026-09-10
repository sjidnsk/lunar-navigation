"""Physical-isolation contract for the pure planner production source tree."""

from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PURE_PLANNER_ROOT = REPOSITORY_ROOT
PRODUCTION_SOURCE = PURE_PLANNER_ROOT / "ros2_ws" / "src"
CORE_PACKAGE = PRODUCTION_SOURCE / "lunar_pure_planner_core"
ROS_PACKAGE = PRODUCTION_SOURCE / "lunar_pure_planner_ros"
CONFIGURATION_ROOT = PURE_PLANNER_ROOT / "config"
SHARED_PLAN_MOTION_ACTION = (
    REPOSITORY_ROOT / "ros2_ws" / "src" / "lunar_planning_msgs" / "action" / "PlanMotion.action"
)


def test_pure_planner_core_is_physically_isolated() -> None:
    """The copied core package must not retain old-core references or artifacts."""
    assert CORE_PACKAGE.is_dir(), "lunar_pure_planner_core package is required"

    forbidden_directories = {".git", "build", "install", "log"}
    present_forbidden_directories = sorted(
        path.relative_to(PURE_PLANNER_ROOT).as_posix()
        for package_root in (CORE_PACKAGE, ROS_PACKAGE)
        for path in package_root.rglob("*")
        if path.is_dir() and path.name in forbidden_directories
    )
    assert not present_forbidden_directories, (
        "pure planner tree contains forbidden directories: "
        f"{present_forbidden_directories}"
    )

    source_files = [
        path
        for path in CORE_PACKAGE.rglob("*")
        if path.is_file()
        and (path.suffix in {".cmake", ".cpp", ".h", ".hpp", ".py"}
             or path.name in {"CMakeLists.txt", "package.xml"})
    ]
    include_violations = []
    namespace_violations = []
    cmake_link_violations = []
    for source_file in source_files:
        contents = source_file.read_text(encoding="utf-8")
        relative_path = source_file.relative_to(CORE_PACKAGE).as_posix()
        if "#include <lunar_planner_core/" in contents:
            include_violations.append(relative_path)
        if "lunar::planning" in contents:
            namespace_violations.append(relative_path)
        if source_file.name == "CMakeLists.txt" and "lunar_planner_core" in contents:
            cmake_link_violations.append(relative_path)

    assert not include_violations, f"old core includes remain: {include_violations}"
    assert not namespace_violations, f"old planning namespace remains: {namespace_violations}"
    assert not cmake_link_violations, (
        f"old core CMake link item remains: {cmake_link_violations}"
    )


def test_shared_plan_motion_action_is_the_only_external_message_contract() -> None:
    """The shared Action has one source definition beside the planner packages."""
    assert SHARED_PLAN_MOTION_ACTION.is_file(), "shared PlanMotion Action is required"
    assert list(PRODUCTION_SOURCE.rglob("PlanMotion.action")) == [
        SHARED_PLAN_MOTION_ACTION
    ]


def test_ros_wrapper_has_fixed_capabilities_launch_and_interface_contracts() -> None:
    """The wrapper exposes three capabilities plus its two fixed runtime contracts."""
    assert ROS_PACKAGE.is_dir(), "lunar_pure_planner_ros package is required"
    assert {
        path.name for path in CONFIGURATION_ROOT.glob("*.yaml")
    } == {
        "wheel.yaml",
        "legged.yaml",
        "hopper.yaml",
        "pure_planner.yaml",
        "external_interfaces.yaml",
        "pure_exploration.yaml",
        "exploration_navigation.yaml",
        "incremental_navigation_interfaces.yaml",
    }

    forbidden_fragments = {
        "observation",
        "freshness",
        "covariance",
        "uncertainty",
        "revision",
        "mission",
    }
    for configuration_name in (
        "wheel.yaml",
        "legged.yaml",
        "hopper.yaml",
        "pure_planner.yaml",
    ):
        configuration_path = CONFIGURATION_ROOT / configuration_name
        contents = configuration_path.read_text(encoding="utf-8").lower()
        violations = sorted(
            fragment for fragment in forbidden_fragments if fragment in contents
        )
        assert not violations, (
            f"{configuration_path.name} contains excluded runtime inputs: "
            f"{violations}"
        )


def test_ros_wrapper_has_no_legacy_planner_adapter_or_tf_static_surface(
    tmp_path: Path,
) -> None:
    """Allow approved input adapters while rejecting named legacy integrations."""
    forbidden_fragments = {
        "lunar_planner_core",
        "lunar_planner_ros",
        "luna_t3_map_adapter",
        "lunar_nav2_adapter",
        "lunar_nav2_plugin",
        "grid_map_adapter",
        "nav2_core",
        "pluginlib",
        "/tf_static",
    }

    def violations_in(package_root: Path) -> list[str]:
        violations = []
        for source_path in package_root.rglob("*"):
            if not source_path.is_file():
                continue
            contents = source_path.read_text(encoding="utf-8").lower()
            matched = sorted(
                fragment for fragment in forbidden_fragments if fragment in contents
            )
            if matched:
                violations.append(
                    f"{source_path.relative_to(package_root).as_posix()}: {matched}"
                )
        return violations

    approved_task12_fixture = tmp_path / "approved_task12_inputs"
    (approved_task12_fixture / "include" / "lunar_pure_planner_ros").mkdir(
        parents=True
    )
    (approved_task12_fixture / "include" / "lunar_pure_planner_ros" / "map_adapters.hpp").write_text(
        '#include "lunar_pure_planner_ros/input_store.hpp"\n'
        'namespace lunar::pure_planner_ros { void AdaptGlobal(); }\n',
        encoding="utf-8",
    )
    (approved_task12_fixture / "include" / "lunar_pure_planner_ros" / "state_adapter.hpp").write_text(
        '#include "lunar_pure_planner_ros/input_store.hpp"\n'
        'namespace lunar::pure_planner_ros { void AdaptOdometry(); }\n',
        encoding="utf-8",
    )
    assert not violations_in(approved_task12_fixture), (
        "approved Task12 map_adapters/state_adapter inputs must stay permitted"
    )

    legacy_adapter_fixture = tmp_path / "legacy_adapter.cpp"
    legacy_adapter_fixture.write_text(
        '#include "luna_t3_map_adapter/map_adapter.hpp"\n', encoding="utf-8"
    )
    assert violations_in(tmp_path) == ["legacy_adapter.cpp: ['luna_t3_map_adapter']"]

    violations = violations_in(ROS_PACKAGE)
    assert not violations, f"ROS wrapper contains forbidden integration surface: {violations}"


def test_ros_cmake_resolves_static_configs_from_the_pure_planner_root() -> None:
    """The compiled test macro must reach pure_planner/config, not repo/config."""
    cmake_path = ROS_PACKAGE / "CMakeLists.txt"
    cmake_contents = cmake_path.read_text(encoding="utf-8")

    assert (ROS_PACKAGE.parents[2] / "config") == CONFIGURATION_ROOT
    assert "${CMAKE_CURRENT_SOURCE_DIR}/../../../config" in cmake_contents
    assert "${CMAKE_CURRENT_SOURCE_DIR}/../../../../config" not in cmake_contents

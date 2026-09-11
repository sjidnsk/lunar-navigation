"""Static launch contracts for the mutually exclusive exploration-navigation modes."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import sys
import types

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
LAUNCH = ROOT / "launch" / "exploration_navigation.launch.py"
STACK_CONFIG = ROOT / "config" / "exploration_navigation.yaml"


class _Context:
    def __init__(self, configurations: dict[str, str]) -> None:
        self.launch_configurations = configurations


def _launch_module(
    monkeypatch: pytest.MonkeyPatch,
    unavailable_packages: set[str] | None = None,
):
    """Load the real launch source with minimal action doubles, not ROS runtime."""
    unavailable = unavailable_packages or set()
    class LaunchConfiguration:
        def __init__(self, name: str) -> None:
            self.name = name

        def perform(self, context: _Context) -> str:
            return context.launch_configurations.get(self.name, "")

    class ParameterValue:
        def __init__(self, value: object, *, value_type: type[bool]) -> None:
            if not isinstance(value, value_type):
                raise TypeError(f"value={value!r} is not a {value_type!r}")
            self.value = value

    class Node:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    class IncludeLaunchDescription:
        def __init__(self, source: object, **kwargs: object) -> None:
            self.source = source
            self.launch_arguments = dict(kwargs["launch_arguments"])
            self.kwargs = kwargs

    class PythonLaunchDescriptionSource:
        def __init__(self, path: str) -> None:
            self.path = path

    class LaunchDescription:
        def __init__(self, actions: list[object]) -> None:
            self.actions = actions

    class DeclareLaunchArgument:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.args = args
            self.kwargs = kwargs

    class OpaqueFunction:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    class IfCondition:
        def __init__(self, predicate: object) -> None:
            self.predicate = predicate

    def package_share(package: str) -> str:
        if package in unavailable:
            raise LookupError(f"package unavailable: {package}")
        return str(ROOT / "ros2_ws/src" / package)

    packages = types.ModuleType("ament_index_python.packages")
    packages.get_package_share_directory = package_share
    ament_index = types.ModuleType("ament_index_python")
    launch = types.ModuleType("launch")
    launch.LaunchDescription = LaunchDescription
    launch_actions = types.ModuleType("launch.actions")
    launch_actions.DeclareLaunchArgument = DeclareLaunchArgument
    launch_actions.IncludeLaunchDescription = IncludeLaunchDescription
    launch_actions.OpaqueFunction = OpaqueFunction
    launch_conditions = types.ModuleType("launch.conditions")
    launch_conditions.IfCondition = IfCondition
    launch_sources = types.ModuleType("launch.launch_description_sources")
    launch_sources.PythonLaunchDescriptionSource = PythonLaunchDescriptionSource
    launch_substitutions = types.ModuleType("launch.substitutions")
    launch_substitutions.LaunchConfiguration = LaunchConfiguration
    launch_ros = types.ModuleType("launch_ros")
    launch_ros_actions = types.ModuleType("launch_ros.actions")
    launch_ros_actions.Node = Node
    launch_ros_substitutions = types.ModuleType("launch_ros.substitutions")
    launch_ros_substitutions.FindPackageShare = LaunchConfiguration
    launch_ros_parameters = types.ModuleType("launch_ros.parameter_descriptions")
    launch_ros_parameters.ParameterValue = ParameterValue
    for name, module in {
        "ament_index_python": ament_index,
        "ament_index_python.packages": packages,
        "launch": launch,
        "launch.actions": launch_actions,
        "launch.conditions": launch_conditions,
        "launch.launch_description_sources": launch_sources,
        "launch.substitutions": launch_substitutions,
        "launch_ros": launch_ros,
        "launch_ros.actions": launch_ros_actions,
        "launch_ros.substitutions": launch_ros_substitutions,
        "launch_ros.parameter_descriptions": launch_ros_parameters,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    spec = importlib.util.spec_from_file_location("task5_stack_launch", LAUNCH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, Node, IncludeLaunchDescription


def _compose(module: object, **overrides: str) -> list[object]:
    return module._compose(_Context({"config_file": str(STACK_CONFIG), **overrides}))


def _tree() -> ast.Module:
    return ast.parse(LAUNCH.read_text(encoding="utf-8"))


def _function(name: str) -> ast.FunctionDef:
    return next(
        node for node in _tree().body if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _keyword_value(call: ast.Call, name: str) -> str | None:
    for keyword in call.keywords:
        if keyword.arg == name and isinstance(keyword.value, ast.Constant):
            if isinstance(keyword.value.value, str):
                return keyword.value.value
    return None


def test_launch_declares_the_single_entrypoint_arguments_and_incremental_default() -> None:
    """Reject a second entrypoint or a default that silently selects legacy."""
    declarations = [
        node
        for node in ast.walk(_function("generate_launch_description"))
        if isinstance(node, ast.Call) and _call_name(node) == "DeclareLaunchArgument"
    ]
    arguments = {
        call.args[0].value: _keyword_value(call, "default_value")
        for call in declarations
        if call.args and isinstance(call.args[0], ast.Constant)
    }

    assert set(arguments) == {
        "stack_mode",
        "platform_type",
        "config_file",
        "use_sim_time",
        "start_navigation",
        "start_exploration",
        "start_rviz",
    }
    assert arguments["stack_mode"] == ""
    assert arguments["platform_type"] == ""
    assert arguments["use_sim_time"] == ""
    assert arguments["start_rviz"] == "false"
    assert arguments["start_navigation"] == "true"
    assert arguments["start_exploration"] == "true"
    assert arguments["config_file"] is None


def test_yaml_defaults_construct_only_incremental_action_server_and_share_endpoints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject dead YAML defaults or endpoint drift between the two v2 nodes."""
    module, Node, IncludeLaunchDescription = _launch_module(monkeypatch)
    actions = _compose(module)

    assert all(isinstance(action, Node) for action in actions)
    assert not any(isinstance(action, IncludeLaunchDescription) for action in actions)
    assert [action.kwargs["executable"] for action in actions] == [
        "lunar_incremental_navigation_node",
        "incremental_exploration_node",
    ]
    navigation, exploration = [action.kwargs["parameters"][0] for action in actions]
    assert navigation["platform_type"] == exploration["platform_selector"] == "wheel"
    assert navigation["action_name"] == exploration["navigation_action"] == "/Car/T4/navigation/navigate_to_pose"
    assert navigation["exploration_map_topic"] == exploration["exploration_map_topic"] == "/Car/T4/mapping/exploration_map"
    assert navigation["local_window_size_m"] == 64.0
    assert navigation["use_sim_time"].value is False
    assert exploration["use_sim_time"].value is False


def test_incremental_launch_passes_local_path_but_starts_no_controller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The controller remains an explicit operation outside stack startup."""
    module, Node, _ = _launch_module(monkeypatch)
    actions = _compose(module)

    assert all(isinstance(action, Node) for action in actions)
    navigation = actions[0].kwargs["parameters"][0]
    assert navigation["local_path_topic"] == "/Car/T4/planning/local_path"
    assert [action.kwargs["package"] for action in actions] == [
        "lunar_incremental_navigation_ros",
        "lunar_pure_exploration_ros",
    ]


def test_explicit_mode_platform_and_time_override_yaml_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject an override layer that cannot supersede the centralized defaults."""
    module, Node, IncludeLaunchDescription = _launch_module(monkeypatch)
    incremental = _compose(
        module, stack_mode="incremental_v2", platform_type="legged", use_sim_time="true"
    )
    legacy = _compose(module, stack_mode="legacy", platform_type="wheel")

    assert all(isinstance(action, Node) for action in incremental)
    incremental_navigation, incremental_exploration = [
        action.kwargs["parameters"][0] for action in incremental
    ]
    assert incremental_navigation["platform_type"] == "legged"
    assert incremental_exploration["platform_selector"] == "legged"
    assert incremental_navigation["use_sim_time"].value is True
    assert all(isinstance(action, IncludeLaunchDescription) for action in legacy)
    assert not any(isinstance(action, Node) for action in legacy)
    assert {action.source.path.rsplit("/", 1)[-1] for action in legacy} == {
        "pure_planner.launch.py",
        "pure_exploration.launch.py",
    }


def test_incremental_time_override_must_resolve_to_a_bool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject string launch values reaching a bool ParameterValue."""
    module, _, _ = _launch_module(monkeypatch)

    with pytest.raises(RuntimeError, match="use_sim_time is invalid"):
        _compose(module, use_sim_time="not-a-bool")


def test_legacy_mode_ignores_invalid_incremental_only_common_contracts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Reject an incremental QoS/frame validation that blocks explicit legacy rollback."""
    config = yaml.safe_load(STACK_CONFIG.read_text(encoding="utf-8"))
    config["common"]["exploration_map_qos"]["depth"] = 2
    config_path = tmp_path / "legacy-with-invalid-v2.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    module, Node, IncludeLaunchDescription = _launch_module(monkeypatch)

    actions = _compose(
        module, config_file=str(config_path), stack_mode="legacy", platform_type="wheel"
    )

    assert all(isinstance(action, IncludeLaunchDescription) for action in actions)
    assert not any(isinstance(action, Node) for action in actions)


def test_explicit_legacy_needs_neither_incremental_package_nor_config_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject a rollback path that resolves v2 resources before selecting legacy."""
    module, Node, IncludeLaunchDescription = _launch_module(
        monkeypatch, {"lunar_incremental_navigation_ros"}
    )

    actions = _compose(
        module,
        stack_mode="legacy",
        config_file="/definitely/missing/exploration_navigation.yaml",
    )

    assert all(isinstance(action, IncludeLaunchDescription) for action in actions)
    assert not any(isinstance(action, Node) for action in actions)
    assert {
        action.source.path.rsplit("/", 1)[-1] for action in actions
    } == {"pure_planner.launch.py", "pure_exploration.launch.py"}


def test_incremental_mode_still_requires_the_incremental_navigation_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject a legacy-only fallback from being silently used for explicit v2."""
    module, _, _ = _launch_module(
        monkeypatch, {"lunar_incremental_navigation_ros"}
    )

    with pytest.raises(LookupError, match="lunar_incremental_navigation_ros"):
        _compose(module, stack_mode="incremental_v2")


def test_default_config_is_owned_by_the_top_level_exploration_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject default-config lookup through the optional incremental package."""
    module, _, _ = _launch_module(
        monkeypatch, {"lunar_incremental_navigation_ros"}
    )

    description = module.generate_launch_description()
    config_argument = next(
        action
        for action in description.actions
        if action.args and action.args[0] == "config_file"
    )

    assert config_argument.kwargs["default_value"] == (
        f"{ROOT / 'ros2_ws/src/lunar_pure_exploration_ros'}/config/"
        "exploration_navigation.yaml"
    )


def test_invalid_mode_does_not_construct_another_action_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject implicit mode selection or an automatic recovery branch."""
    module, _, _ = _launch_module(monkeypatch)

    with pytest.raises(RuntimeError, match="unsupported stack_mode"):
        _compose(module, stack_mode="invalid")


def test_legacy_mode_includes_only_existing_planner_and_explorer_with_verified_capacities() -> None:
    """Reject legacy resource regression or direct construction of another server."""
    source = LAUNCH.read_text(encoding="utf-8")
    compose = _function("_compose")
    includes = [
        node
        for node in ast.walk(compose)
        if isinstance(node, ast.Call) and _call_name(node) == "IncludeLaunchDescription"
    ]

    assert len(includes) == 2
    assert "pure_planner.launch.py" in source
    assert "pure_exploration.launch.py" in source
    for name, value in {
        "maximum_position_probes": 8192,
        "maximum_candidate_views": 4096,
        "maximum_collision_work_units": 4194304,
        "maximum_visibility_work_units": 4096,
        "maximum_task_raster_cells": 1048576,
        "maximum_guidance_grid_cells": 1048576,
        "maximum_guidance_work_units": 8388608,
        "maximum_approach_candidates": 4096,
        "maximum_path_preview_poses": 4096,
        "maximum_executable_path_points": 4096,
        "maximum_failure_entries": 2048,
        "maximum_failure_patch_cells_per_entry": 512,
        "maximum_failure_total_patch_cells": 262144,
    }.items():
        assert f'"{name}": {value}' in source


def test_modes_are_exclusive_and_do_not_implement_runtime_fallback_or_control() -> None:
    """Reject mode hot switching, automatic fallback, dual action servers, or controllers."""
    source = LAUNCH.read_text(encoding="utf-8").lower()

    assert 'if stack_mode == "incremental_v2"' in source
    assert 'if stack_mode == "legacy"' in source
    assert "unsupported stack_mode" in source
    assert "fallback" not in source
    assert "eventhandler" not in source
    assert "lunar_pure_wheeled_controller" not in source
    assert "/car/t5/car_cmd_vel" not in source


def test_configurable_frames_and_qos(monkeypatch):
    module, _, _ = _launch_module(monkeypatch)
    config = yaml.safe_load(STACK_CONFIG.read_text())
    config['common']['frames'] = dict(map='world', odom='local_odom', base_link='robot_base')
    config['common']['local_map_qos'] = dict(reliability='best_effort', durability='volatile')
    config['common']['exploration_map_qos']['durability'] = 'volatile'
    common = module._require_common(config)
    navigation, exploration = module._incremental_parameters(config, common, 'wheel', '/tmp/wheel.yaml', False)
    for parameters in (navigation, exploration):
        assert parameters['map_frame'] == 'world'
        assert parameters['odom_frame'] == 'local_odom'
        assert parameters['base_frame'] == 'robot_base'
        assert parameters['exploration_map_qos_durability'] == 'volatile'


def test_shared_platform_profile_override(monkeypatch):
    module, _, _ = _launch_module(monkeypatch)
    config = yaml.safe_load(STACK_CONFIG.read_text())
    config["common"]["platform_config"] = "/tmp/custom-wheel.yaml"
    navigation, exploration = module._incremental_parameters(
        config, module._require_common(config), "wheel", "/default/wheel.yaml", False
    )
    assert navigation["platform_config"] == "/tmp/custom-wheel.yaml"
    assert exploration["platform_config"] == navigation["platform_config"]


def test_qos_depth_defaults_to_one_and_invalid_depth_names_field(monkeypatch):
    module, _, _ = _launch_module(monkeypatch)
    config = yaml.safe_load(STACK_CONFIG.read_text())
    del config["common"]["exploration_map_qos"]["depth"]
    navigation, exploration = module._incremental_parameters(
        config, module._require_common(config), "wheel", "/default/wheel.yaml", False
    )
    assert navigation["exploration_map_qos_depth"] == 1
    assert exploration["exploration_map_qos_depth"] == 1
    config["common"]["local_map_qos"]["depth"] = 0
    with pytest.raises(RuntimeError, match="local_map_qos"):
        module._require_common(config)

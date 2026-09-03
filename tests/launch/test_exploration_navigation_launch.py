"""Static launch contracts for the mutually exclusive exploration-navigation modes."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LAUNCH = ROOT / "launch" / "exploration_navigation.launch.py"


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

    assert set(arguments) == {"stack_mode", "platform_type", "config_file", "use_sim_time"}
    assert arguments["stack_mode"] == "incremental_v2"
    assert arguments["platform_type"] == "wheel"
    assert arguments["use_sim_time"] == "false"
    assert arguments["config_file"] is None


def test_incremental_mode_constructs_only_its_navigation_and_exploration_nodes() -> None:
    """Reject starting the legacy PlanMotion server beside NavigateToPose."""
    compose = _function("_compose")
    nodes = [node for node in ast.walk(compose) if isinstance(node, ast.Call) and _call_name(node) == "Node"]
    packages = {_keyword_value(node, "package") for node in nodes}
    executables = {_keyword_value(node, "executable") for node in nodes}

    assert packages == {
        "lunar_incremental_navigation_ros",
        "lunar_pure_exploration_ros",
    }
    assert executables == {
        "lunar_incremental_navigation_node",
        "incremental_exploration_node",
    }


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

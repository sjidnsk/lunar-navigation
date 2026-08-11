from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from launch_ros.actions import LifecycleNode


REPOSITORY = Path(__file__).resolve().parents[3]
LAUNCH_FILE = (
    REPOSITORY
    / "ros2_ws/src/lunar_navigation_config/launch"
    / "unreal_tcp_wheeled_path_planning.launch.py"
)


def _load_launch_module():
    spec = spec_from_file_location("unreal_tcp_wheeled_launch", LAUNCH_FILE)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _text(substitutions):
    return "".join(getattr(item, "text", str(item)) for item in substitutions)


def test_real_launch_graph_has_exact_minimal_nodes_and_remap():
    description = _load_launch_module().generate_launch_description()
    nodes = [entity for entity in description.entities if isinstance(entity, LifecycleNode)]
    identities = {
        (node._Node__package, node._Node__node_executable, node._Node__node_name)
        for node in nodes
    }
    assert identities == {
        (
            "lunar_unreal_tcp_bridge",
            "lunar_unreal_tcp_bridge_node",
            "lunar_unreal_tcp_bridge",
        ),
        ("lunar_observed_map", "lunar_observed_map_node", "lunar_observed_map"),
        ("lunar_planner_ros", "lunar_planner_node", "lunar_planner"),
        (
            "lunar_goal_coordinator",
            "lunar_goal_coordinator_node",
            "lunar_goal_coordinator",
        ),
    }

    planner = next(node for node in nodes if node._Node__package == "lunar_planner_ros")
    remappings = {
        (_text(source), _text(target))
        for source, target in planner._Node__remappings
    }
    assert remappings == {
        ("/localization/odometry", "/lunar/unreal/wheeled_odometry")
    }

    for node in nodes:
        parameters = node._Node__parameters
        assert any(
            any(_text(key) == "use_sim_time" and value is True for key, value in item.items())
            for item in parameters
            if isinstance(item, dict)
        )


def test_launch_graph_excludes_exploration_policy_training_and_duplicate_owner():
    description = _load_launch_module().generate_launch_description()
    nodes = [entity for entity in description.entities if isinstance(entity, LifecycleNode)]
    packages = [node._Node__package for node in nodes]
    assert "lunar_exploration_policy" not in packages
    assert "lunar_interface_v1_policy" not in packages
    assert all("training" not in package and "ppo" not in package for package in packages)
    assert packages.count("lunar_goal_coordinator") == 1

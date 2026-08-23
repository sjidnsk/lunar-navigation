"""Launch the one-node isolated pure-frontier explorer."""

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


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


def _require_resource_limits(context):
    missing = [
        name
        for name in RESOURCE_LIMITS
        if not context.launch_configurations.get(name, "").strip()
    ]
    if missing:
        raise RuntimeError(
            "ERROR: pure exploration launch configuration requires "
            + ", ".join(missing)
        )
    return []


def generate_launch_description() -> LaunchDescription:
    """Create an installed-share launch description with explicit resource limits."""
    exploration_share = get_package_share_directory("lunar_pure_exploration_ros")
    planner_share = get_package_share_directory("lunar_pure_planner_ros")
    platform_selector = LaunchConfiguration("platform_selector")
    platform_config = LaunchConfiguration("platform_config")
    exploration_config = LaunchConfiguration("exploration_config")
    use_sim_time = LaunchConfiguration("use_sim_time")
    resource_parameters = {
        name: ParameterValue(LaunchConfiguration(name), value_type=int)
        for name in RESOURCE_LIMITS
    }
    arguments = [
        DeclareLaunchArgument("platform_selector", default_value="wheel"),
        DeclareLaunchArgument(
            "platform_config",
            default_value=[planner_share, "/config/", platform_selector, ".yaml"],
        ),
        DeclareLaunchArgument(
            "exploration_config",
            default_value=f"{exploration_share}/config/pure_exploration.yaml",
        ),
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        *(DeclareLaunchArgument(name) for name in RESOURCE_LIMITS),
    ]
    return LaunchDescription(
        [
            *arguments,
            OpaqueFunction(function=_require_resource_limits),
            Node(
                package="lunar_pure_exploration_ros",
                executable="pure_exploration_node",
                name="pure_exploration",
                namespace="",
                parameters=[
                    exploration_config,
                    {
                        "platform_selector": platform_selector,
                        "platform_config": platform_config,
                        "use_sim_time": ParameterValue(use_sim_time, value_type=bool),
                        **resource_parameters,
                    },
                ],
                output="screen",
            ),
        ]
    )

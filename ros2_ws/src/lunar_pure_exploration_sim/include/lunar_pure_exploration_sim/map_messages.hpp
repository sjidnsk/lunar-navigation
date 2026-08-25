#pragma once

#include <grid_map_msgs/msg/grid_map.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <rclcpp/time.hpp>

#include "lunar_pure_exploration_sim/lunar_scene.hpp"
#include "lunar_pure_exploration_sim/visibility.hpp"

namespace lunar::pure_exploration_sim {

[[nodiscard]] nav_msgs::msg::OccupancyGrid MakeGlobalOverview(
    const LunarScene& scene, const ObservationState& observations,
    const rclcpp::Time& stamp);

[[nodiscard]] grid_map_msgs::msg::GridMap MakeLocalGridMap(
    const LunarScene& scene, const ObservationState& observations, Pose2 pose,
    const rclcpp::Time& stamp);

}  // namespace lunar::pure_exploration_sim

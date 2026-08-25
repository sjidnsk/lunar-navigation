#pragma once

#include <grid_map_msgs/msg/grid_map.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>

#include "lunar_pure_planner_core/types/planning_request.hpp"
#include "lunar_pure_planner_ros/input_store.hpp"

namespace lunar::pure_planner_ros {

[[nodiscard]] AdapterResult<lunar::pure_planning::GridMap> AdaptGlobal(
    const nav_msgs::msg::OccupancyGrid& message);

[[nodiscard]] AdapterResult<lunar::pure_planning::GridMap> AdaptLocal(
    const grid_map_msgs::msg::GridMap& message);

[[nodiscard]] AdapterResult<lunar::pure_planning::MinimalWorldSnapshot>
AdaptSnapshot(lunar::pure_planning::EnvironmentMode mode,
              const InputSnapshot& input,
              bool require_surface_global_map = true);

}  // namespace lunar::pure_planner_ros

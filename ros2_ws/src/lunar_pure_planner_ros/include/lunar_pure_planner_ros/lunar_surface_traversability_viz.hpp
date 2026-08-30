#pragma once

#include <optional>

#include <grid_map_msgs/msg/grid_map.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include "lunar_pure_planner_ros/lunar_surface_scenario.hpp"

namespace lunar::pure_planner_ros {

// Converts planner passability (1 = traversable, 0 = blocked, NaN = unknown)
// into the conventional navigation cost convention used by RViz
// (0 = free/transparent, 100 = lethal, -1 = unknown).
[[nodiscard]] std::optional<nav_msgs::msg::OccupancyGrid>
MakeTraversabilityCostmap(const grid_map_msgs::msg::GridMap& input);

// Builds the static 1 m/cell full-scene GridMap consumed by the same
// traversability core as the rolling local map.  The demo's map->odom is
// identity, while AdaptLocal intentionally requires the input frame "odom".
[[nodiscard]] grid_map_msgs::msg::GridMap MakeGlobalTraversabilityInput(
    const LunarSurfaceScenario& scenario);

[[nodiscard]] visualization_msgs::msg::MarkerArray
MakeClassicGlobalObstacleMarkers(const LunarSurfaceScenario& scenario);

// Returns a display plane below every finite scenario elevation so raw planner
// paths remain visible above all map markers in RViz's depth buffer.
[[nodiscard]] double ClassicMapBaseZ(
    const LunarSurfaceScenario& scenario) noexcept;

[[nodiscard]] std::optional<visualization_msgs::msg::MarkerArray>
MakeClassicGlobalTraversabilityMarkers(
    const LunarSurfaceScenario& scenario,
    const nav_msgs::msg::OccupancyGrid& traversability_costmap,
    double reachable_seed_x_m,
    double reachable_seed_y_m);

[[nodiscard]] std::optional<visualization_msgs::msg::MarkerArray>
MakeClassicLocalOverlayMarkers(
    const nav_msgs::msg::OccupancyGrid& obstacles,
    const nav_msgs::msg::OccupancyGrid& traversability_costmap,
    double reachable_seed_x_m,
    double reachable_seed_y_m,
    double map_base_z = 0.0);

}  // namespace lunar::pure_planner_ros

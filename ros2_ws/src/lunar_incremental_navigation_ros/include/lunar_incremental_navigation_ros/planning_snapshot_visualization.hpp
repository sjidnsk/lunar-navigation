#pragma once

#include <nav_msgs/msg/occupancy_grid.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include "lunar_incremental_navigation_core/global_guidance_snapshot.hpp"
#include "lunar_incremental_navigation_core/local_planning.hpp"
#include "lunar_incremental_navigation_core/traversability_snapshot.hpp"
#include "lunar_incremental_navigation_core/types/platform_capability.hpp"

namespace lunar::incremental_navigation_ros {

struct PlanningVisualizationWindow final {
  lunar::incremental_navigation::Point2 center_map_m;
  double length_m{};
};

struct FineVisualization final {
  nav_msgs::msg::OccupancyGrid state;
  nav_msgs::msg::OccupancyGrid cost;
  nav_msgs::msg::OccupancyGrid risk;
};

[[nodiscard]] FineVisualization ProjectFineVisualization(
    const lunar::incremental_navigation::FineTraversabilitySnapshot& snapshot,
    PlanningVisualizationWindow window, double cost_display_max);
[[nodiscard]] nav_msgs::msg::OccupancyGrid ProjectGuidanceVisualization(
    const lunar::incremental_navigation::GlobalGuidanceSnapshot& snapshot);
[[nodiscard]] visualization_msgs::msg::MarkerArray
ProjectTraversabilityVisualization(
    const lunar::incremental_navigation::FineTraversabilitySnapshot& fine,
    PlanningVisualizationWindow window, double cost_display_max,
    const lunar::incremental_navigation::GlobalGuidanceSnapshot* guidance);
[[nodiscard]] visualization_msgs::msg::MarkerArray
ProjectStartPatchVisualization(
    const lunar::incremental_navigation::RequestLocalPlanningView& view,
    double hard_inflation_radius_m);

}  // namespace lunar::incremental_navigation_ros

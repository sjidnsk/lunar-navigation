#pragma once

#include <optional>
#include <string>

#include <grid_map_msgs/msg/grid_map.hpp>

#include "lunar_pure_planner_core/types/geometry.hpp"

namespace lunar::pure_planner_ros {

// A bridge is intentionally exempt from terrain validation. Its endpoint is
// selected only from a 4-neighbour-connected region already classified as
// traversable by IncrementalTraversability.
struct TrustedBridge final {
  std::string frame_id;
  lunar::pure_planning::Vec3 endpoint_m;
};

[[nodiscard]] std::optional<TrustedBridge> FindTrustedBridge(
    const grid_map_msgs::msg::GridMap& traversability,
    const lunar::pure_planning::Vec3& start_m);

}  // namespace lunar::pure_planner_ros

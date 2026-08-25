#pragma once

#include <grid_map_msgs/msg/grid_map.hpp>

#include <optional>
#include <string_view>

namespace lunar::pure_planner_ros {

// Adds an occupancy layer without inventing obstacle observations: finite
// elevation means known ground (0.0), while NaN/Inf remains unknown (NaN).
// An existing occupancy layer is preserved unless explicitly overwritten.
[[nodiscard]] std::optional<grid_map_msgs::msg::GridMap>
AddOccupancyFromElevation(const grid_map_msgs::msg::GridMap& input,
                          std::string_view elevation_layer,
                          std::string_view occupancy_layer,
                          bool overwrite_existing_occupancy);

}  // namespace lunar::pure_planner_ros

#include "lunar_pure_planner_ros/elevation_occupancy.hpp"

#include <cmath>
#include <limits>
#include <optional>
#include <string>

namespace lunar::pure_planner_ros {

std::optional<grid_map_msgs::msg::GridMap> AddOccupancyFromElevation(
    const grid_map_msgs::msg::GridMap& input, const std::string_view elevation_layer,
    const std::string_view occupancy_layer, const bool overwrite_existing_occupancy) {
  if (elevation_layer.empty() || occupancy_layer.empty() ||
      input.layers.size() != input.data.size()) {
    return std::nullopt;
  }

  std::optional<std::size_t> elevation_index;
  std::optional<std::size_t> occupancy_index;
  for (std::size_t index = 0U; index < input.layers.size(); ++index) {
    if (input.layers[index] == elevation_layer) elevation_index = index;
    if (input.layers[index] == occupancy_layer) occupancy_index = index;
  }
  if (!elevation_index.has_value()) return std::nullopt;
  if (occupancy_index.has_value() && !overwrite_existing_occupancy) return input;

  grid_map_msgs::msg::GridMap output = input;
  std_msgs::msg::Float32MultiArray occupancy = output.data[*elevation_index];
  for (float& value : occupancy.data) {
    value = std::isfinite(value) ? 0.0F : std::numeric_limits<float>::quiet_NaN();
  }
  if (occupancy_index.has_value()) {
    output.data[*occupancy_index] = std::move(occupancy);
  } else {
    output.layers.emplace_back(occupancy_layer);
    output.data.emplace_back(std::move(occupancy));
  }
  return output;
}

}  // namespace lunar::pure_planner_ros

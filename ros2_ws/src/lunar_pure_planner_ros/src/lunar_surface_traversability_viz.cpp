#include "lunar_pure_planner_ros/lunar_surface_traversability_viz.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <queue>
#include <utility>
#include <vector>

#include <geometry_msgs/msg/point.hpp>
#include <std_msgs/msg/color_rgba.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <visualization_msgs/msg/marker.hpp>

namespace lunar::pure_planner_ros {
namespace {

[[nodiscard]] std::optional<std::size_t> CellDimension(
    const double length_m, const double resolution_m) noexcept {
  if (!std::isfinite(length_m) || !std::isfinite(resolution_m) ||
      length_m <= 0.0 || resolution_m <= 0.0) {
    return std::nullopt;
  }
  const double cells = length_m / resolution_m;
  const double rounded = std::round(cells);
  if (!std::isfinite(cells) || rounded < 1.0 ||
      std::abs(cells - rounded) > 1.0e-6 * std::max(1.0, std::abs(cells)) ||
      rounded > static_cast<double>(std::numeric_limits<std::uint32_t>::max())) {
    return std::nullopt;
  }
  return static_cast<std::size_t>(rounded);
}

[[nodiscard]] std_msgs::msg::ColorRGBA FreeColor() noexcept {
  std_msgs::msg::ColorRGBA color;
  color.r = 0.78F;
  color.g = 0.90F;
  color.b = 0.80F;
  color.a = 1.0F;
  return color;
}

[[nodiscard]] std_msgs::msg::ColorRGBA UnknownColor() noexcept {
  std_msgs::msg::ColorRGBA color;
  color.r = 0.45F;
  color.g = 0.45F;
  color.b = 0.45F;
  color.a = 0.72F;
  return color;
}

[[nodiscard]] std_msgs::msg::ColorRGBA DisconnectedColor() noexcept {
  std_msgs::msg::ColorRGBA color;
  color.r = 0.50F;
  color.g = 0.57F;
  color.b = 0.51F;
  color.a = 0.94F;
  return color;
}

[[nodiscard]] std_msgs::msg::ColorRGBA RiskColor(
    const std::int8_t cost) noexcept {
  const float severity = std::clamp(static_cast<float>(cost) / 100.0F,
                                    0.0F, 1.0F);
  std_msgs::msg::ColorRGBA color;
  color.r = 0.90F;
  color.g = 0.62F - 0.44F * severity;
  color.b = 0.05F;
  color.a = 0.78F;
  return color;
}

[[nodiscard]] visualization_msgs::msg::Marker MakeCellMarker(
    const std_msgs::msg::Header& header, const std::string& marker_namespace,
    const int id, const double resolution_m, const double z_m,
    const std::string& name) {
  visualization_msgs::msg::Marker marker;
  marker.header = header;
  marker.header.frame_id = "map";
  marker.ns = marker_namespace;
  marker.id = id;
  marker.type = visualization_msgs::msg::Marker::CUBE_LIST;
  marker.action = visualization_msgs::msg::Marker::ADD;
  marker.pose.orientation.w = 1.0;
  marker.pose.position.z = z_m;
  marker.scale.x = 1.02 * resolution_m;
  marker.scale.y = 1.02 * resolution_m;
  marker.scale.z = 0.06;
  marker.text = name;
  return marker;
}

[[nodiscard]] geometry_msgs::msg::Point GridPoint(
    const nav_msgs::msg::MapMetaData& info, const std::size_t index) {
  const std::size_t x = index % info.width;
  const std::size_t y = index / info.width;
  geometry_msgs::msg::Point point;
  point.x = info.origin.position.x +
      (static_cast<double>(x) + 0.5) * info.resolution;
  point.y = info.origin.position.y +
      (static_cast<double>(y) + 0.5) * info.resolution;
  return point;
}

// Match the planner core's conservative four-neighbour connected-component
// definition.  Cost zero is the only executable state emitted by the demo's
// binary traversability classifier; positive cost is blocked and -1 unknown.
[[nodiscard]] std::vector<std::uint8_t> ReachableSafeCells(
    const nav_msgs::msg::OccupancyGrid& costmap, const double seed_x_m,
    const double seed_y_m) {
  const std::size_t width = costmap.info.width;
  const std::size_t height = costmap.info.height;
  const std::size_t count = width * height;
  std::vector<std::uint8_t> reachable(count, 0U);
  if (!std::isfinite(seed_x_m) || !std::isfinite(seed_y_m) || width == 0U ||
      height == 0U || !std::isfinite(costmap.info.resolution) ||
      costmap.info.resolution <= 0.0F || costmap.data.size() != count) {
    return reachable;
  }

  const double relative_x =
      (seed_x_m - costmap.info.origin.position.x) / costmap.info.resolution;
  const double relative_y =
      (seed_y_m - costmap.info.origin.position.y) / costmap.info.resolution;
  if (!std::isfinite(relative_x) || !std::isfinite(relative_y) ||
      relative_x < 0.0 || relative_y < 0.0 ||
      relative_x >= static_cast<double>(width) ||
      relative_y >= static_cast<double>(height)) {
    return reachable;
  }

  const std::size_t seed_x = static_cast<std::size_t>(std::floor(relative_x));
  const std::size_t seed_y = static_cast<std::size_t>(std::floor(relative_y));
  const std::size_t seed = seed_y * width + seed_x;
  if (costmap.data[seed] != 0) return reachable;

  std::queue<std::pair<std::size_t, std::size_t>> frontier;
  reachable[seed] = 1U;
  frontier.emplace(seed_x, seed_y);
  constexpr std::array<std::pair<std::ptrdiff_t, std::ptrdiff_t>, 4U>
      kNeighbours{{{-1, 0}, {1, 0}, {0, -1}, {0, 1}}};
  while (!frontier.empty()) {
    const auto [x, y] = frontier.front();
    frontier.pop();
    for (const auto [dx, dy] : kNeighbours) {
      const auto next_x = static_cast<std::ptrdiff_t>(x) + dx;
      const auto next_y = static_cast<std::ptrdiff_t>(y) + dy;
      if (next_x < 0 || next_y < 0 ||
          next_x >= static_cast<std::ptrdiff_t>(width) ||
          next_y >= static_cast<std::ptrdiff_t>(height)) {
        continue;
      }
      const std::size_t next = static_cast<std::size_t>(next_y) * width +
                               static_cast<std::size_t>(next_x);
      if (reachable[next] != 0U || costmap.data[next] != 0) continue;
      reachable[next] = 1U;
      frontier.emplace(static_cast<std::size_t>(next_x),
                       static_cast<std::size_t>(next_y));
    }
  }
  return reachable;
}

[[nodiscard]] bool ReachableIsMajority(
    const nav_msgs::msg::OccupancyGrid& costmap,
    const std::vector<std::int8_t>& obstacles,
    const std::vector<std::uint8_t>& reachable) noexcept {
  std::size_t reachable_count = 0U;
  std::size_t disconnected_count = 0U;
  for (std::size_t index = 0U; index < costmap.data.size(); ++index) {
    if (obstacles[index] >= 50 || costmap.data[index] != 0) continue;
    if (reachable[index] != 0U) {
      ++reachable_count;
    } else {
      ++disconnected_count;
    }
  }
  return reachable_count >= disconnected_count;
}

[[nodiscard]] std_msgs::msg::Float32MultiArray WrapLogicalLayer(
    const std::vector<float>& values, const std::size_t width,
    const std::size_t height) {
  std_msgs::msg::Float32MultiArray layer;
  layer.layout.dim.resize(2U);
  layer.layout.dim[0].label = "column_index";
  layer.layout.dim[0].size = height;
  layer.layout.dim[0].stride = width * height;
  layer.layout.dim[1].label = "row_index";
  layer.layout.dim[1].size = width;
  layer.layout.dim[1].stride = width;
  layer.data.resize(values.size());
  for (std::size_t y = 0U; y < height; ++y) {
    for (std::size_t x = 0U; x < width; ++x) {
      const std::size_t logical = y * width + x;
      const std::size_t physical =
          (height - 1U - y) * width + (width - 1U - x);
      layer.data[physical] = values[logical];
    }
  }
  return layer;
}

}  // namespace

std::optional<nav_msgs::msg::OccupancyGrid> MakeTraversabilityCostmap(
    const grid_map_msgs::msg::GridMap& input) {
  const auto width = CellDimension(input.info.length_x, input.info.resolution);
  const auto height = CellDimension(input.info.length_y, input.info.resolution);
  if (input.header.frame_id.empty() || !width.has_value() || !height.has_value() ||
      *width > std::numeric_limits<std::size_t>::max() / *height ||
      input.outer_start_index >= *width || input.inner_start_index >= *height ||
      input.layers.size() != input.data.size()) {
    return std::nullopt;
  }

  std::optional<std::size_t> layer_index;
  for (std::size_t index = 0U; index < input.layers.size(); ++index) {
    if (input.layers[index] == "traversability") {
      layer_index = index;
      break;
    }
  }
  if (!layer_index.has_value()) return std::nullopt;

  const auto& layer = input.data[*layer_index];
  if (layer.layout.dim.size() != 2U) return std::nullopt;
  const auto& outer = layer.layout.dim[0];
  const auto& inner = layer.layout.dim[1];
  const bool column_major =
      outer.label == "column_index" && inner.label == "row_index" &&
      outer.size == *height && inner.size == *width;
  const bool row_major =
      outer.label == "row_index" && inner.label == "column_index" &&
      outer.size == *width && inner.size == *height;
  const std::size_t cell_count = *width * *height;
  if ((!column_major && !row_major) || outer.stride != cell_count ||
      inner.stride != inner.size || layer.layout.data_offset > layer.data.size() ||
      layer.data.size() - layer.layout.data_offset != cell_count) {
    return std::nullopt;
  }

  nav_msgs::msg::OccupancyGrid output;
  output.header = input.header;
  output.info.resolution = static_cast<float>(input.info.resolution);
  output.info.width = static_cast<std::uint32_t>(*width);
  output.info.height = static_cast<std::uint32_t>(*height);
  output.info.origin.position.x = input.info.pose.position.x - input.info.length_x / 2.0;
  output.info.origin.position.y = input.info.pose.position.y - input.info.length_y / 2.0;
  output.info.origin.position.z = input.info.pose.position.z + 0.10;
  output.info.origin.orientation = input.info.pose.orientation;
  output.data.resize(cell_count, -1);

  for (std::size_t y = 0U; y < *height; ++y) {
    for (std::size_t x = 0U; x < *width; ++x) {
      const std::size_t physical_row = (*width - 1U - x + input.outer_start_index) % *width;
      const std::size_t physical_column =
          (*height - 1U - y + input.inner_start_index) % *height;
      const std::size_t source_index = row_major
          ? physical_row * *height + physical_column
          : physical_column * *width + physical_row;
      const float passability =
          layer.data[layer.layout.data_offset + source_index];
      if (!std::isfinite(passability) || passability < 0.0F ||
          passability > 1.0F) {
        continue;
      }
      output.data[y * *width + x] = static_cast<std::int8_t>(
          std::lround((1.0F - passability) * 100.0F));
    }
  }
  return output;
}

grid_map_msgs::msg::GridMap MakeGlobalTraversabilityInput(
    const LunarSurfaceScenario& scenario) {
  std::vector<float> occupancy(scenario.occupancy.size(), 0.0F);
  for (std::size_t index = 0U; index < scenario.occupancy.size(); ++index) {
    occupancy[index] = scenario.occupancy[index] >= 50 ? 1.0F : 0.0F;
  }

  grid_map_msgs::msg::GridMap output;
  output.header.frame_id = "odom";
  output.info.resolution = scenario.resolution_m;
  output.info.length_x =
      static_cast<double>(scenario.width) * scenario.resolution_m;
  output.info.length_y =
      static_cast<double>(scenario.height) * scenario.resolution_m;
  output.info.pose.position.x = scenario.origin_x_m + 0.5 * output.info.length_x;
  output.info.pose.position.y = scenario.origin_y_m + 0.5 * output.info.length_y;
  output.info.pose.orientation.w = 1.0;
  output.layers = {"occupancy", "elevation"};
  output.basic_layers = output.layers;
  output.data = {WrapLogicalLayer(occupancy, scenario.width, scenario.height),
                 WrapLogicalLayer(scenario.elevation_m, scenario.width,
                                  scenario.height)};
  return output;
}

visualization_msgs::msg::MarkerArray MakeClassicGlobalObstacleMarkers(
    const LunarSurfaceScenario& scenario) {
  visualization_msgs::msg::MarkerArray output;
  const double map_base_z = ClassicMapBaseZ(scenario);

  visualization_msgs::msg::Marker background;
  background.header.frame_id = "map";
  background.ns = "classic_global_obstacles";
  background.id = 0;
  background.type = visualization_msgs::msg::Marker::CUBE;
  background.action = visualization_msgs::msg::Marker::ADD;
  background.pose.position.x = scenario.origin_x_m +
      0.5 * static_cast<double>(scenario.width) * scenario.resolution_m;
  background.pose.position.y = scenario.origin_y_m +
      0.5 * static_cast<double>(scenario.height) * scenario.resolution_m;
  background.pose.position.z = map_base_z;
  background.pose.orientation.w = 1.0;
  background.scale.x = static_cast<double>(scenario.width) * scenario.resolution_m;
  background.scale.y = static_cast<double>(scenario.height) * scenario.resolution_m;
  background.scale.z = 0.05;
  background.color = FreeColor();
  output.markers.push_back(std::move(background));

  visualization_msgs::msg::Marker obstacles;
  obstacles.header.frame_id = "map";
  obstacles.ns = "classic_global_obstacles";
  obstacles.id = 3;
  obstacles.type = visualization_msgs::msg::Marker::CUBE_LIST;
  obstacles.action = visualization_msgs::msg::Marker::ADD;
  obstacles.pose.orientation.w = 1.0;
  obstacles.scale.x = 1.01 * scenario.resolution_m;
  obstacles.scale.y = 1.01 * scenario.resolution_m;
  obstacles.scale.z = 0.08;
  obstacles.color.r = 0.04F;
  obstacles.color.g = 0.04F;
  obstacles.color.b = 0.04F;
  obstacles.color.a = 1.0F;
  for (std::size_t y = 0U; y < scenario.height; ++y) {
    for (std::size_t x = 0U; x < scenario.width; ++x) {
      if (scenario.occupancy[scenario.Index({x, y})] < 50) continue;
      geometry_msgs::msg::Point point;
      point.x = scenario.origin_x_m +
          (static_cast<double>(x) + 0.5) * scenario.resolution_m;
      point.y = scenario.origin_y_m +
          (static_cast<double>(y) + 0.5) * scenario.resolution_m;
      point.z = map_base_z + 0.15;
      obstacles.points.push_back(point);
    }
  }
  output.markers.push_back(std::move(obstacles));
  return output;
}

double ClassicMapBaseZ(const LunarSurfaceScenario& scenario) noexcept {
  double minimum_elevation = 0.0;
  bool found_finite = false;
  for (const float elevation : scenario.elevation_m) {
    if (!std::isfinite(elevation)) continue;
    if (!found_finite || elevation < minimum_elevation) {
      minimum_elevation = elevation;
      found_finite = true;
    }
  }
  return minimum_elevation - 2.0;
}

std::optional<visualization_msgs::msg::MarkerArray>
MakeClassicGlobalTraversabilityMarkers(
    const LunarSurfaceScenario& scenario,
    const nav_msgs::msg::OccupancyGrid& traversability_costmap,
    const double reachable_seed_x_m,
    const double reachable_seed_y_m) {
  const std::size_t cell_count = scenario.width * scenario.height;
  const bool valid_geometry =
      (traversability_costmap.header.frame_id == "map" ||
       traversability_costmap.header.frame_id == "odom") &&
      traversability_costmap.info.width == scenario.width &&
      traversability_costmap.info.height == scenario.height &&
      traversability_costmap.info.resolution == scenario.resolution_m &&
      traversability_costmap.info.origin.position.x == scenario.origin_x_m &&
      traversability_costmap.info.origin.position.y == scenario.origin_y_m &&
      traversability_costmap.data.size() == cell_count &&
      scenario.occupancy.size() == cell_count;
  if (!valid_geometry) return std::nullopt;

  auto base = MakeClassicGlobalObstacleMarkers(scenario);
  const double map_base_z = ClassicMapBaseZ(scenario);
  auto risk_cells = MakeCellMarker(
      traversability_costmap.header, "classic_global_map", 1,
      scenario.resolution_m, map_base_z + 0.12, "traversability risk");
  auto unknown_cells = MakeCellMarker(
      traversability_costmap.header, "classic_global_map", 2,
      scenario.resolution_m, map_base_z + 0.10, "unknown");
  auto connectivity_exceptions = MakeCellMarker(
      traversability_costmap.header, "classic_global_map", 3,
      scenario.resolution_m, map_base_z + 0.11, "connectivity exceptions");
  unknown_cells.color = UnknownColor();
  const auto reachable = ReachableSafeCells(
      traversability_costmap, reachable_seed_x_m, reachable_seed_y_m);
  const bool reachable_background = ReachableIsMajority(
      traversability_costmap, scenario.occupancy, reachable);
  base.markers[0].color =
      reachable_background ? FreeColor() : DisconnectedColor();
  connectivity_exceptions.color =
      reachable_background ? DisconnectedColor() : FreeColor();
  connectivity_exceptions.text =
      reachable_background ? "safe but unreachable" : "reachable safe";

  for (std::size_t index = 0U; index < cell_count; ++index) {
    if (scenario.occupancy[index] >= 50) continue;
    const std::int8_t cost = traversability_costmap.data[index];
    if (cost < 0) {
      unknown_cells.points.push_back(
          GridPoint(traversability_costmap.info, index));
      continue;
    }
    if (cost == 0) {
      if ((reachable[index] != 0U) != reachable_background) {
        connectivity_exceptions.points.push_back(
            GridPoint(traversability_costmap.info, index));
      }
      continue;
    }
    risk_cells.points.push_back(GridPoint(traversability_costmap.info, index));
    risk_cells.colors.push_back(RiskColor(cost));
  }

  visualization_msgs::msg::MarkerArray output;
  output.markers = {std::move(base.markers[0]), std::move(risk_cells),
                    std::move(unknown_cells),
                    std::move(connectivity_exceptions),
                    std::move(base.markers[1])};
  return output;
}

std::optional<visualization_msgs::msg::MarkerArray>
MakeClassicLocalOverlayMarkers(
    const nav_msgs::msg::OccupancyGrid& obstacles,
    const nav_msgs::msg::OccupancyGrid& traversability_costmap,
    const double reachable_seed_x_m,
    const double reachable_seed_y_m,
    const double map_base_z) {
  const bool same_geometry =
      obstacles.header.frame_id == traversability_costmap.header.frame_id &&
      obstacles.info.width == traversability_costmap.info.width &&
      obstacles.info.height == traversability_costmap.info.height &&
      obstacles.info.resolution == traversability_costmap.info.resolution &&
      obstacles.info.origin.position.x ==
          traversability_costmap.info.origin.position.x &&
      obstacles.info.origin.position.y ==
          traversability_costmap.info.origin.position.y;
  const std::size_t cell_count =
      static_cast<std::size_t>(obstacles.info.width) * obstacles.info.height;
  if (!same_geometry || obstacles.info.width == 0U ||
      obstacles.info.height == 0U || obstacles.info.resolution <= 0.0F ||
      obstacles.data.size() != cell_count ||
      traversability_costmap.data.size() != cell_count) {
    return std::nullopt;
  }

  visualization_msgs::msg::Marker free_background;
  free_background.header = traversability_costmap.header;
  free_background.header.frame_id = "map";
  free_background.ns = "classic_local_overlay";
  free_background.id = 0;
  free_background.type = visualization_msgs::msg::Marker::CUBE;
  free_background.action = visualization_msgs::msg::Marker::ADD;
  free_background.pose.position.x = obstacles.info.origin.position.x +
      0.5 * static_cast<double>(obstacles.info.width) * obstacles.info.resolution;
  free_background.pose.position.y = obstacles.info.origin.position.y +
      0.5 * static_cast<double>(obstacles.info.height) * obstacles.info.resolution;
  free_background.pose.position.z = map_base_z + 0.30;
  free_background.pose.orientation.w = 1.0;
  free_background.scale.x =
      static_cast<double>(obstacles.info.width) * obstacles.info.resolution;
  free_background.scale.y =
      static_cast<double>(obstacles.info.height) * obstacles.info.resolution;
  free_background.scale.z = 0.04;
  free_background.color = FreeColor();

  visualization_msgs::msg::Marker risk_cells = MakeCellMarker(
      traversability_costmap.header, "classic_local_overlay", 1,
      obstacles.info.resolution, map_base_z + 0.42, "traversability risk");
  visualization_msgs::msg::Marker unknown_cells = MakeCellMarker(
      traversability_costmap.header, "classic_local_overlay", 2,
      obstacles.info.resolution, map_base_z + 0.40, "unknown");
  visualization_msgs::msg::Marker obstacle_cells = MakeCellMarker(
      traversability_costmap.header, "classic_local_overlay", 4,
      obstacles.info.resolution, map_base_z + 0.45, "obstacles");
  visualization_msgs::msg::Marker connectivity_exceptions = MakeCellMarker(
      traversability_costmap.header, "classic_local_overlay", 3,
      obstacles.info.resolution, map_base_z + 0.41,
      "connectivity exceptions");
  obstacle_cells.color.r = 0.05F;
  obstacle_cells.color.g = 0.05F;
  obstacle_cells.color.b = 0.05F;
  obstacle_cells.color.a = 1.0F;
  unknown_cells.color = UnknownColor();
  const auto reachable = ReachableSafeCells(
      traversability_costmap, reachable_seed_x_m, reachable_seed_y_m);
  const bool reachable_background = ReachableIsMajority(
      traversability_costmap, obstacles.data, reachable);
  free_background.color =
      reachable_background ? FreeColor() : DisconnectedColor();
  connectivity_exceptions.color =
      reachable_background ? DisconnectedColor() : FreeColor();
  connectivity_exceptions.text =
      reachable_background ? "safe but unreachable" : "reachable safe";

  for (std::size_t index = 0U; index < cell_count; ++index) {
    if (obstacles.data[index] >= 50) {
      obstacle_cells.points.push_back(GridPoint(obstacles.info, index));
      continue;
    }
    const std::int8_t cost = traversability_costmap.data[index];
    if (cost < 0) {
      unknown_cells.points.push_back(GridPoint(obstacles.info, index));
      continue;
    }
    if (cost == 0) {
      if ((reachable[index] != 0U) != reachable_background) {
        connectivity_exceptions.points.push_back(
            GridPoint(obstacles.info, index));
      }
      continue;
    }
    risk_cells.points.push_back(GridPoint(obstacles.info, index));
    risk_cells.colors.push_back(RiskColor(cost));
  }

  visualization_msgs::msg::MarkerArray output;
  output.markers = {std::move(free_background), std::move(risk_cells),
                    std::move(unknown_cells),
                    std::move(connectivity_exceptions),
                    std::move(obstacle_cells)};
  return output;
}

}  // namespace lunar::pure_planner_ros

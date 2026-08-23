#include "lunar_pure_planner_ros/trusted_bridge.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <queue>
#include <utility>
#include <vector>

namespace lunar::pure_planner_ros {
namespace {

struct Dimensions final {
  std::size_t width{};
  std::size_t height{};
};

[[nodiscard]] std::optional<Dimensions> ReadDimensions(
    const grid_map_msgs::msg::GridMap& map) {
  if (map.layers.size() != 1U || map.layers.front() != "traversability" ||
      map.data.size() != 1U || map.data.front().layout.dim.size() != 2U) {
    return std::nullopt;
  }
  const auto& dimensions = map.data.front().layout.dim;
  if (dimensions[0].label != "column_index" ||
      dimensions[1].label != "row_index" || dimensions[0].size == 0U ||
      dimensions[1].size == 0U) {
    return std::nullopt;
  }
  const Dimensions result{.width = dimensions[1].size, .height = dimensions[0].size};
  if (map.data.front().data.size() != result.width * result.height ||
      !std::isfinite(map.info.resolution) || map.info.resolution <= 0.0 ||
      !std::isfinite(map.info.length_x) || !std::isfinite(map.info.length_y) ||
      !std::isfinite(map.info.pose.position.x) ||
      !std::isfinite(map.info.pose.position.y)) {
    return std::nullopt;
  }
  return result;
}

[[nodiscard]] std::size_t PhysicalIndex(const std::size_t x, const std::size_t y,
                                         const Dimensions dimensions,
                                         const std::size_t outer_start,
                                         const std::size_t inner_start) {
  const std::size_t physical_row =
      (dimensions.width - 1U - x + outer_start) % dimensions.width;
  const std::size_t physical_column =
      (dimensions.height - 1U - y + inner_start) % dimensions.height;
  return physical_column * dimensions.width + physical_row;
}

[[nodiscard]] lunar::pure_planning::Vec3 CellCenter(
    const grid_map_msgs::msg::GridMap& map, const std::size_t x,
    const std::size_t y) {
  return {.x = map.info.pose.position.x - map.info.length_x / 2.0 +
                (static_cast<double>(x) + 0.5) * map.info.resolution,
          .y = map.info.pose.position.y - map.info.length_y / 2.0 +
                (static_cast<double>(y) + 0.5) * map.info.resolution,
          .z = map.info.pose.position.z};
}

}  // namespace

std::optional<TrustedBridge> FindTrustedBridge(
    const grid_map_msgs::msg::GridMap& traversability,
    const lunar::pure_planning::Vec3& start_m) {
  const auto dimensions = ReadDimensions(traversability);
  if (!dimensions.has_value() || traversability.header.frame_id.empty() ||
      !std::isfinite(start_m.x) || !std::isfinite(start_m.y) ||
      !std::isfinite(start_m.z)) {
    return std::nullopt;
  }

  const auto& values = traversability.data.front().data;
  const auto is_free = [&](const std::size_t x, const std::size_t y) {
    return values[PhysicalIndex(x, y, *dimensions,
                                traversability.outer_start_index,
                                traversability.inner_start_index)] == 1.0F;
  };
  const std::size_t cell_count = dimensions->width * dimensions->height;
  std::vector<bool> visited(cell_count, false);
  std::optional<TrustedBridge> nearest;
  double nearest_distance_squared = std::numeric_limits<double>::infinity();

  for (std::size_t y = 0U; y < dimensions->height; ++y) {
    for (std::size_t x = 0U; x < dimensions->width; ++x) {
      const std::size_t first = y * dimensions->width + x;
      if (visited[first] || !is_free(x, y)) {
        continue;
      }
      std::queue<std::pair<std::size_t, std::size_t>> frontier;
      std::vector<std::pair<std::size_t, std::size_t>> component;
      frontier.emplace(x, y);
      visited[first] = true;
      while (!frontier.empty()) {
        const auto [cell_x, cell_y] = frontier.front();
        frontier.pop();
        component.emplace_back(cell_x, cell_y);
        constexpr std::pair<std::ptrdiff_t, std::ptrdiff_t> kNeighbours[] = {
            {-1, 0}, {1, 0}, {0, -1}, {0, 1}};
        for (const auto [dx, dy] : kNeighbours) {
          const std::ptrdiff_t next_x = static_cast<std::ptrdiff_t>(cell_x) + dx;
          const std::ptrdiff_t next_y = static_cast<std::ptrdiff_t>(cell_y) + dy;
          if (next_x < 0 || next_y < 0 ||
              next_x >= static_cast<std::ptrdiff_t>(dimensions->width) ||
              next_y >= static_cast<std::ptrdiff_t>(dimensions->height)) {
            continue;
          }
          const std::size_t next = static_cast<std::size_t>(next_y) *
                                       dimensions->width +
                                   static_cast<std::size_t>(next_x);
          if (!visited[next] && is_free(static_cast<std::size_t>(next_x),
                                        static_cast<std::size_t>(next_y))) {
            visited[next] = true;
            frontier.emplace(static_cast<std::size_t>(next_x),
                             static_cast<std::size_t>(next_y));
          }
        }
      }
      if (component.size() < 2U) {
        continue;
      }
      for (const auto [cell_x, cell_y] : component) {
        const auto center = CellCenter(traversability, cell_x, cell_y);
        const double distance_squared =
            std::pow(center.x - start_m.x, 2.0) +
            std::pow(center.y - start_m.y, 2.0);
        if (distance_squared < nearest_distance_squared) {
          nearest_distance_squared = distance_squared;
          nearest = TrustedBridge{.frame_id = traversability.header.frame_id,
                                  .endpoint_m = center};
        }
      }
    }
  }
  return nearest;
}

}  // namespace lunar::pure_planner_ros

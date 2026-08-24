#include "lunar_pure_planner_ros/incremental_traversability.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <std_msgs/msg/float32_multi_array.hpp>

#include "lunar_pure_planner_core/types/world_snapshot.hpp"
#include "lunar_pure_planner_ros/center_distance_transform.hpp"
#include "lunar_pure_planner_ros/map_adapters.hpp"

namespace lunar::pure_planner_ros {
namespace {

[[nodiscard]] bool SameValue(const float left, const float right) noexcept {
  return left == right || (std::isnan(left) && std::isnan(right));
}

[[nodiscard]] bool ValidValue(const float value) noexcept {
  return std::isfinite(value);
}

[[nodiscard]] std_msgs::msg::Float32MultiArray WrapLayer(
    const std::vector<float>& values, const std::size_t width,
    const std::size_t height, const std::size_t outer_start,
    const std::size_t inner_start) {
  std_msgs::msg::Float32MultiArray layer;
  layer.layout.dim.resize(2);
  layer.layout.dim[0].label = "column_index";
  layer.layout.dim[0].size = height;
  layer.layout.dim[0].stride = width * height;
  layer.layout.dim[1].label = "row_index";
  layer.layout.dim[1].size = width;
  layer.layout.dim[1].stride = width;
  layer.data.resize(values.size());
  for (std::size_t y = 0; y < height; ++y) {
    for (std::size_t x = 0; x < width; ++x) {
      const std::size_t physical_row = (width - 1U - x + outer_start) % width;
      const std::size_t physical_column =
          (height - 1U - y + inner_start) % height;
      layer.data[physical_column * width + physical_row] = values[y * width + x];
    }
  }
  return layer;
}

[[nodiscard]] bool SameGeometry(
    const lunar::pure_planning::GridMap& map, const bool initialized,
    const std::string& frame_id, const std::size_t width,
    const std::size_t height, const double resolution_m,
    const double origin_x_m, const double origin_y_m,
    const double origin_z_m) noexcept {
  return initialized && map.frame_id == frame_id && map.width == width &&
         map.height == height && map.resolution_m == resolution_m &&
         map.origin_m.x == origin_x_m && map.origin_m.y == origin_y_m &&
         map.origin_m.z == origin_z_m;
}

}  // namespace

TraversabilityProfile MakeTraversabilityProfile(
    const lunar::pure_planning::PlatformCapability& capability,
    const double occupancy_threshold) {
  return std::visit(
      [occupancy_threshold](const auto& typed) -> TraversabilityProfile {
        using Capability = std::decay_t<decltype(typed)>;
        if constexpr (std::is_same_v<Capability,
                                     lunar::pure_planning::WheeledCapability>) {
          double footprint_radius{};
          for (const auto& point : typed.footprint_xy_m) {
            footprint_radius = std::max(footprint_radius,
                                        std::hypot(point.x, point.y));
          }
          return {.support_radius_m = footprint_radius + typed.minimum_clearance_m,
                  .maximum_slope_rad = typed.maximum_slope_rad,
                  .occupancy_threshold = occupancy_threshold};
        } else if constexpr (std::is_same_v<Capability,
                                            lunar::pure_planning::LeggedCapability>) {
          return {.support_radius_m =
                      std::hypot(typed.body_extent_m.x, typed.body_extent_m.y) / 2.0 +
                      typed.minimum_body_clearance_m,
                  .maximum_slope_rad = typed.maximum_slope_rad,
                  .occupancy_threshold = occupancy_threshold};
        } else {
          return {.support_radius_m = typed.landing_support_radius_m +
                                       typed.landing_lateral_margin_m,
                  .maximum_slope_rad = typed.maximum_landing_slope_rad,
                  .occupancy_threshold = occupancy_threshold};
        }
      },
      capability);
}

IncrementalTraversability::IncrementalTraversability(TraversabilityProfile profile)
    : profile_(std::move(profile)) {}

std::optional<TraversabilityUpdate> IncrementalTraversability::Update(
    const grid_map_msgs::msg::GridMap& local_map) {
  if (!std::isfinite(profile_.support_radius_m) || profile_.support_radius_m < 0.0 ||
      !std::isfinite(profile_.maximum_slope_rad) ||
      profile_.maximum_slope_rad < 0.0 ||
      !std::isfinite(profile_.occupancy_threshold) ||
      profile_.occupancy_threshold < 0.0 || profile_.occupancy_threshold > 1.0) {
    return std::nullopt;
  }
  const auto adapted = AdaptLocal(local_map);
  if (!adapted.value.has_value()) {
    return std::nullopt;
  }
  const auto* occupancy_layer = std::get_if<std::vector<float>>(
      &adapted.value->layers.at("occupancy").values);
  const auto* elevation_layer = std::get_if<std::vector<float>>(
      &adapted.value->layers.at("elevation").values);
  if (occupancy_layer == nullptr || elevation_layer == nullptr) {
    return std::nullopt;
  }
  const auto& map = *adapted.value;
  const bool full_rebuild = !SameGeometry(
      map, initialized_, frame_id_, width_, height_, resolution_m_, origin_x_m_,
      origin_y_m_, origin_z_m_);
  if (full_rebuild) {
    frame_id_ = map.frame_id;
    width_ = map.width;
    height_ = map.height;
    resolution_m_ = map.resolution_m;
    origin_x_m_ = map.origin_m.x;
    origin_y_m_ = map.origin_m.y;
    origin_z_m_ = map.origin_m.z;
    occupancy_ = *occupancy_layer;
    elevation_ = *elevation_layer;
    traversability_.assign(map.CellCount(), std::numeric_limits<float>::quiet_NaN());
    initialized_ = true;
  }

  std::vector<bool> dirty(map.CellCount(), full_rebuild);
  if (!full_rebuild) {
    const std::size_t halo = static_cast<std::size_t>(std::ceil(
        profile_.support_radius_m / map.resolution_m)) + 1U;
    for (std::size_t index = 0; index < map.CellCount(); ++index) {
      if (SameValue(occupancy_[index], (*occupancy_layer)[index]) &&
          SameValue(elevation_[index], (*elevation_layer)[index])) {
        continue;
      }
      const std::size_t center_x = index % map.width;
      const std::size_t center_y = index / map.width;
      const std::size_t x_min = center_x > halo ? center_x - halo : 0U;
      const std::size_t y_min = center_y > halo ? center_y - halo : 0U;
      const std::size_t x_max = std::min(map.width - 1U, center_x + halo);
      const std::size_t y_max = std::min(map.height - 1U, center_y + halo);
      for (std::size_t y = y_min; y <= y_max; ++y) {
        for (std::size_t x = x_min; x <= x_max; ++x) {
          dirty[y * map.width + x] = true;
        }
      }
    }
    occupancy_ = *occupancy_layer;
    elevation_ = *elevation_layer;
  }

  const std::size_t support_cells = static_cast<std::size_t>(std::ceil(
      profile_.support_radius_m / map.resolution_m));
  const auto slope_classification = [this, &map](const std::size_t x,
                                                  const std::size_t y) {
    const auto derivative = [this, &map, x, y](const bool x_axis)
        -> std::optional<double> {
      const std::ptrdiff_t coordinate = static_cast<std::ptrdiff_t>(x_axis ? x : y);
      const std::ptrdiff_t limit = static_cast<std::ptrdiff_t>(x_axis ? map.width : map.height);
      const auto at = [this, &map, x_axis, x, y](const std::ptrdiff_t value) {
        return elevation_[(x_axis ? y : static_cast<std::size_t>(value)) * map.width +
                          (x_axis ? static_cast<std::size_t>(value) : x)];
      };
      const bool has_low = coordinate > 0 && ValidValue(at(coordinate - 1));
      const bool has_high = coordinate + 1 < limit && ValidValue(at(coordinate + 1));
      if (has_low && has_high) {
        return (static_cast<double>(at(coordinate + 1)) -
                static_cast<double>(at(coordinate - 1))) / (2.0 * map.resolution_m);
      }
      if (has_high) {
        return (static_cast<double>(at(coordinate + 1)) -
                static_cast<double>(elevation_[y * map.width + x])) / map.resolution_m;
      }
      if (has_low) {
        return (static_cast<double>(elevation_[y * map.width + x]) -
                static_cast<double>(at(coordinate - 1))) / map.resolution_m;
      }
      return std::nullopt;
    };
    const auto slope_x = derivative(true);
    const auto slope_y = derivative(false);
    if (!slope_x.has_value() || !slope_y.has_value()) {
      return std::numeric_limits<float>::quiet_NaN();
    }
    const double slope = std::atan(std::hypot(*slope_x, *slope_y));
    return slope > profile_.maximum_slope_rad ? 0.0F : 1.0F;
  };
  const auto classify = [this, &map, support_cells,
                         &slope_classification](const std::size_t x,
                                                const std::size_t y) {
    const std::size_t center = y * map.width + x;
    if (!ValidValue(occupancy_[center]) || !ValidValue(elevation_[center]) ||
        occupancy_[center] < 0.0F || occupancy_[center] > 1.0F) {
      return std::numeric_limits<float>::quiet_NaN();
    }
    for (std::ptrdiff_t dy = -static_cast<std::ptrdiff_t>(support_cells);
         dy <= static_cast<std::ptrdiff_t>(support_cells); ++dy) {
      for (std::ptrdiff_t dx = -static_cast<std::ptrdiff_t>(support_cells);
           dx <= static_cast<std::ptrdiff_t>(support_cells); ++dx) {
        const double distance = std::hypot(static_cast<double>(dx),
                                           static_cast<double>(dy)) * map.resolution_m;
        if (distance > profile_.support_radius_m + 1.0e-9) continue;
        const std::ptrdiff_t neighbour_x = static_cast<std::ptrdiff_t>(x) + dx;
        const std::ptrdiff_t neighbour_y = static_cast<std::ptrdiff_t>(y) + dy;
        if (neighbour_x < 0 || neighbour_y < 0 ||
            neighbour_x >= static_cast<std::ptrdiff_t>(map.width) ||
            neighbour_y >= static_cast<std::ptrdiff_t>(map.height)) {
          return std::numeric_limits<float>::quiet_NaN();
        }
        const std::size_t neighbour =
            static_cast<std::size_t>(neighbour_y) * map.width +
            static_cast<std::size_t>(neighbour_x);
        if (!ValidValue(occupancy_[neighbour]) || !ValidValue(elevation_[neighbour]) ||
            occupancy_[neighbour] < 0.0F || occupancy_[neighbour] > 1.0F) {
          return std::numeric_limits<float>::quiet_NaN();
        }
        if (occupancy_[neighbour] >= profile_.occupancy_threshold) return 0.0F;
      }
    }
    return slope_classification(x, y);
  };

  std::vector<bool> requires_neighbour_scan = dirty;
  if (full_rebuild) {
    std::vector<std::uint8_t> hazards(map.CellCount(), 0U);
    for (std::size_t index = 0U; index < map.CellCount(); ++index) {
      const float occupancy = occupancy_[index];
      const float elevation = elevation_[index];
      hazards[index] = static_cast<std::uint8_t>(
          !ValidValue(occupancy) || !ValidValue(elevation) ||
          occupancy < 0.0F || occupancy > 1.0F ||
          occupancy >= profile_.occupancy_threshold);
    }
    const auto distance = BuildCenterSquaredDistance(map.width, map.height, hazards);
    if (distance.ok()) {
      const double radius_cells = profile_.support_radius_m / map.resolution_m;
      const double radius_with_tolerance = radius_cells + 1.0e-9 / map.resolution_m;
      const double squared_radius = radius_with_tolerance * radius_with_tolerance;
      for (std::size_t y = support_cells; y + support_cells < map.height; ++y) {
        for (std::size_t x = support_cells; x + support_cells < map.width; ++x) {
          const std::size_t index = y * map.width + x;
          requires_neighbour_scan[index] =
              distance.squared_cells[index] <= squared_radius;
        }
      }
    }
  }

  std::size_t recomputed_cells{};
  for (std::size_t index = 0; index < map.CellCount(); ++index) {
    if (!dirty[index]) continue;
    const std::size_t x = index % map.width;
    const std::size_t y = index / map.width;
    traversability_[index] = requires_neighbour_scan[index]
        ? classify(x, y)
        : slope_classification(x, y);
    ++recomputed_cells;
  }
  grid_map_msgs::msg::GridMap output;
  output.header = local_map.header;
  output.info = local_map.info;
  output.outer_start_index = local_map.outer_start_index;
  output.inner_start_index = local_map.inner_start_index;
  output.layers = {"traversability"};
  output.basic_layers = {"traversability"};
  output.data = {WrapLayer(traversability_, map.width, map.height,
                           output.outer_start_index, output.inner_start_index)};
  return TraversabilityUpdate{.map = std::move(output),
                              .full_rebuild = full_rebuild,
                              .recomputed_cells = recomputed_cells};
}

}  // namespace lunar::pure_planner_ros

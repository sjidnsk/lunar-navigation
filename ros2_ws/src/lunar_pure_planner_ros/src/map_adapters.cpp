#include "lunar_pure_planner_ros/map_adapters.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "lunar_pure_planner_ros/state_adapter.hpp"

namespace lunar::pure_planner_ros {
namespace {

constexpr char kInvalidInput[] = "INVALID_INPUT";

[[nodiscard]] bool Finite(const double value) noexcept {
  return std::isfinite(value);
}

[[nodiscard]] bool AxisAlignedQuaternion(
    const geometry_msgs::msg::Quaternion& value) noexcept {
  if (!Finite(value.w) || !Finite(value.x) || !Finite(value.y) || !Finite(value.z)) {
    return false;
  }
  const double norm = std::sqrt(value.w * value.w + value.x * value.x +
                                value.y * value.y + value.z * value.z);
  constexpr double kTolerance = 1.0e-6;
  return Finite(norm) && norm > 0.0 &&
         std::abs(value.x / norm) <= kTolerance &&
         std::abs(value.y / norm) <= kTolerance &&
         std::abs(value.z / norm) <= kTolerance &&
         std::abs(std::abs(value.w / norm) - 1.0) <= kTolerance;
}

[[nodiscard]] bool SafeProduct(const std::size_t left, const std::size_t right,
                               std::size_t* result) noexcept {
  if (left == 0U || right == 0U ||
      left > std::numeric_limits<std::size_t>::max() / right) {
    return false;
  }
  *result = left * right;
  return true;
}

[[nodiscard]] lunar::pure_planning::TimePoint Stamp(
    const builtin_interfaces::msg::Time& stamp) noexcept {
  return {.nanoseconds_since_epoch =
              static_cast<std::int64_t>(stamp.sec) * 1'000'000'000LL + stamp.nanosec};
}

template <typename T>
[[nodiscard]] AdapterResult<T> Invalid() {
  return {.value = std::nullopt, .reason_code = kInvalidInput};
}

[[nodiscard]] bool ValidPose(const geometry_msgs::msg::Pose& pose) noexcept {
  return Finite(pose.position.x) && Finite(pose.position.y) &&
         Finite(pose.position.z) && AxisAlignedQuaternion(pose.orientation);
}

struct LayerLayout final {
  bool row_major{};
  std::size_t data_offset{};
};

[[nodiscard]] std::optional<std::size_t> CellDimension(
    const double length, const double resolution) noexcept {
  if (!Finite(length) || !Finite(resolution) || length <= 0.0 || resolution <= 0.0) {
    return std::nullopt;
  }
  const double cells = length / resolution;
  const double rounded = std::round(cells);
  const double tolerance = 1.0e-6 * std::max(1.0, std::abs(cells));
  if (!Finite(cells) || rounded < 1.0 || std::abs(cells - rounded) > tolerance ||
      rounded > static_cast<double>(std::numeric_limits<std::uint32_t>::max())) {
    return std::nullopt;
  }
  return static_cast<std::size_t>(rounded);
}

[[nodiscard]] std::optional<LayerLayout> ParseLayout(
    const std_msgs::msg::Float32MultiArray& layer, const std::size_t width,
    const std::size_t height) noexcept {
  if (layer.layout.dim.size() != 2U ||
      width > std::numeric_limits<std::size_t>::max() / height) {
    return std::nullopt;
  }
  const auto& outer = layer.layout.dim[0];
  const auto& inner = layer.layout.dim[1];
  bool row_major = false;
  if (outer.label == "column_index" && inner.label == "row_index") {
    if (outer.size != height || inner.size != width) {
      return std::nullopt;
    }
  } else if (outer.label == "row_index" && inner.label == "column_index") {
    row_major = true;
    if (outer.size != width || inner.size != height) {
      return std::nullopt;
    }
  } else {
    return std::nullopt;
  }
  const std::size_t cell_count = width * height;
  if (outer.stride != cell_count || inner.stride != inner.size ||
      layer.layout.data_offset > layer.data.size() ||
      layer.data.size() - layer.layout.data_offset != cell_count) {
    return std::nullopt;
  }
  return LayerLayout{.row_major = row_major, .data_offset = layer.layout.data_offset};
}

[[nodiscard]] std::optional<std::vector<float>> UnwrapLayer(
    const std_msgs::msg::Float32MultiArray& layer, const LayerLayout& layout,
    const std::size_t width, const std::size_t height,
    const std::size_t outer_start, const std::size_t inner_start) {
  if (outer_start >= width || inner_start >= height) {
    return std::nullopt;
  }
  std::size_t count{};
  if (!SafeProduct(width, height, &count)) {
    return std::nullopt;
  }
  std::vector<float> result(count);
  for (std::size_t y = 0U; y < height; ++y) {
    for (std::size_t x = 0U; x < width; ++x) {
      const std::size_t physical_row = (width - 1U - x + outer_start) % width;
      const std::size_t physical_column =
          (height - 1U - y + inner_start) % height;
      const std::size_t source_index = layout.row_major
          ? physical_row * height + physical_column
          : physical_column * width + physical_row;
      result[y * width + x] = layer.data[layout.data_offset + source_index];
    }
  }
  return result;
}

}  // namespace

AdapterResult<lunar::pure_planning::GridMap> AdaptGlobal(
    const nav_msgs::msg::OccupancyGrid& message) {
  std::size_t count{};
  if (message.header.frame_id != "map" || message.info.width == 0U || message.info.height == 0U ||
      !Finite(message.info.resolution) || message.info.resolution <= 0.0F ||
      !SafeProduct(message.info.width, message.info.height, &count) ||
      message.data.size() != count || !ValidPose(message.info.origin)) {
    return Invalid<lunar::pure_planning::GridMap>();
  }
  lunar::pure_planning::GridMap result;
  result.frame_id = message.header.frame_id;
  result.stamp = Stamp(message.header.stamp);
  result.width = message.info.width;
  result.height = message.info.height;
  result.resolution_m = message.info.resolution;
  result.origin_m = {.x = message.info.origin.position.x,
                     .y = message.info.origin.position.y,
                     .z = message.info.origin.position.z};
  result.layers.emplace("occupancy", lunar::pure_planning::GridLayer{
      .values = std::vector<std::int8_t>{message.data.begin(), message.data.end()}});
  return {.value = std::move(result), .reason_code = {}};
}

AdapterResult<lunar::pure_planning::GridMap> AdaptLocal(
    const grid_map_msgs::msg::GridMap& message) {
  const auto width = CellDimension(message.info.length_x, message.info.resolution);
  const auto height = CellDimension(message.info.length_y, message.info.resolution);
  if (message.header.frame_id != "odom" || !width.has_value() || !height.has_value() ||
      !ValidPose(message.info.pose) || message.layers.size() != message.data.size()) {
    return Invalid<lunar::pure_planning::GridMap>();
  }
  std::optional<std::size_t> occupancy_index;
  std::optional<std::size_t> elevation_index;
  for (std::size_t index = 0U; index < message.layers.size(); ++index) {
    if (message.layers[index] == "occupancy") occupancy_index = index;
    if (message.layers[index] == "elevation") elevation_index = index;
  }
  if (!occupancy_index.has_value() || !elevation_index.has_value()) {
    return Invalid<lunar::pure_planning::GridMap>();
  }
  const auto occupancy_layout =
      ParseLayout(message.data[*occupancy_index], *width, *height);
  const auto elevation_layout =
      ParseLayout(message.data[*elevation_index], *width, *height);
  if (!occupancy_layout.has_value() || !elevation_layout.has_value() ||
      message.outer_start_index >= *width || message.inner_start_index >= *height) {
    return Invalid<lunar::pure_planning::GridMap>();
  }
  const auto occupancy = UnwrapLayer(message.data[*occupancy_index], *occupancy_layout,
                                     *width, *height, message.outer_start_index,
                                     message.inner_start_index);
  const auto elevation = UnwrapLayer(message.data[*elevation_index], *elevation_layout,
                                     *width, *height, message.outer_start_index,
                                     message.inner_start_index);
  if (!occupancy.has_value() || !elevation.has_value()) {
    return Invalid<lunar::pure_planning::GridMap>();
  }
  lunar::pure_planning::GridMap result;
  result.frame_id = message.header.frame_id;
  result.stamp = Stamp(message.header.stamp);
  result.width = *width;
  result.height = *height;
  result.resolution_m = message.info.resolution;
  result.origin_m = {.x = message.info.pose.position.x - message.info.length_x / 2.0,
                     .y = message.info.pose.position.y - message.info.length_y / 2.0,
                     .z = message.info.pose.position.z};
  result.layers.emplace("occupancy", lunar::pure_planning::GridLayer{
      .values = std::move(*occupancy)});
  result.layers.emplace("elevation", lunar::pure_planning::GridLayer{
      .values = std::move(*elevation)});
  return {.value = std::move(result), .reason_code = {}};
}

AdapterResult<lunar::pure_planning::MinimalWorldSnapshot> AdaptSnapshot(
    const lunar::pure_planning::EnvironmentMode mode, const InputSnapshot& input,
    const bool require_surface_global_map) {
  if (!input.local_map || !input.odometry || !input.map_from_odom.has_value()) {
    return Invalid<lunar::pure_planning::MinimalWorldSnapshot>();
  }
  const auto local = AdaptLocal(*input.local_map);
  const auto transform = AdaptDirectMapFromOdom(*input.map_from_odom);
  const auto odometry = AdaptOdometry(
      *input.odometry, lunar::pure_planning::PlatformType::kWheeled);
  if (!local.value.has_value() || !transform.value.has_value() ||
      !odometry.value.has_value()) {
    return Invalid<lunar::pure_planning::MinimalWorldSnapshot>();
  }
  lunar::pure_planning::MinimalWorldSnapshot result{
      .local_map = *local.value,
      .map_from_odom = *transform.value,
      .global_map_sequence = input.global_sequence,
      .local_map_sequence = input.local_sequence,
      .odometry_sequence = input.odometry_sequence,
      .tf_sequence = input.tf_sequence,
  };
  if (mode == lunar::pure_planning::EnvironmentMode::kLunarSurface) {
    if (!input.global_map && require_surface_global_map) {
      return Invalid<lunar::pure_planning::MinimalWorldSnapshot>();
    }
    if (input.global_map) {
      const auto global = AdaptGlobal(*input.global_map);
      if (!global.value.has_value()) {
        return Invalid<lunar::pure_planning::MinimalWorldSnapshot>();
      }
      result.global_map = std::move(*global.value);
    }
  } else if (mode != lunar::pure_planning::EnvironmentMode::kLavaTube) {
    return Invalid<lunar::pure_planning::MinimalWorldSnapshot>();
  }
  return {.value = std::move(result), .reason_code = {}};
}

}  // namespace lunar::pure_planner_ros

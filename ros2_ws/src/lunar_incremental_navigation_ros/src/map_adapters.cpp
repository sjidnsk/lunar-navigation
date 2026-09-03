#include "lunar_incremental_navigation_ros/map_adapters.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace lunar::incremental_navigation_ros {
namespace {

constexpr char kInvalidInput[] = "INVALID_INPUT";

template <typename T>
[[nodiscard]] AdapterResult<T> Invalid();

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

[[nodiscard]] std::optional<lunar::incremental_navigation::Quaternion>
NormalizedQuaternion(const geometry_msgs::msg::Quaternion& value) noexcept {
  if (!Finite(value.w) || !Finite(value.x) || !Finite(value.y) ||
      !Finite(value.z)) {
    return std::nullopt;
  }
  const double norm = std::sqrt(value.w * value.w + value.x * value.x +
                                value.y * value.y + value.z * value.z);
  if (!Finite(norm) || norm <= 0.0) {
    return std::nullopt;
  }
  return lunar::incremental_navigation::Quaternion{
      .w = value.w / norm,
      .x = value.x / norm,
      .y = value.y / norm,
      .z = value.z / norm};
}

[[nodiscard]] AdapterResult<lunar::incremental_navigation::RigidTransform>
AdaptCurrentMapFromOdomWithoutStamp(
    const geometry_msgs::msg::TransformStamped& transform) {
  const auto rotation = NormalizedQuaternion(transform.transform.rotation);
  if (transform.header.frame_id != "map" ||
      transform.child_frame_id != "odom" || !rotation ||
      !Finite(transform.transform.translation.x) ||
      !Finite(transform.transform.translation.y) ||
      !Finite(transform.transform.translation.z)) {
    return Invalid<lunar::incremental_navigation::RigidTransform>();
  }
  return {.value = lunar::incremental_navigation::RigidTransform{
              .parent_frame = "map",
              .child_frame = "odom",
              .translation_m = {
                  .x = transform.transform.translation.x,
                  .y = transform.transform.translation.y,
                  .z = transform.transform.translation.z},
              .rotation = *rotation},
          .reason_code = {}};
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
    double length, double resolution) noexcept;
[[nodiscard]] std::optional<LayerLayout> ParseLayout(
    const std_msgs::msg::Float32MultiArray& layer, std::size_t width,
    std::size_t height) noexcept;
[[nodiscard]] std::optional<std::vector<float>> UnwrapLayer(
    const std_msgs::msg::Float32MultiArray& layer, const LayerLayout& layout,
    std::size_t width, std::size_t height, std::size_t outer_start,
    std::size_t inner_start);

[[nodiscard]] AdapterResult<OwnedElevationEvidence> AdaptElevationMessage(
    const grid_map_msgs::msg::GridMap& message,
    lunar::incremental_navigation::RigidTransform map_from_source) {
  const auto width = CellDimension(message.info.length_x, message.info.resolution);
  const auto height = CellDimension(message.info.length_y, message.info.resolution);
  if (!width || !height || message.header.frame_id.empty() ||
      map_from_source.parent_frame != "map" ||
      map_from_source.child_frame != message.header.frame_id ||
      !ValidPose(message.info.pose) ||
      message.layers.size() != message.data.size()) {
    return Invalid<OwnedElevationEvidence>();
  }
  const auto found = std::find(message.layers.begin(), message.layers.end(),
                               "elevation");
  if (found == message.layers.end()) {
    return Invalid<OwnedElevationEvidence>();
  }
  const std::size_t elevation_index =
      static_cast<std::size_t>(std::distance(message.layers.begin(), found));
  const auto layout = ParseLayout(message.data[elevation_index], *width, *height);
  if (!layout || message.outer_start_index >= *width ||
      message.inner_start_index >= *height) {
    return Invalid<OwnedElevationEvidence>();
  }
  auto elevation = UnwrapLayer(message.data[elevation_index], *layout, *width,
                               *height, message.outer_start_index,
                               message.inner_start_index);
  if (!elevation) {
    return Invalid<OwnedElevationEvidence>();
  }
  map_from_source.stamp = {};
  return {.value = OwnedElevationEvidence{
              .geometry = lunar::incremental_navigation::GridGeometry{
                  .frame_id = message.header.frame_id,
                  .width = *width,
                  .height = *height,
                  .resolution_m = message.info.resolution,
                  .origin_m = {
                      .x = message.info.pose.position.x -
                           message.info.length_x / 2.0,
                      .y = message.info.pose.position.y -
                           message.info.length_y / 2.0,
                      .z = message.info.pose.position.z}},
              .elevation_m = std::move(*elevation),
              .map_from_source = std::move(map_from_source)},
          .reason_code = {}};
}

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

lunar::incremental_navigation::ElevationEvidence OwnedElevationEvidence::View()
    const noexcept {
  return {.geometry = geometry,
          .elevation_m = elevation_m,
          .map_from_source = map_from_source};
}

AdapterResult<OwnedElevationEvidence> AdaptLocalElevation(
    const InputSnapshot& input) {
  if (!input.local_map || !input.map_from_odom) {
    return Invalid<OwnedElevationEvidence>();
  }
  auto transform = AdaptCurrentMapFromOdomWithoutStamp(*input.map_from_odom);
  if (!transform.value) {
    return Invalid<OwnedElevationEvidence>();
  }
  return AdaptElevationMessage(*input.local_map, std::move(*transform.value));
}

}  // namespace lunar::incremental_navigation_ros

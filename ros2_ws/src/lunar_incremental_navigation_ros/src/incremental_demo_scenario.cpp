#include "lunar_incremental_navigation_ros/incremental_demo_scenario.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include <geometry_msgs/msg/point.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <std_msgs/msg/multi_array_dimension.hpp>
#include <visualization_msgs/msg/marker.hpp>

namespace lunar::incremental_navigation_ros {
namespace {

[[nodiscard]] bool SupportedResolution(const double fine_resolution_m) {
  return fine_resolution_m == 0.2 || fine_resolution_m == 0.1;
}

[[nodiscard]] std::size_t CellsFor(const double length_m,
                                   const double resolution_m) {
  const double cells = length_m / resolution_m;
  if (!std::isfinite(cells) || cells < 1.0 ||
      cells > static_cast<double>(std::numeric_limits<std::uint16_t>::max()) ||
      std::abs(cells - std::round(cells)) > 1.0e-9) {
    throw std::invalid_argument{"demo extent must be an integral cell count"};
  }
  return static_cast<std::size_t>(std::llround(cells));
}

[[nodiscard]] std_msgs::msg::Float32MultiArray MakeLayer(
    const std::vector<float>& logical_values, const std::size_t width,
    const std::size_t height) {
  std_msgs::msg::Float32MultiArray layer;
  std_msgs::msg::MultiArrayDimension outer;
  outer.label = "column_index";
  outer.size = height;
  outer.stride = width * height;
  std_msgs::msg::MultiArrayDimension inner;
  inner.label = "row_index";
  inner.size = width;
  inner.stride = width;
  layer.layout.dim = {outer, inner};
  layer.data.resize(logical_values.size());
  for (std::size_t y = 0U; y < height; ++y) {
    for (std::size_t x = 0U; x < width; ++x) {
      const std::size_t physical_row = width - 1U - x;
      const std::size_t physical_column = height - 1U - y;
      layer.data[physical_column * width + physical_row] =
          logical_values[y * width + x];
    }
  }
  return layer;
}

void AppendPoint(visualization_msgs::msg::Marker& marker, const double x,
                 const double y) {
  geometry_msgs::msg::Point point;
  point.x = x;
  point.y = y;
  marker.points.push_back(point);
}

}  // namespace

IncrementalDemoScenario::IncrementalDemoScenario(
    IncrementalDemoScenarioConfig config)
    : config_(config) {
  if (!SupportedResolution(config_.fine_resolution_m)) {
    throw std::invalid_argument{
        "fine_resolution_m must be exactly 0.2 or 0.1; restart to change it"};
  }
  if (!std::isfinite(config_.local_window_size_m) ||
      config_.local_window_size_m <= 0.0 ||
      !std::isfinite(config_.task_size_m) || config_.task_size_m <= 0.0) {
    throw std::invalid_argument{"demo extents must be finite and positive"};
  }
  static_cast<void>(
      CellsFor(config_.local_window_size_m, config_.fine_resolution_m));
}

double IncrementalDemoScenario::fine_resolution_m() const noexcept {
  return config_.fine_resolution_m;
}

double IncrementalDemoScenario::local_window_size_m() const noexcept {
  return config_.local_window_size_m;
}

float IncrementalDemoScenario::ElevationAt(const double x_m,
                                           const double y_m) const noexcept {
  // The deterministic test terrain is predominantly flat.  Two compact
  // outcrops make the platform-specific fine view visible while leaving
  // several independent frontier directions reachable.
  const double first = std::hypot(x_m - 5.0, y_m - 2.5);
  const double second = std::hypot(x_m + 4.5, y_m + 5.0);
  if (first < 0.75) {
    return static_cast<float>(0.65 * (1.0 - first / 0.75));
  }
  if (second < 0.55) {
    return static_cast<float>(0.45 * (1.0 - second / 0.55));
  }
  return 0.0F;
}

grid_map_msgs::msg::GridMap IncrementalDemoScenario::MakeLocalObservation(
    const IncrementalDemoPose pose) const {
  const std::size_t width =
      CellsFor(config_.local_window_size_m, config_.fine_resolution_m);
  const std::size_t height = width;
  const double half = config_.local_window_size_m / 2.0;
  std::vector<float> elevation(width * height);
  for (std::size_t y = 0U; y < height; ++y) {
    for (std::size_t x = 0U; x < width; ++x) {
      const double world_x = pose.x_m - half +
                             (static_cast<double>(x) + 0.5) *
                                 config_.fine_resolution_m;
      const double world_y = pose.y_m - half +
                             (static_cast<double>(y) + 0.5) *
                                 config_.fine_resolution_m;
      elevation[y * width + x] = ElevationAt(world_x, world_y);
    }
  }

  grid_map_msgs::msg::GridMap map;
  map.header.frame_id = "odom";
  map.info.resolution = config_.fine_resolution_m;
  map.info.length_x = config_.local_window_size_m;
  map.info.length_y = config_.local_window_size_m;
  map.info.pose.position.x = pose.x_m;
  map.info.pose.position.y = pose.y_m;
  map.info.pose.orientation.w = 1.0;
  map.layers = {"elevation"};
  map.basic_layers = map.layers;
  map.data = {MakeLayer(elevation, width, height)};
  map.outer_start_index = 0U;
  map.inner_start_index = 0U;
  return map;
}

visualization_msgs::msg::MarkerArray IncrementalDemoScenario::MakeLocalWindow(
    const IncrementalDemoPose pose) const {
  visualization_msgs::msg::Marker marker;
  marker.header.frame_id = "map";
  marker.ns = "rolling_local_observation";
  marker.id = 0;
  marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
  marker.action = visualization_msgs::msg::Marker::ADD;
  marker.pose.orientation.w = 1.0;
  marker.pose.position.z = 0.08;
  marker.scale.x = 0.08;
  marker.color.r = 0.20F;
  marker.color.g = 1.0F;
  marker.color.b = 0.35F;
  marker.color.a = 0.75F;
  const double half = config_.local_window_size_m / 2.0;
  AppendPoint(marker, pose.x_m - half, pose.y_m - half);
  AppendPoint(marker, pose.x_m + half, pose.y_m - half);
  AppendPoint(marker, pose.x_m + half, pose.y_m + half);
  AppendPoint(marker, pose.x_m - half, pose.y_m + half);
  AppendPoint(marker, pose.x_m - half, pose.y_m - half);
  visualization_msgs::msg::MarkerArray output;
  output.markers.push_back(std::move(marker));
  return output;
}

nav_msgs::msg::OccupancyGrid IncrementalDemoScenario::MakeGroundTruth() const {
  constexpr double kResolutionM = 1.0;
  const auto width = static_cast<std::uint32_t>(
      std::max(1.0, std::ceil(config_.task_size_m / kResolutionM)));
  nav_msgs::msg::OccupancyGrid map;
  map.header.frame_id = "map";
  map.info.resolution = static_cast<float>(kResolutionM);
  map.info.width = width;
  map.info.height = width;
  map.info.origin.position.x = -0.5 * static_cast<double>(width);
  map.info.origin.position.y = -0.5 * static_cast<double>(width);
  map.info.origin.orientation.w = 1.0;
  map.data.assign(static_cast<std::size_t>(width) * width, 0);
  for (std::uint32_t y = 0U; y < width; ++y) {
    for (std::uint32_t x = 0U; x < width; ++x) {
      const double world_x = map.info.origin.position.x + x + 0.5;
      const double world_y = map.info.origin.position.y + y + 0.5;
      if (ElevationAt(world_x, world_y) > 0.20F) {
        map.data[static_cast<std::size_t>(y) * width + x] = 100;
      }
    }
  }
  return map;
}

}  // namespace lunar::incremental_navigation_ros

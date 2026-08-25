#include "lunar_pure_exploration_sim/map_messages.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <std_msgs/msg/float32_multi_array.hpp>
#include <std_msgs/msg/multi_array_dimension.hpp>

namespace lunar::pure_exploration_sim {
namespace {

constexpr std::size_t kLocalWidth = 320U;
constexpr std::size_t kLocalHeight = 320U;
constexpr double kLocalResolutionM = 0.2;
constexpr double kLocalLengthM = 64.0;

std_msgs::msg::Float32MultiArray EmptyLocalLayer() {
  std_msgs::msg::Float32MultiArray layer;
  layer.layout.dim.resize(2U);
  layer.layout.dim[0].label = "column_index";
  layer.layout.dim[0].size = kLocalHeight;
  layer.layout.dim[0].stride = kLocalWidth * kLocalHeight;
  layer.layout.dim[1].label = "row_index";
  layer.layout.dim[1].size = kLocalWidth;
  layer.layout.dim[1].stride = kLocalWidth;
  layer.data.assign(kLocalWidth * kLocalHeight,
                    std::numeric_limits<float>::quiet_NaN());
  return layer;
}

std::size_t PhysicalIndex(std::size_t logical_x, std::size_t logical_y) {
  const std::size_t physical_row = kLocalWidth - 1U - logical_x;
  const std::size_t physical_column = kLocalHeight - 1U - logical_y;
  return physical_column * kLocalWidth + physical_row;
}

}  // namespace

nav_msgs::msg::OccupancyGrid MakeGlobalOverview(
    const LunarScene& scene, const ObservationState& observations,
    const rclcpp::Time& stamp) {
  nav_msgs::msg::OccupancyGrid message;
  message.header.stamp = stamp;
  message.header.frame_id = "map";
  message.info.resolution = static_cast<float>(scene.global_resolution_m());
  message.info.width = static_cast<std::uint32_t>(scene.global_width());
  message.info.height = static_cast<std::uint32_t>(scene.global_height());
  message.info.origin.position.x = scene.min_x_m();
  message.info.origin.position.y = scene.min_x_m();
  message.info.origin.orientation.w = 1.0;
  message.data.assign(scene.global_width() * scene.global_height(),
                      std::int8_t{-1});
  const auto& truth = scene.GlobalOccupancy();
  const auto& known = observations.KnownGlobalMask();
  for (std::size_t index = 0U; index < message.data.size(); ++index) {
    if (index < known.size() && known[index]) {
      message.data[index] = truth[index];
    }
  }
  return message;
}

grid_map_msgs::msg::GridMap MakeLocalGridMap(
    const LunarScene& scene, const ObservationState& observations, Pose2 pose,
    const rclcpp::Time& stamp) {
  static_cast<void>(scene);
  grid_map_msgs::msg::GridMap message;
  message.header.stamp = stamp;
  message.header.frame_id = "odom";
  message.info.resolution = kLocalResolutionM;
  message.info.length_x = kLocalLengthM;
  message.info.length_y = kLocalLengthM;
  message.info.pose.position.x = pose.x_m;
  message.info.pose.position.y = pose.y_m;
  message.info.pose.orientation.w = 1.0;
  message.layers = {"occupancy", "semantic_id", "elevation", "roughness"};
  message.data = {EmptyLocalLayer(), EmptyLocalLayer(), EmptyLocalLayer(),
                  EmptyLocalLayer()};
  message.outer_start_index = 0U;
  message.inner_start_index = 0U;

  for (std::size_t logical_y = 0U; logical_y < kLocalHeight; ++logical_y) {
    for (std::size_t logical_x = 0U; logical_x < kLocalWidth; ++logical_x) {
      const TruthSample* sample =
          observations.CurrentLocalSample(pose, logical_x, logical_y);
      if (sample == nullptr) {
        continue;
      }
      const std::size_t physical = PhysicalIndex(logical_x, logical_y);
      message.data[0].data[physical] = sample->occupied ? 1.0F : 0.0F;
      message.data[1].data[physical] =
          static_cast<float>(sample->semantic_id);
      message.data[2].data[physical] =
          static_cast<float>(sample->elevation_m);
      message.data[3].data[physical] = static_cast<float>(sample->roughness);
    }
  }
  return message;
}

}  // namespace lunar::pure_exploration_sim

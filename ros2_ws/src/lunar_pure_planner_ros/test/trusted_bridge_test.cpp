#include <limits>

#include <gtest/gtest.h>
#include <std_msgs/msg/float32_multi_array.hpp>

#include "lunar_pure_planner_ros/trusted_bridge.hpp"

namespace lunar::pure_planner_ros {
namespace {

grid_map_msgs::msg::GridMap TraversabilityMap() {
  constexpr std::size_t kWidth = 10U;
  constexpr std::size_t kHeight = 10U;
  grid_map_msgs::msg::GridMap map;
  map.header.frame_id = "odom";
  map.info.resolution = 0.2;
  map.info.length_x = kWidth * map.info.resolution;
  map.info.length_y = kHeight * map.info.resolution;
  map.info.pose.orientation.w = 1.0;
  map.layers = {"traversability"};
  std_msgs::msg::Float32MultiArray layer;
  layer.layout.dim.resize(2);
  layer.layout.dim[0].label = "column_index";
  layer.layout.dim[0].size = kHeight;
  layer.layout.dim[0].stride = kWidth * kHeight;
  layer.layout.dim[1].label = "row_index";
  layer.layout.dim[1].size = kWidth;
  layer.layout.dim[1].stride = kWidth;
  layer.data.assign(kWidth * kHeight, std::numeric_limits<float>::quiet_NaN());
  const auto free = [&layer](const std::size_t x, const std::size_t y) {
    const std::size_t physical_x = 10U - 1U - x;
    const std::size_t physical_y = 10U - 1U - y;
    layer.data[physical_y * 10U + physical_x] = 1.0F;
  };
  free(3U, 3U);  // Must be ignored: this is not a 4-neighbour component.
  free(7U, 5U);
  free(8U, 5U);
  free(7U, 6U);
  free(8U, 6U);
  map.data = {std::move(layer)};
  return map;
}

TEST(TrustedBridge, SelectsNearestCellFromFourConnectedTraversableRegion) {
  const auto bridge = FindTrustedBridge(TraversabilityMap(),
                                        {.x = -0.8, .y = 0.0, .z = 0.0});

  ASSERT_TRUE(bridge.has_value());
  EXPECT_EQ(bridge->frame_id, "odom");
  EXPECT_NEAR(bridge->endpoint_m.x, 0.5, 1.0e-9);
  EXPECT_NEAR(bridge->endpoint_m.y, 0.1, 1.0e-9);
}

}  // namespace
}  // namespace lunar::pure_planner_ros

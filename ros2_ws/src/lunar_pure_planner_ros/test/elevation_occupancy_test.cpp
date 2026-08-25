#include <cmath>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

#include "lunar_pure_planner_ros/elevation_occupancy.hpp"

namespace lunar::pure_planner_ros {
namespace {

grid_map_msgs::msg::GridMap ElevationMap() {
  grid_map_msgs::msg::GridMap map;
  map.layers = {"elevation"};
  std_msgs::msg::Float32MultiArray elevation;
  elevation.data = {0.0F, 1.5F, std::numeric_limits<float>::quiet_NaN(),
                    std::numeric_limits<float>::infinity()};
  map.data = {elevation};
  return map;
}

TEST(ElevationOccupancy, CreatesFreeAndUnknownOccupancyWithoutChangingElevation) {
  const auto output = AddOccupancyFromElevation(ElevationMap(), "elevation", "occupancy", false);
  ASSERT_TRUE(output.has_value());
  ASSERT_EQ(output->layers, std::vector<std::string>({"elevation", "occupancy"}));
  ASSERT_EQ(output->data[0].data.size(), 4U);
  EXPECT_EQ(output->data[0].data[0], 0.0F);
  EXPECT_EQ(output->data[0].data[1], 1.5F);
  EXPECT_TRUE(std::isnan(output->data[0].data[2]));
  EXPECT_TRUE(std::isinf(output->data[0].data[3]));
  ASSERT_EQ(output->data[1].data[0], 0.0F);
  ASSERT_EQ(output->data[1].data[1], 0.0F);
  EXPECT_TRUE(std::isnan(output->data[1].data[2]));
  EXPECT_TRUE(std::isnan(output->data[1].data[3]));
}

TEST(ElevationOccupancy, PreservesExplicitOccupancyByDefault) {
  auto input = ElevationMap();
  input.layers.emplace_back("occupancy");
  input.data.emplace_back();
  input.data.back().data = {1.0F, 0.0F, 1.0F, 0.0F};
  const auto output = AddOccupancyFromElevation(input, "elevation", "occupancy", false);
  ASSERT_TRUE(output.has_value());
  EXPECT_EQ(output->data[1].data, input.data[1].data);
}

}  // namespace
}  // namespace lunar::pure_planner_ros

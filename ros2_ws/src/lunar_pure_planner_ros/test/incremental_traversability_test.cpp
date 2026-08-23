#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include <grid_map_msgs/msg/grid_map.hpp>
#include <gtest/gtest.h>
#include <std_msgs/msg/float32_multi_array.hpp>

#include "lunar_pure_planner_ros/incremental_traversability.hpp"

namespace lunar::pure_planner_ros {
namespace {

std_msgs::msg::Float32MultiArray MakeLayer(
    const std::size_t width, const std::size_t height,
    const std::vector<float>& canonical_values) {
  std_msgs::msg::Float32MultiArray layer;
  layer.layout.dim.resize(2);
  layer.layout.dim[0].label = "column_index";
  layer.layout.dim[0].size = height;
  layer.layout.dim[0].stride = width * height;
  layer.layout.dim[1].label = "row_index";
  layer.layout.dim[1].size = width;
  layer.layout.dim[1].stride = width;
  layer.data.resize(width * height);

  for (std::size_t y = 0; y < height; ++y) {
    for (std::size_t x = 0; x < width; ++x) {
      const std::size_t physical_row = width - 1 - x;
      const std::size_t physical_column = height - 1 - y;
      layer.data[physical_column * width + physical_row] =
          canonical_values[y * width + x];
    }
  }
  return layer;
}

float LayerValue(const std_msgs::msg::Float32MultiArray& layer,
                 const std::size_t width, const std::size_t height,
                 const std::size_t x, const std::size_t y) {
  const std::size_t physical_row = width - 1 - x;
  const std::size_t physical_column = height - 1 - y;
  return layer.data[physical_column * width + physical_row];
}

grid_map_msgs::msg::GridMap MakeMap(const std::size_t width,
                                    const std::size_t height,
                                    const std::vector<float>& occupancy,
                                    const std::vector<float>& elevation) {
  grid_map_msgs::msg::GridMap message;
  message.header.frame_id = "odom";
  message.info.resolution = 1.0;
  message.info.length_x = static_cast<double>(width);
  message.info.length_y = static_cast<double>(height);
  message.info.pose.orientation.w = 1.0;
  message.layers = {"occupancy", "elevation"};
  message.data = {MakeLayer(width, height, occupancy),
                  MakeLayer(width, height, elevation)};
  return message;
}

TEST(IncrementalTraversabilityTest,
     RecomputesOnlyAffectedHaloAndPreservesUnknownCells) {
  constexpr std::size_t kWidth = 7;
  constexpr std::size_t kHeight = 7;
  std::vector<float> occupancy(kWidth * kHeight, 0.0F);
  std::vector<float> elevation(kWidth * kHeight, 0.0F);
  IncrementalTraversability builder(
      TraversabilityProfile{.support_radius_m = 0.0,
                            .maximum_slope_rad = 0.5});

  const auto initial = builder.Update(MakeMap(kWidth, kHeight, occupancy, elevation));
  ASSERT_TRUE(initial.has_value());
  EXPECT_TRUE(initial->full_rebuild);
  EXPECT_EQ(initial->recomputed_cells, kWidth * kHeight);
  EXPECT_FLOAT_EQ(LayerValue(initial->map.data.front(), kWidth, kHeight, 3, 3), 1.0F);

  const auto unchanged = builder.Update(MakeMap(kWidth, kHeight, occupancy, elevation));
  ASSERT_TRUE(unchanged.has_value());
  EXPECT_FALSE(unchanged->full_rebuild);
  EXPECT_EQ(unchanged->recomputed_cells, 0U);

  occupancy[3 * kWidth + 3] = 1.0F;
  const auto blocked = builder.Update(MakeMap(kWidth, kHeight, occupancy, elevation));
  ASSERT_TRUE(blocked.has_value());
  EXPECT_FALSE(blocked->full_rebuild);
  EXPECT_GT(blocked->recomputed_cells, 0U);
  EXPECT_LT(blocked->recomputed_cells, kWidth * kHeight);
  EXPECT_FLOAT_EQ(LayerValue(blocked->map.data.front(), kWidth, kHeight, 3, 3), 0.0F);

  elevation[3 * kWidth + 3] = std::numeric_limits<float>::quiet_NaN();
  const auto unknown = builder.Update(MakeMap(kWidth, kHeight, occupancy, elevation));
  ASSERT_TRUE(unknown.has_value());
  EXPECT_TRUE(std::isnan(LayerValue(unknown->map.data.front(), kWidth, kHeight, 3, 3)));

  const auto resized = builder.Update(MakeMap(8, kHeight,
      std::vector<float>(8 * kHeight, 0.0F),
      std::vector<float>(8 * kHeight, 0.0F)));
  ASSERT_TRUE(resized.has_value());
  EXPECT_TRUE(resized->full_rebuild);
  EXPECT_EQ(resized->recomputed_cells, 8U * kHeight);
}


}  // namespace
}  // namespace lunar::pure_planner_ros

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

#include "lunar_pure_planner_ros/center_distance_transform.hpp"

namespace lunar::pure_planner_ros {
namespace {

TEST(CenterDistanceTransformTest, MeasuresExactSquaredDistanceBetweenCellCenters) {
  constexpr std::size_t kWidth = 5U;
  std::vector<std::uint8_t> seeds(kWidth * kWidth, 0U);
  seeds[2U * kWidth + 2U] = 1U;

  const auto transformed = BuildCenterSquaredDistance(kWidth, kWidth, seeds);

  ASSERT_TRUE(transformed.ok()) << transformed.reason_code;
  ASSERT_EQ(transformed.squared_cells.size(), kWidth * kWidth);
  EXPECT_DOUBLE_EQ(transformed.squared_cells[2U * kWidth + 2U], 0.0);
  EXPECT_DOUBLE_EQ(transformed.squared_cells[2U * kWidth + 3U], 1.0);
  EXPECT_DOUBLE_EQ(transformed.squared_cells[3U * kWidth + 3U], 2.0);
  EXPECT_DOUBLE_EQ(transformed.squared_cells[0U], 8.0);
}

TEST(CenterDistanceTransformTest, PreservesInfinityWhenThereAreNoSeeds) {
  const std::vector<std::uint8_t> seeds(12U, 0U);

  const auto transformed = BuildCenterSquaredDistance(4U, 3U, seeds);

  ASSERT_TRUE(transformed.ok()) << transformed.reason_code;
  for (const double value : transformed.squared_cells) {
    EXPECT_TRUE(std::isinf(value));
  }
}

}  // namespace
}  // namespace lunar::pure_planner_ros

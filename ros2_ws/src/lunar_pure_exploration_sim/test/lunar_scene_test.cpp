#include "lunar_pure_exploration_sim/lunar_scene.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

#include <gtest/gtest.h>

namespace lunar::pure_exploration_sim {
namespace {

TEST(LunarSceneTest, BuildsDeterministicOccupiedSceneWithSafeStart) {
  const auto scene = BuildLunarScene(20260824U);

  EXPECT_DOUBLE_EQ(scene.min_x_m(), -150.0);
  EXPECT_DOUBLE_EQ(scene.max_x_m(), 150.0);
  EXPECT_EQ(scene.global_width(), 300U);
  EXPECT_EQ(scene.global_height(), 300U);
  EXPECT_DOUBLE_EQ(scene.global_resolution_m(), 1.0);
  EXPECT_EQ(scene.GlobalOccupancy().size(), 90'000U);
  EXPECT_GT(std::ranges::count(scene.GlobalOccupancy(), std::int8_t{100}), 0);
  EXPECT_EQ(scene.GlobalOccupancy(),
            BuildLunarScene(20260824U).GlobalOccupancy());
  EXPECT_FALSE(scene.Sample(0.0, 0.0).occupied);
}

TEST(LunarSceneTest, SeedChangesGeneratedObstacleField) {
  EXPECT_NE(BuildLunarScene(20260824U).GlobalOccupancy(),
            BuildLunarScene(20260825U).GlobalOccupancy());
}

TEST(LunarSceneTest, ReservesConnectedCrossAndLoopBackbone) {
  const auto scene = BuildLunarScene(20260824U);
  for (double coordinate = -149.5; coordinate < 150.0; coordinate += 1.0) {
    EXPECT_FALSE(scene.Sample(coordinate, 0.0).occupied);
    EXPECT_FALSE(scene.Sample(0.0, coordinate).occupied);
  }

  for (double coordinate = -80.0; coordinate <= 80.0; coordinate += 1.0) {
    EXPECT_FALSE(scene.Sample(coordinate, -80.0).occupied);
    EXPECT_FALSE(scene.Sample(coordinate, 80.0).occupied);
    EXPECT_FALSE(scene.Sample(-80.0, coordinate).occupied);
    EXPECT_FALSE(scene.Sample(80.0, coordinate).occupied);
  }
}

TEST(LunarSceneTest, ContainsRockAndCraterRimSemantics) {
  const auto scene = BuildLunarScene(20260824U);
  std::array<std::size_t, 3> semantic_counts{};

  for (std::size_t row = 0; row < scene.global_height(); ++row) {
    for (std::size_t column = 0; column < scene.global_width(); ++column) {
      const auto sample = scene.Sample(
          scene.min_x_m() + (static_cast<double>(column) + 0.5) *
                                scene.global_resolution_m(),
          scene.min_x_m() +
              (static_cast<double>(row) + 0.5) * scene.global_resolution_m());
      ASSERT_LT(sample.semantic_id, semantic_counts.size());
      ++semantic_counts[sample.semantic_id];
    }
  }

  EXPECT_GT(semantic_counts[1], 0U);
  EXPECT_GT(semantic_counts[2], 0U);
}

TEST(LunarSceneTest, ReportsMaximumNeighborElevationDifferenceAsRoughness) {
  const auto scene = BuildLunarScene(20260824U);
  constexpr double kX = 23.15;
  constexpr double kY = -41.35;
  constexpr double kOffset = 0.2;
  const auto center = scene.Sample(kX, kY);
  const std::array<double, 4> neighbor_elevations{
      scene.Sample(kX - kOffset, kY).elevation_m,
      scene.Sample(kX + kOffset, kY).elevation_m,
      scene.Sample(kX, kY - kOffset).elevation_m,
      scene.Sample(kX, kY + kOffset).elevation_m,
  };
  double expected_roughness = 0.0;
  for (const double neighbor_elevation : neighbor_elevations) {
    expected_roughness = std::max(
        expected_roughness,
        std::abs(center.elevation_m - neighbor_elevation));
  }

  EXPECT_DOUBLE_EQ(center.roughness, expected_roughness);
}

}  // namespace
}  // namespace lunar::pure_exploration_sim

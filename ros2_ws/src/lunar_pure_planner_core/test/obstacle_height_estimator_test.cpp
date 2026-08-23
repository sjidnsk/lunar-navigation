#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include "shared/map_snapshot.hpp"
#include "shared/obstacle_height_estimator.hpp"

namespace lunar::pure_planning::shared {
namespace {

[[nodiscard]] std::shared_ptr<const MapSnapshot> Snapshot(
    const std::size_t width, std::vector<float> occupancy,
    std::vector<float> elevation, const double resolution_m = 1.0) {
  GridMap map{
      .frame_id = "odom",
      .stamp = TimePoint{.nanoseconds_since_epoch = 1},
      .width = width,
      .height = occupancy.size() / width,
      .resolution_m = resolution_m,
      .origin_m = Vec3{},
      .layers = {},
  };
  map.layers.emplace("occupancy", GridLayer{.values = std::move(occupancy)});
  map.layers.emplace("elevation", GridLayer{.values = std::move(elevation)});
  const auto result = MapSnapshot::Create(std::move(map), MapContract::kLocalElevation);
  EXPECT_TRUE(result.ok()) << result.reason_code;
  return result.snapshot;
}

[[nodiscard]] std::shared_ptr<const MapSnapshot> SlopedObstacle() {
  constexpr std::size_t kWidth = 7U;
  std::vector<float> occupancy(kWidth * kWidth, 0.0F);
  std::vector<float> elevation;
  elevation.reserve(kWidth * kWidth);
  for (std::size_t y = 0U; y < kWidth; ++y) {
    for (std::size_t x = 0U; x < kWidth; ++x) {
      elevation.push_back(static_cast<float>(0.1 * static_cast<double>(x) +
                                             0.2 * static_cast<double>(y)));
    }
  }
  constexpr std::size_t kCenter = 3U * kWidth + 3U;
  occupancy[kCenter] = 0.5F;
  elevation[kCenter] += 0.8F;
  return Snapshot(kWidth, std::move(occupancy), std::move(elevation));
}

TEST(ObstacleHeight, FitsGroundFromSixOrMoreFreeSamples) {
  const auto value = EstimateObstacleHeight(*SlopedObstacle(), GridCell{.x = 3, .y = 3});

  ASSERT_TRUE(value.flyover_allowed);
  EXPECT_NEAR(value.height_m, 0.8, 1.0e-6);
}

TEST(ObstacleHeight, KeepsRelativeHeightStableUnderLargeElevationOffset) {
  constexpr std::size_t kWidth = 7U;
  constexpr std::size_t kCenter = 3U * kWidth + 3U;
  constexpr double kResolutionM = 1.0e-3;
  std::vector<float> occupancy(kWidth * kWidth, 0.0F);
  occupancy[kCenter] = 0.5F;
  std::vector<float> near_zero(kWidth * kWidth, 0.0F);
  near_zero[kCenter] = 4.0F;
  std::vector<float> large_offset(kWidth * kWidth, 10000000.0F);
  large_offset[kCenter] += 4.0F;

  const auto baseline = EstimateObstacleHeight(
      *Snapshot(kWidth, occupancy, std::move(near_zero), kResolutionM),
      GridCell{.x = 3, .y = 3});
  const auto shifted = EstimateObstacleHeight(
      *Snapshot(kWidth, std::move(occupancy), std::move(large_offset),
                kResolutionM),
      GridCell{.x = 3, .y = 3});

  ASSERT_TRUE(baseline.flyover_allowed);
  ASSERT_TRUE(shifted.flyover_allowed);
  EXPECT_NEAR(shifted.height_m, baseline.height_m, 1.0e-9);
  EXPECT_NEAR(shifted.height_m, 4.0, 1.0e-9);
}

TEST(ObstacleHeight, RejectsFewerThanSixFreeSamplesInsideChebyshevWindow) {
  constexpr std::size_t kWidth = 7U;
  std::vector<float> occupancy(kWidth * kWidth, 0.5F);
  std::vector<float> elevation(kWidth * kWidth, 1.0F);
  constexpr std::size_t kCenter = 3U * kWidth + 3U;
  for (const std::size_t index : {0U, 1U, 7U, 8U, 9U}) {
    occupancy[index] = 0.0F;
  }
  elevation[kCenter] = 2.0F;

  EXPECT_FALSE(EstimateObstacleHeight(*Snapshot(kWidth, std::move(occupancy),
                                                 std::move(elevation)),
                                      GridCell{.x = 3, .y = 3})
                   .flyover_allowed);
}

TEST(ObstacleHeight, RejectsRankDeficientGroundSamples) {
  constexpr std::size_t kWidth = 7U;
  std::vector<float> occupancy(kWidth * kWidth, 0.5F);
  std::vector<float> elevation(kWidth * kWidth, 1.0F);
  constexpr std::size_t kCenter = 3U * kWidth + 3U;
  for (std::size_t x = 0U; x < 6U; ++x) {
    occupancy[x] = 0.0F;
  }
  elevation[kCenter] = 2.0F;

  EXPECT_FALSE(EstimateObstacleHeight(*Snapshot(kWidth, std::move(occupancy),
                                                 std::move(elevation)),
                                      GridCell{.x = 3, .y = 3})
                   .flyover_allowed);
}

TEST(ObstacleHeight, RejectsNaNOrNonpositiveOccupiedHeight) {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  constexpr std::size_t kCenter = 3U * 7U + 3U;

  std::vector<float> occupancy(7U * 7U, 0.0F);
  std::vector<float> nan_elevation(7U * 7U, 0.0F);
  occupancy[kCenter] = 0.5F;
  nan_elevation[kCenter] = nan;
  EXPECT_FALSE(EstimateObstacleHeight(*Snapshot(7U, std::move(occupancy),
                                                 std::move(nan_elevation)),
                                      GridCell{.x = 3, .y = 3})
                   .flyover_allowed);

  std::vector<float> flat_occupancy(7U * 7U, 0.0F);
  std::vector<float> flat_elevation(7U * 7U, 1.0F);
  flat_occupancy[kCenter] = 0.5F;
  EXPECT_FALSE(EstimateObstacleHeight(*Snapshot(7U, std::move(flat_occupancy),
                                                 std::move(flat_elevation)),
                                      GridCell{.x = 3, .y = 3})
                   .flyover_allowed);
}

}  // namespace
}  // namespace lunar::pure_planning::shared

#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <ranges>
#include <vector>

#include <gtest/gtest.h>

#include "lunar_pure_planner_core/search_control.hpp"
#include "shared/edge_validation_cache.hpp"
#include "shared/goal_distance_field.hpp"
#include "shared/local_terrain_projection.hpp"
#include "shared/map_snapshot.hpp"

namespace lunar::pure_planning::shared {
namespace {

using namespace std::chrono_literals;

[[nodiscard]] std::shared_ptr<const MapSnapshot> Snapshot(
    const std::size_t width, std::vector<float> occupancy,
    std::vector<float> elevation) {
  GridMap map{
      .frame_id = "odom",
      .stamp = TimePoint{.nanoseconds_since_epoch = 1},
      .width = width,
      .height = occupancy.size() / width,
      .resolution_m = 1.0,
      .origin_m = Vec3{},
      .layers = {},
  };
  map.layers.emplace("occupancy", GridLayer{.values = std::move(occupancy)});
  map.layers.emplace("elevation", GridLayer{.values = std::move(elevation)});
  const auto result = MapSnapshot::Create(std::move(map), MapContract::kLocalElevation);
  EXPECT_TRUE(result.ok()) << result.reason_code;
  return result.snapshot;
}

TEST(LocalTerrainProjection, TreatsCellFaultsAsUnknownWithoutRejectingMap) {
  constexpr std::size_t kWidth = 3U;
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const auto map = Snapshot(
      kWidth, {0.0F, nan, 1.1F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F},
      {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, nan, 0.0F});

  const auto projection = BuildLocalTerrainProjection(map, 0.5F);

  ASSERT_TRUE(projection.ok()) << projection.reason_code;
  EXPECT_EQ(projection.value->free_with_height[1], 0U);
  EXPECT_EQ(projection.value->free_with_height[2], 0U);
  EXPECT_EQ(projection.value->free_with_height[7], 0U);
  EXPECT_EQ(projection.value->free_with_height[0], 1U);
}

TEST(LocalTerrainProjection, DerivesSlopeRoughnessAndClearanceFromTwoLayers) {
  constexpr std::size_t kWidth = 5U;
  std::vector<float> occupancy(kWidth * kWidth, 0.0F);
  occupancy[2U * kWidth + 2U] = 0.5F;
  std::vector<float> elevation;
  elevation.reserve(kWidth * kWidth);
  for (std::size_t y = 0U; y < kWidth; ++y) {
    for (std::size_t x = 0U; x < kWidth; ++x) {
      elevation.push_back(static_cast<float>(0.2 * static_cast<double>(x) +
                                             0.1 * static_cast<double>(y)));
    }
  }
  const auto map = Snapshot(kWidth, std::move(occupancy), std::move(elevation));

  const auto projection = BuildLocalTerrainProjection(map, 0.5F);

  ASSERT_TRUE(projection.ok()) << projection.reason_code;
  const std::size_t corner = 0U;
  const std::size_t obstacle = 2U * kWidth + 2U;
  const std::size_t side = 2U * kWidth + 1U;
  EXPECT_NEAR(projection.value->slope_rad[corner], std::atan(std::sqrt(0.05)),
              1.0e-6);
  EXPECT_NEAR(projection.value->roughness_m[corner], 0.0F, 1.0e-6F);
  EXPECT_FLOAT_EQ(projection.value->clearance_m[obstacle], 0.0F);
  EXPECT_FLOAT_EQ(projection.value->clearance_m[side], 1.0F);
}

TEST(LocalTerrainProjection, KeepsSlopeAndRoughnessStableUnderLargeElevationOffset) {
  constexpr std::size_t kWidth = 3U;
  constexpr std::size_t kCenter = 1U * kWidth + 1U;
  std::vector<float> occupancy(kWidth * kWidth, 0.0F);
  const auto near_zero = Snapshot(kWidth, occupancy,
                                  std::vector<float>(kWidth * kWidth, 0.0F));
  const auto large_offset = Snapshot(
      kWidth, std::move(occupancy),
      std::vector<float>(kWidth * kWidth, 100000000000000.0F));

  const auto baseline = BuildLocalTerrainProjection(near_zero);
  const auto shifted = BuildLocalTerrainProjection(large_offset);

  ASSERT_TRUE(baseline.ok()) << baseline.reason_code;
  ASSERT_TRUE(shifted.ok()) << shifted.reason_code;
  EXPECT_FLOAT_EQ(shifted.value->slope_rad[kCenter],
                  baseline.value->slope_rad[kCenter]);
  EXPECT_FLOAT_EQ(shifted.value->roughness_m[kCenter],
                  baseline.value->roughness_m[kCenter]);
  EXPECT_TRUE(std::isfinite(shifted.value->roughness_m[kCenter]));
}

TEST(LocalTerrainProjection, StopsDuringProjectionAtInjectedDeadline) {
  constexpr std::size_t kWidth = 64U;
  const auto map = Snapshot(
      kWidth, std::vector<float>(kWidth * kWidth, 0.0F),
      std::vector<float>(kWidth * kWidth, 0.0F));
  SteadyClock::time_point now{};
  SearchControl control{
      .deadline = SteadyClock::time_point{5ms},
      .now = [&now] {
        now += 1ms;
        return now;
      },
  };

  const auto projection = BuildLocalTerrainProjection(map, 0.5F, control);

  EXPECT_FALSE(projection.ok());
  EXPECT_EQ(projection.reason_code, "TIMEOUT");
}

TEST(GoalDistanceField, RoutesThroughAnOpeningAndMarksSealedCellsUnreachable) {
  constexpr std::size_t kWidth = 7U;
  constexpr std::size_t kHeight = 5U;
  std::vector<float> with_opening(kWidth * kHeight, 0.0F);
  for (std::size_t y = 0U; y < kHeight - 1U; ++y) {
    with_opening[y * kWidth + 3U] = 1.0F;
  }
  const auto open_map = Snapshot(
      kWidth, std::move(with_opening),
      std::vector<float>(kWidth * kHeight, 0.0F));
  const auto open_projection = BuildLocalTerrainProjection(open_map);
  ASSERT_TRUE(open_projection.ok()) << open_projection.reason_code;

  const auto open_field = BuildGoalDistanceField(
      *open_projection.value, GridCell{.x = 6, .y = 2});
  ASSERT_TRUE(open_field.has_value());
  const std::size_t source_index = open_map->Index(GridCell{.x = 0, .y = 2});
  EXPECT_TRUE(std::isfinite(open_field->distance_m[source_index]));
  EXPECT_GT(open_field->distance_m[source_index], 6.0);

  std::vector<float> sealed(kWidth * kHeight, 0.0F);
  for (std::size_t y = 0U; y < kHeight; ++y) {
    sealed[y * kWidth + 3U] = 1.0F;
  }
  const auto sealed_map = Snapshot(
      kWidth, std::move(sealed), std::vector<float>(kWidth * kHeight, 0.0F));
  const auto sealed_projection = BuildLocalTerrainProjection(sealed_map);
  ASSERT_TRUE(sealed_projection.ok()) << sealed_projection.reason_code;

  const auto sealed_field = BuildGoalDistanceField(
      *sealed_projection.value, GridCell{.x = 6, .y = 2});
  ASSERT_TRUE(sealed_field.has_value());
  EXPECT_FALSE(std::isfinite(
      sealed_field->distance_m[sealed_map->Index(GridCell{.x = 0, .y = 2})]));
}

TEST(GoalDistanceField, StopsWhenControlIsCanceled) {
  constexpr std::size_t kWidth = 5U;
  const auto map = Snapshot(
      kWidth, std::vector<float>(kWidth * kWidth, 0.0F),
      std::vector<float>(kWidth * kWidth, 0.0F));
  const auto projection = BuildLocalTerrainProjection(map);
  ASSERT_TRUE(projection.ok()) << projection.reason_code;
  std::stop_source stop;
  stop.request_stop();

  const auto field = BuildGoalDistanceField(
      *projection.value, GridCell{.x = 2, .y = 2},
      SearchControl{.stop_token = stop.get_token()});

  EXPECT_FALSE(field.has_value());
}

TEST(EdgeValidationCache, EvaluatesEachKeyAtMostOnce) {
  EdgeValidationCache<int, int, std::hash<int>> cache;
  int calls = 0;

  const int& first = cache.GetOrEvaluate(7, [&] { return ++calls; });
  const int& repeated = cache.GetOrEvaluate(7, [&] { return ++calls; });
  const int& second = cache.GetOrEvaluate(8, [&] { return ++calls; });

  EXPECT_EQ(first, 1);
  EXPECT_EQ(repeated, 1);
  EXPECT_EQ(second, 2);
  EXPECT_EQ(calls, 2);
  EXPECT_EQ(cache.evaluation_count(), 2U);
}

}  // namespace
}  // namespace lunar::pure_planning::shared

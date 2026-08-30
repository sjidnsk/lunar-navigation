#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <ranges>
#include <span>
#include <stop_token>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "lunar_pure_planner_core/search_control.hpp"
#include "lunar_pure_planner_core/traversability_map.hpp"
#include "shared/global_occupancy_projection.hpp"
#include "shared/map_snapshot.hpp"

namespace lunar::pure_planning::shared {
namespace {

using namespace std::chrono_literals;

[[nodiscard]] std::shared_ptr<const MapSnapshot> Snapshot(
    std::vector<std::int8_t> occupancy, const std::size_t width,
    const double resolution_m = 1.0) {
  GridMap map{
      .frame_id = "map",
      .stamp = TimePoint{.nanoseconds_since_epoch = 1},
      .width = width,
      .height = occupancy.size() / width,
      .resolution_m = resolution_m,
      .origin_m = Vec3{},
      .layers = {},
  };
  map.layers.emplace(
      "occupancy", GridLayer{.values = std::move(occupancy)});
  const auto result =
      MapSnapshot::Create(std::move(map), MapContract::kGlobalOccupancy);
  EXPECT_TRUE(result.ok()) << result.reason_code;
  return result.snapshot;
}

TEST(GlobalOccupancyProjection, ClassifiesNativeValuesAtDefaultThreshold) {
  const auto map = Snapshot({-1, 0, 49, 50, 100, 101}, 6U);

  const auto result = BuildGlobalOccupancyProjection(map);

  ASSERT_TRUE(result.ok()) << result.reason_code;
  const GlobalOccupancyProjectionView view = result.projection->View();
  ASSERT_EQ(view.hard_feasible.size(), 6U);
  constexpr std::array<std::uint8_t, 6> kExpected{
      0U, 1U, 1U, 0U, 0U, 0U};
  EXPECT_TRUE(std::ranges::equal(view.hard_feasible, kExpected));
  EXPECT_TRUE(view.Valid());
}

TEST(GlobalOccupancyProjection, HonorsAnExplicitIntegerThreshold) {
  const auto map = Snapshot({50}, 1U);

  const auto result = BuildGlobalOccupancyProjection(map, 70);

  ASSERT_TRUE(result.ok()) << result.reason_code;
  const auto view = result.projection->View();
  ASSERT_EQ(view.hard_feasible.size(), 1U);
  EXPECT_EQ(view.hard_feasible[0], 1U);
}

TEST(GlobalOccupancyProjection, DerivesClearanceOnlyFromNativeHazards) {
  const auto unknown_map = Snapshot({0, 0, -1, 0, 0}, 5U);
  const auto occupied_map = Snapshot({0, 0, 100, 0, 0}, 5U);
  const auto clear_map = Snapshot({0, 0, 0, 0, 0}, 5U);

  const auto unknown = BuildGlobalOccupancyProjection(unknown_map);
  const auto occupied = BuildGlobalOccupancyProjection(occupied_map);
  const auto clear = BuildGlobalOccupancyProjection(clear_map);

  ASSERT_TRUE(unknown.ok()) << unknown.reason_code;
  ASSERT_TRUE(occupied.ok()) << occupied.reason_code;
  ASSERT_TRUE(clear.ok()) << clear.reason_code;
  const auto unknown_clearance =
      unknown.projection->View().clearance_m;
  const auto occupied_clearance =
      occupied.projection->View().clearance_m;
  ASSERT_EQ(unknown_clearance.size(), 5U);
  EXPECT_TRUE(std::ranges::equal(unknown_clearance, occupied_clearance));
  EXPECT_FLOAT_EQ(unknown_clearance[0], 1.5F);
  EXPECT_FLOAT_EQ(unknown_clearance[1], 0.5F);
  EXPECT_FLOAT_EQ(unknown_clearance[2], 0.0F);
  EXPECT_FLOAT_EQ(unknown_clearance[3], 0.5F);
  EXPECT_FLOAT_EQ(unknown_clearance[4], 1.5F);
  EXPECT_TRUE(std::ranges::all_of(
      clear.projection->View().clearance_m,
      [](const float value) { return std::isinf(value) && value > 0.0F; }));
}

TEST(GlobalOccupancyProjection, UsesOccupiedCellAreaForDiagonalClearance) {
  std::vector<std::int8_t> occupancy(5U * 5U, 0);
  occupancy[2U * 5U + 2U] = 100;
  const auto map = Snapshot(std::move(occupancy), 5U);

  const auto result = BuildGlobalOccupancyProjection(map, 50);

  ASSERT_TRUE(result.ok()) << result.reason_code;
  const auto view = result.projection->View();
  EXPECT_FLOAT_EQ(view.ClearanceMeters({.x = 2, .y = 2}), 0.0F);
  EXPECT_FLOAT_EQ(view.ClearanceMeters({.x = 3, .y = 2}), 0.5F);
  EXPECT_FLOAT_EQ(view.ClearanceMeters({.x = 4, .y = 2}), 1.5F);
  EXPECT_FLOAT_EQ(view.ClearanceMeters({.x = 3, .y = 3}),
                  static_cast<float>(std::sqrt(0.5)));
}

TEST(GlobalOccupancyProjection, InflatesByStrictOccupiedCellAreaDistance) {
  std::vector<std::int8_t> occupancy(5U * 5U, 0);
  occupancy[2U * 5U + 2U] = 100;
  const auto map = Snapshot(std::move(occupancy), 5U);

  const auto native = BuildGlobalOccupancyProjection(map, 50);
  const auto inflated =
      BuildInflatedGlobalOccupancyProjection(map, 50, 0.9187);
  const auto equality = BuildInflatedGlobalOccupancyProjection(map, 50, 0.5);

  ASSERT_TRUE(native.ok()) << native.reason_code;
  ASSERT_TRUE(inflated.ok()) << inflated.reason_code;
  ASSERT_TRUE(equality.ok()) << equality.reason_code;
  EXPECT_TRUE(native.projection->View().HardFeasible({.x = 3, .y = 2}));
  EXPECT_FALSE(inflated.projection->View().HardFeasible({.x = 3, .y = 2}));
  EXPECT_FALSE(inflated.projection->View().HardFeasible({.x = 3, .y = 3}));
  EXPECT_TRUE(inflated.projection->View().HardFeasible({.x = 4, .y = 2}));
  EXPECT_TRUE(equality.projection->View().HardFeasible({.x = 3, .y = 2}));
  EXPECT_FLOAT_EQ(inflated.projection->View().ClearanceMeters({.x = 3, .y = 2}),
                  0.5F);
}

TEST(GlobalOccupancyProjection,
     BuildsLeggedProjectionFromRawSnapshotAndInflatesExactlyOnce) {
  constexpr std::size_t kWidth = 5U;
  GridMap global{
      .frame_id = "map",
      .width = kWidth,
      .height = kWidth,
      .resolution_m = 1.0,
      .layers = {{"occupancy", GridLayer{.values =
                              std::vector<std::int8_t>(kWidth * kWidth, 0)}}},
  };
  std::vector<float> local_occupancy(kWidth * kWidth, 0.0F);
  local_occupancy[2U * kWidth + 2U] = 0.9F;
  GridMap local{
      .frame_id = "odom",
      .width = kWidth,
      .height = kWidth,
      .resolution_m = 1.0,
      .layers = {
          {"occupancy", GridLayer{.values = std::move(local_occupancy)}},
          {"elevation", GridLayer{.values =
                              std::vector<float>(kWidth * kWidth, 0.0F)}},
      },
  };
  PersistentTraversabilityMap map(TraversabilityProfile{
      .global_occupancy_threshold = 50,
      .local_occupancy_threshold = 0.5,
      .maximum_slope_rad = 0.6,
      .inflation_radius_m = 0.6,
  });
  ASSERT_TRUE(map.UpdateGlobal(global).accepted);
  ASSERT_TRUE(map.UpdateLocal(
      local,
      RigidTransform{.parent_frame = "map", .child_frame = "odom"}, 1U)
                  .accepted);

  const auto result = BuildLeggedTraversabilityProjection(*map.Capture());

  ASSERT_TRUE(result.ok()) << result.reason_code;
  const auto view = result.projection->View();
  EXPECT_FALSE(view.HardFeasible({.x = 2, .y = 2}));
  EXPECT_FALSE(view.HardFeasible({.x = 3, .y = 2}));
  EXPECT_TRUE(view.HardFeasible({.x = 4, .y = 2}));
}

TEST(GlobalOccupancyProjection, RejectsInvalidInflationDistance) {
  const auto map = Snapshot({0, 0, 0, 0}, 2U);

  const auto negative = BuildInflatedGlobalOccupancyProjection(map, 50, -0.1);
  const auto nan = BuildInflatedGlobalOccupancyProjection(
      map, 50, std::numeric_limits<double>::quiet_NaN());

  EXPECT_FALSE(negative.ok());
  EXPECT_EQ(negative.reason_code, "GLOBAL_OCCUPANCY_PROJECTION_INVALID");
  EXPECT_FALSE(nan.ok());
  EXPECT_EQ(nan.reason_code, "GLOBAL_OCCUPANCY_PROJECTION_INVALID");
}

TEST(GlobalOccupancyProjection, StopsDuringProjectionAtInjectedDeadline) {
  constexpr std::size_t kWidth = 128U;
  const auto map = Snapshot(
      std::vector<std::int8_t>(kWidth * kWidth, 100), kWidth);
  SteadyClock::time_point now{};
  SearchControl control{
      .deadline = SteadyClock::time_point{5ms},
      .now = [&now] {
        now += 1ms;
        return now;
      },
  };

  const auto result = BuildGlobalOccupancyProjection(map, 50, control);

  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.reason_code, "TIMEOUT");
}

TEST(GlobalOccupancyProjection, CancellationWinsBeforeProjectionWork) {
  const auto map = Snapshot({0, 0, 0, 0}, 2U);
  std::stop_source stop;
  stop.request_stop();

  const auto result = BuildGlobalOccupancyProjection(
      map, 50, SearchControl{.stop_token = stop.get_token()});

  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.reason_code, "REQUEST_CANCELED");
}

}  // namespace
}  // namespace lunar::pure_planning::shared

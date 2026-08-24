#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "lunar_pure_planner_core/traversability_map.hpp"

namespace lunar::pure_planning {
namespace {

[[nodiscard]] GridMap GlobalMap(const std::size_t width, const std::size_t height,
                                const double resolution_m,
                                std::vector<std::int8_t> occupancy,
                                const Vec3 origin_m = {}) {
  return GridMap{
      .frame_id = "map",
      .stamp = TimePoint{.nanoseconds_since_epoch = 1},
      .width = width,
      .height = height,
      .resolution_m = resolution_m,
      .origin_m = origin_m,
      .layers = {{"occupancy", GridLayer{.values = std::move(occupancy)}}},
  };
}

[[nodiscard]] GridMap LocalMap(const std::size_t width, const std::size_t height,
                               const double resolution_m,
                               std::vector<float> occupancy,
                               std::vector<float> elevation,
                               const Vec3 origin_m = {}) {
  return GridMap{
      .frame_id = "odom",
      .stamp = TimePoint{.nanoseconds_since_epoch = 1},
      .width = width,
      .height = height,
      .resolution_m = resolution_m,
      .origin_m = origin_m,
      .layers = {
          {"occupancy", GridLayer{.values = std::move(occupancy)}},
          {"elevation", GridLayer{.values = std::move(elevation)}},
      },
  };
}

[[nodiscard]] RigidTransform MapFromOdom(
    const Vec3 translation_m = {}, const double yaw_rad = 0.0) {
  return RigidTransform{
      .parent_frame = "map",
      .child_frame = "odom",
      .stamp = TimePoint{.nanoseconds_since_epoch = 1},
      .translation_m = translation_m,
      .rotation = Quaternion{.w = std::cos(yaw_rad / 2.0),
                             .z = std::sin(yaw_rad / 2.0)},
  };
}

[[nodiscard]] TraversabilityProfile Profile(const double inflation_radius_m = 0.0) {
  return TraversabilityProfile{
      .global_occupancy_threshold = 50,
      .local_occupancy_threshold = 0.5,
      .maximum_slope_rad = 1.0,
      .inflation_radius_m = inflation_radius_m,
  };
}

[[nodiscard]] std::shared_ptr<const TraversabilitySnapshot> UpdateKnownLocal(
    PersistentTraversabilityMap& map, const GridMap& local,
    const RigidTransform& map_from_odom, const std::uint64_t sequence) {
  const TraversabilityUpdateResult result =
      map.UpdateLocal(local, map_from_odom, sequence);
  EXPECT_TRUE(result.accepted) << result.reason_code;
  return map.Capture();
}

TEST(TraversabilityMapV1, LocksFirstLocalResolution) {
  PersistentTraversabilityMap map(Profile());
  const GridMap local = LocalMap(1U, 1U, 0.25, {0.0F}, {0.0F});

  const auto snapshot = UpdateKnownLocal(map, local, MapFromOdom(), 1U);

  ASSERT_TRUE(snapshot);
  EXPECT_TRUE(snapshot->valid());
  EXPECT_DOUBLE_EQ(snapshot->resolution_m(), 0.25);
}

TEST(TraversabilityMapV1, RejectsResolutionChangeWithoutMutation) {
  PersistentTraversabilityMap map(Profile());
  const auto first = UpdateKnownLocal(
      map, LocalMap(1U, 1U, 0.25, {1.0F}, {0.0F}), MapFromOdom(), 1U);
  ASSERT_TRUE(first);
  const std::uint64_t first_revision = first->revision();

  const TraversabilityUpdateResult rejected = map.UpdateLocal(
      LocalMap(1U, 1U, 0.5, {0.0F}, {0.0F}), MapFromOdom(), 2U);
  const auto after = map.Capture();

  EXPECT_FALSE(rejected.accepted);
  EXPECT_FALSE(rejected.changed);
  EXPECT_EQ(rejected.reason_code, "MAP_RESOLUTION_MISMATCH");
  ASSERT_TRUE(after);
  EXPECT_EQ(after->revision(), first_revision);
  EXPECT_EQ(after->StateAtWorld(0.125, 0.125), TraversabilityState::kBlocked);
}

TEST(TraversabilityMapV1, AcceptsResolutionWithinRelativeTolerance) {
  PersistentTraversabilityMap map(Profile());
  ASSERT_TRUE(UpdateKnownLocal(
      map, LocalMap(1U, 1U, 0.25, {0.0F}, {0.0F}), MapFromOdom(), 1U));

  const TraversabilityUpdateResult update = map.UpdateLocal(
      LocalMap(1U, 1U, 0.2500001, {0.0F}, {0.0F}), MapFromOdom(), 2U);

  EXPECT_TRUE(update.accepted) << update.reason_code;
  EXPECT_NE(update.reason_code, "MAP_RESOLUTION_MISMATCH");
}

TEST(TraversabilityMapV1, AnchorsCanonicalOriginToGlobalMap) {
  PersistentTraversabilityMap map(Profile());
  const Vec3 global_origin{.x = 10.0, .y = -3.0};
  ASSERT_TRUE(map.UpdateGlobal(GlobalMap(2U, 2U, 1.0, {0, 0, 0, 0},
                                         global_origin))
                  .accepted);

  const auto snapshot = UpdateKnownLocal(
      map, LocalMap(1U, 1U, 0.25,
                    {std::numeric_limits<float>::quiet_NaN()},
                    {std::numeric_limits<float>::quiet_NaN()},
                    Vec3{.x = 5.0, .y = 4.0}),
      MapFromOdom(Vec3{.x = 2.0, .y = 1.0}), 1U);

  ASSERT_TRUE(snapshot);
  EXPECT_EQ(snapshot->origin_m(), global_origin);
}

TEST(TraversabilityMapV1, ReanchorsLocalHistoryWhenGlobalArrivesLater) {
  PersistentTraversabilityMap map(Profile());
  ASSERT_TRUE(map.UpdateLocal(
      LocalMap(1U, 1U, 0.25, {0.9F}, {0.0F}),
      MapFromOdom(Vec3{.x = 10.0, .y = 0.0}), 1U)
                  .accepted);
  ASSERT_EQ(map.Capture()->StateAtWorld(10.125, 0.125),
            TraversabilityState::kBlocked);

  const TraversabilityUpdateResult global = map.UpdateGlobal(
      GlobalMap(20U, 1U, 1.0, std::vector<std::int8_t>(20U, 0)));
  const auto snapshot = map.Capture();

  ASSERT_TRUE(global.accepted) << global.reason_code;
  ASSERT_TRUE(snapshot);
  EXPECT_EQ(snapshot->origin_m(), Vec3{});
  EXPECT_EQ(snapshot->StateAtWorld(10.125, 0.125),
            TraversabilityState::kBlocked);
  EXPECT_EQ(snapshot->StateAtWorld(0.125, 0.125), TraversabilityState::kFree);
}

TEST(TraversabilityMapV1, GlobalStampOnlyChangeDoesNotAdvanceRevision) {
  PersistentTraversabilityMap map(Profile());
  GridMap global = GlobalMap(1U, 1U, 1.0, {0});
  ASSERT_TRUE(map.UpdateGlobal(global).accepted);
  const std::uint64_t revision = map.Capture()->revision();
  global.stamp.nanoseconds_since_epoch = 2;

  const TraversabilityUpdateResult duplicate = map.UpdateGlobal(global);

  EXPECT_TRUE(duplicate.accepted) << duplicate.reason_code;
  EXPECT_FALSE(duplicate.changed);
  EXPECT_EQ(map.Capture()->revision(), revision);
}

TEST(TraversabilityMapV1, UsesGlobalFreeAsLazyPrior) {
  PersistentTraversabilityMap map(Profile());
  const TraversabilityUpdateResult global =
      map.UpdateGlobal(GlobalMap(2U, 1U, 1.0, {0, 0}));
  ASSERT_TRUE(global.accepted) << global.reason_code;
  const auto snapshot = UpdateKnownLocal(
      map, LocalMap(1U, 1U, 0.25,
                    {std::numeric_limits<float>::quiet_NaN()},
                    {std::numeric_limits<float>::quiet_NaN()}),
      MapFromOdom(), 1U);

  ASSERT_TRUE(snapshot);
  EXPECT_EQ(snapshot->StateAtWorld(1.75, 0.75), TraversabilityState::kFree);
}

TEST(TraversabilityMapV1, KnownLocalOverridesPrior) {
  PersistentTraversabilityMap map(Profile());
  ASSERT_TRUE(map.UpdateGlobal(GlobalMap(1U, 1U, 1.0, {0})).accepted);
  const GridMap local = LocalMap(1U, 1U, 0.25, {0.9F}, {0.0F});
  const TraversabilityUpdateResult update = map.UpdateLocal(local, MapFromOdom(), 1U);
  ASSERT_TRUE(update.accepted) << update.reason_code;
  const auto snapshot = map.Capture();

  ASSERT_TRUE(snapshot);
  EXPECT_EQ(snapshot->StateAtWorld(0.125, 0.125), TraversabilityState::kBlocked);
  EXPECT_GT(update.metrics.prior_conflicts, 0U);
}

TEST(TraversabilityMapV1, RecomputesPriorConflictsAfterLocalOverwrite) {
  PersistentTraversabilityMap map(Profile());
  ASSERT_TRUE(map.UpdateGlobal(GlobalMap(1U, 1U, 1.0, {0})).accepted);
  const TraversabilityUpdateResult blocked = map.UpdateLocal(
      LocalMap(1U, 1U, 0.25, {0.9F}, {0.0F}), MapFromOdom(), 1U);
  ASSERT_TRUE(blocked.accepted) << blocked.reason_code;
  ASSERT_GT(blocked.metrics.prior_conflicts, 0U);

  const TraversabilityUpdateResult free = map.UpdateLocal(
      LocalMap(1U, 1U, 0.25, {0.0F}, {0.0F}), MapFromOdom(), 2U);

  ASSERT_TRUE(free.accepted) << free.reason_code;
  EXPECT_EQ(free.metrics.prior_conflicts, 0U);
  EXPECT_EQ(map.Capture()->metrics().prior_conflicts, 0U);
}

TEST(TraversabilityMapV1, RecomputesPriorConflictsAfterGlobalPriorChange) {
  PersistentTraversabilityMap map(Profile());
  ASSERT_TRUE(map.UpdateGlobal(GlobalMap(1U, 1U, 1.0, {0})).accepted);
  const TraversabilityUpdateResult local = map.UpdateLocal(
      LocalMap(1U, 1U, 0.25, {0.9F}, {0.0F}), MapFromOdom(), 1U);
  ASSERT_GT(local.metrics.prior_conflicts, 0U);

  const TraversabilityUpdateResult global =
      map.UpdateGlobal(GlobalMap(1U, 1U, 1.0, {100}));

  ASSERT_TRUE(global.accepted) << global.reason_code;
  EXPECT_EQ(global.metrics.prior_conflicts, 0U);
  EXPECT_EQ(map.Capture()->metrics().prior_conflicts, 0U);
}

TEST(TraversabilityMapV1, LocalFreeOverridesGlobalBlockedAndItsInflation) {
  PersistentTraversabilityMap map(Profile(0.2));
  ASSERT_TRUE(map.UpdateGlobal(GlobalMap(1U, 1U, 1.0, {100})).accepted);
  const auto snapshot = UpdateKnownLocal(
      map, LocalMap(1U, 1U, 0.25, {0.0F}, {0.0F}), MapFromOdom(), 1U);

  ASSERT_TRUE(snapshot);
  EXPECT_EQ(snapshot->StateAtWorld(0.125, 0.125), TraversabilityState::kFree);
}

TEST(TraversabilityMapV1, GlobalBlockedInflatesBySourceRectangleOnce) {
  PersistentTraversabilityMap map(Profile(0.2));
  ASSERT_TRUE(map.UpdateGlobal(GlobalMap(2U, 1U, 1.0, {100, 0})).accepted);
  const auto snapshot = UpdateKnownLocal(
      map, LocalMap(1U, 1U, 0.25,
                    {std::numeric_limits<float>::quiet_NaN()},
                    {std::numeric_limits<float>::quiet_NaN()}),
      MapFromOdom(), 1U);

  ASSERT_TRUE(snapshot);
  EXPECT_EQ(snapshot->StateAtWorld(1.10, 0.125), TraversabilityState::kBlocked);
  EXPECT_EQ(snapshot->StateAtWorld(1.30, 0.125), TraversabilityState::kFree);
}

TEST(TraversabilityMapV1, RasterizesRotatedLocalFreeAtCentersOnly) {
  PersistentTraversabilityMap map(Profile());
  const auto snapshot = UpdateKnownLocal(
      map, LocalMap(1U, 1U, 0.1, {0.0F}, {0.0F}),
      MapFromOdom({}, std::numbers::pi / 4.0), 1U);

  ASSERT_TRUE(snapshot);
  // Its cell rectangle intersects the source diamond, but this center is out.
  EXPECT_EQ(snapshot->StateAtWorld(-0.05, 0.15),
            TraversabilityState::kUnknown);
}

TEST(TraversabilityMapV1, UnknownDoesNotEraseHistory) {
  PersistentTraversabilityMap map(Profile());
  ASSERT_TRUE(UpdateKnownLocal(
      map, LocalMap(1U, 1U, 0.25, {0.9F}, {0.0F}), MapFromOdom(), 1U));

  const auto snapshot = UpdateKnownLocal(
      map, LocalMap(1U, 1U, 0.25,
                    {std::numeric_limits<float>::quiet_NaN()},
                    {std::numeric_limits<float>::quiet_NaN()}),
      MapFromOdom(), 2U);

  ASSERT_TRUE(snapshot);
  EXPECT_EQ(snapshot->StateAtWorld(0.125, 0.125), TraversabilityState::kBlocked);
}

TEST(TraversabilityMapV1, IdenticalFrameDoesNotAdvanceRevision) {
  PersistentTraversabilityMap map(Profile());
  const GridMap local = LocalMap(1U, 1U, 0.25, {0.0F}, {0.0F});
  const auto first = UpdateKnownLocal(map, local, MapFromOdom(), 1U);
  ASSERT_TRUE(first);

  const TraversabilityUpdateResult duplicate =
      map.UpdateLocal(local, MapFromOdom(), 2U);
  const auto after = map.Capture();

  EXPECT_TRUE(duplicate.accepted) << duplicate.reason_code;
  EXPECT_FALSE(duplicate.changed);
  ASSERT_TRUE(after);
  EXPECT_EQ(after->revision(), first->revision());
}

TEST(TraversabilityMapV1, RetainsCellsAfterWindowMoves) {
  PersistentTraversabilityMap map(Profile());
  for (std::uint64_t index = 0U; index < 5U; ++index) {
    ASSERT_TRUE(UpdateKnownLocal(
        map, LocalMap(1U, 1U, 0.25, {0.9F}, {0.0F}),
        MapFromOdom(Vec3{.x = static_cast<double>(index), .y = 0.0}),
        index + 1U));
  }

  const auto snapshot = map.Capture();
  ASSERT_TRUE(snapshot);
  for (std::uint64_t index = 0U; index < 5U; ++index) {
    EXPECT_EQ(snapshot->StateAtWorld(static_cast<double>(index) + 0.125, 0.125),
              TraversabilityState::kBlocked)
        << "translated window " << index;
  }
}

TEST(TraversabilityMapV1, RasterizesRotatedLocalObstacleConservatively) {
  PersistentTraversabilityMap map(Profile());
  const auto snapshot = UpdateKnownLocal(
      map, LocalMap(1U, 1U, 0.1, {0.9F}, {0.0F}),
      MapFromOdom({}, std::numbers::pi / 4.0), 1U);

  ASSERT_TRUE(snapshot);
  // This cell rectangle intersects the rotated source cell, while its center
  // lies outside it. A center-only rasterizer would incorrectly leave it free.
  EXPECT_EQ(snapshot->StateAtWorld(-0.05, 0.15),
            TraversabilityState::kBlocked);
}

TEST(TraversabilityMapV1, InflatesOnceAcrossTileBoundary) {
  PersistentTraversabilityMap map(Profile(0.11));
  ASSERT_TRUE(UpdateKnownLocal(
      map, LocalMap(1U, 1U, 0.1,
                    {std::numeric_limits<float>::quiet_NaN()},
                    {std::numeric_limits<float>::quiet_NaN()}),
      MapFromOdom(), 1U));
  const auto snapshot = UpdateKnownLocal(
      map, LocalMap(1U, 1U, 0.1, {0.9F}, {0.0F}, Vec3{.x = 25.5}),
      MapFromOdom(), 2U);

  ASSERT_TRUE(snapshot);
  // The raw obstacle occupies cell 255. Inflation must cross into cell 256
  // without applying a second inflation pass to the first inflated cell.
  EXPECT_EQ(snapshot->StateAtWorld(25.45, 0.05), TraversabilityState::kBlocked);
  EXPECT_EQ(snapshot->StateAtWorld(25.65, 0.05), TraversabilityState::kBlocked);
}

}  // namespace
}  // namespace lunar::pure_planning

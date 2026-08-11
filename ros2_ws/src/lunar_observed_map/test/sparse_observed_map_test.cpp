#include "lunar_observed_map/sparse_observed_map.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace lunar::observed_map {
namespace {

ObservedPatch Patch(
    const std::int64_t x, const std::int64_t y, const double elevation,
    const bool valid, const std::int64_t time_ns = 1) {
  return ObservedPatch{
      .session_id = "session-a",
      .simulation_time_ns = time_ns,
      .cell_zero = {x, y},
      .width = 1U,
      .height = 1U,
      .elevation_m = {elevation},
      .valid = {static_cast<std::uint8_t>(valid)},
  };
}

TEST(SparseObservedMap, AllocatesOnlyForValidObservation) {
  SparseObservedMap map;
  map.ResetSession("session-a");

  const FuseResult ignored = map.Fuse(Patch(
      0, 0, std::numeric_limits<double>::quiet_NaN(), false));
  EXPECT_TRUE(ignored.accepted);
  EXPECT_EQ(ignored.updated_cells, 0U);
  EXPECT_EQ(map.tile_count(), 0U);

  const FuseResult fused = map.Fuse(Patch(0, 0, 1.25, true));
  EXPECT_TRUE(fused.accepted);
  EXPECT_EQ(fused.updated_cells, 1U);
  EXPECT_EQ(fused.allocated_tiles, 1U);
  EXPECT_EQ(map.tile_count(), 1U);
}

TEST(SparseObservedMap, MapsPositiveNegativeAndBoundaryCellsToTiles) {
  EXPECT_EQ(
      SparseObservedMap::Locate({0, 0}),
      (CellLocation{.tile = {0, 0}, .local_x = 0U, .local_y = 0U}));
  EXPECT_EQ(
      SparseObservedMap::Locate({255, 255}),
      (CellLocation{.tile = {0, 0}, .local_x = 255U, .local_y = 255U}));
  EXPECT_EQ(
      SparseObservedMap::Locate({256, 256}),
      (CellLocation{.tile = {1, 1}, .local_x = 0U, .local_y = 0U}));
  EXPECT_EQ(
      SparseObservedMap::Locate({-1, -1}),
      (CellLocation{.tile = {-1, -1}, .local_x = 255U, .local_y = 255U}));
  EXPECT_EQ(
      SparseObservedMap::Locate({-256, -256}),
      (CellLocation{.tile = {-1, -1}, .local_x = 0U, .local_y = 0U}));
  EXPECT_EQ(
      SparseObservedMap::Locate({-257, -257}),
      (CellLocation{.tile = {-2, -2}, .local_x = 255U, .local_y = 255U}));
}

TEST(SparseObservedMap, InvalidPatchCannotEraseHistory) {
  SparseObservedMap map;
  map.ResetSession("session-a");
  ASSERT_TRUE(map.Fuse(Patch(-1, 2, 4.0, true, 10)).accepted);
  ASSERT_TRUE(map.Fuse(Patch(
      -1, 2, std::numeric_limits<double>::quiet_NaN(), false, 20)).accepted);

  const CellEvidence* evidence = map.Find({-1, 2});
  ASSERT_NE(evidence, nullptr);
  EXPECT_TRUE(evidence->valid);
  EXPECT_DOUBLE_EQ(evidence->elevation_m, 4.0);
  EXPECT_EQ(evidence->observation_count, 1U);
  EXPECT_EQ(evidence->last_observed_time_ns, 10);
}

TEST(SparseObservedMap, WelfordMeanVarianceCountAndAgeAreExact) {
  SparseObservedMap map;
  map.ResetSession("session-a");
  ASSERT_TRUE(map.Fuse(Patch(7, 9, 1.0, true, 10)).accepted);
  ASSERT_TRUE(map.Fuse(Patch(7, 9, 2.0, true, 20)).accepted);
  ASSERT_TRUE(map.Fuse(Patch(7, 9, 3.0, true, 30)).accepted);

  const CellEvidence* evidence = map.Find({7, 9});
  ASSERT_NE(evidence, nullptr);
  EXPECT_DOUBLE_EQ(evidence->elevation_m, 2.0);
  EXPECT_DOUBLE_EQ(evidence->elevation_variance(), 1.0);
  EXPECT_EQ(evidence->observation_count, 3U);
  EXPECT_EQ(evidence->last_observed_time_ns, 30);
  EXPECT_DOUBLE_EQ(evidence->observation_age_s(2'000'000'030LL), 2.0);
}

TEST(SparseObservedMap, SessionResetDropsAllOldSceneEvidence) {
  SparseObservedMap map;
  map.ResetSession("session-a");
  ASSERT_TRUE(map.Fuse(Patch(1, 1, 2.0, true)).accepted);
  ASSERT_NE(map.Find({1, 1}), nullptr);

  map.ResetSession("session-b");

  EXPECT_EQ(map.session_id(), "session-b");
  EXPECT_EQ(map.tile_count(), 0U);
  EXPECT_EQ(map.observed_cell_count(), 0U);
  EXPECT_EQ(map.Find({1, 1}), nullptr);
  EXPECT_FALSE(map.Fuse(Patch(1, 1, 3.0, true)).accepted);
}

TEST(SparseObservedMap, RefusesTheFiveHundredThirteenthTileWithoutEviction) {
  SparseObservedMap map;
  map.ResetSession("session-a");
  ASSERT_TRUE(map.Fuse(Patch(0, 0, 9.0, true)).accepted);

  ObservedPatch oversized;
  oversized.session_id = "session-a";
  oversized.simulation_time_ns = 2;
  oversized.cell_zero = {256, 0};
  oversized.width = kMaximumTilesPerSession * kTileSideCells;
  oversized.height = 1U;
  oversized.elevation_m.assign(oversized.width, 1.0);
  oversized.valid.assign(oversized.width, 0U);
  for (std::size_t tile = 0U; tile < kMaximumTilesPerSession; ++tile) {
    oversized.valid[tile * kTileSideCells] = 1U;
  }

  const FuseResult result = map.Fuse(oversized);

  EXPECT_FALSE(result.accepted);
  EXPECT_EQ(result.reason_code, "MAP_TILE_BUDGET_EXCEEDED");
  EXPECT_EQ(map.tile_count(), 1U);
  const CellEvidence* original = map.Find({0, 0});
  ASSERT_NE(original, nullptr);
  EXPECT_DOUBLE_EQ(original->elevation_m, 9.0);
}

}  // namespace
}  // namespace lunar::observed_map

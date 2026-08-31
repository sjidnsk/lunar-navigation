#include <cstddef>
#include <limits>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include "legged/legged_traversal_projection.hpp"
#include "shared/local_terrain_projection.hpp"
#include "shared/map_snapshot.hpp"

namespace lunar::pure_planning::legged {
namespace {

[[nodiscard]] LeggedCapability Capability() {
  return LeggedCapability{
      .body_extent_m = {0.68, 0.33, 0.3},
      .maximum_slope_rad = 1.2,
      .maximum_step_height_m = 0.2,
  };
}

[[nodiscard]] std::shared_ptr<const shared::LocalTerrainProjection> Terrain(
    std::vector<float> occupancy, std::vector<float> elevation) {
  constexpr std::size_t kWidth = 5U;
  GridMap map{
      .frame_id = "odom",
      .stamp = TimePoint{.nanoseconds_since_epoch = 1},
      .width = kWidth,
      .height = kWidth,
      .resolution_m = 0.2,
      .origin_m = {},
  };
  map.layers.emplace("occupancy", GridLayer{.values = std::move(occupancy)});
  map.layers.emplace("elevation", GridLayer{.values = std::move(elevation)});
  auto snapshot = shared::MapSnapshot::Create(
      std::move(map), shared::MapContract::kLocalElevation);
  EXPECT_TRUE(snapshot.ok()) << snapshot.reason_code;
  auto projection = shared::BuildLocalTerrainProjection(snapshot.snapshot);
  EXPECT_TRUE(projection.ok()) << projection.reason_code;
  return std::make_shared<const shared::LocalTerrainProjection>(
      std::move(*projection.value));
}

TEST(LeggedTraversalProjection,
     PrefixQueryIncludesOccupancyHeightSlopeAndStepFeasibility) {
  constexpr std::size_t kWidth = 5U;
  constexpr std::size_t kCount = kWidth * kWidth;
  constexpr std::size_t kCenter = 2U * kWidth + 2U;
  std::vector<float> occupancy(kCount, 0.0F);
  std::vector<float> elevation(kCount, 0.0F);
  elevation[kCenter + 1U] = 0.3F;
  const auto terrain = Terrain(std::move(occupancy), std::move(elevation));
  auto mutable_terrain = std::make_shared<shared::LocalTerrainProjection>(*terrain);
  mutable_terrain->slope_rad[0U] = 1.3F;
  mutable_terrain->free_with_height[kWidth - 1U] = 0U;

  const auto built = BuildLeggedTraversalProjection(
      std::move(mutable_terrain), Capability(), {});

  ASSERT_TRUE(built.ok()) << built.reason_code;
  EXPECT_EQ(built.value->hard_feasible[0U], 0U);
  EXPECT_EQ(built.value->hard_feasible[kWidth - 1U], 0U);
  EXPECT_EQ(built.value->hard_feasible[kCenter], 1U);
  EXPECT_EQ(built.value->step_feasible[kCenter], 0U);
  EXPECT_FALSE(built.value->AllCellsTraversable({0, 0}, {0, 0}));
  EXPECT_TRUE(built.value->AllCellsTraversable({1, 4}, {3, 4}));
  EXPECT_FALSE(built.value->AllCellsTraversable({1, 1}, {3, 3}));
}

TEST(LeggedTraversalProjection, RejectsNonFiniteElevationThroughBaseProjection) {
  constexpr std::size_t kWidth = 5U;
  constexpr std::size_t kCount = kWidth * kWidth;
  constexpr std::size_t kCenter = 2U * kWidth + 2U;
  std::vector<float> elevation(kCount, 0.0F);
  elevation[kCenter] = std::numeric_limits<float>::quiet_NaN();
  const auto terrain = Terrain(std::vector<float>(kCount, 0.0F),
                               std::move(elevation));

  const auto built = BuildLeggedTraversalProjection(terrain, Capability(), {});

  ASSERT_TRUE(built.ok()) << built.reason_code;
  EXPECT_EQ(built.value->hard_feasible[kCenter], 0U);
  EXPECT_FALSE(built.value->AllCellsTraversable({2, 2}, {2, 2}));
}

TEST(LeggedTraversalProjection,
     NecessaryBodyCenterMaskRejectsInscribedDiskAndMapBoundary) {
  constexpr std::size_t kWidth = 5U;
  constexpr std::size_t kCount = kWidth * kWidth;
  constexpr std::size_t kHazard = 2U * kWidth + 1U;
  constexpr std::size_t kAdjacent = 2U * kWidth + 2U;
  constexpr std::size_t kClearInterior = 2U * kWidth + 3U;
  constexpr std::size_t kBoundary = 4U * kWidth + 3U;
  std::vector<float> occupancy(kCount, 0.0F);
  occupancy[kHazard] = 1.0F;

  const auto built = BuildLeggedTraversalProjection(
      Terrain(std::move(occupancy), std::vector<float>(kCount, 0.0F)),
      Capability(), {});

  ASSERT_TRUE(built.ok()) << built.reason_code;
  ASSERT_EQ(built.value->body_center_feasible.size(), kCount);
  EXPECT_EQ(built.value->body_center_feasible[kHazard], 0U);
  EXPECT_EQ(built.value->body_center_feasible[kAdjacent], 0U);
  EXPECT_EQ(built.value->body_center_feasible[kClearInterior], 1U);
  EXPECT_EQ(built.value->body_center_feasible[kBoundary], 0U);
}

TEST(LeggedTraversalProjection,
     NecessaryBodyCenterMaskDoesNotAddHalfCellInflation) {
  constexpr std::size_t kWidth = 5U;
  constexpr std::size_t kCount = kWidth * kWidth;
  constexpr std::size_t kHazard = 2U * kWidth + 1U;
  constexpr std::size_t kAdjacent = 2U * kWidth + 2U;
  std::vector<float> occupancy(kCount, 0.0F);
  occupancy[kHazard] = 1.0F;
  LeggedCapability capability = Capability();
  capability.body_extent_m.y = 0.1;

  const auto built = BuildLeggedTraversalProjection(
      Terrain(std::move(occupancy), std::vector<float>(kCount, 0.0F)),
      capability, {});

  ASSERT_TRUE(built.ok()) << built.reason_code;
  EXPECT_EQ(built.value->body_center_feasible[kAdjacent], 1U);
}

}  // namespace
}  // namespace lunar::pure_planning::legged

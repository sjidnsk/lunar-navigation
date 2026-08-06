#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <ranges>
#include <stop_token>
#include <utility>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include "hierarchical/landing_spatial_index.hpp"
#include "hierarchical/landing_support_field.hpp"
#include "shared/map_snapshot.hpp"
#include "shared/safe_projection.hpp"
#include "test_fixtures.hpp"

namespace lunar::planning::hierarchical {
namespace {

struct LiteralLandingField final {
  std::shared_ptr<const shared::MapSnapshot> map;
  LandingSupportField field;
};

[[nodiscard]] LiteralLandingField MakeLiteralLandingField() {
  PlannerInput input = test::MakeValidHopperInput();
  input.world.global_map = test::MakeFlatMap("map", 17U, 13U, 0.5);
  auto &capability = std::get<HopperCapability>(input.capability);
  capability.body_half_extent_m.x = 0.1;
  capability.body_half_extent_m.y = 0.1;
  capability.minimum_landing_region_area_m2 = 0.1;

  const shared::MapSnapshotBuildResult map =
      shared::MapSnapshot::Create(input.world.global_map);
  EXPECT_TRUE(map.ok()) << map.reason_code;
  const shared::SafeProjectionBuildResult projection =
      shared::BuildSafeProjection(map.snapshot, input.capability,
                                  input.config.map_safety, {});
  EXPECT_TRUE(projection.ok()) << projection.reason_code;
  LandingSupportFieldBuildResult field =
      BuildLandingSupportField(*projection.projection, capability, {});
  EXPECT_TRUE(field.ok()) << field.reason_code;
  return LiteralLandingField{
      .map = map.snapshot,
      .field = std::move(*field.field),
  };
}

TEST(LandingSpatialIndex, MatchesBruteForceReachableCentersExactly) {
  const LiteralLandingField literal = MakeLiteralLandingField();
  constexpr double kReachM = 1.65;
  const Vec2 source{3.37, 2.19};
  const LandingSpatialIndex index(literal.field, kReachM);
  ASSERT_TRUE(index.ok()) << index.reason_code();

  std::vector<LandingNodeId> expected;
  for (const LandingNodeId id : literal.field.SafeCenterIds()) {
    const shared::GridCell cell{
        .x = static_cast<std::int32_t>(id % literal.map->width()),
        .y = static_cast<std::int32_t>(id / literal.map->width()),
    };
    const Vec3 center = literal.map->CellCenter(cell);
    if (std::hypot(center.x - source.x, center.y - source.y) <=
        kReachM + 1.0e-9) {
      expected.push_back(id);
    }
  }

  const std::vector<LandingNodeId> actual = index.Query(source);

  EXPECT_EQ(actual, expected);
  EXPECT_TRUE(std::ranges::is_sorted(actual));
}

TEST(LandingSpatialIndex, HonorsCancellationDuringConstructionAndQuery) {
  const LiteralLandingField literal = MakeLiteralLandingField();
  std::stop_source stop;
  stop.request_stop();

  const LandingSpatialIndex canceled(literal.field, 1.0, stop.get_token());

  EXPECT_FALSE(canceled.ok());
  EXPECT_EQ(canceled.reason_code(), "REQUEST_CANCELED");
  EXPECT_TRUE(canceled.Query({1.0, 1.0}, stop.get_token()).empty());
}

} // namespace
} // namespace lunar::planning::hierarchical

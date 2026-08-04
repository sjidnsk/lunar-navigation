#include <cmath>
#include <cstddef>
#include <numbers>
#include <variant>

#include <gtest/gtest.h>

#include "lunar_planner_core/traversability_projection.hpp"
#include "test_fixtures.hpp"

namespace lunar::planning {
namespace {

void SetElevation(
    GridMap& map, const std::size_t index, const float elevation_m) {
  std::get<std::vector<float>>(map.layers.at("elevation").values).at(index) =
      elevation_m;
}

TEST(TraversabilityProjection, SameTerrainDiffersByCapability) {
  auto wheel_input = test::MakeValidWheelInput();
  wheel_input.world.local_map = test::MakeFlatMap("odom", 5U, 5U, 1.0);
  const double twenty_degrees = 20.0 * std::numbers::pi / 180.0;
  for (std::size_t y = 0U; y < wheel_input.world.local_map.height; ++y) {
    for (std::size_t x = 0U; x < wheel_input.world.local_map.width; ++x) {
      SetElevation(
          wheel_input.world.local_map,
          y * wheel_input.world.local_map.width + x,
          static_cast<float>(static_cast<double>(x) * std::tan(twenty_degrees)));
    }
  }

  auto strict_wheel = std::get<WheeledCapability>(wheel_input.capability);
  strict_wheel.maximum_slope_rad = 15.0 * std::numbers::pi / 180.0;
  strict_wheel.minimum_clearance_m = 0.0;
  auto permissive_legged =
      std::get<LeggedCapability>(test::MakeValidLeggedInput().capability);
  permissive_legged.maximum_slope_rad = 25.0 * std::numbers::pi / 180.0;
  permissive_legged.maximum_step_height_m = 1.0;
  permissive_legged.minimum_body_clearance_m = 0.0;

  const auto wheel = ProjectTraversability(
      wheel_input.world, PlatformCapability{strict_wheel},
      wheel_input.config.map_safety, {});
  const auto legged = ProjectTraversability(
      wheel_input.world, PlatformCapability{permissive_legged},
      wheel_input.config.map_safety, {});

  ASSERT_TRUE(wheel.ok()) << wheel.reason_code;
  ASSERT_TRUE(legged.ok()) << legged.reason_code;
  ASSERT_EQ(wheel.projection->hard_feasible.size(), 25U);
  ASSERT_EQ(legged.projection->hard_feasible.size(), 25U);
  EXPECT_EQ(wheel.projection->hard_feasible[12U], 0U);
  EXPECT_EQ(legged.projection->hard_feasible[12U], 1U);
}

TEST(TraversabilityProjection, RejectsInvalidLocalMapWithStableReasonCode) {
  auto input = test::MakeValidWheelInput();
  input.world.local_map.layers.erase("forbidden");

  const auto result = ProjectTraversability(
      input.world, input.capability, input.config.map_safety, {});

  EXPECT_FALSE(result.ok());
  EXPECT_FALSE(result.projection.has_value());
  EXPECT_EQ(result.reason_code, "MISSING_MAP_LAYER_FORBIDDEN");
}

}  // namespace
}  // namespace lunar::planning

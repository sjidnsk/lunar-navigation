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

void SetKnown(GridMap& map, const std::size_t index, const bool known) {
  std::get<std::vector<std::uint8_t>>(
      map.layers.at("valid_mask").values).at(index) =
      static_cast<std::uint8_t>(known);
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
  ASSERT_EQ(wheel.projection->intrinsic_feasible.size(), 25U);
  ASSERT_EQ(legged.projection->intrinsic_feasible.size(), 25U);
  EXPECT_EQ(wheel.projection->hard_feasible[12U], 0U);
  EXPECT_EQ(legged.projection->hard_feasible[12U], 1U);
  EXPECT_EQ(wheel.projection->intrinsic_feasible[12U], 0U);
  EXPECT_EQ(legged.projection->intrinsic_feasible[12U], 1U);
}

TEST(TraversabilityProjection,
     UnknownNeighborElevationDoesNotFabricateSlopeAtKnownBoundary) {
  auto input = test::MakeValidWheelInput();
  input.world.local_map = test::MakeFlatMap("odom", 5U, 5U, 1.0);
  auto& capability = std::get<WheeledCapability>(input.capability);
  capability.minimum_clearance_m = 0.0;
  for (std::size_t index = 0U;
       index < input.world.local_map.CellCount(); ++index) {
    SetElevation(input.world.local_map, index, -1385.0F);
  }
  const std::size_t center = 2U * input.world.local_map.width + 2U;
  const std::size_t unknown_positive_x = center + 1U;
  SetKnown(input.world.local_map, unknown_positive_x, false);
  SetElevation(input.world.local_map, unknown_positive_x, 0.0F);

  const auto result = ProjectTraversability(
      input.world, input.capability, input.config.map_safety, {});

  ASSERT_TRUE(result.ok()) << result.reason_code;
  EXPECT_EQ(result.projection->intrinsic_feasible.at(center), 1U);
  EXPECT_NEAR(result.projection->slope_rad.at(center), 0.0F, 1.0e-6F);
}

TEST(TraversabilityProjection,
     UnknownNeighborElevationKeepsLeggedBoundaryIntrinsicFeasible) {
  auto input = test::MakeValidLeggedInput();
  input.world.local_map = test::MakeFlatMap("odom", 5U, 5U, 1.0);
  auto& capability = std::get<LeggedCapability>(input.capability);
  capability.minimum_body_clearance_m = 0.0;
  for (std::size_t index = 0U;
       index < input.world.local_map.CellCount(); ++index) {
    SetElevation(input.world.local_map, index, -1385.0F);
  }
  const std::size_t center = 2U * input.world.local_map.width + 2U;
  const std::size_t unknown_positive_x = center + 1U;
  SetKnown(input.world.local_map, unknown_positive_x, false);
  SetElevation(input.world.local_map, unknown_positive_x, 0.0F);

  const auto result = ProjectTraversability(
      input.world, input.capability, input.config.map_safety, {});

  ASSERT_TRUE(result.ok()) << result.reason_code;
  EXPECT_EQ(result.projection->intrinsic_feasible.at(center), 1U);
}

TEST(TraversabilityProjection, IsolatedKnownCellRemainsIntrinsicInfeasible) {
  auto input = test::MakeValidWheelInput();
  input.world.local_map = test::MakeFlatMap("odom", 5U, 5U, 1.0);
  const std::size_t center = 2U * input.world.local_map.width + 2U;
  for (const std::size_t neighbor : {
           center - input.world.local_map.width,
           center - 1U,
           center + 1U,
           center + input.world.local_map.width,
       }) {
    SetKnown(input.world.local_map, neighbor, false);
  }

  const auto result = ProjectTraversability(
      input.world, input.capability, input.config.map_safety, {});

  ASSERT_TRUE(result.ok()) << result.reason_code;
  EXPECT_EQ(result.projection->intrinsic_feasible.at(center), 0U);
  EXPECT_FALSE(std::isfinite(result.projection->slope_rad.at(center)));
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

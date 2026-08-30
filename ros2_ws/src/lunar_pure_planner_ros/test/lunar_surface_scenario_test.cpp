#include <gtest/gtest.h>

#include "lunar_pure_planner_ros/lunar_surface_local_map.hpp"
#include "lunar_pure_planner_ros/lunar_surface_scenario.hpp"

namespace lunar::pure_planner_ros {
namespace {

TEST(LunarSurfaceScenario, HasFixedGeometryAndReachableDefaultGoal) {
  const LunarSurfaceScenario scenario = BuildLunarSurfaceScenario(20260823U);

  EXPECT_EQ(scenario.width, 1000U);
  EXPECT_EQ(scenario.height, 1000U);
  EXPECT_DOUBLE_EQ(scenario.resolution_m, 1.0);
  EXPECT_DOUBLE_EQ(scenario.origin_x_m, -500.0);
  EXPECT_DOUBLE_EQ(scenario.origin_y_m, -500.0);
  EXPECT_DOUBLE_EQ(static_cast<double>(scenario.width) * scenario.resolution_m, 1000.0);
  EXPECT_DOUBLE_EQ(static_cast<double>(scenario.height) * scenario.resolution_m, 1000.0);
  EXPECT_EQ(scenario.occupancy.size(), 1'000'000U);
  EXPECT_EQ(scenario.elevation_m.size(), 1'000'000U);
  EXPECT_FALSE(scenario.Occupied(scenario.start_cell));
  EXPECT_FALSE(scenario.Occupied(scenario.default_goal_cell));
  EXPECT_TRUE(CellsConnected(scenario, scenario.start_cell,
                             scenario.default_goal_cell));
}

TEST(LunarSurfaceScenario, RetainsCraterRimWithoutAnArtificialDemoCorridor) {
  const LunarSurfaceScenario scenario = BuildLunarSurfaceScenario(20260823U);

  // This fixed-seed cell belongs to a crater rim.  It must remain an obstacle
  // rather than being cleared to guarantee a straight demo route.
  EXPECT_TRUE(scenario.Occupied(LunarSurfaceCell{238U, 101U}));
}

TEST(LunarSurfaceScenario, SamplesSubCellTerrainFromContinuousTruth) {
  const LunarSurfaceScenario scenario = BuildLunarSurfaceScenario(20260823U);

  const auto first = scenario.Sample(0.1, 0.1);
  const auto second = scenario.Sample(0.3, 0.1);

  ASSERT_TRUE(first.has_value());
  ASSERT_TRUE(second.has_value());
  EXPECT_NE(first->elevation_m, second->elevation_m);
  EXPECT_FALSE(scenario.Sample(-500.1, 0.0).has_value());
  EXPECT_FALSE(scenario.Sample(500.0, 0.0).has_value());
}

TEST(LunarSurfaceScenario, BuildsSixtyFourMetreHighResolutionLocalRaster) {
  const LunarSurfaceScenario scenario = BuildLunarSurfaceScenario(20260823U);

  const LunarSurfaceLocalRaster raster =
      BuildLunarSurfaceLocalRaster(scenario, 0.0, 0.0);

  EXPECT_EQ(raster.width, 320U);
  EXPECT_EQ(raster.height, 320U);
  EXPECT_DOUBLE_EQ(raster.resolution_m, 0.2);
  EXPECT_DOUBLE_EQ(raster.length_x_m(), 64.0);
  EXPECT_DOUBLE_EQ(raster.length_y_m(), 64.0);
  EXPECT_EQ(raster.occupancy.size(), 320U * 320U);
  EXPECT_EQ(raster.elevation_m.size(), 320U * 320U);
}

}  // namespace
}  // namespace lunar::pure_planner_ros

#include <gtest/gtest.h>

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

}  // namespace
}  // namespace lunar::pure_planner_ros

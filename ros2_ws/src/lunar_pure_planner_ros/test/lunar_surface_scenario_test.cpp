#include <gtest/gtest.h>

#include "lunar_pure_planner_ros/lunar_surface_scenario.hpp"

namespace lunar::pure_planner_ros {
namespace {

TEST(LunarSurfaceScenario, HasFixedGeometryAndReachableDefaultGoal) {
  const LunarSurfaceScenario scenario = BuildLunarSurfaceScenario(20260823U);

  EXPECT_EQ(scenario.width, 500U);
  EXPECT_EQ(scenario.height, 500U);
  EXPECT_DOUBLE_EQ(scenario.resolution_m, 0.2);
  EXPECT_DOUBLE_EQ(scenario.origin_x_m, -50.0);
  EXPECT_DOUBLE_EQ(scenario.origin_y_m, -50.0);
  EXPECT_DOUBLE_EQ(static_cast<double>(scenario.width) * scenario.resolution_m, 100.0);
  EXPECT_DOUBLE_EQ(static_cast<double>(scenario.height) * scenario.resolution_m, 100.0);
  EXPECT_EQ(scenario.occupancy.size(), 250'000U);
  EXPECT_EQ(scenario.elevation_m.size(), 250'000U);
  EXPECT_FALSE(scenario.Occupied(scenario.start_cell));
  EXPECT_FALSE(scenario.Occupied(scenario.default_goal_cell));
  EXPECT_TRUE(CellsConnected(scenario, scenario.start_cell,
                             scenario.default_goal_cell));
}

}  // namespace
}  // namespace lunar::pure_planner_ros

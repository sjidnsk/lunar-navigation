#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include <gtest/gtest.h>

#include "lunar_pure_planner_ros/lunar_surface_demo_motion.hpp"
#include "lunar_pure_planner_ros/lunar_surface_local_map.hpp"
#include "lunar_pure_planner_ros/lunar_surface_scenario.hpp"

namespace lunar::pure_planner_ros {
namespace {

void AppendUnique(std::vector<LunarSurfaceCell>& cells,
                  const LunarSurfaceCell cell) {
  if (cells.empty() || cells.back() != cell) {
    cells.push_back(cell);
  }
}

[[nodiscard]] std::vector<LunarSurfaceCell> Supercover(
    const LunarSurfaceCell start, const LunarSurfaceCell end) {
  std::vector<LunarSurfaceCell> cells;
  std::int32_t x = static_cast<std::int32_t>(start.x);
  std::int32_t y = static_cast<std::int32_t>(start.y);
  const std::int32_t dx = static_cast<std::int32_t>(end.x) - x;
  const std::int32_t dy = static_cast<std::int32_t>(end.y) - y;
  const std::int32_t sign_x = dx < 0 ? -1 : 1;
  const std::int32_t sign_y = dy < 0 ? -1 : 1;
  const std::int64_t nx = std::llabs(static_cast<std::int64_t>(dx));
  const std::int64_t ny = std::llabs(static_cast<std::int64_t>(dy));
  std::int64_t ix{};
  std::int64_t iy{};
  AppendUnique(cells, start);
  while (ix < nx || iy < ny) {
    const std::int64_t decision = (1 + 2 * ix) * ny - (1 + 2 * iy) * nx;
    if (decision == 0) {
      AppendUnique(cells, LunarSurfaceCell{
                              .x = static_cast<std::size_t>(x + sign_x),
                              .y = static_cast<std::size_t>(y)});
      AppendUnique(cells, LunarSurfaceCell{
                              .x = static_cast<std::size_t>(x),
                              .y = static_cast<std::size_t>(y + sign_y)});
      x += sign_x;
      y += sign_y;
      ++ix;
      ++iy;
    } else if (decision < 0) {
      x += sign_x;
      ++ix;
    } else {
      y += sign_y;
      ++iy;
    }
    AppendUnique(cells,
                 LunarSurfaceCell{.x = static_cast<std::size_t>(x),
                                  .y = static_cast<std::size_t>(y)});
  }
  return cells;
}

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

TEST(LunarSurfaceScenario,
     FixedValidationGoalsAreReachableLongAndRequireObstacleDetours) {
  const LunarSurfaceScenario scenario = BuildLunarSurfaceScenario(20260823U);
  constexpr std::array<LunarSurfaceCell, 3U> kValidationGoals{{
      {325U, 200U},
      {400U, 300U},
      {250U, 700U},
  }};

  for (const LunarSurfaceCell goal : kValidationGoals) {
    SCOPED_TRACE(::testing::Message() << "goal=(" << goal.x << "," << goal.y
                                      << ")");
    EXPECT_FALSE(scenario.Occupied(goal));
    EXPECT_TRUE(CellsConnected(scenario, scenario.start_cell, goal));
    EXPECT_GT(std::hypot(static_cast<double>(goal.x) -
                             static_cast<double>(scenario.start_cell.x),
                         static_cast<double>(goal.y) -
                             static_cast<double>(scenario.start_cell.y)),
              100.0);
    const auto direct_cells = Supercover(scenario.start_cell, goal);
    EXPECT_TRUE(std::ranges::any_of(
        direct_cells,
        [&](const LunarSurfaceCell cell) { return scenario.Occupied(cell); }));
  }
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

TEST(LunarSurfaceScenario, DemoMotionConsumesOneDistanceBudgetPerTick) {
  nav_msgs::msg::Path path;
  for (const double x_m : {0.0, 0.2, 0.4, 0.6, 1.0}) {
    geometry_msgs::msg::PoseStamped pose;
    pose.pose.position.x = x_m;
    path.poses.push_back(pose);
  }
  std::size_t next_pose = 1U;
  double x_m = 0.0;
  double y_m = 0.0;

  AdvanceAlongDemoPath(path, next_pose, x_m, y_m, 0.5);

  EXPECT_DOUBLE_EQ(x_m, 0.5);
  EXPECT_DOUBLE_EQ(y_m, 0.0);
  EXPECT_EQ(next_pose, 3U);
}

}  // namespace
}  // namespace lunar::pure_planner_ros

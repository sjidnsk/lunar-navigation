#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include "hierarchical/global_route_planner.hpp"
#include "test_fixtures.hpp"

namespace lunar::planning::hierarchical {
namespace {

[[nodiscard]] PlannerInput GroundInput(const PlatformType platform) {
  PlannerInput input = platform == PlatformType::kWheeled
                           ? test::MakeValidWheelInput()
                           : test::MakeValidLeggedInput();
  input.world.global_map = test::MakeFlatMap("map", 20U, 12U, 1.0);
  input.world.local_map = test::MakeFlatMap("odom", 12U, 8U, 1.0);
  input.config.global_map.base_resolution_m = 1.0;
  input.goal_map = GoalRegion{
      .goal_id = "far-goal",
      .target =
          PointGoal{
              .position_m = {18.5, 5.5, 0.0},
              .tolerance_m = 0.2,
          },
  };
  if (platform == PlatformType::kWheeled) {
    std::get<WheeledState>(input.current_state).pose.position_m = {1.5, 5.5,
                                                                   0.0};
  } else {
    std::get<LeggedState>(input.current_state).body_pose.position_m = {1.5, 5.5,
                                                                       0.5};
  }
  return input;
}

void SetObstacle(GridMap &map, const std::size_t x, const std::size_t y) {
  auto &values =
      std::get<std::vector<std::uint8_t>>(map.layers.at("obstacle").values);
  values.at(y * map.width + x) = 1U;
}

void SetElevation(GridMap &map, const std::size_t x, const std::size_t y,
                  const float elevation_m) {
  auto &values =
      std::get<std::vector<float>>(map.layers.at("elevation").values);
  values.at(y * map.width + x) = elevation_m;
}

TEST(GlobalRoutePlanner, PlansCompleteDeterministicWheelRoute) {
  const PlannerInput input = GroundInput(PlatformType::kWheeled);

  const auto first = PlanGroundGlobalRoute(input);
  const auto second = PlanGroundGlobalRoute(input);

  ASSERT_TRUE(first.ok());
  ASSERT_TRUE(second.ok());
  EXPECT_EQ(first.reason_code, "GLOBAL_ROUTE_AVAILABLE");
  EXPECT_EQ(first.route->raw_cells, second.route->raw_cells);
  EXPECT_EQ(first.route->simplified_cells, second.route->simplified_cells);
  EXPECT_EQ(first.route->poses_map, second.route->poses_map);
  EXPECT_DOUBLE_EQ(first.route->cost, second.route->cost);
  ASSERT_GE(first.route->poses_map.size(), 2U);
  EXPECT_NEAR(first.route->poses_map.front().position_m.x, 1.5, 1.0e-9);
  EXPECT_NEAR(first.route->poses_map.front().position_m.y, 5.5, 1.0e-9);
  EXPECT_NEAR(first.route->poses_map.back().position_m.x, 18.5, 0.2);
  EXPECT_NEAR(first.route->poses_map.back().position_m.y, 5.5, 0.2);
}

TEST(GlobalRoutePlanner, KeepsWheelFootprintClearOfGlobalObstacles) {
  PlannerInput input = GroundInput(PlatformType::kWheeled);
  input.world.global_map = test::MakeFlatMap("map", 12U, 9U, 1.0);
  input.world.local_map = test::MakeFlatMap("odom", 12U, 9U, 1.0);
  auto &state = std::get<WheeledState>(input.current_state);
  state.pose.position_m = {1.5, 4.5, 0.0};
  auto &capability = std::get<WheeledCapability>(input.capability);
  capability.footprint_xy_m = {
      {-0.9, -0.9}, {0.9, -0.9}, {0.9, 0.9}, {-0.9, 0.9}};
  capability.minimum_clearance_m = 0.0;
  input.goal_map.target = PointGoal{
      .position_m = {10.5, 4.5, 0.0},
      .tolerance_m = 0.2,
  };
  SetObstacle(input.world.global_map, 6U, 5U);

  const auto result = PlanGroundGlobalRoute(input);

  ASSERT_TRUE(result.ok());
  ASSERT_GT(result.route->poses_map.size(), 2U);
  EXPECT_TRUE(std::ranges::all_of(
      result.route->raw_cells, [](const shared::GridCell cell) {
        const double center_x = static_cast<double>(cell.x) + 0.5;
        const double center_y = static_cast<double>(cell.y) + 0.5;
        return std::hypot(center_x - 6.5, center_y - 5.5) >= 2.68;
      }));
}

TEST(GlobalRoutePlanner, DistinguishesGoalInfeasibleFromDisconnectedRoute) {
  PlannerInput blocked_goal = GroundInput(PlatformType::kWheeled);
  SetObstacle(blocked_goal.world.global_map, 18U, 5U);

  const auto infeasible = PlanGroundGlobalRoute(blocked_goal);

  EXPECT_EQ(infeasible.outcome, PlanningOutcome::kGoalInfeasible);
  EXPECT_EQ(infeasible.reason_code, "GLOBAL_GOAL_INFEASIBLE");
  EXPECT_FALSE(infeasible.route.has_value());

  PlannerInput wall = GroundInput(PlatformType::kWheeled);
  for (std::size_t y = 0U; y < wall.world.global_map.height; ++y) {
    SetObstacle(wall.world.global_map, 10U, y);
  }

  const auto disconnected = PlanGroundGlobalRoute(wall);

  EXPECT_EQ(disconnected.outcome, PlanningOutcome::kNoKnownSafeRoute);
  EXPECT_EQ(disconnected.reason_code, "GLOBAL_NO_KNOWN_SAFE_ROUTE");
  EXPECT_FALSE(disconnected.route.has_value());
}

TEST(GlobalRoutePlanner, UsesFrozenMapFromOdomForTheStartPose) {
  PlannerInput input = GroundInput(PlatformType::kWheeled);
  input.world.global_map = test::MakeFlatMap("map", 30U, 30U, 1.0);
  std::get<WheeledState>(input.current_state).pose.position_m = {1.5, 2.5, 0.0};
  input.world.map_from_odom.translation_m = {10.0, 10.0, 0.0};
  input.world.map_from_odom.rotation =
      test::YawQuaternion(std::numbers::pi / 2.0);
  input.goal_map.target = PointGoal{
      .position_m = {7.5, 15.5, 0.0},
      .tolerance_m = 0.2,
  };

  const auto result = PlanGroundGlobalRoute(input);

  ASSERT_TRUE(result.ok());
  ASSERT_FALSE(result.route->poses_map.empty());
  EXPECT_NEAR(result.route->poses_map.front().position_m.x, 7.5, 1.0e-9);
  EXPECT_NEAR(result.route->poses_map.front().position_m.y, 11.5, 1.0e-9);
}

TEST(GlobalRoutePlanner, AcceptsAPlanarGoalRegion) {
  PlannerInput input = GroundInput(PlatformType::kLegged);
  input.goal_map = GoalRegion{
      .goal_id = "region",
      .target =
          PlanarRegionGoal{
              .boundary_m = {{16.0, 4.0, 0.0},
                             {19.0, 4.0, 0.0},
                             {19.0, 7.0, 0.0},
                             {16.0, 7.0, 0.0}},
              .normal_tolerance_m = 0.2,
          },
  };

  const auto result = PlanGroundGlobalRoute(input);

  ASSERT_TRUE(result.ok());
  const Vec3 end = result.route->poses_map.back().position_m;
  EXPECT_GE(end.x, 16.0);
  EXPECT_LE(end.x, 19.0);
  EXPECT_GE(end.y, 4.0);
  EXPECT_LE(end.y, 7.0);
}

TEST(GlobalRoutePlanner, RepresentsATightPointGoalInsideACoarseCell) {
  PlannerInput input = GroundInput(PlatformType::kWheeled);
  input.world.global_map = test::MakeFlatMap("map", 5U, 2U, 2.0);
  input.world.local_map = test::MakeFlatMap("odom", 4U, 2U, 1.0);
  input.config.global_map.maximum_axis_cells = 8U;
  input.config.global_map.maximum_cells = 64U;
  std::get<WheeledState>(input.current_state).pose.position_m = {0.5, 1.0, 0.0};
  input.goal_map.target = PointGoal{
      .position_m = {9.9, 1.0, 0.0},
      .tolerance_m = 0.05,
  };

  const auto result = PlanGroundGlobalRoute(input);

  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result.global_level, 1U);
  EXPECT_EQ(result.route->raw_cells.back(), (shared::GridCell{4, 0}));
}

TEST(GlobalRoutePlanner, AppliesPlatformSpecificTerrainLimits) {
  PlannerInput wheel = GroundInput(PlatformType::kWheeled);
  PlannerInput legged = GroundInput(PlatformType::kLegged);
  wheel.world.global_map = test::MakeFlatMap("map", 8U, 1U, 1.0);
  legged.world.global_map = wheel.world.global_map;
  wheel.goal_map.target =
      PointGoal{.position_m = {6.5, 0.5, 0.6}, .tolerance_m = 0.2};
  legged.goal_map = wheel.goal_map;
  std::get<WheeledState>(wheel.current_state).pose.position_m = {1.5, 0.5, 0.1};
  std::get<LeggedState>(legged.current_state).body_pose.position_m = {1.5, 0.5,
                                                                      0.6};
  std::get<WheeledCapability>(wheel.capability).maximum_slope_rad = 0.05;
  std::get<LeggedCapability>(legged.capability).maximum_slope_rad = 0.4;
  wheel.config.map_safety.project_maximum_slope_rad = 0.5;
  legged.config.map_safety.project_maximum_slope_rad = 0.5;
  for (std::size_t x = 0U; x < 8U; ++x) {
    SetElevation(wheel.world.global_map, x, 0U, 0.1F * static_cast<float>(x));
    SetElevation(legged.world.global_map, x, 0U, 0.1F * static_cast<float>(x));
  }

  const auto wheel_result = PlanGroundGlobalRoute(wheel);
  const auto legged_result = PlanGroundGlobalRoute(legged);

  EXPECT_FALSE(wheel_result.ok());
  ASSERT_TRUE(legged_result.ok());
  EXPECT_NE(wheel_result.reason_code, legged_result.reason_code);
}

TEST(GlobalRoutePlanner, PreservesFinalYawConstraintOnlyAtRouteEnd) {
  PlannerInput input = GroundInput(PlatformType::kWheeled);
  input.goal_map.yaw_rad = std::numbers::pi / 2.0;
  input.goal_map.yaw_tolerance_rad = 0.1;

  const auto result = PlanGroundGlobalRoute(input);

  ASSERT_TRUE(result.ok());
  const Quaternion orientation = result.route->poses_map.back().orientation;
  const double yaw = std::atan2(
      2.0 * (orientation.w * orientation.z + orientation.x * orientation.y),
      1.0 - 2.0 * (orientation.y * orientation.y +
                   orientation.z * orientation.z));
  EXPECT_NEAR(yaw, std::numbers::pi / 2.0, 1.0e-9);
}

} // namespace
} // namespace lunar::planning::hierarchical

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <stop_token>
#include <string>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include "hierarchical/local_frontier.hpp"
#include "legged/legged_planner.hpp"
#include "test_fixtures.hpp"
#include "wheel/wheel_lattice.hpp"
#include "wheel/wheel_planner.hpp"

namespace lunar::planning::hierarchical {
namespace {

template <class Planner, class Problem>
concept PlansProblem = requires(const Planner &planner,
                                const Problem &problem) {
  { planner.Plan(problem) } -> std::same_as<PlannerOutput>;
};

static_assert(PlansProblem<wheel::WheelPlanner, LocalPlanningProblem>);
static_assert(PlansProblem<legged::LeggedPlanner, LocalPlanningProblem>);
static_assert(!PlansProblem<wheel::WheelPlanner, PlannerInput>);
static_assert(!PlansProblem<legged::LeggedPlanner, PlannerInput>);

[[nodiscard]] PlannerInput FrontierInput(const PlatformType platform) {
  PlannerInput input = platform == PlatformType::kWheeled
                           ? test::MakeValidWheelInput()
                           : test::MakeValidLeggedInput();
  input.world.global_map = test::MakeFlatMap("map", 24U, 10U, 1.0);
  input.world.local_map = test::MakeFlatMap("odom", 14U, 10U, 1.0);
  input.config.global_map.base_resolution_m = 1.0;
  input.world.map_from_odom = RigidTransform{
      .parent_frame = "map",
      .child_frame = "odom",
      .stamp = input.state_time,
      .translation_m = {10.0, 1.0, 0.0},
  };
  input.goal_map = GoalRegion{
      .goal_id = "far-goal",
      .target = PointGoal{.position_m = {20.5, 4.5, 0.0}, .tolerance_m = 0.2},
  };
  if (platform == PlatformType::kWheeled) {
    std::get<WheeledState>(input.current_state).pose.position_m = {2.5, 3.5,
                                                                   0.0};
  } else {
    std::get<LeggedState>(input.current_state).body_pose.position_m = {2.5, 3.5,
                                                                       0.5};
  }
  return input;
}

[[nodiscard]] GlobalRoute StraightRoute() {
  return GlobalRoute{
      .raw_cells = {{2, 3}, {10, 3}},
      .simplified_cells = {{2, 3}, {10, 3}},
      .poses_map =
          {
              Pose3{.position_m = {12.5, 4.5, 0.0}},
              Pose3{.position_m = {20.5, 4.5, 0.0}},
          },
      .cost = 8.0,
  };
}

[[nodiscard]] GlobalRoute AngledRoute() {
  return GlobalRoute{
      .raw_cells = {{2, 3}, {10, 7}},
      .simplified_cells = {{2, 3}, {10, 7}},
      .poses_map =
          {
              Pose3{.position_m = {12.5, 4.5, 0.0}},
              Pose3{.position_m = {20.5, 8.5, 0.0}},
          },
      .cost = std::hypot(8.0, 4.0),
  };
}

[[nodiscard]] Vec3 PointTarget(const GoalRegion &goal) {
  return std::get<PointGoal>(goal.target).position_m;
}

void SetValid(GridMap &map, const std::size_t x, const std::size_t y,
              const std::uint8_t value) {
  auto &valid =
      std::get<std::vector<std::uint8_t>>(map.layers.at("valid_mask").values);
  valid.at(y * map.width + x) = value;
}

[[nodiscard]] double DistanceToHorizontalPrefix(const Vec3 point,
                                                const double start_x,
                                                const double end_x,
                                                const double y) {
  const double nearest_x = std::clamp(point.x, start_x, end_x);
  return std::hypot(point.x - nearest_x, point.y - y);
}

TEST(LocalFrontier, TransformsRouteAndReturnsEveryWheelCandidateFarToNear) {
  const PlannerInput input = FrontierInput(PlatformType::kWheeled);

  const LocalFrontierResult result =
      BuildLocalFrontiers(input, StraightRoute());

  ASSERT_TRUE(result.ok()) << result.reason_code;
  EXPECT_EQ(result.reason_code, "LOCAL_FRONTIERS_AVAILABLE");
  ASSERT_EQ(result.problems.size(), 4U);
  ASSERT_EQ(result.frontier_distances_m.size(), result.problems.size());
  EXPECT_NEAR(result.frontier_distances_m[0], 4.0, 1.0e-9);
  EXPECT_GT(result.frontier_distances_m[0], result.frontier_distances_m[1]);
  EXPECT_GT(result.frontier_distances_m[1], result.frontier_distances_m[2]);
  EXPECT_GT(result.frontier_distances_m[2], result.frontier_distances_m[3]);
  EXPECT_NEAR(PointTarget(result.problems[0].goal_odom).x, 6.5, 1.0e-9);
  EXPECT_NEAR(PointTarget(result.problems[0].goal_odom).y, 3.5, 1.0e-9);
  EXPECT_EQ(result.problems[0].local_map_view.frame_id, "odom");
}

TEST(LocalFrontier, WheelTurnFrontierRejectsAHeadingThatMissesTheRouteTurn) {
  PlannerInput input = FrontierInput(PlatformType::kWheeled);
  std::get<WheeledCapability>(input.capability).motion_primitives.push_back(
      WheelMotionPrimitive{
          .primitive_id = "production-sized-turn",
          .kind = WheelPrimitiveKind::kSpinCounterclockwise,
          .relative_end_pose =
              Pose3{.orientation = test::YawQuaternion(
                        std::numbers::pi / 4.0)},
          .nominal_duration = std::chrono::seconds{1},
      });

  const LocalFrontierResult result =
      BuildLocalFrontiers(input, AngledRoute());

  ASSERT_TRUE(result.ok()) << result.reason_code;
  ASSERT_FALSE(result.problems.empty());
  const GoalRegion &goal = result.problems.front().goal_odom;
  const Vec3 target = PointTarget(goal);
  EXPECT_FALSE(wheel::GoalContainsPose(
      goal, wheel::WheelPose{.position_m = target, .yaw_rad = 0.0}));
  EXPECT_TRUE(wheel::GoalContainsPose(
      goal,
      wheel::WheelPose{
          .position_m = target, .yaw_rad = std::numbers::pi / 4.0}));
}

TEST(LocalFrontier, AppliesThreeMetreLeggedHorizon) {
  const PlannerInput input = FrontierInput(PlatformType::kLegged);

  const LocalFrontierResult result =
      BuildLocalFrontiers(input, StraightRoute());

  ASSERT_TRUE(result.ok()) << result.reason_code;
  ASSERT_EQ(result.problems.size(), 3U);
  EXPECT_NEAR(result.frontier_distances_m.front(), 3.0, 1.0e-9);
  EXPECT_NEAR(PointTarget(result.problems.front().goal_odom).x, 5.5, 1.0e-9);
}

TEST(LocalFrontier, StopsAtTheFirstUnknownCellInTheRoutePrefix) {
  PlannerInput input = FrontierInput(PlatformType::kWheeled);
  SetValid(input.world.local_map, 4U, 3U, 0U);

  const LocalFrontierResult result =
      BuildLocalFrontiers(input, StraightRoute());

  ASSERT_TRUE(result.ok()) << result.reason_code;
  ASSERT_FALSE(result.problems.empty());
  EXPECT_LT(result.frontier_distances_m.front(), 2.0);
  EXPECT_LT(PointTarget(result.problems.front().goal_odom).x, 4.0);
}

TEST(LocalFrontier, ReportsInsufficientCoverageWithoutAForwardSafePoint) {
  PlannerInput input = FrontierInput(PlatformType::kWheeled);
  SetValid(input.world.local_map, 3U, 3U, 0U);

  const LocalFrontierResult result =
      BuildLocalFrontiers(input, StraightRoute());

  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.reason_code, "LOCAL_MAP_COVERAGE_INSUFFICIENT");
  EXPECT_TRUE(result.problems.empty());
}

TEST(LocalFrontier, KeepsThePlatformMarginAwayFromTheLocalMapEdge) {
  PlannerInput input = FrontierInput(PlatformType::kWheeled);
  input.world.local_map = test::MakeFlatMap("odom", 7U, 8U, 1.0);

  const LocalFrontierResult result =
      BuildLocalFrontiers(input, StraightRoute());

  ASSERT_TRUE(result.ok()) << result.reason_code;
  EXPECT_NEAR(result.frontier_distances_m.front(), 3.0, 1.0e-9);
  EXPECT_NEAR(PointTarget(result.problems.front().goal_odom).x, 5.5, 1.0e-9);
}

TEST(LocalFrontier, MasksEveryCellOutsideTheHorizonCorridorIntersection) {
  const PlannerInput input = FrontierInput(PlatformType::kWheeled);

  const LocalFrontierResult result =
      BuildLocalFrontiers(input, StraightRoute());

  ASSERT_TRUE(result.ok()) << result.reason_code;
  const GridMap &view = result.problems.front().local_map_view;
  const auto &valid =
      std::get<std::vector<std::uint8_t>>(view.layers.at("valid_mask").values);
  const auto &forbidden =
      std::get<std::vector<std::uint8_t>>(view.layers.at("forbidden").values);
  const Vec3 current{2.5, 3.5, 0.0};
  const Vec3 frontier = PointTarget(result.problems.front().goal_odom);
  const double raster_margin_m =
      std::sqrt(2.0) * 0.5 * view.resolution_m;
  const double mask_half_width_m =
      result.corridor_half_width_m + raster_margin_m;
  const double mask_horizon_m =
      input.config.local_frontier.wheel_horizon_m +
      result.corridor_half_width_m + raster_margin_m;
  for (std::size_t y = 0U; y < view.height; ++y) {
    for (std::size_t x = 0U; x < view.width; ++x) {
      const std::size_t index = y * view.width + x;
      const Vec3 center{
          .x = view.origin_m.x +
               (static_cast<double>(x) + 0.5) * view.resolution_m,
          .y = view.origin_m.y +
               (static_cast<double>(y) + 0.5) * view.resolution_m,
      };
      const bool inside =
          std::hypot(center.x - current.x, center.y - current.y) <=
              mask_horizon_m + 1.0e-9 &&
          DistanceToHorizontalPrefix(center, current.x, frontier.x,
                                     current.y) <=
              mask_half_width_m + 1.0e-9;
      if (!inside) {
        EXPECT_EQ(valid[index], 0U) << "cell " << x << ',' << y;
        EXPECT_EQ(forbidden[index], 1U) << "cell " << x << ',' << y;
      }
    }
  }
}

TEST(LocalFrontier, RejectsANonLevelZeroLocalMapAndHonoursCancellation) {
  PlannerInput wrong_level = FrontierInput(PlatformType::kWheeled);
  wrong_level.world.local_map.resolution_m = 2.0;
  const LocalFrontierResult invalid =
      BuildLocalFrontiers(wrong_level, StraightRoute());
  EXPECT_EQ(invalid.reason_code, "LOCAL_MAP_LEVEL_INVALID");

  PlannerInput canceled = FrontierInput(PlatformType::kWheeled);
  std::stop_source stop;
  stop.request_stop();
  canceled.stop_token = stop.get_token();
  const LocalFrontierResult stopped =
      BuildLocalFrontiers(canceled, StraightRoute());
  EXPECT_EQ(stopped.reason_code, "REQUEST_CANCELED");
}

} // namespace
} // namespace lunar::planning::hierarchical

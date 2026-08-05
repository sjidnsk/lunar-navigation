#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <stop_token>
#include <string>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include "hierarchical/global_route_planner.hpp"
#include "hierarchical/hopper_route_planner.hpp"
#include "lunar_planner_core/planner.hpp"
#include "test_fixtures.hpp"

namespace lunar::planning::hierarchical
{
namespace
{

void SetByte(
  GridMap & map, const std::string & layer, const std::size_t x,
  const std::size_t y, const std::uint8_t value)
{
  std::get<std::vector<std::uint8_t>>(map.layers.at(layer).values)
  .at(y * map.width + x) = value;
}

void SetFloat(
  GridMap & map, const std::string & layer, const std::size_t x,
  const std::size_t y, const float value)
{
  std::get<std::vector<float>>(map.layers.at(layer).values)
  .at(y * map.width + x) = value;
}

[[nodiscard]] PlannerInput DistantGroundInput(const PlatformType platform)
{
  PlannerInput input = platform == PlatformType::kWheeled ?
    test::MakeValidWheelInput() :
    test::MakeValidLeggedInput();
  input.request_id = platform == PlatformType::kWheeled ?
    "regression-distant-wheel" :
    "regression-distant-legged";
  input.world.global_map = test::MakeFlatMap("map", 48U, 12U, 1.0);
  input.world.local_map = test::MakeFlatMap("odom", 12U, 12U, 1.0);
  input.config.global_map.base_resolution_m = 1.0;
  input.goal_map = GoalRegion{
    .goal_id = "distant-ground-goal",
    .target = PointGoal{.position_m = {42.5, 5.5, 0.0},
      .tolerance_m = 0.2},
  };
  if (platform == PlatformType::kWheeled) {
    std::get<WheeledState>(input.current_state).pose.position_m = {2.5, 5.5,
      0.0};
  } else {
    std::get<LeggedState>(input.current_state).body_pose.position_m = {
      2.5, 5.5, 0.5};
  }
  return input;
}

void AddVerticalWall(GridMap & map, const std::size_t x)
{
  for (std::size_t y = 0U; y < map.height; ++y) {
    SetByte(map, "obstacle", x, y, 1U);
    SetFloat(map, "obstacle_height", x, y, 2.0F);
  }
}

[[nodiscard]] PlannerInput ThreeHopInput()
{
  PlannerInput input = test::MakeValidHopperInput();
  input.request_id = "regression-three-hop";
  input.world.global_map = test::MakeFlatMap("map", 20U, 1U, 0.5);
  input.world.local_map = test::MakeFlatMap("odom", 20U, 1U, 0.5);
  input.config.global_map.base_resolution_m = 0.5;
  input.config.global_map.maximum_cells = 1'024U;
  input.config.global_map.maximum_axis_cells = 1'024U;
  input.current_state = HopperState{
    .pose = Pose3{.position_m = {1.5, 0.25, 0.5}},
  };
  input.goal_map = GoalRegion{
    .goal_id = "distant-hopper-goal",
    .target = PointGoal{.position_m = {7.5, 0.25, 0.0},
      .tolerance_m = 0.1},
  };
  auto & capability = std::get<HopperCapability>(input.capability);
  capability.body_half_extent_m.x = 0.1;
  capability.body_half_extent_m.y = 0.1;
  capability.maximum_launch_speed_mps = 2.0;
  capability.maximum_launch_impulse_newton_seconds = 100.0;
  capability.maximum_landing_speed_mps = 2.0;
  capability.minimum_flight_time = std::chrono::milliseconds{500};
  capability.maximum_flight_time = std::chrono::seconds{3};
  return input;
}

void ExpectSameGroundReference(
  const PlannerOutput & left,
  const PlannerOutput & right)
{
  ASSERT_TRUE(left.reference.has_value());
  ASSERT_TRUE(right.reference.has_value());
  EXPECT_EQ(left.reference->plan_id, right.reference->plan_id);
  EXPECT_EQ(left.reference->platform_type, right.reference->platform_type);
  EXPECT_EQ(left.reference->input_time, right.reference->input_time);
  EXPECT_EQ(
    left.reference->preview.poses_map,
    right.reference->preview.poses_map);
  const auto * left_trajectory =
    std::get_if<TrajectoryReference>(&left.reference->data);
  const auto * right_trajectory =
    std::get_if<TrajectoryReference>(&right.reference->data);
  ASSERT_NE(left_trajectory, nullptr);
  ASSERT_NE(right_trajectory, nullptr);
  ASSERT_EQ(left_trajectory->points.size(), right_trajectory->points.size());
  for (std::size_t index = 0U; index < left_trajectory->points.size(); ++index) {
    EXPECT_EQ(
      left_trajectory->points[index].time_from_start,
      right_trajectory->points[index].time_from_start);
    EXPECT_EQ(
      left_trajectory->points[index].pose,
      right_trajectory->points[index].pose);
    EXPECT_EQ(
      left_trajectory->points[index].velocity,
      right_trajectory->points[index].velocity);
  }
}

TEST(HierarchicalRegression, DistantGroundRoutesSucceedAndWallFailsClosed) {
  Planner planner;
  const PlannerOutput wheel =
    planner.Plan(DistantGroundInput(PlatformType::kWheeled));
  const PlannerOutput legged =
    planner.Plan(DistantGroundInput(PlatformType::kLegged));

  ASSERT_EQ(wheel.outcome, PlanningOutcome::kNewReferenceAvailable)
    << wheel.reason_code;
  ASSERT_EQ(legged.outcome, PlanningOutcome::kNewReferenceAvailable)
    << legged.reason_code;
  ASSERT_TRUE(wheel.reference.has_value());
  ASSERT_TRUE(legged.reference.has_value());
  EXPECT_GT(
    wheel.reference->preview.poses_map.back().position_m.x,
    std::get<TrajectoryReference>(wheel.reference->data)
    .points.back()
    .pose.position_m.x);
  EXPECT_GT(
    legged.reference->preview.poses_map.back().position_m.x,
    std::get<TrajectoryReference>(legged.reference->data)
    .points.back()
    .pose.position_m.x);

  PlannerInput blocked = DistantGroundInput(PlatformType::kWheeled);
  AddVerticalWall(blocked.world.global_map, 24U);
  const PlannerOutput wall = planner.Plan(blocked);
  EXPECT_EQ(wall.outcome, PlanningOutcome::kNoKnownSafeRoute);
  EXPECT_EQ(wall.reason_code, "GLOBAL_NO_KNOWN_SAFE_ROUTE");
  EXPECT_FALSE(wall.reference.has_value());
}

TEST(HierarchicalRegression, LeggedPassesTerrainRejectedForWheel) {
  PlannerInput wheel = DistantGroundInput(PlatformType::kWheeled);
  PlannerInput legged = DistantGroundInput(PlatformType::kLegged);
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
    const float elevation = 0.1F * static_cast<float>(x);
    SetFloat(wheel.world.global_map, "elevation", x, 0U, elevation);
    SetFloat(legged.world.global_map, "elevation", x, 0U, elevation);
  }

  const GlobalRoutePlanResult wheel_route = PlanGroundGlobalRoute(wheel);
  const GlobalRoutePlanResult legged_route = PlanGroundGlobalRoute(legged);

  EXPECT_FALSE(wheel_route.ok());
  EXPECT_TRUE(legged_route.ok()) << legged_route.reason_code;
  EXPECT_NE(wheel_route.reason_code, legged_route.reason_code);
}

TEST(
  HierarchicalRegression,
  HopperDistinguishesMultiHopChainResolutionAndResourceFailures) {
  Planner planner;
  const PlannerInput input = ThreeHopInput();
  const HopperRoutePlanResult route = PlanHopperGlobalRoute(input);
  const PlannerOutput success = planner.Plan(input);
  ASSERT_TRUE(route.ok()) << route.reason_code;
  EXPECT_EQ(route.route_hops, 3U);
  ASSERT_EQ(success.outcome, PlanningOutcome::kNewReferenceAvailable)
    << success.reason_code;
  ASSERT_TRUE(success.reference.has_value());
  EXPECT_EQ(success.reference->preview.poses_map.size(), 4U);
  EXPECT_EQ(std::get<HopReference>(success.reference->data).segments.size(), 1U);

  PlannerInput broken = ThreeHopInput();
  for (std::size_t x = 7U; x <= 11U; ++x) {
    SetByte(broken.world.global_map, "valid_mask", x, 0U, 0U);
  }
  const PlannerOutput no_chain = planner.Plan(broken);
  EXPECT_EQ(no_chain.outcome, PlanningOutcome::kNoKnownSafeRoute);
  EXPECT_EQ(no_chain.reason_code, "GLOBAL_NO_KNOWN_SAFE_ROUTE");

  PlannerInput coarse = ThreeHopInput();
  coarse.world.global_map = test::MakeFlatMap("map", 5U, 1U, 2.0);
  coarse.world.local_map = test::MakeFlatMap("odom", 5U, 1U, 2.0);
  coarse.config.global_map.base_resolution_m = 2.0;
  const PlannerOutput resolution = planner.Plan(coarse);
  EXPECT_EQ(resolution.outcome, PlanningOutcome::kResourceExhausted);
  EXPECT_EQ(resolution.reason_code, "HOPPER_GLOBAL_RESOLUTION_INSUFFICIENT");

  PlannerInput limited = ThreeHopInput();
  limited.config.hopper.maximum_graph_nodes = 2U;
  const PlannerOutput resource = planner.Plan(limited);
  EXPECT_EQ(resource.outcome, PlanningOutcome::kResourceExhausted);
  EXPECT_EQ(resource.reason_code, "HOPPER_GLOBAL_ROUTE_RESOURCE_LIMIT");
}

TEST(HierarchicalRegression, SeparatesGlobalSuccessFromLocalCoverageFailure) {
  PlannerInput input = DistantGroundInput(PlatformType::kWheeled);
  const GlobalRoutePlanResult global = PlanGroundGlobalRoute(input);
  ASSERT_TRUE(global.ok()) << global.reason_code;
  auto & valid = std::get<std::vector<std::uint8_t>>(
    input.world.local_map.layers.at("valid_mask").values);
  std::fill(valid.begin(), valid.end(), 0U);

  const PlannerOutput output = Planner{}.Plan(input);

  EXPECT_EQ(output.outcome, PlanningOutcome::kNoKnownSafeRoute);
  EXPECT_EQ(output.reason_code, "LOCAL_MAP_COVERAGE_INSUFFICIENT");
  EXPECT_FALSE(output.reference.has_value());
}

TEST(HierarchicalRegression, AppliesFrozenMapFromOdomTransform) {
  PlannerInput input = DistantGroundInput(PlatformType::kWheeled);
  input.world.global_map = test::MakeFlatMap("map", 30U, 30U, 1.0);
  std::get<WheeledState>(input.current_state).pose.position_m = {1.5, 2.5, 0.0};
  input.world.map_from_odom.translation_m = {10.0, 10.0, 0.0};
  input.world.map_from_odom.rotation =
    test::YawQuaternion(std::numbers::pi / 2.0);
  input.goal_map.target =
    PointGoal{.position_m = {7.5, 15.5, 0.0}, .tolerance_m = 0.2};

  const GlobalRoutePlanResult result = PlanGroundGlobalRoute(input);

  ASSERT_TRUE(result.ok()) << result.reason_code;
  ASSERT_FALSE(result.route->poses_map.empty());
  EXPECT_NEAR(result.route->poses_map.front().position_m.x, 7.5, 1.0e-9);
  EXPECT_NEAR(result.route->poses_map.front().position_m.y, 11.5, 1.0e-9);
}

TEST(HierarchicalRegression, RejectsUnrepresentableScaleAndHonorsCancel) {
  PlannerInput scale = DistantGroundInput(PlatformType::kWheeled);
  scale.world.global_map = test::MakeFlatMap("map", 5'000U, 1U, 3.2);
  scale.world.local_map = test::MakeFlatMap("odom", 12U, 12U, 0.2);
  scale.config.global_map.base_resolution_m = 0.2;
  const PlannerOutput unsupported = Planner{}.Plan(scale);
  EXPECT_EQ(unsupported.outcome, PlanningOutcome::kResourceExhausted);
  EXPECT_EQ(unsupported.reason_code, "GLOBAL_MAP_SCALE_UNSUPPORTED");

  PlannerInput canceled = DistantGroundInput(PlatformType::kWheeled);
  std::stop_source stop;
  stop.request_stop();
  canceled.stop_token = stop.get_token();
  const PlannerOutput stopped = Planner{}.Plan(canceled);
  EXPECT_EQ(stopped.outcome, PlanningOutcome::kCanceled);
  EXPECT_EQ(stopped.reason_code, "REQUEST_CANCELED");
}

TEST(HierarchicalRegression, RepeatedInputIsBitwiseStableExceptTiming) {
  Planner planner;
  const PlannerInput input = DistantGroundInput(PlatformType::kWheeled);
  const PlannerOutput first = planner.Plan(input);
  const PlannerOutput second = planner.Plan(input);

  ASSERT_EQ(first.outcome, PlanningOutcome::kNewReferenceAvailable)
    << first.reason_code;
  EXPECT_EQ(second.outcome, first.outcome);
  EXPECT_EQ(second.directive, first.directive);
  EXPECT_EQ(second.reason_code, first.reason_code);
  EXPECT_EQ(second.diagnostics.planner_name, first.diagnostics.planner_name);
  EXPECT_EQ(
    second.diagnostics.expanded_states,
    first.diagnostics.expanded_states);
  EXPECT_EQ(second.diagnostics.best_cost, first.diagnostics.best_cost);
  EXPECT_EQ(second.diagnostics.warning_codes, first.diagnostics.warning_codes);
  ASSERT_TRUE(first.diagnostics.hierarchical.has_value());
  ASSERT_TRUE(second.diagnostics.hierarchical.has_value());
  EXPECT_EQ(
    second.diagnostics.hierarchical->global_expanded_states,
    first.diagnostics.hierarchical->global_expanded_states);
  EXPECT_EQ(
    second.diagnostics.hierarchical->local_expanded_states,
    first.diagnostics.hierarchical->local_expanded_states);
  EXPECT_EQ(
    second.diagnostics.hierarchical->global_open_peak,
    first.diagnostics.hierarchical->global_open_peak);
  EXPECT_EQ(
    second.diagnostics.hierarchical->raw_route_points,
    first.diagnostics.hierarchical->raw_route_points);
  EXPECT_EQ(
    second.diagnostics.hierarchical->simplified_route_points,
    first.diagnostics.hierarchical->simplified_route_points);
  ExpectSameGroundReference(first, second);
}

} // namespace
} // namespace lunar::planning::hierarchical

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
  input.world.local_map.origin_m.x = -1.0;
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

[[nodiscard]] PlannerInput FiftyMetreInput(const PlatformType platform)
{
  PlannerInput input = platform == PlatformType::kHopper ?
    test::MakeValidHopperInput() : DistantGroundInput(platform);
  input.request_id = "regression-50m-" +
    std::to_string(static_cast<std::uint8_t>(platform));
  input.world.global_map = test::MakeFlatMap("map", 250U, 250U, 0.2);
  input.world.local_map = test::MakeFlatMap("odom", 250U, 250U, 0.2);
  input.world.local_map.origin_m.x = -1.0;
  input.config.global_map.base_resolution_m = 0.2;
  input.goal_map = GoalRegion{
    .goal_id = "regression-50m-goal",
    .target = PointGoal{.position_m = {47.9, 25.1, 0.0},
      .tolerance_m = platform == PlatformType::kHopper ? 0.0 : 0.2},
  };
  if (platform == PlatformType::kWheeled) {
    input.config.wheel.xy_resolution_m = 0.2;
    std::get<WheeledState>(input.current_state).pose.position_m = {
      2.1, 25.1, 0.0};
  } else if (platform == PlatformType::kLegged) {
    input.config.legged.xy_resolution_m = 0.2;
    input.config.local_frontier.legged_horizon_m = 2.0;
    std::get<LeggedState>(input.current_state).body_pose.position_m = {
      2.1, 25.1, 0.5};
  } else {
    std::get<HopperState>(input.current_state).pose.position_m = {
      2.1, 25.1, 0.5};
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

TEST(HierarchicalRegression,
     HistoricalBoundaryWindowUsesHierarchicalDomainConstruction) {
  PlannerInput input = DistantGroundInput(PlatformType::kWheeled);
  input.request_id = "historical-boundary-window";
  input.config.optimization.maximum_iterations = 0U;
  input.config.local_frontier.additional_corridor_margin_m = 2.0;
  auto& state = std::get<WheeledState>(input.current_state);
  state.pose.position_m.y = 3.1;
  std::get<PointGoal>(input.goal_map.target).position_m.y = 3.1;
  const auto& obstacle = std::get<std::vector<std::uint8_t>>(
      input.world.local_map.layers.at("obstacle").values);
  ASSERT_TRUE(std::ranges::all_of(
      obstacle, [](const std::uint8_t value) { return value == 0U; }));

  const PlannerOutput output = Planner{}.Plan(input);

  if (output.outcome != PlanningOutcome::kNewReferenceAvailable) {
    EXPECT_EQ(output.reason_code, "LOCAL_SEARCH_DOMAIN_EXHAUSTED");
    EXPECT_NE(output.reason_code, "WHEEL_GOAL_INFEASIBLE");
    EXPECT_GT(output.diagnostics.expanded_states, 0U);
  }
  ASSERT_EQ(output.outcome, PlanningOutcome::kNewReferenceAvailable)
      << output.reason_code;
  ASSERT_TRUE(output.reference.has_value());
  EXPECT_GT(output.diagnostics.expanded_states, 0U);
  ASSERT_TRUE(output.diagnostics.hierarchical.has_value());
  const auto& metrics = *output.diagnostics.hierarchical;
  EXPECT_GT(metrics.local_expanded_states, 0U);
  EXPECT_DOUBLE_EQ(metrics.additional_corridor_margin_m, 2.0);
  EXPECT_GT(metrics.search_domain_cell_count, 0U);
  EXPECT_LT(
    metrics.search_domain_cell_count, input.world.local_map.CellCount());
  EXPECT_EQ(metrics.search_domain_sha256.size(), 64U);
  EXPECT_TRUE(metrics.physical_goal_feasible);
}

TEST(HierarchicalRegression, FiftyMetreFarGoalsSucceedForAllPlatforms) {
  Planner planner;
  for (const PlatformType platform : {
         PlatformType::kWheeled,
         PlatformType::kLegged,
         PlatformType::kHopper})
  {
    const PlannerOutput output = planner.Plan(FiftyMetreInput(platform));
    ASSERT_EQ(output.outcome, PlanningOutcome::kNewReferenceAvailable)
      << static_cast<int>(platform) << ' ' << output.reason_code;
    ASSERT_TRUE(output.reference.has_value());
    ASSERT_TRUE(output.diagnostics.local_trajectory.has_value());
    EXPECT_EQ(
      output.diagnostics.local_trajectory->collision_validation,
      CollisionValidation::kCertified);
  }
}

TEST(HierarchicalRegression, FiftyMetreFailuresRemainExplicitAndBounded) {
  Planner planner;

  PlannerInput blocked = FiftyMetreInput(PlatformType::kWheeled);
  AddVerticalWall(blocked.world.global_map, 125U);
  const PlannerOutput wall = planner.Plan(blocked);
  EXPECT_EQ(wall.outcome, PlanningOutcome::kNoKnownSafeRoute);
  EXPECT_EQ(wall.reason_code, "GLOBAL_NO_KNOWN_SAFE_ROUTE");

  PlannerInput fallback = FiftyMetreInput(PlatformType::kWheeled);
  fallback.config.optimization.maximum_iterations = 0U;
  const PlannerOutput discrete = planner.Plan(fallback);
  ASSERT_EQ(discrete.outcome, PlanningOutcome::kNewReferenceAvailable)
    << discrete.reason_code;
  ASSERT_TRUE(discrete.diagnostics.local_trajectory.has_value());
  EXPECT_EQ(
    discrete.diagnostics.local_trajectory->trajectory_mode,
    TrajectoryMode::kDiscreteFallback);

  fallback.config.optimization.require_smoothed_execution = true;
  const PlannerOutput required = planner.Plan(fallback);
  EXPECT_EQ(required.outcome, PlanningOutcome::kNoKnownSafeRoute);
  EXPECT_EQ(required.reason_code, "WHEEL_SMOOTHED_EXECUTION_REQUIRED");

  PlannerInput wheel_connector = test::MakeValidWheelInput();
  wheel_connector.world.local_map =
    test::MakeFlatMap("odom", 10U, 10U, 1.0);
  wheel_connector.world.local_map.origin_m = {-1.0, -1.0, 0.0};
  std::get<WheeledState>(wheel_connector.current_state).pose.position_m = {
    2.15, 3.5, 0.0};
  SetByte(wheel_connector.world.local_map, "obstacle", 2U, 4U, 1U);
  const PlannerOutput wheel_blocked = planner.Plan(wheel_connector);
  EXPECT_EQ(
    wheel_blocked.reason_code, "WHEEL_START_CONNECTOR_INFEASIBLE");

  PlannerInput legged_connector = test::MakeValidLeggedInput();
  legged_connector.world.local_map =
    test::MakeFlatMap("odom", 10U, 10U, 1.0);
  legged_connector.world.local_map.origin_m = {-1.0, -1.0, 0.0};
  std::get<LeggedState>(
    legged_connector.current_state).body_pose.position_m = {2.15, 3.5, 0.5};
  SetByte(legged_connector.world.local_map, "obstacle", 2U, 4U, 1U);
  const PlannerOutput legged_blocked = planner.Plan(legged_connector);
  EXPECT_EQ(
    legged_blocked.reason_code, "LEGGED_START_CONNECTOR_INFEASIBLE");

  PlannerInput support = FiftyMetreInput(PlatformType::kHopper);
  SetByte(support.world.local_map, "obstacle", 244U, 125U, 1U);
  const PlannerOutput insufficient = planner.Plan(support);
  EXPECT_EQ(
    insufficient.reason_code, "HOPPER_LANDING_TARGET_OCCUPIED");

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

TEST(HierarchicalRegression, HopperUsesOneDirectRequestWithoutGlobalChain) {
  Planner planner;
  const PlannerOutput success =
    planner.Plan(FiftyMetreInput(PlatformType::kHopper));
  ASSERT_EQ(success.outcome, PlanningOutcome::kNewReferenceAvailable)
    << success.reason_code;
  ASSERT_TRUE(success.reference.has_value());
  EXPECT_EQ(success.reference->preview.poses_map.size(), 2U);
  EXPECT_EQ(std::get<HopReference>(success.reference->data).segments.size(), 1U);
  EXPECT_EQ(success.continuation, nullptr);
}

TEST(HierarchicalRegression, HopperReportsTheValidatedGlobalMapLevel) {
  PlannerInput input = test::MakeValidHopperInput();
  input.world.global_map = test::MakeFlatMap("map", 140U, 15U, 0.8);
  input.world.local_map = test::MakeFlatMap("odom", 560U, 60U, 0.2);
  input.config.global_map.base_resolution_m = 0.2;
  std::get<HopperState>(input.current_state).pose.position_m = {
    5.0, 5.0, 0.0};
  input.goal_map.target = PointGoal{
    .position_m = {105.0, 5.0, 0.0}, .tolerance_m = 0.0};

  const PlannerOutput output = Planner{}.Plan(input);

  ASSERT_EQ(output.outcome, PlanningOutcome::kNewReferenceAvailable)
    << output.reason_code;
  ASSERT_TRUE(output.diagnostics.hierarchical.has_value());
  EXPECT_EQ(output.diagnostics.hierarchical->global_level, 2U);
  EXPECT_DOUBLE_EQ(output.diagnostics.hierarchical->global_resolution_m, 0.8);
  EXPECT_EQ(output.diagnostics.hierarchical->global_cells, 140U * 15U);
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

TEST(HierarchicalRegression, AnchorsWheelPreviewAtTheMapTransformedTrueStart) {
  PlannerInput input = DistantGroundInput(PlatformType::kWheeled);
  input.world.global_map = test::MakeFlatMap("map", 30U, 30U, 1.0);
  input.world.local_map.origin_m.x = -2.0;
  input.world.local_map.origin_m.y = -1.0;
  auto& state = std::get<WheeledState>(input.current_state);
  state.pose.position_m = {1.2, 2.3, 0.0};
  state.pose.orientation = test::YawQuaternion(0.17);
  input.world.map_from_odom.translation_m = {10.0, 10.0, 0.0};
  input.world.map_from_odom.rotation =
      test::YawQuaternion(std::numbers::pi / 2.0);
  input.goal_map.target =
      PointGoal{.position_m = {7.5, 15.5, 0.0}, .tolerance_m = 0.2};

  const PlannerOutput output = Planner{}.Plan(input);

  ASSERT_EQ(output.outcome, PlanningOutcome::kNewReferenceAvailable)
      << output.reason_code;
  ASSERT_TRUE(output.reference.has_value());
  ASSERT_FALSE(output.reference->preview.poses_map.empty());
  const Pose3& start = output.reference->preview.poses_map.front();
  EXPECT_NEAR(start.position_m.x, 7.7, 1.0e-9);
  EXPECT_NEAR(start.position_m.y, 11.2, 1.0e-9);
  EXPECT_NEAR(
      std::atan2(2.0 * start.orientation.w * start.orientation.z,
                 1.0 - 2.0 * start.orientation.z * start.orientation.z),
      std::numbers::pi / 2.0 + 0.17, 1.0e-9);
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
  ASSERT_TRUE(first.diagnostics.local_trajectory.has_value());
  ASSERT_TRUE(second.diagnostics.local_trajectory.has_value());
  EXPECT_EQ(
    second.diagnostics.local_trajectory->trajectory_mode,
    first.diagnostics.local_trajectory->trajectory_mode);
  EXPECT_EQ(
    second.diagnostics.local_trajectory->start_anchor_error_m,
    first.diagnostics.local_trajectory->start_anchor_error_m);
  EXPECT_EQ(
    second.diagnostics.local_trajectory->endpoint_error_m,
    first.diagnostics.local_trajectory->endpoint_error_m);
  EXPECT_EQ(
    second.diagnostics.local_trajectory->maximum_curvature_per_m,
    first.diagnostics.local_trajectory->maximum_curvature_per_m);
  EXPECT_EQ(
    second.diagnostics.local_trajectory->collision_validation,
    first.diagnostics.local_trajectory->collision_validation);
  ExpectSameGroundReference(first, second);
}

} // namespace
} // namespace lunar::planning::hierarchical

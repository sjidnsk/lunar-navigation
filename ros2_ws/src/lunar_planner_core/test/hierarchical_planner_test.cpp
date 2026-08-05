#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stop_token>
#include <string>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include "hierarchical/global_route_planner.hpp"
#include "hierarchical/reference_composer.hpp"
#include "lunar_planner_core/planner.hpp"
#include "test_fixtures.hpp"

namespace lunar::planning::hierarchical {
namespace {

[[nodiscard]] PlannerInput DistantWheelInput() {
  PlannerInput input = test::MakeValidWheelInput();
  input.request_id = "hierarchical-wheel";
  input.world.global_map = test::MakeFlatMap("map", 24U, 8U, 1.0);
  input.world.local_map = test::MakeFlatMap("odom", 12U, 8U, 1.0);
  input.config.global_map.base_resolution_m = 1.0;
  input.goal_map.target = PointGoal{
      .position_m = {18.5, 3.5, 0.0},
      .tolerance_m = 0.2,
  };
  return input;
}

void SetGlobalForbiddenColumn(PlannerInput &input, const std::size_t x) {
  auto &forbidden = std::get<std::vector<std::uint8_t>>(
      input.world.global_map.layers.at("forbidden").values);
  for (std::size_t y = 0U; y < input.world.global_map.height; ++y) {
    forbidden[y * input.world.global_map.width + x] = 1U;
  }
}

[[nodiscard]] PlannerOutput LocalWheelOutput(const PlannerInput &input) {
  return PlannerOutput{
      .outcome = PlanningOutcome::kNewReferenceAvailable,
      .directive = ExecutionDirective::kActivateNewReference,
      .reason_code = "WHEEL_PLAN_AVAILABLE",
      .reference =
          MotionReference{
              .plan_id = "stale-local-id",
              .platform_type = PlatformType::kWheeled,
              .input_time = {},
              .data =
                  TrajectoryReference{
                      .semantics = TrajectorySemantics::kWheeledBase,
                      .points = {TrajectoryPoint{
                          .pose = std::get<WheeledState>(input.current_state)
                                      .pose}},
                  },
          },
  };
}

[[nodiscard]] GlobalRoute TwoPointRoute() {
  return GlobalRoute{
      .raw_cells = {{.x = 0, .y = 0}, {.x = 1, .y = 0}},
      .simplified_cells = {{.x = 0, .y = 0}, {.x = 1, .y = 0}},
      .poses_map =
          {
              Pose3{.position_m = {0.5, 0.5, 0.0}},
              Pose3{.position_m = {1.5, 0.5, 0.0}},
          },
      .cost = 1.0,
  };
}

TEST(ReferenceComposer, FreezesCompletePreviewAndRequestIdentity) {
  const PlannerInput input = DistantWheelInput();
  const GlobalRoute route = TwoPointRoute();

  ReferenceComposeResult result =
      ComposeReference(input, route, LocalWheelOutput(input));

  ASSERT_TRUE(result.ok()) << result.reason_code;
  EXPECT_EQ(result.reference->plan_id, "wheel/" + input.request_id);
  EXPECT_EQ(result.reference->platform_type, PlatformType::kWheeled);
  EXPECT_EQ(result.reference->input_time, input.state_time);
  EXPECT_EQ(result.reference->preview.poses_map, route.poses_map);
  const auto *trajectory =
      std::get_if<TrajectoryReference>(&result.reference->data);
  ASSERT_NE(trajectory, nullptr);
  EXPECT_EQ(trajectory->points.size(), 1U);
}

TEST(ReferenceComposer, RejectsGlobalLocalAndAuthorizationInvariantBreaks) {
  const PlannerInput input = DistantWheelInput();
  GlobalRoute empty;
  EXPECT_EQ(ComposeReference(input, empty, LocalWheelOutput(input)).reason_code,
            "REFERENCE_GLOBAL_PREVIEW_EMPTY");

  GlobalRoute nonfinite = TwoPointRoute();
  nonfinite.poses_map.front().position_m.x =
      std::numeric_limits<double>::quiet_NaN();
  EXPECT_EQ(
      ComposeReference(input, nonfinite, LocalWheelOutput(input)).reason_code,
      "REFERENCE_GLOBAL_PREVIEW_NONFINITE");

  PlannerOutput missing = LocalWheelOutput(input);
  missing.reference.reset();
  EXPECT_EQ(
      ComposeReference(input, TwoPointRoute(), std::move(missing)).reason_code,
      "REFERENCE_LOCAL_DATA_MISSING");

  PlannerOutput mismatch = LocalWheelOutput(input);
  mismatch.reference->platform_type = PlatformType::kLegged;
  EXPECT_EQ(
      ComposeReference(input, TwoPointRoute(), std::move(mismatch)).reason_code,
      "REFERENCE_PLATFORM_MISMATCH");

  PlannerInput hopper = test::MakeValidHopperInput();
  PlannerOutput two_hops{
      .outcome = PlanningOutcome::kNewReferenceAvailable,
      .directive = ExecutionDirective::kActivateNewReference,
      .reason_code = "HOPPER_FIRST_HOP_AVAILABLE",
      .reference =
          MotionReference{
              .platform_type = PlatformType::kHopper,
              .data =
                  HopReference{
                      .segments = {HopSegment{}, HopSegment{}},
                  },
          },
  };
  EXPECT_EQ(ComposeReference(hopper, TwoPointRoute(), std::move(two_hops))
                .reason_code,
            "REFERENCE_HOP_AUTHORIZATION_INVALID");
}

TEST(HierarchicalPlanner, ReturnsCompleteGlobalPreviewAndOneLocalSegment) {
  Planner planner;
  const PlannerInput input = DistantWheelInput();
  const GlobalRoutePlanResult global = PlanGroundGlobalRoute(input);
  ASSERT_TRUE(global.ok()) << global.reason_code;

  const PlannerOutput output = planner.Plan(input);

  ASSERT_EQ(output.outcome, PlanningOutcome::kNewReferenceAvailable)
      << output.reason_code;
  ASSERT_TRUE(output.reference.has_value());
  EXPECT_EQ(output.diagnostics.planner_name, "cpp_v3_hierarchical");
  ASSERT_TRUE(output.diagnostics.best_cost.has_value());
  EXPECT_DOUBLE_EQ(*output.diagnostics.best_cost, global.route->cost);
  EXPECT_EQ(output.reference->preview.poses_map, global.route->poses_map);
  const auto *trajectory =
      std::get_if<TrajectoryReference>(&output.reference->data);
  ASSERT_NE(trajectory, nullptr);
  ASSERT_FALSE(trajectory->points.empty());
  EXPECT_LT(trajectory->points.back().pose.position_m.x,
            output.reference->preview.poses_map.back().position_m.x);
  ASSERT_TRUE(output.diagnostics.hierarchical.has_value());
  EXPECT_GE(output.diagnostics.expanded_states,
            output.diagnostics.hierarchical->global_expanded_states);
}

TEST(HierarchicalPlanner, DistinguishesGlobalAndLocalRejections) {
  Planner planner;
  PlannerInput global_blocked = DistantWheelInput();
  SetGlobalForbiddenColumn(global_blocked, 8U);

  const PlannerOutput global = planner.Plan(global_blocked);

  EXPECT_EQ(global.outcome, PlanningOutcome::kNoKnownSafeRoute);
  EXPECT_EQ(global.reason_code, "GLOBAL_NO_KNOWN_SAFE_ROUTE");
  EXPECT_FALSE(global.reference.has_value());

  PlannerInput local_unavailable = DistantWheelInput();
  auto &valid = std::get<std::vector<std::uint8_t>>(
      local_unavailable.world.local_map.layers.at("valid_mask").values);
  std::fill(valid.begin(), valid.end(), 0U);

  const PlannerOutput local = planner.Plan(local_unavailable);

  EXPECT_EQ(local.outcome, PlanningOutcome::kNoKnownSafeRoute);
  EXPECT_EQ(local.reason_code, "LOCAL_MAP_COVERAGE_INSUFFICIENT");
  EXPECT_FALSE(local.reference.has_value());
}

TEST(HierarchicalPlanner, BacksOffToANearerReachableFrontier) {
  Planner planner;
  PlannerInput input = DistantWheelInput();
  auto &capability = std::get<WheeledCapability>(input.capability);
  capability.motion_primitives = {
      WheelMotionPrimitive{
          .primitive_id = "three-metre-forward",
          .kind = WheelPrimitiveKind::kForward,
          .relative_end_pose = Pose3{.position_m = {3.0, 0.0, 0.0}},
          .nominal_duration = std::chrono::seconds{3},
      },
  };

  const PlannerOutput output = planner.Plan(input);

  ASSERT_EQ(output.outcome, PlanningOutcome::kNewReferenceAvailable)
      << output.reason_code;
  ASSERT_TRUE(output.diagnostics.hierarchical.has_value());
  EXPECT_EQ(output.diagnostics.hierarchical->local_attempts, 2U);
  EXPECT_NE(std::ranges::find(output.diagnostics.warning_codes,
                              "LOCAL_FRONTIER_BACKOFF"),
            output.diagnostics.warning_codes.end());
  const auto *trajectory =
      std::get_if<TrajectoryReference>(&output.reference->data);
  ASSERT_NE(trajectory, nullptr);
  EXPECT_NEAR(trajectory->points.back().pose.position_m.x, 5.5, 0.25);
}

TEST(HierarchicalPlanner, IsStatelessAcrossCancellationAndRepeatedRequests) {
  Planner planner;
  PlannerInput canceled = DistantWheelInput();
  std::stop_source stop;
  stop.request_stop();
  canceled.stop_token = stop.get_token();
  EXPECT_EQ(planner.Plan(canceled).outcome, PlanningOutcome::kCanceled);

  const PlannerInput input = DistantWheelInput();
  const PlannerOutput first = planner.Plan(input);
  const PlannerOutput second = planner.Plan(input);

  ASSERT_TRUE(first.reference.has_value()) << first.reason_code;
  ASSERT_TRUE(second.reference.has_value()) << second.reason_code;
  EXPECT_EQ(first.reference->preview.poses_map,
            second.reference->preview.poses_map);
  EXPECT_EQ(first.reference->plan_id, second.reference->plan_id);
  EXPECT_EQ(first.diagnostics.best_cost, second.diagnostics.best_cost);
}

} // namespace
} // namespace lunar::planning::hierarchical

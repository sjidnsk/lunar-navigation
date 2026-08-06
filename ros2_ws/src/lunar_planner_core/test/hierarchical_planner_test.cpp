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

void SetObstacleBoth(PlannerInput &input, const std::size_t x,
                     const std::size_t y) {
  for (GridMap *map : {&input.world.global_map, &input.world.local_map}) {
    auto &obstacles =
        std::get<std::vector<std::uint8_t>>(map->layers.at("obstacle").values);
    obstacles.at(y * map->width + x) = 1U;
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

[[nodiscard]] GlobalRoute FivePointRoute() {
  GlobalRoute route = TwoPointRoute();
  route.raw_cells = {{.x = 0, .y = 0},
                     {.x = 1, .y = 0},
                     {.x = 2, .y = 0},
                     {.x = 3, .y = 0},
                     {.x = 4, .y = 0}};
  route.simplified_cells = route.raw_cells;
  route.poses_map.clear();
  for (std::int32_t x = 0; x < 5; ++x) {
    route.poses_map.push_back(
        Pose3{.position_m = {static_cast<double>(x) + 0.5, 0.5, 0.0}});
  }
  return route;
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

TEST(ReferenceComposer, ThinsOnlyThePublishedPreviewWithoutLosingEndpoints) {
  PlannerInput input = DistantWheelInput();
  input.config.global_search.maximum_preview_points = 3U;
  const GlobalRoute route = FivePointRoute();

  ReferenceComposeResult result =
      ComposeReference(input, route, LocalWheelOutput(input));

  ASSERT_TRUE(result.ok()) << result.reason_code;
  ASSERT_EQ(result.reference->preview.poses_map.size(), 3U);
  EXPECT_EQ(result.reference->preview.poses_map.front(),
            route.poses_map.front());
  EXPECT_EQ(result.reference->preview.poses_map.back(), route.poses_map.back());
  EXPECT_EQ(route.poses_map.size(), 5U);
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
  ASSERT_NE(output.continuation, nullptr);
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
  EXPECT_FALSE(output.diagnostics.hierarchical->route_reused);
  EXPECT_EQ(output.diagnostics.hierarchical->route_cursor, 0U);
  EXPECT_EQ(output.diagnostics.hierarchical->rolling_request_count, 1U);
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

TEST(HierarchicalPlanner, UsesExactTerminalConnectorBeforeFrontierBackoff) {
  Planner planner;
  PlannerInput input = DistantWheelInput();
  auto &capability = std::get<WheeledCapability>(input.capability);
  capability.motion_primitives = {
      WheelMotionPrimitive{
          .primitive_id = "three-metre-forward",
          .kind = WheelPrimitiveKind::kForward,
          .relative_end_pose = Pose3{.position_m = {3.0, 0.0, 0.0}},
      },
  };

  const PlannerOutput output = planner.Plan(input);

  ASSERT_EQ(output.outcome, PlanningOutcome::kNewReferenceAvailable)
      << output.reason_code;
  ASSERT_TRUE(output.diagnostics.hierarchical.has_value());
  EXPECT_EQ(output.diagnostics.hierarchical->local_attempts, 1U);
  EXPECT_EQ(std::ranges::find(output.diagnostics.warning_codes,
                              "LOCAL_FRONTIER_BACKOFF"),
            output.diagnostics.warning_codes.end());
  const auto *trajectory =
      std::get_if<TrajectoryReference>(&output.reference->data);
  ASSERT_NE(trajectory, nullptr);
  EXPECT_NEAR(trajectory->points.back().pose.position_m.x, 6.5, 1.0e-9);
}

TEST(HierarchicalPlanner, ReportsTheLocalBackendReasonAfterAllFrontiersFail) {
  Planner planner;
  PlannerInput input = DistantWheelInput();
  auto &capability = std::get<WheeledCapability>(input.capability);
  capability.motion_primitives = {
      WheelMotionPrimitive{
          .primitive_id = "spin-only",
          .kind = WheelPrimitiveKind::kSpinCounterclockwise,
          .relative_end_pose =
              Pose3{.orientation = Quaternion{.w = 0.9238795325112867,
                                              .z = 0.3826834323650898}},
      },
  };

  const PlannerOutput output = planner.Plan(input);

  ASSERT_EQ(output.outcome, PlanningOutcome::kNoKnownSafeRoute);
  EXPECT_EQ(output.reason_code, "LOCAL_SEGMENT_INFEASIBLE");
  EXPECT_NE(std::ranges::find(output.diagnostics.warning_codes,
                              "WHEEL_NO_KNOWN_SAFE_ROUTE"),
            output.diagnostics.warning_codes.end());
}

TEST(HierarchicalPlanner, ReroutesAfterAConditionalCorridorFailsLocalSweep) {
  Planner planner;
  PlannerInput input = test::MakeValidWheelInput();
  input.request_id = "conditional-corridor-reroute";
  input.world.global_map = test::MakeFlatMap("map", 14U, 13U, 1.0);
  input.world.local_map = test::MakeFlatMap("odom", 14U, 13U, 1.0);
  input.config.global_map.base_resolution_m = 1.0;
  input.config.local_frontier.wheel_horizon_m = 20.0;
  // The coarse 1 m test lattice needs room to realize the rerouted Manhattan
  // turns; production L0 uses the approved 0.20 m primitives.
  input.config.local_frontier.additional_corridor_margin_m = 0.8;
  auto &state = std::get<WheeledState>(input.current_state);
  state.pose.position_m = {1.5, 4.5, 0.0};
  auto &capability = std::get<WheeledCapability>(input.capability);
  capability.footprint_xy_m = {
      {-0.2, -0.5}, {0.2, -0.5}, {0.2, 0.5}, {-0.2, 0.5}};
  capability.minimum_clearance_m = 0.0;
  input.goal_map.target = PointGoal{
      .position_m = {12.5, 4.5, 0.0},
      .tolerance_m = 0.2,
  };
  for (std::size_t y = 0U; y < 13U; ++y) {
    if (y != 4U && (y < 8U || y > 10U)) {
      SetObstacleBoth(input, 6U, y);
    }
  }
  const GlobalRoutePlanResult shortest = PlanGroundGlobalRoute(input);
  ASSERT_TRUE(shortest.ok()) << shortest.reason_code;
  ASSERT_NE(std::ranges::find(shortest.route->conditional_cells,
                              shared::GridCell{6, 4}),
            shortest.route->conditional_cells.end());

  const PlannerOutput output = planner.Plan(input);

  ASSERT_TRUE(output.reference.has_value())
      << output.reason_code << " first_warning="
      << (output.diagnostics.warning_codes.empty()
              ? std::string{"none"}
              : output.diagnostics.warning_codes.front());
  EXPECT_NE(std::ranges::find(output.diagnostics.warning_codes,
                              "GLOBAL_CONDITIONAL_CORRIDOR_RETRY"),
            output.diagnostics.warning_codes.end());
  EXPECT_NE(output.reference->preview.poses_map, shortest.route->poses_map);
  EXPECT_TRUE(std::ranges::any_of(
      output.reference->preview.poses_map,
      [](const Pose3 &pose) { return pose.position_m.y >= 8.5; }));
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

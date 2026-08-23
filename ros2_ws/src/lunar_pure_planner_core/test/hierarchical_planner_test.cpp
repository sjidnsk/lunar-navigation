#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <future>
#include <limits>
#include <optional>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <thread>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include "hierarchical/global_route_planner.hpp"
#include "hierarchical/local_frontier.hpp"
#include "hierarchical/reference_composer.hpp"
#include "legged/legged_planner.hpp"
#include "lunar_pure_planner_core/planner.hpp"
#include "test_fixtures.hpp"
#include "wheel/wheel_planner.hpp"

namespace lunar::pure_planning::hierarchical {
namespace {

[[nodiscard]] PlannerInput DistantWheelInput() {
  PlannerInput input = test::MakeValidWheelInput();
  input.request_id = "hierarchical-wheel";
  input.world.global_map = test::MakeFlatMap("map", 24U, 8U, 1.0);
  input.world.local_map = test::MakeFlatMap("odom", 12U, 8U, 1.0);
  input.world.local_map.origin_m.x = -1.0;
  input.config.global_map.base_resolution_m = 1.0;
  input.goal_map.target = PointGoal{
      .position_m = {18.5, 3.5, 0.0},
      .tolerance_m = 0.2,
  };
  return input;
}

[[nodiscard]] PlannerInput DistantLeggedInput() {
  PlannerInput input = test::MakeValidLeggedInput();
  input.request_id = "hierarchical-legged";
  input.world.global_map = test::MakeFlatMap("map", 24U, 8U, 1.0);
  input.world.local_map = test::MakeFlatMap("odom", 12U, 8U, 1.0);
  input.world.local_map.origin_m.x = -1.0;
  input.config.global_map.base_resolution_m = 1.0;
  input.goal_map.target = PointGoal{
      .position_m = {18.5, 3.5, 0.0},
      .tolerance_m = 0.2,
  };
  return input;
}

[[nodiscard]] PlannerInput LongRunningLeggedInput() {
  PlannerInput input = DistantLeggedInput();
  input.request_id = "hierarchical-legged-cancel-during-local";
  input.world.global_map = test::MakeFlatMap("map", 250U, 250U, 0.2);
  input.world.local_map = test::MakeFlatMap("odom", 250U, 250U, 0.2);
  input.world.local_map.origin_m.x = -1.0;
  input.config.global_map.base_resolution_m = 0.2;
  input.config.legged.xy_resolution_m = 0.2;
  input.goal_map.target = PointGoal{
      .position_m = {47.9, 25.1, 0.0},
      .tolerance_m = 0.2,
  };
  std::get<LeggedState>(input.current_state).body_pose.position_m = {
      2.1, 25.1, 0.5};
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

TEST(CandidateDispositionContract, DefaultsNonTargetOutcomesToKeep) {
  constexpr std::array<PlanningOutcome, 8U> outcomes{
      PlanningOutcome::kNewReferenceAvailable,
      PlanningOutcome::kSafeFrontierReferenceAvailable,
      PlanningOutcome::kInvalidRequest,
      PlanningOutcome::kStaleInput,
      PlanningOutcome::kNumericalFailure,
      PlanningOutcome::kResourceExhausted,
      PlanningOutcome::kActiveReferenceInvalidated,
      PlanningOutcome::kCanceled,
  };

  for (const PlanningOutcome outcome : outcomes) {
    PlannerOutput output{.outcome = outcome};
    EXPECT_EQ(output.candidate_disposition, CandidateDisposition::kKeep)
        << static_cast<int>(outcome);
  }
}

TEST(CandidateDispositionContract, KeepsConcreteInfrastructureFailures) {
  PlannerInput numerical = DistantWheelInput();
  numerical.config.global_search.slope_weight =
      std::numeric_limits<double>::quiet_NaN();
  const PlannerOutput numerical_output = Planner{}.Plan(numerical);
  ASSERT_EQ(numerical_output.outcome, PlanningOutcome::kNumericalFailure);
  EXPECT_EQ(numerical_output.candidate_disposition,
            CandidateDisposition::kKeep);

  PlannerInput resource = DistantWheelInput();
  resource.world.global_map = test::MakeFlatMap("map", 5'000U, 1U, 3.2);
  resource.world.local_map = test::MakeFlatMap("odom", 12U, 12U, 0.2);
  resource.config.global_map.base_resolution_m = 0.2;
  const PlannerOutput resource_output = Planner{}.Plan(resource);
  ASSERT_EQ(resource_output.outcome, PlanningOutcome::kResourceExhausted);
  EXPECT_EQ(resource_output.candidate_disposition,
            CandidateDisposition::kKeep);
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
  EXPECT_EQ(output.candidate_disposition, CandidateDisposition::kKeep);
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

TEST(HierarchicalPlanner, ReportsTheSuccessfulLeggedSearchProblemDomain) {
  Planner planner;
  const PlannerInput input = DistantLeggedInput();
  const GlobalRoutePlanResult global = PlanGroundGlobalRoute(input);
  ASSERT_TRUE(global.ok()) << global.reason_code;
  const LocalFrontierResult frontiers =
      BuildLocalFrontiers(input, *global.route);
  ASSERT_TRUE(frontiers.ok()) << frontiers.reason_code;

  const PlannerOutput first = planner.Plan(input);
  const PlannerOutput repeated = planner.Plan(input);

  ASSERT_TRUE(first.reference.has_value()) << first.reason_code;
  ASSERT_TRUE(repeated.reference.has_value()) << repeated.reason_code;
  ASSERT_TRUE(first.diagnostics.hierarchical.has_value());
  ASSERT_TRUE(repeated.diagnostics.hierarchical.has_value());
  const auto& metrics = *first.diagnostics.hierarchical;
  ASSERT_GT(metrics.local_frontier_attempts, 0U);
  ASSERT_LE(metrics.local_frontier_attempts, frontiers.problems.size());
  const auto& successful_problem =
      frontiers.problems[metrics.local_frontier_attempts - 1U];
  EXPECT_DOUBLE_EQ(metrics.additional_corridor_margin_m, 2.0);
  EXPECT_DOUBLE_EQ(
      metrics.corridor_half_width_m,
      std::hypot(0.68 / 2.0, 0.33 / 2.0) + 0.3 + 2.0);
  EXPECT_EQ(metrics.search_domain_cell_count,
            successful_problem.search_domain.allowed_cell_count());
  EXPECT_EQ(metrics.search_domain_sha256,
            successful_problem.search_domain.sha256());
  EXPECT_EQ(metrics.local_search_runs, metrics.local_frontier_attempts);
  EXPECT_EQ(metrics.global_replans, 0U);
  EXPECT_TRUE(metrics.physical_goal_feasible);
  EXPECT_EQ(repeated.diagnostics.hierarchical->search_domain_sha256,
            metrics.search_domain_sha256);
}

TEST(HierarchicalPlanner, ReportsTheLastActuallySearchedLeggedProblem) {
  Planner planner;
  PlannerInput input = DistantLeggedInput();
  auto& capability = std::get<LeggedCapability>(input.capability);
  capability.motion_primitives = {
      LeggedBodyPrimitive{
          .primitive_id = "spin-only",
          .kind = LeggedPrimitiveKind::kSpin,
          .yaw_change_rad = 1.5707963267948966,
      },
  };
  const GlobalRoutePlanResult global = PlanGroundGlobalRoute(input);
  ASSERT_TRUE(global.ok()) << global.reason_code;
  const LocalFrontierResult frontiers =
      BuildLocalFrontiers(input, *global.route);
  ASSERT_TRUE(frontiers.ok()) << frontiers.reason_code;
  ASSERT_GT(frontiers.problems.size(), 1U);

  const PlannerOutput output = planner.Plan(input);

  ASSERT_EQ(output.outcome, PlanningOutcome::kNoKnownSafeRoute);
  EXPECT_EQ(output.candidate_disposition,
            CandidateDisposition::kSuppressForCurrentPhysicalSnapshot);
  ASSERT_TRUE(output.diagnostics.hierarchical.has_value());
  const auto& metrics = *output.diagnostics.hierarchical;
  const auto& last_problem = frontiers.problems.back();
  EXPECT_EQ(metrics.local_frontier_attempts, frontiers.problems.size());
  EXPECT_EQ(metrics.local_search_runs, frontiers.problems.size());
  EXPECT_EQ(metrics.search_domain_cell_count,
            last_problem.search_domain.allowed_cell_count());
  EXPECT_EQ(metrics.search_domain_sha256,
            last_problem.search_domain.sha256());
  EXPECT_EQ(metrics.global_replans, 0U);
  EXPECT_TRUE(metrics.physical_goal_feasible);
}

TEST(HierarchicalPlanner, LeavesDomainDiagnosticsEmptyWithoutALocalSearch) {
  Planner planner;
  PlannerInput input = DistantLeggedInput();
  input.config.local_frontier.additional_corridor_margin_m = 0.8;

  const PlannerOutput output = planner.Plan(input);

  ASSERT_EQ(output.outcome, PlanningOutcome::kInvalidRequest);
  EXPECT_EQ(output.candidate_disposition, CandidateDisposition::kKeep);
  EXPECT_EQ(output.reason_code, "LOCAL_FRONTIER_CONFIGURATION_INVALID");
  ASSERT_TRUE(output.diagnostics.hierarchical.has_value());
  const auto& metrics = *output.diagnostics.hierarchical;
  EXPECT_EQ(metrics.search_domain_cell_count, 0U);
  EXPECT_TRUE(metrics.search_domain_sha256.empty());
  EXPECT_EQ(metrics.local_frontier_attempts, 0U);
  EXPECT_EQ(metrics.local_search_runs, 0U);
  EXPECT_FALSE(metrics.physical_goal_feasible);
}

TEST(HierarchicalPlanner, DistinguishesGlobalAndLocalRejections) {
  Planner planner;
  PlannerInput global_blocked = DistantWheelInput();
  SetGlobalForbiddenColumn(global_blocked, 8U);

  const PlannerOutput global = planner.Plan(global_blocked);

  EXPECT_EQ(global.outcome, PlanningOutcome::kNoKnownSafeRoute);
  EXPECT_EQ(global.candidate_disposition,
            CandidateDisposition::kSuppressForCurrentPhysicalSnapshot);
  EXPECT_EQ(global.reason_code, "GLOBAL_NO_KNOWN_SAFE_ROUTE");
  EXPECT_FALSE(global.reference.has_value());

  PlannerInput local_unavailable = DistantWheelInput();
  auto &valid = std::get<std::vector<std::uint8_t>>(
      local_unavailable.world.local_map.layers.at("valid_mask").values);
  std::fill(valid.begin(), valid.end(), 0U);

  const PlannerOutput local = planner.Plan(local_unavailable);

  EXPECT_EQ(local.outcome, PlanningOutcome::kNoKnownSafeRoute);
  EXPECT_EQ(local.candidate_disposition,
            CandidateDisposition::kSuppressForCurrentPhysicalSnapshot);
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
  EXPECT_EQ(output.diagnostics.hierarchical->local_frontier_attempts, 1U);
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
  EXPECT_EQ(output.candidate_disposition,
            CandidateDisposition::kSuppressForCurrentPhysicalSnapshot);
  EXPECT_EQ(output.reason_code, "LOCAL_SEGMENT_INFEASIBLE");
  EXPECT_NE(std::ranges::find(output.diagnostics.warning_codes,
                              "LOCAL_SEARCH_DOMAIN_EXHAUSTED"),
            output.diagnostics.warning_codes.end());
}

TEST(HierarchicalPlanner, ReroutesAfterAConditionalCorridorFailsLocalSweep) {
  Planner planner;
  PlannerInput input = test::MakeValidWheelInput();
  input.request_id = "conditional-corridor-reroute";
  input.world.global_map = test::MakeFlatMap("map", 14U, 13U, 1.0);
  input.world.local_map = test::MakeFlatMap("odom", 18U, 13U, 1.0);
  input.world.local_map.origin_m.x = -2.0;
  input.config.global_map.base_resolution_m = 1.0;
  input.config.local_frontier.wheel_horizon_m = 20.0;
  // The coarse 1 m test lattice needs room to realize the rerouted Manhattan
  // turns; production L0 uses the approved 0.20 m primitives.
  input.config.local_frontier.additional_corridor_margin_m = 2.0;
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
  auto &global_obstacles = std::get<std::vector<std::uint8_t>>(
      input.world.global_map.layers.at("obstacle").values);
  auto &local_obstacles = std::get<std::vector<std::uint8_t>>(
      input.world.local_map.layers.at("obstacle").values);
  for (std::size_t y = 0U; y < 13U; ++y) {
    if (y != 4U && (y < 8U || y > 10U)) {
      global_obstacles.at(y * input.world.global_map.width + 6U) = 1U;
      local_obstacles.at(y * input.world.local_map.width + 8U) = 1U;
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
  ASSERT_TRUE(output.diagnostics.hierarchical.has_value());
  const auto &metrics = *output.diagnostics.hierarchical;
  EXPECT_GE(metrics.global_replans, 1U);
  EXPECT_GE(metrics.local_search_runs, 2U);
  EXPECT_GE(metrics.global_elapsed, std::chrono::nanoseconds::zero());
  EXPECT_GE(metrics.local_elapsed, std::chrono::nanoseconds::zero());
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
  const PlannerOutput canceled_output = planner.Plan(canceled);
  EXPECT_EQ(canceled_output.outcome, PlanningOutcome::kCanceled);
  EXPECT_EQ(canceled_output.candidate_disposition, CandidateDisposition::kKeep);

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

TEST(HierarchicalPlanner,
     CancellationBeforeLocalSearchLeavesGoalFeasibilityUnproven) {
  Planner planner;
  PlannerInput input = DistantLeggedInput();
  std::stop_source stop;
  input.stop_token = stop.get_token();
  bool observer_called = false;

  const PlannerOutput output = planner.Plan(
      input, [&](const ProvisionalGlobalRoute&) {
        observer_called = true;
        stop.request_stop();
      });

  EXPECT_TRUE(observer_called);
  EXPECT_EQ(output.outcome, PlanningOutcome::kCanceled);
  ASSERT_TRUE(output.diagnostics.hierarchical.has_value());
  EXPECT_EQ(output.diagnostics.hierarchical->local_search_runs, 0U);
  EXPECT_FALSE(output.diagnostics.hierarchical->physical_goal_feasible);
}

TEST(HierarchicalPlanner,
     CancellationDuringLocalSearchLeavesGoalFeasibilityUnproven) {
  Planner planner;
  PlannerInput input = LongRunningLeggedInput();
  std::stop_source stop;
  input.stop_token = stop.get_token();
  std::promise<void> global_route_observed;
  std::future<void> global_route_ready = global_route_observed.get_future();

  std::future<PlannerOutput> planning = std::async(
      std::launch::async, [&] {
        return planner.Plan(
            input, [&](const ProvisionalGlobalRoute&) {
              global_route_observed.set_value();
            });
      });
  ASSERT_EQ(global_route_ready.wait_for(std::chrono::seconds{5}),
            std::future_status::ready);
  std::this_thread::sleep_for(std::chrono::milliseconds{100});
  stop.request_stop();
  const PlannerOutput output = planning.get();

  EXPECT_EQ(output.outcome, PlanningOutcome::kCanceled);
  ASSERT_TRUE(output.diagnostics.hierarchical.has_value());
  EXPECT_GT(output.diagnostics.hierarchical->local_search_runs, 0U);
  EXPECT_FALSE(output.diagnostics.hierarchical->physical_goal_feasible);
}

TEST(HierarchicalPlanner,
     InvalidLocalSearchLeavesGoalFeasibilityUnproven) {
  Planner planner;
  PlannerInput input = DistantLeggedInput();
  input.config.legged.yaw_bin_count = 0U;

  const PlannerOutput output = planner.Plan(input);

  EXPECT_EQ(output.outcome, PlanningOutcome::kInvalidRequest);
  EXPECT_EQ(output.candidate_disposition, CandidateDisposition::kKeep);
  EXPECT_EQ(output.reason_code, "LEGGED_LATTICE_REQUEST_INVALID");
  ASSERT_TRUE(output.diagnostics.hierarchical.has_value());
  EXPECT_EQ(output.diagnostics.hierarchical->local_search_runs, 1U);
  EXPECT_FALSE(output.diagnostics.hierarchical->physical_goal_feasible);
}

TEST(HierarchicalPlanner, ReportsGlobalAndWheelLocalProjectionCacheHits) {
  Planner planner;
  const PlannerInput input = DistantWheelInput();

  const PlannerOutput cold = planner.Plan(input);
  const PlannerOutput warm = planner.Plan(input);

  ASSERT_TRUE(cold.reference.has_value()) << cold.reason_code;
  ASSERT_TRUE(warm.reference.has_value()) << warm.reason_code;
  ASSERT_TRUE(cold.diagnostics.hierarchical.has_value());
  ASSERT_TRUE(warm.diagnostics.hierarchical.has_value());
  EXPECT_EQ(cold.diagnostics.hierarchical->global_projection_cache_hits, 0U);
  EXPECT_EQ(cold.diagnostics.hierarchical->local_projection_cache_hits, 0U);
  EXPECT_EQ(warm.diagnostics.hierarchical->global_projection_cache_hits, 1U);
  EXPECT_EQ(warm.diagnostics.hierarchical->local_projection_cache_hits, 1U);
  EXPECT_EQ(cold.reference->preview.poses_map,
            warm.reference->preview.poses_map);
}

TEST(HierarchicalPlanner, PublishesProvisionalRouteAndIsolatesObserverErrors) {
  Planner planner;
  const PlannerInput input = DistantWheelInput();
  std::optional<ProvisionalGlobalRoute> observed;

  const PlannerOutput output = planner.Plan(
      input, [&](const ProvisionalGlobalRoute &route) {
        observed = route;
        throw std::runtime_error{"non-authoritative observer failure"};
      });

  ASSERT_TRUE(output.reference.has_value()) << output.reason_code;
  ASSERT_TRUE(observed.has_value());
  EXPECT_EQ(observed->request_id, input.request_id);
  EXPECT_EQ(observed->route_id, "wheel-route/" + input.request_id);
  EXPECT_EQ(observed->platform_type, PlatformType::kWheeled);
  EXPECT_EQ(observed->poses_map, output.reference->preview.poses_map);
}

TEST(HierarchicalPlanner,
     KeepsLocalBackendFailuresUntilTheHierarchyExhaustsTheTarget) {
  PlannerInput wheel_input = DistantWheelInput();
  auto &wheel_capability = std::get<WheeledCapability>(wheel_input.capability);
  wheel_capability.motion_primitives = {
      WheelMotionPrimitive{
          .primitive_id = "spin-only",
          .kind = WheelPrimitiveKind::kSpinCounterclockwise,
          .relative_end_pose =
              Pose3{.orientation = Quaternion{.w = 0.9238795325112867,
                                              .z = 0.3826834323650898}},
      },
  };
  const GlobalRoutePlanResult wheel_global = PlanGroundGlobalRoute(wheel_input);
  ASSERT_TRUE(wheel_global.ok()) << wheel_global.reason_code;
  const LocalFrontierResult wheel_frontiers =
      BuildLocalFrontiers(wheel_input, *wheel_global.route);
  ASSERT_TRUE(wheel_frontiers.ok()) << wheel_frontiers.reason_code;

  const PlannerOutput wheel_backend =
      wheel::WheelPlanner{}.PlanRanked(wheel_frontiers.problems).output;
  const PlannerOutput wheel_public = Planner{}.Plan(wheel_input);

  ASSERT_EQ(wheel_backend.outcome, PlanningOutcome::kNoKnownSafeRoute);
  EXPECT_EQ(wheel_backend.candidate_disposition, CandidateDisposition::kKeep);
  ASSERT_EQ(wheel_public.outcome, PlanningOutcome::kNoKnownSafeRoute);
  EXPECT_EQ(wheel_public.candidate_disposition,
            CandidateDisposition::kSuppressForCurrentPhysicalSnapshot);

  PlannerInput legged_input = DistantLeggedInput();
  auto &legged_capability = std::get<LeggedCapability>(legged_input.capability);
  legged_capability.motion_primitives = {
      LeggedBodyPrimitive{
          .primitive_id = "spin-only",
          .kind = LeggedPrimitiveKind::kSpin,
          .yaw_change_rad = 1.5707963267948966,
      },
  };
  const GlobalRoutePlanResult legged_global =
      PlanGroundGlobalRoute(legged_input);
  ASSERT_TRUE(legged_global.ok()) << legged_global.reason_code;
  const LocalFrontierResult legged_frontiers =
      BuildLocalFrontiers(legged_input, *legged_global.route);
  ASSERT_TRUE(legged_frontiers.ok()) << legged_frontiers.reason_code;
  ASSERT_FALSE(legged_frontiers.problems.empty());

  const PlannerOutput legged_backend =
      legged::LeggedPlanner{}.Plan(legged_frontiers.problems.back());
  const PlannerOutput legged_public = Planner{}.Plan(legged_input);

  ASSERT_EQ(legged_backend.outcome, PlanningOutcome::kNoKnownSafeRoute);
  EXPECT_EQ(legged_backend.candidate_disposition, CandidateDisposition::kKeep);
  ASSERT_EQ(legged_public.outcome, PlanningOutcome::kNoKnownSafeRoute);
  EXPECT_EQ(legged_public.candidate_disposition,
            CandidateDisposition::kSuppressForCurrentPhysicalSnapshot);
}

} // namespace
} // namespace lunar::pure_planning::hierarchical

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <vector>
#include <variant>

#include <gtest/gtest.h>

#include "hierarchical/global_route_planner.hpp"
#include "hierarchical/local_frontier.hpp"
#include "lunar_pure_planner_core/planner.hpp"
#include "shared/map_snapshot.hpp"
#include "shared/safe_projection.hpp"
#include "test_fixtures.hpp"
#include "wheel/wheel_lattice.hpp"
#include "wheel/wheel_planner.hpp"
#include "wheel/wheel_spline_optimizer.hpp"
#include "wheel/wheel_timing.hpp"

namespace lunar::pure_planning {
namespace {

double Yaw(const Quaternion& orientation) {
  return std::atan2(
      2.0 * (orientation.w * orientation.z +
             orientation.x * orientation.y),
      1.0 - 2.0 * (orientation.y * orientation.y +
                   orientation.z * orientation.z));
}

const TrajectoryReference& WheelTrajectory(const PlannerOutput& output) {
  EXPECT_TRUE(output.reference.has_value());
  EXPECT_EQ(output.reference->platform_type, PlatformType::kWheeled);
  const auto* trajectory =
      std::get_if<TrajectoryReference>(&output.reference->data);
  EXPECT_NE(trajectory, nullptr);
  return *trajectory;
}

const LocalTrajectoryDiagnostics& LocalDiagnostics(
    const PlannerOutput& output) {
  EXPECT_TRUE(output.diagnostics.local_trajectory.has_value());
  return *output.diagnostics.local_trajectory;
}

hierarchical::LocalPlanningProblem MakeLocalWheelProblem(
    const PlannerInput& input, std::vector<std::uint8_t> allowed) {
  return hierarchical::LocalPlanningProblem{
      .request_id = input.request_id,
      .platform_id = input.platform_id,
      .capability_version = input.capability_version,
      .local_map_generation = input.local_map_generation,
      .state_time = input.state_time,
      .current_state = input.current_state,
      .goal_odom = input.goal_map,
      .local_map = input.world.local_map,
      .search_domain = hierarchical::LocalSearchDomain{
          input.world.local_map.width, input.world.local_map.height,
          std::move(allowed)},
      .capability = input.capability,
      .config = input.config,
      .stop_token = input.stop_token,
  };
}

std::vector<std::uint8_t> EmptyDomain(const GridMap& map) {
  return std::vector<std::uint8_t>(map.CellCount(), 0U);
}

void AllowCell(std::vector<std::uint8_t>& allowed, const GridMap& map,
               const std::size_t x, const std::size_t y) {
  allowed.at(y * map.width + x) = 1U;
}

PlannerInput MakeWheelInputWithRequiredLocalCoverage() {
  PlannerInput input = test::MakeValidWheelInput();
  input.world.local_map.origin_m.x = -1.0;
  return input;
}

TEST(WheelPlanner, PlansForwardReferenceWithBoundedTiming) {
  Planner planner;
  auto input = MakeWheelInputWithRequiredLocalCoverage();
  input.config.optimization.maximum_iterations = 0U;

  const PlannerOutput output = planner.Plan(input);

  ASSERT_EQ(output.outcome, PlanningOutcome::kNewReferenceAvailable)
      << output.reason_code;
  EXPECT_EQ(output.directive, ExecutionDirective::kActivateNewReference);
  EXPECT_EQ(output.diagnostics.planner_name, "cpp_v3_hierarchical");
  ASSERT_TRUE(output.diagnostics.best_cost.has_value());
  const TrajectoryReference& trajectory = WheelTrajectory(output);
  ASSERT_GT(trajectory.points.size(), 2U);
  EXPECT_EQ(trajectory.semantics, TrajectorySemantics::kWheeledBase);
  const LocalTrajectoryDiagnostics& diagnostics = LocalDiagnostics(output);
  EXPECT_EQ(diagnostics.trajectory_mode,
            TrajectoryMode::kDiscreteFallback);
  EXPECT_EQ(diagnostics.collision_validation,
            CollisionValidation::kCertified);
  EXPECT_TRUE(std::isfinite(diagnostics.start_anchor_error_m));
  EXPECT_TRUE(std::isfinite(diagnostics.endpoint_error_m));
  EXPECT_TRUE(std::isfinite(diagnostics.maximum_curvature_per_m));
  EXPECT_TRUE(std::isfinite(diagnostics.smoothing_elapsed_s));
  EXPECT_NEAR(trajectory.points.front().pose.position_m.x, 2.5, 1.0e-9);
  EXPECT_NEAR(trajectory.points.back().pose.position_m.x, 4.5, 1.0e-9);
  EXPECT_NEAR(trajectory.points.back().pose.position_m.y, 3.5, 1.0e-9);
  EXPECT_NEAR(diagnostics.endpoint_error_m, 0.0, 1.0e-9);
  for (std::size_t index = 1U; index < trajectory.points.size(); ++index) {
    EXPECT_GT(
        trajectory.points[index].time_from_start,
        trajectory.points[index - 1U].time_from_start);
    EXPECT_LE(
        std::hypot(
            trajectory.points[index].velocity.linear_mps.x,
            trajectory.points[index].velocity.linear_mps.y),
        1.0 + 1.0e-9);
  }
}

TEST(WheelPlanner, SearchDomainBoundaryDoesNotBecomePhysicalObstacle) {
  auto input = test::MakeValidWheelInput();
  input.request_id = "wheel-domain-footprint-overhang";
  input.config.optimization.maximum_iterations = 0U;
  auto& state = std::get<WheeledState>(input.current_state);
  state.pose.position_m.y = 3.1;
  std::get<PointGoal>(input.goal_map.target).position_m.y = 3.1;
  auto allowed = EmptyDomain(input.world.local_map);
  for (std::size_t x = 2U; x <= 4U; ++x) {
    AllowCell(allowed, input.world.local_map, x, 3U);
  }
  const auto problem = MakeLocalWheelProblem(input, std::move(allowed));

  const PlannerOutput output = wheel::WheelPlanner{}.Plan(problem);

  ASSERT_EQ(output.outcome, PlanningOutcome::kNewReferenceAvailable)
      << output.reason_code;
  EXPECT_EQ(output.reason_code, "WHEEL_PLAN_AVAILABLE");
  EXPECT_GT(output.diagnostics.expanded_states, 0U);
  EXPECT_TRUE(output.reference.has_value());
}

TEST(WheelPlanner, RejectsPrimitiveWhoseTargetCenterLeavesSearchDomain) {
  auto input = test::MakeValidWheelInput();
  input.request_id = "wheel-domain-primitive-target";
  std::get<PointGoal>(input.goal_map.target).position_m.x = 6.5;
  auto allowed = EmptyDomain(input.world.local_map);
  AllowCell(allowed, input.world.local_map, 2U, 3U);
  AllowCell(allowed, input.world.local_map, 6U, 3U);
  const auto problem = MakeLocalWheelProblem(input, std::move(allowed));

  const PlannerOutput output = wheel::WheelPlanner{}.Plan(problem);

  EXPECT_EQ(output.outcome, PlanningOutcome::kNoKnownSafeRoute);
  EXPECT_EQ(output.directive, ExecutionDirective::kNoSafeReference);
  EXPECT_EQ(output.reason_code, "LOCAL_SEARCH_DOMAIN_EXHAUSTED");
  EXPECT_GT(output.diagnostics.expanded_states, 0U);
  EXPECT_FALSE(output.reference.has_value());
}

TEST(WheelPlanner, TranslationOnlyDomainBlockedStartReportsDomainExhaustion) {
  auto input = test::MakeValidWheelInput();
  input.request_id = "wheel-domain-translation-only-start";
  std::get<PointGoal>(input.goal_map.target).position_m.x = 6.5;
  auto& capability = std::get<WheeledCapability>(input.capability);
  capability.motion_primitives = {
      WheelMotionPrimitive{
          .primitive_id = "forward",
          .kind = WheelPrimitiveKind::kForward,
          .relative_end_pose = Pose3{.position_m = {1.0, 0.0, 0.0}},
      },
  };
  auto allowed = EmptyDomain(input.world.local_map);
  AllowCell(allowed, input.world.local_map, 2U, 3U);
  const auto problem = MakeLocalWheelProblem(input, std::move(allowed));

  const PlannerOutput output = wheel::WheelPlanner{}.Plan(problem);

  EXPECT_EQ(output.outcome, PlanningOutcome::kNoKnownSafeRoute);
  EXPECT_EQ(output.directive, ExecutionDirective::kNoSafeReference);
  EXPECT_EQ(output.reason_code, "LOCAL_SEARCH_DOMAIN_EXHAUSTED");
  EXPECT_GT(output.diagnostics.expanded_states, 0U);
  EXPECT_FALSE(output.reference.has_value());
}

TEST(WheelPlanner, TranslationOnlyPhysicalBlockKeepsStartInfeasibleReason) {
  auto input = test::MakeValidWheelInput();
  input.request_id = "wheel-domain-before-physical-start";
  auto& state = std::get<WheeledState>(input.current_state);
  state.pose.position_m.y = 3.1;
  auto& goal = std::get<PointGoal>(input.goal_map.target);
  goal.position_m = {6.5, 3.1, 0.0};
  auto& capability = std::get<WheeledCapability>(input.capability);
  capability.motion_primitives = {
      WheelMotionPrimitive{
          .primitive_id = "forward",
          .kind = WheelPrimitiveKind::kForward,
          .relative_end_pose = Pose3{.position_m = {1.0, 0.0, 0.0}},
      },
  };
  auto& obstacle = std::get<std::vector<std::uint8_t>>(
      input.world.local_map.layers.at("obstacle").values);
  obstacle[2U * input.world.local_map.width + 4U] = 1U;
  auto allowed = EmptyDomain(input.world.local_map);
  AllowCell(allowed, input.world.local_map, 2U, 3U);
  const auto problem = MakeLocalWheelProblem(input, std::move(allowed));

  const PlannerOutput output = wheel::WheelPlanner{}.Plan(problem);

  EXPECT_EQ(output.outcome, PlanningOutcome::kNoKnownSafeRoute);
  EXPECT_EQ(output.directive, ExecutionDirective::kNoSafeReference);
  EXPECT_EQ(output.reason_code, "WHEEL_START_CONNECTOR_INFEASIBLE");
  EXPECT_FALSE(output.reference.has_value());
}

TEST(WheelPlanner, PhysicallyUnsafeGoalFailsBeforeSearchExpansion) {
  auto input = test::MakeValidWheelInput();
  input.request_id = "wheel-physical-goal-infeasible";
  auto& obstacle = std::get<std::vector<std::uint8_t>>(
      input.world.local_map.layers.at("obstacle").values);
  obstacle.at(3U * input.world.local_map.width + 4U) = 1U;
  auto allowed = std::vector<std::uint8_t>(
      input.world.local_map.CellCount(), 1U);
  allowed[3U * input.world.local_map.width + 4U] = 0U;
  const auto problem = MakeLocalWheelProblem(input, std::move(allowed));

  const PlannerOutput output = wheel::WheelPlanner{}.Plan(problem);

  EXPECT_EQ(output.outcome, PlanningOutcome::kGoalInfeasible);
  EXPECT_EQ(output.directive, ExecutionDirective::kHoldPosition);
  EXPECT_EQ(output.reason_code, "WHEEL_GOAL_INFEASIBLE");
  EXPECT_EQ(output.diagnostics.expanded_states, 0U);
  EXPECT_FALSE(output.reference.has_value());
}

TEST(WheelPlanner, PhysicallySafeGoalReportsLocalSearchDomainExhaustion) {
  auto input = test::MakeValidWheelInput();
  input.request_id = "wheel-domain-exhausted";
  auto allowed = std::vector<std::uint8_t>(
      input.world.local_map.CellCount(), 1U);
  for (std::size_t y = 0U; y < input.world.local_map.height; ++y) {
    allowed[y * input.world.local_map.width + 3U] = 0U;
  }
  const auto problem = MakeLocalWheelProblem(input, std::move(allowed));

  const PlannerOutput output = wheel::WheelPlanner{}.Plan(problem);

  EXPECT_EQ(output.outcome, PlanningOutcome::kNoKnownSafeRoute);
  EXPECT_EQ(output.directive, ExecutionDirective::kNoSafeReference);
  EXPECT_EQ(output.reason_code, "LOCAL_SEARCH_DOMAIN_EXHAUSTED");
  EXPECT_GT(output.diagnostics.expanded_states, 0U);
  EXPECT_FALSE(output.reference.has_value());
}

TEST(WheelPlanner, LazySearchPreservesContinuousStartAndExactPointGoal) {
  auto input = test::MakeValidWheelInput();
  input.goal_map.yaw_rad = 0.0;
  input.goal_map.yaw_tolerance_rad = 0.05;
  const auto snapshot = shared::MapSnapshot::Create(input.world.local_map);
  ASSERT_TRUE(snapshot.ok()) << snapshot.reason_code;
  const auto projection = shared::BuildSafeProjection(
      snapshot.snapshot, input.capability, input.config.map_safety, {});
  ASSERT_TRUE(projection.ok()) << projection.reason_code;

  const auto search = wheel::SearchWheelLattice(
      std::get<WheeledState>(input.current_state), input.goal_map,
      *projection.projection, std::get<WheeledCapability>(input.capability),
      input.config, {});

  ASSERT_TRUE(search.ok()) << search.reason_code;
  ASSERT_TRUE(search.plan.has_value());
  ASSERT_FALSE(search.plan->transitions.empty());
  EXPECT_EQ(search.plan->transitions.front().source_pose.position_m,
            std::get<WheeledState>(input.current_state).pose.position_m);
  EXPECT_EQ(search.plan->transitions.back().target_pose.position_m,
            std::get<PointGoal>(input.goal_map.target).position_m);
}

TEST(WheelPlanner, RankedSearchReusesExhaustedTreeForNearerFrontier) {
  const auto input = test::MakeValidWheelInput();
  const auto snapshot = shared::MapSnapshot::Create(input.world.local_map);
  ASSERT_TRUE(snapshot.ok()) << snapshot.reason_code;
  const auto projection = shared::BuildSafeProjection(
      snapshot.snapshot, input.capability, input.config.map_safety, {});
  ASSERT_TRUE(projection.ok()) << projection.reason_code;
  const std::vector<GoalRegion> ranked_goals{
      GoalRegion{
          .goal_id = "outside-local-window",
          .target = PointGoal{
              .position_m = {1000.0, 1000.0, 0.0}, .tolerance_m = 0.0},
      },
      input.goal_map,
  };

  const auto search = wheel::SearchWheelLatticeRanked(
      std::get<WheeledState>(input.current_state), ranked_goals,
      *projection.projection,
      hierarchical::LocalSearchDomain{
          input.world.local_map.width, input.world.local_map.height,
          std::vector<std::uint8_t>(input.world.local_map.CellCount(), 1U)},
      std::get<WheeledCapability>(input.capability),
      input.config, {});

  ASSERT_TRUE(search.ok()) << search.reason_code;
  ASSERT_TRUE(search.selected_goal_index.has_value());
  EXPECT_EQ(*search.selected_goal_index, 1U);
  ASSERT_TRUE(search.plan.has_value());
  ASSERT_FALSE(search.plan->transitions.empty());
  EXPECT_EQ(search.plan->transitions.back().target_pose.position_m,
            std::get<PointGoal>(input.goal_map.target).position_m);
}

TEST(WheelPlanner, RankedPlannerBuildsProjectionOnceAndReportsChosenProblem) {
  const auto input = test::MakeValidWheelInput();
  std::vector<hierarchical::LocalPlanningProblem> problems{
      hierarchical::LocalPlanningProblem{
          .request_id = "far",
          .state_time = input.state_time,
          .current_state = input.current_state,
          .goal_odom = GoalRegion{
              .goal_id = "outside-local-window",
              .target = PointGoal{
                  .position_m = {1000.0, 1000.0, 0.0}, .tolerance_m = 0.0},
          },
          .local_map = input.world.local_map,
          .search_domain = hierarchical::LocalSearchDomain{
              input.world.local_map.width, input.world.local_map.height,
              std::vector<std::uint8_t>(input.world.local_map.CellCount(), 1U)},
          .capability = input.capability,
          .config = input.config,
      },
      hierarchical::LocalPlanningProblem{
          .request_id = "near",
          .state_time = input.state_time,
          .current_state = input.current_state,
          .goal_odom = input.goal_map,
          .local_map = input.world.local_map,
          .search_domain = hierarchical::LocalSearchDomain{
              input.world.local_map.width, input.world.local_map.height,
              std::vector<std::uint8_t>(input.world.local_map.CellCount(), 1U)},
          .capability = input.capability,
          .config = input.config,
      },
  };

  const wheel::WheelRankedPlanResult result =
      wheel::WheelPlanner{}.PlanRanked(problems);

  ASSERT_EQ(result.output.outcome, PlanningOutcome::kNewReferenceAvailable)
      << result.output.reason_code;
  ASSERT_TRUE(result.selected_problem_index.has_value());
  EXPECT_EQ(*result.selected_problem_index, 1U);
  ASSERT_TRUE(result.output.reference.has_value());
  EXPECT_EQ(result.output.reference->plan_id, "wheel/near");
}

TEST(WheelPlanner, RankedFallbackFinalValidationUsesTheSearchDomain) {
  const auto input = test::MakeValidWheelInput();
  auto far = MakeLocalWheelProblem(
      input, std::vector<std::uint8_t>(
                 input.world.local_map.CellCount(), 1U));
  far.request_id = "far-wide-domain";
  far.goal_odom = GoalRegion{
      .goal_id = "outside-local-window",
      .target = PointGoal{
          .position_m = {1000.0, 1000.0, 0.0},
          .tolerance_m = 0.0,
      },
  };
  auto narrow_allowed = EmptyDomain(input.world.local_map);
  AllowCell(narrow_allowed, input.world.local_map, 2U, 3U);
  auto near = MakeLocalWheelProblem(input, std::move(narrow_allowed));
  near.request_id = "near-narrow-domain";
  std::vector<hierarchical::LocalPlanningProblem> problems;
  problems.push_back(std::move(far));
  problems.push_back(std::move(near));

  const wheel::WheelRankedPlanResult result =
      wheel::WheelPlanner{}.PlanRanked(problems);

  ASSERT_EQ(result.output.outcome, PlanningOutcome::kNewReferenceAvailable)
      << result.output.reason_code;
  ASSERT_TRUE(result.selected_problem_index.has_value());
  EXPECT_EQ(*result.selected_problem_index, 1U);
  ASSERT_TRUE(result.output.reference.has_value());
  EXPECT_EQ(result.output.reference->plan_id, "wheel/near-narrow-domain");
}

TEST(WheelPlanner, RankedIndexedSearchIsDeterministicAcrossTwentyRuns) {
  const auto input = test::MakeValidWheelInput();
  const auto snapshot = shared::MapSnapshot::Create(input.world.local_map);
  ASSERT_TRUE(snapshot.ok()) << snapshot.reason_code;
  const auto projection = shared::BuildSafeProjection(
      snapshot.snapshot, input.capability, input.config.map_safety, {});
  ASSERT_TRUE(projection.ok()) << projection.reason_code;
  const std::vector<GoalRegion> ranked_goals{
      GoalRegion{
          .goal_id = "outside-local-window",
          .target = PointGoal{
              .position_m = {1000.0, 1000.0, 0.0}, .tolerance_m = 0.0},
      },
      input.goal_map,
  };
  std::optional<wheel::WheelDiscretePlan> expected_plan;
  std::optional<std::size_t> expected_goal;

  for (std::size_t run = 0U; run < 20U; ++run) {
    const auto search = wheel::SearchWheelLatticeRanked(
        std::get<WheeledState>(input.current_state), ranked_goals,
        *projection.projection,
        hierarchical::LocalSearchDomain{
            input.world.local_map.width, input.world.local_map.height,
            std::vector<std::uint8_t>(input.world.local_map.CellCount(), 1U)},
        std::get<WheeledCapability>(input.capability),
        input.config, {});
    ASSERT_TRUE(search.ok()) << "run=" << run << ' ' << search.reason_code;
    if (run == 0U) {
      expected_plan = search.plan;
      expected_goal = search.selected_goal_index;
    } else {
      EXPECT_EQ(search.selected_goal_index, expected_goal);
      ASSERT_TRUE(search.plan.has_value());
      ASSERT_TRUE(expected_plan.has_value());
      EXPECT_EQ(search.plan->transitions, expected_plan->transitions);
      EXPECT_DOUBLE_EQ(search.plan->cost, expected_plan->cost);
      EXPECT_EQ(search.plan->expanded_states, expected_plan->expanded_states);
    }
  }
}

TEST(WheelPlanner, ReusesLocalProjectionOnlyForAnIdenticalContentKey) {
  const auto input = test::MakeValidWheelInput();
  hierarchical::LocalPlanningProblem problem{
      .request_id = "cached-local",
      .platform_id = input.platform_id,
      .capability_version = input.capability_version,
      .local_map_generation = input.local_map_generation,
      .state_time = input.state_time,
      .current_state = input.current_state,
      .goal_odom = input.goal_map,
      .local_map = input.world.local_map,
      .search_domain = hierarchical::LocalSearchDomain{
          input.world.local_map.width, input.world.local_map.height,
          std::vector<std::uint8_t>(input.world.local_map.CellCount(), 1U)},
      .capability = input.capability,
      .config = input.config,
  };
  wheel::WheelPlanner planner;

  const auto cold = planner.PlanRanked(std::span{&problem, 1U});
  const auto warm = planner.PlanRanked(std::span{&problem, 1U});
  auto changed_domain = problem;
  auto restricted = std::vector<std::uint8_t>(
      input.world.local_map.CellCount(), 1U);
  restricted.front() = 0U;
  changed_domain.search_domain = hierarchical::LocalSearchDomain{
      input.world.local_map.width, input.world.local_map.height,
      std::move(restricted)};
  const auto domain_reuse =
      planner.PlanRanked(std::span{&changed_domain, 1U});
  auto changed_generation = problem;
  ++changed_generation.local_map_generation;
  const auto generation_miss =
      planner.PlanRanked(std::span{&changed_generation, 1U});
  auto changed_capability = changed_generation;
  std::get<WheeledCapability>(changed_capability.capability).wheelbase_m += 0.01;
  const auto capability_miss =
      planner.PlanRanked(std::span{&changed_capability, 1U});
  auto changed_safety = changed_capability;
  changed_safety.config.map_safety.minimum_observation_quality -= 0.01;
  const auto safety_miss =
      planner.PlanRanked(std::span{&changed_safety, 1U});

  ASSERT_TRUE(cold.output.reference.has_value()) << cold.output.reason_code;
  ASSERT_TRUE(warm.output.reference.has_value()) << warm.output.reason_code;
  ASSERT_TRUE(domain_reuse.output.reference.has_value())
      << domain_reuse.output.reason_code;
  EXPECT_FALSE(cold.projection_cache_hit);
  EXPECT_TRUE(warm.projection_cache_hit);
  EXPECT_TRUE(domain_reuse.projection_cache_hit);
  EXPECT_FALSE(generation_miss.projection_cache_hit);
  EXPECT_FALSE(capability_miss.projection_cache_hit);
  EXPECT_FALSE(safety_miss.projection_cache_hit);
  const auto& cold_trajectory = WheelTrajectory(cold.output);
  const auto& warm_trajectory = WheelTrajectory(warm.output);
  ASSERT_EQ(cold_trajectory.points.size(), warm_trajectory.points.size());
  EXPECT_EQ(cold_trajectory.points.front().pose,
            warm_trajectory.points.front().pose);
  EXPECT_EQ(cold_trajectory.points.back().pose,
            warm_trajectory.points.back().pose);
  EXPECT_EQ(cold.output.diagnostics.best_cost,
            warm.output.diagnostics.best_cost);
  EXPECT_EQ(cold.output.reason_code, warm.output.reason_code);
}

TEST(WheelPlanner, PlansToExactOffCellGoalWithSmallExecutionTolerance) {
  Planner planner;
  auto input = MakeWheelInputWithRequiredLocalCoverage();
  input.request_id = "wheel-exact-off-cell-goal";
  input.goal_map.target = PointGoal{
      .position_m = {4.37, 3.42, 0.0},
      .tolerance_m = 0.01,
  };

  const PlannerOutput output = planner.Plan(input);

  ASSERT_EQ(output.outcome, PlanningOutcome::kNewReferenceAvailable)
      << output.reason_code;
  const auto& final_pose = WheelTrajectory(output).points.back().pose;
  EXPECT_NEAR(final_pose.position_m.x, 4.37, 1.0e-9);
  EXPECT_NEAR(final_pose.position_m.y, 3.42, 1.0e-9);
  EXPECT_NEAR(LocalDiagnostics(output).endpoint_error_m, 0.0, 1.0e-9);
}

TEST(WheelPlanner, ConnectsAnExactNearGoalAcrossThePointTwoMeterGridGap) {
  auto input = test::MakeValidWheelInput();
  input.request_id = "wheel-near-goal-grid-gap";
  input.config.optimization.maximum_iterations = 0U;
  input.world.global_map = test::MakeFlatMap("map", 250U, 250U, 0.2);
  input.world.global_map.origin_m = {-25.0, -25.0, 0.0};
  input.world.local_map = test::MakeFlatMap("odom", 56U, 56U, 0.2);
  input.world.local_map.origin_m = {11.2, -20.4, 0.0};
  input.config.global_map.base_resolution_m = 0.2;
  input.config.wheel.xy_resolution_m = 0.2;
  std::get<WheeledCapability>(input.capability).motion_primitives = {
      WheelMotionPrimitive{
          .primitive_id = "forward",
          .kind = WheelPrimitiveKind::kForward,
          .relative_end_pose = Pose3{.position_m = {0.2, 0.0, 0.0}},
      },
      WheelMotionPrimitive{
          .primitive_id = "reverse",
          .kind = WheelPrimitiveKind::kReverse,
          .relative_end_pose = Pose3{.position_m = {-0.2, 0.0, 0.0}},
      },
      WheelMotionPrimitive{
          .primitive_id = "forward-arc-left",
          .kind = WheelPrimitiveKind::kForwardArc,
          .relative_end_pose = Pose3{
              .position_m = {0.19509032201612825, 0.01921471959676957, 0.0},
              .orientation = test::YawQuaternion(0.19634954084936207),
          },
      },
      WheelMotionPrimitive{
          .primitive_id = "forward-arc-right",
          .kind = WheelPrimitiveKind::kForwardArc,
          .relative_end_pose = Pose3{
              .position_m = {0.19509032201612825, -0.01921471959676957, 0.0},
              .orientation = test::YawQuaternion(-0.19634954084936207),
          },
      },
      WheelMotionPrimitive{
          .primitive_id = "spin-left",
          .kind = WheelPrimitiveKind::kSpinCounterclockwise,
          .relative_end_pose = Pose3{
              .orientation = test::YawQuaternion(0.19634954084936207),
          },
      },
      WheelMotionPrimitive{
          .primitive_id = "spin-right",
          .kind = WheelPrimitiveKind::kSpinClockwise,
          .relative_end_pose = Pose3{
              .orientation = test::YawQuaternion(-0.19634954084936207),
          },
      },
      WheelMotionPrimitive{
          .primitive_id = "stop-switch",
          .kind = WheelPrimitiveKind::kStopAndSwitch,
      },
  };
  input.current_state = WheeledState{
      .pose = Pose3{
          .position_m = {17.5, -14.9, 0.0},
          .orientation = test::YawQuaternion(0.0),
      },
  };
  input.goal_map = GoalRegion{
      .goal_id = "near-exact-goal",
      .target = PointGoal{
          .position_m = {18.0, -14.8, 0.0},
          .tolerance_m = 0.2,
      },
      .yaw_rad = 0.0,
      .yaw_tolerance_rad = 0.05,
  };
  const auto snapshot = shared::MapSnapshot::Create(input.world.local_map);
  ASSERT_TRUE(snapshot.ok()) << snapshot.reason_code;
  const auto projection = shared::BuildSafeProjection(
      snapshot.snapshot, input.capability, input.config.map_safety, {});
  ASSERT_TRUE(projection.ok()) << projection.reason_code;

  const auto search = wheel::SearchWheelLattice(
      std::get<WheeledState>(input.current_state), input.goal_map,
      *projection.projection, std::get<WheeledCapability>(input.capability),
      input.config, {});

  ASSERT_TRUE(search.ok()) << search.reason_code;
  ASSERT_TRUE(search.plan.has_value());
  ASSERT_FALSE(search.plan->transitions.empty());
  const auto synthetic_begin =
      std::ranges::find_if(search.plan->transitions, [&](const auto& transition) {
        return transition.primitive_index >=
            std::get<WheeledCapability>(input.capability)
                .motion_primitives.size();
      });
  ASSERT_NE(synthetic_begin, search.plan->transitions.end());
  for (auto transition = synthetic_begin;
       transition != search.plan->transitions.end(); ++transition) {
    const double dx = transition->target_pose.position_m.x -
        transition->source_pose.position_m.x;
    const double dy = transition->target_pose.position_m.y -
        transition->source_pose.position_m.y;
    if (transition->path_length_m <= 1.0e-9) {
      EXPECT_NEAR(std::hypot(dx, dy), 0.0, 1.0e-9);
      continue;
    }
    const double travel_heading = std::atan2(dy, dx);
    const double body_heading = transition->reverse
        ? wheel::NormalizeYaw(travel_heading + std::numbers::pi)
        : travel_heading;
    EXPECT_NEAR(
        wheel::ShortestYawDelta(
            transition->source_pose.yaw_rad, body_heading),
        0.0, 1.0e-9);
    EXPECT_NEAR(
        wheel::ShortestYawDelta(
            transition->target_pose.yaw_rad, body_heading),
        0.0, 1.0e-9);
  }
  const auto& terminal = search.plan->transitions.back().target_pose;
  EXPECT_NEAR(terminal.position_m.x, 18.0, 1.0e-9);
  EXPECT_NEAR(terminal.position_m.y, -14.8, 1.0e-9);
  EXPECT_NEAR(terminal.yaw_rad, 0.0, 1.0e-9);

  const auto global = hierarchical::PlanGroundGlobalRoute(input);
  ASSERT_TRUE(global.ok()) << global.reason_code;
  const auto frontiers = hierarchical::BuildLocalFrontiers(input, *global.route);
  ASSERT_TRUE(frontiers.ok()) << frontiers.reason_code;
  ASSERT_FALSE(frontiers.problems.empty());
  bool solved_frontier = false;
  std::string frontier_reasons;
  for (const auto& problem : frontiers.problems) {
    const auto local_snapshot =
        shared::MapSnapshot::Create(problem.local_map);
    ASSERT_TRUE(local_snapshot.ok()) << local_snapshot.reason_code;
    const auto local_projection = shared::BuildSafeProjection(
        local_snapshot.snapshot, problem.capability,
        problem.config.map_safety, {});
    ASSERT_TRUE(local_projection.ok()) << local_projection.reason_code;
    const auto local_search = wheel::SearchWheelLattice(
        std::get<WheeledState>(problem.current_state), problem.goal_odom,
        *local_projection.projection,
        std::get<WheeledCapability>(problem.capability), problem.config, {});
    solved_frontier = solved_frontier || local_search.ok();
    const auto& frontier = std::get<PointGoal>(problem.goal_odom.target);
    frontier_reasons += local_search.reason_code + " goal=(" +
        std::to_string(frontier.position_m.x) + "," +
        std::to_string(frontier.position_m.y) + ") yaw=" +
        (problem.goal_odom.yaw_rad.has_value()
             ? std::to_string(*problem.goal_odom.yaw_rad)
             : std::string{"none"}) + " tol=" +
        std::to_string(frontier.tolerance_m) + ";";
  }
  EXPECT_TRUE(solved_frontier) << frontier_reasons;

  const PlannerOutput output = Planner{}.Plan(input);
  ASSERT_EQ(output.outcome, PlanningOutcome::kNewReferenceAvailable)
      << output.reason_code;
  EXPECT_EQ(
      LocalDiagnostics(output).trajectory_mode,
      TrajectoryMode::kDiscreteFallback);
  for (const auto& point : WheelTrajectory(output).points) {
    const double yaw = Yaw(point.pose.orientation);
    const double lateral_velocity =
        -std::sin(yaw) * point.velocity.linear_mps.x +
        std::cos(yaw) * point.velocity.linear_mps.y;
    EXPECT_NEAR(lateral_velocity, 0.0, 1.0e-8);
  }
}

TEST(WheelPlanner, LazySearchAcceptsAtLeastOneHierarchicalFrontier) {
  const auto input = MakeWheelInputWithRequiredLocalCoverage();
  const auto global = hierarchical::PlanGroundGlobalRoute(input);
  ASSERT_TRUE(global.ok()) << global.reason_code;
  const auto frontiers = hierarchical::BuildLocalFrontiers(
      input, *global.route);
  ASSERT_TRUE(frontiers.ok()) << frontiers.reason_code;
  ASSERT_FALSE(frontiers.problems.empty());
  bool solved = false;
  std::string reasons;
  for (const auto& problem : frontiers.problems) {
    const auto snapshot = shared::MapSnapshot::Create(problem.local_map);
    ASSERT_TRUE(snapshot.ok()) << snapshot.reason_code;
    const auto projection = shared::BuildSafeProjection(
        snapshot.snapshot, problem.capability, problem.config.map_safety, {});
    ASSERT_TRUE(projection.ok()) << projection.reason_code;
    const auto result = wheel::SearchWheelLattice(
        std::get<WheeledState>(problem.current_state), problem.goal_odom,
        *projection.projection,
        std::get<WheeledCapability>(problem.capability), problem.config, {});
    solved = solved || result.ok();
    const auto& point = std::get<PointGoal>(problem.goal_odom.target);
    const auto& state = std::get<WheeledState>(problem.current_state);
    reasons += result.reason_code + "@(" +
        std::to_string(point.position_m.x) + "," +
        std::to_string(point.position_m.y) + ") yaw=" +
        (problem.goal_odom.yaw_rad.has_value()
             ? std::to_string(*problem.goal_odom.yaw_rad)
             : std::string{"none"}) + " start=(" +
        std::to_string(state.pose.position_m.x) + "," +
        std::to_string(state.pose.position_m.y) + ") tol=" +
        std::to_string(point.tolerance_m) + ";";
  }
  EXPECT_TRUE(solved) << reasons;
}

TEST(WheelPlanner, ExposesStationaryModeAtAnAlreadySatisfiedGoal) {
  Planner planner;
  auto input = MakeWheelInputWithRequiredLocalCoverage();
  input.request_id = "wheel-stationary";
  input.goal_map.target = PointGoal{
      .position_m = std::get<WheeledState>(input.current_state).pose.position_m,
      .tolerance_m = 0.05,
  };

  const PlannerOutput output = planner.Plan(input);

  ASSERT_EQ(output.outcome, PlanningOutcome::kNewReferenceAvailable)
      << output.reason_code;
  EXPECT_EQ(LocalDiagnostics(output).trajectory_mode,
            TrajectoryMode::kStationary);
  EXPECT_EQ(LocalDiagnostics(output).collision_validation,
            CollisionValidation::kNotApplicable);
}

TEST(WheelPlanner, FallsBackOrFailsAccordingToRequiredSmoothingPolicy) {
  Planner planner;
  auto fallback = MakeWheelInputWithRequiredLocalCoverage();
  fallback.request_id = "wheel-forced-fallback";
  fallback.config.optimization.maximum_iterations = 0U;

  const PlannerOutput allowed = planner.Plan(fallback);

  ASSERT_EQ(allowed.outcome, PlanningOutcome::kNewReferenceAvailable)
      << allowed.reason_code;
  EXPECT_EQ(LocalDiagnostics(allowed).trajectory_mode,
            TrajectoryMode::kDiscreteFallback);
  EXPECT_NE(std::ranges::find(allowed.diagnostics.warning_codes,
                              "WHEEL_OPTIMIZATION_CONFIG_INVALID"),
            allowed.diagnostics.warning_codes.end());
  EXPECT_NE(std::ranges::find(
                allowed.diagnostics.warning_codes,
                "WHEEL_OPTIMIZATION_DISCRETE_FALLBACK"),
            allowed.diagnostics.warning_codes.end());

  auto required = fallback;
  required.request_id = "wheel-required-smoothing";
  required.config.optimization.require_smoothed_execution = true;
  const PlannerOutput rejected = planner.Plan(required);
  EXPECT_EQ(rejected.outcome, PlanningOutcome::kNoKnownSafeRoute);
  EXPECT_EQ(rejected.directive, ExecutionDirective::kNoSafeReference);
  EXPECT_EQ(rejected.reason_code, "WHEEL_SMOOTHED_EXECUTION_REQUIRED");
  EXPECT_FALSE(rejected.reference.has_value());
}

TEST(WheelPlanner, CapsEmittedSamplesEvenWhenOptimizationFallsBack) {
  Planner planner;
  auto input = MakeWheelInputWithRequiredLocalCoverage();
  input.request_id = "wheel-nine-sample-budget";
  input.config.optimization.maximum_iterations = 0U;
  input.config.optimization.maximum_smoothing_samples = 9U;

  const PlannerOutput output = planner.Plan(input);

  ASSERT_EQ(output.outcome, PlanningOutcome::kNewReferenceAvailable)
      << output.reason_code;
  EXPECT_LE(WheelTrajectory(output).points.size(), 9U);
  EXPECT_EQ(LocalDiagnostics(output).trajectory_mode,
            TrajectoryMode::kDiscreteFallback);
}

TEST(WheelPlanner, PreservesTheTrueOffCenterStartPoseInTrajectoryAndPreview) {
  Planner planner;
  auto input = MakeWheelInputWithRequiredLocalCoverage();
  input.request_id = "wheel-true-start";
  auto& state = std::get<WheeledState>(input.current_state);
  state.pose.position_m = {2.2, 3.2, 0.0};
  state.pose.orientation = test::YawQuaternion(0.12);

  const PlannerOutput output = planner.Plan(input);

  ASSERT_EQ(output.outcome, PlanningOutcome::kNewReferenceAvailable)
      << output.reason_code;
  const TrajectoryReference& trajectory = WheelTrajectory(output);
  ASSERT_FALSE(trajectory.points.empty());
  EXPECT_NEAR(trajectory.points.front().pose.position_m.x, 2.2, 1.0e-9);
  EXPECT_NEAR(trajectory.points.front().pose.position_m.y, 3.2, 1.0e-9);
  EXPECT_NEAR(Yaw(trajectory.points.front().pose.orientation), 0.12, 1.0e-9);
  EXPECT_NEAR(trajectory.points.back().pose.position_m.x, 4.5, 1.0e-9);
  EXPECT_NEAR(trajectory.points.back().pose.position_m.y, 3.5, 1.0e-9);
  ASSERT_TRUE(output.reference.has_value());
  ASSERT_FALSE(output.reference->preview.poses_map.empty());
  EXPECT_NEAR(output.reference->preview.poses_map.front().position_m.x, 2.2,
              1.0e-9);
  EXPECT_NEAR(output.reference->preview.poses_map.front().position_m.y, 3.2,
              1.0e-9);
  EXPECT_NEAR(Yaw(output.reference->preview.poses_map.front().orientation),
              0.12, 1.0e-9);
}

TEST(WheelPlanner, TimingUsesInitialVelocityTerrainAndDirectionStops) {
  auto input = test::MakeValidWheelInput();
  const auto capability = std::get<WheeledCapability>(input.capability);
  const wheel::WheelTransition flat{
      .source_pose = wheel::WheelPose{
          .position_m = {2.5, 3.5, 0.0}, .yaw_rad = 0.0},
      .target_pose = wheel::WheelPose{
          .position_m = {3.5, 3.5, 0.0}, .yaw_rad = 0.0},
      .primitive_kind = WheelPrimitiveKind::kForward,
      .source_mode = wheel::WheelMotionMode::kStart,
      .target_mode = wheel::WheelMotionMode::kForward,
      .path_length_m = 1.0,
  };
  auto rough = flat;
  rough.source_pose.position_m.x = 3.5;
  rough.target_pose.position_m.x = 4.5;
  rough.surface_slope_rad = 0.2;
  rough.roughness_m = 0.1595;
  const wheel::WheelTransition stop{
      .source_pose = rough.target_pose,
      .target_pose = rough.target_pose,
      .primitive_kind = WheelPrimitiveKind::kStopAndSwitch,
      .source_mode = wheel::WheelMotionMode::kForward,
      .target_mode = wheel::WheelMotionMode::kStart,
  };
  wheel::WheelTransition reverse = flat;
  reverse.source_pose = stop.target_pose;
  reverse.target_pose.position_m = {3.5, 3.5, 0.0};
  reverse.primitive_kind = WheelPrimitiveKind::kReverse;
  reverse.source_mode = wheel::WheelMotionMode::kStart;
  reverse.target_mode = wheel::WheelMotionMode::kReverse;
  reverse.reverse = true;
  const Twist3 initial_velocity{.linear_mps = {.x = 0.35}};

  const auto flat_result = wheel::ParameterizeWheelTiming(
      {flat}, capability, initial_velocity, 64U, {});
  const auto terrain_result = wheel::ParameterizeWheelTiming(
      {flat, rough, stop, reverse}, capability, initial_velocity, 128U, {});

  ASSERT_TRUE(flat_result.ok()) << flat_result.reason_code;
  ASSERT_TRUE(terrain_result.ok()) << terrain_result.reason_code;
  EXPECT_EQ(terrain_result.trajectory->points.front().velocity,
            initial_velocity);
  EXPECT_GT(terrain_result.trajectory->points.back().time_from_start,
            flat_result.trajectory->points.back().time_from_start);
  EXPECT_TRUE(std::ranges::any_of(
      terrain_result.trajectory->points,
      [&](const TrajectoryPoint& point) {
        return std::abs(point.pose.position_m.x -
                            stop.target_pose.position_m.x) < 1.0e-9 &&
            std::abs(point.pose.position_m.y -
                         stop.target_pose.position_m.y) < 1.0e-9 &&
            std::hypot(point.velocity.linear_mps.x,
                       point.velocity.linear_mps.y) < 1.0e-9;
      }));
}

TEST(WheelPlanner, WheelV1TimingCapsTheTightestArcSpeed) {
  auto capability =
      std::get<WheeledCapability>(test::MakeValidWheelInput().capability);
  capability.maximum_forward_speed_mps = 0.2;
  capability.maximum_reverse_speed_mps = 0.2;
  capability.maximum_spin_rate_radps = 0.389923188554511;
  capability.maximum_acceleration_mps2 = 0.2;
  capability.maximum_braking_deceleration_mps2 = 0.2;
  capability.maximum_yaw_acceleration_radps2 = 0.389923188554511;
  capability.maximum_lateral_acceleration_mps2 = 0.2;
  capability.maximum_curvature_per_m = 5.0;
  const wheel::WheelTransition tight_arc{
      .source_pose = wheel::WheelPose{
          .position_m = {0.0, 0.0, 0.0}, .yaw_rad = 0.0},
      .target_pose = wheel::WheelPose{
          .position_m = {0.019603428065912, 0.000963054665561, 0.0},
          .yaw_rad = std::numbers::pi / 32.0},
      .curvature_per_m = 5.0,
      .primitive_kind = WheelPrimitiveKind::kForwardArc,
      .source_mode = wheel::WheelMotionMode::kStart,
      .target_mode = wheel::WheelMotionMode::kForward,
      .path_length_m = std::numbers::pi / 160.0,
  };

  const auto result = wheel::ParameterizeWheelTiming(
      {tight_arc}, capability, Twist3{}, 64U, {});

  ASSERT_TRUE(result.ok()) << result.reason_code;
  double observed_peak_speed = 0.0;
  for (const auto& point : result.trajectory->points) {
    const double transition_speed =
        std::abs(point.velocity.angular_radps.z) / 5.0;
    observed_peak_speed = std::max(observed_peak_speed, transition_speed);
    EXPECT_LE(transition_speed, 0.077984637710902 + 1.0e-12);
  }
  EXPECT_GT(observed_peak_speed, 0.0);
  EXPECT_LE(observed_peak_speed, 0.077984637710902 + 1.0e-12);
}

TEST(WheelPlanner, SelectsReverseMotionForGoalBehind) {
  Planner planner;
  auto input = MakeWheelInputWithRequiredLocalCoverage();
  input.world.local_map.origin_m.x = -2.0;
  input.request_id = "wheel-reverse";
  input.goal_map.target = PointGoal{
      .position_m = {0.5, 3.5, 0.0},
      .tolerance_m = 0.2,
  };

  const PlannerOutput output = planner.Plan(input);

  ASSERT_EQ(output.outcome, PlanningOutcome::kNewReferenceAvailable)
      << output.reason_code;
  const TrajectoryReference& trajectory = WheelTrajectory(output);
  EXPECT_NEAR(trajectory.points.back().pose.position_m.x, 1.5, 0.25);
  ASSERT_FALSE(output.reference->preview.poses_map.empty());
  EXPECT_NEAR(
      output.reference->preview.poses_map.back().position_m.x, 0.5, 0.25);
  EXPECT_TRUE(std::ranges::any_of(
      trajectory.points, [](const TrajectoryPoint& point) {
        return point.velocity.linear_mps.x < -1.0e-3;
      }));
}

TEST(WheelPlanner, ProducesInPlaceSpinForYawOnlyGoal) {
  Planner planner;
  auto input = MakeWheelInputWithRequiredLocalCoverage();
  input.request_id = "wheel-spin";
  input.goal_map.target = PointGoal{
      .position_m = {2.5, 3.5, 0.0},
      .tolerance_m = 0.1,
  };
  input.goal_map.yaw_rad = std::numbers::pi / 2.0;
  input.goal_map.yaw_tolerance_rad = 0.05;

  const PlannerOutput output = planner.Plan(input);

  ASSERT_EQ(output.outcome, PlanningOutcome::kNewReferenceAvailable)
      << output.reason_code;
  const TrajectoryReference& trajectory = WheelTrajectory(output);
  ASSERT_GT(trajectory.points.size(), 2U);
  for (const TrajectoryPoint& point : trajectory.points) {
    EXPECT_NEAR(point.pose.position_m.x, 2.5, 1.0e-9);
    EXPECT_NEAR(point.pose.position_m.y, 3.5, 1.0e-9);
  }
  EXPECT_NEAR(
      Yaw(trajectory.points.back().pose.orientation),
      std::numbers::pi / 2.0, 0.05);
}

TEST(WheelPlanner, BuildsABoundedClampedCurveForAForwardArc) {
  const wheel::WheelTransition control{
      .source_pose = wheel::WheelPose{
          .position_m = {2.5, 3.5, 0.0}, .yaw_rad = 0.0},
      .target_pose = wheel::WheelPose{
          .position_m = {3.5, 4.5, 0.0},
          .yaw_rad = std::numbers::pi / 2.0},
      .curvature_per_m = 1.0,
      .primitive_kind = WheelPrimitiveKind::kForwardArc,
      .source_mode = wheel::WheelMotionMode::kStart,
      .target_mode = wheel::WheelMotionMode::kForward,
      .path_length_m = std::numbers::sqrt2,
  };
  const shared::CorridorResult corridor{
      .status = shared::CorridorStatus::kCertified,
      .fallback = shared::CorridorFallback::kNone,
      .cells = {
          shared::ConvexCorridorCell{
              .half_planes = {
                  shared::HalfPlane2{{1.0, 0.0}, 10.0},
                  shared::HalfPlane2{{-1.0, 0.0}, 0.0},
                  shared::HalfPlane2{{0.0, 1.0}, 10.0},
                  shared::HalfPlane2{{0.0, -1.0}, 0.0},
              },
          },
      },
  };
  const OptimizationConfig config;

  const wheel::WheelOptimizationResult result =
      wheel::OptimizeWheelSpline({control}, corridor, config, {});

  ASSERT_TRUE(result.optimized) << result.reason_code;
  ASSERT_GT(result.transitions.size(), 1U);
  EXPECT_LE(1U + 8U * result.transitions.size(),
            config.maximum_smoothing_samples);
  EXPECT_EQ(result.transitions.front().source_pose, control.source_pose);
  EXPECT_EQ(result.transitions.back().target_pose, control.target_pose);
  EXPECT_TRUE(std::ranges::any_of(
      result.transitions, [](const wheel::WheelTransition& transition) {
        const Vec3 point = transition.target_pose.position_m;
        return std::abs((point.y - 3.5) - (point.x - 2.5)) > 1.0e-4;
      }));
}

TEST(WheelPlanner, IsDeterministicForSameTypedSnapshot) {
  Planner planner;
  const auto input = MakeWheelInputWithRequiredLocalCoverage();

  const PlannerOutput first = planner.Plan(input);
  const PlannerOutput second = planner.Plan(input);

  ASSERT_EQ(first.outcome, PlanningOutcome::kNewReferenceAvailable);
  ASSERT_EQ(second.outcome, first.outcome);
  ASSERT_EQ(second.reason_code, first.reason_code);
  ASSERT_EQ(second.diagnostics.best_cost, first.diagnostics.best_cost);
  ASSERT_EQ(second.reference->plan_id, first.reference->plan_id);
  const auto& first_points = WheelTrajectory(first).points;
  const auto& second_points = WheelTrajectory(second).points;
  ASSERT_EQ(second_points.size(), first_points.size());
  for (std::size_t index = 0U; index < first_points.size(); ++index) {
    EXPECT_EQ(
        second_points[index].time_from_start,
        first_points[index].time_from_start);
    EXPECT_DOUBLE_EQ(
        second_points[index].pose.position_m.x,
        first_points[index].pose.position_m.x);
    EXPECT_DOUBLE_EQ(
        second_points[index].pose.position_m.y,
        first_points[index].pose.position_m.y);
    EXPECT_DOUBLE_EQ(
        second_points[index].pose.orientation.w,
        first_points[index].pose.orientation.w);
    EXPECT_DOUBLE_EQ(
        second_points[index].pose.orientation.z,
        first_points[index].pose.orientation.z);
  }
}

}  // namespace
}  // namespace lunar::pure_planning

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include "legged/legged_terrain.hpp"
#include "legged/legged_lattice.hpp"
#include "legged/legged_planner.hpp"
#include "legged/legged_spline_optimizer.hpp"
#include "legged/legged_timing.hpp"
#include "lunar_planner_core/planner.hpp"
#include "shared/map_snapshot.hpp"
#include "shared/safe_projection.hpp"
#include "shared/ara_star.hpp"
#include "test_fixtures.hpp"

namespace lunar::planning {
namespace {

double Yaw(const Quaternion& orientation) {
  return std::atan2(
      2.0 * (orientation.w * orientation.z +
             orientation.x * orientation.y),
      1.0 - 2.0 * (orientation.y * orientation.y +
                   orientation.z * orientation.z));
}

const TrajectoryReference& LeggedTrajectory(const PlannerOutput& output) {
  EXPECT_TRUE(output.reference.has_value());
  EXPECT_EQ(output.reference->platform_type, PlatformType::kLegged);
  const auto* trajectory =
      std::get_if<TrajectoryReference>(&output.reference->data);
  EXPECT_NE(trajectory, nullptr);
  return *trajectory;
}

const LocalTrajectoryDiagnostics& LeggedDiagnostics(
    const PlannerOutput& output) {
  EXPECT_TRUE(output.diagnostics.local_trajectory.has_value());
  return *output.diagnostics.local_trajectory;
}

hierarchical::LocalPlanningProblem MakeLocalLeggedProblem(
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

hierarchical::LocalSearchDomain FullDomain(const GridMap& map) {
  return hierarchical::LocalSearchDomain{
      map.width, map.height,
      std::vector<std::uint8_t>(map.CellCount(), 1U)};
}

void AllowCell(std::vector<std::uint8_t>& allowed, const GridMap& map,
               const std::size_t x, const std::size_t y) {
  allowed.at(y * map.width + x) = 1U;
}

PlannerInput MakeLeggedInputWithRequiredLocalCoverage() {
  PlannerInput input = test::MakeValidLeggedInput();
  input.world.local_map = test::MakeFlatMap("odom", 10U, 10U, 1.0);
  input.world.local_map.origin_m = {-1.0, -1.0, 0.0};
  return input;
}

TEST(LeggedPlanner, ProducesOnlyBodyReferenceWithBoundedKinematics) {
  Planner planner;
  const auto input = MakeLeggedInputWithRequiredLocalCoverage();
  const auto capability = std::get<LeggedCapability>(input.capability);

  const PlannerOutput output = planner.Plan(input);

  ASSERT_EQ(output.outcome, PlanningOutcome::kNewReferenceAvailable)
      << output.reason_code;
  EXPECT_EQ(output.directive, ExecutionDirective::kActivateNewReference);
  EXPECT_EQ(output.diagnostics.planner_name, "cpp_v3_hierarchical");
  ASSERT_TRUE(output.diagnostics.best_cost.has_value());
  const TrajectoryReference& trajectory = LeggedTrajectory(output);
  ASSERT_GT(trajectory.points.size(), 2U);
  EXPECT_LE(trajectory.points.size(),
            input.config.optimization.maximum_smoothing_samples);
  EXPECT_EQ(
      trajectory.semantics,
      TrajectorySemantics::kLeggedBodyReference);
  EXPECT_EQ(LeggedDiagnostics(output).trajectory_mode,
            TrajectoryMode::kOptimized);
  EXPECT_EQ(LeggedDiagnostics(output).collision_validation,
            CollisionValidation::kCertified);
  EXPECT_NEAR(trajectory.points.front().pose.position_m.x, 2.5, 1.0e-9);
  EXPECT_NEAR(trajectory.points.back().pose.position_m.x, 4.5, 0.25);
  for (std::size_t index = 0U; index < trajectory.points.size(); ++index) {
    const TrajectoryPoint& point = trajectory.points[index];
    EXPECT_GE(point.pose.position_m.z, capability.body_height_m.lower - 1.0e-9);
    EXPECT_LE(point.pose.position_m.z, capability.body_height_m.upper + 1.0e-9);
    EXPECT_GE(
        point.velocity.linear_mps.x,
        capability.forward_speed_mps.lower - 1.0e-9);
    EXPECT_LE(
        point.velocity.linear_mps.x,
        capability.forward_speed_mps.upper + 1.0e-9);
    EXPECT_GE(
        point.velocity.linear_mps.y,
        capability.lateral_speed_mps.lower - 1.0e-9);
    EXPECT_LE(
        point.velocity.linear_mps.y,
        capability.lateral_speed_mps.upper + 1.0e-9);
    EXPECT_GE(
        point.velocity.angular_radps.z,
        capability.yaw_rate_radps.lower - 1.0e-9);
    EXPECT_LE(
        point.velocity.angular_radps.z,
        capability.yaw_rate_radps.upper + 1.0e-9);
    if (index > 0U) {
      EXPECT_GT(
          point.time_from_start,
          trajectory.points[index - 1U].time_from_start);
      const double delta_seconds = std::chrono::duration<double>(
          point.time_from_start -
          trajectory.points[index - 1U].time_from_start).count();
      const Vec3 delta_velocity{
          .x = point.velocity.linear_mps.x -
              trajectory.points[index - 1U].velocity.linear_mps.x,
          .y = point.velocity.linear_mps.y -
              trajectory.points[index - 1U].velocity.linear_mps.y,
          .z = point.velocity.linear_mps.z -
              trajectory.points[index - 1U].velocity.linear_mps.z,
      };
      EXPECT_LE(
          std::hypot(
              std::hypot(delta_velocity.x, delta_velocity.y),
              delta_velocity.z) /
              delta_seconds,
          capability.maximum_linear_acceleration_mps2 + 1.0e-9);
      EXPECT_LE(
          std::abs(
              point.velocity.angular_radps.z -
              trajectory.points[index - 1U].velocity.angular_radps.z) /
              delta_seconds,
          capability.maximum_yaw_acceleration_radps2 + 1.0e-9);
    }
  }
}

TEST(LeggedPlanner, DerivesLongSweepSamplingWithoutAFixedCeiling) {
  auto input = test::MakeValidLeggedInput();
  input.world.local_map = test::MakeFlatMap("odom", 120U, 100U, 0.05);
  const auto capability = std::get<LeggedCapability>(input.capability);
  const auto snapshot = shared::MapSnapshot::Create(input.world.local_map);
  ASSERT_TRUE(snapshot.ok()) << snapshot.reason_code;
  const auto projection = shared::BuildSafeProjection(
      snapshot.snapshot, input.capability, input.config.map_safety, {});
  ASSERT_TRUE(projection.ok()) << projection.reason_code;

  const legged::LeggedTerrainGrid terrain_grid{
      *projection.projection, capability, {}};
  ASSERT_TRUE(terrain_grid.ok());
  const legged::LeggedSweepResult result = legged::ValidateLeggedBodySweep(
      legged::LeggedPose{.position_m = {1.0, 2.5, 0.5}},
      legged::LeggedPose{.position_m = {4.0, 2.5, 0.5}},
      Interval{.lower = 0.5, .upper = 0.5},
      *projection.projection, capability, FullDomain(input.world.local_map),
      {}, &terrain_grid);

  EXPECT_TRUE(result.valid) << result.reason_code;
  EXPECT_GT(result.sample_count, 32U);
  EXPECT_EQ(terrain_grid.exact_rectangle_test_count(), 0U);
}

TEST(LeggedPlanner, PhysicallyBlockedBodySupportGoalWinsOverDomainMask) {
  auto input = test::MakeValidLeggedInput();
  input.request_id = "legged-physical-support-goal";
  std::get<LeggedState>(input.current_state).body_pose.position_m.y = 3.1;
  std::get<PointGoal>(input.goal_map.target).position_m.y = 3.1;
  auto& obstacles = std::get<std::vector<std::uint8_t>>(
      input.world.local_map.layers.at("obstacle").values);
  obstacles.at(2U * input.world.local_map.width + 4U) = 1U;
  auto allowed = std::vector<std::uint8_t>(
      input.world.local_map.CellCount(), 1U);
  allowed.at(3U * input.world.local_map.width + 4U) = 0U;
  const auto problem = MakeLocalLeggedProblem(input, std::move(allowed));

  const PlannerOutput output = legged::LeggedPlanner{}.Plan(problem);

  EXPECT_EQ(output.outcome, PlanningOutcome::kGoalInfeasible);
  EXPECT_EQ(output.directive, ExecutionDirective::kHoldPosition);
  EXPECT_EQ(output.reason_code, "LEGGED_GOAL_INFEASIBLE");
  EXPECT_EQ(output.diagnostics.expanded_states, 0U);
  EXPECT_FALSE(output.reference.has_value());
}

TEST(LeggedPlanner, PhysicallySafeGoalReportsLocalSearchDomainExhaustion) {
  auto input = test::MakeValidLeggedInput();
  input.request_id = "legged-domain-exhausted";
  auto allowed = EmptyDomain(input.world.local_map);
  AllowCell(allowed, input.world.local_map, 2U, 3U);
  const auto problem = MakeLocalLeggedProblem(input, std::move(allowed));

  const PlannerOutput output = legged::LeggedPlanner{}.Plan(problem);

  EXPECT_EQ(output.outcome, PlanningOutcome::kNoKnownSafeRoute);
  EXPECT_EQ(output.directive, ExecutionDirective::kNoSafeReference);
  EXPECT_EQ(output.reason_code, "LOCAL_SEARCH_DOMAIN_EXHAUSTED");
  EXPECT_FALSE(output.reference.has_value());
}

TEST(LeggedPlanner, HonorsRequiredSmoothingPolicy) {
  Planner planner;
  auto fallback = MakeLeggedInputWithRequiredLocalCoverage();
  fallback.request_id = "legged-forced-fallback";
  fallback.config.optimization.maximum_iterations = 0U;

  const PlannerOutput allowed = planner.Plan(fallback);

  ASSERT_EQ(allowed.outcome, PlanningOutcome::kNewReferenceAvailable)
      << allowed.reason_code;
  EXPECT_EQ(LeggedDiagnostics(allowed).trajectory_mode,
            TrajectoryMode::kDiscreteFallback);
  EXPECT_NE(std::ranges::find(
                allowed.diagnostics.warning_codes,
                "LEGGED_OPTIMIZATION_DISCRETE_FALLBACK"),
            allowed.diagnostics.warning_codes.end());

  auto required = fallback;
  required.request_id = "legged-required-smoothing";
  required.config.optimization.require_smoothed_execution = true;
  const PlannerOutput rejected = planner.Plan(required);
  EXPECT_EQ(rejected.outcome, PlanningOutcome::kNoKnownSafeRoute);
  EXPECT_EQ(rejected.reason_code, "LEGGED_SMOOTHED_EXECUTION_REQUIRED");
  EXPECT_FALSE(rejected.reference.has_value());
}

TEST(LeggedPlanner, PreservesTheTrueOffCenterBodyPoseAndHeight) {
  Planner planner;
  auto input = MakeLeggedInputWithRequiredLocalCoverage();
  input.request_id = "legged-true-start";
  auto& state = std::get<LeggedState>(input.current_state);
  state.body_pose.position_m = {2.2, 3.2, 0.47};
  state.body_pose.orientation = test::YawQuaternion(0.12);

  const PlannerOutput output = planner.Plan(input);

  ASSERT_EQ(output.outcome, PlanningOutcome::kNewReferenceAvailable)
      << output.reason_code;
  const TrajectoryReference& trajectory = LeggedTrajectory(output);
  ASSERT_FALSE(trajectory.points.empty());
  const Pose3& start = trajectory.points.front().pose;
  EXPECT_NEAR(start.position_m.x, 2.2, 1.0e-9);
  EXPECT_NEAR(start.position_m.y, 3.2, 1.0e-9);
  EXPECT_NEAR(start.position_m.z, 0.47, 1.0e-9);
  EXPECT_NEAR(Yaw(start.orientation), 0.12, 1.0e-9);
  ASSERT_TRUE(output.reference.has_value());
  ASSERT_FALSE(output.reference->preview.poses_map.empty());
  const Pose3& preview_start = output.reference->preview.poses_map.front();
  EXPECT_NEAR(preview_start.position_m.x, 2.2, 1.0e-9);
  EXPECT_NEAR(preview_start.position_m.y, 3.2, 1.0e-9);
  EXPECT_NEAR(preview_start.position_m.z, 0.47, 1.0e-9);
  EXPECT_NEAR(Yaw(preview_start.orientation), 0.12, 1.0e-9);
}

TEST(LeggedPlanner, SupportsLateralBodyPrimitive) {
  Planner planner;
  auto input = MakeLeggedInputWithRequiredLocalCoverage();
  input.request_id = "legged-lateral";
  input.goal_map.target = PointGoal{
      .position_m = {2.5, 5.5, 0.0},
      .tolerance_m = 0.2,
  };

  const PlannerOutput output = planner.Plan(input);

  ASSERT_EQ(output.outcome, PlanningOutcome::kNewReferenceAvailable)
      << output.reason_code;
  const TrajectoryReference& trajectory = LeggedTrajectory(output);
  EXPECT_NEAR(trajectory.points.back().pose.position_m.y, 5.5, 0.25);
  EXPECT_TRUE(std::ranges::any_of(
      trajectory.points, [](const TrajectoryPoint& point) {
        return point.velocity.linear_mps.y > 1.0e-3;
      }));
}

TEST(LeggedPlanner, SmoothsLateralPositionHeightAndYawIndependently) {
  const legged::LeggedTransition control{
      .source_pose = legged::LeggedPose{
          .position_m = {2.5, 3.5, 0.45}, .yaw_rad = 0.0},
      .target_pose = legged::LeggedPose{
          .position_m = {2.5, 4.5, 0.55}, .yaw_rad = 0.6},
      .target_body_z_m = {.lower = 0.4, .upper = 0.6},
      .primitive_kind = LeggedPrimitiveKind::kLateralLeft,
      .path_length_m = std::hypot(1.0, 0.1),
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

  const legged::LeggedOptimizationResult result =
      legged::OptimizeLeggedBodySpline({control}, corridor, config, {});

  ASSERT_TRUE(result.optimized) << result.reason_code;
  ASSERT_GT(result.transitions.size(), 1U);
  EXPECT_LE(1U + 8U * result.transitions.size(),
            config.maximum_smoothing_samples);
  EXPECT_EQ(result.transitions.front().source_pose, control.source_pose);
  EXPECT_EQ(result.transitions.back().target_pose, control.target_pose);
  EXPECT_TRUE(std::ranges::all_of(
      result.transitions, [](const legged::LeggedTransition& transition) {
        return std::abs(transition.source_pose.position_m.x - 2.5) < 1.0e-9 &&
            std::abs(transition.target_pose.position_m.x - 2.5) < 1.0e-9 &&
            transition.source_pose.yaw_rad < 1.0 &&
            transition.target_pose.yaw_rad < 1.0;
      }));
}

TEST(LeggedPlanner, UsesStepVerticalRateAsTimingLowerBound) {
  const auto input = test::MakeValidLeggedInput();
  const auto capability = std::get<LeggedCapability>(input.capability);
  const legged::LeggedTransition transition{
      .source_pose = {{1.0, 1.0, 0.5}, 0.0},
      .target_pose = {{1.2, 1.0, 1.0}, 0.0},
      .target_body_z_m = {.lower = 0.9, .upper = 1.1},
      .primitive_kind = LeggedPrimitiveKind::kForward,
      .path_length_m = std::hypot(0.2, 0.5),
  };

  const auto timed = legged::ParameterizeLeggedBodyTiming(
      {transition}, capability, 64U, {});

  ASSERT_TRUE(timed.ok()) << timed.reason_code;
  ASSERT_FALSE(timed.trajectory->points.empty());
  EXPECT_GE(timed.trajectory->points.back().time_from_start,
            std::chrono::seconds{5});
}

TEST(LeggedPlanner, ReachesExactOffCellGoalAndTaskYaw) {
  Planner planner;
  auto input = MakeLeggedInputWithRequiredLocalCoverage();
  input.request_id = "legged-exact-goal";
  input.goal_map.target = PointGoal{
      .position_m = {4.37, 3.63, 0.0},
      .tolerance_m = 0.01,
  };
  input.goal_map.yaw_rad = 0.23;
  input.goal_map.yaw_tolerance_rad = 0.01;

  const PlannerOutput output = planner.Plan(input);

  ASSERT_EQ(output.outcome, PlanningOutcome::kNewReferenceAvailable)
      << output.reason_code;
  const Pose3& endpoint = LeggedTrajectory(output).points.back().pose;
  EXPECT_NEAR(endpoint.position_m.x, 4.37, 1.0e-9);
  EXPECT_NEAR(endpoint.position_m.y, 3.63, 1.0e-9);
  EXPECT_NEAR(Yaw(endpoint.orientation), 0.23, 1.0e-9);
  EXPECT_NEAR(LeggedDiagnostics(output).endpoint_error_m, 0.0, 1.0e-9);
}

TEST(LeggedPlanner, KeepsTolerantBoundaryGoalInsideItsCertifiedTerrainCell) {
  auto input = test::MakeValidLeggedInput();
  input.request_id = "legged-boundary-goal";
  std::get<LeggedState>(input.current_state).body_pose.position_m.x = 2.2;
  input.config.optimization.maximum_iterations = 0U;
  input.goal_map.target = PointGoal{
      .position_m = {3.0, 3.0, 0.0},
      .tolerance_m = 0.8,
  };

  const auto snapshot = shared::MapSnapshot::Create(input.world.local_map);
  ASSERT_TRUE(snapshot.ok()) << snapshot.reason_code;
  const PointGoal& goal = std::get<PointGoal>(input.goal_map.target);
  const PlannerOutput output = legged::LeggedPlanner{}.Plan(
      hierarchical::LocalPlanningProblem{
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
              std::vector<std::uint8_t>(input.world.local_map.CellCount(), 1U)},
          .capability = input.capability,
          .config = input.config,
      });

  ASSERT_EQ(output.outcome, PlanningOutcome::kNewReferenceAvailable)
      << output.reason_code;
  const Pose3& endpoint = LeggedTrajectory(output).points.back().pose;
  const auto endpoint_cell = snapshot.snapshot->PositionToCell(
      Vec2{.x = endpoint.position_m.x, .y = endpoint.position_m.y});
  ASSERT_TRUE(endpoint_cell.has_value());
  const Vec3 certified_center = snapshot.snapshot->CellCenter(*endpoint_cell);
  EXPECT_NEAR(endpoint.position_m.x, certified_center.x, 1.0e-9);
  EXPECT_NEAR(endpoint.position_m.y, certified_center.y, 1.0e-9);
  EXPECT_LE(
      std::hypot(endpoint.position_m.x - goal.position_m.x,
                 endpoint.position_m.y - goal.position_m.y),
      goal.tolerance_m + 1.0e-9);
}

TEST(LeggedPlanner, IsDeterministicForSameTypedSnapshot) {
  Planner planner;
  const auto input = MakeLeggedInputWithRequiredLocalCoverage();

  const PlannerOutput first = planner.Plan(input);
  const PlannerOutput second = planner.Plan(input);

  ASSERT_EQ(first.outcome, PlanningOutcome::kNewReferenceAvailable);
  ASSERT_EQ(second.outcome, first.outcome);
  ASSERT_EQ(second.reason_code, first.reason_code);
  ASSERT_EQ(second.diagnostics.best_cost, first.diagnostics.best_cost);
  const auto& first_points = LeggedTrajectory(first).points;
  const auto& second_points = LeggedTrajectory(second).points;
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
        second_points[index].pose.position_m.z,
        first_points[index].pose.position_m.z);
  }
}

TEST(LeggedPlanner, LazySearchMatchesCertifiedFullLatticeExactly) {
  const auto input = MakeLeggedInputWithRequiredLocalCoverage();
  const auto snapshot = shared::MapSnapshot::Create(input.world.local_map);
  ASSERT_TRUE(snapshot.ok()) << snapshot.reason_code;
  const auto projection = shared::BuildSafeProjection(
      snapshot.snapshot, input.capability, input.config.map_safety, {});
  ASSERT_TRUE(projection.ok()) << projection.reason_code;
  const auto domain = FullDomain(input.world.local_map);
  const auto& state = std::get<LeggedState>(input.current_state);
  const auto& capability = std::get<LeggedCapability>(input.capability);
  const auto full = legged::BuildLeggedLattice(
      state, input.goal_map, *projection.projection, domain, capability,
      input.config, {});
  ASSERT_TRUE(full.ok()) << full.reason_code;
  const auto full_search = shared::SearchAraStar(
      full.graph->search_problem, {});
  const auto full_plan = legged::ResolveLeggedPlan(*full.graph, full_search);
  ASSERT_TRUE(full_plan.has_value());

  const auto lazy = legged::SearchLeggedLattice(
      state, input.goal_map, *projection.projection, domain, capability,
      input.config, {});

  ASSERT_TRUE(lazy.ok()) << lazy.reason_code;
  ASSERT_TRUE(lazy.plan.has_value());
  auto lazy_transitions = lazy.plan->transitions;
  auto full_transitions = full_plan->transitions;
  for (auto& transition : lazy_transitions) {
    transition.stable_index = 0U;
  }
  for (auto& transition : full_transitions) {
    transition.stable_index = 0U;
  }
  EXPECT_EQ(lazy_transitions, full_transitions);
  EXPECT_DOUBLE_EQ(lazy.plan->cost, full_plan->cost);
  EXPECT_EQ(lazy.plan->expanded_states, full_plan->expanded_states);
  EXPECT_LT(lazy.plan->expanded_states, full.graph->states.size());
}

TEST(LeggedPlanner, LazySearchPreservesNoPathAndCancellationContracts) {
  auto input = MakeLeggedInputWithRequiredLocalCoverage();
  const auto snapshot = shared::MapSnapshot::Create(input.world.local_map);
  ASSERT_TRUE(snapshot.ok()) << snapshot.reason_code;
  const auto projection = shared::BuildSafeProjection(
      snapshot.snapshot, input.capability, input.config.map_safety, {});
  ASSERT_TRUE(projection.ok()) << projection.reason_code;
  auto allowed = EmptyDomain(input.world.local_map);
  AllowCell(allowed, input.world.local_map, 2U, 3U);
  const auto domain = hierarchical::LocalSearchDomain{
      input.world.local_map.width, input.world.local_map.height, allowed};
  const auto& state = std::get<LeggedState>(input.current_state);
  const auto& capability = std::get<LeggedCapability>(input.capability);

  const auto no_path = legged::SearchLeggedLattice(
      state, input.goal_map, *projection.projection, domain, capability,
      input.config, {});
  EXPECT_EQ(no_path.status, legged::LeggedLatticeStatus::kNoPath);
  EXPECT_EQ(no_path.reason_code, "LOCAL_SEARCH_DOMAIN_EXHAUSTED");

  std::stop_source stop_source;
  stop_source.request_stop();
  const auto canceled = legged::SearchLeggedLattice(
      state, input.goal_map, *projection.projection, domain, capability,
      input.config, stop_source.get_token());
  EXPECT_EQ(canceled.status, legged::LeggedLatticeStatus::kCanceled);
  EXPECT_EQ(canceled.reason_code, "REQUEST_CANCELED");
}

TEST(LeggedPlanner, LazySearchTieBreakIsDeterministicAcrossTwentyRuns) {
  auto input = MakeLeggedInputWithRequiredLocalCoverage();
  std::get<PointGoal>(input.goal_map.target).position_m = {4.5, 4.5, 0.0};
  const auto snapshot = shared::MapSnapshot::Create(input.world.local_map);
  ASSERT_TRUE(snapshot.ok()) << snapshot.reason_code;
  const auto projection = shared::BuildSafeProjection(
      snapshot.snapshot, input.capability, input.config.map_safety, {});
  ASSERT_TRUE(projection.ok()) << projection.reason_code;
  const auto domain = FullDomain(input.world.local_map);
  const auto& state = std::get<LeggedState>(input.current_state);
  const auto& capability = std::get<LeggedCapability>(input.capability);
  std::optional<legged::LeggedDiscretePlan> expected;

  for (std::size_t run = 0U; run < 20U; ++run) {
    const auto result = legged::SearchLeggedLattice(
        state, input.goal_map, *projection.projection, domain, capability,
        input.config, {});
    ASSERT_TRUE(result.ok()) << "run=" << run << ' ' << result.reason_code;
    ASSERT_TRUE(result.plan.has_value());
    if (!expected.has_value()) {
      expected = result.plan;
    } else {
      EXPECT_EQ(result.plan->transitions, expected->transitions);
      EXPECT_DOUBLE_EQ(result.plan->cost, expected->cost);
      EXPECT_EQ(result.plan->expanded_states, expected->expanded_states);
    }
  }
}

}  // namespace
}  // namespace lunar::planning

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <variant>

#include <gtest/gtest.h>

#include "lunar_planner_core/planner.hpp"
#include "test_fixtures.hpp"
#include "wheel/wheel_spline_optimizer.hpp"

namespace lunar::planning {
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

TEST(WheelPlanner, PlansForwardReferenceWithBoundedTiming) {
  Planner planner;
  const auto input = test::MakeValidWheelInput();

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
  EXPECT_EQ(diagnostics.trajectory_mode, TrajectoryMode::kOptimized);
  EXPECT_EQ(diagnostics.collision_validation,
            CollisionValidation::kCertified);
  EXPECT_TRUE(std::isfinite(diagnostics.start_anchor_error_m));
  EXPECT_TRUE(std::isfinite(diagnostics.endpoint_error_m));
  EXPECT_TRUE(std::isfinite(diagnostics.maximum_curvature_per_m));
  EXPECT_TRUE(std::isfinite(diagnostics.smoothing_elapsed_s));
  EXPECT_NEAR(trajectory.points.front().pose.position_m.x, 2.5, 1.0e-9);
  EXPECT_NEAR(trajectory.points.back().pose.position_m.x, 4.5, 0.25);
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

TEST(WheelPlanner, ExposesStationaryModeAtAnAlreadySatisfiedGoal) {
  Planner planner;
  auto input = test::MakeValidWheelInput();
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
  auto fallback = test::MakeValidWheelInput();
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
  auto input = test::MakeValidWheelInput();
  input.request_id = "wheel-nine-sample-budget";
  input.config.optimization.maximum_smoothing_samples = 9U;

  const PlannerOutput output = planner.Plan(input);

  ASSERT_EQ(output.outcome, PlanningOutcome::kNewReferenceAvailable)
      << output.reason_code;
  EXPECT_LE(WheelTrajectory(output).points.size(), 9U);
  EXPECT_EQ(LocalDiagnostics(output).trajectory_mode,
            TrajectoryMode::kDiscreteFallback);
}

TEST(WheelPlanner, RejectsValidationSubdivisionBudgetAboveHardCeiling) {
  Planner planner;
  auto input = test::MakeValidWheelInput();
  input.config.wheel.continuous_validation_maximum_subdivisions = 33U;

  const PlannerOutput output = planner.Plan(input);

  EXPECT_EQ(output.outcome, PlanningOutcome::kInvalidRequest);
  EXPECT_EQ(output.reason_code, "WHEEL_LATTICE_REQUEST_INVALID");
  EXPECT_FALSE(output.reference.has_value());
}

TEST(WheelPlanner, PreservesTheTrueOffCenterStartPoseInTrajectoryAndPreview) {
  Planner planner;
  auto input = test::MakeValidWheelInput();
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
  ASSERT_TRUE(output.reference.has_value());
  ASSERT_FALSE(output.reference->preview.poses_map.empty());
  EXPECT_NEAR(output.reference->preview.poses_map.front().position_m.x, 2.2,
              1.0e-9);
  EXPECT_NEAR(output.reference->preview.poses_map.front().position_m.y, 3.2,
              1.0e-9);
  EXPECT_NEAR(Yaw(output.reference->preview.poses_map.front().orientation),
              0.12, 1.0e-9);
}

TEST(WheelPlanner, SelectsReverseMotionForGoalBehind) {
  Planner planner;
  auto input = test::MakeValidWheelInput();
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
  auto input = test::MakeValidWheelInput();
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
      .nominal_duration = std::chrono::seconds{1},
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
  const auto input = test::MakeValidWheelInput();

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
}  // namespace lunar::planning

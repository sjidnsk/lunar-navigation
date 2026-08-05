#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <variant>

#include <gtest/gtest.h>

#include "lunar_planner_core/planner.hpp"
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

const TrajectoryReference& WheelTrajectory(const PlannerOutput& output) {
  EXPECT_TRUE(output.reference.has_value());
  EXPECT_EQ(output.reference->platform_type, PlatformType::kWheeled);
  const auto* trajectory =
      std::get_if<TrajectoryReference>(&output.reference->data);
  EXPECT_NE(trajectory, nullptr);
  return *trajectory;
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

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <variant>

#include <gtest/gtest.h>

#include "lunar_planner_core/planner.hpp"
#include "test_fixtures.hpp"

namespace lunar::planning {
namespace {

const TrajectoryReference& LeggedTrajectory(const PlannerOutput& output) {
  EXPECT_TRUE(output.reference.has_value());
  EXPECT_EQ(output.reference->platform_type, PlatformType::kLegged);
  const auto* trajectory =
      std::get_if<TrajectoryReference>(&output.reference->data);
  EXPECT_NE(trajectory, nullptr);
  return *trajectory;
}

TEST(LeggedPlanner, ProducesOnlyBodyReferenceWithBoundedKinematics) {
  Planner planner;
  const auto input = test::MakeValidLeggedInput();
  const auto capability = std::get<LeggedCapability>(input.capability);

  const PlannerOutput output = planner.Plan(input);

  ASSERT_EQ(output.outcome, PlanningOutcome::kNewReferenceAvailable)
      << output.reason_code;
  EXPECT_EQ(output.directive, ExecutionDirective::kActivateNewReference);
  EXPECT_EQ(output.diagnostics.planner_name, "cpp_v3_hierarchical");
  ASSERT_TRUE(output.diagnostics.best_cost.has_value());
  const TrajectoryReference& trajectory = LeggedTrajectory(output);
  ASSERT_GT(trajectory.points.size(), 2U);
  EXPECT_EQ(
      trajectory.semantics,
      TrajectorySemantics::kLeggedBodyReference);
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
        point.velocity.linear_mps.z,
        capability.vertical_speed_mps.lower - 1.0e-9);
    EXPECT_LE(
        point.velocity.linear_mps.z,
        capability.vertical_speed_mps.upper + 1.0e-9);
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

TEST(LeggedPlanner, SupportsLateralBodyPrimitive) {
  Planner planner;
  auto input = test::MakeValidLeggedInput();
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

TEST(LeggedPlanner, IsDeterministicForSameTypedSnapshot) {
  Planner planner;
  const auto input = test::MakeValidLeggedInput();

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

}  // namespace
}  // namespace lunar::planning

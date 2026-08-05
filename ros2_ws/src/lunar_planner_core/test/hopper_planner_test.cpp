#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <ranges>
#include <variant>

#include <gtest/gtest.h>

#include "lunar_planner_core/planner.hpp"
#include "test_fixtures.hpp"

namespace lunar::planning {
namespace {

const HopReference &HopperReference(const PlannerOutput &output) {
  EXPECT_TRUE(output.reference.has_value());
  EXPECT_EQ(output.reference->platform_type, PlatformType::kHopper);
  const auto *reference = std::get_if<HopReference>(&output.reference->data);
  EXPECT_NE(reference, nullptr);
  return *reference;
}

double Norm(const Vec3 value) {
  return std::hypot(std::hypot(value.x, value.y), value.z);
}

[[nodiscard]] PlannerInput NarrowGoalLargeLandingRegionInput() {
  PlannerInput input = test::MakeValidHopperInput();
  input.request_id = "narrow-goal-large-landing-region";
  input.world.global_map = test::MakeFlatMap("map", 40U, 40U, 0.2);
  input.world.local_map = test::MakeFlatMap("odom", 40U, 40U, 0.2);
  input.config.global_map.base_resolution_m = 0.2;
  input.current_state = HopperState{
      .pose = Pose3{.position_m = {3.1, 3.1, 0.5}},
  };
  input.goal_map.target = PointGoal{
      .position_m = {4.1, 3.1, 0.0},
      .tolerance_m = 0.05,
  };
  auto &capability = std::get<HopperCapability>(input.capability);
  capability.body_half_extent_m.x = 0.1;
  capability.body_half_extent_m.y = 0.1;
  capability.minimum_landing_region_area_m2 = 1.327322;
  return input;
}

TEST(HopperPlanner, ProducesExactlyOnePhysicallyBoundedHop) {
  Planner planner;
  const PlannerInput input = test::MakeValidHopperInput();
  const auto capability = std::get<HopperCapability>(input.capability);
  const auto state = std::get<HopperState>(input.current_state);

  const PlannerOutput output = planner.Plan(input);

  ASSERT_EQ(output.outcome, PlanningOutcome::kNewReferenceAvailable)
      << output.reason_code;
  EXPECT_EQ(output.directive, ExecutionDirective::kActivateNewReference);
  EXPECT_EQ(output.diagnostics.planner_name, "cpp_v3_hierarchical");
  ASSERT_TRUE(output.diagnostics.best_cost.has_value());
  EXPECT_NEAR(*output.diagnostics.best_cost, 1.385398163, 0.35);
  const HopReference &reference = HopperReference(output);
  ASSERT_EQ(reference.segments.size(), 1U);
  const HopSegment &segment = reference.segments.front();
  EXPECT_FALSE(segment.segment_id.empty());
  EXPECT_EQ(segment.launch_pose, state.pose);
  EXPECT_GE(segment.landing_region_boundary_m.size(), 4U);
  EXPECT_GT(segment.flight_tube_radius_m, 0.0);
  EXPECT_GE(segment.flight_time, capability.minimum_flight_time);
  EXPECT_LE(segment.flight_time, capability.maximum_flight_time);
  EXPECT_LE(Norm(segment.launch_velocity_mps),
            capability.maximum_launch_speed_mps + 1.0e-9);
  const Vec3 impulse_delta{
      .x = segment.launch_velocity_mps.x - state.velocity.linear_mps.x,
      .y = segment.launch_velocity_mps.y - state.velocity.linear_mps.y,
      .z = segment.launch_velocity_mps.z - state.velocity.linear_mps.z,
  };
  EXPECT_LE(capability.platform_mass_kg * Norm(impulse_delta),
            capability.maximum_launch_impulse_newton_seconds + 1.0e-9);

  const double seconds =
      std::chrono::duration<double>(segment.flight_time).count();
  const Vec3 landing_position{
      .x = segment.launch_pose.position_m.x +
           segment.launch_velocity_mps.x * seconds +
           0.5 * capability.gravity_mps2.x * seconds * seconds,
      .y = segment.launch_pose.position_m.y +
           segment.launch_velocity_mps.y * seconds +
           0.5 * capability.gravity_mps2.y * seconds * seconds,
      .z = segment.launch_pose.position_m.z +
           segment.launch_velocity_mps.z * seconds +
           0.5 * capability.gravity_mps2.z * seconds * seconds,
  };
  const auto goal = std::get<PointGoal>(input.goal_map.target);
  EXPECT_LE(std::hypot(landing_position.x - goal.position_m.x,
                       landing_position.y - goal.position_m.y),
            goal.tolerance_m + 1.0e-9);
  EXPECT_NEAR(landing_position.z,
              segment.landing_region_boundary_m.front().z +
                  capability.body_half_extent_m.z,
              1.0e-9);
  const Vec3 landing_velocity{
      .x = segment.launch_velocity_mps.x + capability.gravity_mps2.x * seconds,
      .y = segment.launch_velocity_mps.y + capability.gravity_mps2.y * seconds,
      .z = segment.launch_velocity_mps.z + capability.gravity_mps2.z * seconds,
  };
  EXPECT_LE(Norm(landing_velocity),
            capability.maximum_landing_speed_mps + 1.0e-9);
  const double gravity_norm = Norm(capability.gravity_mps2);
  const double downward_speed =
      (landing_velocity.x * capability.gravity_mps2.x +
       landing_velocity.y * capability.gravity_mps2.y +
       landing_velocity.z * capability.gravity_mps2.z) /
      gravity_norm;
  EXPECT_GE(downward_speed,
            capability.minimum_downward_impact_speed_mps - 1.0e-9);
}

TEST(HopperPlanner, AllowsCertifiedLandingBoundaryOutsideNarrowGoalTolerance) {
  Planner planner;
  const PlannerInput input = NarrowGoalLargeLandingRegionInput();
  const PointGoal goal = std::get<PointGoal>(input.goal_map.target);

  const PlannerOutput output = planner.Plan(input);

  ASSERT_EQ(output.outcome, PlanningOutcome::kNewReferenceAvailable)
      << output.reason_code;
  const HopSegment &segment = HopperReference(output).segments.front();
  const double seconds =
      std::chrono::duration<double>(segment.flight_time).count();
  const Vec3 landing_position{
      .x = segment.launch_pose.position_m.x +
           segment.launch_velocity_mps.x * seconds,
      .y = segment.launch_pose.position_m.y +
           segment.launch_velocity_mps.y * seconds,
      .z = segment.launch_pose.position_m.z +
           segment.launch_velocity_mps.z * seconds +
           0.5 * std::get<HopperCapability>(input.capability).gravity_mps2.z *
               seconds * seconds,
  };
  EXPECT_LE(std::hypot(landing_position.x - goal.position_m.x,
                       landing_position.y - goal.position_m.y),
            goal.tolerance_m + 1.0e-9);
  ASSERT_GE(segment.landing_region_boundary_m.size(), 4U);
  EXPECT_TRUE(std::ranges::any_of(
      segment.landing_region_boundary_m, [&](const Vec3 point) {
        return std::hypot(point.x - goal.position_m.x,
                          point.y - goal.position_m.y) >
               goal.tolerance_m + 1.0e-9;
      }));
}

TEST(HopperPlanner, ClampsAuthorizationToTheFirstHop) {
  Planner planner;
  auto input = test::MakeValidHopperInput();
  input.config.hopper.maximum_authorized_hops = 4U;

  const PlannerOutput output = planner.Plan(input);

  ASSERT_EQ(output.outcome, PlanningOutcome::kNewReferenceAvailable)
      << output.reason_code;
  EXPECT_EQ(HopperReference(output).segments.size(), 1U);
  EXPECT_NE(std::ranges::find(output.diagnostics.warning_codes,
                              "HOPPER_AUTHORIZATION_CLAMPED_TO_ONE"),
            output.diagnostics.warning_codes.end());
}

TEST(HopperPlanner, IsDeterministicForSameTypedSnapshot) {
  Planner planner;
  const PlannerInput input = test::MakeValidHopperInput();

  const PlannerOutput first = planner.Plan(input);
  const PlannerOutput second = planner.Plan(input);

  ASSERT_EQ(first.outcome, PlanningOutcome::kNewReferenceAvailable)
      << first.reason_code;
  ASSERT_EQ(second.outcome, first.outcome);
  EXPECT_EQ(second.reason_code, first.reason_code);
  EXPECT_EQ(second.diagnostics.best_cost, first.diagnostics.best_cost);
  const HopSegment &first_hop = HopperReference(first).segments.front();
  const HopSegment &second_hop = HopperReference(second).segments.front();
  EXPECT_EQ(second_hop.segment_id, first_hop.segment_id);
  EXPECT_EQ(second_hop.flight_time, first_hop.flight_time);
  EXPECT_EQ(second_hop.launch_velocity_mps, first_hop.launch_velocity_mps);
  EXPECT_EQ(second_hop.landing_region_boundary_m,
            first_hop.landing_region_boundary_m);
}

} // namespace
} // namespace lunar::planning

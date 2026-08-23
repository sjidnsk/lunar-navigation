#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "lunar_pure_planner_core/planner.hpp"
#include "test_fixtures.hpp"

namespace lunar::pure_planning {
namespace {

const HopReference& HopperReferenceOf(const PlannerOutput& output) {
  return std::get<HopReference>(output.reference->data);
}

PlannerInput LongHopInput(const double distance_m) {
  PlannerInput input = test::MakeValidHopperInput();
  constexpr double local_resolution_m = 0.2;
  constexpr std::array<std::size_t, 6U> scale_factors{1U, 2U, 4U, 8U,
                                                      16U, 20U};
  const double physical_width_m = distance_m + 12.0;
  double global_resolution_m = local_resolution_m;
  for (const std::size_t factor : scale_factors) {
    global_resolution_m =
        local_resolution_m * static_cast<double>(factor);
    if (std::ceil(physical_width_m / global_resolution_m) <= 256.0) {
      break;
    }
  }
  const std::size_t global_width = static_cast<std::size_t>(
      std::ceil(physical_width_m / global_resolution_m));
  const std::size_t local_width = static_cast<std::size_t>(
      std::ceil(physical_width_m / local_resolution_m));
  input.world.global_map =
      test::MakeFlatMap(
          "map", global_width,
          static_cast<std::size_t>(std::ceil(12.0 / global_resolution_m)),
          global_resolution_m);
  input.world.local_map =
      test::MakeFlatMap("odom", local_width, 60U, local_resolution_m);
  auto& state = std::get<HopperState>(input.current_state);
  state.pose.position_m = {5.0, 5.0, 0.0};
  input.goal_map.target = PointGoal{
      .position_m = {5.0 + distance_m, 5.0, 12.0},
      .tolerance_m = 0.0,
  };
  input.config.global_map.base_resolution_m = local_resolution_m;
  return input;
}

void SetObstacle(
    GridMap& map, const double x_m, const double y_m,
    const float height_m) {
  const std::size_t x = static_cast<std::size_t>(
      std::floor((x_m - map.origin_m.x) / map.resolution_m));
  const std::size_t y = static_cast<std::size_t>(
      std::floor((y_m - map.origin_m.y) / map.resolution_m));
  const std::size_t index = y * map.width + x;
  std::get<std::vector<std::uint8_t>>(
      map.layers.at("obstacle").values)[index] = 1U;
  std::get<std::vector<float>>(
      map.layers.at("obstacle_height").values)[index] = height_m;
}

TEST(HopperPlanner, ProducesOneExactEnvelopeCertifiedHopWithoutContinuation) {
  Planner planner;
  const PlannerInput input = test::MakeValidHopperInput();

  const PlannerOutput output = planner.Plan(input);

  ASSERT_EQ(output.outcome, PlanningOutcome::kNewReferenceAvailable)
      << output.reason_code;
  ASSERT_TRUE(output.reference.has_value());
  ASSERT_EQ(output.reference->platform_type, PlatformType::kHopper);
  const HopReference& reference = HopperReferenceOf(output);
  ASSERT_EQ(reference.segments.size(), 1U);
  EXPECT_EQ(output.reason_code, "HOPPER_SINGLE_HOP_AVAILABLE");
  EXPECT_EQ(output.continuation, nullptr);
  ASSERT_EQ(output.certified_hops.size(), 1U);
  EXPECT_TRUE(output.certified_hops.front().promotion_region_map.empty());

  const HopSegment& segment = reference.segments.front();
  EXPECT_DOUBLE_EQ(segment.flight_tube_radius_m, 0.75);
  EXPECT_GT(segment.flight_time.count(), 0);
  ASSERT_GE(segment.landing_region_boundary_m.size(), 3U);
  const double seconds =
      std::chrono::duration<double>(segment.flight_time).count();
  const Vec3 landing{
      segment.launch_pose.position_m.x +
          segment.launch_velocity_mps.x * seconds,
      segment.launch_pose.position_m.y +
          segment.launch_velocity_mps.y * seconds,
      segment.launch_pose.position_m.z +
          segment.launch_velocity_mps.z * seconds -
          0.5 * 1.62 * seconds * seconds,
  };
  const PointGoal goal = std::get<PointGoal>(input.goal_map.target);
  EXPECT_NEAR(landing.x, goal.position_m.x, 1.0e-8);
  EXPECT_NEAR(landing.y, goal.position_m.y, 1.0e-8);
  EXPECT_NEAR(landing.z, 0.0, 1.0e-8);
  EXPECT_NEAR(segment.nominal_landing_point_m.x, landing.x, 1.0e-8);
  EXPECT_NEAR(segment.nominal_landing_point_m.y, landing.y, 1.0e-8);
  EXPECT_NEAR(segment.nominal_landing_point_m.z, landing.z, 1.0e-8);
  EXPECT_GE(segment.available_delta_v_mps, segment.required_delta_v_mps);
  EXPECT_EQ(segment.capability_version, input.capability_version);
  EXPECT_EQ(segment.global_map_generation, input.global_map_generation);
  EXPECT_EQ(segment.local_map_generation, input.local_map_generation);
  ASSERT_TRUE(output.diagnostics.best_cost.has_value());
  EXPECT_DOUBLE_EQ(
      *output.diagnostics.best_cost, segment.required_delta_v_mps);
}

TEST(HopperPlanner, RepeatsSingleHopPlanningWithoutDynamicFuelState) {
  Planner planner;
  PlannerInput first_input = test::MakeValidHopperInput();
  PlannerInput second_input = first_input;
  second_input.request_id = "public-api-hopper-second";

  const PlannerOutput first = planner.Plan(first_input);
  const PlannerOutput second = planner.Plan(second_input);

  ASSERT_EQ(first.outcome, PlanningOutcome::kNewReferenceAvailable)
      << first.reason_code;
  ASSERT_EQ(second.outcome, PlanningOutcome::kNewReferenceAvailable)
      << second.reason_code;
  const HopSegment& first_hop = HopperReferenceOf(first).segments.front();
  const HopSegment& second_hop = HopperReferenceOf(second).segments.front();
  EXPECT_DOUBLE_EQ(first_hop.available_delta_v_mps,
                   second_hop.available_delta_v_mps);
}

TEST(HopperPlanner, UsesConfiguredGravityForTheCertifiedBallisticArc) {
  Planner planner;
  PlannerInput lunar = test::MakeValidHopperInput();
  PlannerInput altered = lunar;
  std::get<HopperCapability>(altered.capability).gravity_mps2 = {
      0.0, 0.0, -0.81};

  const PlannerOutput lunar_output = planner.Plan(lunar);
  const PlannerOutput altered_output = planner.Plan(altered);

  ASSERT_EQ(lunar_output.outcome, PlanningOutcome::kNewReferenceAvailable)
      << lunar_output.reason_code;
  ASSERT_EQ(altered_output.outcome, PlanningOutcome::kNewReferenceAvailable)
      << altered_output.reason_code;
  const HopSegment& lunar_hop = HopperReferenceOf(lunar_output).segments.front();
  const HopSegment& altered_hop = HopperReferenceOf(altered_output).segments.front();
  EXPECT_NE(lunar_hop.launch_velocity_mps, altered_hop.launch_velocity_mps);
  const double seconds = std::chrono::duration<double>(
      altered_hop.flight_time).count();
  const Vec3 landing{
      altered_hop.launch_pose.position_m.x +
          altered_hop.launch_velocity_mps.x * seconds,
      altered_hop.launch_pose.position_m.y +
          altered_hop.launch_velocity_mps.y * seconds,
      altered_hop.launch_pose.position_m.z +
          altered_hop.launch_velocity_mps.z * seconds -
          0.5 * 0.81 * seconds * seconds,
  };
  EXPECT_NEAR(landing.z, altered_hop.nominal_landing_point_m.z, 1.0e-8);
}

TEST(HopperPlanner, HasNoHardCodedDistanceLimitAtHundredMeters) {
  Planner planner;
  const PlannerInput input = LongHopInput(100.0);

  const PlannerOutput output = planner.Plan(input);

  ASSERT_EQ(output.outcome, PlanningOutcome::kNewReferenceAvailable)
      << output.reason_code;
  ASSERT_EQ(HopperReferenceOf(output).segments.size(), 1U);
  const HopSegment& segment = HopperReferenceOf(output).segments.front();
  EXPECT_GE(segment.available_delta_v_mps, segment.required_delta_v_mps);
  const double seconds = std::chrono::duration<double>(
      HopperReferenceOf(output).segments.front().flight_time).count();
  EXPECT_GT(seconds, 0.0);
  EXPECT_LT(seconds, 15.0);
}

TEST(HopperPlanner, SearchesAnAlternateFeasibleTimeWhenMinimumArcIsBlocked) {
  Planner planner;
  PlannerInput input = LongHopInput(20.0);
  SetObstacle(input.world.global_map, 15.0, 5.0, 6.0F);

  const PlannerOutput output = planner.Plan(input);

  ASSERT_EQ(output.outcome, PlanningOutcome::kNewReferenceAvailable)
      << output.reason_code;
  const double seconds = std::chrono::duration<double>(
      HopperReferenceOf(output).segments.front().flight_time).count();
  const double minimum_energy_time = std::sqrt(2.0 * 20.0 / 1.62);
  EXPECT_GT(seconds, minimum_energy_time);
  EXPECT_GT(output.diagnostics.hierarchical->hopper_certification_attempts, 2U);
}

TEST(HopperPlanner, RejectsObstacleThatIntersectsEveryFuelFeasibleFlightTube) {
  Planner planner;
  PlannerInput input = LongHopInput(20.0);
  SetObstacle(input.world.global_map, 15.0, 5.0, 1000.0F);

  const PlannerOutput output = planner.Plan(input);

  EXPECT_EQ(output.outcome, PlanningOutcome::kNoKnownSafeRoute);
  EXPECT_FALSE(output.reference.has_value());
  EXPECT_EQ(output.reason_code, "HOPPER_ALL_FLIGHT_TUBES_BLOCKED");
}

TEST(HopperPlanner, IgnoresCurrentVelocityForStationaryLaunchModel) {
  Planner planner;
  PlannerInput moving = test::MakeValidHopperInput();
  std::get<HopperState>(moving.current_state).velocity = Twist3{
      .linear_mps = {3.0, -2.0, 1.0},
      .angular_radps = {0.4, -0.2, 0.3},
  };
  PlannerInput stationary = moving;
  std::get<HopperState>(stationary.current_state).velocity = {};

  const PlannerOutput first = planner.Plan(moving);
  const PlannerOutput second = planner.Plan(stationary);

  ASSERT_EQ(first.outcome, PlanningOutcome::kNewReferenceAvailable)
      << first.reason_code;
  ASSERT_EQ(second.outcome, PlanningOutcome::kNewReferenceAvailable)
      << second.reason_code;
  EXPECT_EQ(HopperReferenceOf(first).segments.front().flight_time,
            HopperReferenceOf(second).segments.front().flight_time);
  EXPECT_EQ(HopperReferenceOf(first).segments.front().launch_velocity_mps,
            HopperReferenceOf(second).segments.front().launch_velocity_mps);
}

TEST(HopperPlanner, IsDeterministicForSameFrozenSnapshot) {
  Planner planner;
  const PlannerInput input = LongHopInput(20.0);

  const PlannerOutput first = planner.Plan(input);
  const PlannerOutput second = planner.Plan(input);

  ASSERT_EQ(first.outcome, PlanningOutcome::kNewReferenceAvailable)
      << first.reason_code;
  ASSERT_EQ(second.outcome, first.outcome);
  const HopSegment& first_hop = HopperReferenceOf(first).segments.front();
  const HopSegment& second_hop = HopperReferenceOf(second).segments.front();
  EXPECT_EQ(second_hop.segment_id, first_hop.segment_id);
  EXPECT_EQ(second_hop.flight_time, first_hop.flight_time);
  EXPECT_EQ(second_hop.launch_velocity_mps, first_hop.launch_velocity_mps);
  EXPECT_EQ(second_hop.landing_region_boundary_m,
            first_hop.landing_region_boundary_m);
  EXPECT_EQ(second.diagnostics.best_cost, first.diagnostics.best_cost);
}

}  // namespace
}  // namespace lunar::pure_planning

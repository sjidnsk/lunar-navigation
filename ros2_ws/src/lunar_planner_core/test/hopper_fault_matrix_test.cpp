#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stop_token>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "lunar_planner_core/planner.hpp"
#include "test_fixtures.hpp"

namespace lunar::planning {
namespace {

void SetFloat(GridMap &map, const std::string &layer, const std::size_t x,
              const std::size_t y, const float value) {
  std::get<std::vector<float>>(map.layers.at(layer).values)
      .at(y * map.width + x) = value;
}

void SetByte(GridMap &map, const std::string &layer, const std::size_t x,
             const std::size_t y, const std::uint8_t value) {
  std::get<std::vector<std::uint8_t>>(map.layers.at(layer).values)
      .at(y * map.width + x) = value;
}

void MakeExactCellGoal(PlannerInput &input) {
  input.goal_map.target = PointGoal{
      .position_m = {4.25, 3.25, 0.0},
      .tolerance_m = 0.05,
  };
}

void ExpectNoNewHop(const PlannerOutput &output) {
  EXPECT_NE(output.outcome, PlanningOutcome::kNewReferenceAvailable)
      << output.reason_code;
  EXPECT_NE(output.directive, ExecutionDirective::kActivateNewReference);
  EXPECT_FALSE(output.reference.has_value());
}

TEST(HopperFaultMatrix, RejectsLandingSlopeRoughnessAndPlaneResidual) {
  Planner planner;

  auto slope = test::MakeValidHopperInput();
  MakeExactCellGoal(slope);
  SetFloat(slope.world.local_map, "elevation", 9U, 6U, 1.0F);
  ExpectNoNewHop(planner.Plan(slope));

  auto roughness = test::MakeValidHopperInput();
  MakeExactCellGoal(roughness);
  SetFloat(roughness.world.local_map, "elevation_variance", 8U, 6U, 0.0225F);
  ExpectNoNewHop(planner.Plan(roughness));

  auto residual = test::MakeValidHopperInput();
  MakeExactCellGoal(residual);
  residual.config.map_safety.project_maximum_slope_rad = 0.5;
  SetFloat(residual.world.local_map, "elevation", 8U, 6U, 0.08F);
  SetFloat(residual.world.local_map, "elevation", 7U, 6U, 0.0F);
  SetFloat(residual.world.local_map, "elevation", 9U, 6U, 0.0F);
  ExpectNoNewHop(planner.Plan(residual));
}

TEST(HopperFaultMatrix, RejectsInsufficientLandingAreaAndClearance) {
  Planner planner;

  auto area = test::MakeValidHopperInput();
  MakeExactCellGoal(area);
  std::get<HopperCapability>(area.capability).minimum_landing_region_area_m2 =
      0.3;
  SetByte(area.world.global_map, "valid_mask", 9U, 6U, 0U);
  SetByte(area.world.local_map, "valid_mask", 9U, 6U, 0U);
  ExpectNoNewHop(planner.Plan(area));

  auto lateral = test::MakeValidHopperInput();
  MakeExactCellGoal(lateral);
  std::get<HopperCapability>(lateral.capability).minimum_lateral_clearance_m =
      0.3;
  SetByte(lateral.world.local_map, "obstacle", 9U, 6U, 1U);
  SetFloat(lateral.world.local_map, "obstacle_height", 9U, 6U, 1.0F);
  ExpectNoNewHop(planner.Plan(lateral));

  auto overhead = test::MakeValidHopperInput();
  MakeExactCellGoal(overhead);
  std::get<HopperCapability>(overhead.capability).minimum_overhead_clearance_m =
      0.2;
  SetByte(overhead.world.local_map, "obstacle", 7U, 6U, 1U);
  SetFloat(overhead.world.local_map, "obstacle_height", 7U, 6U, 4.0F);
  ExpectNoNewHop(planner.Plan(overhead));
}

TEST(HopperFaultMatrix, RejectsLaunchImpulseFlightAndLandingLimits) {
  Planner planner;

  auto launch_speed = test::MakeValidHopperInput();
  std::get<HopperCapability>(launch_speed.capability).maximum_launch_speed_mps =
      0.5;
  ExpectNoNewHop(planner.Plan(launch_speed));

  auto impulse = test::MakeValidHopperInput();
  std::get<HopperCapability>(impulse.capability)
      .maximum_launch_impulse_newton_seconds = 1.0;
  ExpectNoNewHop(planner.Plan(impulse));

  auto flight_time = test::MakeValidHopperInput();
  auto &flight_capability = std::get<HopperCapability>(flight_time.capability);
  flight_capability.minimum_flight_time = std::chrono::seconds{2};
  flight_capability.maximum_flight_time = std::chrono::seconds{1};
  const PlannerOutput invalid_time = planner.Plan(flight_time);
  EXPECT_EQ(invalid_time.outcome, PlanningOutcome::kInvalidRequest);
  ExpectNoNewHop(invalid_time);

  auto landing_speed = test::MakeValidHopperInput();
  std::get<HopperCapability>(landing_speed.capability)
      .maximum_landing_speed_mps = 0.5;
  ExpectNoNewHop(planner.Plan(landing_speed));
}

TEST(HopperFaultMatrix, RejectsAttitudeUnknownAndNumericalUncertainty) {
  Planner planner;

  auto attitude = test::MakeValidHopperInput();
  std::get<HopperState>(attitude.current_state).velocity.angular_radps.z = 0.3;
  ExpectNoNewHop(planner.Plan(attitude));

  auto unknown = test::MakeValidHopperInput();
  MakeExactCellGoal(unknown);
  SetByte(unknown.world.local_map, "valid_mask", 8U, 6U, 0U);
  ExpectNoNewHop(planner.Plan(unknown));

  auto uncertainty = test::MakeValidHopperInput();
  MakeExactCellGoal(uncertainty);
  SetFloat(uncertainty.world.local_map, "obstacle_variance", 8U, 6U, 0.05F);
  ExpectNoNewHop(planner.Plan(uncertainty));

  auto nonfinite = test::MakeValidHopperInput();
  std::get<HopperState>(nonfinite.current_state).pose.position_m.z =
      std::numeric_limits<double>::quiet_NaN();
  const PlannerOutput invalid_state = planner.Plan(nonfinite);
  EXPECT_EQ(invalid_state.outcome, PlanningOutcome::kInvalidRequest);
  ExpectNoNewHop(invalid_state);
}

TEST(HopperFaultMatrix, CancelsWithoutPublishingAReference) {
  Planner planner;
  auto input = test::MakeValidHopperInput();
  std::stop_source stop_source;
  stop_source.request_stop();
  input.stop_token = stop_source.get_token();

  const PlannerOutput output = planner.Plan(input);

  EXPECT_EQ(output.outcome, PlanningOutcome::kCanceled);
  EXPECT_EQ(output.directive, ExecutionDirective::kHoldPosition);
  EXPECT_EQ(output.reason_code, "REQUEST_CANCELED");
  EXPECT_FALSE(output.reference.has_value());
}

} // namespace
} // namespace lunar::planning

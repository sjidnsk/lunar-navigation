#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <stop_token>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include "lunar_planner_core/planner.hpp"
#include "test_fixtures.hpp"

namespace lunar::planning {
namespace {

void SetByte(
    GridMap& map, const std::string& layer, const std::size_t x,
    const std::size_t y, const std::uint8_t value) {
  std::get<std::vector<std::uint8_t>>(map.layers.at(layer).values)
      .at(y * map.width + x) = value;
}

void SetFloat(
    GridMap& map, const std::string& layer, const std::size_t x,
    const std::size_t y, const float value) {
  std::get<std::vector<float>>(map.layers.at(layer).values)
      .at(y * map.width + x) = value;
}

void ExpectFailure(
    const PlannerOutput& output, const PlanningOutcome outcome,
    const std::string_view reason_code) {
  EXPECT_EQ(output.outcome, outcome);
  EXPECT_EQ(output.reason_code, reason_code);
  EXPECT_FALSE(output.reference.has_value());
  EXPECT_NE(output.directive, ExecutionDirective::kActivateNewReference);
}

TEST(HopperFaultMatrix, RequiresAnExactPointWithoutYaw) {
  Planner planner;

  PlannerInput tolerance = test::MakeValidHopperInput();
  std::get<PointGoal>(tolerance.goal_map.target).tolerance_m = 0.01;
  ExpectFailure(
      planner.Plan(tolerance), PlanningOutcome::kInvalidRequest,
      "HOPPER_EXACT_POINT_REQUIRED");

  PlannerInput yaw = test::MakeValidHopperInput();
  yaw.goal_map.yaw_rad = 0.0;
  ExpectFailure(
      planner.Plan(yaw), PlanningOutcome::kInvalidRequest,
      "HOPPER_EXACT_POINT_REQUIRED");

  PlannerInput region = test::MakeValidHopperInput();
  region.goal_map.target = PlanarRegionGoal{
      .boundary_m = {{3.0, 2.0, 0.0}, {5.0, 2.0, 0.0},
                     {5.0, 4.0, 0.0}},
  };
  ExpectFailure(
      planner.Plan(region), PlanningOutcome::kInvalidRequest,
      "HOPPER_EXACT_POINT_REQUIRED");
}

TEST(HopperFaultMatrix, RejectsMissingInvalidAndInsufficientPropellant) {
  Planner planner;

  PlannerInput missing = test::MakeValidHopperInput();
  missing.hopper_propellant.reset();
  ExpectFailure(
      planner.Plan(missing), PlanningOutcome::kInvalidRequest,
      "HOPPER_PROPELLANT_STATE_INVALID");

  PlannerInput invalid = test::MakeValidHopperInput();
  invalid.hopper_propellant->total_mass_kg = 0.0;
  ExpectFailure(
      planner.Plan(invalid), PlanningOutcome::kInvalidRequest,
      "HOPPER_TOTAL_MASS_INVALID");

  PlannerInput insufficient = test::MakeValidHopperInput();
  insufficient.hopper_propellant->remaining_usable_fuel_mass_kg = 1.0e-6;
  ExpectFailure(
      planner.Plan(insufficient), PlanningOutcome::kGoalInfeasible,
      "HOPPER_FUEL_INSUFFICIENT");
}

TEST(HopperFaultMatrix, RejectsSupportDiskObstacleForbiddenUnknownAndBoundary) {
  Planner planner;

  PlannerInput obstacle = test::MakeValidHopperInput();
  SetByte(obstacle.world.local_map, "obstacle", 8U, 6U, 1U);
  SetFloat(obstacle.world.local_map, "obstacle_height", 8U, 6U, 0.4F);
  ExpectFailure(
      planner.Plan(obstacle), PlanningOutcome::kGoalInfeasible,
      "HOPPER_LANDING_TARGET_OCCUPIED");

  PlannerInput forbidden = test::MakeValidHopperInput();
  SetByte(forbidden.world.local_map, "forbidden", 8U, 6U, 1U);
  ExpectFailure(
      planner.Plan(forbidden), PlanningOutcome::kGoalInfeasible,
      "HOPPER_LANDING_TARGET_OCCUPIED");

  PlannerInput unknown = test::MakeValidHopperInput();
  SetByte(unknown.world.local_map, "valid_mask", 8U, 6U, 0U);
  ExpectFailure(
      planner.Plan(unknown), PlanningOutcome::kGoalInfeasible,
      "LANDING_EVIDENCE_INSUFFICIENT");

  PlannerInput boundary = test::MakeValidHopperInput();
  std::get<PointGoal>(boundary.goal_map.target).position_m = {0.1, 0.1, 0.0};
  ExpectFailure(
      planner.Plan(boundary), PlanningOutcome::kGoalInfeasible,
      "LANDING_EVIDENCE_INSUFFICIENT");
}

TEST(HopperFaultMatrix, EnforcesSlopeAndPlaneResidualButNotRoughness) {
  Planner planner;

  PlannerInput slope = test::MakeValidHopperInput();
  auto& elevations = std::get<std::vector<float>>(
      slope.world.local_map.layers.at("elevation").values);
  const double gradient = std::tan(11.0 * std::numbers::pi / 180.0);
  for (std::size_t y = 0U; y < slope.world.local_map.height; ++y) {
    for (std::size_t x = 0U; x < slope.world.local_map.width; ++x) {
      elevations[y * slope.world.local_map.width + x] =
          static_cast<float>((static_cast<double>(x) + 0.5) *
                             slope.world.local_map.resolution_m * gradient);
    }
  }
  ExpectFailure(
      planner.Plan(slope), PlanningOutcome::kGoalInfeasible,
      "HOPPER_LANDING_SLOPE_EXCEEDED");

  PlannerInput residual = test::MakeValidHopperInput();
  SetFloat(residual.world.local_map, "elevation", 8U, 6U, 0.20F);
  ExpectFailure(
      planner.Plan(residual), PlanningOutcome::kGoalInfeasible,
      "HOPPER_LANDING_PLANE_RESIDUAL_EXCEEDED");

  PlannerInput rough = test::MakeValidHopperInput();
  auto& variance = std::get<std::vector<float>>(
      rough.world.local_map.layers.at("elevation_variance").values);
  std::ranges::fill(variance, 0.25F);
  const PlannerOutput rough_result = planner.Plan(rough);
  EXPECT_EQ(rough_result.outcome, PlanningOutcome::kNewReferenceAvailable)
      << rough_result.reason_code;
}

TEST(HopperFaultMatrix, ReportsCompleteFlightTubeBlockageWithoutAReference) {
  Planner planner;
  PlannerInput blocked = test::MakeValidHopperInput();
  for (std::size_t y = 0U; y < blocked.world.global_map.height; ++y) {
    SetByte(blocked.world.global_map, "forbidden", 7U, y, 1U);
  }

  ExpectFailure(
      planner.Plan(blocked), PlanningOutcome::kNoKnownSafeRoute,
      "HOPPER_ALL_FLIGHT_TUBES_BLOCKED");
}

TEST(HopperFaultMatrix, CancellationNeverPublishesAReference) {
  Planner planner;
  PlannerInput input = test::MakeValidHopperInput();
  std::stop_source stop;
  stop.request_stop();
  input.stop_token = stop.get_token();

  ExpectFailure(
      planner.Plan(input), PlanningOutcome::kCanceled,
      "REQUEST_CANCELED");
}

}  // namespace
}  // namespace lunar::planning

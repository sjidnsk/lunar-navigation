#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <stop_token>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include "lunar_pure_planner_core/planner.hpp"
#include "test_fixtures.hpp"

namespace lunar::pure_planning {
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
    const std::string_view reason_code,
    const CandidateDisposition candidate_disposition) {
  EXPECT_EQ(output.outcome, outcome);
  EXPECT_EQ(output.reason_code, reason_code);
  EXPECT_EQ(output.candidate_disposition, candidate_disposition);
  EXPECT_FALSE(output.reference.has_value());
  EXPECT_NE(output.directive, ExecutionDirective::kActivateNewReference);
}

TEST(HopperFaultMatrix, RequiresAnExactPointWithoutYaw) {
  Planner planner;

  PlannerInput tolerance = test::MakeValidHopperInput();
  std::get<PointGoal>(tolerance.goal_map.target).tolerance_m = 0.01;
  ExpectFailure(planner.Plan(tolerance), PlanningOutcome::kInvalidRequest,
                "HOPPER_EXACT_POINT_REQUIRED", CandidateDisposition::kKeep);

  PlannerInput yaw = test::MakeValidHopperInput();
  yaw.goal_map.yaw_rad = 0.0;
  ExpectFailure(planner.Plan(yaw), PlanningOutcome::kInvalidRequest,
                "HOPPER_EXACT_POINT_REQUIRED", CandidateDisposition::kKeep);

  PlannerInput region = test::MakeValidHopperInput();
  region.goal_map.target = PlanarRegionGoal{
      .boundary_m = {{3.0, 2.0, 0.0}, {5.0, 2.0, 0.0},
                     {5.0, 4.0, 0.0}},
  };
  ExpectFailure(planner.Plan(region), PlanningOutcome::kInvalidRequest,
                "HOPPER_EXACT_POINT_REQUIRED", CandidateDisposition::kKeep);
}

TEST(HopperFaultMatrix, RejectsInvalidAndExceededSingleHopEnvelope) {
  Planner planner;

  PlannerInput invalid = test::MakeValidHopperInput();
  std::get<HopperCapability>(invalid.capability).reference_total_mass_kg = 0.0;
  ExpectFailure(planner.Plan(invalid), PlanningOutcome::kInvalidRequest,
                "HOPPER_CAPABILITY_INVALID", CandidateDisposition::kKeep);

  PlannerInput exceeded = test::MakeValidHopperInput();
  std::get<HopperCapability>(exceeded.capability)
      .reference_propellant_mass_kg = 1.0e-6;
  ExpectFailure(planner.Plan(exceeded), PlanningOutcome::kNoKnownSafeRoute,
                "HOPPER_SINGLE_HOP_ENVELOPE_EXCEEDED",
                CandidateDisposition::kSuppressForCurrentPhysicalSnapshot);
}

TEST(HopperFaultMatrix, RejectsSupportDiskObstacleForbiddenUnknownAndBoundary) {
  Planner planner;

  PlannerInput obstacle = test::MakeValidHopperInput();
  SetByte(obstacle.world.local_map, "obstacle", 8U, 6U, 1U);
  SetFloat(obstacle.world.local_map, "obstacle_height", 8U, 6U, 0.4F);
  ExpectFailure(planner.Plan(obstacle), PlanningOutcome::kGoalInfeasible,
                "HOPPER_LANDING_TARGET_OCCUPIED",
                CandidateDisposition::kSuppressForCurrentPhysicalSnapshot);

  PlannerInput forbidden = test::MakeValidHopperInput();
  SetByte(forbidden.world.local_map, "forbidden", 8U, 6U, 1U);
  ExpectFailure(planner.Plan(forbidden), PlanningOutcome::kGoalInfeasible,
                "HOPPER_LANDING_TARGET_OCCUPIED",
                CandidateDisposition::kSuppressForCurrentPhysicalSnapshot);

  PlannerInput unknown = test::MakeValidHopperInput();
  SetByte(unknown.world.local_map, "valid_mask", 8U, 6U, 0U);
  ExpectFailure(planner.Plan(unknown), PlanningOutcome::kGoalInfeasible,
                "LANDING_EVIDENCE_INSUFFICIENT",
                CandidateDisposition::kSuppressForCurrentPhysicalSnapshot);

  PlannerInput boundary = test::MakeValidHopperInput();
  std::get<PointGoal>(boundary.goal_map.target).position_m = {0.1, 0.1, 0.0};
  ExpectFailure(planner.Plan(boundary), PlanningOutcome::kGoalInfeasible,
                "LANDING_EVIDENCE_INSUFFICIENT",
                CandidateDisposition::kSuppressForCurrentPhysicalSnapshot);
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
  ExpectFailure(planner.Plan(slope), PlanningOutcome::kGoalInfeasible,
                "HOPPER_LANDING_SLOPE_EXCEEDED",
                CandidateDisposition::kSuppressForCurrentPhysicalSnapshot);

  PlannerInput residual = test::MakeValidHopperInput();
  SetFloat(residual.world.local_map, "elevation", 8U, 6U, 0.20F);
  ExpectFailure(planner.Plan(residual), PlanningOutcome::kGoalInfeasible,
                "HOPPER_LANDING_PLANE_RESIDUAL_EXCEEDED",
                CandidateDisposition::kSuppressForCurrentPhysicalSnapshot);

  PlannerInput rough = test::MakeValidHopperInput();
  auto& variance = std::get<std::vector<float>>(
      rough.world.local_map.layers.at("elevation_variance").values);
  std::ranges::fill(variance, 0.25F);
  const PlannerOutput rough_result = planner.Plan(rough);
  EXPECT_EQ(rough_result.outcome, PlanningOutcome::kNewReferenceAvailable)
      << rough_result.reason_code;
  EXPECT_EQ(rough_result.candidate_disposition, CandidateDisposition::kKeep);
}

TEST(HopperFaultMatrix, ReportsCompleteFlightTubeBlockageWithoutAReference) {
  Planner planner;
  PlannerInput blocked = test::MakeValidHopperInput();
  for (std::size_t y = 0U; y < blocked.world.global_map.height; ++y) {
    SetByte(blocked.world.global_map, "forbidden", 7U, y, 1U);
  }

  ExpectFailure(planner.Plan(blocked), PlanningOutcome::kNoKnownSafeRoute,
                "HOPPER_ALL_FLIGHT_TUBES_BLOCKED",
                CandidateDisposition::kSuppressForCurrentPhysicalSnapshot);
}

TEST(HopperFaultMatrix, CancellationNeverPublishesAReference) {
  Planner planner;
  PlannerInput input = test::MakeValidHopperInput();
  std::stop_source stop;
  stop.request_stop();
  input.stop_token = stop.get_token();

  ExpectFailure(planner.Plan(input), PlanningOutcome::kCanceled,
                "REQUEST_CANCELED", CandidateDisposition::kKeep);
}

TEST(HopperFaultMatrix, KeepsCommittedAndInvalidatedActiveReferences) {
  Planner planner;
  PlannerInput committed = test::MakeValidHopperInput();
  committed.previous_execution = ExecutionContext{HopperExecutionContext{
      .state = HopperExecutionState::kJumpCommitted,
      .active_plan_id = "active-plan",
      .active_segment_id = "active-hop",
  }};

  const PlannerOutput continuing = planner.Plan(committed);

  EXPECT_EQ(continuing.directive, ExecutionDirective::kContinueCommittedHop);
  EXPECT_EQ(continuing.candidate_disposition, CandidateDisposition::kKeep);

  PlannerInput invalidated = committed;
  auto& context =
      std::get<HopperExecutionContext>(*invalidated.previous_execution);
  context.active_segment_id.reset();

  const PlannerOutput rejected = planner.Plan(invalidated);

  EXPECT_EQ(rejected.outcome, PlanningOutcome::kActiveReferenceInvalidated);
  EXPECT_EQ(rejected.candidate_disposition, CandidateDisposition::kKeep);
}

}  // namespace
}  // namespace lunar::pure_planning

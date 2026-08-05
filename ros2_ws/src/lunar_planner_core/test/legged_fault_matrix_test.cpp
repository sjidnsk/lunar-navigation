#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stop_token>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "legged/legged_terrain.hpp"
#include "lunar_planner_core/planner.hpp"
#include "shared/map_snapshot.hpp"
#include "shared/safe_projection.hpp"
#include "test_fixtures.hpp"

namespace lunar::planning {
namespace {

void SetFloat(
    GridMap& map, const std::string& layer,
    const std::size_t x, const std::size_t y, const float value) {
  std::get<std::vector<float>>(map.layers.at(layer).values)
      .at(y * map.width + x) = value;
}

void SetByte(
    GridMap& map, const std::string& layer,
    const std::size_t x, const std::size_t y, const std::uint8_t value) {
  std::get<std::vector<std::uint8_t>>(map.layers.at(layer).values)
      .at(y * map.width + x) = value;
}

bool HasReason(
    const legged::LeggedTerrainEvaluation& evaluation,
    const std::string& reason) {
  return std::ranges::find(evaluation.rejection_reasons, reason) !=
      evaluation.rejection_reasons.end();
}

legged::LeggedTerrainEvaluation EvaluateAt(
    PlannerInput input, const shared::GridCell cell) {
  const auto capability = std::get<LeggedCapability>(input.capability);
  const auto snapshot = shared::MapSnapshot::Create(input.world.local_map);
  EXPECT_TRUE(snapshot.ok()) << snapshot.reason_code;
  const auto projection = shared::BuildSafeProjection(
      snapshot.snapshot, input.capability, input.config.map_safety, {});
  EXPECT_TRUE(projection.ok()) << projection.reason_code;
  return legged::EvaluateLeggedTerrainCell(
      *projection.projection, capability, cell, {});
}

TEST(LeggedFaultMatrix, RejectsSlopeRoughnessAndStepViolations) {
  auto slope_input = test::MakeValidLeggedInput();
  SetFloat(slope_input.world.local_map, "elevation", 3U, 3U, 1.0F);
  const auto slope = EvaluateAt(
      slope_input, shared::GridCell{.x = 2, .y = 3});
  EXPECT_FALSE(slope.hard_feasible);
  EXPECT_TRUE(HasReason(slope, "LEGGED_SLOPE_LIMIT"));

  auto rough_input = test::MakeValidLeggedInput();
  std::get<LeggedCapability>(rough_input.capability).maximum_roughness_m = 0.1;
  SetFloat(
      rough_input.world.local_map, "elevation_variance", 2U, 3U, 0.0225F);
  const auto rough = EvaluateAt(
      rough_input, shared::GridCell{.x = 2, .y = 3});
  EXPECT_FALSE(rough.hard_feasible);
  EXPECT_TRUE(HasReason(rough, "LEGGED_ROUGHNESS_LIMIT"));

  auto step_input = test::MakeValidLeggedInput();
  SetFloat(step_input.world.local_map, "elevation", 3U, 3U, 0.35F);
  const auto step = EvaluateAt(
      step_input, shared::GridCell{.x = 2, .y = 3});
  EXPECT_FALSE(step.hard_feasible);
  EXPECT_TRUE(HasReason(step, "LEGGED_STEP_HEIGHT_LIMIT"));
}

TEST(LeggedFaultMatrix, RejectsUnknownTerrainAndBodyClearanceViolation) {
  auto unknown_input = test::MakeValidLeggedInput();
  SetByte(unknown_input.world.local_map, "valid_mask", 2U, 3U, 0U);
  const auto unknown = EvaluateAt(
      unknown_input, shared::GridCell{.x = 2, .y = 3});
  EXPECT_FALSE(unknown.hard_feasible);
  EXPECT_TRUE(HasReason(unknown, "LEGGED_TERRAIN_UNKNOWN"));

  auto clearance_input = test::MakeValidLeggedInput();
  std::get<LeggedCapability>(clearance_input.capability)
      .minimum_body_clearance_m = 1.1;
  SetByte(clearance_input.world.local_map, "obstacle", 3U, 3U, 1U);
  const auto clearance = EvaluateAt(
      clearance_input, shared::GridCell{.x = 2, .y = 3});
  EXPECT_FALSE(clearance.hard_feasible);
  EXPECT_TRUE(HasReason(clearance, "LEGGED_BODY_CLEARANCE_LIMIT"));
}

TEST(LeggedFaultMatrix, ReportsNoRouteAcrossUnknownBarrier) {
  Planner planner;
  auto input = test::MakeValidLeggedInput();
  input.request_id = "legged-unknown-barrier";
  for (std::size_t y = 0U; y < input.world.local_map.height; ++y) {
    SetByte(input.world.local_map, "valid_mask", 3U, y, 0U);
  }

  const PlannerOutput output = planner.Plan(input);

  EXPECT_EQ(output.outcome, PlanningOutcome::kNoKnownSafeRoute);
  EXPECT_EQ(output.directive, ExecutionDirective::kNoSafeReference);
  EXPECT_FALSE(output.reference.has_value());
  EXPECT_EQ(output.diagnostics.planner_name, "cpp_v3_hierarchical");
}

TEST(LeggedFaultMatrix, RejectsObstacleIntersectingOnlyTheTrueBodySweep) {
  Planner planner;
  auto input = test::MakeValidLeggedInput();
  input.request_id = "legged-start-sweep-blocked";
  auto& state = std::get<LeggedState>(input.current_state);
  state.body_pose.position_m = {2.15, 3.5, 0.5};
  SetByte(input.world.local_map, "obstacle", 1U, 3U, 1U);

  const PlannerOutput output = planner.Plan(input);

  EXPECT_EQ(output.outcome, PlanningOutcome::kNoKnownSafeRoute);
  EXPECT_EQ(output.directive, ExecutionDirective::kNoSafeReference);
  EXPECT_EQ(output.reason_code, "LEGGED_START_CONNECTOR_INFEASIBLE");
  EXPECT_FALSE(output.reference.has_value());
}

TEST(LeggedFaultMatrix, RejectsWhenTrueHeightCannotReachAnySuccessorInterval) {
  Planner planner;
  auto input = test::MakeValidLeggedInput();
  input.request_id = "legged-start-height-disconnected";
  auto& state = std::get<LeggedState>(input.current_state);
  state.body_pose.position_m.z = 0.4;
  auto& capability = std::get<LeggedCapability>(input.capability);
  capability.vertical_speed_mps = {-0.01, 0.01};
  capability.motion_primitives.resize(4U);
  for (const auto [x, y] :
       {std::pair{3U, 3U}, std::pair{1U, 3U}, std::pair{2U, 4U},
        std::pair{2U, 2U}}) {
    SetFloat(input.world.local_map, "elevation", x, y, 0.2F);
  }

  const PlannerOutput output = planner.Plan(input);

  EXPECT_EQ(output.outcome, PlanningOutcome::kNoKnownSafeRoute);
  EXPECT_EQ(output.directive, ExecutionDirective::kNoSafeReference);
  EXPECT_EQ(output.reason_code, "LEGGED_START_CONNECTOR_INFEASIBLE");
  EXPECT_FALSE(output.reference.has_value());
}

TEST(LeggedFaultMatrix, CancelsBeforeSearchExpansion) {
  Planner planner;
  auto input = test::MakeValidLeggedInput();
  std::stop_source stop_source;
  stop_source.request_stop();
  input.stop_token = stop_source.get_token();

  const PlannerOutput output = planner.Plan(input);

  EXPECT_EQ(output.outcome, PlanningOutcome::kCanceled);
  EXPECT_EQ(output.directive, ExecutionDirective::kHoldPosition);
  EXPECT_EQ(output.reason_code, "REQUEST_CANCELED");
  EXPECT_FALSE(output.reference.has_value());
}

}  // namespace
}  // namespace lunar::planning

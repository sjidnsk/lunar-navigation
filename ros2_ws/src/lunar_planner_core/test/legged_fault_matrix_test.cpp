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

template <class T>
concept HasMaximumRoughness = requires(T value) {
  value.maximum_roughness_m;
};

template <class T>
concept HasMinimumConfidence = requires(T value) {
  value.minimum_confidence;
};

template <class T>
concept HasVerticalSpeed = requires(T value) {
  value.vertical_speed_mps;
};

template <class T>
concept HasNominalDuration = requires(T value) {
  value.nominal_duration;
};

static_assert(!HasMaximumRoughness<LeggedCapability>);
static_assert(!HasMinimumConfidence<LeggedCapability>);
static_assert(!HasVerticalSpeed<LeggedCapability>);
static_assert(!HasNominalDuration<LeggedBodyPrimitive>);
static_assert(!HasNominalDuration<legged::LeggedTransition>);

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

TEST(LeggedFaultMatrix, RejectsSlopeButReportsRoughnessWithoutUsingIt) {
  auto slope_input = test::MakeValidLeggedInput();
  SetFloat(slope_input.world.local_map, "elevation", 3U, 3U, 2.0F);
  const auto slope = EvaluateAt(
      slope_input, shared::GridCell{.x = 2, .y = 3});
  EXPECT_FALSE(slope.hard_feasible);
  EXPECT_TRUE(HasReason(slope, "LEGGED_SLOPE_LIMIT"));

  auto rough_input = test::MakeValidLeggedInput();
  SetFloat(rough_input.world.local_map, "elevation", 3U, 3U, 0.1F);
  const auto rough = EvaluateAt(
      rough_input, shared::GridCell{.x = 2, .y = 3});
  EXPECT_TRUE(rough.hard_feasible);
  EXPECT_GT(rough.roughness_m, 0.0);
  EXPECT_FALSE(HasReason(rough, "LEGGED_ROUGHNESS_LIMIT"));
}

TEST(LeggedFaultMatrix, AcceptsThirtyDegreeSlopeBoundary) {
  auto input = test::MakeValidLeggedInput();
  SetFloat(input.world.local_map, "elevation", 1U, 3U, 0.0F);
  SetFloat(input.world.local_map, "elevation", 3U, 3U, 1.1547005F);

  const auto boundary = EvaluateAt(
      input, shared::GridCell{.x = 2, .y = 3});

  EXPECT_TRUE(boundary.hard_feasible);
  EXPECT_NEAR(boundary.slope_rad, 0.5235987755982988, 1.0e-6);
}

TEST(LeggedFaultMatrix, RoughnessDoesNotChangeTraversalCost) {
  auto flat_input = test::MakeValidLeggedInput();
  const auto flat_snapshot =
      shared::MapSnapshot::Create(flat_input.world.local_map);
  ASSERT_TRUE(flat_snapshot.ok()) << flat_snapshot.reason_code;
  const auto flat_projection = shared::BuildSafeProjection(
      flat_snapshot.snapshot, flat_input.capability,
      flat_input.config.map_safety, {});
  ASSERT_TRUE(flat_projection.ok()) << flat_projection.reason_code;

  auto rough_input = flat_input;
  SetFloat(rough_input.world.local_map, "elevation", 2U, 3U, 0.1F);
  const auto rough_snapshot =
      shared::MapSnapshot::Create(rough_input.world.local_map);
  ASSERT_TRUE(rough_snapshot.ok()) << rough_snapshot.reason_code;
  const auto rough_projection = shared::BuildSafeProjection(
      rough_snapshot.snapshot, rough_input.capability,
      rough_input.config.map_safety, {});
  ASSERT_TRUE(rough_projection.ok()) << rough_projection.reason_code;
  const shared::GridCell cell{.x = 2, .y = 3};

  EXPECT_GT(rough_projection.projection->RoughnessMeters(cell),
            flat_projection.projection->RoughnessMeters(cell));
  EXPECT_FLOAT_EQ(rough_projection.projection->TraversalCost(cell),
                  flat_projection.projection->TraversalCost(cell));
}

TEST(LeggedFaultMatrix, RejectsUnknownTerrain) {
  auto unknown_input = test::MakeValidLeggedInput();
  SetByte(unknown_input.world.local_map, "valid_mask", 2U, 3U, 0U);
  const auto unknown = EvaluateAt(
      unknown_input, shared::GridCell{.x = 2, .y = 3});
  EXPECT_FALSE(unknown.hard_feasible);
  EXPECT_TRUE(HasReason(unknown, "LEGGED_TERRAIN_UNKNOWN"));
}

TEST(LeggedFaultMatrix, AcceptsHalfMeterStepAndRejectsHigherStep) {
  auto input = test::MakeValidLeggedInput();
  input.world.local_map = test::MakeFlatMap("odom", 30U, 20U, 0.1);
  const auto capability = std::get<LeggedCapability>(input.capability);
  const auto snapshot = shared::MapSnapshot::Create(input.world.local_map);
  ASSERT_TRUE(snapshot.ok()) << snapshot.reason_code;
  const auto projection = shared::BuildSafeProjection(
      snapshot.snapshot, input.capability, input.config.map_safety, {});
  ASSERT_TRUE(projection.ok()) << projection.reason_code;

  SetFloat(input.world.local_map, "elevation", 11U, 10U, 0.5F);
  auto stepped_snapshot = shared::MapSnapshot::Create(input.world.local_map);
  auto stepped_projection = shared::BuildSafeProjection(
      stepped_snapshot.snapshot, input.capability, input.config.map_safety, {});
  ASSERT_TRUE(stepped_projection.ok()) << stepped_projection.reason_code;
  const auto accepted = legged::ValidateLeggedBodySweep(
      {{1.05, 1.05, 0.5}, 0.0}, {{1.15, 1.05, 1.0}, 0.0},
      {.lower = 0.5, .upper = 0.5}, *stepped_projection.projection,
      capability, {});
  EXPECT_TRUE(accepted.valid) << accepted.reason_code;

  SetFloat(input.world.local_map, "elevation", 11U, 10U, 0.51F);
  auto high_snapshot = shared::MapSnapshot::Create(input.world.local_map);
  auto high_projection = shared::BuildSafeProjection(
      high_snapshot.snapshot, input.capability, input.config.map_safety, {});
  ASSERT_TRUE(high_projection.ok()) << high_projection.reason_code;
  const auto rejected = legged::ValidateLeggedBodySweep(
      {{1.05, 1.05, 0.5}, 0.0}, {{1.15, 1.05, 1.01}, 0.0},
      {.lower = 0.5, .upper = 0.5}, *high_projection.projection,
      capability, {});
  EXPECT_FALSE(rejected.valid);
  EXPECT_EQ(rejected.reason_code, "LEGGED_STEP_HEIGHT_LIMIT");
}

TEST(LeggedFaultMatrix, AppliesDirectionalGapWidthInMeters) {
  const auto evaluate = [](const std::size_t unknown_columns) {
    auto input = test::MakeValidLeggedInput();
    input.world.local_map = test::MakeFlatMap("odom", 30U, 20U, 0.1);
    for (std::size_t x = 10U; x < 10U + unknown_columns; ++x) {
      for (std::size_t y = 0U; y < input.world.local_map.height; ++y) {
        SetByte(input.world.local_map, "valid_mask", x, y, 0U);
      }
    }
    const auto capability = std::get<LeggedCapability>(input.capability);
    const auto snapshot = shared::MapSnapshot::Create(input.world.local_map);
    const auto projection = shared::BuildSafeProjection(
        snapshot.snapshot, input.capability, input.config.map_safety, {});
    EXPECT_TRUE(projection.ok()) << projection.reason_code;
    return legged::ValidateLeggedBodySweep(
        {{0.85, 1.05, 0.5}, 0.0},
        {{1.55, 1.05, 0.5}, 0.0},
        {.lower = 0.5, .upper = 0.5}, *projection.projection,
        capability, {});
  };

  const auto at_limit = evaluate(3U);
  EXPECT_TRUE(at_limit.valid) << at_limit.reason_code;
  const auto too_wide = evaluate(4U);
  EXPECT_FALSE(too_wide.valid);
  EXPECT_EQ(too_wide.reason_code, "LEGGED_GAP_WIDTH_LIMIT");
}

TEST(LeggedFaultMatrix, UsesMetricDiagonalGapLength) {
  const auto evaluate = [](const std::size_t diagonal_cells) {
    auto input = test::MakeValidLeggedInput();
    input.world.local_map = test::MakeFlatMap("odom", 30U, 30U, 0.1);
    for (std::size_t offset = 0U; offset < diagonal_cells; ++offset) {
      SetByte(input.world.local_map, "valid_mask", 10U + offset,
              10U + offset, 0U);
    }
    const auto capability = std::get<LeggedCapability>(input.capability);
    const auto snapshot = shared::MapSnapshot::Create(input.world.local_map);
    const auto projection = shared::BuildSafeProjection(
        snapshot.snapshot, input.capability, input.config.map_safety, {});
    EXPECT_TRUE(projection.ok()) << projection.reason_code;
    return legged::ValidateLeggedBodySweep(
        {{0.85, 0.85, 0.5}, 0.7853981633974483},
        {{1.55, 1.55, 0.5}, 0.7853981633974483},
        {.lower = 0.5, .upper = 0.5}, *projection.projection,
        capability, {});
  };

  EXPECT_TRUE(evaluate(2U).valid);
  const auto too_wide = evaluate(3U);
  EXPECT_FALSE(too_wide.valid);
  EXPECT_EQ(too_wide.reason_code, "LEGGED_GAP_WIDTH_LIMIT");
}

TEST(LeggedFaultMatrix, OrientedRectangleDoesNotUseRotatedAabb) {
  auto input = test::MakeValidLeggedInput();
  input.world.local_map = test::MakeFlatMap("odom", 40U, 40U, 0.1);
  SetByte(input.world.local_map, "obstacle", 22U, 22U, 1U);
  const auto capability = std::get<LeggedCapability>(input.capability);
  const auto snapshot = shared::MapSnapshot::Create(input.world.local_map);
  const auto projection = shared::BuildSafeProjection(
      snapshot.snapshot, input.capability, input.config.map_safety, {});
  ASSERT_TRUE(projection.ok()) << projection.reason_code;

  const auto result = legged::ValidateLeggedBodySweep(
      {{1.5, 1.5, 0.5}, 0.7853981633974483},
      {{1.6, 1.5, 0.5}, 0.7853981633974483},
      {.lower = 0.5, .upper = 0.5}, *projection.projection,
      capability, {});

  EXPECT_TRUE(result.valid) << result.reason_code;
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

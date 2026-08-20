#include <cstdint>
#include <stop_token>
#include <type_traits>
#include <utility>
#include <variant>

#include <gtest/gtest.h>

#include "lunar_planner_core/planner.hpp"
#include "lunar_planner_core/traversability_projection.hpp"
#include "test_fixtures.hpp"

namespace lunar::planning {
namespace {

TEST(PlannerDefaults, GroundPlatformsUseSixtyFourYawBins) {
  EXPECT_EQ(WheelPlannerConfig{}.yaw_bin_count, 64U);
  EXPECT_EQ(LeggedPlannerConfig{}.yaw_bin_count, 64U);
  EXPECT_DOUBLE_EQ(WheelPlannerConfig{}.xy_resolution_m, 0.2);
  EXPECT_DOUBLE_EQ(LeggedPlannerConfig{}.xy_resolution_m, 0.2);
}

TEST(PublicApi, RejectsMissingRequiredMapLayersWithoutCallingBackend) {
  Planner planner;
  auto input = test::MakeValidWheelInput();
  input.world.local_map.layers.erase("forbidden");

  const auto output = planner.Plan(input);

  EXPECT_EQ(output.outcome, PlanningOutcome::kInvalidRequest);
  EXPECT_EQ(output.directive, ExecutionDirective::kNoSafeReference);
  EXPECT_EQ(output.reason_code, "MISSING_MAP_LAYER_FORBIDDEN");
  EXPECT_FALSE(output.reference.has_value());
}

TEST(PublicApi, KeepsFrozenWireFacingEnumValues) {
  EXPECT_EQ(static_cast<std::uint8_t>(PlatformType::kWheeled), 1U);
  EXPECT_EQ(static_cast<std::uint8_t>(PlatformType::kLegged), 2U);
  EXPECT_EQ(static_cast<std::uint8_t>(PlatformType::kHopper), 3U);
  EXPECT_EQ(static_cast<std::uint8_t>(PlanningOutcome::kNewReferenceAvailable), 0U);
  EXPECT_EQ(static_cast<std::uint8_t>(PlanningOutcome::kCanceled), 9U);
  EXPECT_EQ(static_cast<std::uint8_t>(ExecutionDirective::kActivateNewReference), 0U);
  EXPECT_EQ(static_cast<std::uint8_t>(ExecutionDirective::kNoSafeReference), 4U);
}

TEST(PublicApi, ExposesStableBoundedSmoothingAndDiagnosticContracts) {
  const OptimizationConfig config;
  EXPECT_EQ(config.maximum_smoothing_control_points, 64U);
  EXPECT_EQ(config.maximum_smoothing_samples, 512U);
  EXPECT_EQ(config.maximum_iterations, 128U);
  EXPECT_EQ(config.maximum_trust_region_reductions, 8U);
  EXPECT_FALSE(config.require_smoothed_execution);

  EXPECT_EQ(ToString(TrajectoryMode::kStationary), "STATIONARY");
  EXPECT_EQ(ToString(TrajectoryMode::kOptimized), "OPTIMIZED");
  EXPECT_EQ(ToString(TrajectoryMode::kDiscreteFallback),
            "DISCRETE_FALLBACK");
  EXPECT_EQ(ToString(TrajectoryMode::kCertifiedHop), "CERTIFIED_HOP");
  EXPECT_EQ(ToString(CollisionValidation::kCertified), "CERTIFIED");
  EXPECT_EQ(ToString(CollisionValidation::kNotApplicable), "NOT_APPLICABLE");

  static_assert(std::is_copy_constructible_v<LocalTrajectoryDiagnostics>);
  static_assert(std::is_move_constructible_v<LocalTrajectoryDiagnostics>);
  static_assert(std::is_copy_assignable_v<LocalTrajectoryDiagnostics>);
  static_assert(std::is_move_assignable_v<LocalTrajectoryDiagnostics>);
}

TEST(PublicApi, RetainsTypedContiguousRowMajorLayers) {
  const auto input = test::MakeValidWheelInput();
  const GridLayer& forbidden = input.world.local_map.layers.at("forbidden");
  const auto* bytes = std::get_if<std::vector<std::uint8_t>>(&forbidden.values);

  ASSERT_NE(bytes, nullptr);
  EXPECT_EQ(bytes->size(), input.world.local_map.CellCount());
  EXPECT_TRUE(input.world.local_map.HasConsistentLayerSizes());
}

TEST(PublicApi, IsMoveOnlyAndAcceptsCooperativeCancellation) {
  static_assert(std::is_move_constructible_v<Planner>);
  static_assert(std::is_move_assignable_v<Planner>);
  static_assert(!std::is_copy_constructible_v<Planner>);
  static_assert(!std::is_copy_assignable_v<Planner>);

  Planner source;
  Planner moved{std::move(source)};
  auto input = test::MakeValidWheelInput();
  std::stop_source stop_source;
  stop_source.request_stop();
  input.stop_token = stop_source.get_token();

  const auto output = moved.Plan(input);

  EXPECT_EQ(output.outcome, PlanningOutcome::kCanceled);
  EXPECT_EQ(output.directive, ExecutionDirective::kHoldPosition);
  EXPECT_EQ(output.reason_code, "REQUEST_CANCELED");
  EXPECT_FALSE(output.reference.has_value());
}

}  // namespace
}  // namespace lunar::planning

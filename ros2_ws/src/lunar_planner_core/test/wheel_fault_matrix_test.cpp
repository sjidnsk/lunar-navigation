#include <cstddef>
#include <cstdint>
#include <numbers>
#include <stop_token>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "lunar_planner_core/planner.hpp"
#include "shared/map_snapshot.hpp"
#include "shared/safe_projection.hpp"
#include "test_fixtures.hpp"
#include "wheel/wheel_sweep_validator.hpp"
#include "wheel/wheel_types.hpp"

namespace lunar::planning {
namespace {

void SetObstacle(GridMap& map, const std::size_t x, const std::size_t y) {
  auto& obstacle = std::get<std::vector<std::uint8_t>>(
      map.layers.at("obstacle").values);
  obstacle.at(y * map.width + x) = 1U;
}

TEST(WheelFaultMatrix, EnforcesCurvatureLimitBeforeSweep) {
  auto input = test::MakeValidWheelInput();
  auto capability = std::get<WheeledCapability>(input.capability);
  capability.maximum_curvature_per_m = 0.1;
  const auto snapshot = shared::MapSnapshot::Create(input.world.local_map);
  ASSERT_TRUE(snapshot.ok()) << snapshot.reason_code;
  const auto projection = shared::BuildSafeProjection(
      snapshot.snapshot, PlatformCapability{capability},
      input.config.map_safety, {});
  ASSERT_TRUE(projection.ok()) << projection.reason_code;
  const wheel::WheelSweepValidator validator{
      *projection.projection, capability,
      input.config.wheel.continuous_validation_maximum_subdivisions};
  const wheel::WheelTransition transition{
      .source_pose = wheel::WheelPose{.position_m = {2.5, 3.5, 0.0}},
      .target_pose = wheel::WheelPose{
          .position_m = {3.5, 3.5, 0.0},
          .yaw_rad = 0.5,
      },
      .curvature_per_m = 0.5,
  };

  const auto result = validator.Validate(transition, {});

  EXPECT_FALSE(result.valid);
  EXPECT_EQ(result.reason_code, "WHEEL_CURVATURE_LIMIT");
}

TEST(WheelFaultMatrix, RejectsObstacleIntersectingContinuousFootprintSweep) {
  auto input = test::MakeValidWheelInput();
  SetObstacle(input.world.local_map, 3U, 3U);
  const auto capability = std::get<WheeledCapability>(input.capability);
  const auto snapshot = shared::MapSnapshot::Create(input.world.local_map);
  ASSERT_TRUE(snapshot.ok()) << snapshot.reason_code;
  const auto projection = shared::BuildSafeProjection(
      snapshot.snapshot, input.capability, input.config.map_safety, {});
  ASSERT_TRUE(projection.ok()) << projection.reason_code;
  const wheel::WheelSweepValidator validator{
      *projection.projection, capability,
      input.config.wheel.continuous_validation_maximum_subdivisions};
  const wheel::WheelTransition transition{
      .source_pose = wheel::WheelPose{.position_m = {2.5, 3.5, 0.0}},
      .target_pose = wheel::WheelPose{.position_m = {4.5, 3.5, 0.0}},
  };

  const auto result = validator.Validate(transition, {});

  EXPECT_FALSE(result.valid);
  EXPECT_EQ(result.reason_code, "WHEEL_SWEEP_COLLISION");
}

TEST(WheelFaultMatrix, IgnoresUnsafeCellsOutsideTheRotatedFootprintPolygon) {
  auto input = test::MakeValidWheelInput();
  SetObstacle(input.world.local_map, 4U, 4U);
  auto capability = std::get<WheeledCapability>(input.capability);
  capability.footprint_xy_m = {
      {-0.6, -0.2}, {0.6, -0.2}, {0.6, 0.2}, {-0.6, 0.2}};
  const auto snapshot = shared::MapSnapshot::Create(input.world.local_map);
  ASSERT_TRUE(snapshot.ok()) << snapshot.reason_code;
  const auto projection = shared::BuildSafeProjection(
      snapshot.snapshot, PlatformCapability{capability},
      input.config.map_safety, {});
  ASSERT_TRUE(projection.ok()) << projection.reason_code;
  const wheel::WheelSweepValidator validator{
      *projection.projection, capability,
      input.config.wheel.continuous_validation_maximum_subdivisions};
  const wheel::WheelTransition transition{
      .source_pose = wheel::WheelPose{
          .position_m = {3.5, 3.5, 0.0},
          .yaw_rad = std::numbers::pi / 4.0,
      },
      .target_pose = wheel::WheelPose{
          .position_m = {3.5, 3.5, 0.0},
          .yaw_rad = std::numbers::pi / 4.0,
      },
  };

  const auto result = validator.Validate(transition, {});

  EXPECT_TRUE(result.valid) << result.reason_code;
  EXPECT_EQ(result.reason_code, "WHEEL_SWEEP_VALID");
}

TEST(WheelFaultMatrix, ReportsNoPathAcrossFullBarrier) {
  Planner planner;
  auto input = test::MakeValidWheelInput();
  input.request_id = "wheel-blocked";
  for (std::size_t y = 0U; y < input.world.local_map.height; ++y) {
    SetObstacle(input.world.local_map, 3U, y);
  }

  const PlannerOutput output = planner.Plan(input);

  EXPECT_EQ(output.outcome, PlanningOutcome::kNoKnownSafeRoute);
  EXPECT_EQ(output.directive, ExecutionDirective::kNoSafeReference);
  EXPECT_FALSE(output.reference.has_value());
  EXPECT_EQ(output.diagnostics.planner_name, "cpp_v3_hierarchical");
}

TEST(WheelFaultMatrix, RejectsObstacleIntersectingOnlyTheTrueStartConnector) {
  Planner planner;
  auto input = test::MakeValidWheelInput();
  input.request_id = "wheel-start-connector-blocked";
  auto& state = std::get<WheeledState>(input.current_state);
  state.pose.position_m = {2.15, 3.5, 0.0};
  SetObstacle(input.world.local_map, 1U, 3U);

  const PlannerOutput output = planner.Plan(input);

  EXPECT_EQ(output.outcome, PlanningOutcome::kNoKnownSafeRoute);
  EXPECT_EQ(output.directive, ExecutionDirective::kNoSafeReference);
  EXPECT_EQ(output.reason_code, "WHEEL_START_CONNECTOR_INFEASIBLE");
  EXPECT_FALSE(output.reference.has_value());
}

TEST(WheelFaultMatrix, CancelsBeforeSearchExpansion) {
  Planner planner;
  auto input = test::MakeValidWheelInput();
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

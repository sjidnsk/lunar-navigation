#include <cstddef>
#include <cstdint>
#include <cmath>
#include <functional>
#include <numbers>
#include <ranges>
#include <stop_token>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "lunar_planner_core/planner.hpp"
#include "shared/map_snapshot.hpp"
#include "shared/safe_projection.hpp"
#include "shared/terrain_checks.hpp"
#include "test_fixtures.hpp"
#include "wheel/wheel_sweep_validator.hpp"
#include "wheel/wheel_types.hpp"

namespace lunar::planning {
namespace {

template <class T>
concept HasNominalDuration = requires(T value) {
  value.nominal_duration;
};

static_assert(!HasNominalDuration<WheelMotionPrimitive>);
static_assert(!HasNominalDuration<wheel::WheelTransition>);

void SetObstacle(GridMap& map, const std::size_t x, const std::size_t y) {
  auto& obstacle = std::get<std::vector<std::uint8_t>>(
      map.layers.at("obstacle").values);
  obstacle.at(y * map.width + x) = 1U;
}

void FillElevation(GridMap& map,
                   const std::function<double(double, double)>& surface) {
  auto& elevation = std::get<std::vector<float>>(
      map.layers.at("elevation").values);
  for (std::size_t y = 0U; y < map.height; ++y) {
    for (std::size_t x = 0U; x < map.width; ++x) {
      const double px = map.origin_m.x +
          (static_cast<double>(x) + 0.5) * map.resolution_m;
      const double py = map.origin_m.y +
          (static_cast<double>(y) + 0.5) * map.resolution_m;
      elevation[y * map.width + x] =
          static_cast<float>(surface(px, py));
    }
  }
}

bool HasReason(const shared::WheelTerrainPoseEvaluation& evaluation,
               const std::string& reason) {
  return std::ranges::find(evaluation.rejection_codes, reason) !=
      evaluation.rejection_codes.end();
}

PlannerInput MakeWheelInputWithRequiredLocalCoverage() {
  PlannerInput input = test::MakeValidWheelInput();
  input.world.local_map.origin_m.x = -1.0;
  return input;
}

TEST(WheelFaultMatrix, FitsFourWheelSupportPlaneAtArbitrarySlopeBoundary) {
  auto input = test::MakeValidWheelInput();
  auto capability = std::get<WheeledCapability>(input.capability);
  capability.footprint_xy_m = {
      {-0.591, -0.409}, {0.591, -0.409},
      {0.591, 0.409}, {-0.591, 0.409}};
  capability.maximum_slope_rad = std::numbers::pi / 9.0;
  GridMap map = test::MakeFlatMap("odom", 120U, 120U, 0.05);
  constexpr double direction = 0.63;
  FillElevation(map, [&](const double x, const double y) {
    return std::tan(capability.maximum_slope_rad) *
        (std::cos(direction) * x + std::sin(direction) * y);
  });
  const auto snapshot = shared::MapSnapshot::Create(map);
  ASSERT_TRUE(snapshot.ok()) << snapshot.reason_code;

  const auto boundary = shared::EvaluateWheelTerrainPose(
      *snapshot.snapshot, Vec2{3.0, 3.0}, 0.41, capability);

  EXPECT_TRUE(boundary.feasible);
  EXPECT_NEAR(boundary.surface_slope_rad,
              capability.maximum_slope_rad, 2.0e-6);
  EXPECT_NEAR(boundary.roughness_m, 0.0, 2.0e-6);

  FillElevation(map, [&](const double x, const double y) {
    return std::tan(capability.maximum_slope_rad + 0.01) *
        (std::cos(direction) * x + std::sin(direction) * y);
  });
  const auto steeper_snapshot = shared::MapSnapshot::Create(map);
  ASSERT_TRUE(steeper_snapshot.ok()) << steeper_snapshot.reason_code;
  const auto rejected = shared::EvaluateWheelTerrainPose(
      *steeper_snapshot.snapshot, Vec2{3.0, 3.0}, 0.41, capability);
  EXPECT_FALSE(rejected.feasible);
  EXPECT_TRUE(HasReason(rejected, "WHEEL_SLOPE_LIMIT"));
}

TEST(WheelFaultMatrix, SeparatesPhysicalRoughnessFromMapUncertainty) {
  auto input = test::MakeValidWheelInput();
  auto capability = std::get<WheeledCapability>(input.capability);
  capability.footprint_xy_m = {
      {-0.591, -0.409}, {0.591, -0.409},
      {0.591, 0.409}, {-0.591, 0.409}};
  GridMap map = test::MakeFlatMap("odom", 120U, 120U, 0.05);
  FillElevation(map, [](const double x, const double y) {
    return 0.008 * std::sin(5.0 * x) * std::cos(4.0 * y);
  });
  const auto snapshot = shared::MapSnapshot::Create(map);
  ASSERT_TRUE(snapshot.ok()) << snapshot.reason_code;

  const auto rough = shared::EvaluateWheelTerrainPose(
      *snapshot.snapshot, Vec2{3.0, 3.0}, 0.2, capability);

  EXPECT_TRUE(rough.feasible);
  EXPECT_GT(rough.roughness_m, 1.0e-4);

  auto& variance = std::get<std::vector<float>>(
      map.layers.at("elevation_variance").values);
  const std::size_t center_index = 60U * map.width + 60U;
  variance[center_index] = 0.25F;
  const auto uncertain_snapshot = shared::MapSnapshot::Create(map);
  ASSERT_TRUE(uncertain_snapshot.ok()) << uncertain_snapshot.reason_code;
  const auto projection = shared::BuildSafeProjection(
      uncertain_snapshot.snapshot, PlatformCapability{capability},
      input.config.map_safety, {});
  ASSERT_TRUE(projection.ok()) << projection.reason_code;
  EXPECT_FALSE(projection.projection->IntrinsicFeasible({60, 60}));
}

TEST(WheelFaultMatrix, RejectsUnsupportedReliefDiscontinuityAndUnderbodyStrike) {
  auto input = test::MakeValidWheelInput();
  auto capability = std::get<WheeledCapability>(input.capability);
  capability.footprint_xy_m = {
      {-0.591, -0.409}, {0.591, -0.409},
      {0.591, 0.409}, {-0.591, 0.409}};
  capability.maximum_slope_rad = 1.2;
  GridMap map = test::MakeFlatMap("odom", 80U, 80U, 0.05);
  const Vec2 center{2.0, 2.0};

  auto& valid = std::get<std::vector<std::uint8_t>>(
      map.layers.at("valid_mask").values);
  valid[40U * map.width + 40U] = 0U;
  auto unsupported_snapshot = shared::MapSnapshot::Create(map);
  ASSERT_TRUE(unsupported_snapshot.ok()) << unsupported_snapshot.reason_code;
  const auto unsupported = shared::EvaluateWheelTerrainPose(
      *unsupported_snapshot.snapshot, center, 0.0, capability);
  EXPECT_FALSE(unsupported.feasible);
  EXPECT_TRUE(HasReason(unsupported, "WHEEL_UNSUPPORTED_GAP"));

  valid[40U * map.width + 40U] = 1U;
  auto& elevation = std::get<std::vector<float>>(
      map.layers.at("elevation").values);
  elevation[40U * map.width + 40U] = 0.205F;
  auto discontinuity_snapshot = shared::MapSnapshot::Create(map);
  ASSERT_TRUE(discontinuity_snapshot.ok())
      << discontinuity_snapshot.reason_code;
  const auto discontinuity = shared::EvaluateWheelTerrainPose(
      *discontinuity_snapshot.snapshot, center, 0.0, capability);
  EXPECT_FALSE(discontinuity.feasible);
  EXPECT_TRUE(HasReason(discontinuity, "WHEEL_SURFACE_DISCONTINUITY"));
  EXPECT_TRUE(HasReason(discontinuity, "WHEEL_LOCAL_RELIEF_LIMIT"));

  capability.maximum_local_obstacle_relief_m = 1.0;
  elevation[40U * map.width + 40U] = 0.22F;
  auto strike_snapshot = shared::MapSnapshot::Create(map);
  ASSERT_TRUE(strike_snapshot.ok()) << strike_snapshot.reason_code;
  const auto strike = shared::EvaluateWheelTerrainPose(
      *strike_snapshot.snapshot, center, 0.0, capability);
  EXPECT_FALSE(strike.feasible);
  EXPECT_TRUE(HasReason(strike, "WHEEL_UNDERBODY_CLEARANCE_LIMIT"));
}

TEST(WheelFaultMatrix, AcceptsReliefAtExactTwentyCentimeterBoundary) {
  auto input = test::MakeValidWheelInput();
  auto capability = std::get<WheeledCapability>(input.capability);
  capability.maximum_slope_rad = 1.56;
  GridMap map = test::MakeFlatMap("odom", 80U, 80U, 0.05);
  auto& elevation = std::get<std::vector<float>>(
      map.layers.at("elevation").values);
  elevation[40U * map.width + 40U] = 0.20F;
  auto snapshot = shared::MapSnapshot::Create(map);
  ASSERT_TRUE(snapshot.ok()) << snapshot.reason_code;

  const auto boundary = shared::EvaluateWheelTerrainPose(
      *snapshot.snapshot, Vec2{2.0, 2.0}, 0.0, capability);

  EXPECT_TRUE(boundary.feasible);
  EXPECT_NEAR(boundary.maximum_positive_relief_m, 0.20, 1.0e-6);
  EXPECT_NEAR(boundary.minimum_underbody_clearance_m, 0.01, 1.0e-6);

  elevation[40U * map.width + 40U] = 0.201F;
  snapshot = shared::MapSnapshot::Create(map);
  ASSERT_TRUE(snapshot.ok()) << snapshot.reason_code;
  const auto over_limit = shared::EvaluateWheelTerrainPose(
      *snapshot.snapshot, Vec2{2.0, 2.0}, 0.0, capability);
  EXPECT_FALSE(over_limit.feasible);
  EXPECT_TRUE(HasReason(over_limit, "WHEEL_LOCAL_RELIEF_LIMIT"));
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
      *projection.projection, capability};
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
      *projection.projection, capability};
  const wheel::WheelTransition transition{
      .source_pose = wheel::WheelPose{.position_m = {2.5, 3.5, 0.0}},
      .target_pose = wheel::WheelPose{.position_m = {4.5, 3.5, 0.0}},
  };

  const auto result = validator.Validate(transition, {});

  EXPECT_FALSE(result.valid);
  EXPECT_EQ(result.reason_code, "WHEEL_SWEEP_COLLISION");
}

TEST(WheelFaultMatrix, SweepMemoizesExactTerrainPosesPerValidator) {
  auto input = test::MakeValidWheelInput();
  const auto capability = std::get<WheeledCapability>(input.capability);
  const auto snapshot = shared::MapSnapshot::Create(input.world.local_map);
  ASSERT_TRUE(snapshot.ok()) << snapshot.reason_code;
  const auto projection = shared::BuildSafeProjection(
      snapshot.snapshot, input.capability, input.config.map_safety, {});
  ASSERT_TRUE(projection.ok()) << projection.reason_code;
  const wheel::WheelSweepValidator validator{
      *projection.projection, capability};
  const wheel::WheelTransition transition{
      .source_pose = wheel::WheelPose{.position_m = {2.5, 3.5, 0.0}},
      .target_pose = wheel::WheelPose{.position_m = {4.5, 3.5, 0.0}},
  };

  const auto first = validator.Validate(transition, {});
  const std::size_t first_count = validator.terrain_evaluation_count();
  const auto second = validator.Validate(transition, {});

  EXPECT_GT(first_count, 0U);
  EXPECT_EQ(validator.terrain_evaluation_count(), first_count);
  EXPECT_EQ(second.valid, first.valid);
  EXPECT_EQ(second.canceled, first.canceled);
  EXPECT_EQ(second.sample_count, first.sample_count);
  EXPECT_EQ(second.maximum_surface_slope_rad,
            first.maximum_surface_slope_rad);
  EXPECT_EQ(second.maximum_roughness_m, first.maximum_roughness_m);
  EXPECT_EQ(second.maximum_positive_relief_m,
            first.maximum_positive_relief_m);
  EXPECT_EQ(second.minimum_underbody_clearance_m,
            first.minimum_underbody_clearance_m);
  EXPECT_EQ(second.reason_code, first.reason_code);
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
      *projection.projection, capability};
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

TEST(WheelFaultMatrix, DerivesLongSweepSamplingWithoutAFixedCeiling) {
  auto input = test::MakeValidWheelInput();
  input.world.local_map = test::MakeFlatMap("odom", 200U, 100U, 0.05);
  const auto capability = std::get<WheeledCapability>(input.capability);
  const auto snapshot = shared::MapSnapshot::Create(input.world.local_map);
  ASSERT_TRUE(snapshot.ok()) << snapshot.reason_code;
  const auto projection = shared::BuildSafeProjection(
      snapshot.snapshot, input.capability, input.config.map_safety, {});
  ASSERT_TRUE(projection.ok()) << projection.reason_code;
  const wheel::WheelSweepValidator validator{
      *projection.projection, capability};
  const wheel::WheelTransition transition{
      .source_pose = wheel::WheelPose{.position_m = {1.0, 2.5, 0.0}},
      .target_pose = wheel::WheelPose{.position_m = {5.0, 2.5, 0.0}},
  };

  const auto result = validator.Validate(transition, {});

  EXPECT_TRUE(result.valid) << result.reason_code;
  EXPECT_GT(result.sample_count, 32U);
}

TEST(WheelFaultMatrix, RejectsAForbiddenContinuousCenterBetweenAllowedEndpoints) {
  auto input = test::MakeValidWheelInput();
  const auto capability = std::get<WheeledCapability>(input.capability);
  const auto snapshot = shared::MapSnapshot::Create(input.world.local_map);
  ASSERT_TRUE(snapshot.ok()) << snapshot.reason_code;
  const auto projection = shared::BuildSafeProjection(
      snapshot.snapshot, input.capability, input.config.map_safety, {});
  ASSERT_TRUE(projection.ok()) << projection.reason_code;
  auto allowed = std::vector<std::uint8_t>(
      input.world.local_map.CellCount(), 1U);
  allowed[3U * input.world.local_map.width + 3U] = 0U;
  const hierarchical::LocalSearchDomain search_domain{
      input.world.local_map.width, input.world.local_map.height,
      std::move(allowed)};
  const wheel::WheelSweepValidator validator{
      *projection.projection, capability, search_domain};
  const wheel::WheelTransition transition{
      .source_pose = wheel::WheelPose{.position_m = {2.5, 3.5, 0.0}},
      .target_pose = wheel::WheelPose{.position_m = {4.5, 3.5, 0.0}},
      .primitive_kind = WheelPrimitiveKind::kForward,
      .source_mode = wheel::WheelMotionMode::kStart,
      .target_mode = wheel::WheelMotionMode::kForward,
      .path_length_m = 2.0,
  };

  const auto result = validator.Validate(transition, {});

  EXPECT_FALSE(result.valid);
  EXPECT_GT(result.sample_count, 1U);
  EXPECT_EQ(result.reason_code, "WHEEL_SWEEP_OUTSIDE_SEARCH_DOMAIN");
}

TEST(WheelFaultMatrix, ReportsNoPathAcrossFullBarrier) {
  Planner planner;
  auto input = MakeWheelInputWithRequiredLocalCoverage();
  input.request_id = "wheel-blocked";
  for (std::size_t y = 0U; y < input.world.local_map.height; ++y) {
    SetObstacle(input.world.local_map, 4U, y);
  }

  const PlannerOutput output = planner.Plan(input);

  EXPECT_EQ(output.outcome, PlanningOutcome::kNoKnownSafeRoute);
  EXPECT_EQ(output.directive, ExecutionDirective::kNoSafeReference);
  EXPECT_FALSE(output.reference.has_value());
  EXPECT_EQ(output.diagnostics.planner_name, "cpp_v3_hierarchical");
}

TEST(WheelFaultMatrix, RejectsObstacleIntersectingOnlyTheTrueStartConnector) {
  Planner planner;
  auto input = MakeWheelInputWithRequiredLocalCoverage();
  input.request_id = "wheel-start-connector-blocked";
  auto& state = std::get<WheeledState>(input.current_state);
  state.pose.position_m = {2.15, 3.5, 0.0};
  SetObstacle(input.world.local_map, 2U, 3U);

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

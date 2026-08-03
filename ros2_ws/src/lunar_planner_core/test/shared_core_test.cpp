#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <stop_token>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "shared/bounded_qp_solver.hpp"
#include "shared/convex_corridor.hpp"
#include "shared/map_snapshot.hpp"
#include "shared/safe_projection.hpp"
#include "shared/terrain_checks.hpp"
#include "test_fixtures.hpp"

namespace lunar::planning::shared {
namespace {

void SetFloat(
    GridMap& map, const std::string& layer, const std::size_t index,
    const float value) {
  std::get<std::vector<float>>(map.layers.at(layer).values).at(index) = value;
}

void SetByte(
    GridMap& map, const std::string& layer, const std::size_t index,
    const std::uint8_t value) {
  std::get<std::vector<std::uint8_t>>(map.layers.at(layer).values).at(index) =
      value;
}

void SetCount(
    GridMap& map, const std::string& layer, const std::size_t index,
    const std::uint32_t value) {
  std::get<std::vector<std::uint32_t>>(map.layers.at(layer).values).at(index) =
      value;
}

std::shared_ptr<const MapSnapshot> MakeSnapshot(const GridMap& map) {
  const auto result = MapSnapshot::Create(map);
  EXPECT_TRUE(result.ok()) << result.reason_code;
  return result.snapshot;
}

TEST(MapSnapshot, RejectsMissingLayerWrongTypeAndNan) {
  auto missing = test::MakeFlatMap("odom", 3U, 2U);
  missing.layers.erase("forbidden");
  const auto missing_result = MapSnapshot::Create(missing);
  EXPECT_FALSE(missing_result.ok());
  EXPECT_EQ(missing_result.reason_code, "MISSING_MAP_LAYER_FORBIDDEN");

  auto wrong_type = test::MakeFlatMap("odom", 3U, 2U);
  wrong_type.layers.at("observation_count") = test::MakeFloatLayer(6U, 1.0F);
  const auto type_result = MapSnapshot::Create(wrong_type);
  EXPECT_FALSE(type_result.ok());
  EXPECT_EQ(type_result.reason_code, "MAP_LAYER_TYPE_OBSERVATION_COUNT");

  auto nan_map = test::MakeFlatMap("odom", 3U, 2U);
  SetFloat(
      nan_map, "elevation", 4U, std::numeric_limits<float>::quiet_NaN());
  const auto nan_result = MapSnapshot::Create(nan_map);
  EXPECT_FALSE(nan_result.ok());
  EXPECT_EQ(nan_result.reason_code, "MAP_NONFINITE_ELEVATION");
}

TEST(MapSnapshot, PreservesUnwrappedRowMajorTypedLayers) {
  auto map = test::MakeFlatMap("odom", 3U, 2U);
  auto& elevation =
      std::get<std::vector<float>>(map.layers.at("elevation").values);
  elevation = {0.0F, 1.0F, 2.0F, 10.0F, 11.0F, 12.0F};

  const auto snapshot = MakeSnapshot(map);

  ASSERT_NE(snapshot, nullptr);
  EXPECT_EQ(snapshot->Index(GridCell{.x = 1, .y = 1}), 4U);
  EXPECT_FLOAT_EQ(snapshot->FloatLayer("elevation")[4], 11.0F);
  EXPECT_EQ(snapshot->ByteLayer("valid_mask")[5], 1U);
  EXPECT_EQ(snapshot->CountLayer("observation_count")[0], 1U);
}

TEST(SafeProjection, RejectsUnknownForbiddenObstacleAndQualityFaults) {
  auto map = test::MakeFlatMap("odom", 7U, 5U, 1.0);
  constexpr std::size_t kUnknown = 8U;
  constexpr std::size_t kForbidden = 9U;
  constexpr std::size_t kObstacle = 10U;
  constexpr std::size_t kElevationVariance = 11U;
  constexpr std::size_t kObstacleVariance = 12U;
  constexpr std::size_t kOld = 15U;
  constexpr std::size_t kLowQuality = 16U;
  constexpr std::size_t kNoObservations = 17U;
  SetByte(map, "valid_mask", kUnknown, 0U);
  SetByte(map, "forbidden", kForbidden, 1U);
  SetByte(map, "obstacle", kObstacle, 1U);
  SetFloat(map, "elevation_variance", kElevationVariance, 0.05F);
  SetFloat(map, "obstacle_variance", kObstacleVariance, 0.05F);
  SetFloat(map, "observation_age_s", kOld, 2.1F);
  SetFloat(map, "observation_quality", kLowQuality, 0.7F);
  SetCount(map, "observation_count", kNoObservations, 0U);

  const auto input = test::MakeValidWheelInput();
  const auto result = BuildSafeProjection(
      MakeSnapshot(map), input.capability, input.config.map_safety, {});

  ASSERT_TRUE(result.ok()) << result.reason_code;
  const auto& projection = *result.projection;
  for (const std::size_t index : {
           kUnknown, kForbidden, kObstacle, kElevationVariance,
           kObstacleVariance, kOld, kLowQuality, kNoObservations}) {
    const GridCell cell{
        .x = static_cast<std::int32_t>(index % map.width),
        .y = static_cast<std::int32_t>(index / map.width),
    };
    EXPECT_FALSE(projection.HardFeasible(cell)) << index;
  }
  EXPECT_TRUE(projection.HardFeasible(GridCell{.x = 5, .y = 3}));
}

TEST(SafeProjection, AppliesClearanceAndMinimumSlopeLimit) {
  auto clearance_map = test::MakeFlatMap("odom", 7U, 7U, 1.0);
  SetByte(clearance_map, "obstacle", 3U * 7U + 3U, 1U);
  auto input = test::MakeValidWheelInput();
  auto wheel = std::get<WheeledCapability>(input.capability);
  wheel.minimum_clearance_m = 1.1;
  const auto clearance = BuildSafeProjection(
      MakeSnapshot(clearance_map), PlatformCapability{wheel},
      input.config.map_safety, {});
  ASSERT_TRUE(clearance.ok()) << clearance.reason_code;
  EXPECT_FALSE(clearance.projection->HardFeasible(GridCell{.x = 4, .y = 3}));
  EXPECT_TRUE(clearance.projection->HardFeasible(GridCell{.x = 5, .y = 3}));

  auto slope_map = test::MakeFlatMap("odom", 5U, 5U, 1.0);
  const double twenty_degrees = 20.0 * std::numbers::pi / 180.0;
  for (std::size_t y = 0U; y < slope_map.height; ++y) {
    for (std::size_t x = 0U; x < slope_map.width; ++x) {
      SetFloat(
          slope_map, "elevation", y * slope_map.width + x,
          static_cast<float>(static_cast<double>(x) * std::tan(twenty_degrees)));
    }
  }
  wheel.minimum_clearance_m = 0.0;
  wheel.maximum_slope_rad = 15.0 * std::numbers::pi / 180.0;
  const auto capability_limited = BuildSafeProjection(
      MakeSnapshot(slope_map), PlatformCapability{wheel},
      input.config.map_safety, {});
  ASSERT_TRUE(capability_limited.ok()) << capability_limited.reason_code;
  EXPECT_FALSE(
      capability_limited.projection->HardFeasible(GridCell{.x = 2, .y = 2}));

  wheel.maximum_slope_rad = 45.0 * std::numbers::pi / 180.0;
  auto thirty_five_map = test::MakeFlatMap("odom", 5U, 5U, 1.0);
  const double thirty_five_degrees = 35.0 * std::numbers::pi / 180.0;
  for (std::size_t y = 0U; y < thirty_five_map.height; ++y) {
    for (std::size_t x = 0U; x < thirty_five_map.width; ++x) {
      SetFloat(
          thirty_five_map, "elevation", y * thirty_five_map.width + x,
          static_cast<float>(
              static_cast<double>(x) * std::tan(thirty_five_degrees)));
    }
  }
  auto permissive_config = input.config.map_safety;
  permissive_config.project_maximum_slope_rad = std::numbers::pi / 4.0;
  const auto project_limited = BuildSafeProjection(
      MakeSnapshot(thirty_five_map), PlatformCapability{wheel},
      permissive_config, {});
  ASSERT_TRUE(project_limited.ok()) << project_limited.reason_code;
  EXPECT_NEAR(
      project_limited.projection->maximum_slope_rad(),
      std::numbers::pi / 6.0, 1.0e-12);
  EXPECT_FALSE(project_limited.projection->HardFeasible(
      GridCell{.x = 2, .y = 2}));
}

TEST(ConvexCorridor, BuildsCertifiedCellsAndHonorsCancellation) {
  const auto input = test::MakeValidWheelInput();
  const auto projection = BuildSafeProjection(
      MakeSnapshot(test::MakeFlatMap("odom", 8U, 5U, 1.0)),
      input.capability, input.config.map_safety, {});
  ASSERT_TRUE(projection.ok()) << projection.reason_code;
  const std::vector<Vec2> centerline{{1.5, 2.5}, {3.5, 2.5}, {5.5, 2.5}};
  const CorridorTightening tightening{
      .footprint_support_radius_m = 0.2,
      .tracking_error_bound_m = 0.1,
      .additional_margin_m = 0.05,
  };

  const auto corridor = BuildConvexCorridor(
      *projection.projection, centerline, tightening, input.config.corridor, {});
  ASSERT_EQ(corridor.status, CorridorStatus::kCertified)
      << corridor.reason_code;
  ASSERT_FALSE(corridor.cells.empty());
  EXPECT_EQ(corridor.fallback, CorridorFallback::kNone);
  for (const auto& cell : corridor.cells) {
    EXPECT_EQ(cell.half_planes.size(), 4U);
  }

  std::stop_source stop_source;
  stop_source.request_stop();
  const auto canceled = BuildConvexCorridor(
      *projection.projection, centerline, tightening, input.config.corridor,
      stop_source.get_token());
  EXPECT_EQ(canceled.status, CorridorStatus::kCanceled);
  EXPECT_EQ(canceled.reason_code, "REQUEST_CANCELED");
}

TEST(BoundedQpSolver, SolvesBoxConstrainedProblemAndRejectsNan) {
  const BoundedQpProblem problem{
      .dimension = 2U,
      .hessian = {2.0, 0.0, 0.0, 2.0},
      .gradient = {-4.0, 2.0},
      .lower_bounds = {-1.0, -1.0},
      .upper_bounds = {1.0, 1.0},
  };
  const BoundedQpSettings settings{
      .maximum_iterations = 256U,
      .absolute_tolerance = 1.0e-10,
  };

  const auto solution = SolveBoundedQp(problem, settings, {});

  ASSERT_EQ(solution.termination, QpTermination::kSolved)
      << solution.reason_code;
  ASSERT_EQ(solution.primal.size(), 2U);
  EXPECT_NEAR(solution.primal[0], 1.0, 1.0e-8);
  EXPECT_NEAR(solution.primal[1], -1.0, 1.0e-8);
  EXPECT_NEAR(solution.objective, -4.0, 1.0e-8);

  auto invalid = problem;
  invalid.gradient[0] = std::numeric_limits<double>::quiet_NaN();
  const auto rejected = SolveBoundedQp(invalid, settings, {});
  EXPECT_EQ(rejected.termination, QpTermination::kInvalidProblem);
  EXPECT_EQ(rejected.reason_code, "QP_NONFINITE_GRADIENT");
}

}  // namespace
}  // namespace lunar::planning::shared

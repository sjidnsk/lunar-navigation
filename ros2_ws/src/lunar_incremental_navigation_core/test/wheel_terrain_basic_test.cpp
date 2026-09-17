#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <numbers>
#include <vector>

#include "lunar_incremental_navigation_core/fine_traversability_builder.hpp"

namespace lunar::incremental_navigation {
namespace {
struct Terrain {
  GridGeometry geometry{
      .frame_id = "map", .width = 15, .height = 15, .resolution_m = .2};
  std::vector<float> z = std::vector<float>(225, 0.F);
  PersistentElevationMap map;
  float& at(int x, int y) { return z[y * 15 + x]; }
  std::shared_ptr<const ElevationSnapshot> snapshot() {
    static_cast<void>(map.Apply(
        {.geometry = geometry,
         .elevation_m = z,
         .map_from_source = {.parent_frame = "map", .child_frame = "map"}}));
    return map.Snapshot();
  }
};
WheeledCapability Capability() {
  WheeledCapability c;
  c.maximum_slope_rad = 20. * std::numbers::pi / 180.;
  c.maximum_local_obstacle_relief_m = .2;
  c.minimum_underbody_clearance_m = .21;
  return c;
}
IntrinsicTraversalEvaluation Evaluate(Terrain& t, GridIndex p = {7, 7}) {
  return MakePlatformElevationEvaluator(Capability())
      ->Evaluate(*t.snapshot(), p);
}
TEST(WheelTerrainBasic, FlatNoisyGroundDoesNotTurnPairwiseNoiseIntoSlope) {
  Terrain t;
  for (int y = 0; y < 15; ++y)
    for (int x = 0; x < 15; ++x)
      t.at(x, y) = float(.012 * std::sin(x * 3.1 + y * 1.7));
  t.at(8, 7) = .085F;
  EXPECT_EQ(Evaluate(t).state, IntrinsicCellState::kFree);
  EXPECT_LT(Evaluate(t).slope_rad, 5. * std::numbers::pi / 180.);
}
TEST(WheelTerrainBasic, SmoothDiagonalRampIsNotATwentyCentimetreStep) {
  Terrain t;
  const double gradient =
      std::tan(19.8 * std::numbers::pi / 180.) / std::sqrt(2.);
  for (int y = 0; y < 15; ++y)
    for (int x = 0; x < 15; ++x) t.at(x, y) = float((x + y) * .2 * gradient);
  const auto result = Evaluate(t);
  EXPECT_EQ(result.state, IntrinsicCellState::kFree);
  EXPECT_NEAR(result.slope_rad, 19.8 * std::numbers::pi / 180., 1e-5);
  EXPECT_LT(result.relief_m, 1e-5);
}
TEST(WheelTerrainBasic, SteepRampRemainsBlocked) {
  Terrain t;
  for (int y = 0; y < 15; ++y)
    for (int x = 0; x < 15; ++x)
      t.at(x, y) = float(x * .2 * std::tan(25. * std::numbers::pi / 180.));
  EXPECT_EQ(Evaluate(t).state, IntrinsicCellState::kBlocked);
}
TEST(WheelTerrainBasic, IsolatedTallReturnIsUnknownNotFreeOrConfirmedObstacle) {
  for (float height : {.35F, -.35F}) {
    Terrain t;
    t.at(7, 7) = height;
    EXPECT_EQ(Evaluate(t).state, IntrinsicCellState::kUnknown);
    EXPECT_EQ(Evaluate(t, {6, 7}).state, IntrinsicCellState::kUnknown);
  }
}
TEST(WheelTerrainBasic, SupportedRockAndPitRemainBlocked) {
  for (float height : {.3F, -.3F}) {
    Terrain t;
    for (int y = 7; y <= 8; ++y)
      for (int x = 7; x <= 8; ++x) t.at(x, y) = height;
    EXPECT_EQ(Evaluate(t).state, IntrinsicCellState::kBlocked);
  }
}
TEST(WheelTerrainBasic, AbruptStepMustNotBecomeAFittedRamp) {
  for (float height : {.25F, -.25F}) {
    Terrain t;
    for (int y = 0; y < 15; ++y)
      for (int x = 8; x < 15; ++x) t.at(x, y) = height;
    EXPECT_EQ(Evaluate(t).state, IntrinsicCellState::kBlocked);
    EXPECT_EQ(Evaluate(t, {8, 7}).state, IntrinsicCellState::kBlocked);
  }
}
TEST(WheelTerrainBasic, JustOverLimitStepsOnSlopesRemainBlocked) {
  for (float height : {.21F, -.21F, .25F, -.25F}) {
    for (double slope_deg : {0., 10., -10.}) {
      Terrain t;
      for (int y = 0; y < 15; ++y)
        for (int x = 0; x < 15; ++x)
          t.at(x, y) =
              float(x * .2 * std::tan(slope_deg * std::numbers::pi / 180.) +
                    (x >= 8 ? height : 0.));
      SCOPED_TRACE(std::to_string(height) + "," + std::to_string(slope_deg));
      EXPECT_EQ(Evaluate(t).state, IntrinsicCellState::kBlocked);
      EXPECT_EQ(Evaluate(t, {8, 7}).state, IntrinsicCellState::kBlocked);
    }
  }
}
TEST(WheelTerrainBasic, MissingCentreAndInsufficientSupportRemainUnknown) {
  Terrain t;
  t.at(7, 7) = std::numeric_limits<float>::quiet_NaN();
  EXPECT_EQ(Evaluate(t).state, IntrinsicCellState::kUnknown);
  Terrain sparse;
  for (auto& z : sparse.z) z = std::numeric_limits<float>::quiet_NaN();
  for (int x = 5; x <= 9; ++x) sparse.at(x, 7) = 0;
  EXPECT_EQ(Evaluate(sparse).state, IntrinsicCellState::kUnknown);
}
TEST(WheelTerrainBasic, AWellSupportedWindowCanContainAFewMissingCells) {
  Terrain t;
  t.at(8, 8) = std::numeric_limits<float>::quiet_NaN();
  EXPECT_EQ(Evaluate(t).state, IntrinsicCellState::kFree);
}
}  // namespace
}  // namespace lunar::incremental_navigation

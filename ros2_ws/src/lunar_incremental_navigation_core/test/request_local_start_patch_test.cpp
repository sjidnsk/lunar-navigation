#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <numbers>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "lunar_incremental_navigation_core/elevation_map.hpp"
#include "lunar_incremental_navigation_core/fine_traversability_builder.hpp"
#include "lunar_incremental_navigation_core/request_local_start_patch.hpp"
#include "lunar_incremental_navigation_core/wheel_local_planner.hpp"
#include "lunar_incremental_navigation_core/local_target_selector.hpp"
#include "lunar_incremental_navigation_core/types/platform_capability.hpp"

namespace lunar::incremental_navigation {
namespace {

[[nodiscard]] GridGeometry Geometry(const std::size_t width = 25U,
                                    const std::size_t height = 25U,
                                    const double resolution_m = 1.0) {
  return GridGeometry{.frame_id = "map",
                      .width = width,
                      .height = height,
                      .resolution_m = resolution_m};
}

[[nodiscard]] RigidTransform IdentityMapTransform() {
  return RigidTransform{.parent_frame = "map", .child_frame = "map"};
}

[[nodiscard]] RigidTransform RotatedMapTransform(const double yaw_rad) {
  return RigidTransform{
      .parent_frame = "map",
      .child_frame = "map",
      .rotation = Quaternion{.w = std::cos(yaw_rad / 2.0),
                             .z = std::sin(yaw_rad / 2.0)},
  };
}

[[nodiscard]] std::size_t Offset(const GridGeometry& geometry,
                                 const GridIndex index) {
  return static_cast<std::size_t>(index.y) * geometry.width +
         static_cast<std::size_t>(index.x);
}

[[nodiscard]] Pose2 PoseAt(const GridGeometry& geometry,
                           const GridIndex index) {
  return Pose2{.position_m =
                   {.x = geometry.origin_m.x +
                             (static_cast<double>(index.x) + 0.5) *
                                 geometry.resolution_m,
                    .y = geometry.origin_m.y +
                             (static_cast<double>(index.y) + 0.5) *
                                 geometry.resolution_m}};
}

[[nodiscard]] WheeledCapability WheelCapability() {
  return WheeledCapability{
      .footprint_xy_m = {{-0.1, -0.1}, {0.1, -0.1},
                         {0.1, 0.1}, {-0.1, 0.1}},
      .body_extent_m = {.x = 0.2, .y = 0.2, .z = 0.3},
      .wheel_diameter_m = 0.3,
      .wheel_width_m = 0.08,
      .wheelbase_m = 0.4,
      .track_width_m = 0.3,
      .minimum_underbody_clearance_m = 0.5,
      .maximum_local_obstacle_relief_m = 0.2,
      .allow_unsupported_gap = false,
      .minimum_body_z_m = 0.0,
      .maximum_body_z_m = 1.0,
      .maximum_forward_speed_mps = 1.0,
      .maximum_reverse_speed_mps = 0.5,
      .maximum_spin_rate_radps = 1.0,
      .maximum_acceleration_mps2 = 1.0,
      .maximum_braking_deceleration_mps2 = 1.0,
      .maximum_yaw_acceleration_radps2 = 1.0,
      .maximum_lateral_acceleration_mps2 = 1.0,
      .maximum_curvature_per_m = 2.0,
      .maximum_slope_rad = std::numbers::pi / 2.0,
      .minimum_clearance_m = 0.2,
  };
}

[[nodiscard]] LeggedCapability LeggedCapabilityForTest() {
  return LeggedCapability{
      .body_extent_m = {.x = 0.2, .y = 0.2, .z = 0.3},
      .nominal_body_height_m = 0.5,
      .platform_mass_kg = 20.0,
      .nominal_payload_kg = 2.0,
      .maximum_payload_kg = 10.0,
      .maximum_slope_rad = std::numbers::pi / 2.0,
      .maximum_step_height_m = 0.6,
      .maximum_gap_width_m = 0.2,
      .minimum_body_clearance_m = 0.3,
      .step_vertical_rate_mps = 0.2,
      .body_height_m = {.lower = 0.4, .upper = 0.6},
      .forward_speed_mps = {.lower = -0.5, .upper = 0.8},
      .lateral_speed_mps = {.lower = -0.4, .upper = 0.4},
      .yaw_rate_radps = {.lower = -0.8, .upper = 0.8},
      .maximum_linear_acceleration_mps2 = 1.0,
      .maximum_yaw_acceleration_radps2 = 1.0,
      .unknown_is_traversable = false,
  };
}

[[nodiscard]] TraversabilityProfile Profile(
    const double margin_m = 2.0,
    const TraversalCostWeights weights = {},
    const double preferred_clearance_m = 0.0) {
  return TraversabilityProfile{
      .planar_envelope_xy_m = {{-0.1, -0.1}, {0.1, -0.1},
                               {0.1, 0.1}, {-0.1, 0.1}},
      .preferred_clearance_m = preferred_clearance_m,
      .slope_weight = weights.slope,
      .relief_weight = weights.relief,
      .clearance_weight = weights.clearance,
      .start_blind_zone_margin_m = margin_m,
  };
}

struct SnapshotFixture final {
  std::shared_ptr<const ElevationSnapshot> raw;
  std::shared_ptr<const FineTraversabilitySnapshot> fine;
};

[[nodiscard]] SnapshotFixture MakeFixture(
    const GridGeometry& geometry, const std::vector<float>& elevation,
    const PlatformCapability& capability,
    const TraversabilityProfile& profile) {
  PersistentElevationMap map;
  const ElevationUpdateResult update = map.Apply(ElevationEvidence{
      .geometry = geometry,
      .elevation_m = elevation,
      .map_from_source = IdentityMapTransform(),
  });
  if (update.status != ElevationUpdateResult::Status::kApplied) {
    throw std::runtime_error("test elevation fixture was rejected");
  }
  auto raw = map.Snapshot();
  FineTraversabilityBuilder fine_builder;
  auto fine = fine_builder.Derive(raw, capability, profile);
  return {.raw = std::move(raw), .fine = std::move(fine)};
}

[[nodiscard]] std::vector<float> FlatWithMissingSquare(
    const GridGeometry& geometry, const GridIndex center,
    const std::int64_t half_extent) {
  std::vector<float> elevation(geometry.CellCount(), 0.0F);
  for (std::int64_t dy = -half_extent; dy <= half_extent; ++dy) {
    for (std::int64_t dx = -half_extent; dx <= half_extent; ++dx) {
      elevation[Offset(geometry,
                       {.x = center.x + dx, .y = center.y + dy})] =
          std::numeric_limits<float>::quiet_NaN();
    }
  }
  return elevation;
}

TEST(RequestLocalStartPatch,
     UsesExactRadiusAndCellAreaIntersectionForMissingRawCells) {
  const GridGeometry geometry = Geometry();
  const GridIndex start{.x = 12, .y = 12};
  std::vector<float> elevation = FlatWithMissingSquare(geometry, start, 1);
  elevation[Offset(geometry, {.x = 13, .y = 13})] =
      std::numeric_limits<float>::infinity();
  const TraversabilityProfile profile = Profile(2.0);
  const auto fixture =
      MakeFixture(geometry, elevation, WheelCapability(), profile);

  RequestLocalStartPatchBuilder builder;
  const StartPatchResult result = builder.Build(
      fixture.fine, PoseAt(geometry, start), WheelCapability(), profile);

  ASSERT_EQ(result.status, StartPatchResult::Status::kReady);
  ASSERT_TRUE(result.view);
  EXPECT_DOUBLE_EQ(result.view->start_patch_radius_m(),
                   std::hypot(0.1, 0.1) + 2.0);
  EXPECT_EQ(result.view->Source({.x = 13, .y = 13}),
            LocalCellSource::kStartAssumedFree);
  EXPECT_EQ(result.view->Source({.x = 15, .y = 12}),
            LocalCellSource::kEvidenceFree);
  EXPECT_GT(result.assumed_cells, 0U);
}

TEST(RequestLocalStartPatch,
     ExportsOnlyNativeEvidenceAtTheEndOfTheAssumedStartPrefix) {
  const GridGeometry geometry = Geometry();
  const GridIndex start{.x = 12, .y = 12};
  const auto fixture = MakeFixture(
      geometry, FlatWithMissingSquare(geometry, start, 1), WheelCapability(),
      Profile(2.0));

  const StartConnectionsResult connections =
      RequestLocalStartPatchBuilder().BuildStartConnections(
          fixture.fine, fixture.fine->geometry(), PoseAt(geometry, start),
          WheelCapability(), Profile(2.0));

  ASSERT_EQ(connections.status, StartPatchResult::Status::kReady);
  ASSERT_FALSE(connections.connections.empty());
  for (const StartConnection connection : connections.connections) {
    EXPECT_EQ(connection.phase, StartPhase::kNormal);
    EXPECT_EQ(fixture.fine->State(connection.index), FineCellState::kFree);
  }
}

TEST(RequestLocalStartPatch,
     FitsUniquePlaneFromThreeNonCollinearNonblockedSamples) {
  const GridGeometry geometry = Geometry();
  const GridIndex start{.x = 12, .y = 12};
  std::vector<float> elevation(
      geometry.CellCount(), std::numeric_limits<float>::quiet_NaN());
  for (const GridIndex sample : {GridIndex{.x = 10, .y = 10},
                                 GridIndex{.x = 14, .y = 10},
                                 GridIndex{.x = 10, .y = 14},
                                 GridIndex{.x = 0, .y = 0},
                                 GridIndex{.x = 24, .y = 24}}) {
    elevation[Offset(geometry, sample)] = 0.0F;
  }
  for (std::int64_t y = 10; y <= 14; ++y) {
    for (std::int64_t x = 15; x <= 18; ++x) {
      elevation[Offset(geometry, {.x = x, .y = y})] = 0.0F;
    }
  }
  const TraversabilityProfile profile = Profile(3.0);
  const auto fixture =
      MakeFixture(geometry, elevation, WheelCapability(), profile);

  const StartPatchResult result = RequestLocalStartPatchBuilder().Build(
      fixture.fine, PoseAt(geometry, start), WheelCapability(), profile);

  ASSERT_EQ(result.status, StartPatchResult::Status::kReady);
  ASSERT_TRUE(result.view);
  EXPECT_EQ(result.view->Source(start), LocalCellSource::kStartAssumedFree);
}

TEST(RequestLocalStartPatch, RejectsCollinearSupportWithoutAddingQualityGate) {
  const GridGeometry geometry = Geometry();
  const GridIndex start{.x = 12, .y = 12};
  std::vector<float> elevation(
      geometry.CellCount(), std::numeric_limits<float>::quiet_NaN());
  for (const GridIndex sample : {GridIndex{.x = 10, .y = 11},
                                 GridIndex{.x = 12, .y = 11},
                                 GridIndex{.x = 14, .y = 11},
                                 GridIndex{.x = 0, .y = 0},
                                 GridIndex{.x = 24, .y = 24}}) {
    elevation[Offset(geometry, sample)] = 0.0F;
  }
  const TraversabilityProfile profile = Profile(2.0);
  const auto fixture =
      MakeFixture(geometry, elevation, WheelCapability(), profile);

  const StartPatchResult result = RequestLocalStartPatchBuilder().Build(
      fixture.fine, PoseAt(geometry, start), WheelCapability(), profile);

  EXPECT_EQ(result.status, StartPatchResult::Status::kUnresolved);
  EXPECT_FALSE(result.view);
  EXPECT_EQ(result.assumed_cells, 0U);
}

TEST(RequestLocalStartPatch,
     ReportsUnresolvedWhenAssumedPrefixCannotReachBaseEvidence) {
  const GridGeometry geometry = Geometry();
  const GridIndex start{.x = 12, .y = 12};
  std::vector<float> elevation(
      geometry.CellCount(), std::numeric_limits<float>::quiet_NaN());
  for (const GridIndex sample : {GridIndex{.x = 10, .y = 10},
                                 GridIndex{.x = 14, .y = 10},
                                 GridIndex{.x = 10, .y = 14},
                                 GridIndex{.x = 0, .y = 0},
                                 GridIndex{.x = 24, .y = 24}}) {
    elevation[Offset(geometry, sample)] = 0.0F;
  }
  const TraversabilityProfile profile = Profile(3.0);
  const auto fixture =
      MakeFixture(geometry, elevation, WheelCapability(), profile);

  const StartPatchResult result = RequestLocalStartPatchBuilder().Build(
      fixture.fine, PoseAt(geometry, start), WheelCapability(), profile);

  EXPECT_EQ(result.status, StartPatchResult::Status::kUnresolved);
  EXPECT_FALSE(result.view);
  EXPECT_GT(result.assumed_cells, 0U);
}

TEST(RequestLocalStartPatch, KeepsFiniteObstacleAndItsInflationAuthoritative) {
  const GridGeometry geometry = Geometry();
  const GridIndex start{.x = 12, .y = 12};
  std::vector<float> elevation = FlatWithMissingSquare(geometry, start, 1);
  elevation[Offset(geometry, {.x = 11, .y = 12})] = 1.0F;
  const TraversabilityProfile profile = Profile(2.0);
  const auto fixture =
      MakeFixture(geometry, elevation, WheelCapability(), profile);

  const StartPatchResult result = RequestLocalStartPatchBuilder().Build(
      fixture.fine, PoseAt(geometry, start), WheelCapability(), profile);

  EXPECT_EQ(result.status, StartPatchResult::Status::kStartBlocked);
  EXPECT_FALSE(result.view);
  EXPECT_EQ(fixture.fine->State(start), FineCellState::kUnknown);
}

TEST(RequestLocalStartPatch,
     PreservesFiniteElevationRangeWhenFittedNeighborCompletesEvaluation) {
  const GridGeometry geometry = Geometry();
  const GridIndex start{.x = 5, .y = 5};
  std::vector<float> elevation = FlatWithMissingSquare(geometry, start, 1);
  PersistentElevationMap map;
  ASSERT_EQ(map.Apply(ElevationEvidence{
                          .geometry = geometry,
                          .elevation_m = elevation,
                          .map_from_source = IdentityMapTransform(),
                      })
                .status,
            ElevationUpdateResult::Status::kApplied);
  std::vector<float> colliding_values(
      36U, std::numeric_limits<float>::quiet_NaN());
  colliding_values[5U * 6U + 4U] = -0.4F;
  colliding_values[5U * 6U + 5U] = 0.4F;
  ASSERT_EQ(map.Apply(ElevationEvidence{
                          .geometry = Geometry(6U, 6U),
                          .elevation_m = colliding_values,
                          .map_from_source = RotatedMapTransform(
                              5.0 * std::numbers::pi / 180.0),
                      })
                .status,
            ElevationUpdateResult::Status::kApplied);
  const auto raw = map.Snapshot();
  const ElevationRange obstacle_range{.min_m = -0.4F, .max_m = 0.4F};
  ASSERT_EQ(raw->ElevationRangeAt({.x = 4, .y = 5}), obstacle_range);
  const TraversabilityProfile profile = Profile(2.0);
  const auto fine = FineTraversabilityBuilder().Derive(
      raw, WheelCapability(), profile);
  ASSERT_EQ(fine->State(start), FineCellState::kUnknown);

  const StartPatchResult result = RequestLocalStartPatchBuilder().Build(
      fine, PoseAt(geometry, start), WheelCapability(), profile);

  EXPECT_EQ(result.status, StartPatchResult::Status::kStartBlocked);
  EXPECT_FALSE(result.view);
  EXPECT_EQ(raw->ElevationRangeAt({.x = 4, .y = 5}), obstacle_range);
}

TEST(RequestLocalStartPatch,
     KeepsBaseSnapshotImmutableAndClipsOverlayToCapturedGeometry) {
  const GridGeometry geometry = Geometry(8U, 8U);
  const GridIndex start{.x = 1, .y = 1};
  std::vector<float> elevation = FlatWithMissingSquare(geometry, start, 1);
  const TraversabilityProfile profile = Profile(3.0);
  const auto fixture =
      MakeFixture(geometry, elevation, WheelCapability(), profile);
  const std::uint64_t raw_revision = fixture.raw->raw_elevation_revision();
  const std::uint64_t fine_revision =
      fixture.fine->fine_traversability_revision();

  const StartPatchResult result = RequestLocalStartPatchBuilder().Build(
      fixture.fine, PoseAt(geometry, start), WheelCapability(), profile);

  ASSERT_EQ(result.status, StartPatchResult::Status::kReady);
  ASSERT_TRUE(result.view);
  EXPECT_EQ(result.view->base(), fixture.fine);
  EXPECT_EQ(fixture.raw->raw_elevation_revision(), raw_revision);
  EXPECT_EQ(fixture.fine->fine_traversability_revision(), fine_revision);
  EXPECT_TRUE(std::ranges::all_of(
      result.view->overrides(), [&](const LocalCellOverride& cell) {
        return fixture.fine->geometry().Contains(cell.index) &&
               fixture.raw->geometry().Contains(cell.index);
      }));
}

TEST(RequestLocalStartPatch,
     EnforcesIrreversiblePrefixAndEvidenceOnlyEndpoint) {
  const GridGeometry geometry = Geometry();
  const GridIndex start{.x = 12, .y = 12};
  const GridIndex evidence{.x = 15, .y = 12};
  std::vector<float> elevation = FlatWithMissingSquare(geometry, start, 1);
  const TraversabilityProfile profile = Profile(2.0);
  const auto fixture =
      MakeFixture(geometry, elevation, WheelCapability(), profile);
  const StartPatchResult result = RequestLocalStartPatchBuilder().Build(
      fixture.fine, PoseAt(geometry, start), WheelCapability(), profile);
  ASSERT_EQ(result.status, StartPatchResult::Status::kReady);
  ASSERT_TRUE(result.view);
  ASSERT_EQ(result.view->Source(start), LocalCellSource::kStartAssumedFree);
  ASSERT_EQ(result.view->Source(evidence), LocalCellSource::kEvidenceFree);

  EXPECT_EQ(result.view->AdvancePhase(StartPhase::kStartPrefix, start),
            StartPhase::kStartPrefix);
  EXPECT_EQ(result.view->AdvancePhase(StartPhase::kStartPrefix, evidence),
            StartPhase::kNormal);
  EXPECT_FALSE(result.view->AdvancePhase(StartPhase::kNormal, start));
  EXPECT_TRUE(result.view->CanBeEndpoint(evidence));
  EXPECT_FALSE(result.view->CanBeEndpoint(start));
}

TEST(RequestLocalStartPatch,
     WheelAndLeggedSharePrefixButLeggedAssumptionIsInitialSupportOnly) {
  const GridGeometry geometry = Geometry();
  const GridIndex start{.x = 12, .y = 12};
  std::vector<float> elevation = FlatWithMissingSquare(geometry, start, 1);
  const TraversabilityProfile profile = Profile(2.0);
  const auto wheel_fixture =
      MakeFixture(geometry, elevation, WheelCapability(), profile);
  const auto legged_fixture = MakeFixture(
      geometry, elevation, LeggedCapabilityForTest(), profile);
  RequestLocalStartPatchBuilder builder;

  const StartPatchResult wheel = builder.Build(
      wheel_fixture.fine, PoseAt(geometry, start), WheelCapability(), profile);
  const StartPatchResult legged = builder.Build(
      legged_fixture.fine, PoseAt(geometry, start),
      LeggedCapabilityForTest(), profile);

  ASSERT_EQ(wheel.status, StartPatchResult::Status::kReady);
  ASSERT_TRUE(wheel.view);
  EXPECT_TRUE(wheel.view->CanCertifyLeggedSupport(start, true));
  EXPECT_FALSE(wheel.view->CanCertifyLeggedSupport(start, false));
  ASSERT_EQ(legged.status, StartPatchResult::Status::kReady);
  ASSERT_TRUE(legged.view);
  EXPECT_TRUE(legged.view->CanCertifyLeggedSupport(start, true));
  EXPECT_FALSE(legged.view->CanCertifyLeggedSupport(start, false));
}

TEST(RequestLocalStartPatch,
     AssumedSlopeUsesSameNonzeroSoftCostAsObservedSlope) {
  const GridGeometry geometry = Geometry();
  const GridIndex start{.x = 12, .y = 12};
  std::vector<float> elevation(geometry.CellCount(), 0.0F);
  for (std::size_t y = 0U; y < geometry.height; ++y) {
    for (std::size_t x = 0U; x < geometry.width; ++x) {
      elevation[y * geometry.width + x] = 0.02F * static_cast<float>(x);
    }
  }
  for (std::int64_t dy = -1; dy <= 1; ++dy) {
    for (std::int64_t dx = -1; dx <= 1; ++dx) {
      elevation[Offset(geometry,
                       {.x = start.x + dx, .y = start.y + dy})] =
          std::numeric_limits<float>::quiet_NaN();
    }
  }
  const TraversabilityProfile profile =
      Profile(2.0, {.slope = 5.0, .relief = 2.0});
  const auto fixture =
      MakeFixture(geometry, elevation, WheelCapability(), profile);

  const StartPatchResult result = RequestLocalStartPatchBuilder().Build(
      fixture.fine, PoseAt(geometry, start), WheelCapability(), profile);

  ASSERT_EQ(result.status, StartPatchResult::Status::kReady);
  ASSERT_TRUE(result.view);
  ASSERT_EQ(result.view->Source(start), LocalCellSource::kStartAssumedFree);
  EXPECT_GT(result.view->TraversalCost(start), 0.0);
  EXPECT_NEAR(result.view->TraversalCost(start),
              fixture.fine->TraversalCost({.x = 16, .y = 12}), 1.0e-5);
}

TEST(RequestLocalStartPatch,
     NearbyUnknownAddsClearanceCostWithoutChangingAssumedState) {
  const GridGeometry geometry = Geometry();
  const GridIndex start{.x = 12, .y = 12};
  std::vector<float> elevation = FlatWithMissingSquare(geometry, start, 1);
  elevation[Offset(geometry, {.x = 15, .y = 12})] =
      std::numeric_limits<float>::quiet_NaN();
  const TraversabilityProfile no_clearance = Profile();
  const TraversabilityProfile with_clearance =
      Profile(2.0, {.clearance = 4.0}, 3.0);
  const auto zero_fixture =
      MakeFixture(geometry, elevation, WheelCapability(), no_clearance);
  const auto cost_fixture =
      MakeFixture(geometry, elevation, WheelCapability(), with_clearance);
  RequestLocalStartPatchBuilder builder;

  const StartPatchResult zero = builder.Build(
      zero_fixture.fine, PoseAt(geometry, start), WheelCapability(),
      no_clearance);
  const StartPatchResult costed = builder.Build(
      cost_fixture.fine, PoseAt(geometry, start), WheelCapability(),
      with_clearance);

  ASSERT_EQ(zero.status, StartPatchResult::Status::kReady);
  ASSERT_EQ(costed.status, StartPatchResult::Status::kReady);
  ASSERT_TRUE(zero.view);
  ASSERT_TRUE(costed.view);
  EXPECT_EQ(zero.view->Source(start), LocalCellSource::kStartAssumedFree);
  EXPECT_EQ(costed.view->Source(start), LocalCellSource::kStartAssumedFree);
  EXPECT_DOUBLE_EQ(zero.view->TraversalCost(start), 0.0);
  EXPECT_GT(costed.view->TraversalCost(start), 0.0);
}

TEST(RequestLocalStartPatch, ReportsNotNeededAndRejectsInvalidMargin) {
  const GridGeometry geometry = Geometry();
  std::vector<float> elevation(geometry.CellCount(), 0.0F);
  const GridIndex start{.x = 12, .y = 12};
  TraversabilityProfile profile = Profile(0.0);
  const auto fixture =
      MakeFixture(geometry, elevation, WheelCapability(), profile);
  RequestLocalStartPatchBuilder builder;

  const StartPatchResult not_needed = builder.Build(
      fixture.fine, PoseAt(geometry, start), WheelCapability(), profile);
  EXPECT_EQ(not_needed.status, StartPatchResult::Status::kNotNeeded);
  ASSERT_TRUE(not_needed.view);
  EXPECT_TRUE(not_needed.view->overrides().empty());

  profile.start_blind_zone_margin_m = -0.1;
  EXPECT_THROW(builder.Build(fixture.fine, PoseAt(geometry, start),
                             WheelCapability(), profile),
               std::invalid_argument);
  profile.start_blind_zone_margin_m =
      std::numeric_limits<double>::infinity();
  EXPECT_THROW(builder.Build(fixture.fine, PoseAt(geometry, start),
                             WheelCapability(), profile),
               std::invalid_argument);
}

TEST(RequestLocalStartPatch, StationarySupportDoesNotRequireTranslationExit) {
  const auto geometry = Geometry(25U, 25U, 0.2);
  const GridIndex start{12, 12};
  std::vector<float> heights(geometry.CellCount(), std::numeric_limits<float>::quiet_NaN());
  for (const GridIndex cell : {start, GridIndex{13, 12}, GridIndex{13, 13}})
    heights[Offset(geometry, cell)] = 0.0F;
  const auto profile = Profile(0.4);
  const auto fixture = MakeFixture(geometry, heights, WheelCapability(), profile);
  const auto pose = PoseAt(geometry, start);
  ASSERT_EQ(fixture.fine->State(start), FineCellState::kUnknown);
  const auto ordinary = RequestLocalStartPatchBuilder{}.Build(fixture.fine, pose, WheelCapability(), profile);
  EXPECT_EQ(ordinary.status, StartPatchResult::Status::kUnresolved);
  const auto stationary = RequestLocalStartPatchBuilder{}.BuildStationary(fixture.fine, pose, WheelCapability(), profile);
  ASSERT_EQ(stationary.status, StartPatchResult::Status::kReady);
  ASSERT_TRUE(stationary.view);
  EXPECT_TRUE(stationary.view->CanCertifyStationary(pose.position_m));
  EXPECT_FALSE(stationary.view->CanCertifyStationary({pose.position_m.x + 0.01, pose.position_m.y}));
  EXPECT_FALSE(stationary.view->CanBeEndpoint(start));
  const FinalGoal goal{.target_x_m=pose.position_m.x, .target_y_m=pose.position_m.y,
                       .has_target_yaw=true, .target_yaw_rad=1.57};
  const auto selected=LocalTargetSelector{}.SelectRolling(*fixture.fine,
      fixture.fine->geometry(), pose.position_m, goal, std::nullopt);
  ASSERT_TRUE(selected);
  const auto plan=WheelLocalPlanner(WheelCapability()).Plan(*stationary.view,pose,*selected,
      SearchDeadline::max(),StopToken{});
  ASSERT_EQ(plan.status,LocalPlanResult::Status::kPlanFound);
  ASSERT_EQ(plan.path.size(),1U);
  EXPECT_DOUBLE_EQ(plan.path[0].pose.position_m.x,pose.position_m.x);
  EXPECT_TRUE(plan.reaches_final_goal);
  auto translation=*selected;
  translation.center.x+=0.01;
  EXPECT_EQ(WheelLocalPlanner(WheelCapability()).Plan(*stationary.view,pose,translation,
      SearchDeadline::max(),StopToken{}).status,LocalPlanResult::Status::kNoPath);

  EXPECT_EQ(fixture.fine->State(start), FineCellState::kUnknown);
  EXPECT_LT(stationary.view->geometry().min_inclusive().x, fixture.fine->geometry().min_inclusive().x);
}

TEST(RequestLocalStartPatch, StationaryPreservesKnownBlockersAndRequiresMeasuredPlane) {
  const auto g=Geometry(25U,25U,0.2);
  const GridIndex start{12,12};
  auto h=FlatWithMissingSquare(g,start,1);
  h[Offset(g,{13,12})]=1.0F;
  const auto fixture=MakeFixture(g,h,WheelCapability(),Profile(.4));
  EXPECT_EQ(RequestLocalStartPatchBuilder{}.BuildStationary(fixture.fine,PoseAt(g,start),
      WheelCapability(),Profile(.4)).status,StartPatchResult::Status::kStartBlocked);
  h.assign(g.CellCount(),std::numeric_limits<float>::quiet_NaN());
  h[Offset(g,start)]=0.0F;
  const auto unknown=MakeFixture(g,h,WheelCapability(),Profile(.4));
  EXPECT_EQ(RequestLocalStartPatchBuilder{}.BuildStationary(unknown.fine,PoseAt(g,start),
      WheelCapability(),Profile(.4)).status,StartPatchResult::Status::kUnresolved);
  EXPECT_EQ(RequestLocalStartPatchBuilder{}.BuildStationary(fixture.fine,PoseAt(g,start),
      LeggedCapabilityForTest(),Profile(.4)).status,StartPatchResult::Status::kUnresolved);
}

TEST(RequestLocalStartPatch, CanonicalWheelNearWallWedgeCertifiesStationarySupport) {
  GridGeometry g; g.frame_id="map";g.width=60;g.height=60;g.resolution_m=.2;g.origin_m={-4.,-4.,0.};
  std::vector<float> heights(g.width*g.height,0.f);
  for(int y=0;y<60;++y)for(int x=26;x<60;++x)heights[y*60+x]=2.f;
  RigidTransform identity;identity.parent_frame="map";identity.child_frame="map";
  PersistentElevationMap truth_map;
  const auto applied=truth_map.Apply({g,heights,identity});
  auto truth=truth_map.Snapshot();
  WheeledCapability wheel;
  wheel.footprint_xy_m={{.591,.409},{.591,-.409},{-.591,-.409},{-.591,.409}};
  wheel.body_extent_m={1.182,.818,1.29996};wheel.wheel_diameter_m=.319;wheel.wheel_width_m=.148;
  wheel.wheelbase_m=.8175;wheel.track_width_m=.67;wheel.minimum_underbody_clearance_m=.21;
  wheel.maximum_local_obstacle_relief_m=.2;wheel.maximum_forward_speed_mps=.2;
  wheel.maximum_reverse_speed_mps=.2;wheel.maximum_spin_rate_radps=1.;
  wheel.maximum_acceleration_mps2=.5;wheel.maximum_braking_deceleration_mps2=.5;
  wheel.maximum_yaw_acceleration_radps2=.5;wheel.maximum_lateral_acceleration_mps2=.5;
  wheel.maximum_curvature_per_m=1.;wheel.maximum_slope_rad=.3490658503988659;wheel.minimum_clearance_m=.2;
  PlatformCapability capability=wheel;
  TraversabilityProfile profile;profile.planar_envelope_xy_m=wheel.footprint_xy_m;
  profile.preferred_clearance_m=.2;profile.slope_weight=1.;profile.relief_weight=1.;
  profile.maximum_slope_rad=wheel.maximum_slope_rad;profile.start_blind_zone_margin_m=.2;
  profile.goal_position_tolerance_m=.3;profile.goal_yaw_tolerance_rad=.2617993877991494;
  auto true_fine=FineTraversabilityBuilder{}.Derive(truth,capability,profile);
  std::vector<float> measured(3600,std::numeric_limits<float>::quiet_NaN());
  std::vector<LocalTerrainMeasurements> stats(3600);
  for(int x=20;x<=25;++x)for(int y=20-(x-20);y<=20+(x-20);++y){
    const int i=y*60+x;measured[i]=heights[i];stats[i]=MeasureLocalTerrain(*truth,{x,y});
    stats[i].slope_rad=static_cast<float>(stats[i].slope_rad);
    stats[i].relief_m=static_cast<float>(stats[i].relief_m);
    stats[i].positive_rise_m=static_cast<float>(stats[i].positive_rise_m);
  }
  PersistentElevationMap measured_map;
  const auto measured_result=measured_map.Apply({g,measured,identity,stats});
  auto raw=measured_map.Snapshot();
  auto fine=FineTraversabilityBuilder{}.Derive(raw,capability,profile);
  const Pose2 pose{.position_m={.1,.1},.yaw_rad=0.};
  auto patch=RequestLocalStartPatchBuilder{}.BuildStationary(fine,pose,capability,profile);
  auto connections=RequestLocalStartPatchBuilder{}.BuildStartConnections(fine,fine->geometry(),pose,capability,profile);
  ASSERT_EQ(true_fine->State({20,20}),FineCellState::kFree);
  ASSERT_EQ(fine->State({20,20}),FineCellState::kUnknown);
  ASSERT_TRUE(connections.connections.empty());
  ASSERT_EQ(patch.status,StartPatchResult::Status::kReady);
  ASSERT_TRUE(patch.view->CanCertifyStationary(pose.position_m));
  EXPECT_FALSE(patch.view->CanBeEndpoint({20,20}));
}

}  // namespace
}  // namespace lunar::incremental_navigation

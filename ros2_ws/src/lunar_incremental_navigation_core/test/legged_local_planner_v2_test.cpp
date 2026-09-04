#include <chrono>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <memory>
#include <numbers>
#include <optional>
#include <stop_token>
#include <stdexcept>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "lunar_incremental_navigation_core/elevation_map.hpp"
#include "lunar_incremental_navigation_core/fine_traversability_builder.hpp"
#include "lunar_incremental_navigation_core/legged_local_planner.hpp"
#include "lunar_incremental_navigation_core/local_planning_window.hpp"

namespace lunar::incremental_navigation {
namespace {

constexpr std::size_t kWidth = 12U;
constexpr std::size_t kHeight = 9U;
constexpr double kResolutionM = 1.0;

[[nodiscard]] std::size_t Offset(const GridIndex index) {
  return static_cast<std::size_t>(index.y) * kWidth +
         static_cast<std::size_t>(index.x);
}

[[nodiscard]] Pose2 PoseAt(const GridIndex index, const double yaw = 0.0) {
  return Pose2{.position_m = {.x = static_cast<double>(index.x) + 0.5,
                             .y = static_cast<double>(index.y) + 0.5},
               .yaw_rad = yaw};
}

[[nodiscard]] LeggedCapability Capability() {
  return LeggedCapability{
      .body_extent_m = {.x = 0.6, .y = 0.4, .z = 0.3},
      .maximum_slope_rad = 1.0,
      .maximum_step_height_m = 0.25,
      .maximum_gap_width_m = 1.0,
      .minimum_body_clearance_m = 0.3,
      .body_height_m = {.lower = 0.3, .upper = 0.6},
      .motion_primitives = {
          {.primitive_id = "forward",
           .kind = LeggedPrimitiveKind::kForward,
           .body_frame_displacement_m = {.x = 1.0}},
          {.primitive_id = "backward",
           .kind = LeggedPrimitiveKind::kBackward,
           .body_frame_displacement_m = {.x = -1.0}},
          {.primitive_id = "spin-left",
           .kind = LeggedPrimitiveKind::kSpin,
           .yaw_change_rad = std::numbers::pi / 2.0},
          {.primitive_id = "spin-right",
           .kind = LeggedPrimitiveKind::kSpin,
           .yaw_change_rad = -std::numbers::pi / 2.0},
      },
  };
}

struct Fixture final {
  std::shared_ptr<const FineTraversabilitySnapshot> fine;
  std::shared_ptr<const RequestLocalPlanningView> view;
};

[[nodiscard]] Fixture MakeFixture(
    std::vector<FineCellState> states = {}, std::vector<float> elevation = {},
    std::vector<double> costs = {},
    std::vector<LocalCellOverride> overrides = {}) {
  const std::size_t count = kWidth * kHeight;
  if (states.empty()) {
    states.assign(count, FineCellState::kFree);
  }
  if (elevation.empty()) {
    elevation.assign(count, 0.0F);
  }
  if (costs.empty()) {
    costs.assign(count, 0.0);
  }
  GridGeometry raw_geometry{.frame_id = "map",
                            .width = kWidth,
                            .height = kHeight,
                            .resolution_m = kResolutionM};
  PersistentElevationMap map;
  const auto update = map.Apply(ElevationEvidence{
      .geometry = raw_geometry,
      .elevation_m = elevation,
      .map_from_source =
          RigidTransform{.parent_frame = "map", .child_frame = "map"},
  });
  if (update.status != ElevationUpdateResult::Status::kApplied) {
    throw std::runtime_error("test elevation fixture rejected");
  }
  auto raw = map.Snapshot();
  FineTraversabilityTile::StateArray tile_states;
  FineTraversabilityTile::CostArray tile_costs;
  tile_states.fill(FineCellState::kUnknown);
  tile_costs.fill(0.0);
  for (std::int64_t y = 0; y < static_cast<std::int64_t>(kHeight); ++y) {
    for (std::int64_t x = 0; x < static_cast<std::int64_t>(kWidth); ++x) {
      const GridIndex cell{.x = x, .y = y};
      tile_states[TileCellOffset(cell)] = states[Offset(cell)];
      tile_costs[TileCellOffset(cell)] = costs[Offset(cell)];
    }
  }
  FineTraversabilityTileDirectory directory(raw->geometry());
  directory = directory.WithTile(
      {.x = 0, .y = 0},
      std::make_shared<const FineTraversabilityTile>(
          std::move(tile_states), std::move(tile_costs)));
  auto fine = std::make_shared<const FineTraversabilitySnapshot>(
      raw->geometry(), raw->raw_elevation_revision(), 1U, "legged-test",
      0.37, 0.0, TraversalCostWeights{}, std::move(raw),
      std::move(directory), std::vector<TileIndex>{{.x = 0, .y = 0}},
      std::vector<TileIndex>{{.x = 0, .y = 0}}, FineSnapshotMetrics{});
  auto view = std::make_shared<const RequestLocalPlanningView>(
      fine, PoseAt({.x = 2, .y = 4}), 3.0, std::move(overrides));
  return {.fine = std::move(fine), .view = std::move(view)};
}

[[nodiscard]] LocalTarget TargetAt(const Point2 position,
                                   const bool final = false,
                                   std::optional<double> yaw = std::nullopt) {
  return LocalTarget{.center = position,
                     .position_tolerance_m = 0.0,
                     .is_final_goal = final,
                     .terminal_yaw_rad = yaw};
}

[[nodiscard]] Fixture MakeLargeDerivedFixture(
    const std::size_t width, const std::size_t height,
    const LeggedCapability& capability, const double resolution_m = 1.0) {
  GridGeometry geometry{.frame_id = "map",
                        .width = width,
                        .height = height,
                        .resolution_m = resolution_m};
  std::vector<float> elevation(width * height, 0.0F);
  PersistentElevationMap map;
  const auto update = map.Apply(ElevationEvidence{
      .geometry = geometry,
      .elevation_m = elevation,
      .map_from_source =
          RigidTransform{.parent_frame = "map", .child_frame = "map"},
  });
  if (update.status != ElevationUpdateResult::Status::kApplied) {
    throw std::runtime_error("large test elevation fixture rejected");
  }
  const TraversabilityProfile profile{
      .planar_envelope_xy_m = {{-0.1, -0.1}, {0.1, -0.1},
                               {0.1, 0.1}, {-0.1, 0.1}},
  };
  auto fine = FineTraversabilityBuilder().Derive(
      map.Snapshot(), capability, profile);
  const Pose2 anchor{.position_m = {
                         .x = (static_cast<double>(width / 2U) + 0.5) *
                              resolution_m,
                         .y = (static_cast<double>(height / 2U) + 0.5) *
                              resolution_m}};
  auto view = std::make_shared<const RequestLocalPlanningView>(
      fine, anchor, 0.0, std::vector<LocalCellOverride>{});
  return {.fine = std::move(fine), .view = std::move(view)};
}

TEST(LeggedLocalPlannerV2,
     UsesPlanarSearchAndPreservesContinuousStartAndFinalEndpoint) {
  const Fixture fixture = MakeFixture();
  LeggedCapability forward_only = Capability();
  forward_only.motion_primitives = {forward_only.motion_primitives.front()};
  const Pose2 start{.position_m = {.x = 2.25, .y = 4.5}, .yaw_rad = 0.0};
  const LocalTarget target = TargetAt({.x = 6.75, .y = 4.5}, true);

  const LocalPlanResult result = LeggedLocalPlanner(forward_only).Plan(
      *fixture.view, start, target, SteadyClock::time_point::max(), {});

  ASSERT_EQ(result.status, LocalPlanResult::Status::kPlanFound);
  EXPECT_GT(result.statistics.open_peak, 0U);
  EXPECT_GT(result.postprocess_elapsed, std::chrono::nanoseconds::zero());
  ASSERT_GE(result.path.size(), 2U);
  EXPECT_DOUBLE_EQ(result.path.front().pose.position_m.x, 2.25);
  EXPECT_DOUBLE_EQ(result.path.front().pose.position_m.y, 4.5);
  EXPECT_DOUBLE_EQ(result.path.back().pose.position_m.x, 6.75);
  EXPECT_DOUBLE_EQ(result.path.back().pose.position_m.y, 4.5);
  EXPECT_TRUE(result.reaches_final_goal);

  const Pose2 facing_away{.position_m = start.position_m,
                          .yaw_rad = std::numbers::pi};
  const LocalPlanResult independent_of_start_heading =
      LeggedLocalPlanner(forward_only).Plan(
      *fixture.view, facing_away, target,
      SteadyClock::time_point::max(), {});
  EXPECT_EQ(independent_of_start_heading.status,
            LocalPlanResult::Status::kPlanFound);

  const LocalPlanResult arbitrary_continuous_endpoint =
      LeggedLocalPlanner(forward_only).Plan(
      *fixture.view, start, TargetAt({.x = 6.5, .y = 4.75}, true),
      SteadyClock::time_point::max(), {});
  EXPECT_EQ(arbitrary_continuous_endpoint.status,
            LocalPlanResult::Status::kPlanFound);
}

TEST(LeggedLocalPlannerV2,
     UsesDiagonalTransitionsAndSimplifiesWithTheCertifiedLeggedEdgeCache) {
  const Fixture fixture = MakeFixture();
  LeggedCapability capability = Capability();
  capability.motion_primitives = {{
      .primitive_id = "long-reference-step",
      .kind = LeggedPrimitiveKind::kForward,
      .body_frame_displacement_m = {.x = 3.0},
  }};
  const LocalPlanResult result = LeggedLocalPlanner(capability).Plan(
      *fixture.view, PoseAt({.x = 2, .y = 4}),
      TargetAt(PoseAt({.x = 6, .y = 8}).position_m, true),
      SteadyClock::time_point::max(), {});

  ASSERT_EQ(result.status, LocalPlanResult::Status::kPlanFound);
  ASSERT_GE(result.raw_path.size(), 4U);
  ASSERT_LT(result.path.size(), result.raw_path.size());
  bool saw_diagonal = false;
  for (std::size_t index = 1U; index < result.raw_path.size(); ++index) {
    const Vec3& previous = result.raw_path[index - 1U].pose.position_m;
    const Vec3& current = result.raw_path[index].pose.position_m;
    saw_diagonal = saw_diagonal ||
                   (std::abs(current.x - previous.x) > 0.5 &&
                    std::abs(current.y - previous.y) > 0.5);
  }
  EXPECT_TRUE(saw_diagonal);
}

TEST(LeggedLocalPlannerV2, PublishesPlanarPathOnNonzeroElevation) {
  const Fixture fixture = MakeFixture(
      {}, std::vector<float>(kWidth * kHeight, 2.5F));
  LeggedCapability forward_only = Capability();
  forward_only.motion_primitives = {forward_only.motion_primitives.front()};
  const Pose2 start{.position_m = {.x = 2.25, .y = 4.5}, .yaw_rad = 0.0};
  const LocalTarget target = TargetAt({.x = 6.75, .y = 4.5}, true);

  const LocalPlanResult result = LeggedLocalPlanner(forward_only).Plan(
      *fixture.view, start, target, SteadyClock::time_point::max(), {});

  ASSERT_EQ(result.status, LocalPlanResult::Status::kPlanFound);
  ASSERT_FALSE(result.raw_path.empty());
  ASSERT_FALSE(result.path.empty());
  for (const PathPoint& point : result.raw_path) {
    EXPECT_DOUBLE_EQ(point.pose.position_m.z, 0.0);
  }
  for (const PathPoint& point : result.path) {
    EXPECT_DOUBLE_EQ(point.pose.position_m.z, 0.0);
  }
}

TEST(LeggedLocalPlannerV2,
     QuantizesAnArbitraryTwoDimensionalStartIntoTheFirstCertifiedEdge) {
  const Fixture fixture = MakeFixture();
  LeggedCapability forward_only = Capability();
  forward_only.motion_primitives = {forward_only.motion_primitives.front()};
  const Pose2 start{.position_m = {.x = 2.9, .y = 4.1}, .yaw_rad = 0.0};

  const LocalPlanResult result = LeggedLocalPlanner(forward_only).Plan(
      *fixture.view, start, TargetAt(PoseAt({.x = 5, .y = 4}).position_m,
                                        true),
      SteadyClock::time_point::max(), {});

  ASSERT_EQ(result.status, LocalPlanResult::Status::kPlanFound);
  ASSERT_GE(result.raw_path.size(), 2U);
  EXPECT_DOUBLE_EQ(result.raw_path.front().pose.position_m.x,
                   start.position_m.x);
  EXPECT_DOUBLE_EQ(result.raw_path.front().pose.position_m.y,
                   start.position_m.y);
  EXPECT_GT(result.statistics.generated_states, 1U);
}

TEST(LeggedLocalPlannerV2,
     ReachesAContinuousGoalRegionAtTheCertifiedLatticeEndpoint) {
  const Fixture fixture = MakeFixture();
  LeggedCapability forward_only = Capability();
  forward_only.motion_primitives = {forward_only.motion_primitives.front()};
  const LocalTarget target{
      .center = {.x = 5.9, .y = 4.8},
      .position_tolerance_m = 0.6,
      .is_final_goal = true,
  };

  const LocalPlanResult result = LeggedLocalPlanner(forward_only).Plan(
      *fixture.view, PoseAt({.x = 2, .y = 4}), target,
      SteadyClock::time_point::max(), {});

  ASSERT_EQ(result.status, LocalPlanResult::Status::kPlanFound);
  ASSERT_FALSE(result.raw_path.empty());
  EXPECT_TRUE(result.reaches_final_goal);
  EXPECT_DOUBLE_EQ(result.raw_path.back().pose.position_m.x, 5.5);
  EXPECT_DOUBLE_EQ(result.raw_path.back().pose.position_m.y, 4.5);
}

TEST(LeggedLocalPlannerV2,
     RejectsBlockedUnknownAndStepInDirectedEdgeCertification) {
  LeggedCapability capability = Capability();
  capability.motion_primitives = {capability.motion_primitives.front()};
  const LocalTarget target = TargetAt(PoseAt({.x = 6, .y = 4}).position_m);
  const auto no_path_across_column = [&](const FineCellState column_state,
                                         const float column_elevation) {
    std::vector<FineCellState> states(kWidth * kHeight,
                                      FineCellState::kFree);
    std::vector<float> elevation(kWidth * kHeight, 0.0F);
    for (std::int64_t y = 0; y < static_cast<std::int64_t>(kHeight); ++y) {
      states[Offset({.x = 4, .y = y})] = column_state;
      elevation[Offset({.x = 4, .y = y})] = column_elevation;
    }
    const Fixture fixture = MakeFixture(std::move(states), std::move(elevation));
    return LeggedLocalPlanner(capability).Plan(
        *fixture.view, PoseAt({.x = 2, .y = 4}), target,
        SteadyClock::time_point::max(), {});
  };

  EXPECT_EQ(no_path_across_column(FineCellState::kBlocked, 0.0F).status,
            LocalPlanResult::Status::kNoPath);
  EXPECT_EQ(no_path_across_column(FineCellState::kUnknown, 0.0F).status,
            LocalPlanResult::Status::kNoPath);
  EXPECT_EQ(no_path_across_column(FineCellState::kFree, 0.5F).status,
            LocalPlanResult::Status::kNoPath);
}

TEST(LeggedLocalPlannerV2, PlanarEdgeUsesSupercoverAtGridCorners) {
  std::vector<FineCellState> states(kWidth * kHeight,
                                    FineCellState::kBlocked);
  states[Offset({.x = 2, .y = 4})] = FineCellState::kFree;
  states[Offset({.x = 3, .y = 5})] = FineCellState::kFree;
  const Fixture fixture = MakeFixture(std::move(states));
  LeggedCapability capability = Capability();
  capability.motion_primitives = {{
      .primitive_id = "diagonal",
      .kind = LeggedPrimitiveKind::kCoupled,
      .body_frame_displacement_m = {.x = 1.0, .y = 1.0},
  }};

  const LocalPlanResult result = LeggedLocalPlanner(capability).Plan(
      *fixture.view, PoseAt({.x = 2, .y = 4}),
      TargetAt(PoseAt({.x = 3, .y = 5}).position_m),
      SteadyClock::time_point::max(), {});

  EXPECT_EQ(result.status, LocalPlanResult::Status::kNoPath);
}

TEST(LeggedLocalPlannerV2,
     CertifiesTheSameCellCenterGeometryThatReconstructionOutputs) {
  std::vector<FineCellState> states(kWidth * kHeight,
                                    FineCellState::kBlocked);
  states[Offset({.x = 2, .y = 4})] = FineCellState::kFree;
  states[Offset({.x = 3, .y = 5})] = FineCellState::kFree;
  const Fixture fixture = MakeFixture(std::move(states));
  LeggedCapability capability = Capability();
  capability.motion_primitives = {{
      .primitive_id = "off-center-diagonal",
      .kind = LeggedPrimitiveKind::kCoupled,
      .body_frame_displacement_m = {.x = 1.0, .y = 0.51},
  }};

  const LocalPlanResult result = LeggedLocalPlanner(capability).Plan(
      *fixture.view, PoseAt({.x = 2, .y = 4}),
      TargetAt(PoseAt({.x = 3, .y = 5}).position_m),
      SteadyClock::time_point::max(), {});

  EXPECT_EQ(result.status, LocalPlanResult::Status::kNoPath);
}

TEST(LeggedLocalPlannerV2,
     StartPatchAllowsMultiCellTrustedPrefixBeforeEvidenceExit) {
  std::vector<FineCellState> states(kWidth * kHeight,
                                    FineCellState::kBlocked);
  std::vector<float> elevation(kWidth * kHeight, 0.0F);
  states[Offset({.x = 2, .y = 4})] = FineCellState::kUnknown;
  states[Offset({.x = 3, .y = 4})] = FineCellState::kUnknown;
  states[Offset({.x = 4, .y = 4})] = FineCellState::kFree;
  states[Offset({.x = 5, .y = 4})] = FineCellState::kFree;
  elevation[Offset({.x = 2, .y = 4})] =
      std::numeric_limits<float>::quiet_NaN();
  elevation[Offset({.x = 3, .y = 4})] =
      std::numeric_limits<float>::quiet_NaN();
  const Fixture fixture = MakeFixture(
      std::move(states), std::move(elevation), {},
      {{.index = {.x = 2, .y = 4},
        .source = LocalCellSource::kStartAssumedFree},
       {.index = {.x = 3, .y = 4},
        .source = LocalCellSource::kStartAssumedFree}});
  LeggedCapability capability = Capability();
  capability.maximum_gap_width_m = 0.0;
  capability.motion_primitives = {capability.motion_primitives.front()};

  const LocalPlanResult result = LeggedLocalPlanner(capability).Plan(
      *fixture.view, PoseAt({.x = 2, .y = 4}),
      TargetAt(PoseAt({.x = 5, .y = 4}).position_m),
      SteadyClock::time_point::max(), {});

  ASSERT_EQ(result.status, LocalPlanResult::Status::kPlanFound);
  ASSERT_GE(result.raw_path.size(), 4U);
  EXPECT_EQ(result.raw_path[0].phase, StartPhase::kStartPrefix);
  EXPECT_EQ(result.raw_path[1].phase, StartPhase::kStartPrefix);
  EXPECT_EQ(result.raw_path[2].phase, StartPhase::kNormal);
  EXPECT_EQ(result.raw_path.back().phase, StartPhase::kNormal);
}

TEST(LeggedLocalPlannerV2, StartPatchCannotReenterAssumedCellsAfterEvidence) {
  std::vector<FineCellState> states(kWidth * kHeight,
                                    FineCellState::kBlocked);
  std::vector<float> elevation(kWidth * kHeight, 0.0F);
  states[Offset({.x = 2, .y = 4})] = FineCellState::kUnknown;
  states[Offset({.x = 3, .y = 4})] = FineCellState::kFree;
  states[Offset({.x = 4, .y = 4})] = FineCellState::kUnknown;
  states[Offset({.x = 5, .y = 4})] = FineCellState::kFree;
  elevation[Offset({.x = 2, .y = 4})] =
      std::numeric_limits<float>::quiet_NaN();
  elevation[Offset({.x = 4, .y = 4})] =
      std::numeric_limits<float>::quiet_NaN();
  const Fixture fixture = MakeFixture(
      std::move(states), std::move(elevation), {},
      {{.index = {.x = 2, .y = 4},
        .source = LocalCellSource::kStartAssumedFree},
       {.index = {.x = 4, .y = 4},
        .source = LocalCellSource::kStartAssumedFree}});
  LeggedCapability capability = Capability();
  capability.motion_primitives = {capability.motion_primitives.front()};

  EXPECT_EQ(LeggedLocalPlanner(capability)
                .Plan(*fixture.view, PoseAt({.x = 2, .y = 4}),
                      TargetAt(PoseAt({.x = 5, .y = 4}).position_m),
                      SteadyClock::time_point::max(), {})
                .status,
            LocalPlanResult::Status::kNoPath);
}

TEST(LeggedLocalPlannerV2, StartPatchDoesNotOverrideBlockedExit) {
  std::vector<FineCellState> states(kWidth * kHeight,
                                    FineCellState::kBlocked);
  std::vector<float> elevation(kWidth * kHeight, 0.0F);
  states[Offset({.x = 2, .y = 4})] = FineCellState::kUnknown;
  states[Offset({.x = 4, .y = 4})] = FineCellState::kFree;
  elevation[Offset({.x = 2, .y = 4})] =
      std::numeric_limits<float>::quiet_NaN();
  const Fixture fixture = MakeFixture(
      std::move(states), std::move(elevation), {},
      {{.index = {.x = 2, .y = 4},
        .source = LocalCellSource::kStartAssumedFree}});
  LeggedCapability capability = Capability();
  capability.motion_primitives = {capability.motion_primitives.front()};

  EXPECT_EQ(LeggedLocalPlanner(capability)
                .Plan(*fixture.view, PoseAt({.x = 2, .y = 4}),
                      TargetAt(PoseAt({.x = 4, .y = 4}).position_m),
                      SteadyClock::time_point::max(), {})
                .status,
            LocalPlanResult::Status::kNoPath);
}

TEST(LeggedLocalPlannerV2,
     InitialAssumedSupportCanTurnThenEnterEvidenceWithZeroGapLimit) {
  std::vector<FineCellState> states(kWidth * kHeight, FineCellState::kFree);
  std::vector<float> elevation(kWidth * kHeight, 0.0F);
  states[Offset({.x = 2, .y = 4})] = FineCellState::kUnknown;
  elevation[Offset({.x = 2, .y = 4})] =
      std::numeric_limits<float>::quiet_NaN();
  const Fixture fixture = MakeFixture(
      std::move(states), std::move(elevation), {},
      {{.index = {.x = 2, .y = 4},
        .source = LocalCellSource::kStartAssumedFree}});
  LeggedCapability capability = Capability();
  capability.maximum_gap_width_m = 0.0;
  capability.motion_primitives = {
      {.primitive_id = "spin-left",
       .kind = LeggedPrimitiveKind::kSpin,
       .yaw_change_rad = std::numbers::pi / 2.0},
      {.primitive_id = "forward",
       .kind = LeggedPrimitiveKind::kForward,
       .body_frame_displacement_m = {.x = 1.0}},
  };

  const LocalPlanResult result = LeggedLocalPlanner(capability).Plan(
      *fixture.view, PoseAt({.x = 2, .y = 4}),
      TargetAt(PoseAt({.x = 2, .y = 5}).position_m),
      SteadyClock::time_point::max(), {});

  ASSERT_EQ(result.status, LocalPlanResult::Status::kPlanFound);
  ASSERT_EQ(result.raw_path.size(), 2U);
  EXPECT_EQ(result.raw_path.front().phase, StartPhase::kStartPrefix);
  EXPECT_EQ(result.raw_path.back().phase, StartPhase::kNormal);
}

TEST(LeggedLocalPlannerV2,
     InitialAssumedSupportCanUseAShortenedTerminalExitToEvidence) {
  std::vector<FineCellState> states(kWidth * kHeight, FineCellState::kFree);
  std::vector<float> elevation(kWidth * kHeight, 0.0F);
  states[Offset({.x = 2, .y = 4})] = FineCellState::kUnknown;
  elevation[Offset({.x = 2, .y = 4})] =
      std::numeric_limits<float>::quiet_NaN();
  const Pose2 start{.position_m = {.x = 2.9, .y = 4.5}, .yaw_rad = 0.0};
  const Fixture fixture = MakeFixture(
      std::move(states), std::move(elevation), {},
      {{.index = {.x = 2, .y = 4},
        .source = LocalCellSource::kStartAssumedFree}});
  LeggedCapability capability = Capability();
  capability.maximum_gap_width_m = 0.0;
  capability.motion_primitives = {capability.motion_primitives.front()};

  const LocalPlanResult result = LeggedLocalPlanner(capability).Plan(
      *fixture.view, start, TargetAt({.x = 3.1, .y = 4.5}),
      SteadyClock::time_point::max(), {});

  ASSERT_EQ(result.status, LocalPlanResult::Status::kPlanFound);
  ASSERT_EQ(result.raw_path.size(), 2U);
  EXPECT_EQ(result.raw_path.front().phase, StartPhase::kStartPrefix);
  EXPECT_EQ(result.raw_path.back().phase, StartPhase::kNormal);
}

TEST(LeggedLocalPlannerV2, EnforcesGapWidthOnACachedDirectedEdge) {
  std::vector<FineCellState> states(kWidth * kHeight,
                                    FineCellState::kBlocked);
  std::vector<float> elevation(kWidth * kHeight, 0.0F);
  states[Offset({.x = 2, .y = 4})] = FineCellState::kFree;
  states[Offset({.x = 3, .y = 4})] = FineCellState::kFree;
  states[Offset({.x = 4, .y = 4})] = FineCellState::kFree;
  elevation[Offset({.x = 3, .y = 4})] =
      std::numeric_limits<float>::quiet_NaN();
  const Fixture fixture = MakeFixture(std::move(states), std::move(elevation));
  LeggedCapability capability = Capability();
  capability.motion_primitives = {{
      .primitive_id = "long-step",
      .kind = LeggedPrimitiveKind::kForward,
      .body_frame_displacement_m = {.x = 2.0},
  }};
  capability.maximum_gap_width_m = 0.5;
  const LocalTarget target =
      TargetAt(PoseAt({.x = 4, .y = 4}).position_m);

  EXPECT_EQ(LeggedLocalPlanner(capability)
                .Plan(*fixture.view, PoseAt({.x = 2, .y = 4}), target,
                      SteadyClock::time_point::max(), {})
                .status,
            LocalPlanResult::Status::kNoPath);

  capability.maximum_gap_width_m = 1.0;
  EXPECT_EQ(LeggedLocalPlanner(capability)
                .Plan(*fixture.view, PoseAt({.x = 2, .y = 4}), target,
                      SteadyClock::time_point::max(), {})
                .status,
            LocalPlanResult::Status::kPlanFound);
}

TEST(LeggedLocalPlannerV2,
     RequiresAFeasibleTerminalActionForOptionalFinalYaw) {
  const Fixture fixture = MakeFixture();
  LeggedCapability capability = Capability();
  const LocalTarget target = TargetAt(
      PoseAt({.x = 5, .y = 4}).position_m, true, std::numbers::pi / 2.0);

  const LocalPlanResult feasible = LeggedLocalPlanner(capability).Plan(
      *fixture.view, PoseAt({.x = 2, .y = 4}), target,
      SteadyClock::time_point::max(), {});
  ASSERT_EQ(feasible.status, LocalPlanResult::Status::kPlanFound);
  ASSERT_FALSE(feasible.path.empty());
  const Quaternion& orientation = feasible.path.back().pose.orientation;
  EXPECT_NEAR(orientation.w, std::cos(std::numbers::pi / 4.0), 1.0e-9);
  EXPECT_NEAR(orientation.z, std::sin(std::numbers::pi / 4.0), 1.0e-9);

  const LocalPlanResult two_spin_terminal =
      LeggedLocalPlanner(capability).Plan(
          *fixture.view, PoseAt({.x = 2, .y = 4}),
          TargetAt(PoseAt({.x = 5, .y = 4}).position_m, true,
                   std::numbers::pi),
          SteadyClock::time_point::max(), {});
  ASSERT_EQ(two_spin_terminal.status, LocalPlanResult::Status::kPlanFound);
  ASSERT_FALSE(two_spin_terminal.path.empty());
  EXPECT_NEAR(two_spin_terminal.path.back().pose.orientation.w, 0.0, 1.0e-9);
  EXPECT_NEAR(two_spin_terminal.path.back().pose.orientation.z, 1.0, 1.0e-9);

  capability.motion_primitives.resize(2U);
  std::vector<FineCellState> corridor(kWidth * kHeight,
                                      FineCellState::kBlocked);
  for (std::int64_t x = 2; x <= 5; ++x) {
    corridor[Offset({.x = x, .y = 4})] = FineCellState::kFree;
  }
  const Fixture corridor_fixture = MakeFixture(std::move(corridor));
  const LocalPlanResult infeasible = LeggedLocalPlanner(capability).Plan(
      *corridor_fixture.view, PoseAt({.x = 2, .y = 4}), target,
      SteadyClock::time_point::max(), {});
  EXPECT_EQ(infeasible.status, LocalPlanResult::Status::kNoPath);

  capability.motion_primitives = {Capability().motion_primitives.front()};
  const LocalPlanResult tiny_optional_yaw =
      LeggedLocalPlanner(capability).Plan(
          *fixture.view, PoseAt({.x = 5, .y = 4}),
          TargetAt(PoseAt({.x = 5, .y = 4}).position_m, true, 0.01),
          SteadyClock::time_point::max(), {});
  ASSERT_EQ(tiny_optional_yaw.status, LocalPlanResult::Status::kPlanFound);
  ASSERT_FALSE(tiny_optional_yaw.path.empty());
  EXPECT_NEAR(tiny_optional_yaw.path.back().pose.orientation.w, 1.0, 1.0e-12);
  EXPECT_NEAR(tiny_optional_yaw.path.back().pose.orientation.z, 0.0, 1.0e-12);

  const LocalPlanResult strict_tiny_yaw =
      LeggedLocalPlanner(
          capability, {.terminal_yaw_tolerance_rad = 0.0})
          .Plan(*fixture.view, PoseAt({.x = 5, .y = 4}),
                TargetAt(PoseAt({.x = 5, .y = 4}).position_m, true, 0.01),
                SteadyClock::time_point::max(), {});
  EXPECT_EQ(strict_tiny_yaw.status, LocalPlanResult::Status::kNoPath);
  EXPECT_EQ(strict_tiny_yaw.reason_code, "TERMINAL_YAW_UNREACHABLE");
}

TEST(LeggedLocalPlannerV2, FiniteTraversalCostOrdersButNeverBlocksAnEdge) {
  std::vector<double> costs(kWidth * kHeight, 0.0);
  for (std::int64_t x = 3; x <= 6; ++x) {
    costs[Offset({.x = x, .y = 4})] =
        std::numeric_limits<double>::max();
  }
  const Fixture fixture = MakeFixture({}, {}, std::move(costs));
  LeggedCapability capability = Capability();
  capability.motion_primitives = {capability.motion_primitives.front()};

  const LocalPlanResult result = LeggedLocalPlanner(capability).Plan(
      *fixture.view, PoseAt({.x = 2, .y = 4}),
      TargetAt(PoseAt({.x = 6, .y = 4}).position_m),
      SteadyClock::time_point::max(), {});

  EXPECT_EQ(result.status, LocalPlanResult::Status::kPlanFound);
}

TEST(LeggedLocalPlannerV2,
     EdgeCostMultipliesLengthByOnePlusTargetTraversalCost) {
  std::vector<double> costs(kWidth * kHeight, 0.0);
  costs[Offset({.x = 4, .y = 4})] = 10.0;
  const Fixture fixture = MakeFixture({}, {}, std::move(costs));
  LeggedCapability capability = Capability();
  capability.motion_primitives = {
      {.primitive_id = "long",
       .kind = LeggedPrimitiveKind::kForward,
       .body_frame_displacement_m = {.x = 2.0}},
      {.primitive_id = "short",
       .kind = LeggedPrimitiveKind::kForward,
       .body_frame_displacement_m = {.x = 1.0}},
  };

  const LocalPlanResult result = LeggedLocalPlanner(capability).Plan(
      *fixture.view, PoseAt({.x = 2, .y = 4}),
      TargetAt(PoseAt({.x = 4, .y = 4}).position_m),
      SteadyClock::time_point::max(), {});

  ASSERT_EQ(result.status, LocalPlanResult::Status::kPlanFound);
  EXPECT_GE(result.raw_path.size(), 3U);
  EXPECT_DOUBLE_EQ(result.raw_path.at(1U).pose.position_m.x, 3.5);
}

TEST(LeggedLocalPlannerV2, HonorsCancelDeadlineAndDeterministicTies) {
  const Fixture fixture = MakeFixture();
  LeggedLocalPlanner planner(Capability());
  const LocalTarget target =
      TargetAt(PoseAt({.x = 7, .y = 4}).position_m);
  std::stop_source source;
  source.request_stop();

  EXPECT_EQ(planner.Plan(*fixture.view, PoseAt({.x = 2, .y = 4}), target,
                         SteadyClock::time_point::max(), source.get_token())
                .status,
            LocalPlanResult::Status::kCanceled);
  EXPECT_EQ(planner.Plan(*fixture.view, PoseAt({.x = 2, .y = 4}), target,
                         SteadyClock::time_point::min(), {})
                .status,
            LocalPlanResult::Status::kTimeout);

  const LocalPlanResult first = planner.Plan(
      *fixture.view, PoseAt({.x = 2, .y = 4}), target,
      SteadyClock::time_point::max(), {});
  const LocalPlanResult second = planner.Plan(
      *fixture.view, PoseAt({.x = 2, .y = 4}), target,
      SteadyClock::time_point::max(), {});
  EXPECT_EQ(first.status, LocalPlanResult::Status::kPlanFound);
  ASSERT_EQ(first.raw_path.size(), second.raw_path.size());
  ASSERT_EQ(first.path.size(), second.path.size());
  for (std::size_t index = 0U; index < first.path.size(); ++index) {
    EXPECT_EQ(first.path[index].pose, second.path[index].pose);
    EXPECT_EQ(first.path[index].phase, second.path[index].phase);
  }
}

TEST(LeggedLocalPlannerV2,
     AdjacentGoalIn320By320WindowOnlyEvaluatesExpandedStateWork) {
  LeggedCapability capability = Capability();
  capability.motion_primitives = {capability.motion_primitives.front()};
  const Fixture fixture = MakeLargeDerivedFixture(320U, 320U, capability);
  const Pose2 start = PoseAt({.x = 160, .y = 160});

  const auto begin = SteadyClock::now();
  const LocalPlanResult result = LeggedLocalPlanner(capability).Plan(
      *fixture.view, start,
      TargetAt(PoseAt({.x = 161, .y = 160}).position_m),
      SteadyClock::time_point::max(), {});
  const auto elapsed = std::chrono::duration<double, std::milli>(
      SteadyClock::now() - begin);

  ASSERT_EQ(result.status, LocalPlanResult::Status::kPlanFound);
  EXPECT_LE(result.statistics.expanded_states, 2U);
  EXPECT_GT(result.statistics.evaluated_transitions, 0U);
  EXPECT_LE(result.statistics.evaluated_transitions, 8U);
  std::cout << "legged_local_320x320_ms=" << elapsed.count()
            << " expanded=" << result.statistics.expanded_states
            << " generated=" << result.statistics.generated_states
            << " open_peak=" << result.statistics.open_peak
            << " path_points=" << result.path.size() << '\n';
}

TEST(LeggedLocalPlannerV2,
     PlansThe64MeterPointOneMeterWindowWithoutReducingItsCellResolution) {
  LeggedCapability capability = Capability();
  capability.motion_primitives = {capability.motion_primitives.front()};
  const Fixture fixture = MakeLargeDerivedFixture(
      640U, 640U, capability, 0.1);
  const Pose2 start{.position_m = {.x = 32.05, .y = 32.05}};
  const LocalTarget target = TargetAt({.x = 33.05, .y = 32.05});

  const auto begin = SteadyClock::now();
  const LocalPlanResult result = LeggedLocalPlanner(capability).Plan(
      *fixture.view, start, target, SteadyClock::time_point::max(), {});
  const auto elapsed = std::chrono::duration<double, std::milli>(
      SteadyClock::now() - begin);

  ASSERT_EQ(result.status, LocalPlanResult::Status::kPlanFound);
  EXPECT_LE(result.statistics.expanded_states, 2U);
  std::cout << "legged_local_640x640_point1_ms=" << elapsed.count()
            << " expanded=" << result.statistics.expanded_states
            << " generated=" << result.statistics.generated_states
            << " open_peak=" << result.statistics.open_peak
            << " path_points=" << result.path.size() << '\n';
}

TEST(LeggedLocalPlannerV2,
     BoundedViewKeepsAGrowingFineSnapshotInsideTheLocalEnvelope) {
  LeggedCapability capability = Capability();
  capability.motion_primitives = {capability.motion_primitives.front()};
  const Fixture fixture = MakeLargeDerivedFixture(641U, 640U, capability);
  const Pose2 start = PoseAt({.x = 320, .y = 320});

  EXPECT_EQ(LeggedLocalPlanner(capability)
                .Plan(*fixture.view, start,
                      TargetAt(PoseAt({.x = 321, .y = 320}).position_m),
                      SteadyClock::time_point::max(), {})
                .status,
            LocalPlanResult::Status::kNoPath);

  const auto local_window = BuildLocalPlanningWindow(
      *fixture.fine, start.position_m, 640.0);
  ASSERT_EQ(local_window.status, LocalPlanningWindowStatus::kReady);
  ASSERT_TRUE(local_window.geometry);
  EXPECT_EQ(local_window.geometry->CellCount(), 640U * 640U);
  const RequestLocalPlanningView bounded(
      fixture.fine, *local_window.geometry, start, 0.0,
      std::vector<LocalCellOverride>{});
  EXPECT_EQ(LeggedLocalPlanner(capability)
                .Plan(bounded, start,
                      TargetAt(PoseAt({.x = 321, .y = 320}).position_m),
                      SteadyClock::time_point::max(), {})
                .status,
            LocalPlanResult::Status::kPlanFound);
}

TEST(LeggedLocalPlannerV2, DeadlineAndStopInterruptPrimitiveCertification) {
  const Fixture fixture = MakeFixture();
  LeggedCapability capability = Capability();
  capability.motion_primitives.assign(
      500000U,
      LeggedBodyPrimitive{.primitive_id = "forward",
                          .kind = LeggedPrimitiveKind::kForward,
                          .body_frame_displacement_m = {.x = 1.0}});
  LeggedLocalPlanner planner(capability);
  const LocalTarget unreachable =
      TargetAt(PoseAt({.x = 2, .y = 7}).position_m);

  const LocalPlanResult timeout = planner.Plan(
      *fixture.view, PoseAt({.x = 2, .y = 4}), unreachable,
      SteadyClock::now() + std::chrono::milliseconds(1), {});
  EXPECT_EQ(timeout.status, LocalPlanResult::Status::kTimeout);
  EXPECT_GT(timeout.statistics.evaluated_transitions, 0U);
  EXPECT_LT(timeout.statistics.evaluated_transitions,
            capability.motion_primitives.size());

  std::stop_source stop_source;
  std::jthread stopper([&stop_source] {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    stop_source.request_stop();
  });
  const LocalPlanResult canceled = planner.Plan(
      *fixture.view, PoseAt({.x = 2, .y = 4}), unreachable,
      SteadyClock::time_point::max(), stop_source.get_token());
  EXPECT_EQ(canceled.status, LocalPlanResult::Status::kCanceled);
  EXPECT_GT(canceled.statistics.evaluated_transitions, 0U);
  EXPECT_LT(canceled.statistics.evaluated_transitions,
            capability.motion_primitives.size());
}

}  // namespace
}  // namespace lunar::incremental_navigation

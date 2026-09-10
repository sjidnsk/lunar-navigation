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
#include <vector>

#include <gtest/gtest.h>

#include "lunar_incremental_navigation_core/elevation_map.hpp"
#include "lunar_incremental_navigation_core/fine_traversability_builder.hpp"
#include "lunar_incremental_navigation_core/legged_local_planner.hpp"
#include "lunar_incremental_navigation_core/local_planning_window.hpp"
#include "lunar_incremental_navigation_core/wheel_local_planner.hpp"
#include "lunar_incremental_navigation_core/local_goal_region.hpp"

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


class ReachableLocalGoals : public ::testing::TestWithParam<bool> {};
LocalPlanResult RunPlatform(const Fixture& fixture, const Pose2& start,
                    const LocalTarget& target, bool legged) {
  if (legged) {
    return LeggedLocalPlanner(Capability()).Plan(*fixture.view, start, target,
        SteadyClock::now() + std::chrono::seconds(2), {});
  }
  return WheelLocalPlanner().Plan(*fixture.view, start, target,
      SteadyClock::now() + std::chrono::seconds(2), {});
}
std::optional<LocalTarget> Rolling(const Fixture& fixture, Point2 start,
    FinalGoal goal, const std::optional<GlobalRoute>& route = {}) {
  return LocalTargetSelector().SelectRolling(*fixture.fine,
      fixture.view->geometry(), start, goal, route);
}
TEST_P(ReachableLocalGoals, DisconnectedFinalIslandUsesReachableWindowExit) {
  std::vector<FineCellState> cells(kWidth*kHeight, FineCellState::kUnknown);
  for (int y=4; y<9; ++y) cells[Offset({2,y})]=FineCellState::kFree;
  cells[Offset({3,4})]=FineCellState::kFree;
  cells[Offset({4,4})]=FineCellState::kFree;
  cells[Offset({8,4})]=FineCellState::kFree;
  const auto fixture=MakeFixture(cells);
  const Pose2 start=PoseAt({4,4});
  GlobalRoute route;
  for (Point2 point : {Point2{4.5,4.5}, Point2{4.5,12.5},
                       Point2{8.5,12.5}, Point2{8.5,4.5}})
    route.poses_map.push_back({.position_m={point.x,point.y,0}});
  const auto target=Rolling(fixture,start.position_m,
      FinalGoal{.target_x_m=8.5,.target_y_m=4.5},route);
  ASSERT_TRUE(target);
  const auto result=RunPlatform(fixture,start,*target,GetParam());
  ASSERT_EQ(result.status,LocalPlanResult::Status::kPlanFound);
  ASSERT_GT(result.path.size(),1U);
  EXPECT_FALSE(result.reaches_final_goal);
  EXPECT_DOUBLE_EQ(result.path.back().pose.position_m.x,2.5);
  EXPECT_DOUBLE_EQ(result.path.back().pose.position_m.y,8.5);
  // The first safe leg must move away from the final goal to follow the U.
  ASSERT_GT(result.raw_path.size(),1U);
  EXPECT_LT(result.raw_path[1].pose.position_m.x,start.position_m.x);
}
TEST_P(ReachableLocalGoals, CurrentPoseAloneNeverBecomesNewSuccessfulGoal) {
  std::vector<FineCellState> cells(kWidth*kHeight,FineCellState::kUnknown);
  cells[Offset({2,4})]=FineCellState::kFree;
  const auto fixture=MakeFixture(cells);
  const Pose2 start=PoseAt({2,4});
  const auto target=Rolling(fixture,start.position_m,
      FinalGoal{.target_x_m=10.5,.target_y_m=4.5});
  ASSERT_TRUE(target);
  const auto result=RunPlatform(fixture,start,*target,GetParam());
  EXPECT_EQ(result.status,LocalPlanResult::Status::kNoPath);
  EXPECT_EQ(result.reason_code,"LOCAL_WAITING_FOR_MAP");
  EXPECT_TRUE(result.path.empty());
}

TEST_P(ReachableLocalGoals, FrontierDoesNotSlideSidewaysWithoutRouteProgress) {
  std::vector<FineCellState> cells(kWidth*kHeight,FineCellState::kUnknown);
  for (int y=0;y<int(kHeight);++y)
    for(int x=0;x<=5;++x) cells[Offset({x,y})]=FineCellState::kFree;
  const auto fixture=MakeFixture(cells);
  const Pose2 start=PoseAt({5,4});
  const auto target=Rolling(fixture,start.position_m,
      FinalGoal{.target_x_m=10.5,.target_y_m=4.5});
  ASSERT_TRUE(target);
  const auto result=RunPlatform(fixture,start,*target,GetParam());
  EXPECT_EQ(result.status,LocalPlanResult::Status::kNoPath);
  EXPECT_EQ(result.reason_code,"LOCAL_WAITING_FOR_MAP");
  EXPECT_TRUE(result.path.empty());
}
TEST_P(ReachableLocalGoals, ReachableFinalRetainsContinuousEndpointAndYaw) {
  const auto fixture=MakeFixture();
  const Pose2 start=PoseAt({2,4});
  auto target=Rolling(fixture,start.position_m,
      FinalGoal{.target_x_m=7.6,.target_y_m=4.5,
                .has_target_yaw=true,.target_yaw_rad=0.0});
  ASSERT_TRUE(target);
  target->position_tolerance_m=0.0;
  const auto result=RunPlatform(fixture,start,*target,GetParam());
  ASSERT_EQ(result.status,LocalPlanResult::Status::kPlanFound);
  ASSERT_FALSE(result.path.empty());
  EXPECT_TRUE(result.reaches_final_goal);
  EXPECT_NEAR(result.path.back().pose.position_m.x,7.6,1.e-9);
  EXPECT_NEAR(result.path.back().pose.position_m.y,4.5,1.e-9);
  EXPECT_NEAR(result.path.back().pose.orientation.z,0.0,1.e-9);
}
TEST(ReachableLocalGoalsSupport, LeggedRejectsRegionalRouteAcrossStepOrMissingSupport) {
  std::vector<FineCellState> cells(kWidth*kHeight,FineCellState::kBlocked);
  for (int y=4;y<9;++y) cells[Offset({2,y})]=FineCellState::kFree;
  cells[Offset({3,4})]=FineCellState::kFree;
  cells[Offset({4,4})]=FineCellState::kFree;
  const Pose2 start=PoseAt({4,4});
  GlobalRoute route;
  for (Point2 point : {Point2{4.5,4.5},Point2{4.5,12.5},Point2{8.5,12.5}})
    route.poses_map.push_back({.position_m={point.x,point.y,0}});
  for (float barrier : {1.0F,std::numeric_limits<float>::quiet_NaN()}) {
    std::vector<float> elevation(kWidth*kHeight,0.0F);
    elevation[Offset({2,6})]=barrier;
    const auto fixture=MakeFixture(cells,elevation);
    const auto target=Rolling(fixture,start.position_m,
        FinalGoal{.target_x_m=8.5,.target_y_m=12.5},route);
    ASSERT_TRUE(target);
    const auto wheel=RunPlatform(fixture,start,*target,false);
    ASSERT_EQ(wheel.status,LocalPlanResult::Status::kPlanFound);
    const auto legged=RunPlatform(fixture,start,*target,true);
    EXPECT_EQ(legged.status,LocalPlanResult::Status::kNoPath);
    EXPECT_EQ(legged.reason_code,"LOCAL_NO_PROGRESS");
    EXPECT_TRUE(legged.path.empty());
  }
}

TEST_P(ReachableLocalGoals, FinalGoalAtCurrentPoseStillHonorsFinalRequest) {
  const auto fixture=MakeFixture();
  const Pose2 start=PoseAt({2,4});
  const auto target=Rolling(fixture,start.position_m,
      FinalGoal{.target_x_m=2.5,.target_y_m=4.5,
                .has_target_yaw=true,.target_yaw_rad=0.0});
  ASSERT_TRUE(target);
  const auto result=RunPlatform(fixture,start,*target,GetParam());
  ASSERT_EQ(result.status,LocalPlanResult::Status::kPlanFound);
  EXPECT_TRUE(result.reaches_final_goal);
}

TEST(ReachableLocalGoalRegion, MinimumInt64NeighborsStayWithinRepresentableCells) {
  const auto minimum = std::numeric_limits<std::int64_t>::min();
  for (const GridIndex free_cell : {GridIndex{minimum, 0}, GridIndex{0, minimum}}) {
    PersistentElevationMap map;
    const auto apply = [&](const Vec3 origin) {
      return map.Apply(ElevationEvidence{
          .geometry = {.frame_id = "map", .width = 1U, .height = 1U,
                       .resolution_m = 1.0, .origin_m = origin},
          .elevation_m = std::vector<float>{0.0F},
          .map_from_source = {.parent_frame = "map", .child_frame = "map"}});
    };
    ASSERT_EQ(apply({}).status, ElevationUpdateResult::Status::kApplied);
    ASSERT_EQ(apply({static_cast<double>(free_cell.x),
                     static_cast<double>(free_cell.y), 0.0}).status,
              ElevationUpdateResult::Status::kApplied);
    const auto elevation = map.Snapshot();
    const SparseGridGeometry window("map", 1.0, {}, free_cell,
                                    {free_cell.x + 1, free_cell.y + 1});
    ASSERT_TRUE(window.valid());
    FineTraversabilityTile::StateArray states;
    states.fill(FineCellState::kUnknown);
    states[TileCellOffset(free_cell)] = FineCellState::kFree;
    FineTraversabilityTile::CostArray costs;
    costs.fill(0.0);
    FineTraversabilityTileDirectory directory(window);
    directory = directory.WithTile(
        TileForCell(free_cell), std::make_shared<const FineTraversabilityTile>(
                                   std::move(states), std::move(costs)));
    const FineTraversabilitySnapshot fine(
        window, elevation->raw_elevation_revision(), 1U, "extreme-index-test",
        0.0, 0.0, TraversalCostWeights{}, elevation, std::move(directory), {}, {}, {});
    const Point2 position{static_cast<double>(free_cell.x),
                          static_cast<double>(free_cell.y)};
    // Run this fixture with UBSan as well: evaluating a missing -1 neighbor
    // before checking map containment used to overflow at either axis minimum.
    const auto target = LocalTargetSelector().SelectRolling(
        fine, window, position,
        FinalGoal{.target_x_m = position.x, .target_y_m = position.y}, {});
    ASSERT_TRUE(target);
    ASSERT_TRUE(target->region);
    const auto* candidate = target->region->At(free_cell);
    ASSERT_NE(candidate, nullptr);
    EXPECT_TRUE(candidate->target.is_final_goal);
    EXPECT_EQ(candidate->target.center, position);
  }
}

TEST_P(ReachableLocalGoals, UnknownAtDisconnectedFreeIslandIsNotWaitingEvidence) {
  std::vector<FineCellState> cells(kWidth * kHeight, FineCellState::kBlocked);
  cells[Offset({2, 4})] = FineCellState::kFree;
  cells[Offset({8, 4})] = FineCellState::kFree;
  cells[Offset({9, 4})] = FineCellState::kUnknown;
  const auto fixture = MakeFixture(cells);
  const Pose2 start = PoseAt({2, 4});
  const auto target = Rolling(fixture, start.position_m,
      FinalGoal{.target_x_m = 10.5, .target_y_m = 4.5});
  ASSERT_TRUE(target);
  ASSERT_TRUE(target->region);
  ASSERT_TRUE(target->region->has_unknown_boundary);
  const auto result = RunPlatform(fixture, start, *target, GetParam());
  EXPECT_EQ(result.status, LocalPlanResult::Status::kNoPath);
  EXPECT_EQ(result.reason_code, "LOCAL_NO_PROGRESS");
  EXPECT_TRUE(result.path.empty());
}

TEST_P(ReachableLocalGoals, ReachableUnknownBehindTheRouteIsNotWaitingEvidence) {
  std::vector<FineCellState> cells(kWidth * kHeight, FineCellState::kBlocked);
  for (int x = 1; x <= 4; ++x) cells[Offset({x, 4})] = FineCellState::kFree;
  cells[Offset({0, 4})] = FineCellState::kUnknown;
  cells[Offset({1, 5})] = FineCellState::kUnknown;
  const auto fixture = MakeFixture(cells);
  const Pose2 start = PoseAt({4, 4});
  const auto target = Rolling(fixture, start.position_m,
      FinalGoal{.target_x_m = 10.5, .target_y_m = 4.5});
  ASSERT_TRUE(target);
  ASSERT_TRUE(target->region->has_unknown_boundary);
  const auto result = RunPlatform(fixture, start, *target, GetParam());
  EXPECT_GT(result.statistics.expanded_states, 1U);
  EXPECT_EQ(result.status, LocalPlanResult::Status::kNoPath);
  EXPECT_EQ(result.reason_code, "LOCAL_NO_PROGRESS");
}

TEST_P(ReachableLocalGoals, CertifiedForwardFrontierIsReachedBeforeWaiting) {
  std::vector<FineCellState> cells(kWidth * kHeight, FineCellState::kBlocked);
  for (int x = 2; x <= 5; ++x) cells[Offset({x, 4})] = FineCellState::kFree;
  cells[Offset({6, 4})] = FineCellState::kUnknown;
  const auto fixture = MakeFixture(cells);
  const FinalGoal goal{.target_x_m = 10.5, .target_y_m = 4.5};
  const Pose2 start = PoseAt({2, 4});
  const auto approach_target = Rolling(fixture, start.position_m, goal);
  ASSERT_TRUE(approach_target);
  const auto approach = RunPlatform(fixture, start, *approach_target, GetParam());
  ASSERT_EQ(approach.status, LocalPlanResult::Status::kPlanFound);
  ASSERT_FALSE(approach.path.empty());
  EXPECT_DOUBLE_EQ(approach.path.back().pose.position_m.x, 5.5);
  const Pose2 frontier = PoseAt({5, 4});
  const auto waiting_target = Rolling(fixture, frontier.position_m, goal);
  ASSERT_TRUE(waiting_target);
  const auto waiting = RunPlatform(fixture, frontier, *waiting_target, GetParam());
  EXPECT_EQ(waiting.status, LocalPlanResult::Status::kNoPath);
  EXPECT_EQ(waiting.reason_code, "LOCAL_WAITING_FOR_MAP");
  EXPECT_TRUE(waiting.path.empty());
}

TEST_P(ReachableLocalGoals, OutsideFineGoalWaitsOnlyAtAReachableForwardMapBoundary) {
  std::vector<FineCellState> cells(kWidth * kHeight, FineCellState::kBlocked);
  cells[Offset({2, 4})] = FineCellState::kFree;
  cells[Offset({11, 4})] = FineCellState::kFree;
  const auto fixture = MakeFixture(cells);
  const FinalGoal goal{.target_x_m = 14.5, .target_y_m = 4.5};
  for (const bool at_boundary : {false, true}) {
    const Pose2 start = PoseAt({at_boundary ? 11 : 2, 4});
    const auto target = Rolling(fixture, start.position_m, goal);
    ASSERT_TRUE(target);
    ASSERT_TRUE(target->region->has_unknown_boundary);
    const auto result = RunPlatform(fixture, start, *target, GetParam());
    EXPECT_EQ(result.status, LocalPlanResult::Status::kNoPath);
    EXPECT_EQ(result.reason_code,
              at_boundary ? "LOCAL_WAITING_FOR_MAP" : "LOCAL_NO_PROGRESS");
    EXPECT_TRUE(result.path.empty());
  }
}

TEST_P(ReachableLocalGoals, UnknownTouchingOnlyABlockedCornerIsNotWaitingEvidence) {
  std::vector<FineCellState> cells(kWidth * kHeight, FineCellState::kBlocked);
  cells[Offset({2, 4})] = FineCellState::kFree;
  cells[Offset({3, 5})] = FineCellState::kUnknown;
  const auto fixture = MakeFixture(cells);
  const Pose2 start = PoseAt({2, 4});
  const auto target = Rolling(fixture, start.position_m,
      FinalGoal{.target_x_m = 10.5, .target_y_m = 12.5});
  ASSERT_TRUE(target);
  ASSERT_TRUE(target->region->has_unknown_boundary);
  const auto result = RunPlatform(fixture, start, *target, GetParam());
  EXPECT_EQ(result.status, LocalPlanResult::Status::kNoPath);
  EXPECT_EQ(result.reason_code, "LOCAL_NO_PROGRESS");
}

TEST_P(ReachableLocalGoals, CardinalUnknownOnDiagonalGuidanceStillPermitsWaiting) {
  std::vector<FineCellState> cells(kWidth * kHeight, FineCellState::kBlocked);
  cells[Offset({2, 4})] = FineCellState::kFree;
  cells[Offset({3, 4})] = FineCellState::kUnknown;
  const auto fixture = MakeFixture(cells);
  const Pose2 start = PoseAt({2, 4});
  const auto target = Rolling(fixture, start.position_m,
      FinalGoal{.target_x_m = 6.5, .target_y_m = 8.5});
  ASSERT_TRUE(target);
  const auto result = RunPlatform(fixture, start, *target, GetParam());
  EXPECT_EQ(result.status, LocalPlanResult::Status::kNoPath);
  EXPECT_EQ(result.reason_code, "LOCAL_WAITING_FOR_MAP");
  EXPECT_TRUE(result.path.empty());
}
INSTANTIATE_TEST_SUITE_P(Platforms,ReachableLocalGoals,::testing::Bool());
}  // namespace
}  // namespace lunar::incremental_navigation

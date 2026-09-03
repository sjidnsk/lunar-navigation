#include <chrono>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <memory>
#include <numbers>
#include <stop_token>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "lunar_incremental_navigation_core/elevation_map.hpp"
#include "lunar_incremental_navigation_core/wheel_local_planner.hpp"

namespace lunar::incremental_navigation {
namespace {

using Cell = std::pair<GridIndex, std::pair<FineCellState, double>>;

[[nodiscard]] RigidTransform IdentityMapTransform() {
  return RigidTransform{.parent_frame = "map", .child_frame = "map"};
}

[[nodiscard]] std::shared_ptr<const FineTraversabilitySnapshot> MakeFine(
    const std::size_t width, const std::size_t height,
    const std::vector<Cell>& cells, const bool default_free = false,
    const double resolution_m = 1.0, std::vector<float> elevation = {}) {
  PersistentElevationMap elevation_map;
  const GridGeometry geometry{.frame_id = "map",
                              .width = width,
                              .height = height,
                              .resolution_m = resolution_m};
  if (elevation.empty()) {
    elevation.assign(geometry.CellCount(), 0.0F);
  }
  if (elevation.size() != geometry.CellCount()) {
    throw std::runtime_error("test elevation fixture has an invalid size");
  }
  if (elevation_map.Apply(ElevationEvidence{
          .geometry = geometry,
          .elevation_m = elevation,
          .map_from_source = IdentityMapTransform(),
      }).status != ElevationUpdateResult::Status::kApplied) {
    throw std::runtime_error("test elevation fixture was rejected");
  }
  const auto raw = elevation_map.Snapshot();
  FineTraversabilityTileDirectory directory(raw->geometry());
  const std::size_t tile_columns = (width + 255U) / 256U;
  const std::size_t tile_rows = (height + 255U) / 256U;
  for (std::size_t tile_y = 0; tile_y < tile_rows; ++tile_y) {
    for (std::size_t tile_x = 0; tile_x < tile_columns; ++tile_x) {
      FineTraversabilityTile::StateArray states;
      states.fill(FineCellState::kUnknown);
      FineTraversabilityTile::CostArray costs;
      costs.fill(0.0);
      if (default_free) {
        const std::size_t min_x = tile_x * 256U;
        const std::size_t min_y = tile_y * 256U;
        const std::size_t max_x = std::min(width, min_x + 256U);
        const std::size_t max_y = std::min(height, min_y + 256U);
        for (std::size_t y = min_y; y < max_y; ++y) {
          for (std::size_t x = min_x; x < max_x; ++x) {
            states[TileCellOffset({.x = static_cast<std::int64_t>(x),
                                   .y = static_cast<std::int64_t>(y)})] =
                FineCellState::kFree;
          }
        }
      }
      for (const auto& [index, value] : cells) {
        if (TileForCell(index) ==
            TileIndex{.x = static_cast<std::int64_t>(tile_x),
                      .y = static_cast<std::int64_t>(tile_y)}) {
          states[TileCellOffset(index)] = value.first;
          costs[TileCellOffset(index)] = value.second;
        }
      }
      directory = directory.WithTile(
          {.x = static_cast<std::int64_t>(tile_x),
           .y = static_cast<std::int64_t>(tile_y)},
          std::make_shared<const FineTraversabilityTile>(std::move(states),
                                                         std::move(costs)));
    }
  }
  return std::make_shared<const FineTraversabilitySnapshot>(
      raw->geometry(), raw->raw_elevation_revision(), 1U, "wheel-profile",
      0.2, 0.0, TraversalCostWeights{}, raw, std::move(directory),
      std::vector<TileIndex>{}, std::vector<TileIndex>{},
      FineSnapshotMetrics{});
}

[[nodiscard]] RequestLocalPlanningView View(
    std::shared_ptr<const FineTraversabilitySnapshot> fine,
    std::vector<LocalCellOverride> overrides = {}) {
  return RequestLocalPlanningView(std::move(fine),
                                  Pose2{.position_m = {.x = 0.5, .y = 0.5}},
                                  20.0, std::move(overrides));
}

[[nodiscard]] LocalTarget Target(const double x, const double y,
                                 const bool final = true,
                                 std::optional<double> yaw = std::nullopt) {
  return LocalTarget{.center = {.x = x, .y = y},
                     .position_tolerance_m = 0.01,
                     .is_final_goal = final,
                     .terminal_yaw_rad = yaw};
}

[[nodiscard]] double Yaw(const Quaternion& q) {
  return std::atan2(2.0 * (q.w * q.z + q.x * q.y),
                    1.0 - 2.0 * (q.y * q.y + q.z * q.z));
}

TEST(WheelLocalPlanner, UsesEightNeighborsAndPreservesContinuousEndpoints) {
  const auto fine = MakeFine(6U, 6U, {}, true);
  const Pose2 start{.position_m = {.x = 0.2, .y = 0.3}, .yaw_rad = 0.4};
  const LocalPlanResult result = WheelLocalPlanner{}.Plan(
      View(fine), start, Target(4.8, 4.7),
      SteadyClock::time_point::max(), StopToken{});

  ASSERT_EQ(result.status, LocalPlanResult::Status::kPlanFound);
  EXPECT_GT(result.statistics.open_peak, 0U);
  EXPECT_GT(result.postprocess_elapsed, std::chrono::nanoseconds::zero());
  ASSERT_FALSE(result.raw_path.empty());
  EXPECT_DOUBLE_EQ(result.raw_path.front().pose.position_m.x, 0.2);
  EXPECT_DOUBLE_EQ(result.raw_path.front().pose.position_m.y, 0.3);
  EXPECT_DOUBLE_EQ(result.path.back().pose.position_m.x, 4.8);
  EXPECT_DOUBLE_EQ(result.path.back().pose.position_m.y, 4.7);
  EXPECT_LT(result.raw_path.size(), 8U);
}

TEST(WheelLocalPlanner, PublishesPlanarPathOnNonzeroElevation) {
  const auto fine = MakeFine(
      6U, 6U, {}, true, 1.0, std::vector<float>(36U, 2.5F));
  const LocalPlanResult result = WheelLocalPlanner{}.Plan(
      View(fine), {.position_m = {.x = 0.2, .y = 0.3}}, Target(4.8, 4.7),
      SteadyClock::time_point::max(), StopToken{});

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

TEST(WheelLocalPlanner, SoftCostOrdersPathsButNeverBlocksOnlyCorridor) {
  std::vector<Cell> cells;
  for (std::int64_t x = 0; x < 7; ++x) {
    cells.push_back({{.x = x, .y = 1}, {FineCellState::kFree, 1000.0}});
    cells.push_back({{.x = x, .y = 2}, {FineCellState::kFree, 0.0}});
  }
  const auto fine = MakeFine(7U, 4U, cells);
  const auto preferred = WheelLocalPlanner{}.Plan(
      View(fine), {.position_m = {.x = 0.5, .y = 1.5}}, Target(6.5, 1.5),
      SteadyClock::time_point::max(), StopToken{});
  ASSERT_EQ(preferred.status, LocalPlanResult::Status::kPlanFound);
  EXPECT_TRUE(std::any_of(preferred.raw_path.begin(), preferred.raw_path.end(),
                          [](const PathPoint& point) {
                            return point.pose.position_m.y > 2.0;
                          }));

  const auto corridor = MakeFine(
      7U, 3U,
      {{{.x = 0, .y = 1}, {FineCellState::kFree, 1000.0}},
       {{.x = 1, .y = 1}, {FineCellState::kFree, 1000.0}},
       {{.x = 2, .y = 1}, {FineCellState::kFree, 1000.0}},
       {{.x = 3, .y = 1}, {FineCellState::kFree, 1000.0}},
       {{.x = 4, .y = 1}, {FineCellState::kFree, 1000.0}},
       {{.x = 5, .y = 1}, {FineCellState::kFree, 1000.0}},
       {{.x = 6, .y = 1}, {FineCellState::kFree, 1000.0}}});
  const auto only = WheelLocalPlanner{}.Plan(
      View(corridor), {.position_m = {.x = 0.5, .y = 1.5}},
      Target(6.5, 1.5), SteadyClock::time_point::max(), StopToken{});
  EXPECT_EQ(only.status, LocalPlanResult::Status::kPlanFound);

  const auto extreme = MakeFine(
      2U, 1U,
      {{{.x = 0, .y = 0},
        {FineCellState::kFree, std::numeric_limits<double>::max()}},
       {{.x = 1, .y = 0},
        {FineCellState::kFree, std::numeric_limits<double>::max()}}});
  EXPECT_EQ(WheelLocalPlanner{}
                .Plan(View(extreme), {.position_m = {.x = 0.5, .y = 0.5}},
                      Target(1.5, 0.5), SteadyClock::time_point::max(),
                      StopToken{})
                .status,
            LocalPlanResult::Status::kPlanFound);
}

TEST(WheelLocalPlanner, PrefixBecomesNormalOnceAndCannotReturnToAssumed) {
  const auto fine = MakeFine(
      5U, 1U,
      {{{.x = 2, .y = 0}, {FineCellState::kFree, 0.0}},
       {{.x = 4, .y = 0}, {FineCellState::kFree, 0.0}}});
  const auto result = WheelLocalPlanner{}.Plan(
      View(fine, {{{.x = 0, .y = 0},
                   LocalCellSource::kStartAssumedFree, 0.0},
                  {{.x = 1, .y = 0},
                   LocalCellSource::kStartAssumedFree, 0.0},
                  {{.x = 3, .y = 0},
                   LocalCellSource::kStartAssumedFree, 0.0}}),
      {.position_m = {.x = 0.5, .y = 0.5}}, Target(4.5, 0.5),
      SteadyClock::time_point::max(), StopToken{});
  EXPECT_EQ(result.status, LocalPlanResult::Status::kNoPath);
}

TEST(WheelLocalPlanner, OptionalFinalYawRequiresCertifiedTerminalConnection) {
  const auto fine = MakeFine(
      5U, 3U,
      {{{.x = 0, .y = 1}, {FineCellState::kFree, 0.0}},
       {{.x = 1, .y = 1}, {FineCellState::kFree, 0.0}},
       {{.x = 2, .y = 1}, {FineCellState::kFree, 0.0}},
       {{.x = 3, .y = 1}, {FineCellState::kFree, 0.0}},
       {{.x = 4, .y = 1}, {FineCellState::kFree, 0.0}}});
  const auto no_spin = WheelLocalPlanner{}.Plan(
      View(fine), {.position_m = {.x = 0.5, .y = 1.5}},
      Target(4.5, 1.5, true, std::numbers::pi / 2.0),
      SteadyClock::time_point::max(), StopToken{});
  EXPECT_EQ(no_spin.status, LocalPlanResult::Status::kNoPath);

  WheeledCapability capability;
  capability.motion_primitives.push_back(WheelMotionPrimitive{
      .primitive_id = "spin",
      .kind = WheelPrimitiveKind::kSpinCounterclockwise});
  const auto with_spin = WheelLocalPlanner(capability).Plan(
      View(fine), {.position_m = {.x = 0.5, .y = 1.5}},
      Target(4.5, 1.5, true, std::numbers::pi / 2.0),
      SteadyClock::time_point::max(), StopToken{});
  ASSERT_EQ(with_spin.status, LocalPlanResult::Status::kPlanFound);
  ASSERT_GE(with_spin.path.size(), 2U);
  EXPECT_NEAR(Yaw(with_spin.path.back().pose.orientation),
              std::numbers::pi / 2.0, 1.0e-12);
  EXPECT_EQ(with_spin.path[with_spin.path.size() - 2U].pose.position_m,
            with_spin.path.back().pose.position_m);
}

TEST(WheelLocalPlanner, IntermediateTargetYawIsPathTangent) {
  const auto fine = MakeFine(5U, 3U, {}, true);
  const auto result = WheelLocalPlanner{}.Plan(
      View(fine), {.position_m = {.x = 0.5, .y = 1.5}},
      Target(4.5, 1.5, false, std::numbers::pi / 2.0),
      SteadyClock::time_point::max(), StopToken{});
  ASSERT_EQ(result.status, LocalPlanResult::Status::kPlanFound);
  EXPECT_NEAR(Yaw(result.path.back().pose.orientation), 0.0, 1.0e-12);
}

TEST(WheelLocalPlanner, SimplificationPreservesCertifiedFinalArrivalYaw) {
  const auto fine = MakeFine(6U, 6U, {}, true);
  const auto result = WheelLocalPlanner{}.Plan(
      View(fine), {.position_m = {.x = 0.5, .y = 2.5}},
      Target(4.5, 2.5, true, std::numbers::pi / 2.0),
      SteadyClock::time_point::max(), StopToken{});
  ASSERT_EQ(result.status, LocalPlanResult::Status::kPlanFound);
  ASSERT_GE(result.path.size(), 2U);
  const auto& before_goal = result.path[result.path.size() - 2U].pose.position_m;
  const auto& goal = result.path.back().pose.position_m;
  EXPECT_NEAR(std::atan2(goal.y - before_goal.y, goal.x - before_goal.x),
              std::numbers::pi / 2.0, 0.2617993877991494);
  EXPECT_NEAR(Yaw(result.path.back().pose.orientation),
              std::numbers::pi / 2.0, 1.0e-12);
}

TEST(WheelLocalPlanner,
     NeighborGoalYawUsesTheContinuousStartInsteadOfTheStartCellCenter) {
  const auto fine = MakeFine(2U, 1U, {}, true);
  const Pose2 start{.position_m = {.x = 0.99, .y = 0.01}, .yaw_rad = 0.0};
  const auto result = WheelLocalPlanner{}.Plan(
      View(fine), start, Target(1.01, 0.99, true, std::numbers::pi / 2.0),
      SteadyClock::time_point::max(), StopToken{});
  ASSERT_EQ(result.status, LocalPlanResult::Status::kPlanFound);
  ASSERT_GE(result.raw_path.size(), 2U);
  const auto& before_goal =
      result.raw_path[result.raw_path.size() - 2U].pose.position_m;
  const auto& goal = result.raw_path.back().pose.position_m;
  EXPECT_NEAR(std::atan2(goal.y - before_goal.y, goal.x - before_goal.x),
              std::numbers::pi / 2.0, 0.2617993877991494);
}

TEST(WheelLocalPlanner,
     InFlightCancelAndDeadlineNeverPublishPartiallyProcessedPaths) {
  const auto fine = MakeFine(40U, 1U, {}, true);
  std::stop_source source;
  int cancel_checks = 0;
  WheelLocalPlannerConfig cancel_config;
  cancel_config.now = [&] {
    if (++cancel_checks == 50) {
      source.request_stop();
    }
    return SteadyClock::time_point{};
  };
  const auto canceled = WheelLocalPlanner({}, cancel_config).Plan(
      View(fine), {.position_m = {.x = 0.5, .y = 0.5}},
      Target(39.5, 0.5), SteadyClock::time_point::max(), source.get_token());
  EXPECT_EQ(canceled.status, LocalPlanResult::Status::kCanceled);
  EXPECT_TRUE(canceled.raw_path.empty());
  EXPECT_TRUE(canceled.path.empty());

  int deadline_checks = 0;
  const auto deadline = SteadyClock::time_point{} + std::chrono::seconds(1);
  WheelLocalPlannerConfig deadline_config;
  deadline_config.now = [&] {
    return ++deadline_checks < 50 ? SteadyClock::time_point{} : deadline;
  };
  const auto timed_out = WheelLocalPlanner({}, deadline_config).Plan(
      View(fine), {.position_m = {.x = 0.5, .y = 0.5}},
      Target(39.5, 0.5), deadline, StopToken{});
  EXPECT_EQ(timed_out.status, LocalPlanResult::Status::kTimeout);
  EXPECT_TRUE(timed_out.raw_path.empty());
  EXPECT_TRUE(timed_out.path.empty());
}

TEST(WheelLocalPlanner, ReportsCanceledAndTimeoutDeterministically) {
  const auto fine = MakeFine(5U, 3U, {}, true);
  std::stop_source source;
  source.request_stop();
  EXPECT_EQ(WheelLocalPlanner{}
                .Plan(View(fine), {.position_m = {.x = 0.5, .y = 1.5}},
                      Target(4.5, 1.5), SteadyClock::time_point::max(),
                      source.get_token())
                .status,
            LocalPlanResult::Status::kCanceled);
  EXPECT_EQ(WheelLocalPlanner{}
                .Plan(View(fine), {.position_m = {.x = 0.5, .y = 1.5}},
                      Target(4.5, 1.5), SteadyClock::time_point::min(),
                      StopToken{})
                .status,
            LocalPlanResult::Status::kTimeout);
}

TEST(WheelLocalPlanner, DeterministicTiesProduceTheSameRawPath) {
  const auto fine = MakeFine(9U, 9U, {}, true);
  WheelLocalPlanner planner;
  const auto first = planner.Plan(
      View(fine), {.position_m = {.x = 0.5, .y = 4.5}}, Target(8.5, 4.5),
      SteadyClock::time_point::max(), StopToken{});
  const auto second = planner.Plan(
      View(fine), {.position_m = {.x = 0.5, .y = 4.5}}, Target(8.5, 4.5),
      SteadyClock::time_point::max(), StopToken{});
  ASSERT_EQ(first.status, LocalPlanResult::Status::kPlanFound);
  ASSERT_EQ(first.raw_path.size(), second.raw_path.size());
  for (std::size_t i = 0; i < first.raw_path.size(); ++i) {
    EXPECT_EQ(first.raw_path[i].pose, second.raw_path[i].pose);
    EXPECT_EQ(first.raw_path[i].phase, second.raw_path[i].phase);
  }
}

TEST(WheelLocalPlanner, ReusesDenseBuffersForThe320By320Envelope) {
  const auto fine = MakeFine(320U, 320U, {}, true, 0.2);
  WheelLocalPlanner planner;
  const auto begin = SteadyClock::now();
  const auto result = planner.Plan(
      View(fine), {.position_m = {.x = 0.1, .y = 0.1}},
      Target(63.9, 63.9), SteadyClock::time_point::max(), StopToken{});
  const auto elapsed = std::chrono::duration<double, std::milli>(
      SteadyClock::now() - begin);
  ASSERT_EQ(result.status, LocalPlanResult::Status::kPlanFound);
  EXPECT_EQ(planner.buffer_capacity_cells(), 320U * 320U);
  const std::size_t allocations = planner.buffer_allocation_count();
  EXPECT_EQ(planner.Plan(View(fine),
                         {.position_m = {.x = 0.1, .y = 0.1}},
                         Target(63.9, 63.9),
                         SteadyClock::time_point::max(), StopToken{})
                .status,
            LocalPlanResult::Status::kPlanFound);
  EXPECT_EQ(planner.buffer_allocation_count(), allocations);
  std::cout << "wheel_local_320x320_ms=" << elapsed.count()
            << " expanded=" << result.statistics.expanded_states
            << " generated=" << result.statistics.generated_states
            << " open_peak=" << result.statistics.open_peak
            << " path_points=" << result.path.size() << '\n';
}

TEST(WheelLocalPlanner,
     PlansThe64MeterPointOneMeterWindowWithoutReducingItsCellResolution) {
  const auto fine = MakeFine(640U, 640U, {}, true, 0.1);
  const Pose2 start{.position_m = {.x = 0.05, .y = 0.05}};
  const auto local_window = BuildLocalPlanningWindow(*fine, start.position_m, 64.0);
  ASSERT_EQ(local_window.status, LocalPlanningWindowStatus::kReady);
  ASSERT_TRUE(local_window.geometry);
  ASSERT_EQ(local_window.geometry->width(), 640U);
  ASSERT_EQ(local_window.geometry->height(), 640U);
  const RequestLocalPlanningView view(
      fine, *local_window.geometry, start, 0.0,
      std::vector<LocalCellOverride>{});
  WheelLocalPlanner planner;

  const auto begin = SteadyClock::now();
  const auto result = planner.Plan(
      view, start, Target(63.95, 63.95), SteadyClock::time_point::max(),
      StopToken{});
  const auto elapsed = std::chrono::duration<double, std::milli>(
      SteadyClock::now() - begin);

  ASSERT_EQ(result.status, LocalPlanResult::Status::kPlanFound);
  EXPECT_EQ(planner.buffer_capacity_cells(), 640U * 640U);
  std::cout << "wheel_local_640x640_point1_ms=" << elapsed.count()
            << " expanded=" << result.statistics.expanded_states
            << " generated=" << result.statistics.generated_states
            << " open_peak=" << result.statistics.open_peak
            << " path_points=" << result.path.size() << '\n';
}

TEST(WheelLocalPlanner, RejectsWindowsLargerThanTheFrozenEnvelope) {
  const auto fine = MakeFine(641U, 1U, {}, true);
  WheelLocalPlanner planner;
  EXPECT_EQ(planner
                .Plan(View(fine), {.position_m = {.x = 0.5, .y = 0.5}},
                      Target(640.5, 0.5), SteadyClock::time_point::max(),
                      StopToken{})
                .status,
            LocalPlanResult::Status::kNoPath);
  EXPECT_EQ(planner.buffer_capacity_cells(), 0U);
  EXPECT_EQ(planner.buffer_allocation_count(), 0U);

  const Pose2 start{.position_m = {.x = 320.5, .y = 0.5}};
  const auto local_window = BuildLocalPlanningWindow(
      *fine, start.position_m, 640.0);
  ASSERT_EQ(local_window.status, LocalPlanningWindowStatus::kReady);
  ASSERT_TRUE(local_window.geometry);
  EXPECT_EQ(local_window.geometry->width(), 640U);
  EXPECT_EQ(local_window.geometry->height(), 1U);
  const RequestLocalPlanningView bounded(
      fine, *local_window.geometry, start, 0.0,
      std::vector<LocalCellOverride>{});
  EXPECT_EQ(planner.Plan(bounded, start, Target(321.5, 0.5),
                         SteadyClock::time_point::max(), StopToken{})
                .status,
            LocalPlanResult::Status::kPlanFound);

  WheelLocalPlannerConfig oversized;
  oversized.maximum_width_cells =
      static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) + 1U;
  EXPECT_THROW((void)WheelLocalPlanner({}, oversized), std::invalid_argument);
}

}  // namespace
}  // namespace lunar::incremental_navigation

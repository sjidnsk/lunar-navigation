#include <chrono>
#include <limits>
#include <memory>
#include <set>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "local/grid_supercover.hpp"
#include "lunar_incremental_navigation_core/elevation_map.hpp"
#include "lunar_incremental_navigation_core/wheel_local_planner.hpp"

namespace lunar::incremental_navigation {
namespace {

[[nodiscard]] RigidTransform IdentityMapTransform() {
  return RigidTransform{.parent_frame = "map", .child_frame = "map"};
}

[[nodiscard]] std::shared_ptr<const FineTraversabilitySnapshot> MakeFine(
    const GridGeometry& geometry,
    std::vector<std::pair<GridIndex, FineCellState>> cells) {
  PersistentElevationMap map;
  const std::vector<float> elevation(geometry.CellCount(), 0.0F);
  if (map.Apply({.geometry = geometry,
                 .elevation_m = elevation,
                 .map_from_source = IdentityMapTransform()})
          .status != ElevationUpdateResult::Status::kApplied) {
    throw std::runtime_error("fixture failure");
  }
  const auto raw = map.Snapshot();
  FineTraversabilityTile::StateArray states;
  states.fill(FineCellState::kUnknown);
  FineTraversabilityTile::CostArray costs;
  costs.fill(0.0);
  for (const auto& [index, state] : cells) {
    states[TileCellOffset(index)] = state;
  }
  FineTraversabilityTileDirectory directory(raw->geometry());
  directory = directory.WithTile(
      {}, std::make_shared<const FineTraversabilityTile>(std::move(states),
                                                         std::move(costs)));
  return std::make_shared<const FineTraversabilitySnapshot>(
      raw->geometry(), raw->raw_elevation_revision(), 1U, "wheel", 0.2, 0.0,
      TraversalCostWeights{}, raw, std::move(directory),
      std::vector<TileIndex>{}, std::vector<TileIndex>{},
      FineSnapshotMetrics{});
}

[[nodiscard]] std::shared_ptr<const FineTraversabilitySnapshot> MakeFine(
    std::vector<std::pair<GridIndex, FineCellState>> cells) {
  return MakeFine(GridGeometry{.frame_id = "map",
                               .width = 8U,
                               .height = 4U,
                               .resolution_m = 1.0},
                  std::move(cells));
}

[[nodiscard]] PathPoint P(const double x, const double y,
                          const StartPhase phase) {
  return PathPoint{.pose = {.position_m = {.x = x, .y = y}}, .phase = phase};
}

// Captured failing terminal edges from the real planned outbound/return probe.
// Only the four cells touching the terminal corner intersect these segments.
TEST(GridSupercover, CapturedTerminalCornersDoNotTraversePastTheEndpoint) {
  const SparseGridGeometry geometry("map", 0.2, {.x = -12.0, .y = -12.0},
                                    {.x = 0, .y = 0}, {.x = 120, .y = 120});
  struct Case {
    Vec3 from;
    Vec3 to;
    std::set<GridIndex> expected;
  };
  const std::vector<Case> cases{
      {{.x = 7.900000000000001, .y = 0.10000000000000067},
       {.x = 8.0, .y = 0.0}, {{99, 60}, {100, 60}, {99, 59}, {100, 59}}},
      {{.x = 4.1000000000000005, .y = 4.900000000000001},
       {.x = 4.0, .y = 5.0}, {{80, 84}, {79, 84}, {80, 85}, {79, 85}}}};
  for (const auto& test : cases) {
    for (const bool reverse : {false, true}) {
      SCOPED_TRACE(reverse);
      std::set<GridIndex> visited;
      EXPECT_TRUE(local::VisitSupercoverCells(
          geometry, reverse ? test.to : test.from,
          reverse ? test.from : test.to, [&](const GridIndex index) {
            visited.insert(index);
            // Fail promptly if the implementation walks down the extension.
            return test.expected.contains(index);
          }));
      EXPECT_EQ(visited, test.expected);
      for (const GridIndex blocked : test.expected) {
        // Termination must happen after ALL corner neighbours are certified.
        EXPECT_FALSE(local::VisitSupercoverCells(
            geometry, reverse ? test.to : test.from,
            reverse ? test.from : test.to,
            [&](const GridIndex index) { return index != blocked; }));
      }
    }
  }
}

TEST(PhaseAwareSimplifier, PreservesSafeRawPathEndingAtMixedDirectionCorner) {
  std::vector<std::pair<GridIndex, FineCellState>> cells;
  for (std::int64_t y = 0; y < 4; ++y) {
    for (std::int64_t x = 0; x < 8; ++x) {
      cells.emplace_back(GridIndex{x, y}, FineCellState::kFree);
    }
  }
  const auto fine = MakeFine(std::move(cells));
  const RequestLocalPlanningView view(
      fine, {.position_m = {.x = 0.5, .y = 2.5}}, 0.0, {});
  const std::vector<PathPoint> raw{
      P(0.5, 2.5, StartPhase::kNormal),
      P(1.5, 2.5, StartPhase::kNormal),
      P(2.0, 2.0, StartPhase::kNormal)};
  const auto path = SimplifyPhaseAwarePath(view, raw);
  ASSERT_FALSE(path.empty());
  EXPECT_EQ(path.front().pose, raw.front().pose);
  EXPECT_EQ(path.back().pose, raw.back().pose);
}

TEST(PhaseAwareSimplifier, SimplifiesPrefixAndNormalWithoutCrossingSwitch) {
  const auto fine = MakeFine({{{.x = 2, .y = 1}, FineCellState::kFree},
                              {{.x = 3, .y = 1}, FineCellState::kFree},
                              {{.x = 4, .y = 1}, FineCellState::kFree},
                              {{.x = 5, .y = 1}, FineCellState::kFree}});
  const RequestLocalPlanningView view(
      fine, {.position_m = {.x = 0.5, .y = 1.5}}, 3.0,
      {{{.x = 0, .y = 1}, LocalCellSource::kStartAssumedFree, 0.0},
       {{.x = 1, .y = 1}, LocalCellSource::kStartAssumedFree, 0.0}});
  const std::vector<PathPoint> raw{
      P(0.5, 1.5, StartPhase::kStartPrefix),
      P(1.5, 1.5, StartPhase::kStartPrefix),
      P(2.5, 1.5, StartPhase::kNormal),
      P(3.5, 1.5, StartPhase::kNormal),
      P(5.5, 1.5, StartPhase::kNormal)};

  const auto simplified = SimplifyPhaseAwarePath(view, raw);
  ASSERT_EQ(simplified.size(), 4U);
  EXPECT_EQ(simplified[0].pose, raw[0].pose);
  EXPECT_EQ(simplified[1].pose, raw[1].pose);
  EXPECT_EQ(simplified[2].pose, raw[2].pose);
  EXPECT_EQ(simplified[3].pose, raw[4].pose);
}

TEST(PhaseAwareSimplifier, SupercoverRejectsBlockedCornerShortcut) {
  const auto fine = MakeFine({{{.x = 0, .y = 0}, FineCellState::kFree},
                              {{.x = 1, .y = 0}, FineCellState::kBlocked},
                              {{.x = 0, .y = 1}, FineCellState::kFree},
                              {{.x = 1, .y = 1}, FineCellState::kFree},
                              {{.x = 2, .y = 1}, FineCellState::kFree}});
  const RequestLocalPlanningView view(
      fine, {.position_m = {.x = 0.5, .y = 0.5}}, 0.0, {});
  const std::vector<PathPoint> raw{
      P(0.5, 0.5, StartPhase::kNormal),
      P(0.5, 1.5, StartPhase::kNormal),
      P(1.5, 1.5, StartPhase::kNormal),
      P(2.5, 1.5, StartPhase::kNormal)};

  const auto simplified = SimplifyPhaseAwarePath(view, raw);
  ASSERT_GE(simplified.size(), 3U);
  EXPECT_EQ(simplified.front().pose, raw.front().pose);
  EXPECT_EQ(simplified.back().pose, raw.back().pose);
}

TEST(PhaseAwareSimplifier,
     ConservativelyVisitsBothSidesOfANonCenterMathematicalCorner) {
  constexpr double kStartX = 0.1;
  constexpr double kStartY = 0.4;
  constexpr double kEndX = 2.8;
  constexpr double kEndY = 2.2;
  const double t_at_x_boundary = (1.0 - kStartX) / (kEndX - kStartX);
  const double t_at_y_boundary = (1.0 - kStartY) / (kEndY - kStartY);
  ASSERT_NE(t_at_x_boundary, t_at_y_boundary);
  ASSERT_GT(t_at_x_boundary, t_at_y_boundary);
  ASSERT_NEAR(t_at_x_boundary, t_at_y_boundary,
              2.0 * std::numeric_limits<double>::epsilon());

  const auto fine = MakeFine({{{.x = 0, .y = 0}, FineCellState::kFree},
                              {{.x = 1, .y = 0}, FineCellState::kBlocked},
                              {{.x = 0, .y = 1}, FineCellState::kFree},
                              {{.x = 1, .y = 1}, FineCellState::kFree},
                              {{.x = 2, .y = 1}, FineCellState::kFree},
                              {{.x = 2, .y = 2}, FineCellState::kFree}});
  const RequestLocalPlanningView view(
      fine, {.position_m = {.x = kStartX, .y = kStartY}}, 0.0, {});
  const std::vector<PathPoint> raw{
      P(kStartX, kStartY, StartPhase::kNormal),
      P(0.1, 1.4, StartPhase::kNormal),
      P(1.1, 1.4, StartPhase::kNormal),
      P(2.1, 1.4, StartPhase::kNormal),
      P(kEndX, kEndY, StartPhase::kNormal)};

  const auto simplified = SimplifyPhaseAwarePath(view, raw);
  ASSERT_GT(simplified.size(), 2U);
  EXPECT_EQ(simplified.front().pose, raw.front().pose);
  EXPECT_EQ(simplified.back().pose, raw.back().pose);
}

TEST(PhaseAwareSimplifier, StopAndDeadlineAbortWithoutReturningAPartialPath) {
  const auto fine = MakeFine({{{.x = 0, .y = 1}, FineCellState::kFree},
                              {{.x = 1, .y = 1}, FineCellState::kFree},
                              {{.x = 2, .y = 1}, FineCellState::kFree},
                              {{.x = 3, .y = 1}, FineCellState::kFree},
                              {{.x = 4, .y = 1}, FineCellState::kFree},
                              {{.x = 5, .y = 1}, FineCellState::kFree},
                              {{.x = 6, .y = 1}, FineCellState::kFree}});
  const RequestLocalPlanningView view(
      fine, {.position_m = {.x = 0.5, .y = 1.5}}, 0.0, {});
  const std::vector<PathPoint> raw{
      P(0.5, 1.5, StartPhase::kNormal),
      P(1.5, 1.5, StartPhase::kNormal),
      P(2.5, 1.5, StartPhase::kNormal),
      P(3.5, 1.5, StartPhase::kNormal),
      P(4.5, 1.5, StartPhase::kNormal),
      P(5.5, 1.5, StartPhase::kNormal),
      P(6.5, 1.5, StartPhase::kNormal)};

  std::stop_source source;
  int cancel_checks = 0;
  SearchControl cancel_control{
      .deadline = SteadyClock::time_point::max(),
      .stop_token = source.get_token(),
      .now = [&] {
        if (++cancel_checks == 3) {
          source.request_stop();
        }
        return SteadyClock::time_point{};
      }};
  EXPECT_TRUE(SimplifyPhaseAwarePath(view, raw, cancel_control).empty());

  int deadline_checks = 0;
  const auto deadline = SteadyClock::time_point{} + std::chrono::seconds(1);
  SearchControl deadline_control{
      .deadline = deadline,
      .now = [&] {
        return ++deadline_checks < 3 ? SteadyClock::time_point{}
                                     : deadline;
      }};
  EXPECT_TRUE(SimplifyPhaseAwarePath(view, raw, deadline_control).empty());
}

TEST(PhaseAwareSimplifier, FailedShortcutKeepsSafeRawSubpath) {
  const auto fine = MakeFine({{{.x = 0, .y = 0}, FineCellState::kFree},
                              {{.x = 0, .y = 1}, FineCellState::kFree},
                              {{.x = 1, .y = 1}, FineCellState::kFree},
                              {{.x = 2, .y = 1}, FineCellState::kFree},
                              {{.x = 2, .y = 0}, FineCellState::kFree},
                              {{.x = 1, .y = 0}, FineCellState::kBlocked}});
  const RequestLocalPlanningView view(
      fine, {.position_m = {.x = 0.5, .y = 0.5}}, 0.0, {});
  const std::vector<PathPoint> raw{
      P(0.5, 0.5, StartPhase::kNormal),
      P(0.5, 1.5, StartPhase::kNormal),
      P(1.5, 1.5, StartPhase::kNormal),
      P(2.5, 1.5, StartPhase::kNormal),
      P(2.5, 0.5, StartPhase::kNormal)};

  const auto simplified = SimplifyPhaseAwarePath(view, raw);
  EXPECT_GE(simplified.size(), 3U);
  EXPECT_EQ(simplified.front().pose, raw.front().pose);
  EXPECT_EQ(simplified.back().pose, raw.back().pose);
}

TEST(PhaseAwareSimplifier,
     CollinearRunsAreReducedBeforeBlockedShortcutSearch) {
  std::vector<std::pair<GridIndex, FineCellState>> cells;
  for (std::int64_t y = 0; y <= 2; ++y) {
    cells.emplace_back(GridIndex{.x = 0, .y = y}, FineCellState::kFree);
    cells.emplace_back(GridIndex{.x = 6, .y = y}, FineCellState::kFree);
  }
  for (std::int64_t x = 0; x <= 6; ++x) {
    cells.emplace_back(GridIndex{.x = x, .y = 2}, FineCellState::kFree);
  }
  for (std::int64_t x = 1; x < 6; ++x) {
    cells.emplace_back(GridIndex{.x = x, .y = 0}, FineCellState::kBlocked);
  }
  const auto fine = MakeFine(std::move(cells));
  const RequestLocalPlanningView view(
      fine, {.position_m = {.x = 0.5, .y = 0.5}}, 0.0, {});
  std::size_t sparse_probe_work = 0U;
  for (const std::size_t density : {1U, 10U}) {
    std::vector<PathPoint> raw;
    for (std::size_t step = 0U; step <= 40U * density; ++step) {
      raw.push_back(P(0.5, 0.5 + 2.0 * static_cast<double>(step) / (40.0 * density),
                      StartPhase::kNormal));
    }
    for (std::size_t step = 1U; step <= 120U * density; ++step) {
      raw.push_back(P(0.5 + 6.0 * static_cast<double>(step) / (120.0 * density), 2.5,
                      StartPhase::kNormal));
    }
    for (std::size_t step = 1U; step <= 40U * density; ++step) {
      raw.push_back(P(6.5, 2.5 - 2.0 * static_cast<double>(step) / (40.0 * density),
                      StartPhase::kNormal));
    }
    std::size_t clock_checks = 0U;
    const SearchControl control{
        .deadline = SteadyClock::time_point::max(),
        .now = [&] {
          ++clock_checks;
          return SteadyClock::time_point{};
        }};

    const auto simplified = SimplifyPhaseAwarePath(view, raw, control);

    ASSERT_EQ(simplified.size(), 4U);
    EXPECT_EQ(simplified.front().pose, raw.front().pose);
    EXPECT_EQ(simplified[1].pose, P(0.5, 2.5, StartPhase::kNormal).pose);
    EXPECT_EQ(simplified[2].pose, P(6.5, 2.5, StartPhase::kNormal).pose);
    EXPECT_EQ(simplified.back().pose, raw.back().pose);
    // Preprocessing checks interruption once per raw vertex. Increasing input
    // density tenfold must leave the remaining LOS work unchanged for this
    // fixed geometry, including every conservative closed-cell contact. This
    // detects rescanning dense candidates without encoding a visitor-count cap.
    ASSERT_GE(clock_checks, raw.size());
    const auto probe_work = clock_checks - raw.size();
    if (density == 1U) sparse_probe_work = probe_work;
    else EXPECT_EQ(probe_work, sparse_probe_work);
  }
}

TEST(PhaseAwareSimplifier,
     PrecompressedEdgesRemainCertifiedAtSmallValidResolution) {
  constexpr double kResolutionM = 1.0e-8;
  std::vector<std::pair<GridIndex, FineCellState>> cells;
  for (std::int64_t y = 0; y < 3; ++y) {
    for (std::int64_t x = 0; x < 4; ++x) {
      cells.emplace_back(GridIndex{.x = x, .y = y}, FineCellState::kFree);
    }
  }
  cells.emplace_back(GridIndex{.x = 1, .y = 1}, FineCellState::kBlocked);
  const auto fine = MakeFine(
      GridGeometry{.frame_id = "map",
                   .width = 4U,
                   .height = 3U,
                   .resolution_m = kResolutionM},
      std::move(cells));
  const RequestLocalPlanningView view(
      fine,
      {.position_m = {.x = 0.5 * kResolutionM,
                      .y = 0.5 * kResolutionM}},
      0.0, {});
  const std::vector<PathPoint> raw{
      P(0.5 * kResolutionM, 0.5 * kResolutionM, StartPhase::kNormal),
      P(2.5 * kResolutionM, 0.5 * kResolutionM, StartPhase::kNormal),
      P(3.5 * kResolutionM, 2.5 * kResolutionM, StartPhase::kNormal)};

  const auto simplified = SimplifyPhaseAwarePath(view, raw);

  ASSERT_GT(simplified.size(), 2U);
  EXPECT_EQ(simplified.front().pose, raw.front().pose);
  EXPECT_EQ(simplified.back().pose, raw.back().pose);
}

TEST(PhaseAwareSimplifier,
     PrefixShortcutCannotReenterAssumedCellsAfterCrossingEvidence) {
  const auto fine = MakeFine({{{.x = 1, .y = 0}, FineCellState::kFree}});
  const RequestLocalPlanningView view(
      fine, {.position_m = {.x = 0.5, .y = 0.5}}, 4.0,
      {{{.x = 0, .y = 0}, LocalCellSource::kStartAssumedFree, 0.0},
       {{.x = 0, .y = 1}, LocalCellSource::kStartAssumedFree, 0.0},
       {{.x = 1, .y = 1}, LocalCellSource::kStartAssumedFree, 0.0},
       {{.x = 2, .y = 1}, LocalCellSource::kStartAssumedFree, 0.0},
       {{.x = 2, .y = 0}, LocalCellSource::kStartAssumedFree, 0.0}});
  const std::vector<PathPoint> raw{
      P(0.5, 0.5, StartPhase::kStartPrefix),
      P(0.5, 1.5, StartPhase::kStartPrefix),
      P(1.5, 1.5, StartPhase::kStartPrefix),
      P(2.5, 1.5, StartPhase::kStartPrefix),
      P(2.5, 0.5, StartPhase::kStartPrefix)};

  const auto simplified = SimplifyPhaseAwarePath(view, raw);
  ASSERT_GT(simplified.size(), 2U);
  EXPECT_EQ(simplified.front().pose, raw.front().pose);
  EXPECT_EQ(simplified.back().pose, raw.back().pose);
}

TEST(PhaseAwareSimplifier,
     HorizontalGridBoundaryChecksIntersectedCellsOnBothSides) {
  const auto fine = MakeFine({{{.x = 0, .y = 0}, FineCellState::kFree},
                              {{.x = 1, .y = 0}, FineCellState::kBlocked},
                              {{.x = 2, .y = 0}, FineCellState::kFree},
                              {{.x = 0, .y = 1}, FineCellState::kFree},
                              {{.x = 1, .y = 1}, FineCellState::kFree},
                              {{.x = 2, .y = 1}, FineCellState::kFree}});
  const RequestLocalPlanningView view(
      fine, {.position_m = {.x = 0.2, .y = 1.0}}, 0.0, {});
  const std::vector<PathPoint> raw{
      P(0.2, 1.0, StartPhase::kNormal),
      P(0.2, 1.5, StartPhase::kNormal),
      P(2.8, 1.5, StartPhase::kNormal),
      P(2.8, 1.0, StartPhase::kNormal)};

  EXPECT_GT(SimplifyPhaseAwarePath(view, raw).size(), 2U);
}

TEST(PhaseAwareSimplifier,
     VerticalGridBoundaryChecksIntersectedCellsOnBothSides) {
  const auto fine = MakeFine({{{.x = 0, .y = 0}, FineCellState::kFree},
                              {{.x = 0, .y = 1}, FineCellState::kBlocked},
                              {{.x = 0, .y = 2}, FineCellState::kFree},
                              {{.x = 1, .y = 0}, FineCellState::kFree},
                              {{.x = 1, .y = 1}, FineCellState::kFree},
                              {{.x = 1, .y = 2}, FineCellState::kFree}});
  const RequestLocalPlanningView view(
      fine, {.position_m = {.x = 1.0, .y = 0.2}}, 0.0, {});
  const std::vector<PathPoint> raw{
      P(1.0, 0.2, StartPhase::kNormal),
      P(1.5, 0.2, StartPhase::kNormal),
      P(1.5, 2.8, StartPhase::kNormal),
      P(1.0, 2.8, StartPhase::kNormal)};

  EXPECT_GT(SimplifyPhaseAwarePath(view, raw).size(), 2U);
}

}  // namespace
}  // namespace lunar::incremental_navigation

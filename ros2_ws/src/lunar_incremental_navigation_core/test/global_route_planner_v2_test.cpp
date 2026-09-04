#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <stop_token>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "lunar_incremental_navigation_core/global_route_planner.hpp"

namespace lunar::incremental_navigation {
namespace {

struct CellValue final {
  GuidanceCellState state{GuidanceCellState::kUnknown};
  double risk{};
};

[[nodiscard]] std::shared_ptr<const GlobalGuidanceSnapshot> MakeSnapshot(
    const SparseGridGeometry& geometry,
    const std::map<GridIndex, CellValue>& values = {},
    const std::uint64_t revision = 1U,
    std::string profile_hash = "wheel-profile",
    std::vector<TileIndex> changed_halo_tiles = {}) {
  GlobalGuidanceTileDirectory directory(geometry);

  struct MutableTile final {
    GlobalGuidanceTile::StateArray states;
    GlobalGuidanceTile::RiskArray risks;

    MutableTile() {
      states.fill(GuidanceCellState::kUnknown);
      risks.fill(0.0);
    }
  };
  std::map<TileIndex, MutableTile> tiles;
  for (const auto& [index, value] : values) {
    if (!geometry.Contains(index)) {
      throw std::invalid_argument("test cell lies outside guidance geometry");
    }
    MutableTile& tile = tiles[TileForCell(index)];
    tile.states[TileCellOffset(index)] = value.state;
    tile.risks[TileCellOffset(index)] = value.risk;
  }
  for (auto& [index, tile] : tiles) {
    directory = directory.WithTile(
        index, std::make_shared<const GlobalGuidanceTile>(
                   std::move(tile.states), std::move(tile.risks)));
  }
  return std::make_shared<const GlobalGuidanceSnapshot>(
      geometry, 1U, 1U, revision, std::move(profile_hash),
      std::move(directory), std::vector<TileIndex>{},
      std::move(changed_halo_tiles));
}

[[nodiscard]] std::shared_ptr<const GlobalGuidanceSnapshot> MakeSnapshot(
    const std::int64_t width, const std::int64_t height,
    const std::map<GridIndex, CellValue>& values = {},
    const std::uint64_t revision = 1U,
    std::string profile_hash = "wheel-profile",
    std::vector<TileIndex> changed_halo_tiles = {}) {
  return MakeSnapshot(
      SparseGridGeometry("map", 1.0, Vec3{}, GridIndex{},
                         GridIndex{.x = width, .y = height}),
      values, revision, std::move(profile_hash),
      std::move(changed_halo_tiles));
}

[[nodiscard]] Point2 CellPoint(const std::int64_t x, const std::int64_t y,
                               const double dx = 0.5,
                               const double dy = 0.5) {
  return Point2{.x = static_cast<double>(x) + dx,
                .y = static_cast<double>(y) + dy};
}

[[nodiscard]] SearchDeadline GenerousDeadline() {
  return SteadyClock::now() + std::chrono::seconds(5);
}

TEST(GlobalRoutePlannerV2Test,
     FindsDeterministicEightNeighborRouteAndPreservesExactEndpoints) {
  std::map<GridIndex, CellValue> candidates;
  for (std::int64_t y = 0; y < 5; ++y) {
    for (std::int64_t x = 0; x < 5; ++x) {
      candidates[{.x = x, .y = y}] = {
          .state = GuidanceCellState::kCandidate, .risk = 0.0};
    }
  }
  const auto snapshot = MakeSnapshot(5, 5, candidates);
  const Point2 start = CellPoint(0, 0, 0.1, 0.2);
  const Point2 goal = CellPoint(4, 4, 0.8, 0.7);

  GlobalRoutePlanner first_planner(
      GlobalRoutePlannerConfig{.global_detour_margin_m = 0.0,
                               .unknown_step_risk = 5.0});
  GlobalRoutePlanner second_planner(
      GlobalRoutePlannerConfig{.global_detour_margin_m = 0.0,
                               .unknown_step_risk = 5.0});
  const GlobalRouteResult first = first_planner.Plan(
      *snapshot, start, goal, GenerousDeadline(), StopToken{});
  const GlobalRouteResult second = second_planner.Plan(
      *snapshot, start, goal, GenerousDeadline(), StopToken{});

  ASSERT_EQ(first.status, GuidanceStatus::kAvailable);
  ASSERT_TRUE(first.route.has_value());
  ASSERT_EQ(second.status, GuidanceStatus::kAvailable);
  ASSERT_TRUE(second.route.has_value());
  EXPECT_FALSE(first.reused_cache);
  EXPECT_GT(first.statistics.expanded_states, 0U);
  EXPECT_GT(first.statistics.open_peak, 0U);
  EXPECT_EQ(first.route->poses_map, second.route->poses_map);
  ASSERT_GE(first.route->poses_map.size(), 2U);
  EXPECT_DOUBLE_EQ(first.route->poses_map.front().position_m.x, start.x);
  EXPECT_DOUBLE_EQ(first.route->poses_map.front().position_m.y, start.y);
  EXPECT_DOUBLE_EQ(first.route->poses_map.back().position_m.x, goal.x);
  EXPECT_DOUBLE_EQ(first.route->poses_map.back().position_m.y, goal.y);
  EXPECT_DOUBLE_EQ(first.route->poses_map[1U].position_m.x, 1.5);
  EXPECT_DOUBLE_EQ(first.route->poses_map[1U].position_m.y, 1.5);
}

TEST(GlobalRoutePlannerV2Test,
     UnknownRemainsSearchableButCostsStrictlyMoreThanCandidateTerrain) {
  std::map<GridIndex, CellValue> values;
  for (std::int64_t y = 0; y < 3; ++y) {
    for (std::int64_t x = 0; x < 5; ++x) {
      values[{.x = x, .y = y}] = {
          .state = GuidanceCellState::kProvenBlocked, .risk = 0.0};
    }
  }
  for (std::int64_t x = 0; x < 5; ++x) {
    values[{.x = x, .y = 0}] = {
        .state = GuidanceCellState::kCandidate, .risk = 0.0};
  }
  values[{.x = 0, .y = 1}] = {
      .state = GuidanceCellState::kCandidate, .risk = 0.0};
  values[{.x = 4, .y = 1}] = {
      .state = GuidanceCellState::kCandidate, .risk = 0.0};
  for (std::int64_t x = 1; x < 4; ++x) {
    values[{.x = x, .y = 1}] = {
        .state = GuidanceCellState::kUnknown, .risk = 0.0};
  }
  const auto snapshot = MakeSnapshot(5, 3, values);
  GlobalRoutePlanner planner(
      GlobalRoutePlannerConfig{.global_detour_margin_m = 1.0,
                               .unknown_step_risk = 10.0});

  const GlobalRouteResult lower_risk = planner.Plan(
      *snapshot, CellPoint(0, 1), CellPoint(4, 1), GenerousDeadline(),
      StopToken{});

  ASSERT_EQ(lower_risk.status, GuidanceStatus::kAvailable);
  ASSERT_TRUE(lower_risk.route.has_value());
  ASSERT_GT(lower_risk.route->poses_map.size(), 2U);
  for (std::size_t i = 1U; i + 1U < lower_risk.route->poses_map.size(); ++i) {
    EXPECT_LT(lower_risk.route->poses_map[i].position_m.y, 1.0);
  }

  const auto unknown_only = MakeSnapshot(5, 1);
  const GlobalRouteResult finite_unknown = planner.Plan(
      *unknown_only, CellPoint(0, 0), CellPoint(4, 0), GenerousDeadline(),
      StopToken{});
  EXPECT_EQ(finite_unknown.status, GuidanceStatus::kAvailable);
  EXPECT_TRUE(finite_unknown.route.has_value());
}

TEST(GlobalRoutePlannerV2Test,
     NonnegativeTerrainRiskAffectsCostWithoutBecomingAHardGate) {
  std::map<GridIndex, CellValue> values;
  for (std::int64_t y = 0; y < 3; ++y) {
    for (std::int64_t x = 0; x < 5; ++x) {
      values[{.x = x, .y = y}] = {
          .state = GuidanceCellState::kCandidate, .risk = 0.0};
    }
  }
  for (std::int64_t x = 1; x < 4; ++x) {
    values[{.x = x, .y = 1}].risk = 50.0;
  }
  const auto snapshot = MakeSnapshot(5, 3, values);
  GlobalRoutePlanner planner(
      GlobalRoutePlannerConfig{.global_detour_margin_m = 1.0,
                               .unknown_step_risk = 5.0});

  const GlobalRouteResult result = planner.Plan(
      *snapshot, CellPoint(0, 1), CellPoint(4, 1), GenerousDeadline(),
      StopToken{});

  ASSERT_EQ(result.status, GuidanceStatus::kAvailable);
  ASSERT_TRUE(result.route.has_value());
  ASSERT_GT(result.route->poses_map.size(), 2U);
  bool left_high_risk_row = false;
  for (std::size_t i = 1U; i + 1U < result.route->poses_map.size(); ++i) {
    left_high_risk_row = left_high_risk_row ||
                         result.route->poses_map[i].position_m.y != 1.5;
  }
  EXPECT_TRUE(left_high_risk_row);
}

TEST(GlobalRoutePlannerV2Test,
     ProvenBlockedCellsAndBlockedOrthogonalCornersCannotBeCrossed) {
  const auto blocked_corner = MakeSnapshot(
      2, 2,
      {{{.x = 0, .y = 0}, {.state = GuidanceCellState::kCandidate}},
       {{.x = 1, .y = 1}, {.state = GuidanceCellState::kCandidate}},
       {{.x = 1, .y = 0}, {.state = GuidanceCellState::kProvenBlocked}},
       {{.x = 0, .y = 1}, {.state = GuidanceCellState::kProvenBlocked}}});
  GlobalRoutePlanner planner(
      GlobalRoutePlannerConfig{.global_detour_margin_m = 0.0,
                               .unknown_step_risk = 5.0});

  const GlobalRouteResult result = planner.Plan(
      *blocked_corner, CellPoint(0, 0), CellPoint(1, 1),
      GenerousDeadline(), StopToken{});

  EXPECT_EQ(result.status, GuidanceStatus::kNoRoute);
  EXPECT_FALSE(result.route.has_value());
}

TEST(GlobalRoutePlannerV2Test, SearchCannotLeaveFrozenDetourDomain) {
  std::map<GridIndex, CellValue> values;
  for (std::int64_t y = 0; y < 7; ++y) {
    for (std::int64_t x = 0; x < 7; ++x) {
      values[{.x = x, .y = y}] = {
          .state = GuidanceCellState::kCandidate, .risk = 0.0};
    }
  }
  for (std::int64_t y = 1; y <= 5; ++y) {
    values[{.x = 3, .y = y}] = {
        .state = GuidanceCellState::kProvenBlocked, .risk = 0.0};
  }
  const auto snapshot = MakeSnapshot(7, 7, values);
  const Point2 start = CellPoint(1, 3);
  const Point2 goal = CellPoint(5, 3);

  GlobalRoutePlanner narrow(
      GlobalRoutePlannerConfig{.global_detour_margin_m = 1.0,
                               .unknown_step_risk = 5.0});
  GlobalRoutePlanner wide(
      GlobalRoutePlannerConfig{.global_detour_margin_m = 3.0,
                               .unknown_step_risk = 5.0});

  EXPECT_EQ(narrow.Plan(*snapshot, start, goal, GenerousDeadline(),
                        StopToken{})
                .status,
            GuidanceStatus::kNoRoute);
  EXPECT_EQ(wide.Plan(*snapshot, start, goal, GenerousDeadline(), StopToken{})
                .status,
            GuidanceStatus::kAvailable);
}

TEST(GlobalRoutePlannerV2Test,
     DeadlineStopAndUnavailableInputsRemainGuidanceStatuses) {
  const auto snapshot = MakeSnapshot(4, 1);
  GlobalRoutePlanner planner;
  const Point2 start = CellPoint(0, 0);
  const Point2 goal = CellPoint(3, 0);

  const GlobalRouteResult expired = planner.Plan(
      *snapshot, start, goal, SteadyClock::now() - std::chrono::seconds(1),
      StopToken{});
  EXPECT_EQ(expired.status, GuidanceStatus::kTimeout);
  EXPECT_FALSE(expired.route.has_value());

  std::stop_source source;
  source.request_stop();
  const GlobalRouteResult stopped = planner.Plan(
      *snapshot, start, goal, GenerousDeadline(), source.get_token());
  EXPECT_EQ(stopped.status, GuidanceStatus::kTimeout);
  EXPECT_FALSE(stopped.route.has_value());

  const GlobalRouteResult outside = planner.Plan(
      *snapshot, CellPoint(-1, 0), goal, GenerousDeadline(), StopToken{});
  EXPECT_EQ(outside.status, GuidanceStatus::kUnavailable);
  EXPECT_FALSE(outside.route.has_value());

  const GlobalRouteResult nonfinite = planner.Plan(
      *snapshot,
      Point2{.x = std::numeric_limits<double>::quiet_NaN(), .y = 0.5}, goal,
      GenerousDeadline(), StopToken{});
  EXPECT_EQ(nonfinite.status, GuidanceStatus::kUnavailable);
  EXPECT_FALSE(nonfinite.route.has_value());
}

TEST(GlobalRoutePlannerV2Test,
     CacheUsesRevisionCellsAndProfileAndRebindsOnlySafeSuccessors) {
  const auto first_snapshot = MakeSnapshot(520, 3);
  GlobalRoutePlanner planner(
      GlobalRoutePlannerConfig{.global_detour_margin_m = 0.0,
                               .unknown_step_risk = 2.0});
  const Point2 start = CellPoint(1, 1, 0.1, 0.2);
  const Point2 goal = CellPoint(10, 1, 0.8, 0.7);

  const GlobalRouteResult first = planner.Plan(
      *first_snapshot, start, goal, GenerousDeadline(), StopToken{});
  ASSERT_EQ(first.status, GuidanceStatus::kAvailable);
  EXPECT_FALSE(first.reused_cache);
  EXPECT_GT(first.statistics.expanded_states, 0U);

  const GlobalRouteResult exact = planner.Plan(
      *first_snapshot, CellPoint(1, 1, 0.3, 0.4),
      CellPoint(10, 1, 0.6, 0.9), GenerousDeadline(), StopToken{});
  ASSERT_EQ(exact.status, GuidanceStatus::kAvailable);
  ASSERT_TRUE(exact.route.has_value());
  EXPECT_TRUE(exact.reused_cache);
  EXPECT_EQ(exact.statistics.expanded_states, 0U);
  EXPECT_DOUBLE_EQ(exact.route->poses_map.front().position_m.x, 1.3);
  EXPECT_DOUBLE_EQ(exact.route->poses_map.back().position_m.y, 1.9);

  const auto far_change = MakeSnapshot(
      520, 3, {}, 2U, "wheel-profile", {{.x = 1, .y = 0}});
  const GlobalRouteResult safely_rebound = planner.Plan(
      *far_change, start, goal, GenerousDeadline(), StopToken{});
  EXPECT_TRUE(safely_rebound.reused_cache);
  EXPECT_EQ(safely_rebound.statistics.expanded_states, 0U);

  const auto route_change = MakeSnapshot(
      520, 3,
      {{{.x = 5, .y = 1},
        {.state = GuidanceCellState::kCandidate, .risk = 3.0}}},
      3U, "wheel-profile", {{.x = 0, .y = 0}});
  const GlobalRouteResult invalidated = planner.Plan(
      *route_change, start, goal, GenerousDeadline(), StopToken{});
  EXPECT_FALSE(invalidated.reused_cache);
  EXPECT_GT(invalidated.statistics.expanded_states, 0U);

  const GlobalRouteResult new_start_cell = planner.Plan(
      *route_change, CellPoint(2, 1), goal, GenerousDeadline(), StopToken{});
  EXPECT_TRUE(new_start_cell.reused_cache);
  EXPECT_EQ(new_start_cell.statistics.expanded_states, 0U);

  const auto new_profile = MakeSnapshot(520, 3, {}, 4U, "legged-profile");
  const GlobalRouteResult profile_miss = planner.Plan(
      *new_profile, CellPoint(2, 1), goal, GenerousDeadline(), StopToken{});
  EXPECT_FALSE(profile_miss.reused_cache);

  GlobalRoutePlanner skipped_revision_planner;
  ASSERT_EQ(skipped_revision_planner
                .Plan(*first_snapshot, start, goal, GenerousDeadline(),
                      StopToken{})
                .status,
            GuidanceStatus::kAvailable);
  const auto skipped_revision = MakeSnapshot(
      520, 3, {}, 3U, "wheel-profile", {{.x = 1, .y = 0}});
  EXPECT_FALSE(skipped_revision_planner
                   .Plan(*skipped_revision, start, goal, GenerousDeadline(),
                         StopToken{})
                   .reused_cache);
}

TEST(GlobalRoutePlannerV2Test,
     CacheReusesAnOnRouteMovedStartSuffixButNotAnOffRouteStart) {
  const auto snapshot = MakeSnapshot(64, 3);
  GlobalRoutePlanner planner(
      GlobalRoutePlannerConfig{.global_detour_margin_m = 0.0,
                               .unknown_step_risk = 2.0});
  const Point2 start = CellPoint(1, 1, 0.1, 0.2);
  const Point2 goal = CellPoint(20, 1, 0.8, 0.7);
  const GlobalRouteResult first = planner.Plan(
      *snapshot, start, goal, GenerousDeadline(), StopToken{});
  ASSERT_EQ(first.status, GuidanceStatus::kAvailable);
  ASSERT_TRUE(first.route.has_value());

  const Point2 moved_start = CellPoint(5, 1, 0.2, 0.3);
  const GlobalRouteResult moved = planner.Plan(
      *snapshot, moved_start, goal, GenerousDeadline(), StopToken{});

  ASSERT_EQ(moved.status, GuidanceStatus::kAvailable);
  ASSERT_TRUE(moved.route.has_value());
  EXPECT_TRUE(moved.reused_cache);
  EXPECT_EQ(moved.statistics.expanded_states, 0U);
  EXPECT_DOUBLE_EQ(moved.route->poses_map.front().position_m.x,
                   moved_start.x);
  EXPECT_DOUBLE_EQ(moved.route->poses_map.front().position_m.y,
                   moved_start.y);
  EXPECT_DOUBLE_EQ(moved.route->poses_map.back().position_m.x, goal.x);
  EXPECT_DOUBLE_EQ(moved.route->poses_map.back().position_m.y, goal.y);
  EXPECT_EQ(moved.route->poses_map.size(),
            first.route->poses_map.size() - 4U);

  const GlobalRouteResult off_route = planner.Plan(
      *snapshot, CellPoint(5, 0), goal, GenerousDeadline(), StopToken{});
  EXPECT_EQ(off_route.status, GuidanceStatus::kAvailable);
  EXPECT_FALSE(off_route.reused_cache);
  EXPECT_GT(off_route.statistics.expanded_states, 0U);
}

TEST(GlobalRoutePlannerV2Test,
     EqualRevisionIndependentSnapshotCannotReuseABlockedMovedStart) {
  const std::map<GridIndex, CellValue> free_line{
      {{.x = 0, .y = 0}, {.state = GuidanceCellState::kCandidate}},
      {{.x = 1, .y = 0}, {.state = GuidanceCellState::kCandidate}},
      {{.x = 2, .y = 0}, {.state = GuidanceCellState::kCandidate}}};
  const auto first = MakeSnapshot(3, 1, free_line, 1U, "wheel-profile");
  auto blocked_line = free_line;
  blocked_line[{.x = 1, .y = 0}].state =
      GuidanceCellState::kProvenBlocked;
  const auto independent =
      MakeSnapshot(3, 1, blocked_line, 1U, "wheel-profile");
  GlobalRoutePlanner planner(
      GlobalRoutePlannerConfig{.global_detour_margin_m = 0.0,
                               .unknown_step_risk = 2.0});
  const Point2 goal = CellPoint(2, 0);
  ASSERT_EQ(planner.Plan(*first, CellPoint(0, 0), goal,
                         GenerousDeadline(), StopToken{})
                .status,
            GuidanceStatus::kAvailable);

  const GlobalRouteResult result = planner.Plan(
      *independent, CellPoint(1, 0), goal, GenerousDeadline(), StopToken{});

  EXPECT_EQ(result.status, GuidanceStatus::kNoRoute);
  EXPECT_FALSE(result.route.has_value());
  EXPECT_FALSE(result.reused_cache);
}

TEST(GlobalRoutePlannerV2Test,
     EqualRevisionIndependentSnapshotCannotReuseABlockedRouteInterior) {
  std::map<GridIndex, CellValue> free_line;
  for (std::int64_t x = 0; x < 4; ++x) {
    free_line[{.x = x, .y = 0}] = {
        .state = GuidanceCellState::kCandidate};
  }
  const auto first = MakeSnapshot(4, 1, free_line, 1U, "wheel-profile");
  auto blocked_line = free_line;
  blocked_line[{.x = 2, .y = 0}].state =
      GuidanceCellState::kProvenBlocked;
  const auto independent =
      MakeSnapshot(4, 1, blocked_line, 1U, "wheel-profile");
  GlobalRoutePlanner planner(
      GlobalRoutePlannerConfig{.global_detour_margin_m = 0.0,
                               .unknown_step_risk = 2.0});
  const Point2 start = CellPoint(0, 0);
  const Point2 goal = CellPoint(3, 0);
  ASSERT_EQ(planner.Plan(*first, start, goal, GenerousDeadline(), StopToken{})
                .status,
            GuidanceStatus::kAvailable);

  const GlobalRouteResult result = planner.Plan(
      *independent, start, goal, GenerousDeadline(), StopToken{});

  EXPECT_EQ(result.status, GuidanceStatus::kNoRoute);
  EXPECT_FALSE(result.route.has_value());
  EXPECT_FALSE(result.reused_cache);
}

TEST(GlobalRoutePlannerV2Test,
     EqualRevisionIndependentSnapshotRechecksRouteTerrainRisk) {
  std::map<GridIndex, CellValue> free_line;
  for (std::int64_t x = 0; x < 4; ++x) {
    free_line[{.x = x, .y = 0}] = {
        .state = GuidanceCellState::kCandidate};
  }
  const auto first = MakeSnapshot(4, 1, free_line, 1U, "wheel-profile");
  auto changed_risk = free_line;
  changed_risk[{.x = 2, .y = 0}].risk = 7.0;
  const auto independent =
      MakeSnapshot(4, 1, changed_risk, 1U, "wheel-profile");
  GlobalRoutePlanner planner(
      GlobalRoutePlannerConfig{.global_detour_margin_m = 0.0,
                               .unknown_step_risk = 2.0});
  const Point2 start = CellPoint(0, 0);
  const Point2 goal = CellPoint(3, 0);
  ASSERT_EQ(planner.Plan(*first, start, goal, GenerousDeadline(), StopToken{})
                .status,
            GuidanceStatus::kAvailable);

  const GlobalRouteResult result = planner.Plan(
      *independent, start, goal, GenerousDeadline(), StopToken{});

  EXPECT_EQ(result.status, GuidanceStatus::kAvailable);
  EXPECT_TRUE(result.route.has_value());
  EXPECT_FALSE(result.reused_cache);
  EXPECT_GT(result.statistics.expanded_states, 0U);
}

TEST(GlobalRoutePlannerV2Test,
     ReusesSuccessorWhenSameTileChangeIsBeyondExactRouteHalo) {
  const auto first_snapshot = MakeSnapshot(256, 3);
  GlobalRoutePlanner planner(
      GlobalRoutePlannerConfig{.global_detour_margin_m = 0.0,
                               .unknown_step_risk = 2.0});
  const Point2 start = CellPoint(1, 1);
  const Point2 goal = CellPoint(10, 1);
  ASSERT_EQ(planner.Plan(*first_snapshot, start, goal, GenerousDeadline(),
                         StopToken{})
                .status,
            GuidanceStatus::kAvailable);

  const auto far_same_tile_change = MakeSnapshot(
      256, 3,
      {{{.x = 100, .y = 1},
        {.state = GuidanceCellState::kCandidate, .risk = 3.0}}},
      2U, "wheel-profile", {{.x = 0, .y = 0}});
  const GlobalRouteResult result = planner.Plan(
      *far_same_tile_change, start, goal, GenerousDeadline(), StopToken{});

  EXPECT_EQ(result.status, GuidanceStatus::kAvailable);
  EXPECT_TRUE(result.reused_cache);
  EXPECT_EQ(result.statistics.expanded_states, 0U);
}

TEST(GlobalRoutePlannerV2Test,
     RecomputesSuccessorWhenChangedCellTouchesExactRouteHalo) {
  const auto first_snapshot = MakeSnapshot(256, 3);
  GlobalRoutePlanner planner(
      GlobalRoutePlannerConfig{.global_detour_margin_m = 0.0,
                               .unknown_step_risk = 2.0});
  const Point2 start = CellPoint(1, 1);
  const Point2 goal = CellPoint(10, 1);
  ASSERT_EQ(planner.Plan(*first_snapshot, start, goal, GenerousDeadline(),
                         StopToken{})
                .status,
            GuidanceStatus::kAvailable);

  const auto halo_change = MakeSnapshot(
      256, 3,
      {{{.x = 5, .y = 2},
        {.state = GuidanceCellState::kCandidate, .risk = 3.0}}},
      2U, "wheel-profile", {{.x = 0, .y = 0}});
  const GlobalRouteResult result = planner.Plan(
      *halo_change, start, goal, GenerousDeadline(), StopToken{});

  EXPECT_EQ(result.status, GuidanceStatus::kAvailable);
  EXPECT_FALSE(result.reused_cache);
  EXPECT_GT(result.statistics.expanded_states, 0U);
}

TEST(GlobalRoutePlannerV2Test,
     DeadlineCoversLongCacheValidationWithoutAdvancingRevision) {
  const auto first_snapshot = MakeSnapshot(256, 3);
  const auto base_time = SteadyClock::time_point{};
  bool expire = false;
  std::size_t clock_calls = 0U;
  GlobalRoutePlanner planner(GlobalRoutePlannerConfig{
      .global_detour_margin_m = 0.0,
      .unknown_step_risk = 2.0,
      .now = [&] {
        ++clock_calls;
        return expire && clock_calls > 4U
                   ? base_time + std::chrono::nanoseconds(2)
                   : base_time;
      },
  });
  const Point2 start = CellPoint(1, 1);
  const Point2 goal = CellPoint(200, 1);
  ASSERT_EQ(planner.Plan(*first_snapshot, start, goal,
                         base_time + std::chrono::nanoseconds(1), StopToken{})
                .status,
            GuidanceStatus::kAvailable);

  const std::map<GridIndex, CellValue> far_value{
      {{.x = 250, .y = 1},
       {.state = GuidanceCellState::kCandidate, .risk = 3.0}}};
  const auto successor = MakeSnapshot(
      256, 3, far_value, 2U, "wheel-profile", {{.x = 0, .y = 0}});
  clock_calls = 0U;
  expire = true;
  const GlobalRouteResult timed_out = planner.Plan(
      *successor, start, goal,
      base_time + std::chrono::nanoseconds(1), StopToken{});
  EXPECT_EQ(timed_out.status, GuidanceStatus::kTimeout);
  EXPECT_FALSE(timed_out.route.has_value());

  const auto skipped_revision = MakeSnapshot(
      256, 3, far_value, 3U, "wheel-profile", {{.x = 0, .y = 0}});
  clock_calls = 0U;
  expire = false;
  const GlobalRouteResult after_timeout = planner.Plan(
      *skipped_revision, start, goal,
      base_time + std::chrono::nanoseconds(1), StopToken{});
  EXPECT_EQ(after_timeout.status, GuidanceStatus::kAvailable);
  EXPECT_FALSE(after_timeout.reused_cache);
  EXPECT_GT(after_timeout.statistics.expanded_states, 0U);
}

TEST(GlobalRoutePlannerV2Test,
     PostSearchDeadlineDoesNotPublishRouteOrPopulateCache) {
  const auto snapshot = MakeSnapshot(1, 1);
  const auto base_time = SteadyClock::time_point{};
  bool expire = true;
  std::size_t clock_calls = 0U;
  GlobalRoutePlanner planner(GlobalRoutePlannerConfig{
      .global_detour_margin_m = 0.0,
      .unknown_step_risk = 2.0,
      .now = [&] {
        ++clock_calls;
        return expire && clock_calls > 5U
                   ? base_time + std::chrono::nanoseconds(2)
                   : base_time;
      },
  });
  const Point2 point = CellPoint(0, 0);

  const GlobalRouteResult timed_out = planner.Plan(
      *snapshot, point, point,
      base_time + std::chrono::nanoseconds(1), StopToken{});
  EXPECT_EQ(timed_out.status, GuidanceStatus::kTimeout);
  EXPECT_FALSE(timed_out.route.has_value());

  expire = false;
  clock_calls = 0U;
  const GlobalRouteResult recomputed = planner.Plan(
      *snapshot, point, point,
      base_time + std::chrono::nanoseconds(1), StopToken{});
  EXPECT_EQ(recomputed.status, GuidanceStatus::kAvailable);
  EXPECT_FALSE(recomputed.reused_cache);
  EXPECT_GT(recomputed.statistics.expanded_states, 0U);
}

TEST(GlobalRoutePlannerV2Test,
     CheckedOffsetsSupportLegalInt64MinimumSingleCellGeometry) {
  const std::int64_t minimum = std::numeric_limits<std::int64_t>::min();
  const SparseGridGeometry geometry(
      "map", 1.0,
      Vec3{.x = std::ldexp(1.0, 63), .y = 0.0, .z = 0.0},
      GridIndex{.x = minimum, .y = 0},
      GridIndex{.x = minimum + 1, .y = 1});
  const auto snapshot = MakeSnapshot(geometry);
  GlobalRoutePlanner planner;

  const GlobalRouteResult result = planner.Plan(
      *snapshot, Point2{.x = 0.0, .y = 0.5},
      Point2{.x = 0.0, .y = 0.5}, GenerousDeadline(), StopToken{});

  EXPECT_EQ(result.status, GuidanceStatus::kAvailable);
  EXPECT_TRUE(result.route.has_value());
}

TEST(GlobalRoutePlannerV2Test, RejectsTwoToThe63CellMarginBeforeConversion) {
  const auto snapshot = MakeSnapshot(1, 1);
  GlobalRoutePlanner planner(GlobalRoutePlannerConfig{
      .global_detour_margin_m = std::ldexp(1.0, 63),
      .unknown_step_risk = 1.0,
  });

  EXPECT_THROW(planner.Plan(*snapshot, CellPoint(0, 0), CellPoint(0, 0),
                            GenerousDeadline(), StopToken{}),
               std::invalid_argument);
}

TEST(GlobalRoutePlannerV2Test,
     MultipleMaximumFiniteTerrainRisksRemainSearchable) {
  const double maximum = std::numeric_limits<double>::max();
  const auto snapshot = MakeSnapshot(
      3, 1,
      {{{.x = 0, .y = 0},
        {.state = GuidanceCellState::kCandidate, .risk = 0.0}},
       {{.x = 1, .y = 0},
        {.state = GuidanceCellState::kCandidate, .risk = maximum}},
       {{.x = 2, .y = 0},
        {.state = GuidanceCellState::kCandidate, .risk = maximum}}});
  GlobalRoutePlanner planner;

  const GlobalRouteResult result = planner.Plan(
      *snapshot, CellPoint(0, 0), CellPoint(2, 0), GenerousDeadline(),
      StopToken{});

  EXPECT_EQ(result.status, GuidanceStatus::kAvailable);
  EXPECT_TRUE(result.route.has_value());
}

}  // namespace
}  // namespace lunar::incremental_navigation

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <stop_token>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "hierarchical/global_route.hpp"
#include "hierarchical/grid_search.hpp"
#include "shared/map_snapshot.hpp"
#include "shared/safe_projection.hpp"
#include "test_fixtures.hpp"

namespace lunar::planning::hierarchical {
namespace {

[[nodiscard]] shared::SafeProjection
Projection(const std::size_t width, const std::size_t height,
           const std::vector<shared::GridCell> &obstacles = {}) {
  GridMap map = test::MakeFlatMap("map", width, height, 1.0);
  auto *values =
      std::get_if<std::vector<std::uint8_t>>(&map.layers.at("obstacle").values);
  if (values == nullptr) {
    throw std::runtime_error{"obstacle fixture has wrong type"};
  }
  for (const shared::GridCell cell : obstacles) {
    values->at(static_cast<std::size_t>(cell.y) * width +
               static_cast<std::size_t>(cell.x)) = 1U;
  }
  const auto snapshot = shared::MapSnapshot::Create(map);
  if (!snapshot.ok()) {
    throw std::runtime_error{snapshot.reason_code};
  }
  auto capability = test::MakeValidWheelInput().capability;
  auto *wheel = std::get_if<WheeledCapability>(&capability);
  if (wheel == nullptr) {
    throw std::runtime_error{"wheel fixture has wrong type"};
  }
  wheel->minimum_clearance_m = 0.0;
  const auto projection = shared::BuildSafeProjection(
      snapshot.snapshot, capability, MapSafetyConfig{}, std::stop_token{});
  if (!projection.ok()) {
    throw std::runtime_error{projection.reason_code};
  }
  return std::move(*projection.projection);
}

[[nodiscard]] shared::SafeProjection SerpentineProjection(
    const std::size_t width, const std::size_t height) {
  std::vector<shared::GridCell> obstacles;
  obstacles.reserve(width * height / 2U);
  for (std::size_t y = 1U; y < height; y += 2U) {
    const std::size_t connector_x = ((y / 2U) % 2U == 0U) ? width - 1U : 0U;
    for (std::size_t x = 0U; x < width; ++x) {
      if (x != connector_x) {
        obstacles.push_back(shared::GridCell{
            .x = static_cast<std::int32_t>(x),
            .y = static_cast<std::int32_t>(y),
        });
      }
    }
  }
  return Projection(width, height, obstacles);
}

[[nodiscard]] std::vector<std::uint8_t>
GoalMask(const std::size_t width, const std::size_t height,
         const std::vector<shared::GridCell> &goals) {
  std::vector<std::uint8_t> mask(width * height, 0U);
  for (const shared::GridCell goal : goals) {
    mask.at(static_cast<std::size_t>(goal.y) * width +
            static_cast<std::size_t>(goal.x)) = 1U;
  }
  return mask;
}

[[nodiscard]] GlobalGridSearchResult
Search(const shared::SafeProjection &projection, const shared::GridCell start,
       const std::vector<std::uint8_t> &goals, GlobalSearchConfig config = {},
       const std::stop_token stop_token = {}) {
  config.slope_weight = 0.0;
  config.roughness_weight = 0.0;
  config.clearance_weight = 0.0;
  return SearchGlobalGrid(GlobalGridSearchProblem{
      .projection = projection,
      .start = start,
      .goal_mask = goals,
      .maximum_speed_mps = 1.0,
      .config = config,
      .stop_token = stop_token,
  });
}

TEST(GlobalGridSearch, RejectsDiagonalCornerCutting) {
  const auto projection = Projection(3U, 3U, {{1, 0}, {0, 1}});
  const auto goals = GoalMask(3U, 3U, {{1, 1}});

  const auto result = Search(projection, {0, 0}, goals);

  EXPECT_EQ(result.status, GlobalSearchStatus::kNoPath);
  EXPECT_EQ(result.reason_code, "GLOBAL_NO_KNOWN_SAFE_ROUTE");
  EXPECT_TRUE(result.path_cells.empty());
}

TEST(GlobalGridSearch, ReturnsIdenticalRouteForSymmetricTies) {
  const auto projection = Projection(5U, 3U, {{2, 1}});
  const auto goals = GoalMask(5U, 3U, {{4, 1}});

  const auto first = Search(projection, {0, 1}, goals);
  const auto second = Search(projection, {0, 1}, goals);

  ASSERT_EQ(first.status, GlobalSearchStatus::kSolved);
  ASSERT_EQ(second.status, GlobalSearchStatus::kSolved);
  EXPECT_EQ(first.path_cells, second.path_cells);
  EXPECT_DOUBLE_EQ(first.cost, second.cost);
  EXPECT_EQ(first.expanded_states, second.expanded_states);
  ASSERT_FALSE(first.path_cells.empty());
  EXPECT_EQ(first.path_cells.front(), (shared::GridCell{0, 1}));
  EXPECT_EQ(first.path_cells.back(), (shared::GridCell{4, 1}));
}

TEST(GlobalGridSearch, AcceptsAnyHardFeasibleGoalMaskCell) {
  const auto projection = Projection(6U, 2U);
  const auto goals = GoalMask(6U, 2U, {{4, 0}, {5, 0}});

  const auto result = Search(projection, {0, 0}, goals);

  ASSERT_EQ(result.status, GlobalSearchStatus::kSolved);
  EXPECT_EQ(result.path_cells.back(), (shared::GridCell{4, 0}));
}

TEST(GlobalGridSearch, StopsCooperativelyBeforeExpanding) {
  const auto projection = Projection(4U, 1U);
  const auto goals = GoalMask(4U, 1U, {{3, 0}});
  std::stop_source source;
  source.request_stop();

  const auto result = Search(projection, {0, 0}, goals, GlobalSearchConfig{},
                             source.get_token());

  EXPECT_EQ(result.status, GlobalSearchStatus::kCanceled);
  EXPECT_EQ(result.reason_code, "REQUEST_CANCELED");
  EXPECT_EQ(result.expanded_states, 0U);
}

TEST(GlobalGridSearch, CrossesMoreCellsThanTheFormerExpansionLimit) {
  constexpr std::size_t kWidth = 512U;
  constexpr std::size_t kHeight = 512U;
  const auto projection = SerpentineProjection(kWidth, kHeight);
  const auto goals = GoalMask(kWidth, kHeight, {{0, 510}});

  const auto result = Search(projection, {0, 0}, goals);

  ASSERT_EQ(result.status, GlobalSearchStatus::kSolved);
  EXPECT_GT(result.expanded_states, 100'000U);
  EXPECT_EQ(result.path_cells.front(), (shared::GridCell{0, 0}));
  EXPECT_EQ(result.path_cells.back(), (shared::GridCell{0, 510}));
}

TEST(GlobalGridSearch, ReportsNoPathOnlyAfterFiniteOpenSetExhaustion) {
  constexpr std::size_t kWidth = 9U;
  constexpr std::size_t kHeight = 5U;
  const auto projection = Projection(
      kWidth, kHeight, {{4, 0}, {4, 1}, {4, 2}, {4, 3}, {4, 4}});
  const auto goals = GoalMask(kWidth, kHeight, {{8, 2}});
  const auto start = shared::GridCell{0, 2};
  const std::int32_t start_component = projection.ConnectedComponent(start);
  std::size_t reachable_safe_cells = 0U;
  for (std::size_t y = 0U; y < kHeight; ++y) {
    for (std::size_t x = 0U; x < kWidth; ++x) {
      const shared::GridCell cell{
          .x = static_cast<std::int32_t>(x),
          .y = static_cast<std::int32_t>(y),
      };
      if (projection.HardFeasible(cell) &&
          projection.ConnectedComponent(cell) == start_component) {
        ++reachable_safe_cells;
      }
    }
  }

  const auto result = Search(projection, start, goals);

  ASSERT_GT(reachable_safe_cells, 0U);
  EXPECT_EQ(result.status, GlobalSearchStatus::kNoPath);
  EXPECT_EQ(result.reason_code, "GLOBAL_NO_KNOWN_SAFE_ROUTE");
  EXPECT_EQ(result.expanded_states, reachable_safe_cells);
}

TEST(GlobalGridSearch, SimplifiesOnlyAcrossSafeSupercoverCells) {
  const std::vector<shared::GridCell> around_obstacle{
      {0, 0}, {1, 0}, {2, 0}, {2, 1}, {2, 2}};
  const auto blocked_projection = Projection(3U, 3U, {{1, 1}});

  const auto blocked =
      SimplifyRouteSupercover(blocked_projection, around_obstacle, 16U);

  ASSERT_GE(blocked.size(), 3U);
  EXPECT_EQ(blocked.front(), around_obstacle.front());
  EXPECT_EQ(blocked.back(), around_obstacle.back());

  const auto open_projection = Projection(3U, 3U);
  const auto open =
      SimplifyRouteSupercover(open_projection, around_obstacle, 16U);
  ASSERT_EQ(open.size(), 2U);
  EXPECT_EQ(open.front(), around_obstacle.front());
  EXPECT_EQ(open.back(), around_obstacle.back());
}

TEST(GlobalGridSearch, RejectsAnOutputLimitThatCannotRetainSafeTurns) {
  const auto projection = Projection(3U, 3U, {{1, 1}});
  const std::vector<shared::GridCell> route{
      {0, 0}, {1, 0}, {2, 0}, {2, 1}, {2, 2}};

  EXPECT_TRUE(SimplifyRouteSupercover(projection, route, 2U).empty());
}

} // namespace
} // namespace lunar::planning::hierarchical

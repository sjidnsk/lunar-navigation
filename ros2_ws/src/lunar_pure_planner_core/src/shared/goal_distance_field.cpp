#include "shared/goal_distance_field.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <queue>
#include <span>
#include <tuple>
#include <vector>

#include "shared/controlled_work.hpp"

namespace lunar::pure_planning::shared {
namespace {

using QueueEntry =
    std::tuple<double, std::size_t, std::size_t, std::size_t>;

[[nodiscard]] GridCell CellFromIndex(const MapSnapshot& map,
                                     const std::size_t index) noexcept {
  return GridCell{.x = static_cast<std::int32_t>(index % map.width()),
                  .y = static_cast<std::int32_t>(index / map.width())};
}

}  // namespace

std::optional<GoalDistanceField> BuildGoalDistanceField(
    const MapSnapshot& map, const std::span<const std::uint8_t> feasible,
    const std::span<const GridCell> goals,
    const SearchControl control) {
  if (feasible.size() != map.cell_count() ||
      StopReason(control).has_value()) {
    return std::nullopt;
  }

  const std::size_t invalid_goal_index =
      std::numeric_limits<std::size_t>::max();
  GoalDistanceField result{
      .distance_m = std::vector<double>(map.cell_count(),
                                        std::numeric_limits<double>::infinity()),
      .nearest_goal_index = std::vector<std::size_t>(
          map.cell_count(), invalid_goal_index),
  };
  std::priority_queue<QueueEntry, std::vector<QueueEntry>, std::greater<>> open;
  std::size_t insertion_sequence = 0U;
  for (std::size_t source_index = 0U; source_index < goals.size();
       ++source_index) {
    if (ControlCheckDue(source_index) && StopReason(control).has_value()) {
      return std::nullopt;
    }
    const GridCell goal = goals[source_index];
    if (!map.InBounds(goal)) {
      continue;
    }
    const std::size_t goal_index = map.Index(goal);
    if (feasible[goal_index] == 0U ||
        result.nearest_goal_index[goal_index] != invalid_goal_index) {
      continue;
    }
    result.distance_m[goal_index] = 0.0;
    result.nearest_goal_index[goal_index] = source_index;
    open.emplace(0.0, source_index, goal_index, insertion_sequence++);
  }
  if (open.empty()) {
    return std::nullopt;
  }

  constexpr std::array<std::int32_t, 8U> kDx{-1, 0, 1, -1, 1, -1, 0, 1};
  constexpr std::array<std::int32_t, 8U> kDy{-1, -1, -1, 0, 0, 1, 1, 1};
  std::size_t expanded{};
  while (!open.empty()) {
    if (ControlCheckDue(expanded++) && StopReason(control).has_value()) {
      return std::nullopt;
    }
    const auto [distance, source_index, index, sequence] = open.top();
    static_cast<void>(sequence);
    open.pop();
    if (distance != result.distance_m[index] ||
        source_index != result.nearest_goal_index[index]) {
      continue;
    }
    const GridCell current = CellFromIndex(map, index);
    for (std::size_t neighbor = 0U; neighbor < kDx.size(); ++neighbor) {
      const GridCell next{.x = current.x + kDx[neighbor],
                          .y = current.y + kDy[neighbor]};
      if (!map.InBounds(next)) {
        continue;
      }
      const std::size_t next_index = map.Index(next);
      if (feasible[next_index] == 0U) {
        continue;
      }
      if (kDx[neighbor] != 0 && kDy[neighbor] != 0) {
        const GridCell side_x{.x = current.x + kDx[neighbor],
                              .y = current.y};
        const GridCell side_y{.x = current.x,
                              .y = current.y + kDy[neighbor]};
        if (feasible[map.Index(side_x)] == 0U ||
            feasible[map.Index(side_y)] == 0U) {
          continue;
        }
      }
      const double step = map.resolution_m() *
                          (kDx[neighbor] != 0 && kDy[neighbor] != 0
                               ? std::numbers::sqrt2
                               : 1.0);
      const double candidate = distance + step;
      if (candidate < result.distance_m[next_index] ||
          (candidate == result.distance_m[next_index] &&
           source_index < result.nearest_goal_index[next_index])) {
        result.distance_m[next_index] = candidate;
        result.nearest_goal_index[next_index] = source_index;
        open.emplace(candidate, source_index, next_index,
                     insertion_sequence++);
      }
    }
  }
  return result;
}

std::optional<GoalDistanceField> BuildGoalDistanceField(
    const LocalTerrainProjection& terrain,
    const std::span<const GridCell> goals,
    const SearchControl control) {
  if (terrain.map == nullptr) {
    return std::nullopt;
  }
  return BuildGoalDistanceField(*terrain.map, terrain.free_with_height, goals,
                                control);
}

std::optional<GoalDistanceField> BuildGoalDistanceField(
    const LocalTerrainProjection& terrain, const GridCell goal,
    const SearchControl control) {
  return BuildGoalDistanceField(
      terrain, std::span<const GridCell>{&goal, 1U}, control);
}

}  // namespace lunar::pure_planning::shared

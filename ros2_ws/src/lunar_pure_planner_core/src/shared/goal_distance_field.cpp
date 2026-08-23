#include "shared/goal_distance_field.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <queue>
#include <utility>
#include <vector>

#include "shared/controlled_work.hpp"

namespace lunar::pure_planning::shared {
namespace {

using QueueEntry = std::pair<double, std::size_t>;

[[nodiscard]] GridCell CellFromIndex(const MapSnapshot& map,
                                     const std::size_t index) noexcept {
  return GridCell{.x = static_cast<std::int32_t>(index % map.width()),
                  .y = static_cast<std::int32_t>(index / map.width())};
}

}  // namespace

std::optional<GoalDistanceField> BuildGoalDistanceField(
    const LocalTerrainProjection& terrain, const GridCell goal,
    const SearchControl control) {
  if (terrain.map == nullptr ||
      terrain.free_with_height.size() != terrain.map->cell_count() ||
      !terrain.map->InBounds(goal) ||
      terrain.free_with_height[terrain.map->Index(goal)] == 0U ||
      StopReason(control).has_value()) {
    return std::nullopt;
  }

  GoalDistanceField result{
      .distance_m = std::vector<double>(terrain.map->cell_count(),
                                        std::numeric_limits<double>::infinity()),
  };
  const std::size_t goal_index = terrain.map->Index(goal);
  result.distance_m[goal_index] = 0.0;
  std::priority_queue<QueueEntry, std::vector<QueueEntry>, std::greater<>> open;
  open.emplace(0.0, goal_index);

  constexpr std::array<std::int32_t, 8U> kDx{-1, 0, 1, -1, 1, -1, 0, 1};
  constexpr std::array<std::int32_t, 8U> kDy{-1, -1, -1, 0, 0, 1, 1, 1};
  std::size_t expanded{};
  while (!open.empty()) {
    if (ControlCheckDue(expanded++) && StopReason(control).has_value()) {
      return std::nullopt;
    }
    const auto [distance, index] = open.top();
    open.pop();
    if (distance > result.distance_m[index]) {
      continue;
    }
    const GridCell current = CellFromIndex(*terrain.map, index);
    for (std::size_t neighbor = 0U; neighbor < kDx.size(); ++neighbor) {
      const GridCell next{.x = current.x + kDx[neighbor],
                          .y = current.y + kDy[neighbor]};
      if (!terrain.map->InBounds(next)) {
        continue;
      }
      const std::size_t next_index = terrain.map->Index(next);
      if (terrain.free_with_height[next_index] == 0U) {
        continue;
      }
      const double step = terrain.map->resolution_m() *
                          (kDx[neighbor] != 0 && kDy[neighbor] != 0
                               ? std::numbers::sqrt2
                               : 1.0);
      const double candidate = distance + step;
      if (candidate < result.distance_m[next_index]) {
        result.distance_m[next_index] = candidate;
        open.emplace(candidate, next_index);
      }
    }
  }
  return result;
}

}  // namespace lunar::pure_planning::shared

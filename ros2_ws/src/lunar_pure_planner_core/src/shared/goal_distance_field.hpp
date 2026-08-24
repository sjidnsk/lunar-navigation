#pragma once

#include <cstddef>
#include <optional>
#include <span>
#include <vector>

#include "lunar_pure_planner_core/search_control.hpp"
#include "shared/local_terrain_projection.hpp"

namespace lunar::pure_planning::shared {

struct GoalDistanceField final {
  std::vector<double> distance_m;
  std::vector<std::size_t> nearest_goal_index;
};

[[nodiscard]] std::optional<GoalDistanceField> BuildGoalDistanceField(
    const LocalTerrainProjection& terrain, std::span<const GridCell> goals,
    SearchControl control = {});

[[nodiscard]] std::optional<GoalDistanceField> BuildGoalDistanceField(
    const LocalTerrainProjection& terrain, GridCell goal,
    SearchControl control = {});

}  // namespace lunar::pure_planning::shared

#pragma once

#include <optional>
#include <vector>

#include "lunar_pure_planner_core/search_control.hpp"
#include "shared/local_terrain_projection.hpp"

namespace lunar::pure_planning::shared {

struct GoalDistanceField final {
  std::vector<double> distance_m;
};

[[nodiscard]] std::optional<GoalDistanceField> BuildGoalDistanceField(
    const LocalTerrainProjection& terrain, GridCell goal,
    SearchControl control = {});

}  // namespace lunar::pure_planning::shared

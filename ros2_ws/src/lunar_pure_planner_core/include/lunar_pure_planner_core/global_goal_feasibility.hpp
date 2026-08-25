#pragma once

#include <cstdint>
#include <string>

#include "lunar_pure_planner_core/types/geometry.hpp"
#include "lunar_pure_planner_core/types/world_snapshot.hpp"

namespace lunar::pure_planning {

struct GlobalGoalFeasibilityRequest final {
  GridMap global_map;
  Vec2 goal_position_m;
  std::int32_t obstacle_threshold_percent{50};
  double inflation_m{};
};

struct GlobalGoalFeasibilityResult final {
  bool feasible{false};
  std::string reason_code;
};

[[nodiscard]] GlobalGoalFeasibilityResult EvaluateGlobalGoalFeasibility(
    GlobalGoalFeasibilityRequest request);

}  // namespace lunar::pure_planning

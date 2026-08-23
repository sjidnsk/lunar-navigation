#pragma once

#include <optional>
#include <string>

#include "lunar_pure_planner_core/planner.hpp"

namespace lunar::pure_planning::hierarchical {

struct LocalGoalSelectionResult final {
  std::optional<GoalRegion> goal;
  std::string reason_code;

  [[nodiscard]] bool ok() const noexcept {
    return goal.has_value() && reason_code.empty();
  }
};

[[nodiscard]] GlobalStageResult PlanSurfaceGlobal(
    const PlanningRequest& input, SearchControl control);

[[nodiscard]] GlobalStageResult PlanSurfaceGlobal(
    const PlanningRequest& input, SearchControl control, double inflation_m);

[[nodiscard]] LocalGoalSelectionResult SelectSurfaceLocalGoal(
    const PlanningRequest& input, const GlobalRoute& route,
    SearchControl control);

[[nodiscard]] std::optional<GoalRegion> GoalMapToOdomPlanar(
    const PlanningRequest& input, const GoalRegion& goal_map);

[[nodiscard]] bool GoalInsideLocalMap(const PlanningRequest& input,
                                      const GoalRegion& goal_map);

}  // namespace lunar::pure_planning::hierarchical

#pragma once

#include <optional>
#include <string>

#include "lunar_pure_planner_core/planner.hpp"

namespace lunar::pure_planning::hierarchical {

struct SurfacePortalSetResult;
struct SurfaceRollingDecision;

struct LocalGoalSetResult final {
  std::optional<LocalGoalSet> goals;
  std::string reason_code;

  [[nodiscard]] bool ok() const noexcept {
    return goals.has_value() && !goals->goals_odom.empty() &&
           reason_code.empty();
  }
};

[[nodiscard]] GlobalStageResult PlanSurfaceGlobal(
    const PlanningRequest& input, SearchControl control);

[[nodiscard]] GlobalStageResult PlanSurfaceGlobal(
    const PlanningRequest& input, SearchControl control, double inflation_m);

[[nodiscard]] LocalGoalSetResult SelectSurfaceLocalGoals(
    const PlanningRequest& input, const GlobalRoute& route,
    SearchControl control);

[[nodiscard]] LocalGoalSetResult ConvertSurfacePortalsToLocalGoals(
    const SurfacePortalSetResult& portals,
    const SurfaceRollingDecision& decision, const Pose3& current_pose_odom,
    const GridMap& global_map, const GridMap& local_map);

[[nodiscard]] std::optional<GoalRegion> GoalMapToOdomPlanar(
    const PlanningRequest& input, const GoalRegion& goal_map);

[[nodiscard]] bool GoalInsideLocalMap(const PlanningRequest& input,
                                      const GoalRegion& goal_map);

}  // namespace lunar::pure_planning::hierarchical

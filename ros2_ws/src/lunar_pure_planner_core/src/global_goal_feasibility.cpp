#include "lunar_pure_planner_core/global_goal_feasibility.hpp"

#include <utility>

#include "shared/global_occupancy_projection.hpp"
#include "shared/map_snapshot.hpp"

namespace lunar::pure_planning {

GlobalGoalFeasibilityResult EvaluateGlobalGoalFeasibility(
    GlobalGoalFeasibilityRequest request) {
  const auto snapshot = shared::MapSnapshot::Create(
      std::move(request.global_map), shared::MapContract::kGlobalOccupancy);
  if (!snapshot.ok()) {
    return {.reason_code = "GLOBAL_OCCUPANCY_PROJECTION_INVALID"};
  }
  const auto projection = shared::BuildInflatedGlobalOccupancyProjection(
      snapshot.snapshot, request.obstacle_threshold_percent, request.inflation_m);
  if (!projection.ok()) {
    return {.reason_code = std::move(projection.reason_code)};
  }
  const auto cell = snapshot.snapshot->PositionToCell(request.goal_position_m);
  if (!cell.has_value()) {
    return {.reason_code = "GLOBAL_GOAL_OUT_OF_BOUNDS"};
  }
  if (!projection.projection->View().HardFeasible(*cell)) {
    return {.reason_code = "GLOBAL_GOAL_INFEASIBLE"};
  }
  return {.feasible = true};
}

}  // namespace lunar::pure_planning

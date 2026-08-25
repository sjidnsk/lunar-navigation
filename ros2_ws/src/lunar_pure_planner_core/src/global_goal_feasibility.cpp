#include "lunar_pure_planner_core/global_goal_feasibility.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "shared/cell_area_distance_transform.hpp"
#include "shared/map_snapshot.hpp"

namespace lunar::pure_planning {

GlobalGoalFeasibilityResult EvaluateGlobalGoalFeasibility(
    GlobalGoalFeasibilityRequest request) {
  const auto snapshot = shared::MapSnapshot::Create(
      std::move(request.global_map), shared::MapContract::kGlobalOccupancy);
  if (!snapshot.ok()) {
    return {.reason_code = "GLOBAL_OCCUPANCY_PROJECTION_INVALID"};
  }
  if (request.obstacle_threshold_percent < 0 ||
      request.obstacle_threshold_percent > 100 ||
      !std::isfinite(request.inflation_m) || request.inflation_m < 0.0) {
    return {.reason_code = "GLOBAL_OCCUPANCY_PROJECTION_INVALID"};
  }
  const auto cell = snapshot.snapshot->PositionToCell(request.goal_position_m);
  if (!cell.has_value()) {
    return {.reason_code = "GLOBAL_GOAL_OUT_OF_BOUNDS"};
  }
  const auto occupancy = snapshot.snapshot->Int8Layer("occupancy");
  if (occupancy.size() != snapshot.snapshot->cell_count()) {
    return {.reason_code = "GLOBAL_OCCUPANCY_PROJECTION_INVALID"};
  }
  const std::size_t goal_index = snapshot.snapshot->Index(*cell);
  const auto goal_value = static_cast<std::int32_t>(occupancy[goal_index]);
  if (goal_value < 0 || goal_value > 100 ||
      goal_value >= request.obstacle_threshold_percent) {
    return {.reason_code = "GLOBAL_GOAL_INFEASIBLE"};
  }

  std::vector<std::uint8_t> occupied_mask(occupancy.size(), 0U);
  for (std::size_t index = 0U; index < occupancy.size(); ++index) {
    const auto value = static_cast<std::int32_t>(occupancy[index]);
    occupied_mask[index] = static_cast<std::uint8_t>(
        value >= request.obstacle_threshold_percent && value <= 100);
  }
  const auto clearance = shared::BuildCellAreaClearance(
      snapshot.snapshot->width(), snapshot.snapshot->height(),
      snapshot.snapshot->resolution_m(), occupied_mask);
  if (!clearance.ok()) {
    return {.reason_code = clearance.reason_code};
  }
  if (static_cast<double>(clearance.clearance_m[goal_index]) <
      request.inflation_m) {
    return {.reason_code = "GLOBAL_GOAL_INFEASIBLE"};
  }
  return {.feasible = true};
}

}  // namespace lunar::pure_planning

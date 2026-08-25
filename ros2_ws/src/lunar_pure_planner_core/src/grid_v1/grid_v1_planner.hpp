#pragma once

#include <cstddef>
#include <vector>

#include "lunar_pure_planner_core/traversability_map.hpp"
#include "lunar_pure_planner_core/types/planning_request.hpp"

namespace lunar::pure_planning::grid_v1 {

[[nodiscard]] PlanningResult Plan(const PlanningRequest& request) noexcept;

[[nodiscard]] bool PathIsFree(const TraversabilitySnapshot& snapshot,
                              const std::vector<Pose3>& path,
                              std::size_t* checked_cells) noexcept;

}  // namespace lunar::pure_planning::grid_v1

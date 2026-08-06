#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "hierarchical/global_route.hpp"
#include "hierarchical/local_planning_problem.hpp"
#include "lunar_planner_core/types/planner_io.hpp"

namespace lunar::planning::hierarchical {

enum class LocalFrontierStatus : std::uint8_t {
  kReady,
  kCoverageInsufficient,
  kGoalInfeasible,
  kInvalidRequest,
  kCanceled,
};

struct LocalFrontierResult final {
  LocalFrontierStatus status{LocalFrontierStatus::kInvalidRequest};
  std::vector<LocalPlanningProblem> problems;
  std::vector<double> frontier_distances_m;
  std::vector<shared::GridCell> conditional_corridor_cells;
  double corridor_half_width_m{};
  std::string reason_code;

  [[nodiscard]] bool ok() const noexcept {
    return status == LocalFrontierStatus::kReady && !problems.empty() &&
           problems.size() == frontier_distances_m.size() &&
           reason_code == "LOCAL_FRONTIERS_AVAILABLE";
  }
};

[[nodiscard]] LocalFrontierResult BuildLocalFrontiers(const PlannerInput &input,
                                                      const GlobalRoute &route);

} // namespace lunar::planning::hierarchical

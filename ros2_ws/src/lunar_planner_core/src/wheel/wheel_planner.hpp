#pragma once

#include <optional>
#include <span>

#include "hierarchical/local_planning_problem.hpp"
#include "lunar_planner_core/types/planner_io.hpp"
#include "shared/projection_cache.hpp"

namespace lunar::planning::wheel {

struct WheelRankedPlanResult final {
  PlannerOutput output;
  std::optional<std::size_t> selected_problem_index;
  bool projection_cache_hit{};
};

class WheelPlanner final {
 public:
  [[nodiscard]] PlannerOutput Plan(
      const hierarchical::LocalPlanningProblem& problem) const;

  [[nodiscard]] WheelRankedPlanResult PlanRanked(
      std::span<const hierarchical::LocalPlanningProblem> problems) const;

 private:
  mutable shared::ProjectionCache projection_cache_;
};

}  // namespace lunar::planning::wheel

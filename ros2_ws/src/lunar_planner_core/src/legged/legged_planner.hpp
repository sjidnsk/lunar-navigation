#pragma once

#include "hierarchical/local_planning_problem.hpp"
#include "lunar_planner_core/types/planner_io.hpp"

namespace lunar::planning::legged {

class LeggedPlanner final {
 public:
  [[nodiscard]] PlannerOutput Plan(
      const hierarchical::LocalPlanningProblem& problem) const;
};

}  // namespace lunar::planning::legged

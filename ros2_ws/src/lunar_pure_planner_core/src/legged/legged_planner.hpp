#pragma once

#include "hierarchical/local_planning_problem.hpp"
#include "lunar_pure_planner_core/types/planner_io.hpp"

namespace lunar::pure_planning::legged {

class LeggedPlanner final {
 public:
  [[nodiscard]] PlannerOutput Plan(
      const hierarchical::LocalPlanningProblem& problem) const;
};

}  // namespace lunar::pure_planning::legged

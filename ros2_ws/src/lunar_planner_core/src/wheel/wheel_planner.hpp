#pragma once

#include "hierarchical/local_planning_problem.hpp"
#include "lunar_planner_core/types/planner_io.hpp"

namespace lunar::planning::wheel {

class WheelPlanner final {
 public:
  [[nodiscard]] PlannerOutput Plan(
      const hierarchical::LocalPlanningProblem& problem) const;
};

}  // namespace lunar::planning::wheel

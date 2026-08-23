#pragma once

#include "lunar_pure_planner_core/types/planner_io.hpp"

namespace lunar::pure_planning::hopper {

class HopperPlanner final {
 public:
  [[nodiscard]] PlannerOutput Plan(const PlannerInput& input) const;
};

}  // namespace lunar::pure_planning::hopper

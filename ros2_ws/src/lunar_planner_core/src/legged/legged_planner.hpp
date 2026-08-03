#pragma once

#include "lunar_planner_core/types/planner_io.hpp"

namespace lunar::planning::legged {

class LeggedPlanner final {
 public:
  [[nodiscard]] PlannerOutput Plan(const PlannerInput& input) const;
};

}  // namespace lunar::planning::legged

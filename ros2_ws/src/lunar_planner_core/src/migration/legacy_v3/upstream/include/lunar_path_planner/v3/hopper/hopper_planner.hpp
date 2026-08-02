#pragma once

#include "lunar_path_planner/v3/goal/terminal_resolver.hpp"
#include "lunar_path_planner/v3/hopper/hop_certifier.hpp"

namespace lunar::planning::v3 {

class HopperPlanner final {
 public:
  [[nodiscard]] Result<HopperReference> Plan(
      const PlanningRequest& request,
      const ResolvedTerminalSet& terminal_set) const noexcept;
};

}  // namespace lunar::planning::v3

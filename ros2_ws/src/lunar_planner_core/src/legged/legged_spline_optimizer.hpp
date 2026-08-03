#pragma once

#include <stop_token>
#include <string>
#include <vector>

#include "legged/legged_types.hpp"
#include "lunar_planner_core/types/planner_config.hpp"
#include "shared/convex_corridor.hpp"

namespace lunar::planning::legged {

struct LeggedOptimizationResult final {
  std::vector<LeggedTransition> transitions;
  bool optimized{};
  bool canceled{};
  std::string reason_code;
};

[[nodiscard]] LeggedOptimizationResult OptimizeLeggedBodySpline(
    const std::vector<LeggedTransition>& discrete_transitions,
    const shared::CorridorResult& corridor,
    const OptimizationConfig& config,
    std::stop_token stop_token);

}  // namespace lunar::planning::legged

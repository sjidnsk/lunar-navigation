#pragma once

#include <stop_token>
#include <string>
#include <vector>

#include "lunar_planner_core/types/planner_config.hpp"
#include "shared/convex_corridor.hpp"
#include "wheel/wheel_types.hpp"

namespace lunar::planning::wheel {

struct WheelOptimizationResult final {
  std::vector<WheelTransition> transitions;
  bool optimized{};
  bool canceled{};
  std::string reason_code;
};

[[nodiscard]] WheelOptimizationResult OptimizeWheelSpline(
    const std::vector<WheelTransition>& discrete_transitions,
    const shared::CorridorResult& corridor,
    const OptimizationConfig& config,
    std::stop_token stop_token);

}  // namespace lunar::planning::wheel

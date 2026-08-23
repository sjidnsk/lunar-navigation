#pragma once

#include <cstddef>
#include <optional>
#include <stop_token>
#include <string>
#include <vector>

#include "legged/legged_types.hpp"
#include "lunar_pure_planner_core/types/motion_reference.hpp"
#include "lunar_pure_planner_core/types/platform_capability.hpp"

namespace lunar::pure_planning::legged {

struct LeggedTimingResult final {
  std::optional<TrajectoryReference> trajectory;
  bool canceled{};
  std::string reason_code;

  [[nodiscard]] bool ok() const noexcept {
    return trajectory.has_value() && !canceled && reason_code.empty();
  }
};

[[nodiscard]] LeggedTimingResult ParameterizeLeggedBodyTiming(
    const std::vector<LeggedTransition>& transitions,
    const LeggedCapability& capability,
    std::size_t maximum_samples,
    std::stop_token stop_token);

}  // namespace lunar::pure_planning::legged

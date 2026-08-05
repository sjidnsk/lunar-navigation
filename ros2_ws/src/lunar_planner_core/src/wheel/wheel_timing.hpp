#pragma once

#include <cstddef>
#include <optional>
#include <stop_token>
#include <string>
#include <vector>

#include "lunar_planner_core/types/motion_reference.hpp"
#include "lunar_planner_core/types/platform_capability.hpp"
#include "wheel/wheel_types.hpp"

namespace lunar::planning::wheel {

struct WheelTimingResult final {
  std::optional<TrajectoryReference> trajectory;
  bool canceled{};
  std::string reason_code;

  [[nodiscard]] bool ok() const noexcept {
    return trajectory.has_value() && !canceled && reason_code.empty();
  }
};

[[nodiscard]] WheelTimingResult ParameterizeWheelTiming(
    const std::vector<WheelTransition>& transitions,
    const WheeledCapability& capability,
    std::size_t maximum_samples,
    std::stop_token stop_token);

}  // namespace lunar::planning::wheel

#pragma once

#include <cstddef>
#include <stop_token>
#include <string>

#include "lunar_planner_core/types/platform_capability.hpp"
#include "shared/safe_projection.hpp"
#include "wheel/wheel_types.hpp"

namespace lunar::planning::wheel {

struct WheelSweepValidation final {
  bool valid{};
  bool canceled{};
  std::size_t sample_count{};
  std::string reason_code;
};

class WheelSweepValidator final {
 public:
  WheelSweepValidator(
      const shared::SafeProjection& projection,
      const WheeledCapability& capability) noexcept;

  [[nodiscard]] WheelSweepValidation Validate(
      const WheelTransition& transition,
      std::stop_token stop_token) const;

 private:
  const shared::SafeProjection* projection_{};
  const WheeledCapability* capability_{};
  double footprint_support_radius_m_{};
};

}  // namespace lunar::planning::wheel

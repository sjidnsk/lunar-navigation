#pragma once

#include "lunar_path_planner/v3/contracts/profiles.hpp"
#include "lunar_path_planner/v3/contracts/status.hpp"

namespace lunar::planning::v3 {

class WheelCapabilityView final {
 public:
  [[nodiscard]] static Result<WheelCapabilityView> Create(
      const SafetyCapabilityProfile& profile);

  [[nodiscard]] const ContentRef& content_ref() const noexcept;
  [[nodiscard]] const WheeledCapability& wheel() const noexcept;
  [[nodiscard]] const WheelHardLimits& hard_limits() const noexcept;
  [[nodiscard]] const ExtrudedConvexFootprint& collision_envelope()
      const noexcept;
  [[nodiscard]] double footprint_support_radius_m() const noexcept;
  [[nodiscard]] double certified_horizontal_position_error_m()
      const noexcept;

 private:
  ContentRef content_ref_;
  WheeledCapability wheel_;
  double footprint_support_radius_m_{};
  double certified_horizontal_position_error_m_{};
};

}  // namespace lunar::planning::v3

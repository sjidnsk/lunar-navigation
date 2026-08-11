#pragma once

#include <cstddef>
#include <stop_token>
#include <string>

#include "hierarchical/local_planning_problem.hpp"
#include "lunar_planner_core/types/platform_capability.hpp"
#include "shared/safe_projection.hpp"
#include "wheel/wheel_types.hpp"

namespace lunar::planning::wheel {

struct WheelSweepValidation final {
  bool valid{};
  bool canceled{};
  std::size_t sample_count{};
  double maximum_surface_slope_rad{};
  double maximum_roughness_m{};
  double maximum_positive_relief_m{};
  double minimum_underbody_clearance_m{};
  std::string reason_code;
};

class WheelSweepValidator final {
 public:
  WheelSweepValidator(
      const shared::SafeProjection& projection,
      const WheeledCapability& capability) noexcept;
  WheelSweepValidator(
      const shared::SafeProjection& physical_projection,
      const WheeledCapability& capability,
      const hierarchical::LocalSearchDomain& search_domain) noexcept;

  [[nodiscard]] WheelSweepValidation Validate(
      const WheelTransition& transition,
      std::stop_token stop_token) const;

 private:
  const shared::SafeProjection* projection_{};
  const WheeledCapability* capability_{};
  const hierarchical::LocalSearchDomain* search_domain_{};
  double footprint_support_radius_m_{};
};

}  // namespace lunar::planning::wheel

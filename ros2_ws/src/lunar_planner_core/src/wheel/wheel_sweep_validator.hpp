#pragma once

#include <cstddef>
#include <stop_token>
#include <string>
#include <unordered_map>

#include "hierarchical/local_planning_problem.hpp"
#include "lunar_planner_core/types/platform_capability.hpp"
#include "shared/safe_projection.hpp"
#include "shared/terrain_checks.hpp"
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

struct WheelTerrainPoseKey final {
  std::uint64_t x_bits{};
  std::uint64_t y_bits{};
  std::uint64_t yaw_bits{};

  bool operator==(const WheelTerrainPoseKey&) const = default;
};

struct WheelTerrainPoseKeyHash final {
  [[nodiscard]] std::size_t operator()(
      const WheelTerrainPoseKey& key) const noexcept;
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
  [[nodiscard]] std::size_t terrain_evaluation_count() const noexcept;

 private:
  const shared::SafeProjection* projection_{};
  const WheeledCapability* capability_{};
  const hierarchical::LocalSearchDomain* search_domain_{};
  double footprint_support_radius_m_{};
  mutable std::unordered_map<
      WheelTerrainPoseKey, shared::WheelTerrainPoseEvaluation,
      WheelTerrainPoseKeyHash> terrain_cache_;
  mutable std::size_t terrain_evaluation_count_{};
};

}  // namespace lunar::planning::wheel

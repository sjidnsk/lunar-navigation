#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <stop_token>
#include <string>
#include <vector>

#include "lunar_planner_core/types/planner_config.hpp"
#include "lunar_planner_core/types/platform_capability.hpp"
#include "shared/map_snapshot.hpp"

namespace lunar::planning::shared {

struct SafeProjectionBuildResult;

enum class ClearanceClass : std::uint8_t {
  kRejected = 0,
  kConditional = 1,
  kUnconditional = 2,
};

struct ClearanceBands final {
  double rejected_below_m{};
  double unconditional_at_or_above_m{};
};

[[nodiscard]] std::optional<ClearanceBands>
ResolveClearanceBands(const PlatformCapability &capability) noexcept;

[[nodiscard]] ClearanceClass
ClassifyClearance(double clearance_m, const ClearanceBands &bands) noexcept;

struct SafeProjectionClearanceMargins final {
  double hazard_m{};
  double boundary_m{};
};

class SafeProjection final {
public:
  [[nodiscard]] const std::shared_ptr<const MapSnapshot> &
  source_map() const noexcept;
  [[nodiscard]] bool InBounds(GridCell cell) const noexcept;
  [[nodiscard]] bool Known(GridCell cell) const noexcept;
  [[nodiscard]] bool IntrinsicFeasible(GridCell cell) const noexcept;
  [[nodiscard]] bool HardFeasible(GridCell cell) const noexcept;
  [[nodiscard]] ClearanceClass
  ClearanceClassification(GridCell cell) const noexcept;
  [[nodiscard]] float ClearanceMeters(GridCell cell) const noexcept;
  [[nodiscard]] float SlopeRadians(GridCell cell) const noexcept;
  [[nodiscard]] float RoughnessMeters(GridCell cell) const noexcept;
  [[nodiscard]] float TraversalCost(GridCell cell) const noexcept;
  [[nodiscard]] std::int32_t ConnectedComponent(GridCell cell) const noexcept;
  [[nodiscard]] double maximum_slope_rad() const noexcept;
  [[nodiscard]] PlatformType platform_type() const noexcept;

private:
  friend struct SafeProjectionBuildResult;
  friend SafeProjectionBuildResult
  BuildSafeProjection(std::shared_ptr<const MapSnapshot>,
                      const PlatformCapability &, const MapSafetyConfig &,
                      std::stop_token, SafeProjectionClearanceMargins);

  std::shared_ptr<const MapSnapshot> source_map_;
  PlatformType platform_type_{PlatformType::kWheeled};
  double maximum_slope_rad_{};
  std::vector<std::uint8_t> known_mask_;
  std::vector<std::uint8_t> intrinsic_feasible_mask_;
  std::vector<std::uint8_t> hard_feasible_mask_;
  std::vector<std::uint8_t> clearance_class_mask_;
  std::vector<float> clearance_m_;
  std::vector<float> slope_rad_;
  std::vector<float> roughness_m_;
  std::vector<float> traversal_cost_;
  std::vector<std::int32_t> connected_component_;
};

struct SafeProjectionBuildResult final {
  std::optional<SafeProjection> projection;
  std::string reason_code;

  [[nodiscard]] bool ok() const noexcept {
    return projection.has_value() && reason_code.empty();
  }
};

[[nodiscard]] SafeProjectionBuildResult
BuildSafeProjection(std::shared_ptr<const MapSnapshot> map,
                    const PlatformCapability &capability,
                    const MapSafetyConfig &config, std::stop_token stop_token,
                    SafeProjectionClearanceMargins clearance_margins = {});

} // namespace lunar::planning::shared

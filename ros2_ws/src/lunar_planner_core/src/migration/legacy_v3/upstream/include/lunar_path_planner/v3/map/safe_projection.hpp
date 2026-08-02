#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "lunar_path_planner/v3/contracts/platform_reference.hpp"
#include "lunar_path_planner/v3/contracts/profiles.hpp"
#include "lunar_path_planner/v3/contracts/status.hpp"
#include "lunar_path_planner/v3/cost/learned_cost_snapshot.hpp"
#include "lunar_path_planner/v3/map/immutable_snapshot.hpp"

namespace lunar::planning::v3 {

struct Cell final {
  std::int32_t x{};
  std::int32_t y{};

  auto operator<=>(const Cell&) const = default;
};

struct SafetyProjectionLimits final {
  PlatformType platform_type{PlatformType::kWheeled};
  double maximum_slope_rad{};
  std::optional<double> maximum_roughness_m;
  std::optional<double> maximum_step_height_m;
  double minimum_clearance_m{};
  double minimum_confidence{};
  double maximum_linear_speed_mps{};
  double maximum_linear_deceleration_mps2{};
  double maximum_yaw_rate_radps{};
  double maximum_yaw_deceleration_radps2{};
  bool supports_safe_stop_anchor{};
};

struct SafeProjectionRequest;

class SafeProjection final {
 public:
  GridGeometry geometry;
  std::vector<std::uint8_t> known_mask;
  std::vector<std::uint8_t> hard_feasible_mask;
  std::vector<float> esdf_clearance_m;
  std::vector<float> analytic_time_cost_s;
  std::vector<float> analytic_energy;
  std::vector<float> analytic_nonfatal_risk;
  std::vector<float> resolved_energy;
  std::vector<float> resolved_nonfatal_risk;
  std::vector<float> conservative_speed_limit_mps;
  std::vector<std::int32_t> connected_component;
  std::vector<std::uint8_t> safe_stop_candidate_mask;
  SoftCostSource soft_cost_source{SoftCostSource::kAnalyticOnly};
  std::optional<std::string> soft_cost_fallback_reason_code;

  [[nodiscard]] const std::shared_ptr<const ImmutableMapSnapshot>&
  source_map() const noexcept;
  [[nodiscard]] std::span<const float> ElevationMeters() const noexcept;
  [[nodiscard]] std::span<const float> RoughnessMeters() const noexcept;
  [[nodiscard]] SurfaceNormalGridView SurfaceNormals() const noexcept;
  [[nodiscard]] bool InBounds(Cell cell) const noexcept;
  [[nodiscard]] bool Known(Cell cell) const noexcept;
  [[nodiscard]] bool HardFeasible(Cell cell) const noexcept;
  [[nodiscard]] float ClearanceMeters(Cell cell) const noexcept;
  [[nodiscard]] PlatformType platform_type() const noexcept;
  [[nodiscard]] const ContentRef& capability_ref() const noexcept;

 private:
  friend Result<SafeProjection> BuildSafeProjection(
      const SafeProjectionRequest&);
  std::shared_ptr<const ImmutableMapSnapshot> source_map_;
  PlatformType platform_type_{PlatformType::kWheeled};
  ContentRef capability_ref_;
};

struct SafeProjectionRequest final {
  std::shared_ptr<const ImmutableMapSnapshot> map;
  std::shared_ptr<const SafetyCapabilityProfile> capability;
  std::shared_ptr<const PlannerAlgorithmConfig> algorithm_config;
  std::shared_ptr<const LearnedCostSnapshot> learned_cost;
};

[[nodiscard]] Result<SafetyProjectionLimits> ResolveProjectionLimits(
    const SafetyCapabilityProfile& capability);

[[nodiscard]] Result<SafeProjection> BuildSafeProjection(
    const SafeProjectionRequest& request);

[[nodiscard]] Result<SafeStopAnchor> ResolveSafeStopAnchor(
    const SafeProjection& projection,
    const WheeledOrLeggedState& state,
    const SafetyCapabilityProfile& capability);

}  // namespace lunar::planning::v3

#pragma once

#include <optional>
#include <string>
#include <vector>

#include "lunar_planner_core/types/planner_config.hpp"
#include "lunar_planner_core/types/platform_capability.hpp"
#include "shared/map_snapshot.hpp"

namespace lunar::planning::shared {

struct TerrainLimits final {
  PlatformType platform_type{PlatformType::kWheeled};
  double maximum_slope_rad{};
  std::optional<double> maximum_roughness_m;
  std::optional<double> maximum_step_height_m;
  double minimum_clearance_m{};
  double minimum_confidence{};
  double maximum_speed_mps{};
};

struct TerrainLimitsResult final {
  std::optional<TerrainLimits> limits;
  std::string reason_code;

  [[nodiscard]] bool ok() const noexcept {
    return limits.has_value() && reason_code.empty();
  }
};

struct TerrainCellEvaluation final {
  bool hard_feasible{};
  double slope_rad{};
  double roughness_m{};
  double clearance_m{};
  double conservative_speed_mps{};
  std::vector<std::string> rejection_codes;
};

struct WheelTerrainPoseEvaluation final {
  bool feasible{};
  double surface_slope_rad{};
  double roughness_m{};
  double maximum_positive_relief_m{};
  double minimum_underbody_clearance_m{};
  std::vector<std::string> rejection_codes;
};

[[nodiscard]] double EffectiveMaximumSlopeRad(
    const PlatformCapability& capability,
    const MapSafetyConfig& config) noexcept;

[[nodiscard]] TerrainLimitsResult ResolveTerrainLimits(
    const PlatformCapability& capability,
    const MapSafetyConfig& config);

[[nodiscard]] double ComputeSlopeRadians(
    const MapSnapshot& map, GridCell cell) noexcept;

[[nodiscard]] double ComputeMaximumNeighborStep(
    const MapSnapshot& map, GridCell cell) noexcept;

[[nodiscard]] double ComputeSpatialRoughnessMeters(
    const MapSnapshot& map, GridCell cell) noexcept;

[[nodiscard]] WheelTerrainPoseEvaluation EvaluateWheelTerrainPose(
    const MapSnapshot& map, Vec2 center_m, double yaw_rad,
    const WheeledCapability& capability);

[[nodiscard]] TerrainCellEvaluation EvaluateTerrainCell(
    const MapSnapshot& map, GridCell cell, const TerrainLimits& limits,
    const MapSafetyConfig& config, double clearance_m);

}  // namespace lunar::planning::shared

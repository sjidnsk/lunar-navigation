#include "shared/terrain_checks.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numbers>
#include <type_traits>
#include <utility>
#include <variant>

namespace lunar::planning::shared {
namespace {

constexpr double kProjectHardSlopeLimitRad = std::numbers::pi / 6.0;
constexpr double kComparisonTolerance = 1.0e-9;

[[nodiscard]] bool IsFiniteNonNegative(const double value) noexcept {
  return std::isfinite(value) && value >= 0.0;
}

[[nodiscard]] bool IsFinitePositive(const double value) noexcept {
  return std::isfinite(value) && value > 0.0;
}

[[nodiscard]] bool IsValidSlope(const double value) noexcept {
  return IsFinitePositive(value) && value < std::numbers::pi / 2.0;
}

[[nodiscard]] double MaximumAbsolute(const Interval& interval) noexcept {
  return std::max(std::abs(interval.lower), std::abs(interval.upper));
}

[[nodiscard]] bool IsFiniteOrdered(const Interval& interval) noexcept {
  return std::isfinite(interval.lower) && std::isfinite(interval.upper) &&
         interval.lower <= interval.upper;
}

[[nodiscard]] double CapabilitySlope(
    const PlatformCapability& capability) noexcept {
  return std::visit(
      [](const auto& concrete) noexcept {
        using Capability = std::decay_t<decltype(concrete)>;
        if constexpr (std::is_same_v<Capability, WheeledCapability> ||
                      std::is_same_v<Capability, LeggedCapability>) {
          return concrete.maximum_slope_rad;
        } else {
          return concrete.maximum_landing_slope_rad;
        }
      },
      capability);
}

[[nodiscard]] TerrainLimitsResult Invalid(std::string reason_code) {
  return TerrainLimitsResult{
      .limits = std::nullopt,
      .reason_code = std::move(reason_code),
  };
}

[[nodiscard]] double ElevationAt(
    const MapSnapshot& map, const GridCell cell) noexcept {
  if (!map.InBounds(cell)) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return static_cast<double>(map.FloatLayer("elevation")[map.Index(cell)]);
}

[[nodiscard]] double AxisGradient(
    const MapSnapshot& map, const GridCell cell,
    const std::int32_t dx, const std::int32_t dy) noexcept {
  const GridCell negative{.x = cell.x - dx, .y = cell.y - dy};
  const GridCell positive{.x = cell.x + dx, .y = cell.y + dy};
  const double center = ElevationAt(map, cell);
  if (map.InBounds(negative) && map.InBounds(positive)) {
    return (ElevationAt(map, positive) - ElevationAt(map, negative)) /
           (2.0 * map.resolution_m());
  }
  if (map.InBounds(positive)) {
    return (ElevationAt(map, positive) - center) / map.resolution_m();
  }
  if (map.InBounds(negative)) {
    return (center - ElevationAt(map, negative)) / map.resolution_m();
  }
  return 0.0;
}

}  // namespace

double EffectiveMaximumSlopeRad(
    const PlatformCapability& capability,
    const MapSafetyConfig& config) noexcept {
  const double capability_slope = CapabilitySlope(capability);
  if (!IsValidSlope(capability_slope) ||
      !IsValidSlope(config.project_maximum_slope_rad)) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return std::min(
      {capability_slope, config.project_maximum_slope_rad,
       kProjectHardSlopeLimitRad});
}

TerrainLimitsResult ResolveTerrainLimits(
    const PlatformCapability& capability,
    const MapSafetyConfig& config) {
  if (!IsValidSlope(config.project_maximum_slope_rad) ||
      !IsFiniteNonNegative(config.maximum_elevation_variance_m2) ||
      !IsFiniteNonNegative(config.maximum_obstacle_variance_m2) ||
      !IsFiniteNonNegative(config.maximum_observation_age_s) ||
      !std::isfinite(config.minimum_observation_quality) ||
      config.minimum_observation_quality < 0.0 ||
      config.minimum_observation_quality > 1.0 ||
      config.minimum_observation_count == 0U) {
    return Invalid("MAP_SAFETY_CONFIG_INVALID");
  }

  const double maximum_slope =
      EffectiveMaximumSlopeRad(capability, config);
  if (!IsValidSlope(maximum_slope)) {
    return Invalid("CAPABILITY_SLOPE_INVALID");
  }

  return std::visit(
      [&](const auto& concrete) -> TerrainLimitsResult {
        using Capability = std::decay_t<decltype(concrete)>;
        if constexpr (std::is_same_v<Capability, WheeledCapability>) {
          const double maximum_speed = std::max(
              concrete.maximum_forward_speed_mps,
              concrete.maximum_reverse_speed_mps);
          if (!IsFinitePositive(concrete.maximum_forward_speed_mps) ||
              !IsFiniteNonNegative(concrete.maximum_reverse_speed_mps) ||
              !IsFinitePositive(maximum_speed) ||
              !IsFiniteNonNegative(concrete.minimum_clearance_m)) {
            return Invalid("WHEELED_TERRAIN_CAPABILITY_INVALID");
          }
          return TerrainLimitsResult{
              .limits = TerrainLimits{
                  .platform_type = PlatformType::kWheeled,
                  .maximum_slope_rad = maximum_slope,
                  .maximum_roughness_m = std::nullopt,
                  .maximum_step_height_m = std::nullopt,
                  .minimum_clearance_m = concrete.minimum_clearance_m,
                  .minimum_confidence =
                      config.minimum_observation_quality,
                  .maximum_speed_mps = maximum_speed,
              },
              .reason_code = {},
          };
        } else if constexpr (std::is_same_v<Capability, LeggedCapability>) {
          const double maximum_speed = std::hypot(
              MaximumAbsolute(concrete.forward_speed_mps),
              MaximumAbsolute(concrete.lateral_speed_mps),
              MaximumAbsolute(concrete.vertical_speed_mps));
          if (!IsFiniteOrdered(concrete.forward_speed_mps) ||
              !IsFiniteOrdered(concrete.lateral_speed_mps) ||
              !IsFiniteOrdered(concrete.vertical_speed_mps) ||
              !IsFiniteOrdered(concrete.yaw_rate_radps) ||
              !IsFinitePositive(maximum_speed) ||
              !IsFiniteNonNegative(concrete.maximum_roughness_m) ||
              !IsFiniteNonNegative(concrete.maximum_step_height_m) ||
              !IsFiniteNonNegative(concrete.minimum_body_clearance_m) ||
              !std::isfinite(concrete.minimum_confidence) ||
              concrete.minimum_confidence < 0.0 ||
              concrete.minimum_confidence > 1.0) {
            return Invalid("LEGGED_TERRAIN_CAPABILITY_INVALID");
          }
          return TerrainLimitsResult{
              .limits = TerrainLimits{
                  .platform_type = PlatformType::kLegged,
                  .maximum_slope_rad = maximum_slope,
                  .maximum_roughness_m = concrete.maximum_roughness_m,
                  .maximum_step_height_m = concrete.maximum_step_height_m,
                  .minimum_clearance_m = concrete.minimum_body_clearance_m,
                  .minimum_confidence = std::max(
                      config.minimum_observation_quality,
                      concrete.minimum_confidence),
                  .maximum_speed_mps = maximum_speed,
              },
              .reason_code = {},
          };
        } else {
          const double minimum_clearance = std::max(
              {concrete.minimum_overhead_clearance_m,
               concrete.minimum_lateral_clearance_m,
               concrete.minimum_landing_clearance_m});
          if (!IsFinitePositive(concrete.maximum_launch_speed_mps) ||
              !IsFiniteNonNegative(concrete.maximum_landing_roughness_m) ||
              !IsFiniteNonNegative(minimum_clearance)) {
            return Invalid("HOPPER_TERRAIN_CAPABILITY_INVALID");
          }
          return TerrainLimitsResult{
              .limits = TerrainLimits{
                  .platform_type = PlatformType::kHopper,
                  .maximum_slope_rad = maximum_slope,
                  .maximum_roughness_m =
                      concrete.maximum_landing_roughness_m,
                  .maximum_step_height_m = std::nullopt,
                  .minimum_clearance_m = minimum_clearance,
                  .minimum_confidence =
                      config.minimum_observation_quality,
                  .maximum_speed_mps = concrete.maximum_launch_speed_mps,
              },
              .reason_code = {},
          };
        }
      },
      capability);
}

double ComputeSlopeRadians(
    const MapSnapshot& map, const GridCell cell) noexcept {
  if (!map.InBounds(cell)) {
    return std::numeric_limits<double>::infinity();
  }
  const double gradient_x = AxisGradient(map, cell, 1, 0);
  const double gradient_y = AxisGradient(map, cell, 0, 1);
  if (!std::isfinite(gradient_x) || !std::isfinite(gradient_y)) {
    return std::numeric_limits<double>::infinity();
  }
  return std::atan(std::hypot(gradient_x, gradient_y));
}

double ComputeMaximumNeighborStep(
    const MapSnapshot& map, const GridCell cell) noexcept {
  if (!map.InBounds(cell)) {
    return std::numeric_limits<double>::infinity();
  }
  constexpr std::array<std::int32_t, 4> kDx{-1, 1, 0, 0};
  constexpr std::array<std::int32_t, 4> kDy{0, 0, -1, 1};
  const double center = ElevationAt(map, cell);
  double maximum_step = 0.0;
  for (std::size_t neighbor = 0U; neighbor < kDx.size(); ++neighbor) {
    const GridCell adjacent{
        .x = cell.x + kDx[neighbor],
        .y = cell.y + kDy[neighbor],
    };
    if (!map.InBounds(adjacent)) {
      continue;
    }
    maximum_step = std::max(
        maximum_step, std::abs(ElevationAt(map, adjacent) - center));
  }
  return maximum_step;
}

TerrainCellEvaluation EvaluateTerrainCell(
    const MapSnapshot& map, const GridCell cell,
    const TerrainLimits& limits, const MapSafetyConfig& config,
    const double clearance_m) {
  TerrainCellEvaluation evaluation{
      .hard_feasible = false,
      .slope_rad = ComputeSlopeRadians(map, cell),
      .roughness_m = 0.0,
      .clearance_m = clearance_m,
      .conservative_speed_mps = 0.0,
      .rejection_codes = {},
  };
  if (!map.InBounds(cell)) {
    evaluation.rejection_codes.emplace_back("OUT_OF_BOUNDS");
    return evaluation;
  }

  const std::size_t index = map.Index(cell);
  const auto valid = map.ByteLayer("valid_mask");
  const auto forbidden = map.ByteLayer("forbidden");
  const auto obstacle = map.ByteLayer("obstacle");
  const auto elevation_variance = map.FloatLayer("elevation_variance");
  const auto obstacle_variance = map.FloatLayer("obstacle_variance");
  const auto observation_age = map.FloatLayer("observation_age_s");
  const auto observation_quality = map.FloatLayer("observation_quality");
  const auto observation_count = map.CountLayer("observation_count");

  evaluation.roughness_m =
      std::sqrt(static_cast<double>(elevation_variance[index]));
  if (valid[index] == 0U) {
    evaluation.rejection_codes.emplace_back("UNKNOWN_CELL");
  }
  if (forbidden[index] != 0U) {
    evaluation.rejection_codes.emplace_back("FORBIDDEN_CELL");
  }
  if (obstacle[index] != 0U) {
    evaluation.rejection_codes.emplace_back("HARD_OBSTACLE");
  }
  if (static_cast<double>(elevation_variance[index]) >
      config.maximum_elevation_variance_m2 + kComparisonTolerance) {
    evaluation.rejection_codes.emplace_back("ELEVATION_VARIANCE_LIMIT");
  }
  if (static_cast<double>(obstacle_variance[index]) >
      config.maximum_obstacle_variance_m2 + kComparisonTolerance) {
    evaluation.rejection_codes.emplace_back("OBSTACLE_VARIANCE_LIMIT");
  }
  if (static_cast<double>(observation_age[index]) >
      config.maximum_observation_age_s + kComparisonTolerance) {
    evaluation.rejection_codes.emplace_back("OBSERVATION_TOO_OLD");
  }
  if (static_cast<double>(observation_quality[index]) +
          kComparisonTolerance <
      limits.minimum_confidence) {
    evaluation.rejection_codes.emplace_back("OBSERVATION_QUALITY_LIMIT");
  }
  if (observation_count[index] < config.minimum_observation_count) {
    evaluation.rejection_codes.emplace_back("OBSERVATION_COUNT_LIMIT");
  }
  if (!std::isfinite(evaluation.slope_rad) ||
      evaluation.slope_rad > limits.maximum_slope_rad + kComparisonTolerance) {
    evaluation.rejection_codes.emplace_back("SLOPE_LIMIT");
  }
  if (limits.maximum_roughness_m.has_value() &&
      evaluation.roughness_m >
          *limits.maximum_roughness_m + kComparisonTolerance) {
    evaluation.rejection_codes.emplace_back("ROUGHNESS_LIMIT");
  }
  if (limits.maximum_step_height_m.has_value() &&
      ComputeMaximumNeighborStep(map, cell) >
          *limits.maximum_step_height_m + kComparisonTolerance) {
    evaluation.rejection_codes.emplace_back("STEP_HEIGHT_LIMIT");
  }
  if (std::isnan(clearance_m) ||
      clearance_m + kComparisonTolerance < limits.minimum_clearance_m) {
    evaluation.rejection_codes.emplace_back("CLEARANCE_LIMIT");
  }

  evaluation.hard_feasible = evaluation.rejection_codes.empty();
  if (evaluation.hard_feasible) {
    evaluation.conservative_speed_mps =
        limits.maximum_speed_mps * std::max(0.05, std::cos(evaluation.slope_rad));
  }
  return evaluation;
}

}  // namespace lunar::planning::shared

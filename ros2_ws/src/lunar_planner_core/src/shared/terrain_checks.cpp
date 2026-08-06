#include "shared/terrain_checks.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <numbers>
#include <type_traits>
#include <utility>
#include <variant>

namespace lunar::planning::shared {
namespace {

constexpr double kProjectHardSlopeLimitRad = std::numbers::pi / 6.0;
constexpr double kComparisonTolerance = 1.0e-9;
// Elevation layers are float-valued.  A plane sampled from an exactly
// boundary-valued slope therefore needs a small angular tolerance after the
// deterministic least-squares fit.
constexpr double kSlopeComparisonToleranceRad = 2.0e-6;
constexpr double kElevationComparisonToleranceM = 1.0e-6;

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

struct PlaneFit final {
  double slope_x{};
  double slope_y{};
  double offset{};
};

struct ElevationPoint final {
  double x{};
  double y{};
  double z{};
};

[[nodiscard]] std::optional<PlaneFit> FitPlane(
    const std::vector<ElevationPoint>& points) noexcept {
  if (points.size() < 3U) {
    return std::nullopt;
  }
  double mean_x = 0.0;
  double mean_y = 0.0;
  double mean_z = 0.0;
  for (const ElevationPoint& point : points) {
    mean_x += point.x;
    mean_y += point.y;
    mean_z += point.z;
  }
  const double count = static_cast<double>(points.size());
  mean_x /= count;
  mean_y /= count;
  mean_z /= count;
  double xx = 0.0;
  double xy = 0.0;
  double yy = 0.0;
  double xz = 0.0;
  double yz = 0.0;
  for (const ElevationPoint& point : points) {
    const double dx = point.x - mean_x;
    const double dy = point.y - mean_y;
    const double dz = point.z - mean_z;
    xx += dx * dx;
    xy += dx * dy;
    yy += dy * dy;
    xz += dx * dz;
    yz += dy * dz;
  }
  const double determinant = xx * yy - xy * xy;
  if (!std::isfinite(determinant) ||
      determinant <= 1.0e-18 * std::max(1.0, xx * yy)) {
    return std::nullopt;
  }
  const double slope_x = (xz * yy - yz * xy) / determinant;
  const double slope_y = (yz * xx - xz * xy) / determinant;
  const double offset = mean_z - slope_x * mean_x - slope_y * mean_y;
  if (!std::isfinite(slope_x) || !std::isfinite(slope_y) ||
      !std::isfinite(offset)) {
    return std::nullopt;
  }
  return PlaneFit{
      .slope_x = slope_x,
      .slope_y = slope_y,
      .offset = offset,
  };
}

[[nodiscard]] double PlaneHeight(const PlaneFit& plane, const double x,
                                 const double y) noexcept {
  return plane.slope_x * x + plane.slope_y * y + plane.offset;
}

[[nodiscard]] bool PointOnSegment(const Vec2 point, const Vec2 start,
                                  const Vec2 finish) noexcept {
  const double cross = (finish.x - start.x) * (point.y - start.y) -
                       (finish.y - start.y) * (point.x - start.x);
  return std::abs(cross) <= kComparisonTolerance &&
         point.x + kComparisonTolerance >= std::min(start.x, finish.x) &&
         point.x <= std::max(start.x, finish.x) + kComparisonTolerance &&
         point.y + kComparisonTolerance >= std::min(start.y, finish.y) &&
         point.y <= std::max(start.y, finish.y) + kComparisonTolerance;
}

[[nodiscard]] bool PointInPolygon(const Vec2 point,
                                  const std::vector<Vec2>& polygon) noexcept {
  if (polygon.size() < 3U) {
    return false;
  }
  bool inside = false;
  for (std::size_t current = 0U, previous = polygon.size() - 1U;
       current < polygon.size(); previous = current++) {
    const Vec2& start = polygon[previous];
    const Vec2& finish = polygon[current];
    if (PointOnSegment(point, start, finish)) {
      return true;
    }
    if ((start.y > point.y) == (finish.y > point.y)) {
      continue;
    }
    const double crossing_x =
        start.x + (point.y - start.y) * (finish.x - start.x) /
                      (finish.y - start.y);
    if (point.x < crossing_x) {
      inside = !inside;
    }
  }
  return inside;
}

void AddReason(std::vector<std::string>& reasons, std::string reason) {
  if (std::ranges::find(reasons, reason) == reasons.end()) {
    reasons.push_back(std::move(reason));
  }
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

double ComputeSpatialRoughnessMeters(const MapSnapshot& map,
                                     const GridCell cell) noexcept {
  if (!map.InBounds(cell)) {
    return std::numeric_limits<double>::infinity();
  }
  std::vector<ElevationPoint> points;
  points.reserve(9U);
  const auto valid = map.ByteLayer("valid_mask");
  for (std::int32_t dy = -1; dy <= 1; ++dy) {
    for (std::int32_t dx = -1; dx <= 1; ++dx) {
      const GridCell sample{.x = cell.x + dx, .y = cell.y + dy};
      if (!map.InBounds(sample) || valid[map.Index(sample)] == 0U) {
        continue;
      }
      const Vec3 center = map.CellCenter(sample);
      points.push_back(ElevationPoint{center.x, center.y, center.z});
    }
  }
  const auto plane = FitPlane(points);
  if (!plane.has_value()) {
    return 0.0;
  }
  double mean = 0.0;
  std::vector<double> residuals;
  residuals.reserve(points.size());
  for (const ElevationPoint& point : points) {
    const double residual =
        point.z - PlaneHeight(*plane, point.x, point.y);
    residuals.push_back(residual);
    mean += residual;
  }
  mean /= static_cast<double>(residuals.size());
  double variance = 0.0;
  for (const double residual : residuals) {
    variance += (residual - mean) * (residual - mean);
  }
  return std::sqrt(variance / static_cast<double>(residuals.size()));
}

WheelTerrainPoseEvaluation EvaluateWheelTerrainPose(
    const MapSnapshot& map, const Vec2 center_m, const double yaw_rad,
    const WheeledCapability& capability) {
  WheelTerrainPoseEvaluation evaluation{
      .feasible = false,
      .surface_slope_rad = std::numeric_limits<double>::infinity(),
      .roughness_m = std::numeric_limits<double>::infinity(),
      .maximum_positive_relief_m =
          std::numeric_limits<double>::infinity(),
      .minimum_underbody_clearance_m =
          -std::numeric_limits<double>::infinity(),
  };
  if (!std::isfinite(center_m.x) || !std::isfinite(center_m.y) ||
      !std::isfinite(yaw_rad) ||
      capability.footprint_xy_m.size() < 3U ||
      !IsFinitePositive(capability.wheelbase_m) ||
      !IsFinitePositive(capability.track_width_m) ||
      !IsValidSlope(capability.maximum_slope_rad) ||
      !IsFiniteNonNegative(capability.maximum_local_obstacle_relief_m) ||
      !IsFinitePositive(capability.minimum_underbody_clearance_m)) {
    AddReason(evaluation.rejection_codes,
              "WHEEL_TERRAIN_CAPABILITY_INVALID");
    return evaluation;
  }

  const double cosine = std::cos(yaw_rad);
  const double sine = std::sin(yaw_rad);
  const auto world_point = [&](const Vec2 relative) {
    return Vec2{
        .x = center_m.x + cosine * relative.x - sine * relative.y,
        .y = center_m.y + sine * relative.x + cosine * relative.y,
    };
  };
  std::vector<ElevationPoint> wheel_support;
  wheel_support.reserve(4U);
  for (const double x : {-0.5 * capability.wheelbase_m,
                         0.5 * capability.wheelbase_m}) {
    for (const double y : {-0.5 * capability.track_width_m,
                           0.5 * capability.track_width_m}) {
      const Vec2 point = world_point(Vec2{x, y});
      const auto elevation = map.SampleElevationBilinear(point);
      if (!elevation.has_value()) {
        AddReason(evaluation.rejection_codes, "WHEEL_UNSUPPORTED_GAP");
        return evaluation;
      }
      wheel_support.push_back(ElevationPoint{point.x, point.y, *elevation});
    }
  }
  const auto support_plane = FitPlane(wheel_support);
  if (!support_plane.has_value()) {
    AddReason(evaluation.rejection_codes, "WHEEL_SUPPORT_PLANE_INVALID");
    return evaluation;
  }
  evaluation.surface_slope_rad = std::atan(std::hypot(
      support_plane->slope_x, support_plane->slope_y));
  if (evaluation.surface_slope_rad >
      capability.maximum_slope_rad + kSlopeComparisonToleranceRad) {
    AddReason(evaluation.rejection_codes, "WHEEL_SLOPE_LIMIT");
  }

  std::vector<Vec2> footprint_world;
  footprint_world.reserve(capability.footprint_xy_m.size());
  double minimum_x = std::numeric_limits<double>::infinity();
  double maximum_x = -std::numeric_limits<double>::infinity();
  double minimum_y = std::numeric_limits<double>::infinity();
  double maximum_y = -std::numeric_limits<double>::infinity();
  for (const Vec2 relative : capability.footprint_xy_m) {
    if (!std::isfinite(relative.x) || !std::isfinite(relative.y)) {
      AddReason(evaluation.rejection_codes,
                "WHEEL_TERRAIN_CAPABILITY_INVALID");
      return evaluation;
    }
    const Vec2 point = world_point(relative);
    footprint_world.push_back(point);
    minimum_x = std::min(minimum_x, point.x);
    maximum_x = std::max(maximum_x, point.x);
    minimum_y = std::min(minimum_y, point.y);
    maximum_y = std::max(maximum_y, point.y);
  }
  const auto minimum_cell = map.PositionToCell(
      Vec2{minimum_x + kComparisonTolerance,
           minimum_y + kComparisonTolerance});
  const auto maximum_cell = map.PositionToCell(
      Vec2{maximum_x - kComparisonTolerance,
           maximum_y - kComparisonTolerance});
  if (!minimum_cell.has_value() || !maximum_cell.has_value()) {
    AddReason(evaluation.rejection_codes, "WHEEL_UNSUPPORTED_GAP");
    return evaluation;
  }

  struct TerrainSample final {
    GridCell cell;
    double elevation_m{};
    double residual_m{};
  };
  std::vector<TerrainSample> samples;
  const auto valid = map.ByteLayer("valid_mask");
  for (std::int32_t y = minimum_cell->y; y <= maximum_cell->y; ++y) {
    for (std::int32_t x = minimum_cell->x; x <= maximum_cell->x; ++x) {
      const GridCell cell{x, y};
      const Vec3 sample = map.CellCenter(cell);
      if (!PointInPolygon(Vec2{sample.x, sample.y}, footprint_world)) {
        continue;
      }
      if (valid[map.Index(cell)] == 0U) {
        if (!capability.allow_unsupported_gap) {
          AddReason(evaluation.rejection_codes, "WHEEL_UNSUPPORTED_GAP");
        }
        continue;
      }
      const double residual = sample.z -
          PlaneHeight(*support_plane, sample.x, sample.y);
      samples.push_back(TerrainSample{cell, sample.z, residual});
    }
  }
  if (samples.empty()) {
    const auto center_cell = map.PositionToCell(center_m);
    const auto center_elevation = map.SampleElevationBilinear(center_m);
    if (!center_cell.has_value() || !center_elevation.has_value() ||
        valid[map.Index(*center_cell)] == 0U) {
      AddReason(evaluation.rejection_codes, "WHEEL_UNSUPPORTED_GAP");
      return evaluation;
    }
    samples.push_back(TerrainSample{
        .cell = *center_cell,
        .elevation_m = *center_elevation,
        .residual_m = *center_elevation - PlaneHeight(
            *support_plane, center_m.x, center_m.y),
    });
  }

  evaluation.maximum_positive_relief_m = 0.0;
  double mean_residual = 0.0;
  std::map<GridCell, double> elevation_by_cell;
  for (const TerrainSample& sample : samples) {
    evaluation.maximum_positive_relief_m = std::max(
        evaluation.maximum_positive_relief_m, sample.residual_m);
    mean_residual += sample.residual_m;
    elevation_by_cell.emplace(sample.cell, sample.elevation_m);
  }
  mean_residual /= static_cast<double>(samples.size());
  double residual_variance = 0.0;
  for (const TerrainSample& sample : samples) {
    residual_variance += (sample.residual_m - mean_residual) *
                         (sample.residual_m - mean_residual);
  }
  evaluation.roughness_m =
      std::sqrt(residual_variance / static_cast<double>(samples.size()));
  evaluation.minimum_underbody_clearance_m =
      capability.minimum_underbody_clearance_m -
      evaluation.maximum_positive_relief_m;

  constexpr std::array<GridCell, 2> kForwardNeighbors{
      GridCell{1, 0}, GridCell{0, 1}};
  const double maximum_continuous_step_m =
      std::tan(capability.maximum_slope_rad) * map.resolution_m();
  for (const auto& [cell, elevation_m] : elevation_by_cell) {
    for (const GridCell offset : kForwardNeighbors) {
      const auto found = elevation_by_cell.find(
          GridCell{cell.x + offset.x, cell.y + offset.y});
      if (found != elevation_by_cell.end() &&
          std::abs(found->second - elevation_m) >
              maximum_continuous_step_m + kElevationComparisonToleranceM) {
        AddReason(evaluation.rejection_codes,
                  "WHEEL_SURFACE_DISCONTINUITY");
      }
    }
  }
  if (evaluation.maximum_positive_relief_m >
      capability.maximum_local_obstacle_relief_m +
          kElevationComparisonToleranceM) {
    AddReason(evaluation.rejection_codes, "WHEEL_LOCAL_RELIEF_LIMIT");
  }
  if (evaluation.minimum_underbody_clearance_m <
      -kElevationComparisonToleranceM) {
    AddReason(evaluation.rejection_codes,
              "WHEEL_UNDERBODY_CLEARANCE_LIMIT");
  }
  evaluation.feasible = evaluation.rejection_codes.empty();
  return evaluation;
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

  evaluation.roughness_m = ComputeSpatialRoughnessMeters(map, cell);
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

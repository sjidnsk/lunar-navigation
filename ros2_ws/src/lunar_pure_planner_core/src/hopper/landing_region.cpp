#include "hopper/landing_region.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace lunar::pure_planning::hopper {
namespace {

constexpr double kTolerance = 1.0e-9;

struct PlaneFit final {
  double a{};
  double b{};
  double c{};
  double slope_rad{};
  double maximum_residual_m{};
  double maximum_roughness_m{};
};

struct DiskCheck final {
  std::optional<PlaneFit> plane;
  std::size_t inspected_cells{};
  std::string reason_code;

  [[nodiscard]] bool ok() const noexcept {
    return plane.has_value() && reason_code.empty();
  }
};

[[nodiscard]] bool FinitePositive(const double value) noexcept {
  return std::isfinite(value) && value > 0.0;
}

[[nodiscard]] bool CellIntersectsDisk(
    const shared::MapSnapshot& map, const shared::GridCell cell,
    const Vec2 center, const double radius_m) noexcept {
  const double x0 = map.origin_m().x + static_cast<double>(cell.x) *
      map.resolution_m();
  const double y0 = map.origin_m().y + static_cast<double>(cell.y) *
      map.resolution_m();
  const double x1 = x0 + map.resolution_m();
  const double y1 = y0 + map.resolution_m();
  const double nearest_x = std::clamp(center.x, x0, x1);
  const double nearest_y = std::clamp(center.y, y0, y1);
  return std::hypot(center.x - nearest_x, center.y - nearest_y) <=
      radius_m + kTolerance;
}

[[nodiscard]] bool KnownForLanding(
    const shared::MapSnapshot& map, const std::size_t index,
    const MapSafetyConfig& safety) noexcept {
  return map.ByteLayer("valid_mask")[index] != 0U &&
      static_cast<double>(map.FloatLayer("observation_age_s")[index]) <=
          safety.maximum_observation_age_s + kTolerance &&
      static_cast<double>(map.FloatLayer("observation_quality")[index]) +
              kTolerance >=
          safety.minimum_observation_quality &&
      map.CountLayer("observation_count")[index] >=
          safety.minimum_observation_count &&
      static_cast<double>(map.FloatLayer("obstacle_variance")[index]) <=
          safety.maximum_obstacle_variance_m2 + kTolerance;
}

[[nodiscard]] std::optional<PlaneFit> FitPlane(
    const std::vector<Vec3>& samples,
    const double maximum_roughness_m) noexcept {
  if (samples.size() < 3U) {
    return std::nullopt;
  }
  double mean_x = 0.0;
  double mean_y = 0.0;
  double mean_z = 0.0;
  for (const Vec3 sample : samples) {
    mean_x += sample.x;
    mean_y += sample.y;
    mean_z += sample.z;
  }
  const double count = static_cast<double>(samples.size());
  mean_x /= count;
  mean_y /= count;
  mean_z /= count;

  double xx = 0.0;
  double xy = 0.0;
  double yy = 0.0;
  double xz = 0.0;
  double yz = 0.0;
  for (const Vec3 sample : samples) {
    const double x = sample.x - mean_x;
    const double y = sample.y - mean_y;
    const double z = sample.z - mean_z;
    xx += x * x;
    xy += x * y;
    yy += y * y;
    xz += x * z;
    yz += y * z;
  }
  const double determinant = xx * yy - xy * xy;
  const double scale = std::max({1.0, std::abs(xx * yy), std::abs(xy * xy)});
  if (!std::isfinite(determinant) ||
      std::abs(determinant) <= std::numeric_limits<double>::epsilon() * scale) {
    return std::nullopt;
  }
  const double a = (xz * yy - yz * xy) / determinant;
  const double b = (yz * xx - xz * xy) / determinant;
  const double c = mean_z - a * mean_x - b * mean_y;
  double residual = 0.0;
  for (const Vec3 sample : samples) {
    residual = std::max(
        residual, std::abs(sample.z - (a * sample.x + b * sample.y + c)));
  }
  const double slope = std::atan(std::hypot(a, b));
  if (!std::isfinite(a) || !std::isfinite(b) || !std::isfinite(c) ||
      !std::isfinite(residual) || !std::isfinite(slope)) {
    return std::nullopt;
  }
  return PlaneFit{
      .a = a,
      .b = b,
      .c = c,
      .slope_rad = slope,
      .maximum_residual_m = residual,
      .maximum_roughness_m = maximum_roughness_m,
  };
}

[[nodiscard]] DiskCheck CheckDisk(
    const shared::MapSnapshot& map, const Vec2 center,
    const double radius_m, const HopperCapability& capability,
    const MapSafetyConfig& safety, const std::stop_token stop_token) {
  const double minimum_x = center.x - radius_m;
  const double maximum_x = center.x + radius_m;
  const double minimum_y = center.y - radius_m;
  const double maximum_y = center.y + radius_m;
  const auto to_index = [&](const double coordinate, const double origin) {
    return static_cast<long long>(
        std::floor((coordinate - origin) / map.resolution_m()));
  };
  const long long x0 = to_index(minimum_x, map.origin_m().x);
  const long long x1 = to_index(maximum_x, map.origin_m().x);
  const long long y0 = to_index(minimum_y, map.origin_m().y);
  const long long y1 = to_index(maximum_y, map.origin_m().y);
  if (x0 < 0 || y0 < 0 || x1 >= static_cast<long long>(map.width()) ||
      y1 >= static_cast<long long>(map.height())) {
    return {.reason_code = "LANDING_EVIDENCE_INSUFFICIENT"};
  }

  std::vector<Vec3> samples;
  std::size_t inspected = 0U;
  double maximum_roughness = 0.0;
  for (long long y = y0; y <= y1; ++y) {
    for (long long x = x0; x <= x1; ++x) {
      const shared::GridCell cell{
          .x = static_cast<std::int32_t>(x),
          .y = static_cast<std::int32_t>(y),
      };
      if (!CellIntersectsDisk(map, cell, center, radius_m)) {
        continue;
      }
      if (stop_token.stop_requested()) {
        return {
            .inspected_cells = inspected,
            .reason_code = "REQUEST_CANCELED",
        };
      }
      ++inspected;
      const std::size_t index = map.Index(cell);
      if (!KnownForLanding(map, index, safety)) {
        return {
            .inspected_cells = inspected,
            .reason_code = "LANDING_EVIDENCE_INSUFFICIENT",
        };
      }
      if (map.ByteLayer("obstacle")[index] != 0U ||
          map.ByteLayer("forbidden")[index] != 0U) {
        return {
            .inspected_cells = inspected,
            .reason_code = "HOPPER_LANDING_TARGET_OCCUPIED",
        };
      }
      const Vec3 sample = map.CellCenter(cell);
      samples.push_back(sample);
      maximum_roughness = std::max(
          maximum_roughness,
          std::sqrt(std::max(
              0.0, static_cast<double>(
                       map.FloatLayer("elevation_variance")[index]))));
    }
  }
  const auto plane = FitPlane(samples, maximum_roughness);
  if (!plane.has_value()) {
    return {
        .inspected_cells = inspected,
        .reason_code = "HOPPER_LANDING_REGION_NUMERICAL_INDETERMINATE",
    };
  }
  const double maximum_slope = std::min(
      capability.maximum_landing_slope_rad,
      safety.project_maximum_slope_rad);
  if (plane->slope_rad > maximum_slope + kTolerance) {
    return {
        .inspected_cells = inspected,
        .reason_code = "HOPPER_LANDING_SLOPE_EXCEEDED",
    };
  }
  if (plane->maximum_residual_m >
      capability.maximum_landing_plane_residual_m + kTolerance) {
    return {
        .inspected_cells = inspected,
        .reason_code = "HOPPER_LANDING_PLANE_RESIDUAL_EXCEEDED",
    };
  }
  return {
      .plane = plane,
      .inspected_cells = inspected,
  };
}

[[nodiscard]] LandingRegionResult Failure(
    const LandingRegionStatus status, std::string reason_code,
    const std::size_t inspected) {
  return {
      .status = status,
      .inspected_cells = inspected,
      .reason_code = std::move(reason_code),
  };
}

}  // namespace

LandingRegionResult CertifyExactLandingRegion(
    const shared::MapSnapshot& map, const GoalRegion& goal,
    const HopperCapability& capability, const MapSafetyConfig& map_safety,
    const std::stop_token stop_token) {
  if (stop_token.stop_requested()) {
    return Failure(LandingRegionStatus::kCanceled, "REQUEST_CANCELED", 0U);
  }
  const auto* point = std::get_if<PointGoal>(&goal.target);
  if (point == nullptr || point->tolerance_m != 0.0 ||
      goal.yaw_rad.has_value() || goal.yaw_tolerance_rad != 0.0 ||
      !std::isfinite(point->position_m.x) ||
      !std::isfinite(point->position_m.y) ||
      !std::isfinite(point->position_m.z)) {
    return Failure(
        LandingRegionStatus::kInvalidRequest,
        "HOPPER_EXACT_POINT_REQUIRED", 0U);
  }
  if (!FinitePositive(capability.landing_support_radius_m) ||
      !std::isfinite(capability.landing_lateral_margin_m) ||
      capability.landing_lateral_margin_m < 0.0 ||
      !FinitePositive(capability.maximum_landing_slope_rad) ||
      !std::isfinite(capability.maximum_landing_plane_residual_m) ||
      capability.maximum_landing_plane_residual_m < 0.0) {
    return Failure(
        LandingRegionStatus::kInvalidRequest,
        "HOPPER_CAPABILITY_INVALID", 0U);
  }

  const Vec2 center{point->position_m.x, point->position_m.y};
  const double support_radius = capability.landing_support_radius_m +
      capability.landing_lateral_margin_m;
  DiskCheck exact = CheckDisk(
      map, center, support_radius, capability, map_safety, stop_token);
  if (!exact.ok()) {
    const LandingRegionStatus status =
        exact.reason_code == "REQUEST_CANCELED"
            ? LandingRegionStatus::kCanceled
            : (exact.reason_code.find("NUMERICAL") != std::string::npos
                   ? LandingRegionStatus::kInvalidRequest
                   : LandingRegionStatus::kInfeasible);
    return Failure(status, std::move(exact.reason_code), exact.inspected_cells);
  }
  const auto elevation = map.SampleElevationBilinear(center);
  const auto containing = map.PositionToCell(center);
  if (!elevation.has_value() || !containing.has_value() ||
      !std::isfinite(*elevation)) {
    return Failure(
        LandingRegionStatus::kInfeasible,
        "LANDING_EVIDENCE_INSUFFICIENT", exact.inspected_cells);
  }

  double half_extent = 0.125 * map.resolution_m();
  DiskCheck expanded;
  std::size_t inspected = exact.inspected_cells;
  const double minimum_extent =
      map.resolution_m() * std::sqrt(std::numeric_limits<double>::epsilon());
  while (half_extent > minimum_extent) {
    const double expanded_radius =
        support_radius + std::numbers::sqrt2 * half_extent;
    expanded = CheckDisk(
        map, center, expanded_radius, capability, map_safety, stop_token);
    inspected += expanded.inspected_cells;
    if (expanded.ok()) {
      break;
    }
    if (expanded.reason_code == "REQUEST_CANCELED") {
      return Failure(
          LandingRegionStatus::kCanceled, "REQUEST_CANCELED", inspected);
    }
    half_extent *= 0.5;
  }
  if (!expanded.ok()) {
    return Failure(
        LandingRegionStatus::kInfeasible,
        "LANDING_REGION_NOT_CERTIFIABLE", inspected);
  }

  const PlaneFit& plane = *expanded.plane;
  const auto height = [&](const double x, const double y) {
    return plane.a * x + plane.b * y + plane.c;
  };
  const double x0 = center.x - half_extent;
  const double x1 = center.x + half_extent;
  const double y0 = center.y - half_extent;
  const double y1 = center.y + half_extent;
  return {
      .status = LandingRegionStatus::kCertified,
      .region = CertifiedLandingRegion{
          .seed_cell = *containing,
          .aim_position_on_surface_m = {center.x, center.y, *elevation},
          .plane_normal = {-plane.a, -plane.b, 1.0},
          .boundary_m = {
              Vec3{x0, y0, height(x0, y0)},
              Vec3{x1, y0, height(x1, y0)},
              Vec3{x1, y1, height(x1, y1)},
              Vec3{x0, y1, height(x0, y1)},
          },
          .area_m2 = 4.0 * half_extent * half_extent,
          .maximum_slope_rad = plane.slope_rad,
          .maximum_roughness_m = plane.maximum_roughness_m,
          .maximum_plane_residual_m = plane.maximum_residual_m,
          .minimum_clearance_m = support_radius,
      },
      .inspected_cells = inspected,
  };
}

LandingRegionResult CertifyLandingRegion(
    const shared::SafeProjection& projection, const GoalRegion& goal,
    const HopperCapability& capability, const MapSafetyConfig& map_safety,
    const std::stop_token stop_token) {
  if (projection.source_map() == nullptr) {
    return Failure(
        LandingRegionStatus::kInvalidRequest,
        "LANDING_EVIDENCE_INSUFFICIENT", 0U);
  }
  return CertifyExactLandingRegion(
      *projection.source_map(), goal, capability, map_safety, stop_token);
}

LandingRegionResult CertifyHoldingRegion(
    const shared::SafeProjection& projection, const Vec3 holding_position_m,
    const HopperCapability& capability, const MapSafetyConfig& map_safety,
    const std::stop_token stop_token) {
  GoalRegion goal{
      .goal_id = "hopper-holding-region",
      .target = PointGoal{
          .position_m = holding_position_m,
          .tolerance_m = 0.0,
      },
      .yaw_rad = std::nullopt,
      .yaw_tolerance_rad = 0.0,
  };
  return CertifyLandingRegion(
      projection, goal, capability, map_safety, stop_token);
}

}  // namespace lunar::pure_planning::hopper

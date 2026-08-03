#include "hopper/landing_region.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <ranges>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "shared/terrain_checks.hpp"

namespace lunar::planning::hopper {
namespace {

constexpr double kTolerance = 1.0e-9;

struct PreferredPoint final {
  Vec3 point_m;
  bool valid{};
};

struct CellCertification final {
  bool safe{};
  double slope_rad{};
  double roughness_m{};
  double plane_residual_m{};
  double clearance_m{};
};

struct CellRectangle final {
  std::int32_t minimum_x{};
  std::int32_t maximum_x{};
  std::int32_t minimum_y{};
  std::int32_t maximum_y{};
};

[[nodiscard]] bool IsFinite(const Vec3 value) noexcept {
  return std::isfinite(value.x) && std::isfinite(value.y) &&
      std::isfinite(value.z);
}

[[nodiscard]] double Norm(const Vec3 value) noexcept {
  return std::hypot(std::hypot(value.x, value.y), value.z);
}

[[nodiscard]] Vec3 Normalize(const Vec3 value) noexcept {
  const double length = Norm(value);
  if (!std::isfinite(length) || length <= kTolerance) {
    return Vec3{};
  }
  return {value.x / length, value.y / length, value.z / length};
}

[[nodiscard]] bool PointInPolygon(
    const Vec2 point, const std::vector<Vec3>& boundary) noexcept {
  if (boundary.size() < 3U || !std::isfinite(point.x) ||
      !std::isfinite(point.y)) {
    return false;
  }
  bool inside = false;
  for (std::size_t current = 0U, previous = boundary.size() - 1U;
       current < boundary.size(); previous = current++) {
    const Vec3 a = boundary[current];
    const Vec3 b = boundary[previous];
    if (!IsFinite(a) || !IsFinite(b)) {
      return false;
    }
    const bool crosses = (a.y > point.y) != (b.y > point.y);
    if (crosses) {
      const double x_intersection =
          (b.x - a.x) * (point.y - a.y) / (b.y - a.y) + a.x;
      if (point.x < x_intersection) {
        inside = !inside;
      }
    }
  }
  return inside;
}

[[nodiscard]] PreferredPoint GoalPreferredPoint(const GoalRegion& goal) {
  return std::visit(
      [](const auto& target) -> PreferredPoint {
        using Target = std::decay_t<decltype(target)>;
        if constexpr (std::is_same_v<Target, PointGoal>) {
          return PreferredPoint{
              .point_m = target.position_m,
              .valid = IsFinite(target.position_m) &&
                  std::isfinite(target.tolerance_m) &&
                  target.tolerance_m >= 0.0,
          };
        } else {
          if (target.boundary_m.size() < 3U ||
              !std::isfinite(target.normal_tolerance_m) ||
              target.normal_tolerance_m < 0.0) {
            return {};
          }
          Vec3 center{};
          for (const Vec3 point : target.boundary_m) {
            if (!IsFinite(point)) {
              return {};
            }
            center.x += point.x;
            center.y += point.y;
            center.z += point.z;
          }
          const double count = static_cast<double>(target.boundary_m.size());
          center.x /= count;
          center.y /= count;
          center.z /= count;
          return PreferredPoint{.point_m = center, .valid = true};
        }
      },
      goal.target);
}

[[nodiscard]] bool GoalContainsCellCenter(
    const GoalRegion& goal, const Vec3 center) noexcept {
  return std::visit(
      [&](const auto& target) {
        using Target = std::decay_t<decltype(target)>;
        if constexpr (std::is_same_v<Target, PointGoal>) {
          return std::hypot(
                     center.x - target.position_m.x,
                     center.y - target.position_m.y) <=
              target.tolerance_m + kTolerance;
        } else {
          return PointInPolygon(Vec2{center.x, center.y}, target.boundary_m);
        }
      },
      goal.target);
}

[[nodiscard]] double Elevation(
    const shared::MapSnapshot& map,
    const shared::GridCell cell) noexcept {
  if (!map.InBounds(cell)) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return static_cast<double>(map.FloatLayer("elevation")[map.Index(cell)]);
}

[[nodiscard]] double AxisGradient(
    const shared::MapSnapshot& map, const shared::GridCell cell,
    const std::int32_t dx, const std::int32_t dy) noexcept {
  const shared::GridCell negative{.x = cell.x - dx, .y = cell.y - dy};
  const shared::GridCell positive{.x = cell.x + dx, .y = cell.y + dy};
  const double center = Elevation(map, cell);
  if (map.InBounds(negative) && map.InBounds(positive)) {
    return (Elevation(map, positive) - Elevation(map, negative)) /
        (2.0 * map.resolution_m());
  }
  if (map.InBounds(positive)) {
    return (Elevation(map, positive) - center) / map.resolution_m();
  }
  if (map.InBounds(negative)) {
    return (center - Elevation(map, negative)) / map.resolution_m();
  }
  return 0.0;
}

[[nodiscard]] double PlaneResidual(
    const shared::SafeProjection& projection,
    const shared::GridCell cell) noexcept {
  if (projection.source_map() == nullptr) {
    return std::numeric_limits<double>::infinity();
  }
  const shared::MapSnapshot& map = *projection.source_map();
  const double center_elevation = Elevation(map, cell);
  const Vec3 center = map.CellCenter(cell);
  const double gradient_x = AxisGradient(map, cell, 1, 0);
  const double gradient_y = AxisGradient(map, cell, 0, 1);
  if (!std::isfinite(center_elevation) || !std::isfinite(gradient_x) ||
      !std::isfinite(gradient_y)) {
    return std::numeric_limits<double>::infinity();
  }
  double residual = 0.0;
  for (std::int32_t dy = -1; dy <= 1; ++dy) {
    for (std::int32_t dx = -1; dx <= 1; ++dx) {
      const shared::GridCell neighbor{.x = cell.x + dx, .y = cell.y + dy};
      if (!map.InBounds(neighbor)) {
        continue;
      }
      if (!projection.Known(neighbor) ||
          !projection.HardFeasible(neighbor)) {
        return std::numeric_limits<double>::infinity();
      }
      const Vec3 point = map.CellCenter(neighbor);
      const double predicted = center_elevation +
          gradient_x * (point.x - center.x) +
          gradient_y * (point.y - center.y);
      residual = std::max(
          residual, std::abs(Elevation(map, neighbor) - predicted));
    }
  }
  return residual;
}

[[nodiscard]] bool CellRange(
    const shared::MapSnapshot& map, const Vec3 center,
    const HopperCapability& capability,
    std::int32_t& minimum_x, std::int32_t& maximum_x,
    std::int32_t& minimum_y, std::int32_t& maximum_y) noexcept {
  const double lateral = capability.minimum_lateral_clearance_m;
  const double min_x_m = center.x - capability.body_half_extent_m.x - lateral;
  const double max_x_m = center.x + capability.body_half_extent_m.x + lateral;
  const double min_y_m = center.y - capability.body_half_extent_m.y - lateral;
  const double max_y_m = center.y + capability.body_half_extent_m.y + lateral;
  const auto index = [&](const double coordinate, const double origin) {
    return static_cast<long long>(
        std::floor((coordinate - origin) / map.resolution_m()));
  };
  const long long low_x = index(min_x_m, map.origin_m().x);
  const long long high_x = index(max_x_m, map.origin_m().x);
  const long long low_y = index(min_y_m, map.origin_m().y);
  const long long high_y = index(max_y_m, map.origin_m().y);
  if (low_x < 0 || low_y < 0 ||
      high_x >= static_cast<long long>(map.width()) ||
      high_y >= static_cast<long long>(map.height()) ||
      low_x > high_x || low_y > high_y) {
    return false;
  }
  minimum_x = static_cast<std::int32_t>(low_x);
  maximum_x = static_cast<std::int32_t>(high_x);
  minimum_y = static_cast<std::int32_t>(low_y);
  maximum_y = static_cast<std::int32_t>(high_y);
  return true;
}

[[nodiscard]] CellCertification CertifyCell(
    const shared::SafeProjection& projection,
    const shared::GridCell cell,
    const HopperCapability& capability,
    const MapSafetyConfig& map_safety) {
  CellCertification result;
  if (projection.source_map() == nullptr || !projection.InBounds(cell)) {
    return result;
  }
  const shared::MapSnapshot& map = *projection.source_map();
  const Vec3 center = map.CellCenter(cell);
  std::int32_t minimum_x = 0;
  std::int32_t maximum_x = -1;
  std::int32_t minimum_y = 0;
  std::int32_t maximum_y = -1;
  if (!CellRange(
          map, center, capability, minimum_x, maximum_x,
          minimum_y, maximum_y)) {
    return result;
  }

  result.slope_rad = 0.0;
  result.roughness_m = 0.0;
  result.plane_residual_m = 0.0;
  result.clearance_m = std::numeric_limits<double>::infinity();
  for (std::int32_t y = minimum_y; y <= maximum_y; ++y) {
    for (std::int32_t x = minimum_x; x <= maximum_x; ++x) {
      const shared::GridCell footprint_cell{.x = x, .y = y};
      if (!projection.Known(footprint_cell) ||
          !projection.HardFeasible(footprint_cell)) {
        return result;
      }
      result.slope_rad = std::max(
          result.slope_rad,
          static_cast<double>(projection.SlopeRadians(footprint_cell)));
      result.roughness_m = std::max(
          result.roughness_m,
          static_cast<double>(projection.RoughnessMeters(footprint_cell)));
      result.plane_residual_m = std::max(
          result.plane_residual_m,
          PlaneResidual(projection, footprint_cell));
      result.clearance_m = std::min(
          result.clearance_m,
          static_cast<double>(projection.ClearanceMeters(footprint_cell)));
    }
  }

  const double required_center_clearance = std::max(
      capability.minimum_landing_clearance_m,
      std::hypot(
          capability.body_half_extent_m.x,
          capability.body_half_extent_m.y) +
          capability.minimum_lateral_clearance_m);
  const double center_clearance =
      static_cast<double>(projection.ClearanceMeters(cell));
  const double effective_slope = std::min(
      capability.maximum_landing_slope_rad,
      std::min(map_safety.project_maximum_slope_rad, projection.maximum_slope_rad()));
  result.safe = std::isfinite(result.slope_rad) &&
      std::isfinite(result.roughness_m) &&
      std::isfinite(result.plane_residual_m) &&
      std::isfinite(center_clearance) &&
      result.slope_rad <= effective_slope + kTolerance &&
      result.roughness_m <=
          capability.maximum_landing_roughness_m + kTolerance &&
      result.plane_residual_m <=
          capability.maximum_plane_residual_m + kTolerance &&
      center_clearance + kTolerance >= required_center_clearance;
  result.clearance_m = std::min(result.clearance_m, center_clearance);
  return result;
}

[[nodiscard]] bool RectangleSafe(
    const shared::MapSnapshot& map,
    const std::vector<std::uint8_t>& safe_mask,
    const CellRectangle rectangle) noexcept {
  for (std::int32_t y = rectangle.minimum_y; y <= rectangle.maximum_y; ++y) {
    for (std::int32_t x = rectangle.minimum_x; x <= rectangle.maximum_x; ++x) {
      const shared::GridCell cell{.x = x, .y = y};
      if (!map.InBounds(cell) || safe_mask[map.Index(cell)] == 0U) {
        return false;
      }
    }
  }
  return true;
}

[[nodiscard]] LandingRegionResult Certify(
    const shared::SafeProjection& projection,
    const GoalRegion& goal,
    const HopperCapability& capability,
    const MapSafetyConfig& map_safety,
    const std::stop_token stop_token,
    const bool holding_region) {
  if (stop_token.stop_requested()) {
    return LandingRegionResult{
        .status = LandingRegionStatus::kCanceled,
        .region = std::nullopt,
        .inspected_cells = 0U,
        .reason_code = "REQUEST_CANCELED",
    };
  }
  const PreferredPoint preferred = GoalPreferredPoint(goal);
  if (!preferred.valid || projection.source_map() == nullptr) {
    return LandingRegionResult{
        .status = LandingRegionStatus::kInvalidRequest,
        .region = std::nullopt,
        .inspected_cells = 0U,
        .reason_code = "HOPPER_LANDING_GOAL_INVALID",
    };
  }
  const shared::MapSnapshot& map = *projection.source_map();
  const auto containing = map.PositionToCell(
      Vec2{preferred.point_m.x, preferred.point_m.y});
  if (!containing.has_value()) {
    return LandingRegionResult{
        .status = LandingRegionStatus::kInfeasible,
        .region = std::nullopt,
        .inspected_cells = 0U,
        .reason_code = holding_region
            ? "HOPPER_START_NOT_SAFE"
            : "HOPPER_GOAL_INFEASIBLE",
    };
  }

  std::vector<std::uint8_t> safe_mask(map.cell_count(), 0U);
  std::vector<CellCertification> certifications(map.cell_count());
  std::vector<shared::GridCell> safe_cells;
  std::size_t inspected = 0U;
  for (std::size_t y = 0U; y < map.height(); ++y) {
    for (std::size_t x = 0U; x < map.width(); ++x) {
      if (stop_token.stop_requested()) {
        return LandingRegionResult{
            .status = LandingRegionStatus::kCanceled,
            .region = std::nullopt,
            .inspected_cells = inspected,
            .reason_code = "REQUEST_CANCELED",
        };
      }
      const shared::GridCell cell{
          .x = static_cast<std::int32_t>(x),
          .y = static_cast<std::int32_t>(y),
      };
      const bool candidate = holding_region
          ? true
          : (GoalContainsCellCenter(goal, map.CellCenter(cell)) ||
             cell == *containing);
      if (!candidate) {
        continue;
      }
      ++inspected;
      const std::size_t index = map.Index(cell);
      certifications[index] =
          CertifyCell(projection, cell, capability, map_safety);
      if (certifications[index].safe) {
        safe_mask[index] = 1U;
        safe_cells.push_back(cell);
      }
    }
  }
  if (safe_cells.empty()) {
    return LandingRegionResult{
        .status = LandingRegionStatus::kInfeasible,
        .region = std::nullopt,
        .inspected_cells = inspected,
        .reason_code = holding_region
            ? "HOPPER_START_NOT_SAFE"
            : "HOPPER_GOAL_INFEASIBLE",
    };
  }
  std::ranges::sort(
      safe_cells,
      [&](const shared::GridCell lhs, const shared::GridCell rhs) {
        const Vec3 left = map.CellCenter(lhs);
        const Vec3 right = map.CellCenter(rhs);
        const double left_distance = std::hypot(
            left.x - preferred.point_m.x,
            left.y - preferred.point_m.y);
        const double right_distance = std::hypot(
            right.x - preferred.point_m.x,
            right.y - preferred.point_m.y);
        return std::tuple{left_distance, lhs.y, lhs.x} <
            std::tuple{right_distance, rhs.y, rhs.x};
      });
  const shared::GridCell seed = safe_cells.front();
  CellRectangle rectangle{
      .minimum_x = seed.x,
      .maximum_x = seed.x,
      .minimum_y = seed.y,
      .maximum_y = seed.y,
  };
  bool expanded = true;
  while (expanded) {
    expanded = false;
    for (const std::uint8_t side : {0U, 1U, 2U, 3U}) {
      CellRectangle candidate = rectangle;
      if (side == 0U) {
        --candidate.minimum_x;
      } else if (side == 1U) {
        ++candidate.maximum_x;
      } else if (side == 2U) {
        --candidate.minimum_y;
      } else {
        ++candidate.maximum_y;
      }
      if (RectangleSafe(map, safe_mask, candidate)) {
        rectangle = candidate;
        expanded = true;
      }
    }
  }

  const double resolution = map.resolution_m();
  const double area =
      static_cast<double>(rectangle.maximum_x - rectangle.minimum_x + 1) *
      static_cast<double>(rectangle.maximum_y - rectangle.minimum_y + 1) *
      resolution * resolution;
  if (!std::isfinite(area) ||
      area + kTolerance < capability.minimum_landing_region_area_m2) {
    return LandingRegionResult{
        .status = LandingRegionStatus::kInfeasible,
        .region = std::nullopt,
        .inspected_cells = inspected,
        .reason_code = holding_region
            ? "HOPPER_START_REGION_AREA_INSUFFICIENT"
            : "HOPPER_LANDING_REGION_AREA_INSUFFICIENT",
    };
  }

  double maximum_slope = 0.0;
  double maximum_roughness = 0.0;
  double maximum_residual = 0.0;
  double minimum_clearance = std::numeric_limits<double>::infinity();
  for (std::int32_t y = rectangle.minimum_y; y <= rectangle.maximum_y; ++y) {
    for (std::int32_t x = rectangle.minimum_x; x <= rectangle.maximum_x; ++x) {
      const CellCertification& certification =
          certifications[map.Index(shared::GridCell{.x = x, .y = y})];
      maximum_slope = std::max(maximum_slope, certification.slope_rad);
      maximum_roughness =
          std::max(maximum_roughness, certification.roughness_m);
      maximum_residual =
          std::max(maximum_residual, certification.plane_residual_m);
      minimum_clearance =
          std::min(minimum_clearance, certification.clearance_m);
    }
  }

  const double gradient_x = AxisGradient(map, seed, 1, 0);
  const double gradient_y = AxisGradient(map, seed, 0, 1);
  const Vec3 normal = Normalize(Vec3{-gradient_x, -gradient_y, 1.0});
  if (!IsFinite(normal) || Norm(normal) <= kTolerance) {
    return LandingRegionResult{
        .status = LandingRegionStatus::kInvalidRequest,
        .region = std::nullopt,
        .inspected_cells = inspected,
        .reason_code = "HOPPER_LANDING_PLANE_NUMERICAL_FAILURE",
    };
  }
  const Vec3 seed_center = map.CellCenter(seed);
  const auto plane_height = [&](const double x, const double y) {
    return seed_center.z + gradient_x * (x - seed_center.x) +
        gradient_y * (y - seed_center.y);
  };
  const double x0 = map.origin_m().x +
      static_cast<double>(rectangle.minimum_x) * resolution;
  const double x1 = map.origin_m().x +
      static_cast<double>(rectangle.maximum_x + 1) * resolution;
  const double y0 = map.origin_m().y +
      static_cast<double>(rectangle.minimum_y) * resolution;
  const double y1 = map.origin_m().y +
      static_cast<double>(rectangle.maximum_y + 1) * resolution;
  const double aim_x = std::clamp(preferred.point_m.x, x0, x1);
  const double aim_y = std::clamp(preferred.point_m.y, y0, y1);

  return LandingRegionResult{
      .status = LandingRegionStatus::kCertified,
      .region = CertifiedLandingRegion{
          .seed_cell = seed,
          .aim_position_on_surface_m =
              Vec3{aim_x, aim_y, plane_height(aim_x, aim_y)},
          .plane_normal = normal,
          .boundary_m =
              {
                  Vec3{x0, y0, plane_height(x0, y0)},
                  Vec3{x1, y0, plane_height(x1, y0)},
                  Vec3{x1, y1, plane_height(x1, y1)},
                  Vec3{x0, y1, plane_height(x0, y1)},
              },
          .area_m2 = area,
          .maximum_slope_rad = maximum_slope,
          .maximum_roughness_m = maximum_roughness,
          .maximum_plane_residual_m = maximum_residual,
          .minimum_clearance_m = minimum_clearance,
      },
      .inspected_cells = inspected,
      .reason_code = {},
  };
}

}  // namespace

LandingRegionResult CertifyLandingRegion(
    const shared::SafeProjection& projection,
    const GoalRegion& goal,
    const HopperCapability& capability,
    const MapSafetyConfig& map_safety,
    const std::stop_token stop_token) {
  return Certify(
      projection, goal, capability, map_safety, stop_token, false);
}

LandingRegionResult CertifyHoldingRegion(
    const shared::SafeProjection& projection,
    const Vec3 holding_position_m,
    const HopperCapability& capability,
    const MapSafetyConfig& map_safety,
    const std::stop_token stop_token) {
  return Certify(
      projection,
      GoalRegion{
          .goal_id = "hopper-holding-region",
          .target = PointGoal{
              .position_m = holding_position_m,
              .tolerance_m = 0.0,
          },
          .yaw_rad = std::nullopt,
          .yaw_tolerance_rad = 0.0,
      },
      capability, map_safety, stop_token, true);
}

}  // namespace lunar::planning::hopper

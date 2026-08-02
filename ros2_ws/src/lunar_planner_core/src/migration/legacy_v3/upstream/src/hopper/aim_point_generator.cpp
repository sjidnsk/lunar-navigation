#include "lunar_path_planner/v3/hopper/aim_point_generator.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <limits>
#include <numbers>
#include <ranges>
#include <tuple>
#include <type_traits>

namespace lunar::planning::v3 {
namespace {

[[nodiscard]] double BoundaryDistance(
    const Eigen::Vector2d& point,
    const ConvexPolygon2d& polygon) {
  double minimum = std::numeric_limits<double>::infinity();
  for (std::size_t index = 0U;
       index < polygon.vertices_ccw.size(); ++index) {
    const Eigen::Vector2d& start = polygon.vertices_ccw[index];
    const Eigen::Vector2d& end =
        polygon.vertices_ccw[
            (index + 1U) % polygon.vertices_ccw.size()];
    const Eigen::Vector2d edge = end - start;
    const double distance =
        (edge.x() * (point.y() - start.y()) -
         edge.y() * (point.x() - start.x())) /
        edge.norm();
    minimum = std::min(minimum, distance);
  }
  return minimum;
}

[[nodiscard]] Eigen::Vector2d PolygonCenter(
    const ConvexPolygon2d& polygon) {
  Eigen::Vector2d center = Eigen::Vector2d::Zero();
  for (const auto& vertex : polygon.vertices_ccw) {
    center += vertex;
  }
  return center / static_cast<double>(polygon.vertices_ccw.size());
}

[[nodiscard]] Eigen::Vector2d GoalPointUv(
    const GoalRegion& goal, const LandingPlane& plane) {
  return std::visit(
      [&plane](const auto& target) -> Eigen::Vector2d {
        using Target = std::decay_t<decltype(target)>;
        if constexpr (std::is_same_v<Target, PointGoal>) {
          return world_to_plane_uv(
              plane,
              Eigen::Vector3d{
                  target.position_m.x, target.position_m.y,
                  target.position_m.z});
        } else {
          Eigen::Vector2d center = Eigen::Vector2d::Zero();
          for (const Vec2& vertex : target.polygon.vertices_uv) {
            center += Eigen::Vector2d{vertex.x, vertex.y};
          }
          return center /
                 static_cast<double>(
                     target.polygon.vertices_uv.size());
        }
      },
      goal.target);
}

[[nodiscard]] Eigen::Vector2d InsetTowardCenter(
    const Eigen::Vector2d& candidate,
    const Eigen::Vector2d& center,
    const TerrainCertifiedLandingRegion& region) {
  Eigen::Vector2d result = candidate;
  for (int iteration = 0; iteration < 64; ++iteration) {
    if (point_in_convex_polygon(
            result, region.vertices_uv_ccw,
            region.inward_safety_margin_m + 1.0e-9)) {
      return result;
    }
    result = 0.5 * (result + center);
  }
  return center;
}

[[nodiscard]] std::int64_t Quantize(const double value) {
  return static_cast<std::int64_t>(
      std::llround(value * 1.0e9));
}

}  // namespace

AimPointGenerator::AimPointGenerator(
    HopperCapabilityView capability,
    HopperPlannerLimits limits)
    : capability_(std::move(capability)),
      limits_(std::move(limits)) {}

std::vector<AimPointCandidate> AimPointGenerator::generate(
    const TerrainCertifiedLandingRegion& region,
    const GoalRegion& goal,
    const std::optional<Eigen::Vector3d>&
        mission_direction_frame) const {
  std::vector<std::pair<AimPointSource, Eigen::Vector2d>> raw;
  if (!validate_convex_polygon(region.vertices_uv_ccw).ok()) {
    return {};
  }
  const Eigen::Vector2d center =
      PolygonCenter(region.vertices_uv_ccw);
  raw.emplace_back(AimPointSource::kChebyshevCenter, center);
  raw.emplace_back(
      AimPointSource::kGoalProjection,
      InsetTowardCenter(
          GoalPointUv(goal, region.landing_plane), center, region));

  if (mission_direction_frame.has_value() &&
      mission_direction_frame->allFinite()) {
    const Eigen::Vector3d basis_u{
        region.landing_plane.basis_u.x,
        region.landing_plane.basis_u.y,
        region.landing_plane.basis_u.z};
    const Eigen::Vector3d basis_v{
        region.landing_plane.basis_v.x,
        region.landing_plane.basis_v.y,
        region.landing_plane.basis_v.z};
    Eigen::Vector2d direction{
        mission_direction_frame->dot(basis_u),
        mission_direction_frame->dot(basis_v)};
    if (direction.norm() > 1.0e-12) {
      direction.normalize();
      raw.emplace_back(
          AimPointSource::kMissionDirectionInset,
          InsetTowardCenter(
              center + direction, center, region));
    }
  }

  for (std::size_t index = 0U;
       index < limits_.support_direction_count; ++index) {
    const double angle =
        2.0 * std::numbers::pi * static_cast<double>(index) /
        static_cast<double>(limits_.support_direction_count);
    const Eigen::Vector2d direction{
        std::cos(angle), std::sin(angle)};
    raw.emplace_back(
        AimPointSource::kSupportDirectionInset,
        InsetTowardCenter(center + direction, center, region));
  }

  std::vector<AimPointCandidate> candidates;
  const Eigen::Vector3d normal{
      region.landing_plane.normal.x,
      region.landing_plane.normal.y,
      region.landing_plane.normal.z};
  for (const auto& [source, uv] : raw) {
    const double distance =
        BoundaryDistance(uv, region.vertices_uv_ccw);
    if (!std::isfinite(distance) ||
        distance + 1.0e-9 <
            region.inward_safety_margin_m) {
      continue;
    }
    const auto quantized =
        std::tuple{Quantize(uv.x()), Quantize(uv.y())};
    if (std::ranges::any_of(
            candidates, [&quantized](const auto& existing) {
              return std::tuple{
                         Quantize(existing.position_uv.x()),
                         Quantize(existing.position_uv.y())} ==
                     quantized;
            })) {
      continue;
    }
    candidates.push_back(
        {
            .aim_point_id = std::format(
                "{}-aim-{}-{}", region.region_id,
                std::get<0>(quantized), std::get<1>(quantized)),
            .region_id = region.region_id,
            .frame_id = region.frame_id,
            .position_uv = uv,
            .position_m =
                plane_uv_to_world(region.landing_plane, uv) +
                normal *
                    capability_.actuator_or_impulse_profile
                        .nominal_landing_center_normal_offset_m,
            .source = source,
            .minimum_boundary_distance_m = distance,
        });
  }
  std::sort(
      candidates.begin(), candidates.end(),
      [](const auto& lhs, const auto& rhs) {
        return std::tuple{
                   static_cast<std::uint8_t>(lhs.source),
                   -lhs.minimum_boundary_distance_m,
                   Quantize(lhs.position_uv.x()),
                   Quantize(lhs.position_uv.y()),
                   lhs.aim_point_id} <
               std::tuple{
                   static_cast<std::uint8_t>(rhs.source),
                   -rhs.minimum_boundary_distance_m,
                   Quantize(rhs.position_uv.x()),
                   Quantize(rhs.position_uv.y()),
                   rhs.aim_point_id};
      });
  if (candidates.size() >
      limits_.maximum_nominal_aim_points_per_region) {
    candidates.resize(
        limits_.maximum_nominal_aim_points_per_region);
  }
  return candidates;
}

bool point_strictly_inside_region(
    const AimPointCandidate& point,
    const TerrainCertifiedLandingRegion& region) {
  return point.region_id == region.region_id &&
         point.frame_id == region.frame_id &&
         point.position_uv.allFinite() &&
         point.position_m.allFinite() &&
         point_in_convex_polygon(
             point.position_uv, region.vertices_uv_ccw,
             region.inward_safety_margin_m);
}

}  // namespace lunar::planning::v3

#include "lunar_path_planner/v3/hopper/landing_region.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <limits>
#include <numeric>
#include <ranges>
#include <set>
#include <string>

#include <Eigen/Eigenvalues>

namespace lunar::planning::v3 {
namespace {

struct CellRectangle final {
  int minimum_x{};
  int maximum_x{};
  int minimum_y{};
  int maximum_y{};
};

[[nodiscard]] bool RectangleSafe(
    const SafePoseMask& mask, const CellRectangle& rectangle) {
  for (int y = rectangle.minimum_y; y <= rectangle.maximum_y; ++y) {
    for (int x = rectangle.minimum_x; x <= rectangle.maximum_x; ++x) {
      if (!is_landing_cell_safe(mask, Cell{x, y})) {
        return false;
      }
    }
  }
  return true;
}

[[nodiscard]] Eigen::Vector3d CellPoint(
    const ImmutableMapSnapshot& map, const Cell cell) {
  const auto& geometry = map.geometry();
  const std::size_t index =
      landing_cell_index(geometry, cell);
  return {
      geometry.origin_m.x +
          (static_cast<double>(cell.x) + 0.5) *
              geometry.resolution_m,
      geometry.origin_m.y +
          (static_cast<double>(cell.y) + 0.5) *
              geometry.resolution_m,
      map.ElevationMeters()[index],
  };
}

[[nodiscard]] LandingPlane FitPlane(
    const ImmutableMapSnapshot& map,
    const CellRectangle& rectangle) {
  std::vector<Eigen::Vector3d> points;
  for (int y = rectangle.minimum_y; y <= rectangle.maximum_y; ++y) {
    for (int x = rectangle.minimum_x; x <= rectangle.maximum_x; ++x) {
      points.push_back(CellPoint(map, Cell{x, y}));
    }
  }
  Eigen::Vector3d center = Eigen::Vector3d::Zero();
  for (const auto& point : points) {
    center += point;
  }
  center /= static_cast<double>(points.size());
  Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
  for (const auto& point : points) {
    const Eigen::Vector3d delta = point - center;
    covariance += delta * delta.transpose();
  }
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(covariance);
  Eigen::Vector3d normal =
      solver.eigenvectors().col(0).normalized();
  if (normal.z() < 0.0) {
    normal = -normal;
  }
  const Eigen::Vector3d reference =
      std::abs(normal.z()) < 0.9
          ? Eigen::Vector3d::UnitZ()
          : Eigen::Vector3d::UnitX();
  const Eigen::Vector3d basis_u =
      reference.cross(normal).normalized();
  const Eigen::Vector3d basis_v =
      normal.cross(basis_u).normalized();
  double residual = 0.0;
  for (const auto& point : points) {
    residual =
        std::max(residual, std::abs(normal.dot(point - center)));
  }
  return {
      .origin_m = {center.x(), center.y(), center.z()},
      .normal = {normal.x(), normal.y(), normal.z()},
      .basis_u = {basis_u.x(), basis_u.y(), basis_u.z()},
      .basis_v = {basis_v.x(), basis_v.y(), basis_v.z()},
      .residual_bound_m = residual,
  };
}

[[nodiscard]] ConvexPolygon2d RectanglePolygon(
    const ImmutableMapSnapshot& map,
    const LandingPlane& plane,
    const CellRectangle& rectangle) {
  const auto& geometry = map.geometry();
  const double x0 = geometry.origin_m.x +
      static_cast<double>(rectangle.minimum_x) *
          geometry.resolution_m;
  const double x1 = geometry.origin_m.x +
      static_cast<double>(rectangle.maximum_x + 1) *
          geometry.resolution_m;
  const double y0 = geometry.origin_m.y +
      static_cast<double>(rectangle.minimum_y) *
          geometry.resolution_m;
  const double y1 = geometry.origin_m.y +
      static_cast<double>(rectangle.maximum_y + 1) *
          geometry.resolution_m;
  const Eigen::Vector3d normal{
      plane.normal.x, plane.normal.y, plane.normal.z};
  const Eigen::Vector3d origin{
      plane.origin_m.x, plane.origin_m.y, plane.origin_m.z};
  const auto on_plane = [&normal, &origin](
                            const double x, const double y) {
    double z = origin.z();
    if (std::abs(normal.z()) > 1.0e-12) {
      z = origin.z() -
          (normal.x() * (x - origin.x()) +
           normal.y() * (y - origin.y())) /
              normal.z();
    }
    return Eigen::Vector3d{x, y, z};
  };
  ConvexPolygon2d polygon;
  for (const auto& point :
       {on_plane(x0, y0), on_plane(x1, y0),
        on_plane(x1, y1), on_plane(x0, y1)}) {
    polygon.vertices_ccw.push_back(
        world_to_plane_uv(plane, point));
  }
  if (polygon_signed_area(polygon) < 0.0) {
    std::reverse(
        polygon.vertices_ccw.begin(),
        polygon.vertices_ccw.end());
  }
  return polygon;
}

[[nodiscard]] ValidationIssue Issue(
    std::string path, std::string code, std::string message) {
  return {std::move(path), std::move(code), std::move(message)};
}

}  // namespace

LandingRegionGenerator::LandingRegionGenerator(
    HopperCapabilityView capability,
    HopperPlannerLimits limits)
    : capability_(std::move(capability)),
      limits_(std::move(limits)) {}

std::vector<TerrainCertifiedLandingRegion>
LandingRegionGenerator::generate(
    const ImmutableMapSnapshot& map,
    const SafePoseMask& mask,
    const std::span<const LandingSeed> seeds) const {
  std::vector<TerrainCertifiedLandingRegion> regions;
  std::set<std::string> identities;
  for (const LandingSeed& seed : seeds) {
    if (regions.size() >= limits_.maximum_landing_regions ||
        !is_landing_cell_safe(mask, seed.cell)) {
      break;
    }
    CellRectangle rectangle{
        seed.cell.x, seed.cell.x, seed.cell.y, seed.cell.y};
    for (std::size_t iteration = 0U;
         iteration < limits_.landing_region_inflation_iterations;
         ++iteration) {
      bool expanded = false;
      for (const int side : {0, 1, 2, 3}) {
        CellRectangle candidate = rectangle;
        if (side == 0) {
          --candidate.minimum_x;
        } else if (side == 1) {
          ++candidate.maximum_x;
        } else if (side == 2) {
          --candidate.minimum_y;
        } else {
          ++candidate.maximum_y;
        }
        if (RectangleSafe(mask, candidate)) {
          rectangle = candidate;
          expanded = true;
        }
      }
      if (!expanded) {
        break;
      }
    }
    const LandingPlane plane = FitPlane(map, rectangle);
    if (!validate_landing_plane(plane).ok() ||
        plane.residual_bound_m >
            capability_.landing_terrain_thresholds
                .maximum_plane_residual_m) {
      continue;
    }
    const ConvexPolygon2d polygon =
        RectanglePolygon(map, plane, rectangle);
    if (!validate_convex_polygon(polygon).ok() ||
        std::abs(polygon_signed_area(polygon)) <
            capability_.landing_terrain_thresholds
                .minimum_landing_region_area_m2) {
      continue;
    }
    const std::string region_id = std::format(
        "hopper-region-{}-{}-{}-{}-{}",
        rectangle.minimum_x, rectangle.maximum_x,
        rectangle.minimum_y, rectangle.maximum_y,
        seed.stable_id);
    if (!identities.insert(region_id).second) {
      continue;
    }
    TerrainCertifiedLandingRegion region{
        .region_id = region_id,
        .frame_id = capability_.frame_id,
        .landing_plane = plane,
        .vertices_uv_ccw = polygon,
        .allowed_yaw_interval =
            mask.certified_yaw_interval,
        .terrain_certification =
            {
                .certification_id =
                    region_id + "-terrain-certification",
                .source_snapshot_ref = map.snapshot_ref(),
                .maximum_slope_rad =
                    capability_.landing_terrain_thresholds
                        .maximum_slope_rad,
                .maximum_roughness_m =
                    capability_.landing_terrain_thresholds
                        .maximum_roughness_m,
                .maximum_plane_residual_m =
                    plane.residual_bound_m,
                .minimum_clearance_m = seed.clearance_m,
            },
        .inward_safety_margin_m =
            std::max(
                capability_.landing_terrain_thresholds
                    .minimum_lateral_clearance_m,
                1.0e-6),
        .source_seed = seed.cell,
    };
    if (validate_terrain_certified_region(
            region, map, mask, capability_).ok()) {
      regions.push_back(std::move(region));
    }
  }
  std::sort(
      regions.begin(), regions.end(),
      [](const auto& lhs, const auto& rhs) {
        return lhs.region_id < rhs.region_id;
      });
  return regions;
}

ValidationReport validate_terrain_certified_region(
    const TerrainCertifiedLandingRegion& region,
    const ImmutableMapSnapshot& map,
    const SafePoseMask& mask,
    const HopperCapabilityView& capability) {
  ValidationReport report;
  if (region.region_id.empty() ||
      region.frame_id != capability.frame_id ||
      region.frame_id != map.frame_id() ||
      region.terrain_certification.source_snapshot_ref !=
          map.snapshot_ref() ||
      region.terrain_certification.source_snapshot_ref !=
          mask.source_snapshot_ref) {
    report.issues.push_back(Issue(
        "", "landing_region_provenance_mismatch",
        "landing region provenance must match map and capability"));
  }
  const ValidationReport plane =
      validate_landing_plane(region.landing_plane);
  report.issues.insert(
      report.issues.end(), plane.issues.begin(), plane.issues.end());
  const ValidationReport polygon =
      validate_convex_polygon(region.vertices_uv_ccw);
  report.issues.insert(
      report.issues.end(),
      polygon.issues.begin(), polygon.issues.end());
  if (std::abs(polygon_signed_area(region.vertices_uv_ccw)) +
          1.0e-12 <
      capability.landing_terrain_thresholds
          .minimum_landing_region_area_m2) {
    report.issues.push_back(Issue(
        "vertices_uv_ccw", "landing_region_area_too_small",
        "landing region must meet the capability minimum area"));
  }
  if (region.landing_plane.residual_bound_m >
      capability.landing_terrain_thresholds
              .maximum_plane_residual_m +
          1.0e-9) {
    report.issues.push_back(Issue(
        "landing_plane", "landing_plane_residual_exceeded",
        "landing plane residual exceeds capability"));
  }
  if (!is_landing_cell_safe(mask, region.source_seed)) {
    report.issues.push_back(Issue(
        "source_seed", "unsafe_landing_seed",
        "region seed must remain safe"));
  }
  if (!validate_circular_yaw_interval(
           region.allowed_yaw_interval).ok()) {
    report.issues.push_back(Issue(
        "allowed_yaw_interval", "invalid_yaw_interval",
        "region yaw interval must be canonical"));
  }
  return report;
}

TerrainCertifiedLandingRegion simplify_region_inward(
    const TerrainCertifiedLandingRegion& region,
    const std::size_t maximum_vertex_count) {
  if (maximum_vertex_count < 3U ||
      region.vertices_uv_ccw.vertices_ccw.size() <=
          maximum_vertex_count) {
    return region;
  }
  TerrainCertifiedLandingRegion result = region;
  result.vertices_uv_ccw.vertices_ccw.clear();
  const std::size_t count =
      region.vertices_uv_ccw.vertices_ccw.size();
  for (std::size_t index = 0U;
       index < maximum_vertex_count; ++index) {
    const std::size_t source =
        index * count / maximum_vertex_count;
    result.vertices_uv_ccw.vertices_ccw.push_back(
        region.vertices_uv_ccw.vertices_ccw[source]);
  }
  return result;
}

}  // namespace lunar::planning::v3

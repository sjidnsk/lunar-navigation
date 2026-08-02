#include "lunar_path_planner/v3/hopper/landing_geometry.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <string>

namespace lunar::planning::v3 {
namespace {

constexpr double kGeometryTolerance = 1.0e-9;
constexpr double kTwoPi = 2.0 * std::numbers::pi;

[[nodiscard]] Eigen::Vector3d ToEigen(const Vec3& value) {
  return {value.x, value.y, value.z};
}

[[nodiscard]] ValidationIssue Issue(
    std::string path, std::string reason, std::string message) {
  return {
      .field_path = std::move(path),
      .reason_code = std::move(reason),
      .message = std::move(message),
  };
}

}  // namespace

ValidationReport validate_circular_yaw_interval(
    const CircularYawInterval& interval) {
  ValidationReport report;
  if (!std::isfinite(interval.start_rad) ||
      interval.start_rad < -std::numbers::pi ||
      interval.start_rad >= std::numbers::pi) {
    report.issues.push_back(Issue(
        "start_rad", "noncanonical_yaw_interval_start",
        "yaw interval start must be finite and in [-pi, pi)"));
  }
  if (!std::isfinite(interval.span_rad) ||
      interval.span_rad < 0.0 || interval.span_rad > kTwoPi) {
    report.issues.push_back(Issue(
        "span_rad", "yaw_interval_span_out_of_range",
        "yaw interval span must be finite and in [0, 2*pi]"));
  }
  if (!interval.closed) {
    report.issues.push_back(Issue(
        "closed", "closed_yaw_interval_required",
        "yaw interval must be closed"));
  }
  return report;
}

ValidationReport validate_landing_plane(const LandingPlane& plane) {
  ValidationReport report;
  const Eigen::Vector3d origin = ToEigen(plane.origin_m);
  const Eigen::Vector3d normal = ToEigen(plane.normal);
  const Eigen::Vector3d basis_u = ToEigen(plane.basis_u);
  const Eigen::Vector3d basis_v = ToEigen(plane.basis_v);
  if (!origin.allFinite() || !normal.allFinite() ||
      !basis_u.allFinite() || !basis_v.allFinite() ||
      !std::isfinite(plane.residual_bound_m) ||
      plane.residual_bound_m < 0.0) {
    report.issues.push_back(Issue(
        "", "invalid_landing_plane_numeric",
        "landing-plane values must be finite and residual nonnegative"));
    return report;
  }
  if (std::abs(normal.norm() - 1.0) > kGeometryTolerance ||
      std::abs(basis_u.norm() - 1.0) > kGeometryTolerance ||
      std::abs(basis_v.norm() - 1.0) > kGeometryTolerance) {
    report.issues.push_back(Issue(
        "", "unit_vector_required",
        "landing-plane basis vectors must be unit length"));
  }
  if (std::abs(normal.dot(basis_u)) > kGeometryTolerance ||
      std::abs(normal.dot(basis_v)) > kGeometryTolerance ||
      std::abs(basis_u.dot(basis_v)) > kGeometryTolerance) {
    report.issues.push_back(Issue(
        "", "orthogonal_plane_basis_required",
        "landing-plane basis vectors must be orthogonal"));
  }
  if (basis_u.cross(basis_v).dot(normal) <
      1.0 - kGeometryTolerance) {
    report.issues.push_back(Issue(
        "", "right_handed_plane_basis_required",
        "basis_u cross basis_v must align with normal"));
  }
  return report;
}

ValidationReport validate_convex_polygon(
    const ConvexPolygon2d& polygon) {
  ValidationReport report;
  if (polygon.vertices_ccw.size() < 3U) {
    report.issues.push_back(Issue(
        "vertices_ccw", "polygon_requires_three_vertices",
        "convex polygon needs at least three vertices"));
    return report;
  }
  for (const auto& vertex : polygon.vertices_ccw) {
    if (!vertex.allFinite()) {
      report.issues.push_back(Issue(
          "vertices_ccw", "nonfinite_polygon_vertex",
          "polygon vertices must be finite"));
      return report;
    }
  }
  for (std::size_t index = 0U;
       index < polygon.vertices_ccw.size(); ++index) {
    const Eigen::Vector2d& a = polygon.vertices_ccw[index];
    const Eigen::Vector2d& b =
        polygon.vertices_ccw[
            (index + 1U) % polygon.vertices_ccw.size()];
    const Eigen::Vector2d& c =
        polygon.vertices_ccw[
            (index + 2U) % polygon.vertices_ccw.size()];
    const double cross =
        (b.x() - a.x()) * (c.y() - b.y()) -
        (b.y() - a.y()) * (c.x() - b.x());
    if (!(cross > kGeometryTolerance)) {
      report.issues.push_back(Issue(
          "vertices_ccw", "strict_convex_ccw_polygon_required",
          "polygon must be strictly convex and CCW"));
      return report;
    }
  }
  return report;
}

double canonical_yaw(const double yaw_rad) {
  if (!std::isfinite(yaw_rad)) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  double value =
      std::fmod(yaw_rad + std::numbers::pi, kTwoPi);
  if (value < 0.0) {
    value += kTwoPi;
  }
  return value - std::numbers::pi;
}

double yaw_interval_end_unwrapped(
    const CircularYawInterval& interval) {
  return interval.start_rad + interval.span_rad;
}

bool yaw_interval_contains(
    const CircularYawInterval& interval, const double yaw_rad) {
  if (!validate_circular_yaw_interval(interval).ok() ||
      !std::isfinite(yaw_rad)) {
    return false;
  }
  if (interval.span_rad >= kTwoPi - kGeometryTolerance) {
    return true;
  }
  double delta = canonical_yaw(yaw_rad) - interval.start_rad;
  if (delta < 0.0) {
    delta += kTwoPi;
  }
  return delta <= interval.span_rad + kGeometryTolerance;
}

bool yaw_interval_is_subset(
    const CircularYawInterval& child,
    const CircularYawInterval& parent) {
  if (!validate_circular_yaw_interval(child).ok() ||
      !validate_circular_yaw_interval(parent).ok()) {
    return false;
  }
  if (parent.span_rad >= kTwoPi - kGeometryTolerance) {
    return true;
  }
  if (child.span_rad >= kTwoPi - kGeometryTolerance) {
    return false;
  }
  double offset =
      canonical_yaw(child.start_rad) -
      canonical_yaw(parent.start_rad);
  if (offset < 0.0) {
    offset += kTwoPi;
  }
  return offset <= parent.span_rad + kGeometryTolerance &&
         offset + child.span_rad <=
             parent.span_rad + kGeometryTolerance;
}

Eigen::Vector3d plane_uv_to_world(
    const LandingPlane& plane, const Eigen::Vector2d& uv) {
  return ToEigen(plane.origin_m) +
         uv.x() * ToEigen(plane.basis_u) +
         uv.y() * ToEigen(plane.basis_v);
}

Eigen::Vector2d world_to_plane_uv(
    const LandingPlane& plane, const Eigen::Vector3d& point_m) {
  const Eigen::Vector3d relative =
      point_m - ToEigen(plane.origin_m);
  return {
      relative.dot(ToEigen(plane.basis_u)),
      relative.dot(ToEigen(plane.basis_v)),
  };
}

double polygon_signed_area(const ConvexPolygon2d& polygon) {
  double twice_area = 0.0;
  for (std::size_t index = 0U;
       index < polygon.vertices_ccw.size(); ++index) {
    const auto& current = polygon.vertices_ccw[index];
    const auto& next =
        polygon.vertices_ccw[
            (index + 1U) % polygon.vertices_ccw.size()];
    twice_area +=
        current.x() * next.y() - current.y() * next.x();
  }
  return 0.5 * twice_area;
}

bool point_in_convex_polygon(
    const Eigen::Vector2d& point,
    const ConvexPolygon2d& polygon,
    const double inward_margin_m,
    const double tolerance_m) {
  if (!point.allFinite() ||
      !std::isfinite(inward_margin_m) ||
      inward_margin_m < 0.0 ||
      !validate_convex_polygon(polygon).ok()) {
    return false;
  }
  for (std::size_t index = 0U;
       index < polygon.vertices_ccw.size(); ++index) {
    const Eigen::Vector2d& start = polygon.vertices_ccw[index];
    const Eigen::Vector2d& end =
        polygon.vertices_ccw[
            (index + 1U) % polygon.vertices_ccw.size()];
    const Eigen::Vector2d edge = end - start;
    const double cross =
        edge.x() * (point.y() - start.y()) -
        edge.y() * (point.x() - start.x());
    if (cross + tolerance_m <
        inward_margin_m * edge.norm()) {
      return false;
    }
  }
  return true;
}

ConvexPolygonUv to_contract_polygon(
    const ConvexPolygon2d& polygon) {
  ConvexPolygonUv result;
  result.vertices_uv.reserve(polygon.vertices_ccw.size());
  for (const Eigen::Vector2d& vertex : polygon.vertices_ccw) {
    result.vertices_uv.push_back({vertex.x(), vertex.y()});
  }
  return result;
}

ConvexPolygon2d from_contract_polygon(
    const ConvexPolygonUv& polygon) {
  ConvexPolygon2d result;
  result.vertices_ccw.reserve(polygon.vertices_uv.size());
  for (const Vec2& vertex : polygon.vertices_uv) {
    result.vertices_ccw.emplace_back(vertex.x, vertex.y);
  }
  return result;
}

}  // namespace lunar::planning::v3

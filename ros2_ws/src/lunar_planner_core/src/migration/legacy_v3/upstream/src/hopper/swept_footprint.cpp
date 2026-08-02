#include "lunar_path_planner/v3/hopper/swept_footprint.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <numeric>
#include <string>
#include <tuple>

#include <Eigen/LU>

namespace lunar::planning::v3 {
namespace {

constexpr double kTwoPi = 2.0 * std::numbers::pi;

[[nodiscard]] Error Invalid(
    std::string path, std::string message) {
  return {
      .code = ErrorCode::kInvalidArgument,
      .field_path = std::move(path),
      .message = std::move(message),
  };
}

[[nodiscard]] double ProjectionAt(
    const Eigen::Vector2d& normal,
    const Eigen::Vector2d& vertex,
    const double angle) {
  const double cosine = std::cos(angle);
  const double sine = std::sin(angle);
  const Eigen::Vector2d rotated{
      cosine * vertex.x() - sine * vertex.y(),
      sine * vertex.x() + cosine * vertex.y()};
  return normal.dot(rotated);
}

[[nodiscard]] double MaximumProjection(
    const Eigen::Vector2d& normal,
    const Eigen::Vector2d& vertex,
    const CircularYawInterval& yaw) {
  const double a = normal.dot(vertex);
  const double b =
      normal.dot(Eigen::Vector2d{-vertex.y(), vertex.x()});
  const double start = yaw.start_rad;
  const double end = yaw_interval_end_unwrapped(yaw);
  double maximum = std::max(
      ProjectionAt(normal, vertex, start),
      ProjectionAt(normal, vertex, end));
  const double base = std::atan2(b, a);
  const int first_turn = static_cast<int>(
      std::floor((start - base) / kTwoPi)) - 1;
  const int last_turn = static_cast<int>(
      std::ceil((end - base) / kTwoPi)) + 1;
  for (int turn = first_turn; turn <= last_turn; ++turn) {
    const double stationary =
        base + static_cast<double>(turn) * kTwoPi;
    if (stationary >= start - 1.0e-12 &&
        stationary <= end + 1.0e-12) {
      maximum = std::max(
          maximum,
          ProjectionAt(normal, vertex, stationary));
    }
  }
  return maximum;
}

[[nodiscard]] bool Satisfies(
    const Eigen::Vector2d& point,
    std::span<const SupportHalfspace2> halfspaces,
    const double tolerance) {
  return std::ranges::all_of(
      halfspaces, [&point, tolerance](const auto& halfspace) {
        return halfspace.outward_unit_normal.dot(point) <=
               halfspace.upper_bound_m + tolerance;
      });
}

[[nodiscard]] Result<ConvexPolygon2d> IntersectHalfspaces(
    std::span<const SupportHalfspace2> halfspaces) {
  std::vector<Eigen::Vector2d> vertices;
  for (std::size_t first = 0U; first < halfspaces.size(); ++first) {
    for (std::size_t second = first + 1U;
         second < halfspaces.size(); ++second) {
      Eigen::Matrix2d matrix;
      matrix.row(0) =
          halfspaces[first].outward_unit_normal.transpose();
      matrix.row(1) =
          halfspaces[second].outward_unit_normal.transpose();
      if (std::abs(matrix.determinant()) <= 1.0e-12) {
        continue;
      }
      const Eigen::Vector2d point =
          matrix.inverse() *
          Eigen::Vector2d{
              halfspaces[first].upper_bound_m,
              halfspaces[second].upper_bound_m};
      if (point.allFinite() &&
          Satisfies(point, halfspaces, 1.0e-9) &&
          !std::ranges::any_of(
              vertices, [&point](const Eigen::Vector2d& other) {
                return (point - other).norm() <= 1.0e-9;
              })) {
        vertices.push_back(point);
      }
    }
  }
  if (vertices.size() < 3U) {
    return Invalid(
        "support_directions",
        "SWEPT_FOOTPRINT_HALFSPACE_INTERSECTION_UNBOUNDED");
  }
  const Eigen::Vector2d center =
      std::accumulate(
          vertices.begin(), vertices.end(),
          Eigen::Vector2d::Zero().eval()) /
      static_cast<double>(vertices.size());
  std::sort(
      vertices.begin(), vertices.end(),
      [&center](const Eigen::Vector2d& lhs,
                const Eigen::Vector2d& rhs) {
        return std::tuple{
                   std::atan2(lhs.y() - center.y(),
                              lhs.x() - center.x()),
                   lhs.x(), lhs.y()} <
               std::tuple{
                   std::atan2(rhs.y() - center.y(),
                              rhs.x() - center.x()),
                   rhs.x(), rhs.y()};
      });
  ConvexPolygon2d polygon{std::move(vertices)};
  if (!validate_convex_polygon(polygon).ok()) {
    return Invalid(
        "support_directions",
        "SWEPT_FOOTPRINT_INTERSECTION_DEGENERATE");
  }
  return polygon;
}

}  // namespace

std::vector<Eigen::Vector2d> make_uniform_unit_directions(
    const std::size_t count) {
  std::vector<Eigen::Vector2d> directions;
  if (count < 4U) {
    return directions;
  }
  directions.reserve(count);
  for (std::size_t index = 0U; index < count; ++index) {
    const double angle =
        kTwoPi * static_cast<double>(index) /
        static_cast<double>(count);
    directions.emplace_back(std::cos(angle), std::sin(angle));
  }
  return directions;
}

Result<SweptFootprintEnvelope>
outer_approximate_rotated_footprint(
    const ConvexPolygon2d& body_frame_footprint,
    const CircularYawInterval& yaw_interval,
    const std::span<const Eigen::Vector2d> support_directions,
    const double deterministic_margin_m) {
  if (!validate_convex_polygon(body_frame_footprint).ok()) {
    return Invalid(
        "body_frame_footprint", "INVALID_LANDING_FOOTPRINT");
  }
  if (!validate_circular_yaw_interval(yaw_interval).ok()) {
    return Invalid("yaw_interval", "INVALID_YAW_INTERVAL");
  }
  if (!std::isfinite(deterministic_margin_m) ||
      deterministic_margin_m < 0.0 ||
      support_directions.size() < 4U) {
    return Invalid(
        "support_directions", "INVALID_SUPPORT_DIRECTIONS");
  }
  for (const Eigen::Vector2d& direction : support_directions) {
    if (!direction.allFinite() ||
        std::abs(direction.norm() - 1.0) > 1.0e-9) {
      return Invalid(
          "support_directions", "NON_UNIT_SUPPORT_DIRECTION");
    }
  }

  SweptFootprintEnvelope result;
  result.certified_yaw_interval = yaw_interval;
  result.deterministic_margin_m = deterministic_margin_m;
  result.halfspaces.reserve(support_directions.size());
  for (const Eigen::Vector2d& direction : support_directions) {
    double bound = -std::numeric_limits<double>::infinity();
    for (const Eigen::Vector2d& vertex :
         body_frame_footprint.vertices_ccw) {
      bound = std::max(
          bound,
          MaximumProjection(direction, vertex, yaw_interval));
    }
    result.halfspaces.push_back(
        {direction, bound + deterministic_margin_m});
  }
  const auto polygon = IntersectHalfspaces(result.halfspaces);
  if (!IsOk(polygon)) {
    return std::get<Error>(polygon);
  }
  result.vertices_ccw = std::get<ConvexPolygon2d>(polygon);
  return result;
}

bool polygon_is_contained(
    const ConvexPolygon2d& polygon,
    const SweptFootprintEnvelope& envelope,
    const double tolerance_m) {
  if (!std::isfinite(tolerance_m) || tolerance_m < 0.0) {
    return false;
  }
  for (const Eigen::Vector2d& vertex : polygon.vertices_ccw) {
    if (!Satisfies(vertex, envelope.halfspaces, tolerance_m)) {
      return false;
    }
  }
  return true;
}

}  // namespace lunar::planning::v3

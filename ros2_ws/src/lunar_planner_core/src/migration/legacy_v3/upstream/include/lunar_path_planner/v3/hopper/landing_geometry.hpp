#pragma once

#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "lunar_path_planner/v3/codec/semantic_validator.hpp"
#include "lunar_path_planner/v3/contracts/planning_request.hpp"

namespace lunar::planning::v3 {

struct ConvexPolygon2d final {
  std::vector<Eigen::Vector2d> vertices_ccw;
};

[[nodiscard]] ValidationReport validate_circular_yaw_interval(
    const CircularYawInterval& interval);
[[nodiscard]] ValidationReport validate_landing_plane(
    const LandingPlane& plane);
[[nodiscard]] ValidationReport validate_convex_polygon(
    const ConvexPolygon2d& polygon);

[[nodiscard]] double canonical_yaw(double yaw_rad);
[[nodiscard]] double yaw_interval_end_unwrapped(
    const CircularYawInterval& interval);
[[nodiscard]] bool yaw_interval_contains(
    const CircularYawInterval& interval, double yaw_rad);
[[nodiscard]] bool yaw_interval_is_subset(
    const CircularYawInterval& child,
    const CircularYawInterval& parent);

[[nodiscard]] Eigen::Vector3d plane_uv_to_world(
    const LandingPlane& plane, const Eigen::Vector2d& uv);
[[nodiscard]] Eigen::Vector2d world_to_plane_uv(
    const LandingPlane& plane, const Eigen::Vector3d& point_m);

[[nodiscard]] double polygon_signed_area(
    const ConvexPolygon2d& polygon);
[[nodiscard]] bool point_in_convex_polygon(
    const Eigen::Vector2d& point,
    const ConvexPolygon2d& polygon,
    double inward_margin_m = 0.0,
    double tolerance_m = 1.0e-10);

[[nodiscard]] ConvexPolygonUv to_contract_polygon(
    const ConvexPolygon2d& polygon);
[[nodiscard]] ConvexPolygon2d from_contract_polygon(
    const ConvexPolygonUv& polygon);

}  // namespace lunar::planning::v3

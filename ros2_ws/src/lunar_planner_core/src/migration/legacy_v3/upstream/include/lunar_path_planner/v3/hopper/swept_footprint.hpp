#pragma once

#include <span>
#include <vector>

#include <Eigen/Core>

#include "lunar_path_planner/v3/contracts/status.hpp"
#include "lunar_path_planner/v3/hopper/landing_geometry.hpp"

namespace lunar::planning::v3 {

struct SupportHalfspace2 final {
  Eigen::Vector2d outward_unit_normal;
  double upper_bound_m{};
};

struct SweptFootprintEnvelope final {
  std::vector<SupportHalfspace2> halfspaces;
  ConvexPolygon2d vertices_ccw;
  CircularYawInterval certified_yaw_interval;
  double deterministic_margin_m{};
};

[[nodiscard]] std::vector<Eigen::Vector2d>
make_uniform_unit_directions(std::size_t count);

[[nodiscard]] Result<SweptFootprintEnvelope>
outer_approximate_rotated_footprint(
    const ConvexPolygon2d& body_frame_footprint,
    const CircularYawInterval& yaw_interval,
    std::span<const Eigen::Vector2d> support_directions,
    double deterministic_margin_m);

[[nodiscard]] bool polygon_is_contained(
    const ConvexPolygon2d& polygon,
    const SweptFootprintEnvelope& envelope,
    double tolerance_m);

}  // namespace lunar::planning::v3

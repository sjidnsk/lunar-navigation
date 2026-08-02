#include "lunar_path_planner/v3/hopper/landing_set_propagator.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <type_traits>
#include <utility>

#include "lunar_path_planner/v3/hopper/landing_geometry.hpp"

namespace lunar::planning::v3 {
namespace {

[[nodiscard]] Eigen::Vector3d ToEigen(const Vec3 value) {
  return {value.x, value.y, value.z};
}

[[nodiscard]] Vec3 ToContract(const Eigen::Vector3d& value) {
  return {value.x(), value.y(), value.z()};
}

[[nodiscard]] Eigen::Vector3d BoxRadius(
    const AxisAlignedBox3& box) {
  return ToEigen(box.center).cwiseAbs() +
         ToEigen(box.half_extent);
}

[[nodiscard]] bool ValidBox(const AxisAlignedBox3& box) {
  const Eigen::Vector3d center = ToEigen(box.center);
  const Eigen::Vector3d half = ToEigen(box.half_extent);
  return center.allFinite() && half.allFinite() &&
         (half.array() >= 0.0).all();
}

[[nodiscard]] DurationNanoseconds DurationFloor(
    const double seconds) {
  return DurationNanoseconds{
      std::chrono::nanoseconds{
          static_cast<std::int64_t>(
              std::floor(seconds * 1.0e9))}};
}

[[nodiscard]] DurationNanoseconds DurationCeil(
    const double seconds) {
  return DurationNanoseconds{
      std::chrono::nanoseconds{
          static_cast<std::int64_t>(
              std::ceil(seconds * 1.0e9))}};
}

[[nodiscard]] bool SameVec(
    const Vec3 lhs, const Vec3 rhs,
    const double tolerance = 1.0e-9) {
  return (ToEigen(lhs) - ToEigen(rhs)).norm() <= tolerance;
}

void AddIssue(
    ValidationReport& report, std::string field,
    std::string code, std::string message) {
  report.issues.push_back(
      {std::move(field), std::move(code), std::move(message)});
}

}  // namespace

LandingSetPropagator::LandingSetPropagator(
    HopperCapabilityView capability,
    HopperPlannerLimits limits)
    : capability_(std::move(capability)),
      limits_(std::move(limits)) {}

LandingSetPropagationResult LandingSetPropagator::propagate(
    const LandingSetPropagationInput& input) const {
  LandingSetPropagationResult result{};
  result.diagnostics.support_direction_count =
      limits_.support_direction_count;
  if (!validate_landing_plane(input.landing_plane).ok() ||
      !validate_circular_yaw_interval(
           input.certified_landing_yaw_interval)
           .ok() ||
      !std::isfinite(input.arc.flight_time_s) ||
      input.arc.flight_time_s <= 0.0 ||
      !input.arc.landing_velocity_mps.allFinite() ||
      !ValidBox(input.initial_position_error_m) ||
      !ValidBox(input.initial_velocity_error_mps) ||
      !ValidBox(input.gravity_error_mps2) ||
      !ValidBox(input.launch_execution_velocity_error_mps) ||
      !ValidBox(input.landing_plane_error.origin_error_m) ||
      !std::isfinite(
          input.landing_plane_error.normal_error.radius_rad) ||
      input.landing_plane_error.normal_error.radius_rad < 0.0 ||
      !std::isfinite(
          input.landing_plane_error.residual_error_m.center) ||
      !std::isfinite(
          input.landing_plane_error.residual_error_m.half_width) ||
      input.landing_plane_error.residual_error_m.half_width < 0.0) {
    result.diagnostics.rejection_reason = "invalid_input";
    return result;
  }

  const Eigen::Vector3d normal =
      ToEigen(input.landing_plane.normal).normalized();
  const double time = input.arc.flight_time_s;
  const Eigen::Vector3d velocity_radius =
      BoxRadius(input.initial_velocity_error_mps) +
      BoxRadius(input.launch_execution_velocity_error_mps) +
      time * BoxRadius(input.gravity_error_mps2);
  const double normal_velocity_uncertainty =
      velocity_radius.dot(normal.cwiseAbs()) +
      input.arc.landing_velocity_mps.norm() *
          input.landing_plane_error.normal_error.radius_rad;
  const double worst_upward_normal_velocity =
      input.arc.landing_velocity_mps.dot(normal) +
      normal_velocity_uncertainty;
  result.diagnostics.worst_case_downward_normal_speed_mps =
      -worst_upward_normal_velocity;
  if (!(worst_upward_normal_velocity <
        -capability_.launch_limits
             .minimum_downward_impact_speed_mps)) {
    result.diagnostics.rejection_reason =
        "downward_transversality_not_proven";
    return result;
  }

  const Eigen::Vector3d position_radius =
      BoxRadius(input.initial_position_error_m) +
      time *
          (BoxRadius(input.initial_velocity_error_mps) +
           BoxRadius(
               input.launch_execution_velocity_error_mps)) +
      0.5 * time * time *
          BoxRadius(input.gravity_error_mps2) +
      BoxRadius(input.landing_plane_error.origin_error_m);
  const double scalar_plane_error =
      std::abs(
          input.landing_plane_error.residual_error_m.center) +
      input.landing_plane_error.residual_error_m.half_width +
      capability_.actuator_or_impulse_profile
              .nominal_landing_center_normal_offset_m *
          input.landing_plane_error.normal_error.radius_rad;
  const Eigen::Vector3d basis_u =
      ToEigen(input.landing_plane.basis_u).normalized();
  const Eigen::Vector3d basis_v =
      ToEigen(input.landing_plane.basis_v).normalized();
  const double radius_u =
      basis_u.cwiseAbs().dot(position_radius) +
      scalar_plane_error;
  const double radius_v =
      basis_v.cwiseAbs().dot(position_radius) +
      scalar_plane_error;
  if (!std::isfinite(radius_u) || !std::isfinite(radius_v)) {
    result.diagnostics.rejection_reason =
        "landing_set_numerical_failure";
    return result;
  }
  const Eigen::Vector2d center =
      world_to_plane_uv(
          input.landing_plane, input.arc.aim_position_m);
  const double safe_radius_u = std::max(radius_u, 1.0e-12);
  const double safe_radius_v = std::max(radius_v, 1.0e-12);
  ConvexPolygonUv polygon{
      .vertices_uv =
          {
              {center.x() - safe_radius_u,
               center.y() - safe_radius_v},
              {center.x() + safe_radius_u,
               center.y() - safe_radius_v},
              {center.x() + safe_radius_u,
               center.y() + safe_radius_v},
              {center.x() - safe_radius_u,
               center.y() + safe_radius_v},
          },
  };

  const double signed_distance_radius =
      normal.cwiseAbs().dot(position_radius) +
      scalar_plane_error;
  const double minimum_downward_speed =
      -worst_upward_normal_velocity;
  const double time_radius =
      signed_distance_radius /
      std::max(minimum_downward_speed, 1.0e-12);
  const double time_begin = std::max(0.0, time - time_radius);
  const double time_end = time + time_radius;
  const Eigen::Vector3d acceleration_radius =
      BoxRadius(input.gravity_error_mps2);
  const Eigen::Vector3d total_velocity_radius =
      velocity_radius +
      acceleration_radius * time_radius;
  result.footprint = PredictedLandingFootprint{
      .landing_plane = input.landing_plane,
      .convex_center_landing_polygon = std::move(polygon),
      .landing_time_window =
          {
              .start_offset = DurationFloor(time_begin),
              .end_offset = DurationCeil(time_end),
          },
      .landing_velocity_bounds =
          {
              .lower = ToContract(
                  input.arc.landing_velocity_mps -
                  total_velocity_radius),
              .upper = ToContract(
                  input.arc.landing_velocity_mps +
                  total_velocity_radius),
          },
      .landing_yaw_interval =
          input.certified_landing_yaw_interval,
      .source_error_model_ref = input.source_error_model_ref,
      .outer_approximation_margin_m = 0.0,
  };
  return result;
}

ValidationReport validate_landing_containment(
    const PredictedLandingFootprint& footprint,
    const NextLandingRegion& region,
    const AttitudeBoundary& attitude_boundary) {
  ValidationReport report{};
  if (region.frame_id.empty() ||
      !SameVec(
          footprint.landing_plane.origin_m,
          region.landing_plane.origin_m) ||
      !SameVec(
          footprint.landing_plane.normal,
          region.landing_plane.normal) ||
      !SameVec(
          footprint.landing_plane.basis_u,
          region.landing_plane.basis_u) ||
      !SameVec(
          footprint.landing_plane.basis_v,
          region.landing_plane.basis_v)) {
    AddIssue(
        report, "landing_plane", "landing_plane_mismatch",
        "landing footprint and region planes must match");
    return report;
  }
  const ConvexPolygon2d region_polygon =
      from_contract_polygon(region.convex_polygon);
  const auto region_validation =
      validate_convex_polygon(region_polygon);
  if (!region_validation.ok()) {
    AddIssue(
        report, "next_landing_region.convex_polygon",
        "invalid_landing_region", "landing region is invalid");
    return report;
  }
  const double total_margin =
      region.inward_safety_margin_m +
      footprint.outer_approximation_margin_m;
  for (const Vec2 vertex :
       footprint.convex_center_landing_polygon.vertices_uv) {
    if (!point_in_convex_polygon(
            Eigen::Vector2d{vertex.x, vertex.y},
            region_polygon, total_margin)) {
      AddIssue(
          report,
          "predicted_landing_footprint."
          "convex_center_landing_polygon",
          "landing_footprint_not_contained",
          "margin-expanded landing set leaves region");
      break;
    }
  }
  if (!yaw_interval_is_subset(
          footprint.landing_yaw_interval,
          attitude_boundary.target_attitude_set
              .allowed_yaw_interval)) {
    AddIssue(
        report, "predicted_landing_footprint.landing_yaw_interval",
        "landing_yaw_not_subset_of_target_attitude",
        "landing yaw must be inside target attitude yaw");
  }
  if (!yaw_interval_is_subset(
          attitude_boundary.target_attitude_set
              .allowed_yaw_interval,
          region.allowed_yaw_interval)) {
    AddIssue(
        report,
        "attitude_boundary.target_attitude_set."
        "allowed_yaw_interval",
        "target_attitude_yaw_not_subset_of_landing_region",
        "target attitude yaw must be inside landing region yaw");
  }
  return report;
}

}  // namespace lunar::planning::v3

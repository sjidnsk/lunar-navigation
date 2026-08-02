#include "lunar_path_planner/v3/hopper/ballistic_kinematics.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace lunar::planning::v3 {
namespace {

[[nodiscard]] Eigen::Vector3d ToEigen(const Vec3 value) {
  return {value.x, value.y, value.z};
}

[[nodiscard]] bool PointInsideBox(
    const Eigen::Vector3d& point, const AxisAlignedBox3& box) {
  const Eigen::Vector3d center = ToEigen(box.center);
  const Eigen::Vector3d half_extent = ToEigen(box.half_extent);
  return point.allFinite() && center.allFinite() &&
         half_extent.allFinite() &&
         (half_extent.array() >= 0.0).all() &&
         ((point - center).array().abs() <=
          half_extent.array() + 1.0e-12)
             .all();
}

[[nodiscard]] Error Invalid(
    std::string field_path, std::string message) {
  return {
      .code = ErrorCode::kInvalidArgument,
      .field_path = std::move(field_path),
      .message = std::move(message),
  };
}

}  // namespace

GravityModelView gravity_model_view(const GravityModel& model) {
  return {
      .model_ref = model.content_ref,
      .frame_id = model.frame_id,
      .nominal_acceleration_mps2 =
          model.nominal_acceleration_mps2,
      .acceleration_error_mps2 =
          model.acceleration_error_mps2,
      .spatial_validity =
          SpatialValidityRegion{model.spatial_validity_m},
      .time_validity =
          TimeValidityInterval{
              model.valid_from, model.valid_until},
  };
}

Result<NominalBallisticArc> make_nominal_ballistic_arc(
    const Eigen::Vector3d& launch_position_m,
    const std::string_view launch_frame_id,
    const AimPointCandidate& aim_point,
    const GravityModelView& gravity,
    const double flight_time_s) {
  if (launch_frame_id.empty() ||
      aim_point.frame_id != launch_frame_id ||
      gravity.frame_id != launch_frame_id) {
    return Invalid("frame_id", "ballistic frames must match");
  }
  if (!launch_position_m.allFinite() ||
      !aim_point.position_m.allFinite() ||
      !gravity.nominal_acceleration_mps2.allFinite() ||
      !std::isfinite(flight_time_s) || flight_time_s <= 0.0) {
    return Invalid(
        "flight_time_s",
        "ballistic inputs must be finite and time positive");
  }
  const Eigen::Vector3d displacement =
      aim_point.position_m - launch_position_m;
  const Eigen::Vector3d launch_velocity =
      displacement / flight_time_s -
      0.5 * gravity.nominal_acceleration_mps2 *
          flight_time_s;
  const Eigen::Vector3d landing_velocity =
      launch_velocity +
      gravity.nominal_acceleration_mps2 * flight_time_s;
  if (!launch_velocity.allFinite() ||
      !landing_velocity.allFinite()) {
    return Error{
        .code = ErrorCode::kNumericalFailure,
        .field_path = "launch_velocity_mps",
        .message = "ballistic velocity is nonfinite",
    };
  }

  NominalBallisticArc arc{
      .aim_point_id = aim_point.aim_point_id,
      .frame_id = std::string(launch_frame_id),
      .launch_position_m = launch_position_m,
      .aim_position_m = aim_point.position_m,
      .gravity_mps2 = gravity.nominal_acceleration_mps2,
      .launch_velocity_mps = launch_velocity,
      .landing_velocity_mps = landing_velocity,
      .flight_time_s = flight_time_s,
  };

  const double apex_time = ballistic_apex_time_s(arc);
  if (!PointInsideBox(
          arc.launch_position_m,
          gravity.spatial_validity.bounds_m) ||
      !PointInsideBox(
          arc.aim_position_m,
          gravity.spatial_validity.bounds_m) ||
      !PointInsideBox(
          evaluate_ballistic_state(arc, apex_time).position_m,
          gravity.spatial_validity.bounds_m)) {
    return Invalid(
        "gravity.spatial_validity",
        "nominal ballistic arc leaves gravity validity bounds");
  }
  return arc;
}

BallisticState evaluate_ballistic_state(
    const NominalBallisticArc& arc, const double time_s) {
  return {
      .position_m =
          arc.launch_position_m +
          arc.launch_velocity_mps * time_s +
          0.5 * arc.gravity_mps2 * time_s * time_s,
      .velocity_mps =
          arc.launch_velocity_mps + arc.gravity_mps2 * time_s,
  };
}

double ballistic_apex_time_s(const NominalBallisticArc& arc) {
  const double gravity_squared = arc.gravity_mps2.squaredNorm();
  if (!std::isfinite(gravity_squared) ||
      gravity_squared <=
          std::numeric_limits<double>::epsilon()) {
    return 0.0;
  }
  const double stationary =
      -arc.launch_velocity_mps.dot(arc.gravity_mps2) /
      gravity_squared;
  return std::clamp(stationary, 0.0, arc.flight_time_s);
}

}  // namespace lunar::planning::v3

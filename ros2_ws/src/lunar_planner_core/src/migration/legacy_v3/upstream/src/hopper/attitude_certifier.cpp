#include "lunar_path_planner/v3/hopper/attitude_certifier.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numbers>
#include <type_traits>
#include <utility>
#include <vector>

#include "lunar_path_planner/v3/hopper/landing_geometry.hpp"

namespace lunar::planning::v3 {
namespace {

[[nodiscard]] Eigen::Vector3d ToEigen(const Vec3 value) {
  return {value.x, value.y, value.z};
}

[[nodiscard]] Vec3 ToContract(const Eigen::Vector3d& value) {
  return {value.x(), value.y(), value.z()};
}

[[nodiscard]] Quaternion ToContract(
    Eigen::Quaterniond value) {
  value.normalize();
  if (value.w() < 0.0) {
    value.coeffs() *= -1.0;
  }
  return {value.w(), value.x(), value.y(), value.z()};
}

[[nodiscard]] double MaximumVectorNorm(
    const DeterministicVectorSet3& set) {
  return std::visit(
      [](const auto& value) {
        using Set = std::decay_t<decltype(value)>;
        const Eigen::Vector3d center = ToEigen(value.center);
        if constexpr (std::is_same_v<Set, AxisAlignedBox3>) {
          const Eigen::Vector3d half = ToEigen(value.half_extent);
          return (center.cwiseAbs() + half).norm();
        } else {
          return center.norm() + value.radius;
        }
      },
      set);
}

[[nodiscard]] bool FiniteQuaternion(
    const Eigen::Quaterniond& value) {
  return value.coeffs().allFinite() && value.norm() > 1.0e-12;
}

[[nodiscard]] Eigen::Quaterniond TargetOrientation(
    const LandingPlane& plane, const double yaw_rad) {
  const Eigen::Vector3d normal = ToEigen(plane.normal).normalized();
  const Eigen::Vector3d basis_u = ToEigen(plane.basis_u).normalized();
  const Eigen::Vector3d basis_v = ToEigen(plane.basis_v).normalized();
  const Eigen::Vector3d forward =
      std::cos(yaw_rad) * basis_u +
      std::sin(yaw_rad) * basis_v;
  const Eigen::Vector3d lateral =
      normal.cross(forward).normalized();
  Eigen::Matrix3d rotation;
  rotation.col(0) = forward.normalized();
  rotation.col(1) = lateral;
  rotation.col(2) = normal;
  Eigen::Quaterniond result{rotation};
  result.normalize();
  if (result.w() < 0.0) {
    result.coeffs() *= -1.0;
  }
  return result;
}

struct TargetChoice final {
  Eigen::Quaterniond orientation{Eigen::Quaterniond::Identity()};
  double yaw_rad{};
  double angle_rad{std::numeric_limits<double>::infinity()};
};

[[nodiscard]] std::optional<TargetChoice> ChooseTarget(
    const Eigen::Quaterniond& initial,
    const LandingPlane& plane,
    const CircularYawInterval& allowed_yaw) {
  if (!validate_landing_plane(plane).ok() ||
      !validate_circular_yaw_interval(allowed_yaw).ok()) {
    return std::nullopt;
  }
  const std::vector<double> yaw_candidates{
      canonical_yaw(allowed_yaw.start_rad),
      canonical_yaw(
          allowed_yaw.start_rad + 0.5 * allowed_yaw.span_rad),
      canonical_yaw(
          allowed_yaw.start_rad + allowed_yaw.span_rad),
  };
  TargetChoice best{};
  for (const double yaw : yaw_candidates) {
    const Eigen::Quaterniond target =
        TargetOrientation(plane, yaw);
    const double angle =
        shortest_quaternion_angle_rad(initial, target);
    if (angle < best.angle_rad - 1.0e-12 ||
        (std::abs(angle - best.angle_rad) <= 1.0e-12 &&
         yaw < best.yaw_rad)) {
      best = {target, yaw, angle};
    }
  }
  return std::isfinite(best.angle_rad)
             ? std::optional<TargetChoice>{best}
             : std::nullopt;
}

struct EffectiveLimits final {
  double speed{};
  double acceleration{};
};

[[nodiscard]] EffectiveLimits EffectiveForAngle(
    const ArbitraryAxisAttitudeCapability& capability,
    const double angle_rad) {
  EffectiveLimits result{
      capability.maximum_angular_speed_radps,
      capability.maximum_angular_acceleration_radps2};
  if (capability.resolved_attitude_tightening_table &&
      !capability.resolved_attitude_tightening_table
           ->entries.empty()) {
    const auto& entries =
        capability.resolved_attitude_tightening_table->entries;
    const auto iterator = std::find_if(
        entries.begin(), entries.end(),
        [angle_rad](const AttitudeTighteningEntry& entry) {
          return angle_rad <= entry.maximum_rotation_angle_rad +
                                  1.0e-12;
        });
    const AttitudeTighteningEntry& selected =
        iterator == entries.end() ? entries.back() : *iterator;
    result.speed =
        std::min(result.speed, selected.maximum_angular_speed_radps);
    result.acceleration = std::min(
        result.acceleration,
        selected.maximum_angular_acceleration_radps2);
  }
  return result;
}

[[nodiscard]] Error Invalid(std::string message) {
  return {
      .code = ErrorCode::kInvalidArgument,
      .field_path = "attitude",
      .message = std::move(message),
  };
}

}  // namespace

ArbitraryAxisAttitudeCapability attitude_capability_view(
    const HopperCapabilityView& capability) {
  std::shared_ptr<const AttitudeTighteningTable> table;
  std::optional<ContentRef> table_ref;
  if (capability.attitude_tightening_table.has_value()) {
    table = std::make_shared<const AttitudeTighteningTable>(
        *capability.attitude_tightening_table);
    table_ref = table->content_ref;
  }
  return {
      .maximum_angular_speed_radps =
          capability.attitude_envelope.maximum_angular_speed_radps,
      .maximum_angular_acceleration_radps2 =
          capability.attitude_envelope
              .maximum_angular_acceleration_radps2,
      .maximum_initial_angular_speed_radps =
          capability.attitude_envelope
              .maximum_initial_angular_speed_radps,
      .minimum_settle_guard =
          capability.attitude_envelope.minimum_settle_guard,
      .source_safety_capability_ref =
          capability.source_safety_capability_ref,
      .attitude_tightening_table_ref = table_ref,
      .resolved_attitude_tightening_table = std::move(table),
      .certified_state_error_bounds =
          capability.certified_state_error_bounds,
  };
}

AttitudeCertifier::AttitudeCertifier(
    ArbitraryAxisAttitudeCapability capability)
    : capability_(std::move(capability)) {}

Result<double> AttitudeCertifier::conservative_required_time_s(
    const Eigen::Quaterniond& initial_orientation_body_to_frame,
    const Eigen::Vector3d& initial_angular_velocity_radps,
    const RotationVectorBall& initial_orientation_error_set,
    const DeterministicVectorSet3&
        initial_angular_velocity_error_set_radps,
    const LandingPlane& landing_plane,
    const CircularYawInterval& allowed_yaw_interval) const {
  if (!FiniteQuaternion(initial_orientation_body_to_frame) ||
      !initial_angular_velocity_radps.allFinite() ||
      !std::isfinite(initial_orientation_error_set.radius_rad) ||
      initial_orientation_error_set.radius_rad < 0.0) {
    return Invalid("nonfinite or invalid initial attitude");
  }
  const auto target = ChooseTarget(
      initial_orientation_body_to_frame.normalized(),
      landing_plane, allowed_yaw_interval);
  if (!target.has_value()) {
    return Invalid("invalid target attitude set");
  }
  const double omega0_bound =
      initial_angular_velocity_radps.norm() +
      MaximumVectorNorm(
          initial_angular_velocity_error_set_radps);
  if (!std::isfinite(omega0_bound) ||
      omega0_bound >
          capability_.maximum_initial_angular_speed_radps +
              1.0e-12) {
    return Invalid(
        "launch angular velocity outside certified bound");
  }
  const double angle_bound = std::min(
      std::numbers::pi,
      target->angle_rad +
          initial_orientation_error_set.radius_rad);
  const EffectiveLimits limits =
      EffectiveForAngle(capability_, angle_bound);
  if (!std::isfinite(limits.speed) ||
      !std::isfinite(limits.acceleration) ||
      limits.speed <= 0.0 || limits.acceleration <= 0.0) {
    return Invalid("invalid effective attitude limits");
  }
  const double brake_time = omega0_bound / limits.acceleration;
  const double adverse_brake_angle =
      omega0_bound * omega0_bound /
      (2.0 * limits.acceleration);
  return brake_time +
         bang_bang_rotation_time_s(
             std::min(
                 std::numbers::pi,
                 angle_bound + adverse_brake_angle),
             limits.speed, limits.acceleration);
}

AttitudeCertificationResult AttitudeCertifier::certify(
    const AttitudeCertificationInput& input) const {
  AttitudeCertificationResult result{};
  const auto target = ChooseTarget(
      input.initial_orientation_body_to_frame,
      input.landing_plane, input.allowed_yaw_interval);
  if (!target.has_value()) {
    result.rejection_reason = "invalid_target_attitude_set";
    return result;
  }
  result.shortest_rotation_angle_rad = target->angle_rad;
  result.worst_case_initial_angular_speed_radps =
      input.initial_angular_velocity_radps.norm() +
      MaximumVectorNorm(
          input.initial_angular_velocity_error_set_radps);
  if (result.worst_case_initial_angular_speed_radps >
      capability_.maximum_initial_angular_speed_radps +
          1.0e-12) {
    result.rejection_reason =
        "launch_angular_velocity_outside_certified_bound";
    return result;
  }
  const auto required = conservative_required_time_s(
      input.initial_orientation_body_to_frame,
      input.initial_angular_velocity_radps,
      input.initial_orientation_error_set,
      input.initial_angular_velocity_error_set_radps,
      input.landing_plane, input.allowed_yaw_interval);
  if (!IsOk(required)) {
    result.rejection_reason = "attitude_numerical_failure";
    return result;
  }
  result.certified_required_rotation_time_s =
      std::get<double>(required);
  const EffectiveLimits effective = EffectiveForAngle(
      capability_,
      std::min(
          std::numbers::pi,
          target->angle_rad +
              input.initial_orientation_error_set.radius_rad));
  result.effective_maximum_speed_radps = effective.speed;
  result.effective_maximum_acceleration_radps2 =
      effective.acceleration;
  const double settle_seconds = std::chrono::duration<double>(
      capability_.minimum_settle_guard.value)
                                    .count();
  if (!std::isfinite(input.flight_time_s) ||
      input.flight_time_s <
          result.certified_required_rotation_time_s +
              settle_seconds) {
    result.rejection_reason =
        "insufficient_attitude_settle_time";
    return result;
  }
  result.boundary = AttitudeBoundary{
      .initial_orientation_error_set =
          input.initial_orientation_error_set,
      .initial_angular_velocity_error_set_radps =
          input.initial_angular_velocity_error_set_radps,
      .target_attitude_set =
          {
              .nominal_orientation_body_to_frame =
                  ToContract(target->orientation),
              .orientation_error_set = {0.0},
              .allowed_yaw_interval =
                  input.allowed_yaw_interval,
          },
      .landing_angular_velocity_bounds_radps =
          EuclideanBall3{
              .center = ToContract(Eigen::Vector3d::Zero()),
              .radius = 0.0,
          },
      .settle_guard = capability_.minimum_settle_guard,
      .certification_ref =
          capability_.attitude_tightening_table_ref.value_or(
              capability_.source_safety_capability_ref),
  };
  return result;
}

double shortest_quaternion_angle_rad(
    const Eigen::Quaterniond& from,
    const Eigen::Quaterniond& to) {
  if (!FiniteQuaternion(from) || !FiniteQuaternion(to)) {
    return std::numeric_limits<double>::infinity();
  }
  const double dot = std::clamp(
      std::abs(from.normalized().dot(to.normalized())),
      0.0, 1.0);
  return 2.0 * std::acos(dot);
}

double bang_bang_rotation_time_s(
    const double angle_rad,
    const double maximum_speed_radps,
    const double maximum_acceleration_radps2) {
  if (!std::isfinite(angle_rad) ||
      !std::isfinite(maximum_speed_radps) ||
      !std::isfinite(maximum_acceleration_radps2) ||
      angle_rad < 0.0 || maximum_speed_radps <= 0.0 ||
      maximum_acceleration_radps2 <= 0.0) {
    return std::numeric_limits<double>::infinity();
  }
  const double switching_angle =
      maximum_speed_radps * maximum_speed_radps /
      maximum_acceleration_radps2;
  if (angle_rad <= switching_angle) {
    return 2.0 *
           std::sqrt(angle_rad /
                     maximum_acceleration_radps2);
  }
  return 2.0 * maximum_speed_radps /
             maximum_acceleration_radps2 +
         (angle_rad - switching_angle) /
             maximum_speed_radps;
}

}  // namespace lunar::planning::v3

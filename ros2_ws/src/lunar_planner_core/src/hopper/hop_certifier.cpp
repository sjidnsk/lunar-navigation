#include "hopper/hop_certifier.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "hopper/ballistic_envelope.hpp"
#include "hopper/flight_tube_certifier.hpp"
#include "shared/map_snapshot.hpp"

namespace lunar::planning::hopper {
namespace {

constexpr double kTolerance = 1.0e-9;

[[nodiscard]] bool IsFinite(const Vec3 value) noexcept {
  return std::isfinite(value.x) && std::isfinite(value.y) &&
      std::isfinite(value.z);
}

[[nodiscard]] bool IsFinite(const Quaternion value) noexcept {
  return std::isfinite(value.w) && std::isfinite(value.x) &&
      std::isfinite(value.y) && std::isfinite(value.z);
}

[[nodiscard]] double Dot(const Vec3 lhs, const Vec3 rhs) noexcept {
  return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
}

[[nodiscard]] double Norm(const Vec3 value) noexcept {
  return std::hypot(std::hypot(value.x, value.y), value.z);
}

[[nodiscard]] double QuaternionNorm(const Quaternion value) noexcept {
  return std::hypot(
      std::hypot(value.w, value.x), std::hypot(value.y, value.z));
}

[[nodiscard]] double WrapAngle(double angle) noexcept {
  while (angle > std::numbers::pi) {
    angle -= 2.0 * std::numbers::pi;
  }
  while (angle < -std::numbers::pi) {
    angle += 2.0 * std::numbers::pi;
  }
  return angle;
}

[[nodiscard]] double Yaw(const Quaternion value) noexcept {
  return std::atan2(
      2.0 * (value.w * value.z + value.x * value.y),
      1.0 - 2.0 * (value.y * value.y + value.z * value.z));
}

[[nodiscard]] Vec3 BodyUp(const Quaternion value) noexcept {
  return Vec3{
      .x = 2.0 * (value.x * value.z + value.w * value.y),
      .y = 2.0 * (value.y * value.z - value.w * value.x),
      .z = 1.0 - 2.0 * (value.x * value.x + value.y * value.y),
  };
}

[[nodiscard]] std::optional<double> RequiredAttitudeTime(
    const HopperState& state,
    const GoalRegion& goal,
    const CertifiedLandingRegion& target_region,
    const HopperCapability& capability) noexcept {
  if (!IsFinite(state.pose.orientation) ||
      !IsFinite(state.velocity.angular_radps) ||
      !IsFinite(target_region.plane_normal)) {
    return std::nullopt;
  }
  const double quaternion_norm = QuaternionNorm(state.pose.orientation);
  const double initial_angular_speed = Norm(state.velocity.angular_radps);
  if (!std::isfinite(quaternion_norm) ||
      std::abs(quaternion_norm - 1.0) > 1.0e-6 ||
      !std::isfinite(initial_angular_speed) ||
      initial_angular_speed >
          capability.maximum_initial_angular_speed_radps + kTolerance ||
      !std::isfinite(capability.maximum_angular_speed_radps) ||
      capability.maximum_angular_speed_radps <= 0.0 ||
      !std::isfinite(capability.maximum_angular_acceleration_radps2) ||
      capability.maximum_angular_acceleration_radps2 <= 0.0) {
    return std::nullopt;
  }
  const Vec3 up = BodyUp(state.pose.orientation);
  const double up_norm = Norm(up);
  const double normal_norm = Norm(target_region.plane_normal);
  if (up_norm <= kTolerance || normal_norm <= kTolerance) {
    return std::nullopt;
  }
  const double tilt = std::acos(std::clamp(
      Dot(up, target_region.plane_normal) / (up_norm * normal_norm),
      -1.0, 1.0));
  const double current_yaw = Yaw(state.pose.orientation);
  const double target_yaw = goal.yaw_rad.value_or(current_yaw);
  if (!std::isfinite(current_yaw) || !std::isfinite(target_yaw) ||
      !std::isfinite(goal.yaw_tolerance_rad) ||
      goal.yaw_tolerance_rad < 0.0) {
    return std::nullopt;
  }
  const double yaw_rotation = std::max(
      0.0, std::abs(WrapAngle(target_yaw - current_yaw)) -
          goal.yaw_tolerance_rad);
  const double rotation = std::max(tilt, yaw_rotation);
  const double acceleration = capability.maximum_angular_acceleration_radps2;
  const double speed = capability.maximum_angular_speed_radps;
  const double acceleration_angle = speed * speed / acceleration;
  double maneuver_time = 0.0;
  if (rotation <= acceleration_angle) {
    maneuver_time = 2.0 * std::sqrt(rotation / acceleration);
  } else {
    maneuver_time = 2.0 * speed / acceleration +
        (rotation - acceleration_angle) / speed;
  }
  maneuver_time += initial_angular_speed / acceleration;
  maneuver_time += std::chrono::duration<double>(
      capability.minimum_settle_guard).count();
  return std::isfinite(maneuver_time)
      ? std::optional<double>{maneuver_time}
      : std::nullopt;
}

[[nodiscard]] HopCertificationResult Failure(
    const HopCertificationStatus status,
    std::string reason_code,
    const std::size_t examined_intervals = 0U) {
  return HopCertificationResult{
      .status = status,
      .segment = std::nullopt,
      .examined_intervals = examined_intervals,
      .cost = 0.0,
      .reason_code = std::move(reason_code),
  };
}

[[nodiscard]] HopCertificationStatus EnvelopeFailureStatus(
    const BallisticEnvelopeStatus status) noexcept {
  switch (status) {
    case BallisticEnvelopeStatus::kInfeasible:
      return HopCertificationStatus::kInfeasible;
    case BallisticEnvelopeStatus::kCanceled:
      return HopCertificationStatus::kCanceled;
    case BallisticEnvelopeStatus::kInvalid:
      return HopCertificationStatus::kInvalid;
    case BallisticEnvelopeStatus::kNumericalIndeterminate:
      return HopCertificationStatus::kNumericalIndeterminate;
    case BallisticEnvelopeStatus::kSolved:
      return HopCertificationStatus::kNumericalIndeterminate;
  }
  return HopCertificationStatus::kNumericalIndeterminate;
}

[[nodiscard]] HopCertificationStatus FlightTubeFailureStatus(
    const std::string_view reason_code) noexcept {
  if (reason_code == "HOPPER_FLIGHT_TUBE_RESOURCE_EXHAUSTED") {
    return HopCertificationStatus::kResourceExhausted;
  }
  if (reason_code == "HOPPER_FLIGHT_TUBE_INPUT_INVALID") {
    return HopCertificationStatus::kInvalid;
  }
  if (reason_code == "HOPPER_FLIGHT_TUBE_NUMERICAL_INDETERMINATE") {
    return HopCertificationStatus::kNumericalIndeterminate;
  }
  return HopCertificationStatus::kInfeasible;
}

}  // namespace

HopCertificationResult CertifyFirstHop(
    const hierarchical::LocalPlanningProblem& problem,
    const CertifiedLandingRegion& source_region,
    const CertifiedLandingRegion& target_region) {
  const auto* state = std::get_if<HopperState>(&problem.current_state);
  const auto* capability =
      std::get_if<HopperCapability>(&problem.capability);
  const shared::MapSnapshotBuildResult map =
      shared::MapSnapshot::Create(problem.local_map_view);
  if (state == nullptr || capability == nullptr || !map.ok()) {
    return Failure(
        HopCertificationStatus::kInvalid,
        map.ok() ? "HOPPER_CERTIFICATION_INPUT_INVALID" : map.reason_code);
  }
  if (problem.stop_token.stop_requested()) {
    return Failure(HopCertificationStatus::kCanceled, "REQUEST_CANCELED");
  }
  if (problem.config.hopper.maximum_authorized_hops == 0U) {
    return Failure(
        HopCertificationStatus::kResourceExhausted,
        "HOPPER_CERTIFICATION_RESOURCE_EXHAUSTED");
  }
  const auto required_attitude_time = RequiredAttitudeTime(
      *state, problem.goal_odom, target_region, *capability);
  if (!required_attitude_time.has_value()) {
    return Failure(
        HopCertificationStatus::kInfeasible,
        "HOPPER_ATTITUDE_NOT_CERTIFIED");
  }
  const Vec3 launch_position = state->pose.position_m;
  const Vec3 landing_position{
      .x = target_region.aim_position_on_surface_m.x,
      .y = target_region.aim_position_on_surface_m.y,
      .z = target_region.aim_position_on_surface_m.z +
          capability->body_half_extent_m.z,
  };
  const BallisticEnvelopeResult envelope = SolveBallisticEnvelope(
      launch_position, landing_position, state->velocity.linear_mps,
      *capability, *required_attitude_time, problem.stop_token);
  if (!envelope.ok()) {
    return Failure(
        EnvelopeFailureStatus(envelope.status), envelope.reason_code,
        envelope.examined_intervals);
  }
  if (!envelope.arc.has_value()) {
    return Failure(
        HopCertificationStatus::kNumericalIndeterminate,
        "HOPPER_BALLISTIC_NUMERICAL_INDETERMINATE",
        envelope.examined_intervals);
  }

  const BallisticArc& arc = *envelope.arc;
  const FlightTubeCertificationResult tube = CertifyFlightTube(
      arc, *map.snapshot, source_region, target_region, *capability,
      problem.config, problem.stop_token);
  if (tube.canceled) {
    return Failure(
        HopCertificationStatus::kCanceled, "REQUEST_CANCELED",
        envelope.examined_intervals);
  }
  if (!tube.certified) {
    return Failure(
        FlightTubeFailureStatus(tube.reason_code), tube.reason_code,
        envelope.examined_intervals);
  }

  const auto flight_time = std::chrono::nanoseconds{
      static_cast<std::int64_t>(
          std::llround(arc.flight_time_s * 1.0e9))};
  const double settle_s = std::chrono::duration<double>(
      capability->minimum_settle_guard).count();
  return HopCertificationResult{
      .status = HopCertificationStatus::kCertified,
      .segment = HopSegment{
          .segment_id = problem.request_id + "-hop-0",
          .launch_pose = state->pose,
          .landing_region_boundary_m = target_region.boundary_m,
          .flight_time = flight_time,
          .launch_velocity_mps = arc.launch_velocity_mps,
          .flight_tube_radius_m = tube.radius_m,
      },
      .examined_intervals = envelope.examined_intervals,
      .cost = arc.flight_time_s + 2.0 * settle_s,
      .reason_code = {},
  };
}

}  // namespace lunar::planning::hopper

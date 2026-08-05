#include "hopper/hop_certifier.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <optional>
#include <ranges>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "hopper/ballistic_kinematics.hpp"
#include "hopper/flight_tube_certifier.hpp"

namespace lunar::planning::hopper {
namespace {

constexpr double kTolerance = 1.0e-9;

struct CertifiedCandidate final {
  BallisticArc arc;
  FlightTubeCertificationResult tube;
  double physical_score{};
};

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

[[nodiscard]] std::vector<double> CandidateFlightTimes(
    const Vec3 displacement,
    const Vec3 gravity,
    const double minimum_s,
    const double maximum_s,
    const std::size_t attempt_limit) {
  std::vector<double> times;
  if (attempt_limit == 0U) {
    return times;
  }
  const double gravity_norm = Norm(gravity);
  const double displacement_norm = Norm(displacement);
  if (gravity_norm > kTolerance && std::isfinite(displacement_norm)) {
    times.push_back(std::clamp(
        std::sqrt(2.0 * displacement_norm / gravity_norm),
        minimum_s, maximum_s));
  }
  times.push_back(minimum_s);
  times.push_back(maximum_s);
  const std::size_t sample_count = std::max<std::size_t>(2U, attempt_limit);
  for (std::size_t index = 0U; index < sample_count; ++index) {
    const double ratio = sample_count == 1U
        ? 0.0
        : static_cast<double>(index) /
            static_cast<double>(sample_count - 1U);
    times.push_back(minimum_s + ratio * (maximum_s - minimum_s));
  }
  std::vector<double> unique;
  unique.reserve(times.size());
  for (const double time : times) {
    if (std::ranges::none_of(unique, [&](const double existing) {
          return std::abs(existing - time) <= 1.0e-12;
        })) {
      unique.push_back(time);
    }
  }
  return unique;
}

[[nodiscard]] bool Better(
    const CertifiedCandidate& candidate,
    const CertifiedCandidate& selected) noexcept {
  return std::tuple{
             candidate.physical_score,
             candidate.arc.flight_time_s,
             candidate.arc.launch_velocity_mps.x,
             candidate.arc.launch_velocity_mps.y,
             candidate.arc.launch_velocity_mps.z} <
      std::tuple{
             selected.physical_score,
             selected.arc.flight_time_s,
             selected.arc.launch_velocity_mps.x,
             selected.arc.launch_velocity_mps.y,
             selected.arc.launch_velocity_mps.z};
}

}  // namespace

HopCertificationResult CertifyFirstHop(
    const PlannerInput& input,
    const HopperState& state,
    const HopperCapability& capability,
    const shared::MapSnapshot& map,
    const CertifiedLandingRegion& source_region,
    const CertifiedLandingRegion& target_region) {
  if (input.stop_token.stop_requested()) {
    return HopCertificationResult{
        .segment = std::nullopt,
        .canceled = true,
        .resource_exhausted = false,
        .attempted_candidates = 0U,
        .cost = 0.0,
        .reason_code = "REQUEST_CANCELED",
    };
  }
  if (input.config.hopper.maximum_certification_attempts == 0U ||
      input.config.hopper.maximum_authorized_hops == 0U) {
    return HopCertificationResult{
        .segment = std::nullopt,
        .canceled = false,
        .resource_exhausted = true,
        .attempted_candidates = 0U,
        .cost = 0.0,
        .reason_code = "HOPPER_CERTIFICATION_RESOURCE_EXHAUSTED",
    };
  }
  const auto required_attitude_time = RequiredAttitudeTime(
      state, input.goal_map, target_region, capability);
  if (!required_attitude_time.has_value()) {
    return HopCertificationResult{
        .segment = std::nullopt,
        .canceled = false,
        .resource_exhausted = false,
        .attempted_candidates = 0U,
        .cost = 0.0,
        .reason_code = "HOPPER_ATTITUDE_NOT_CERTIFIED",
    };
  }
  const Vec3 launch_position = state.pose.position_m;
  const Vec3 landing_position{
      .x = target_region.aim_position_on_surface_m.x,
      .y = target_region.aim_position_on_surface_m.y,
      .z = target_region.aim_position_on_surface_m.z +
          capability.body_half_extent_m.z,
  };
  const Vec3 displacement{
      .x = landing_position.x - launch_position.x,
      .y = landing_position.y - launch_position.y,
      .z = landing_position.z - launch_position.z,
  };
  const double minimum_s = std::chrono::duration<double>(
      capability.minimum_flight_time).count();
  const double maximum_s = std::chrono::duration<double>(
      capability.maximum_flight_time).count();
  if (!IsFinite(launch_position) || !IsFinite(landing_position) ||
      !IsFinite(state.velocity.linear_mps) ||
      !std::isfinite(minimum_s) || !std::isfinite(maximum_s) ||
      minimum_s <= 0.0 || maximum_s < minimum_s) {
    return HopCertificationResult{
        .segment = std::nullopt,
        .canceled = false,
        .resource_exhausted = false,
        .attempted_candidates = 0U,
        .cost = 0.0,
        .reason_code = "HOPPER_CERTIFICATION_INPUT_INVALID",
    };
  }

  const std::vector<double> times = CandidateFlightTimes(
      displacement, capability.gravity_mps2, minimum_s, maximum_s,
      input.config.hopper.maximum_certification_attempts);
  const double gravity_norm = Norm(capability.gravity_mps2);
  std::optional<CertifiedCandidate> selected;
  std::size_t attempts = 0U;
  std::string last_rejection = "HOPPER_NO_BALLISTIC_CANDIDATE";
  for (const double time_s : times) {
    if (attempts >= input.config.hopper.maximum_certification_attempts) {
      break;
    }
    if (input.stop_token.stop_requested()) {
      return HopCertificationResult{
          .segment = std::nullopt,
          .canceled = true,
          .resource_exhausted = false,
          .attempted_candidates = attempts,
          .cost = 0.0,
          .reason_code = "REQUEST_CANCELED",
      };
    }
    ++attempts;
    if (time_s + kTolerance < *required_attitude_time) {
      last_rejection = "HOPPER_ATTITUDE_TIME_LIMIT";
      continue;
    }
    const BallisticSolveResult solved = SolveBallisticArc(
        launch_position, landing_position, capability.gravity_mps2, time_s);
    if (!solved.ok()) {
      last_rejection = solved.reason_code;
      continue;
    }
    const BallisticArc& arc = *solved.arc;
    const double launch_speed = Norm(arc.launch_velocity_mps);
    const double landing_speed = Norm(arc.landing_velocity_mps);
    const Vec3 velocity_change{
        .x = arc.launch_velocity_mps.x - state.velocity.linear_mps.x,
        .y = arc.launch_velocity_mps.y - state.velocity.linear_mps.y,
        .z = arc.launch_velocity_mps.z - state.velocity.linear_mps.z,
    };
    const double impulse = capability.platform_mass_kg * Norm(velocity_change);
    const double downward_speed = gravity_norm > kTolerance
        ? Dot(arc.landing_velocity_mps, capability.gravity_mps2) / gravity_norm
        : -std::numeric_limits<double>::infinity();
    if (!std::isfinite(launch_speed) ||
        launch_speed > capability.maximum_launch_speed_mps + kTolerance) {
      last_rejection = "HOPPER_LAUNCH_SPEED_LIMIT";
      continue;
    }
    if (!std::isfinite(impulse) ||
        impulse >
            capability.maximum_launch_impulse_newton_seconds + kTolerance) {
      last_rejection = "HOPPER_LAUNCH_IMPULSE_LIMIT";
      continue;
    }
    if (!std::isfinite(landing_speed) ||
        landing_speed > capability.maximum_landing_speed_mps + kTolerance) {
      last_rejection = "HOPPER_LANDING_SPEED_LIMIT";
      continue;
    }
    if (!std::isfinite(downward_speed) ||
        downward_speed + kTolerance <
            capability.minimum_downward_impact_speed_mps) {
      last_rejection = "HOPPER_DOWNWARD_IMPACT_SPEED_LIMIT";
      continue;
    }
    const FlightTubeCertificationResult tube = CertifyFlightTube(
        arc, map, source_region, target_region, capability,
        input.config, input.stop_token);
    if (tube.canceled) {
      return HopCertificationResult{
          .segment = std::nullopt,
          .canceled = true,
          .resource_exhausted = false,
          .attempted_candidates = attempts,
          .cost = 0.0,
          .reason_code = "REQUEST_CANCELED",
      };
    }
    if (!tube.certified) {
      last_rejection = tube.reason_code;
      continue;
    }
    CertifiedCandidate candidate{
        .arc = arc,
        .tube = tube,
        .physical_score = launch_speed + landing_speed,
    };
    if (!selected.has_value() || Better(candidate, *selected)) {
      selected = std::move(candidate);
    }
  }
  if (!selected.has_value()) {
    return HopCertificationResult{
        .segment = std::nullopt,
        .canceled = false,
        .resource_exhausted = false,
        .attempted_candidates = attempts,
        .cost = 0.0,
        .reason_code = std::move(last_rejection),
    };
  }

  const auto flight_time = std::chrono::nanoseconds{
      static_cast<std::int64_t>(
          std::llround(selected->arc.flight_time_s * 1.0e9))};
  const double settle_s = std::chrono::duration<double>(
      capability.minimum_settle_guard).count();
  return HopCertificationResult{
      .segment = HopSegment{
          .segment_id = input.request_id + "-hop-0",
          .launch_pose = state.pose,
          .landing_region_boundary_m = target_region.boundary_m,
          .flight_time = flight_time,
          .launch_velocity_mps = selected->arc.launch_velocity_mps,
          .flight_tube_radius_m = selected->tube.radius_m,
      },
      .canceled = false,
      .resource_exhausted = false,
      .attempted_candidates = attempts,
      .cost = selected->arc.flight_time_s + 2.0 * settle_s,
      .reason_code = {},
  };
}

}  // namespace lunar::planning::hopper

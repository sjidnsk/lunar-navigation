#include "wheel/wheel_timing.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stop_token>
#include <string>
#include <utility>
#include <vector>

namespace lunar::planning::wheel {
namespace {

constexpr std::size_t kPreferredSamplesPerTransition = 8U;
constexpr double kComparisonTolerance = 1.0e-12;
constexpr double kWheelRoughnessReferenceM = 0.1595;

struct SpeedProfile final {
  double distance{};
  double initial_speed{};
  double peak_speed{};
  double final_speed{};
  double acceleration{};
  double deceleration{};
  double acceleration_time{};
  double cruise_time{};
  double deceleration_time{};

  [[nodiscard]] double duration() const noexcept {
    return acceleration_time + cruise_time + deceleration_time;
  }
};

struct ProfileSample final {
  double distance{};
  double speed{};
};

[[nodiscard]] WheelTimingResult Failure(
    std::string reason_code, const bool canceled = false) {
  return WheelTimingResult{
      .trajectory = std::nullopt,
      .canceled = canceled,
      .reason_code = std::move(reason_code),
  };
}

[[nodiscard]] bool Finite(const Vec3& value) noexcept {
  return std::isfinite(value.x) && std::isfinite(value.y) &&
      std::isfinite(value.z);
}

[[nodiscard]] bool PositiveFinite(const double value) noexcept {
  return std::isfinite(value) && value > 0.0;
}

[[nodiscard]] bool ValidateCapability(
    const WheeledCapability& capability) noexcept {
  return PositiveFinite(capability.wheel_diameter_m) &&
      PositiveFinite(capability.maximum_forward_speed_mps) &&
      PositiveFinite(capability.maximum_reverse_speed_mps) &&
      PositiveFinite(capability.maximum_spin_rate_radps) &&
      PositiveFinite(capability.maximum_acceleration_mps2) &&
      PositiveFinite(capability.maximum_braking_deceleration_mps2) &&
      PositiveFinite(capability.maximum_yaw_acceleration_radps2) &&
      PositiveFinite(capability.maximum_lateral_acceleration_mps2) &&
      PositiveFinite(capability.maximum_curvature_per_m);
}

[[nodiscard]] bool IsStop(const WheelTransition& transition) noexcept {
  return transition.primitive_kind == WheelPrimitiveKind::kStopAndSwitch;
}

[[nodiscard]] bool IsTranslation(
    const WheelTransition& transition) noexcept {
  return transition.path_length_m > kComparisonTolerance;
}

[[nodiscard]] std::optional<double> TranslationSpeedLimit(
    const WheelTransition& transition,
    const WheeledCapability& capability) noexcept {
  if (!std::isfinite(transition.path_length_m) ||
      transition.path_length_m < 0.0 ||
      !std::isfinite(transition.curvature_per_m) ||
      !std::isfinite(transition.surface_slope_rad) ||
      !std::isfinite(transition.roughness_m) ||
      transition.roughness_m < 0.0) {
    return std::nullopt;
  }
  double limit = transition.reverse
      ? capability.maximum_reverse_speed_mps
      : capability.maximum_forward_speed_mps;
  const double curvature = std::abs(transition.curvature_per_m);
  if (curvature > kComparisonTolerance) {
    limit = std::min(
        {limit,
         capability.maximum_spin_rate_radps / curvature,
         std::sqrt(
             capability.maximum_lateral_acceleration_mps2 / curvature)});
  }
  const double roughness_ratio =
      transition.roughness_m / kWheelRoughnessReferenceM;
  const double roughness_scale =
      1.0 / (1.0 + roughness_ratio * roughness_ratio);
  limit *= std::max(0.0, std::cos(transition.surface_slope_rad)) *
      roughness_scale;
  return PositiveFinite(limit) ? std::optional<double>{limit} : std::nullopt;
}

[[nodiscard]] std::optional<SpeedProfile> MakeProfile(
    const double distance, const double initial_speed,
    const double final_speed, const double speed_limit,
    const double acceleration, const double deceleration) noexcept {
  if (!PositiveFinite(distance) || !PositiveFinite(speed_limit) ||
      !PositiveFinite(acceleration) || !PositiveFinite(deceleration) ||
      !std::isfinite(initial_speed) || !std::isfinite(final_speed) ||
      initial_speed < 0.0 || final_speed < 0.0 ||
      initial_speed > speed_limit + kComparisonTolerance ||
      final_speed > speed_limit + kComparisonTolerance) {
    return std::nullopt;
  }
  const double unconstrained_peak_squared =
      (2.0 * acceleration * deceleration * distance +
       deceleration * initial_speed * initial_speed +
       acceleration * final_speed * final_speed) /
      (acceleration + deceleration);
  if (!std::isfinite(unconstrained_peak_squared) ||
      unconstrained_peak_squared < 0.0) {
    return std::nullopt;
  }
  const double peak = std::min(
      speed_limit, std::sqrt(unconstrained_peak_squared));
  if (peak + kComparisonTolerance < initial_speed ||
      peak + kComparisonTolerance < final_speed) {
    return std::nullopt;
  }
  const double acceleration_distance = std::max(
      0.0, (peak * peak - initial_speed * initial_speed) /
          (2.0 * acceleration));
  const double deceleration_distance = std::max(
      0.0, (peak * peak - final_speed * final_speed) /
          (2.0 * deceleration));
  const double cruise_distance = std::max(
      0.0, distance - acceleration_distance - deceleration_distance);
  SpeedProfile profile{
      .distance = distance,
      .initial_speed = initial_speed,
      .peak_speed = peak,
      .final_speed = final_speed,
      .acceleration = acceleration,
      .deceleration = deceleration,
      .acceleration_time = (peak - initial_speed) / acceleration,
      .cruise_time = cruise_distance /
          std::max(peak, kComparisonTolerance),
      .deceleration_time = (peak - final_speed) / deceleration,
  };
  return PositiveFinite(profile.duration())
      ? std::optional<SpeedProfile>{profile}
      : std::nullopt;
}

[[nodiscard]] ProfileSample SampleProfile(
    const SpeedProfile& profile, const double time) noexcept {
  const double clamped_time = std::clamp(time, 0.0, profile.duration());
  if (clamped_time <= profile.acceleration_time) {
    const double speed = profile.initial_speed +
        profile.acceleration * clamped_time;
    return ProfileSample{
        .distance = profile.initial_speed * clamped_time +
            0.5 * profile.acceleration * clamped_time * clamped_time,
        .speed = speed,
    };
  }
  const double acceleration_distance =
      profile.initial_speed * profile.acceleration_time +
      0.5 * profile.acceleration * profile.acceleration_time *
          profile.acceleration_time;
  if (clamped_time <=
      profile.acceleration_time + profile.cruise_time) {
    const double cruise_elapsed = clamped_time - profile.acceleration_time;
    return ProfileSample{
        .distance = acceleration_distance +
            profile.peak_speed * cruise_elapsed,
        .speed = profile.peak_speed,
    };
  }
  const double deceleration_elapsed = clamped_time -
      profile.acceleration_time - profile.cruise_time;
  const double cruise_distance =
      profile.peak_speed * profile.cruise_time;
  return ProfileSample{
      .distance = std::min(
          profile.distance,
          acceleration_distance + cruise_distance +
              profile.peak_speed * deceleration_elapsed -
              0.5 * profile.deceleration * deceleration_elapsed *
                  deceleration_elapsed),
      .speed = std::max(
          profile.final_speed,
          profile.peak_speed -
              profile.deceleration * deceleration_elapsed),
  };
}

[[nodiscard]] Pose3 PoseAt(
    const WheelTransition& transition, const double progress) noexcept {
  const double yaw_delta = ShortestYawDelta(
      transition.source_pose.yaw_rad, transition.target_pose.yaw_rad);
  return Pose3{
      .position_m = Vec3{
          .x = transition.source_pose.position_m.x +
              progress * (transition.target_pose.position_m.x -
                          transition.source_pose.position_m.x),
          .y = transition.source_pose.position_m.y +
              progress * (transition.target_pose.position_m.y -
                          transition.source_pose.position_m.y),
          .z = transition.source_pose.position_m.z +
              progress * (transition.target_pose.position_m.z -
                          transition.source_pose.position_m.z),
      },
      .orientation = QuaternionFromYaw(
          transition.source_pose.yaw_rad + progress * yaw_delta),
  };
}

[[nodiscard]] Twist3 TranslationVelocity(
    const WheelTransition& transition, const double speed) noexcept {
  if (transition.path_length_m <= kComparisonTolerance) {
    return {};
  }
  const double dx = transition.target_pose.position_m.x -
      transition.source_pose.position_m.x;
  const double dy = transition.target_pose.position_m.y -
      transition.source_pose.position_m.y;
  const double dz = transition.target_pose.position_m.z -
      transition.source_pose.position_m.z;
  const double yaw_delta = ShortestYawDelta(
      transition.source_pose.yaw_rad, transition.target_pose.yaw_rad);
  return Twist3{
      .linear_mps = Vec3{
          .x = speed * dx / transition.path_length_m,
          .y = speed * dy / transition.path_length_m,
          .z = speed * dz / transition.path_length_m,
      },
      .angular_radps = Vec3{
          .z = speed * yaw_delta / transition.path_length_m,
      },
  };
}

}  // namespace

WheelTimingResult ParameterizeWheelTiming(
    const std::vector<WheelTransition>& transitions,
    const WheeledCapability& capability,
    const Twist3& initial_velocity,
    const std::size_t maximum_samples,
    const std::stop_token stop_token) {
  if (stop_token.stop_requested()) {
    return Failure("REQUEST_CANCELED", true);
  }
  if (transitions.empty()) {
    return Failure("WHEEL_TIMING_PATH_EMPTY");
  }
  if (!ValidateCapability(capability) ||
      !Finite(initial_velocity.linear_mps) ||
      !Finite(initial_velocity.angular_radps)) {
    return Failure("WHEEL_TIMING_CAPABILITY_INVALID");
  }
  if (maximum_samples <= 1U ||
      transitions.size() > maximum_samples - 1U) {
    return Failure("WHEEL_TIMING_SAMPLE_LIMIT");
  }
  const std::size_t samples_per_transition = std::min(
      kPreferredSamplesPerTransition,
      (maximum_samples - 1U) / transitions.size());
  if (samples_per_transition == 0U) {
    return Failure("WHEEL_TIMING_SAMPLE_LIMIT");
  }

  std::vector<double> speed_limits(transitions.size(), 0.0);
  std::vector<double> boundary_speeds(transitions.size() + 1U, 0.0);
  for (std::size_t index = 0U; index < transitions.size(); ++index) {
    if (IsTranslation(transitions[index])) {
      const auto limit = TranslationSpeedLimit(transitions[index], capability);
      if (!limit.has_value()) {
        return Failure("WHEEL_TIMING_DURATION_INVALID");
      }
      speed_limits[index] = *limit;
    }
  }
  if (IsTranslation(transitions.front())) {
    const Vec3 delta{
        .x = transitions.front().target_pose.position_m.x -
            transitions.front().source_pose.position_m.x,
        .y = transitions.front().target_pose.position_m.y -
            transitions.front().source_pose.position_m.y,
        .z = transitions.front().target_pose.position_m.z -
            transitions.front().source_pose.position_m.z,
    };
    boundary_speeds.front() = std::clamp(
        (initial_velocity.linear_mps.x * delta.x +
         initial_velocity.linear_mps.y * delta.y +
         initial_velocity.linear_mps.z * delta.z) /
            transitions.front().path_length_m,
        0.0, speed_limits.front());
  }
  for (std::size_t boundary = 1U;
       boundary < transitions.size(); ++boundary) {
    if (IsStop(transitions[boundary - 1U]) ||
        IsStop(transitions[boundary]) ||
        !IsTranslation(transitions[boundary - 1U]) ||
        !IsTranslation(transitions[boundary]) ||
        transitions[boundary - 1U].reverse !=
            transitions[boundary].reverse) {
      boundary_speeds[boundary] = 0.0;
    } else {
      boundary_speeds[boundary] = std::min(
          speed_limits[boundary - 1U], speed_limits[boundary]);
    }
  }
  boundary_speeds.back() = 0.0;
  for (std::size_t index = 0U; index < transitions.size(); ++index) {
    if (!IsTranslation(transitions[index])) {
      boundary_speeds[index] = 0.0;
      boundary_speeds[index + 1U] = 0.0;
      continue;
    }
    boundary_speeds[index + 1U] = std::min(
        boundary_speeds[index + 1U],
        std::sqrt(
            boundary_speeds[index] * boundary_speeds[index] +
            2.0 * capability.maximum_acceleration_mps2 *
                transitions[index].path_length_m));
  }
  for (std::size_t reverse_index = transitions.size();
       reverse_index > 0U; --reverse_index) {
    const std::size_t index = reverse_index - 1U;
    if (!IsTranslation(transitions[index])) {
      continue;
    }
    boundary_speeds[index] = std::min(
        boundary_speeds[index],
        std::sqrt(
            boundary_speeds[index + 1U] *
                boundary_speeds[index + 1U] +
            2.0 * capability.maximum_braking_deceleration_mps2 *
                transitions[index].path_length_m));
  }

  TrajectoryReference trajectory{
      .semantics = TrajectorySemantics::kWheeledBase,
      .points = {},
  };
  trajectory.points.reserve(
      1U + transitions.size() * samples_per_transition);
  trajectory.points.push_back(TrajectoryPoint{
      .time_from_start = std::chrono::nanoseconds{0},
      .pose = PoseAt(transitions.front(), 0.0),
      .velocity = initial_velocity,
  });
  std::chrono::nanoseconds elapsed{0};
  for (std::size_t index = 0U; index < transitions.size(); ++index) {
    if (stop_token.stop_requested()) {
      return Failure("REQUEST_CANCELED", true);
    }
    const WheelTransition& transition = transitions[index];
    const double yaw_distance = std::abs(ShortestYawDelta(
        transition.source_pose.yaw_rad, transition.target_pose.yaw_rad));
    std::optional<SpeedProfile> profile;
    bool angular_profile = false;
    if (IsTranslation(transition)) {
      profile = MakeProfile(
          transition.path_length_m, boundary_speeds[index],
          boundary_speeds[index + 1U], speed_limits[index],
          capability.maximum_acceleration_mps2,
          capability.maximum_braking_deceleration_mps2);
    } else if (yaw_distance > kComparisonTolerance) {
      profile = MakeProfile(
          yaw_distance, 0.0, 0.0,
          capability.maximum_spin_rate_radps,
          capability.maximum_yaw_acceleration_radps2,
          capability.maximum_yaw_acceleration_radps2);
      angular_profile = true;
    }
    double duration_s = 1.0e-3;
    if (profile.has_value()) {
      duration_s = profile->duration();
    } else if (!IsStop(transition)) {
      return Failure("WHEEL_TIMING_DURATION_INVALID");
    }
    if (!PositiveFinite(duration_s) ||
        duration_s >
            static_cast<double>(std::numeric_limits<std::int64_t>::max()) /
                1.0e9) {
      return Failure("WHEEL_TIMING_DURATION_INVALID");
    }
    const auto duration = std::chrono::nanoseconds{
        std::max<std::int64_t>(
            1, static_cast<std::int64_t>(std::ceil(duration_s * 1.0e9)))};
    if (elapsed.count() >
        std::numeric_limits<std::int64_t>::max() - duration.count()) {
      return Failure("WHEEL_TIMING_DURATION_INVALID");
    }
    for (std::size_t sample = 1U;
         sample <= samples_per_transition; ++sample) {
      const double ratio = static_cast<double>(sample) /
          static_cast<double>(samples_per_transition);
      double progress = ratio;
      Twist3 velocity{};
      if (profile.has_value()) {
        const ProfileSample profile_sample =
            SampleProfile(*profile, ratio * profile->duration());
        progress = std::clamp(
            profile_sample.distance / profile->distance, 0.0, 1.0);
        if (angular_profile) {
          const double direction = ShortestYawDelta(
              transition.source_pose.yaw_rad,
              transition.target_pose.yaw_rad) >= 0.0 ? 1.0 : -1.0;
          velocity.angular_radps.z = direction * profile_sample.speed;
        } else {
          velocity = TranslationVelocity(transition, profile_sample.speed);
        }
      }
      const std::int64_t sample_count =
          static_cast<std::int64_t>(samples_per_transition);
      const std::int64_t sample_index = static_cast<std::int64_t>(sample);
      const auto sample_offset = std::chrono::nanoseconds{
          (duration.count() / sample_count) * sample_index +
          ((duration.count() % sample_count) * sample_index) / sample_count};
      trajectory.points.push_back(TrajectoryPoint{
          .time_from_start = elapsed + sample_offset,
          .pose = PoseAt(transition, progress),
          .velocity = velocity,
      });
    }
    elapsed += duration;
  }
  return WheelTimingResult{
      .trajectory = std::move(trajectory),
      .canceled = false,
      .reason_code = {},
  };
}

}  // namespace lunar::planning::wheel

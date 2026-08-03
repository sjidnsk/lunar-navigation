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

constexpr std::size_t kSamplesPerTransition = 8U;
constexpr double kComparisonTolerance = 1.0e-12;

[[nodiscard]] WheelTimingResult Failure(
    std::string reason_code, const bool canceled = false) {
  return WheelTimingResult{
      .trajectory = std::nullopt,
      .canceled = canceled,
      .reason_code = std::move(reason_code),
  };
}

[[nodiscard]] bool PositiveFinite(const double value) noexcept {
  return std::isfinite(value) && value > 0.0;
}

[[nodiscard]] bool ValidateCapability(
    const WheeledCapability& capability) noexcept {
  return PositiveFinite(capability.maximum_forward_speed_mps) &&
      PositiveFinite(capability.maximum_reverse_speed_mps) &&
      PositiveFinite(capability.maximum_spin_rate_radps) &&
      PositiveFinite(capability.maximum_acceleration_mps2) &&
      PositiveFinite(capability.maximum_braking_deceleration_mps2) &&
      PositiveFinite(capability.maximum_yaw_acceleration_radps2) &&
      PositiveFinite(capability.maximum_lateral_acceleration_mps2) &&
      PositiveFinite(capability.maximum_curvature_per_m);
}

[[nodiscard]] std::optional<std::chrono::nanoseconds> SegmentDuration(
    const WheelTransition& transition,
    const WheeledCapability& capability) noexcept {
  if (transition.nominal_duration.count() <= 0 ||
      !std::isfinite(transition.path_length_m) ||
      transition.path_length_m < 0.0 ||
      !std::isfinite(transition.curvature_per_m)) {
    return std::nullopt;
  }
  const double yaw_distance = std::abs(ShortestYawDelta(
      transition.source_pose.yaw_rad, transition.target_pose.yaw_rad));
  double speed_limit = transition.reverse
      ? capability.maximum_reverse_speed_mps
      : capability.maximum_forward_speed_mps;
  if (std::abs(transition.curvature_per_m) > kComparisonTolerance) {
    speed_limit = std::min(
        speed_limit,
        std::sqrt(capability.maximum_lateral_acceleration_mps2 /
                  std::abs(transition.curvature_per_m)));
  }
  if (!PositiveFinite(speed_limit)) {
    return std::nullopt;
  }
  const double linear_acceleration_limit = std::min(
      capability.maximum_acceleration_mps2,
      capability.maximum_braking_deceleration_mps2);
  double duration_s = std::chrono::duration<double>(
      transition.nominal_duration).count();
  duration_s = std::max(
      duration_s, 1.5 * transition.path_length_m / speed_limit);
  duration_s = std::max(
      duration_s,
      std::sqrt(6.0 * transition.path_length_m /
                linear_acceleration_limit));
  duration_s = std::max(
      duration_s, 1.5 * yaw_distance /
          capability.maximum_spin_rate_radps);
  duration_s = std::max(
      duration_s,
      std::sqrt(6.0 * yaw_distance /
                capability.maximum_yaw_acceleration_radps2));
  if (!PositiveFinite(duration_s) ||
      duration_s >
          static_cast<double>(std::numeric_limits<std::int64_t>::max()) /
              1.0e9) {
    return std::nullopt;
  }
  const auto nanoseconds = static_cast<std::int64_t>(
      std::ceil(duration_s * 1.0e9));
  return nanoseconds > 0
      ? std::optional<std::chrono::nanoseconds>{
            std::chrono::nanoseconds{
                std::max<std::int64_t>(
                    nanoseconds,
                    static_cast<std::int64_t>(kSamplesPerTransition))}}
      : std::nullopt;
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

}  // namespace

WheelTimingResult ParameterizeWheelTiming(
    const std::vector<WheelTransition>& transitions,
    const WheeledCapability& capability,
    const std::stop_token stop_token) {
  if (stop_token.stop_requested()) {
    return Failure("REQUEST_CANCELED", true);
  }
  if (transitions.empty()) {
    return Failure("WHEEL_TIMING_PATH_EMPTY");
  }
  if (!ValidateCapability(capability)) {
    return Failure("WHEEL_TIMING_CAPABILITY_INVALID");
  }

  TrajectoryReference trajectory{
      .semantics = TrajectorySemantics::kWheeledBase,
      .points = {},
  };
  trajectory.points.reserve(
      1U + transitions.size() * kSamplesPerTransition);
  trajectory.points.push_back(TrajectoryPoint{
      .time_from_start = std::chrono::nanoseconds{0},
      .pose = PoseAt(transitions.front(), 0.0),
      .velocity = {},
  });
  std::chrono::nanoseconds elapsed{0};
  for (const WheelTransition& transition : transitions) {
    if (stop_token.stop_requested()) {
      return Failure("REQUEST_CANCELED", true);
    }
    const auto duration = SegmentDuration(transition, capability);
    if (!duration.has_value() ||
        elapsed.count() >
            std::numeric_limits<std::int64_t>::max() - duration->count()) {
      return Failure("WHEEL_TIMING_DURATION_INVALID");
    }
    const double duration_s =
        std::chrono::duration<double>(*duration).count();
    const double yaw_delta = ShortestYawDelta(
        transition.source_pose.yaw_rad, transition.target_pose.yaw_rad);
    for (std::size_t sample = 1U;
         sample <= kSamplesPerTransition; ++sample) {
      if (stop_token.stop_requested()) {
        return Failure("REQUEST_CANCELED", true);
      }
      const double ratio = static_cast<double>(sample) /
          static_cast<double>(kSamplesPerTransition);
      const double progress = ratio * ratio * (3.0 - 2.0 * ratio);
      const double progress_rate =
          6.0 * ratio * (1.0 - ratio) / duration_s;
      const std::int64_t samples =
          static_cast<std::int64_t>(kSamplesPerTransition);
      const std::int64_t sample_index =
          static_cast<std::int64_t>(sample);
      const auto sample_offset = std::chrono::nanoseconds{
          (duration->count() / samples) * sample_index +
          ((duration->count() % samples) * sample_index) / samples};
      trajectory.points.push_back(TrajectoryPoint{
          .time_from_start = elapsed + sample_offset,
          .pose = PoseAt(transition, progress),
          .velocity = Twist3{
              .linear_mps = Vec3{
                  .x = progress_rate *
                      (transition.target_pose.position_m.x -
                       transition.source_pose.position_m.x),
                  .y = progress_rate *
                      (transition.target_pose.position_m.y -
                       transition.source_pose.position_m.y),
                  .z = progress_rate *
                      (transition.target_pose.position_m.z -
                       transition.source_pose.position_m.z),
              },
              .angular_radps = Vec3{.z = progress_rate * yaw_delta},
          },
      });
    }
    elapsed += *duration;
  }
  return WheelTimingResult{
      .trajectory = std::move(trajectory),
      .canceled = false,
      .reason_code = {},
  };
}

}  // namespace lunar::planning::wheel

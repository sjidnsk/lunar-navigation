#include "legged/legged_timing.hpp"

#include <algorithm>
#include <array>
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

namespace lunar::pure_planning::legged {
namespace {

constexpr std::size_t kPreferredSamplesPerTransition = 8U;
constexpr double kTolerance = 1.0e-12;

[[nodiscard]] LeggedTimingResult Failure(
    std::string reason_code, const bool canceled = false) {
  return LeggedTimingResult{
      .trajectory = std::nullopt,
      .canceled = canceled,
      .reason_code = std::move(reason_code),
  };
}

[[nodiscard]] bool ValidVelocityInterval(
    const Interval& interval) noexcept {
  return ValidInterval(interval) && interval.lower <= 0.0 &&
      interval.upper >= 0.0 && interval.lower < interval.upper;
}

[[nodiscard]] double DirectionalLimit(
    const double displacement, const Interval& interval) noexcept {
  if (displacement > kTolerance) {
    return interval.upper;
  }
  if (displacement < -kTolerance) {
    return -interval.lower;
  }
  return std::numeric_limits<double>::infinity();
}

[[nodiscard]] std::optional<std::chrono::nanoseconds> DurationFor(
    const LeggedTransition& transition,
    const LeggedCapability& capability) noexcept {
  if (!std::isfinite(transition.path_length_m) ||
      transition.path_length_m < 0.0) {
    return std::nullopt;
  }
  const double cosine = std::cos(transition.source_pose.yaw_rad);
  const double sine = std::sin(transition.source_pose.yaw_rad);
  const double dx = transition.target_pose.position_m.x -
      transition.source_pose.position_m.x;
  const double dy = transition.target_pose.position_m.y -
      transition.source_pose.position_m.y;
  const double dz = transition.target_pose.position_m.z -
      transition.source_pose.position_m.z;
  const double forward = cosine * dx + sine * dy;
  const double lateral = -sine * dx + cosine * dy;
  const double yaw = ShortestYawDelta(
      transition.source_pose.yaw_rad,
      transition.target_pose.yaw_rad);
  double seconds = std::abs(dz) / capability.step_vertical_rate_mps;
  const std::array<std::pair<double, Interval>, 3> components{
      std::pair{forward, capability.forward_speed_mps},
      std::pair{lateral, capability.lateral_speed_mps},
      std::pair{yaw, capability.yaw_rate_radps},
  };
  for (const auto& [displacement, interval] : components) {
    const double limit = DirectionalLimit(displacement, interval);
    if (!std::isfinite(limit)) {
      continue;
    }
    if (limit <= 0.0) {
      return std::nullopt;
    }
    seconds = std::max(seconds, 1.5 * std::abs(displacement) / limit);
  }
  seconds = std::max(
      seconds,
      std::sqrt(6.0 * transition.path_length_m /
                capability.maximum_linear_acceleration_mps2));
  seconds = std::max(
      seconds,
      std::sqrt(6.0 * std::abs(yaw) /
                capability.maximum_yaw_acceleration_radps2));
  if (!std::isfinite(seconds) || seconds <= 0.0 ||
      seconds >
          static_cast<double>(std::numeric_limits<std::int64_t>::max()) /
              1.0e9) {
    return std::nullopt;
  }
  const auto count = std::max<std::int64_t>(
      static_cast<std::int64_t>(std::ceil(seconds * 1.0e9)),
      static_cast<std::int64_t>(kPreferredSamplesPerTransition));
  return std::chrono::nanoseconds{count};
}

[[nodiscard]] Pose3 PoseAt(
    const LeggedTransition& transition, const double progress) noexcept {
  const double yaw_delta = ShortestYawDelta(
      transition.source_pose.yaw_rad,
      transition.target_pose.yaw_rad);
  return Pose3{
      .position_m = Vec3{
          .x = transition.source_pose.position_m.x + progress *
              (transition.target_pose.position_m.x -
               transition.source_pose.position_m.x),
          .y = transition.source_pose.position_m.y + progress *
              (transition.target_pose.position_m.y -
               transition.source_pose.position_m.y),
          .z = transition.source_pose.position_m.z + progress *
              (transition.target_pose.position_m.z -
               transition.source_pose.position_m.z),
      },
      .orientation = QuaternionFromYaw(
          transition.source_pose.yaw_rad + progress * yaw_delta),
  };
}

}  // namespace

LeggedTimingResult ParameterizeLeggedBodyTiming(
    const std::vector<LeggedTransition>& transitions,
    const LeggedCapability& capability,
    const std::size_t maximum_samples,
    const std::stop_token stop_token) {
  if (stop_token.stop_requested()) {
    return Failure("REQUEST_CANCELED", true);
  }
  if (transitions.empty()) {
    return Failure("LEGGED_TIMING_PATH_EMPTY");
  }
  if (!ValidVelocityInterval(capability.forward_speed_mps) ||
      !ValidVelocityInterval(capability.lateral_speed_mps) ||
      !ValidVelocityInterval(capability.yaw_rate_radps) ||
      !std::isfinite(capability.step_vertical_rate_mps) ||
      capability.step_vertical_rate_mps <= 0.0 ||
      !std::isfinite(capability.maximum_linear_acceleration_mps2) ||
      capability.maximum_linear_acceleration_mps2 <= 0.0 ||
      !std::isfinite(capability.maximum_yaw_acceleration_radps2) ||
      capability.maximum_yaw_acceleration_radps2 <= 0.0) {
    return Failure("LEGGED_TIMING_CAPABILITY_INVALID");
  }
  if (maximum_samples <= 1U ||
      transitions.size() > maximum_samples - 1U) {
    return Failure("LEGGED_TIMING_SAMPLE_LIMIT");
  }
  const std::size_t samples_per_transition = std::min(
      kPreferredSamplesPerTransition,
      (maximum_samples - 1U) / transitions.size());
  if (samples_per_transition == 0U) {
    return Failure("LEGGED_TIMING_SAMPLE_LIMIT");
  }

  TrajectoryReference trajectory{
      .semantics = TrajectorySemantics::kLeggedBodyReference,
      .points = {},
  };
  trajectory.points.reserve(
      1U + transitions.size() * samples_per_transition);
  trajectory.points.push_back(TrajectoryPoint{
      .time_from_start = std::chrono::nanoseconds{0},
      .pose = PoseAt(transitions.front(), 0.0),
      .velocity = {},
  });
  std::chrono::nanoseconds elapsed{0};
  for (const LeggedTransition& transition : transitions) {
    if (stop_token.stop_requested()) {
      return Failure("REQUEST_CANCELED", true);
    }
    const auto duration = DurationFor(transition, capability);
    if (!duration.has_value() ||
        elapsed.count() >
            std::numeric_limits<std::int64_t>::max() - duration->count()) {
      return Failure("LEGGED_TIMING_DURATION_INVALID");
    }
    const double seconds = std::chrono::duration<double>(*duration).count();
    const double yaw_delta = ShortestYawDelta(
        transition.source_pose.yaw_rad,
        transition.target_pose.yaw_rad);
    for (std::size_t sample = 1U;
         sample <= samples_per_transition; ++sample) {
      if (stop_token.stop_requested()) {
        return Failure("REQUEST_CANCELED", true);
      }
      const double ratio = static_cast<double>(sample) /
          static_cast<double>(samples_per_transition);
      const double progress = ratio * ratio * (3.0 - 2.0 * ratio);
      const double rate = 6.0 * ratio * (1.0 - ratio) / seconds;
      const auto sample_count =
          static_cast<std::int64_t>(samples_per_transition);
      const auto sample_index = static_cast<std::int64_t>(sample);
      const std::chrono::nanoseconds offset{
          (duration->count() / sample_count) * sample_index +
          ((duration->count() % sample_count) * sample_index) /
              sample_count};
      trajectory.points.push_back(TrajectoryPoint{
          .time_from_start = elapsed + offset,
          .pose = PoseAt(transition, progress),
          .velocity = Twist3{
              .linear_mps = Vec3{
                  .x = rate * (transition.target_pose.position_m.x -
                               transition.source_pose.position_m.x),
                  .y = rate * (transition.target_pose.position_m.y -
                               transition.source_pose.position_m.y),
                  .z = rate * (transition.target_pose.position_m.z -
                               transition.source_pose.position_m.z),
              },
              .angular_radps = Vec3{.z = rate * yaw_delta},
          },
      });
    }
    elapsed += *duration;
  }
  return LeggedTimingResult{
      .trajectory = std::move(trajectory),
      .canceled = false,
      .reason_code = {},
  };
}

}  // namespace lunar::pure_planning::legged

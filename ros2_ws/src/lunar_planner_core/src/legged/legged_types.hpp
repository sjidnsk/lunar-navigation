#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <optional>
#include <vector>

#include "lunar_planner_core/types/geometry.hpp"
#include "lunar_planner_core/types/platform_capability.hpp"

namespace lunar::planning::legged {

struct LeggedPose final {
  Vec3 position_m;
  double yaw_rad{};

  bool operator==(const LeggedPose&) const = default;
};

struct LeggedLatticeState final {
  std::int32_t cell_x{};
  std::int32_t cell_y{};
  std::int32_t yaw_bin{};
  Interval reachable_body_z_m;
};

struct LeggedTransition final {
  LeggedPose source_pose;
  LeggedPose target_pose;
  Interval target_body_z_m;
  std::size_t primitive_index{};
  LeggedPrimitiveKind primitive_kind{LeggedPrimitiveKind::kForward};
  double path_length_m{};
  std::size_t stable_index{};

  bool operator==(const LeggedTransition&) const = default;
};

struct LeggedDiscretePlan final {
  std::vector<LeggedTransition> transitions;
  double cost{};
  std::size_t expanded_states{};
};

[[nodiscard]] inline bool ValidInterval(
    const Interval& interval) noexcept {
  return std::isfinite(interval.lower) &&
      std::isfinite(interval.upper) && interval.lower <= interval.upper;
}

[[nodiscard]] inline std::optional<Interval> IntersectIntervals(
    const Interval& lhs, const Interval& rhs) noexcept {
  if (!ValidInterval(lhs) || !ValidInterval(rhs)) {
    return std::nullopt;
  }
  const Interval result{
      .lower = std::max(lhs.lower, rhs.lower),
      .upper = std::min(lhs.upper, rhs.upper),
  };
  return ValidInterval(result) ? std::optional<Interval>{result}
                               : std::nullopt;
}

[[nodiscard]] inline double NormalizeYaw(double yaw_rad) noexcept {
  if (!std::isfinite(yaw_rad)) {
    return yaw_rad;
  }
  yaw_rad = std::remainder(yaw_rad, 2.0 * std::numbers::pi);
  if (yaw_rad <= -std::numbers::pi) {
    yaw_rad += 2.0 * std::numbers::pi;
  }
  return yaw_rad;
}

[[nodiscard]] inline double ShortestYawDelta(
    const double from_rad, const double to_rad) noexcept {
  return NormalizeYaw(to_rad - from_rad);
}

[[nodiscard]] inline Quaternion QuaternionFromYaw(
    const double yaw_rad) noexcept {
  return Quaternion{
      .w = std::cos(yaw_rad / 2.0),
      .z = std::sin(yaw_rad / 2.0),
  };
}

[[nodiscard]] inline std::optional<double> YawFromQuaternion(
    const Quaternion& quaternion) noexcept {
  const double norm = std::sqrt(
      quaternion.w * quaternion.w + quaternion.x * quaternion.x +
      quaternion.y * quaternion.y + quaternion.z * quaternion.z);
  if (!std::isfinite(norm) || std::abs(norm - 1.0) > 1.0e-6) {
    return std::nullopt;
  }
  const double yaw = std::atan2(
      2.0 * (quaternion.w * quaternion.z +
             quaternion.x * quaternion.y),
      1.0 - 2.0 * (quaternion.y * quaternion.y +
                   quaternion.z * quaternion.z));
  return std::isfinite(yaw) ? std::optional<double>{NormalizeYaw(yaw)}
                            : std::nullopt;
}

}  // namespace lunar::planning::legged

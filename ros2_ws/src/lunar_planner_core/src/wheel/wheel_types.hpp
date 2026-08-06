#pragma once

#include <cmath>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <optional>
#include <string>
#include <vector>

#include "lunar_planner_core/types/geometry.hpp"
#include "lunar_planner_core/types/platform_capability.hpp"

namespace lunar::planning::wheel {

enum class WheelMotionMode : std::uint8_t {
  kStart,
  kForward,
  kReverse,
};

struct WheelPose final {
  Vec3 position_m;
  double yaw_rad{};

  bool operator==(const WheelPose&) const = default;
};

struct WheelLatticeState final {
  std::int32_t cell_x{};
  std::int32_t cell_y{};
  std::int32_t yaw_bin{};
  WheelMotionMode motion_mode{WheelMotionMode::kStart};

  auto operator<=>(const WheelLatticeState&) const = default;
};

struct WheelTransition final {
  WheelPose source_pose;
  WheelPose target_pose;
  double curvature_per_m{};
  std::size_t primitive_index{};
  WheelPrimitiveKind primitive_kind{WheelPrimitiveKind::kForward};
  WheelMotionMode source_mode{WheelMotionMode::kStart};
  WheelMotionMode target_mode{WheelMotionMode::kStart};
  double path_length_m{};
  double surface_slope_rad{};
  double roughness_m{};
  bool reverse{};
  std::size_t stable_index{};

  bool operator==(const WheelTransition&) const = default;
};

struct WheelDiscretePlan final {
  std::vector<WheelTransition> transitions;
  double cost{};
  std::size_t expanded_states{};
};

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
      .x = 0.0,
      .y = 0.0,
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

}  // namespace lunar::planning::wheel

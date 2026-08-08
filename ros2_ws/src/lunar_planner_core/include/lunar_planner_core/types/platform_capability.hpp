#pragma once

#include <chrono>
#include <cstdint>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

#include "lunar_planner_core/types/geometry.hpp"

namespace lunar::planning {

enum class PlatformType : std::uint8_t {
  kWheeled = 1,
  kLegged = 2,
  kHopper = 3,
};

enum class WheelPrimitiveKind : std::uint8_t {
  kForward,
  kReverse,
  kForwardArc,
  kReverseArc,
  kSpinClockwise,
  kSpinCounterclockwise,
  kStopAndSwitch,
};

struct WheelMotionPrimitive final {
  std::string primitive_id;
  WheelPrimitiveKind kind{};
  Pose3 relative_end_pose;
};

struct WheeledCapability final {
  std::vector<Vec2> footprint_xy_m;
  Vec3 body_extent_m;
  double wheel_diameter_m{};
  double wheel_width_m{};
  double wheelbase_m{};
  double track_width_m{};
  double minimum_underbody_clearance_m{};
  double maximum_local_obstacle_relief_m{};
  bool allow_unsupported_gap{};
  double minimum_body_z_m{};
  double maximum_body_z_m{};
  double maximum_forward_speed_mps{};
  double maximum_reverse_speed_mps{};
  double maximum_spin_rate_radps{};
  double maximum_acceleration_mps2{};
  double maximum_braking_deceleration_mps2{};
  double maximum_yaw_acceleration_radps2{};
  double maximum_lateral_acceleration_mps2{};
  double maximum_curvature_per_m{};
  double maximum_slope_rad{};
  double minimum_clearance_m{};
  std::vector<WheelMotionPrimitive> motion_primitives;
};

enum class LeggedPrimitiveKind : std::uint8_t {
  kForward,
  kBackward,
  kLateralLeft,
  kLateralRight,
  kSpin,
  kCoupled,
};

struct LeggedBodyPrimitive final {
  std::string primitive_id;
  LeggedPrimitiveKind kind{};
  Vec3 body_frame_displacement_m;
  double yaw_change_rad{};
};

struct LeggedCapability final {
  Vec3 body_extent_m;
  double platform_mass_kg{};
  double maximum_payload_kg{};
  double maximum_slope_rad{};
  double maximum_step_height_m{};
  double maximum_gap_width_m{};
  double minimum_body_clearance_m{};
  double step_vertical_rate_mps{};
  Interval body_height_m;
  Interval forward_speed_mps;
  Interval lateral_speed_mps;
  Interval yaw_rate_radps;
  double maximum_linear_acceleration_mps2{};
  double maximum_yaw_acceleration_radps2{};
  std::vector<LeggedBodyPrimitive> motion_primitives;
};

struct HopperCapability final {
  double specific_impulse_s{};
  double reference_total_mass_kg{};
  double reference_propellant_mass_kg{};
  double landing_support_radius_m{};
  double flight_collision_radius_m{};
  double maximum_landing_plane_residual_m{};
  double landing_lateral_margin_m{};
  double flight_map_margin_m{};
  double reachability_delta_v_margin_ratio{};
  double standard_gravity_mps2{};
  double maximum_landing_slope_rad{};
};

using PlatformCapability =
    std::variant<WheeledCapability, LeggedCapability, HopperCapability>;

[[nodiscard]] inline PlatformType CapabilityPlatform(
    const PlatformCapability& capability) noexcept {
  return std::visit(
      [](const auto& typed_capability) {
        using Capability = std::decay_t<decltype(typed_capability)>;
        if constexpr (std::is_same_v<Capability, WheeledCapability>) {
          return PlatformType::kWheeled;
        } else if constexpr (std::is_same_v<Capability, LeggedCapability>) {
          return PlatformType::kLegged;
        } else {
          return PlatformType::kHopper;
        }
      },
      capability);
}

}  // namespace lunar::planning

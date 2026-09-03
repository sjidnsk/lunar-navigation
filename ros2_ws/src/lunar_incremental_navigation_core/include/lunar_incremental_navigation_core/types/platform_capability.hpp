#pragma once

#include <chrono>
#include <cstdint>
#include <string>
#include <variant>
#include <vector>

#include "lunar_incremental_navigation_core/types/geometry.hpp"

namespace lunar::incremental_navigation {

using Point2 = Vec2;

struct TraversabilityProfile final {
  // Transitional Grid V1 fields. The fine derivation path below does not read
  // occupancy thresholds or the legacy inflation radius.
  std::int32_t global_occupancy_threshold{50};
  double local_occupancy_threshold{0.5};
  double maximum_slope_rad{};
  double inflation_radius_m{};
  std::vector<Point2> planar_envelope_xy_m;
  double preferred_clearance_m{};
  double slope_weight{};
  double relief_weight{};
  double clearance_weight{};
  double start_blind_zone_margin_m{};
  double goal_position_tolerance_m{};
  double goal_yaw_tolerance_rad{};
};

enum class PlatformType : std::uint8_t {
  kWheeled = 1,
  kLegged = 2,
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
  double nominal_body_height_m{};
  double platform_mass_kg{};
  double nominal_payload_kg{};
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
  bool unknown_is_traversable{};
  std::vector<LeggedBodyPrimitive> motion_primitives;
};

using PlatformCapability =
    std::variant<WheeledCapability, LeggedCapability>;

[[nodiscard]] inline PlatformType CapabilityPlatform(
    const PlatformCapability& capability) noexcept {
  return std::holds_alternative<WheeledCapability>(capability)
             ? PlatformType::kWheeled
             : PlatformType::kLegged;
}

}  // namespace lunar::incremental_navigation

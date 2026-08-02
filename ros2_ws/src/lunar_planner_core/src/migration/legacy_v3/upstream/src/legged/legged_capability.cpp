#include "lunar_path_planner/v3/legged/legged_capability.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <numeric>
#include <string>

namespace lunar::planning::v3 {
namespace {

[[nodiscard]] Error Invalid(std::string path, std::string message) {
  return Error{
      .code = ErrorCode::kInvalidArgument,
      .field_path = std::move(path),
      .message = std::move(message),
  };
}

[[nodiscard]] bool IsFinite(const Vec3& value) noexcept {
  return std::isfinite(value.x) && std::isfinite(value.y) &&
         std::isfinite(value.z);
}

[[nodiscard]] bool IsValidInterval(const Interval& value) noexcept {
  return std::isfinite(value.lower) && std::isfinite(value.upper) &&
         value.lower <= value.upper;
}

[[nodiscard]] bool IsNonNegativeFinite(double value) noexcept {
  return std::isfinite(value) && value >= 0.0;
}

[[nodiscard]] bool IsValidCollisionEnvelope(
    const BodyConvexPolytope& envelope) noexcept {
  const auto& halfspaces = envelope.body_frame_halfspaces.halfspaces;
  if (halfspaces.size() < 4U) {
    return false;
  }
  return std::ranges::all_of(
      halfspaces, [](const Halfspace3& halfspace) {
        const double norm =
            std::hypot(halfspace.normal.x, halfspace.normal.y,
                       halfspace.normal.z);
        return IsFinite(halfspace.normal) &&
               std::isfinite(halfspace.offset_m) &&
               halfspace.offset_m >= 0.0 && norm > 0.0;
      });
}

}  // namespace

Result<LeggedCapabilityView> LeggedCapabilityView::Create(
    const SafetyCapabilityProfile& profile) {
  if (!std::holds_alternative<LeggedCapability>(profile.content)) {
    return Invalid("safety_capability.content",
                   "legged platform requires a legged capability");
  }
  const auto& capability = std::get<LeggedCapability>(profile.content);
  if (profile.content_ref.id.empty() ||
      profile.content_ref.content_hash.size() != 64U) {
    return Invalid("safety_capability.content_ref",
                   "capability content reference is incomplete");
  }
  if (capability.frame_id.empty()) {
    return Invalid("safety_capability.content.frame_id",
                   "frame id must not be empty");
  }
  if (capability.reference_point_id.empty()) {
    return Invalid("safety_capability.content.reference_point_id",
                   "a fixed body reference point is required");
  }
  if (capability.footstep_feasibility_guaranteed) {
    return Invalid(
        "safety_capability.content.footstep_feasibility_guaranteed",
        "the body-only planner cannot guarantee footstep feasibility");
  }
  if (!IsValidCollisionEnvelope(capability.collision_envelope)) {
    return Invalid("safety_capability.content.collision_envelope",
                   "collision envelope must be a finite convex polytope");
  }

  const auto& terrain = capability.terrain_thresholds;
  if (!IsNonNegativeFinite(terrain.maximum_slope_rad) ||
      terrain.maximum_slope_rad > std::numbers::pi / 2.0 ||
      !IsNonNegativeFinite(terrain.maximum_roughness_m) ||
      !IsNonNegativeFinite(terrain.maximum_step_height_m) ||
      !IsNonNegativeFinite(terrain.maximum_gap_width_m) ||
      !std::isfinite(terrain.minimum_confidence) ||
      terrain.minimum_confidence < 0.0 ||
      terrain.minimum_confidence > 1.0 ||
      !IsNonNegativeFinite(terrain.minimum_body_clearance_m) ||
      !IsNonNegativeFinite(terrain.minimum_body_height_m) ||
      !IsNonNegativeFinite(terrain.maximum_body_height_m) ||
      terrain.minimum_body_height_m > terrain.maximum_body_height_m) {
    return Invalid("safety_capability.content.terrain_thresholds",
                   "terrain thresholds must be finite and ordered");
  }

  const auto& velocity = capability.body_velocity_limits;
  if (!IsValidInterval(velocity.forward_mps) ||
      !IsValidInterval(velocity.lateral_mps) ||
      !IsValidInterval(velocity.vertical_mps) ||
      !IsValidInterval(velocity.yaw_rate_radps) ||
      !std::isfinite(velocity.linear_acceleration_mps2) ||
      velocity.linear_acceleration_mps2 <= 0.0 ||
      !std::isfinite(velocity.yaw_acceleration_radps2) ||
      velocity.yaw_acceleration_radps2 <= 0.0) {
    return Invalid("safety_capability.content.body_velocity_limits",
                   "velocity and acceleration limits are invalid");
  }

  return LeggedCapabilityView{
      .content_ref = profile.content_ref,
      .frame_id = capability.frame_id,
      .reference_point_id = capability.reference_point_id,
      .collision_envelope = capability.collision_envelope,
      .terrain_thresholds = terrain,
      .velocity_envelope =
          BodyFrameVelocityEnvelope{
              .forward_mps = velocity.forward_mps,
              .lateral_mps = velocity.lateral_mps,
              .vertical_mps = velocity.vertical_mps,
              .yaw_rate_radps = velocity.yaw_rate_radps,
          },
      .certified_state_error_bounds =
          capability.certified_state_error_bounds,
      .preferred_body_height_m =
          std::midpoint(terrain.minimum_body_height_m,
                        terrain.maximum_body_height_m),
      .maximum_linear_acceleration_mps2 =
          velocity.linear_acceleration_mps2,
      .maximum_yaw_acceleration_radps2 =
          velocity.yaw_acceleration_radps2,
  };
}

}  // namespace lunar::planning::v3

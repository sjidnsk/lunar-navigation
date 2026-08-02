#include "lunar_path_planner/v3/wheel/wheel_capability.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <variant>

namespace lunar::planning::v3 {
namespace {

[[nodiscard]] Error Invalid(std::string path,
                            std::string message) {
  return {
      .code = ErrorCode::kInvalidArgument,
      .field_path = std::move(path),
      .message = std::move(message),
  };
}

[[nodiscard]] bool FiniteNonnegative(double value) noexcept {
  return std::isfinite(value) && value >= 0.0;
}

[[nodiscard]] bool FinitePositive(double value) noexcept {
  return std::isfinite(value) && value > 0.0;
}

[[nodiscard]] double HorizontalErrorRadius(
    const DeterministicVectorSet3& bound) noexcept {
  return std::visit(
      [](const auto& concrete) {
        using Bound = std::decay_t<decltype(concrete)>;
        if constexpr (std::is_same_v<Bound, AxisAlignedBox3>) {
          return std::hypot(std::abs(concrete.center.x) +
                                concrete.half_extent.x,
                            std::abs(concrete.center.y) +
                                concrete.half_extent.y);
        } else {
          return std::hypot(concrete.center.x, concrete.center.y) +
                 concrete.radius;
        }
      },
      bound);
}

}  // namespace

Result<WheelCapabilityView> WheelCapabilityView::Create(
    const SafetyCapabilityProfile& profile) {
  const auto* wheel = std::get_if<WheeledCapability>(&profile.content);
  if (wheel == nullptr) {
    return Invalid("/content",
                   "WHEEL_CAPABILITY_REQUIRED");
  }
  const auto& limits = wheel->hard_limits;
  if (!FinitePositive(limits.maximum_forward_speed_mps) ||
      !FinitePositive(limits.maximum_reverse_speed_mps) ||
      !FinitePositive(limits.maximum_spin_rate_radps) ||
      !FinitePositive(limits.maximum_forward_acceleration_mps2) ||
      !FinitePositive(limits.maximum_braking_deceleration_mps2) ||
      !FinitePositive(limits.maximum_yaw_acceleration_radps2) ||
      !FinitePositive(limits.maximum_lateral_acceleration_mps2) ||
      !FinitePositive(limits.maximum_drive_curvature_per_m) ||
      !FiniteNonnegative(limits.maximum_slope_rad) ||
      !FiniteNonnegative(limits.minimum_clearance_m)) {
    return Invalid("/content/hard_limits",
                   "INVALID_WHEEL_HARD_LIMIT");
  }
  if (wheel->collision_envelope.vertices_xy_m.size() < 3U) {
    return Invalid("/content/collision_envelope/vertices_xy_m",
                   "INVALID_WHEEL_COLLISION_ENVELOPE");
  }

  double support_radius = 0.0;
  for (const Vec2& vertex :
       wheel->collision_envelope.vertices_xy_m) {
    if (!std::isfinite(vertex.x) || !std::isfinite(vertex.y)) {
      return Invalid("/content/collision_envelope/vertices_xy_m",
                     "INVALID_WHEEL_COLLISION_ENVELOPE");
    }
    support_radius =
        std::max(support_radius, std::hypot(vertex.x, vertex.y));
  }
  const double error_radius = HorizontalErrorRadius(
      wheel->certified_state_error_bounds.position_bound_m);
  if (!FiniteNonnegative(error_radius) ||
      !std::isfinite(support_radius)) {
    return Invalid("/content/certified_state_error_bounds",
                   "INVALID_WHEEL_ERROR_BOUND");
  }

  WheelCapabilityView view;
  view.content_ref_ = profile.content_ref;
  view.wheel_ = *wheel;
  view.footprint_support_radius_m_ = support_radius;
  view.certified_horizontal_position_error_m_ = error_radius;
  return view;
}

const ContentRef& WheelCapabilityView::content_ref() const noexcept {
  return content_ref_;
}

const WheeledCapability& WheelCapabilityView::wheel() const noexcept {
  return wheel_;
}

const WheelHardLimits& WheelCapabilityView::hard_limits() const noexcept {
  return wheel_.hard_limits;
}

const ExtrudedConvexFootprint&
WheelCapabilityView::collision_envelope() const noexcept {
  return wheel_.collision_envelope;
}

double WheelCapabilityView::footprint_support_radius_m() const noexcept {
  return footprint_support_radius_m_;
}

double WheelCapabilityView::certified_horizontal_position_error_m()
    const noexcept {
  return certified_horizontal_position_error_m_;
}

}  // namespace lunar::planning::v3

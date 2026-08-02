#include "lunar_path_planner/v3/legged/body_motion_primitive.hpp"

#include <algorithm>
#include <cmath>
#include <set>
#include <string>
#include <tuple>

namespace lunar::planning::v3 {
namespace {

constexpr double kMotionTolerance = 1.0e-12;

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

[[nodiscard]] bool Contains(const Interval& interval,
                            double value) noexcept {
  return value >= interval.lower - kMotionTolerance &&
         value <= interval.upper + kMotionTolerance;
}

[[nodiscard]] Result<BodyMotionKind> ResolveKind(
    const LeggedBodyPrimitive& primitive) {
  const auto& displacement = primitive.body_frame_displacement_m;
  const bool translates =
      std::hypot(displacement.x, displacement.y) > kMotionTolerance ||
      std::abs(displacement.z) > kMotionTolerance;
  const bool rotates =
      std::abs(primitive.yaw_change_rad) > kMotionTolerance;
  switch (primitive.kind) {
    case LeggedBodyPrimitive::Kind::kForward:
      if (!(displacement.x > kMotionTolerance) || rotates) {
        return Invalid("primitive.body_frame_displacement_m",
                       "forward primitive must translate forward only");
      }
      return BodyMotionKind::kForward;
    case LeggedBodyPrimitive::Kind::kBackward:
      if (!(displacement.x < -kMotionTolerance) || rotates) {
        return Invalid("primitive.body_frame_displacement_m",
                       "backward primitive must translate backward only");
      }
      return BodyMotionKind::kBackward;
    case LeggedBodyPrimitive::Kind::kLateral:
      if (std::abs(displacement.y) <= kMotionTolerance || rotates) {
        return Invalid("primitive.body_frame_displacement_m",
                       "lateral primitive must have signed lateral motion");
      }
      return displacement.y > 0.0 ? BodyMotionKind::kLateralLeft
                                  : BodyMotionKind::kLateralRight;
    case LeggedBodyPrimitive::Kind::kDiagonal:
      if (std::abs(displacement.x) <= kMotionTolerance ||
          std::abs(displacement.y) <= kMotionTolerance || rotates) {
        return Invalid("primitive.body_frame_displacement_m",
                       "diagonal primitive needs longitudinal and lateral motion");
      }
      return BodyMotionKind::kDiagonal;
    case LeggedBodyPrimitive::Kind::kSpin:
      if (translates || !rotates) {
        return Invalid("primitive.yaw_change_rad",
                       "spin primitive must rotate without translation");
      }
      return BodyMotionKind::kSpin;
    case LeggedBodyPrimitive::Kind::kCoupled:
      if (!translates || !rotates) {
        return Invalid("primitive",
                       "coupled primitive needs translation and yaw motion");
      }
      return BodyMotionKind::kCoupledTranslationYaw;
  }
  return Invalid("primitive.kind", "unsupported primitive kind");
}

[[nodiscard]] bool ValidRef(const ContentRef& ref) noexcept {
  return !ref.id.empty() && ref.content_hash.size() == 64U;
}

}  // namespace

Result<BodyMotionPrimitiveCatalog> BodyMotionPrimitiveCatalog::Create(
    const SafetyCapabilityProfile& profile) {
  const auto view_result = LeggedCapabilityView::Create(profile);
  if (!IsOk(view_result)) {
    return std::get<Error>(view_result);
  }
  const auto& source = std::get<LeggedCapability>(profile.content);
  if (source.motion_primitives.empty()) {
    return Invalid("safety_capability.content.motion_primitives",
                   "at least one certified body primitive is required");
  }

  BodyMotionPrimitiveCatalog catalog;
  catalog.content_ref_ = profile.content_ref;
  catalog.capability_ = std::get<LeggedCapabilityView>(view_result);
  catalog.ordered_primitives_.reserve(source.motion_primitives.size());
  std::set<PrimitiveId> ids;
  bool has_lateral_left = false;
  bool has_lateral_right = false;
  bool has_spin = false;

  for (const auto& primitive : source.motion_primitives) {
    if (primitive.primitive_id.empty() ||
        !ids.insert(primitive.primitive_id).second) {
      return Invalid("safety_capability.content.motion_primitives",
                     "primitive ids must be nonempty and unique");
    }
    if (!IsFinite(primitive.body_frame_displacement_m) ||
        !std::isfinite(primitive.yaw_change_rad) ||
        primitive.nominal_duration.value.count() <= 0 ||
        !ValidRef(primitive.sampled_body_sweep_ref)) {
      return Invalid("safety_capability.content.motion_primitives",
                     "primitive motion, duration, or sweep ref is invalid");
    }
    const auto kind_result = ResolveKind(primitive);
    if (!IsOk(kind_result)) {
      return std::get<Error>(kind_result);
    }
    const BodyMotionKind kind = std::get<BodyMotionKind>(kind_result);
    const double seconds =
        std::chrono::duration<double>(primitive.nominal_duration.value)
            .count();
    const double forward_rate =
        primitive.body_frame_displacement_m.x / seconds;
    const double lateral_rate =
        primitive.body_frame_displacement_m.y / seconds;
    const double vertical_rate =
        primitive.body_frame_displacement_m.z / seconds;
    const double yaw_rate = primitive.yaw_change_rad / seconds;
    const auto& envelope = catalog.capability_.velocity_envelope;
    if (!Contains(envelope.forward_mps, forward_rate) ||
        !Contains(envelope.lateral_mps, lateral_rate) ||
        !Contains(envelope.vertical_mps, vertical_rate) ||
        !Contains(envelope.yaw_rate_radps, yaw_rate)) {
      return Invalid("safety_capability.content.motion_primitives",
                     "primitive exceeds the certified velocity envelope");
    }

    has_lateral_left =
        has_lateral_left || kind == BodyMotionKind::kLateralLeft;
    has_lateral_right =
        has_lateral_right || kind == BodyMotionKind::kLateralRight;
    has_spin = has_spin || kind == BodyMotionKind::kSpin;
    catalog.ordered_primitives_.push_back(BodyMotionPrimitive{
        .id = primitive.primitive_id,
        .kind = kind,
        .relative_end =
            PoseXyzYaw{
                .position_m = primitive.body_frame_displacement_m,
                .yaw_rad = primitive.yaw_change_rad,
            },
        .nominal_duration = primitive.nominal_duration,
        .nominal_envelope = envelope,
        .normalized_samples = {0.0, 0.25, 0.5, 0.75, 1.0},
        .secondary_costs = SecondaryCostVector{},
        .sampled_body_sweep_ref = primitive.sampled_body_sweep_ref,
        .maximum_up_delta_m =
            std::max(0.0, primitive.body_frame_displacement_m.z),
        .maximum_down_delta_m =
            std::max(0.0, -primitive.body_frame_displacement_m.z),
    });
  }

  if (!has_lateral_left || !has_lateral_right || !has_spin) {
    return Invalid("safety_capability.content.motion_primitives",
                   "both lateral directions and spin are required");
  }
  std::ranges::sort(
      catalog.ordered_primitives_,
      [](const BodyMotionPrimitive& lhs,
         const BodyMotionPrimitive& rhs) {
        return std::tuple{lhs.kind, lhs.id} <
               std::tuple{rhs.kind, rhs.id};
      });
  return catalog;
}

std::span<const BodyMotionPrimitive>
BodyMotionPrimitiveCatalog::ordered_primitives() const noexcept {
  return ordered_primitives_;
}

const ContentRef& BodyMotionPrimitiveCatalog::content_ref() const noexcept {
  return content_ref_;
}

const LeggedCapabilityView&
BodyMotionPrimitiveCatalog::capability() const noexcept {
  return capability_;
}

}  // namespace lunar::planning::v3

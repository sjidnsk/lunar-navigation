#include "lunar_path_planner/v3/wheel/wheel_primitive.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <ranges>
#include <set>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <variant>

namespace lunar::planning::v3 {
namespace {

constexpr std::array<WheelMotionMode, 4U> kActiveModes{
    WheelMotionMode::kForward,
    WheelMotionMode::kReverse,
    WheelMotionMode::kSpinClockwise,
    WheelMotionMode::kSpinCounterClockwise,
};

[[nodiscard]] Error Invalid(std::string path,
                            std::string message) {
  return {
      .code = ErrorCode::kInvalidArgument,
      .field_path = std::move(path),
      .message = std::move(message),
  };
}

[[nodiscard]] std::size_t ModeIndex(WheelMotionMode mode) noexcept {
  return static_cast<std::size_t>(mode);
}

[[nodiscard]] std::string_view ModeName(
    WheelMotionMode mode) noexcept {
  switch (mode) {
    case WheelMotionMode::kStart:
      return "start";
    case WheelMotionMode::kForward:
      return "forward";
    case WheelMotionMode::kReverse:
      return "reverse";
    case WheelMotionMode::kSpinClockwise:
      return "spin-cw";
    case WheelMotionMode::kSpinCounterClockwise:
      return "spin-ccw";
  }
  return "invalid";
}

[[nodiscard]] bool FinitePose(const PoseXyzYaw& pose) noexcept {
  return std::isfinite(pose.position_m.x) &&
         std::isfinite(pose.position_m.y) &&
         std::isfinite(pose.position_m.z) &&
         std::isfinite(pose.yaw_rad);
}

[[nodiscard]] bool IsZeroPose(const PoseXyzYaw& pose) noexcept {
  return pose.position_m.x == 0.0 &&
         pose.position_m.y == 0.0 &&
         pose.position_m.z == 0.0 && pose.yaw_rad == 0.0;
}

[[nodiscard]] WheelMotionMode ActiveMode(
    WheelMotionPrimitive::Kind kind) noexcept {
  switch (kind) {
    case WheelMotionPrimitive::Kind::kDriveForwardLine:
    case WheelMotionPrimitive::Kind::kDriveForwardArc:
      return WheelMotionMode::kForward;
    case WheelMotionPrimitive::Kind::kDriveReverseLine:
    case WheelMotionPrimitive::Kind::kDriveReverseArc:
      return WheelMotionMode::kReverse;
    case WheelMotionPrimitive::Kind::kSpinCw:
      return WheelMotionMode::kSpinClockwise;
    case WheelMotionPrimitive::Kind::kSpinCcw:
      return WheelMotionMode::kSpinCounterClockwise;
    case WheelMotionPrimitive::Kind::kStopAndSwitch:
      return WheelMotionMode::kStart;
  }
  return WheelMotionMode::kStart;
}

[[nodiscard]] WheelPrimitiveKind PrimitiveKindOf(
    WheelMotionPrimitive::Kind kind) noexcept {
  switch (kind) {
    case WheelMotionPrimitive::Kind::kDriveForwardLine:
    case WheelMotionPrimitive::Kind::kDriveReverseLine:
      return WheelPrimitiveKind::kDriveLine;
    case WheelMotionPrimitive::Kind::kDriveForwardArc:
    case WheelMotionPrimitive::Kind::kDriveReverseArc:
      return WheelPrimitiveKind::kDriveArc;
    case WheelMotionPrimitive::Kind::kSpinCw:
    case WheelMotionPrimitive::Kind::kSpinCcw:
      return WheelPrimitiveKind::kSpin;
    case WheelMotionPrimitive::Kind::kStopAndSwitch:
      return WheelPrimitiveKind::kModeSwitch;
  }
  return WheelPrimitiveKind::kModeSwitch;
}

[[nodiscard]] WheelPrimitive ConvertActive(
    const WheelMotionPrimitive& spec) {
  const WheelMotionMode mode = ActiveMode(spec.kind);
  return {
      .id = spec.primitive_id,
      .capability_primitive_id = spec.primitive_id,
      .kind = PrimitiveKindOf(spec.kind),
      .source_mode = mode,
      .target_mode = mode,
      .relative_end = spec.relative_end_pose,
      .nominal_duration = spec.nominal_duration,
      .secondary_costs = {},
      .sweep = {.validation_ref = spec.swept_geometry_ref},
  };
}

[[nodiscard]] WheelPrimitive ConvertSwitch(
    const WheelMotionPrimitive& spec,
    WheelMotionMode source,
    WheelMotionMode target) {
  return {
      .id = spec.primitive_id + ":" + std::string{ModeName(source)} +
            ":" + std::string{ModeName(target)},
      .capability_primitive_id = spec.primitive_id,
      .kind = WheelPrimitiveKind::kModeSwitch,
      .source_mode = source,
      .target_mode = target,
      .relative_end = spec.relative_end_pose,
      .nominal_duration = spec.nominal_duration,
      .secondary_costs = {},
      .sweep = {.validation_ref = spec.swept_geometry_ref},
  };
}

[[nodiscard]] auto PrimitiveOrderKey(
    const WheelPrimitive& primitive) {
  return std::tuple{
      primitive.capability_primitive_id,
      static_cast<int>(primitive.source_mode),
      static_cast<int>(primitive.target_mode),
      static_cast<int>(primitive.kind),
      primitive.id,
  };
}

}  // namespace

Result<WheelPrimitiveCatalog> WheelPrimitiveCatalog::Create(
    const SafetyCapabilityProfile& profile) {
  const auto capability_result = WheelCapabilityView::Create(profile);
  if (!IsOk(capability_result)) {
    return std::get<Error>(capability_result);
  }
  const auto& wheel =
      std::get<WheelCapabilityView>(capability_result).wheel();

  bool has_forward = false;
  bool has_reverse = false;
  bool has_spin_cw = false;
  bool has_spin_ccw = false;
  bool has_switch = false;
  std::set<PrimitiveId> unique_ids;
  std::vector<WheelMotionPrimitive> ordered_specs =
      wheel.motion_primitives;
  std::ranges::sort(
      ordered_specs, {}, &WheelMotionPrimitive::primitive_id);

  for (const auto& spec : ordered_specs) {
    if (spec.primitive_id.empty() ||
        !unique_ids.insert(spec.primitive_id).second) {
      return Invalid("/content/motion_primitives/primitive_id",
                     "INVALID_WHEEL_PRIMITIVE_ID");
    }
    if (!FinitePose(spec.relative_end_pose)) {
      return Invalid("/content/motion_primitives/relative_end_pose",
                     "INVALID_WHEEL_PRIMITIVE_POSE");
    }
    if (spec.nominal_duration.value.count() <= 0) {
      return Invalid(
          "/content/motion_primitives/nominal_duration_ns",
          "INVALID_WHEEL_PRIMITIVE_DURATION");
    }
    switch (spec.kind) {
      case WheelMotionPrimitive::Kind::kDriveForwardLine:
      case WheelMotionPrimitive::Kind::kDriveForwardArc:
        has_forward = true;
        break;
      case WheelMotionPrimitive::Kind::kDriveReverseLine:
      case WheelMotionPrimitive::Kind::kDriveReverseArc:
        has_reverse = true;
        break;
      case WheelMotionPrimitive::Kind::kSpinCw:
        has_spin_cw = true;
        break;
      case WheelMotionPrimitive::Kind::kSpinCcw:
        has_spin_ccw = true;
        break;
      case WheelMotionPrimitive::Kind::kStopAndSwitch:
        has_switch = true;
        if (!IsZeroPose(spec.relative_end_pose)) {
          return Invalid(
              "/content/motion_primitives/relative_end_pose",
              "NON_STATIONARY_WHEEL_MODE_SWITCH");
        }
        break;
    }
  }

  if (!has_forward || !has_reverse || !has_spin_cw ||
      !has_spin_ccw || !has_switch) {
    return Invalid("/content/motion_primitives",
                   "INCOMPLETE_WHEEL_PRIMITIVE_CATALOG");
  }

  WheelPrimitiveCatalog catalog;
  catalog.content_ref_ = profile.content_ref;
  for (const auto& spec : ordered_specs) {
    if (spec.kind != WheelMotionPrimitive::Kind::kStopAndSwitch) {
      auto primitive = ConvertActive(spec);
      catalog.outgoing_[ModeIndex(primitive.source_mode)]
          .push_back(primitive);
      catalog.all_.push_back(std::move(primitive));
      continue;
    }
    for (const WheelMotionMode active : kActiveModes) {
      auto enter =
          ConvertSwitch(spec, WheelMotionMode::kStart, active);
      catalog.outgoing_[ModeIndex(enter.source_mode)]
          .push_back(enter);
      catalog.all_.push_back(std::move(enter));

      auto stop =
          ConvertSwitch(spec, active, WheelMotionMode::kStart);
      catalog.outgoing_[ModeIndex(stop.source_mode)]
          .push_back(stop);
      catalog.all_.push_back(std::move(stop));
    }
  }

  for (auto& outgoing : catalog.outgoing_) {
    std::ranges::sort(outgoing, [](const WheelPrimitive& lhs,
                                  const WheelPrimitive& rhs) {
      return PrimitiveOrderKey(lhs) < PrimitiveOrderKey(rhs);
    });
  }
  std::ranges::sort(
      catalog.all_, [](const WheelPrimitive& lhs,
                       const WheelPrimitive& rhs) {
        return PrimitiveOrderKey(lhs) < PrimitiveOrderKey(rhs);
      });
  return catalog;
}

std::span<const WheelPrimitive> WheelPrimitiveCatalog::Outgoing(
    WheelMotionMode mode) const noexcept {
  const std::size_t index = ModeIndex(mode);
  if (index >= outgoing_.size()) {
    return {};
  }
  return outgoing_[index];
}

std::span<const WheelPrimitive> WheelPrimitiveCatalog::All()
    const noexcept {
  return all_;
}

const ContentRef& WheelPrimitiveCatalog::content_ref() const noexcept {
  return content_ref_;
}

}  // namespace lunar::planning::v3

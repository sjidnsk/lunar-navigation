#pragma once

#include <array>
#include <span>
#include <vector>

#include "lunar_path_planner/v3/search/platform_adapter.hpp"
#include "lunar_path_planner/v3/wheel/wheel_capability.hpp"

namespace lunar::planning::v3 {

enum class WheelMotionMode {
  kStart,
  kForward,
  kReverse,
  kSpinClockwise,
  kSpinCounterClockwise,
};

enum class WheelPrimitiveKind {
  kDriveLine,
  kDriveArc,
  kSpin,
  kModeSwitch,
};

struct WheelSweepDescriptor final {
  ContentRef validation_ref;
};

struct WheelPrimitive final {
  PrimitiveId id;
  PrimitiveId capability_primitive_id;
  WheelPrimitiveKind kind{WheelPrimitiveKind::kDriveLine};
  WheelMotionMode source_mode{WheelMotionMode::kStart};
  WheelMotionMode target_mode{WheelMotionMode::kStart};
  PoseXyzYaw relative_end;
  DurationNanoseconds nominal_duration;
  SecondaryCostVector secondary_costs;
  WheelSweepDescriptor sweep;
};

class WheelPrimitiveCatalog final {
 public:
  [[nodiscard]] static Result<WheelPrimitiveCatalog> Create(
      const SafetyCapabilityProfile& profile);

  [[nodiscard]] std::span<const WheelPrimitive> Outgoing(
      WheelMotionMode mode) const noexcept;
  [[nodiscard]] std::span<const WheelPrimitive> All() const noexcept;
  [[nodiscard]] const ContentRef& content_ref() const noexcept;

 private:
  static constexpr std::size_t kModeCount = 5U;

  ContentRef content_ref_;
  std::array<std::vector<WheelPrimitive>, kModeCount> outgoing_;
  std::vector<WheelPrimitive> all_;
};

}  // namespace lunar::planning::v3

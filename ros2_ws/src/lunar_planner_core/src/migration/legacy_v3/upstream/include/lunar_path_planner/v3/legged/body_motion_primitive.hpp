#pragma once

#include <span>
#include <vector>

#include "lunar_path_planner/v3/legged/legged_capability.hpp"
#include "lunar_path_planner/v3/search/platform_adapter.hpp"

namespace lunar::planning::v3 {

enum class BodyMotionKind {
  kForward,
  kBackward,
  kLateralLeft,
  kLateralRight,
  kDiagonal,
  kSpin,
  kCoupledTranslationYaw,
};

struct BodyMotionPrimitive final {
  PrimitiveId id;
  BodyMotionKind kind{BodyMotionKind::kForward};
  PoseXyzYaw relative_end;
  DurationNanoseconds nominal_duration;
  BodyFrameVelocityEnvelope nominal_envelope;
  std::vector<double> normalized_samples;
  SecondaryCostVector secondary_costs;
  ContentRef sampled_body_sweep_ref;
  double maximum_up_delta_m{};
  double maximum_down_delta_m{};
};

class BodyMotionPrimitiveCatalog final {
 public:
  [[nodiscard]] static Result<BodyMotionPrimitiveCatalog> Create(
      const SafetyCapabilityProfile& profile);

  [[nodiscard]] std::span<const BodyMotionPrimitive> ordered_primitives()
      const noexcept;
  [[nodiscard]] const ContentRef& content_ref() const noexcept;
  [[nodiscard]] const LeggedCapabilityView& capability() const noexcept;

 private:
  ContentRef content_ref_;
  LeggedCapabilityView capability_;
  std::vector<BodyMotionPrimitive> ordered_primitives_;
};

}  // namespace lunar::planning::v3

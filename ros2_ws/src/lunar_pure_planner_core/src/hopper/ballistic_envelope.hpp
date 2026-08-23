#pragma once

#include <optional>
#include <string>

#include "hopper/hopper_types.hpp"
#include "lunar_pure_planner_core/types/platform_capability.hpp"

namespace lunar::pure_planning::hopper {

struct AvailableSingleHopDeltaVResult final {
  std::optional<double> delta_v_mps;
  std::string reason_code;

  [[nodiscard]] bool ok() const noexcept {
    return delta_v_mps.has_value() && reason_code.empty();
  }
};

struct SingleHopEnvelopeEvidence final {
  double available_delta_v_mps{};
  double ideal_delta_v_mps{};
  double required_delta_v_mps{};
};

struct SingleHopEnvelopeResult final {
  std::optional<SingleHopEnvelopeEvidence> evidence;
  std::string reason_code;

  [[nodiscard]] bool ok() const noexcept {
    return evidence.has_value() && reason_code.empty();
  }
};

struct MinimumSingleHopEnvelopeEvidence final {
  BallisticArc arc;
  SingleHopEnvelopeEvidence envelope;
};

struct MinimumSingleHopEnvelopeResult final {
  std::optional<MinimumSingleHopEnvelopeEvidence> evidence;
  std::string reason_code;

  [[nodiscard]] bool ok() const noexcept {
    return evidence.has_value() && reason_code.empty();
  }
};

[[nodiscard]] AvailableSingleHopDeltaVResult AvailableSingleHopDeltaV(
    const HopperCapability& capability) noexcept;

[[nodiscard]] SingleHopEnvelopeResult EvaluateSingleHopEnvelope(
    const BallisticArc& arc,
    const HopperCapability& capability) noexcept;

[[nodiscard]] MinimumSingleHopEnvelopeResult
EvaluateMinimumSingleHopEnvelope(
    Vec3 launch_position_m,
    Vec3 landing_position_m,
    Vec3 gravity_mps2,
    double evidence_scale_m,
    const HopperCapability& capability) noexcept;

}  // namespace lunar::pure_planning::hopper

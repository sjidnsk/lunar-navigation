#pragma once

#include <optional>
#include <string>

#include "hopper/hopper_types.hpp"
#include "lunar_planner_core/types/planner_io.hpp"
#include "lunar_planner_core/types/platform_capability.hpp"

namespace lunar::planning::hopper {

struct AvailableDeltaVResult final {
  std::optional<double> delta_v_mps;
  std::string reason_code;

  [[nodiscard]] bool ok() const noexcept {
    return delta_v_mps.has_value() && reason_code.empty();
  }
};

struct PropellantEvidence final {
  double available_delta_v_mps{};
  double ideal_delta_v_mps{};
  double certified_delta_v_mps{};
  double ideal_fuel_required_kg{};
  double certified_fuel_required_kg{};
  double expected_remaining_usable_fuel_kg{};
};

struct PropellantEvaluationResult final {
  std::optional<PropellantEvidence> evidence;
  std::string reason_code;

  [[nodiscard]] bool ok() const noexcept {
    return evidence.has_value() && reason_code.empty();
  }
};

[[nodiscard]] AvailableDeltaVResult AvailableDeltaV(
    const HopperPropellantState& propellant,
    const HopperCapability& capability) noexcept;

[[nodiscard]] PropellantEvaluationResult EvaluatePropellant(
    const BallisticArc& arc,
    const HopperPropellantState& propellant,
    const HopperCapability& capability) noexcept;

}  // namespace lunar::planning::hopper

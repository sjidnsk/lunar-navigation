#include "hopper/propellant_model.hpp"

#include <algorithm>
#include <cmath>
#include <optional>
#include <string>

namespace lunar::planning::hopper {
namespace {

[[nodiscard]] bool FinitePositive(const double value) noexcept {
  return std::isfinite(value) && value > 0.0;
}

[[nodiscard]] bool Finite(const Vec3 value) noexcept {
  return std::isfinite(value.x) && std::isfinite(value.y) &&
      std::isfinite(value.z);
}

[[nodiscard]] double Norm(const Vec3 value) noexcept {
  return std::hypot(std::hypot(value.x, value.y), value.z);
}

[[nodiscard]] bool ValidCapability(
    const HopperCapability& capability) noexcept {
  return FinitePositive(capability.specific_impulse_s) &&
      FinitePositive(capability.standard_gravity_mps2) &&
      std::isfinite(capability.reachability_delta_v_margin_ratio) &&
      capability.reachability_delta_v_margin_ratio >= 0.0;
}

[[nodiscard]] std::optional<double> FuelForDeltaV(
    const double delta_v_mps, const double total_mass_kg,
    const double exhaust_velocity_mps) noexcept {
  if (!std::isfinite(delta_v_mps) || delta_v_mps < 0.0 ||
      !FinitePositive(total_mass_kg) || !FinitePositive(exhaust_velocity_mps)) {
    return std::nullopt;
  }
  const double exponent = -delta_v_mps / exhaust_velocity_mps;
  const double fuel = -total_mass_kg * std::expm1(exponent);
  if (!std::isfinite(fuel) || fuel < 0.0 || fuel >= total_mass_kg) {
    return std::nullopt;
  }
  return fuel;
}

}  // namespace

AvailableDeltaVResult AvailableDeltaV(
    const HopperPropellantState& propellant,
    const HopperCapability& capability) noexcept {
  if (!FinitePositive(propellant.total_mass_kg)) {
    return {.reason_code = "HOPPER_TOTAL_MASS_INVALID"};
  }
  if (!FinitePositive(propellant.remaining_usable_fuel_mass_kg) ||
      propellant.remaining_usable_fuel_mass_kg >= propellant.total_mass_kg) {
    return {.reason_code = "HOPPER_USABLE_FUEL_INVALID"};
  }
  if (!ValidCapability(capability)) {
    return {.reason_code = "HOPPER_CAPABILITY_INVALID"};
  }

  const double dry_mass =
      propellant.total_mass_kg - propellant.remaining_usable_fuel_mass_kg;
  if (!FinitePositive(dry_mass)) {
    return {.reason_code = "HOPPER_USABLE_FUEL_INVALID"};
  }
  const double exhaust_velocity =
      capability.specific_impulse_s * capability.standard_gravity_mps2;
  const double mass_ratio =
      propellant.remaining_usable_fuel_mass_kg / dry_mass;
  const double delta_v = exhaust_velocity * std::log1p(mass_ratio);
  if (!std::isfinite(delta_v) || delta_v <= 0.0) {
    return {.reason_code = "HOPPER_BALLISTIC_NUMERICAL_INDETERMINATE"};
  }
  return {.delta_v_mps = delta_v};
}

PropellantEvaluationResult EvaluatePropellant(
    const BallisticArc& arc,
    const HopperPropellantState& propellant,
    const HopperCapability& capability) noexcept {
  const AvailableDeltaVResult available =
      AvailableDeltaV(propellant, capability);
  if (!available.ok()) {
    return {.reason_code = available.reason_code};
  }
  if (!Finite(arc.launch_velocity_mps) || !Finite(arc.landing_velocity_mps) ||
      !std::isfinite(arc.flight_time_s) || arc.flight_time_s <= 0.0) {
    return {.reason_code = "HOPPER_BALLISTIC_NUMERICAL_INDETERMINATE"};
  }

  const double ideal_delta_v =
      Norm(arc.launch_velocity_mps) + Norm(arc.landing_velocity_mps);
  const double certified_delta_v =
      (1.0 + capability.reachability_delta_v_margin_ratio) * ideal_delta_v;
  const double exhaust_velocity =
      capability.specific_impulse_s * capability.standard_gravity_mps2;
  const auto ideal_fuel = FuelForDeltaV(
      ideal_delta_v, propellant.total_mass_kg, exhaust_velocity);
  const auto certified_fuel = FuelForDeltaV(
      certified_delta_v, propellant.total_mass_kg, exhaust_velocity);
  if (!std::isfinite(ideal_delta_v) || ideal_delta_v < 0.0 ||
      !std::isfinite(certified_delta_v) || !ideal_fuel.has_value() ||
      !certified_fuel.has_value()) {
    return {.reason_code = "HOPPER_BALLISTIC_NUMERICAL_INDETERMINATE"};
  }
  if (*certified_fuel > propellant.remaining_usable_fuel_mass_kg) {
    return {.reason_code = "HOPPER_FUEL_INSUFFICIENT"};
  }

  return {
      .evidence = PropellantEvidence{
          .available_delta_v_mps = *available.delta_v_mps,
          .ideal_delta_v_mps = ideal_delta_v,
          .certified_delta_v_mps = certified_delta_v,
          .ideal_fuel_required_kg = *ideal_fuel,
          .certified_fuel_required_kg = *certified_fuel,
          .expected_remaining_usable_fuel_kg = std::max(
              0.0, propellant.remaining_usable_fuel_mass_kg - *certified_fuel),
      },
  };
}

}  // namespace lunar::planning::hopper

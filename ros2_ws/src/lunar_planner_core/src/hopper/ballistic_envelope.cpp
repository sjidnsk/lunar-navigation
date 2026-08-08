#include "hopper/ballistic_envelope.hpp"

#include <cmath>

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
      FinitePositive(capability.reference_total_mass_kg) &&
      FinitePositive(capability.reference_propellant_mass_kg) &&
      capability.reference_propellant_mass_kg <
          capability.reference_total_mass_kg &&
      FinitePositive(capability.standard_gravity_mps2) &&
      std::isfinite(capability.reachability_delta_v_margin_ratio) &&
      capability.reachability_delta_v_margin_ratio >= 0.0;
}

}  // namespace

AvailableSingleHopDeltaVResult AvailableSingleHopDeltaV(
    const HopperCapability& capability) noexcept {
  if (!ValidCapability(capability)) {
    return {.reason_code = "HOPPER_CAPABILITY_INVALID"};
  }
  const double dry_mass = capability.reference_total_mass_kg -
      capability.reference_propellant_mass_kg;
  const double exhaust_velocity =
      capability.specific_impulse_s * capability.standard_gravity_mps2;
  const double delta_v = exhaust_velocity * std::log(
      capability.reference_total_mass_kg / dry_mass);
  if (!std::isfinite(delta_v) || delta_v <= 0.0) {
    return {.reason_code = "HOPPER_BALLISTIC_NUMERICAL_INDETERMINATE"};
  }
  return {.delta_v_mps = delta_v};
}

SingleHopEnvelopeResult EvaluateSingleHopEnvelope(
    const BallisticArc& arc,
    const HopperCapability& capability) noexcept {
  const AvailableSingleHopDeltaVResult available =
      AvailableSingleHopDeltaV(capability);
  if (!available.ok()) {
    return {.reason_code = available.reason_code};
  }
  if (!Finite(arc.launch_velocity_mps) || !Finite(arc.landing_velocity_mps) ||
      !std::isfinite(arc.flight_time_s) || arc.flight_time_s <= 0.0) {
    return {.reason_code = "HOPPER_BALLISTIC_NUMERICAL_INDETERMINATE"};
  }
  const double ideal_delta_v =
      Norm(arc.launch_velocity_mps) + Norm(arc.landing_velocity_mps);
  const double required_delta_v =
      (1.0 + capability.reachability_delta_v_margin_ratio) * ideal_delta_v;
  if (!std::isfinite(ideal_delta_v) || ideal_delta_v < 0.0 ||
      !std::isfinite(required_delta_v)) {
    return {.reason_code = "HOPPER_BALLISTIC_NUMERICAL_INDETERMINATE"};
  }
  if (required_delta_v > *available.delta_v_mps) {
    return {.reason_code = "HOPPER_SINGLE_HOP_ENVELOPE_EXCEEDED"};
  }
  return {
      .evidence = SingleHopEnvelopeEvidence{
          .available_delta_v_mps = *available.delta_v_mps,
          .ideal_delta_v_mps = ideal_delta_v,
          .required_delta_v_mps = required_delta_v,
      },
  };
}

}  // namespace lunar::planning::hopper

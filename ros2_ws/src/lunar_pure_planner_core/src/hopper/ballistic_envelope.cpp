#include "hopper/ballistic_envelope.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "hopper/ballistic_kinematics.hpp"

namespace lunar::pure_planning::hopper {
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

struct TimedArc final {
  BallisticArc arc;
  double required_delta_v_mps{};
};

[[nodiscard]] TimedArc AtTime(
    const Vec3 launch,
    const Vec3 landing,
    const Vec3 gravity,
    const double time_s,
    const HopperCapability& capability) noexcept {
  const auto solved = SolveBallisticArc(
      launch, landing, gravity, time_s);
  if (!solved.ok()) {
    return {
        .arc = BallisticArc{
            .flight_time_s = std::numeric_limits<double>::quiet_NaN()},
        .required_delta_v_mps =
            std::numeric_limits<double>::infinity(),
    };
  }
  const double ideal = Norm(solved.arc->launch_velocity_mps) +
      Norm(solved.arc->landing_velocity_mps);
  return {
      .arc = *solved.arc,
      .required_delta_v_mps =
          (1.0 + capability.reachability_delta_v_margin_ratio) * ideal,
  };
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

MinimumSingleHopEnvelopeResult EvaluateMinimumSingleHopEnvelope(
    const Vec3 launch_position_m,
    const Vec3 landing_position_m,
    const Vec3 gravity_mps2,
    const double evidence_scale_m,
    const HopperCapability& capability) noexcept {
  if (!Finite(launch_position_m) || !Finite(landing_position_m) ||
      !Finite(gravity_mps2) || !FinitePositive(evidence_scale_m) ||
      !ValidCapability(capability)) {
    return {
        .reason_code = "HOPPER_BALLISTIC_NUMERICAL_INDETERMINATE"};
  }
  const Vec3 displacement{
      landing_position_m.x - launch_position_m.x,
      landing_position_m.y - launch_position_m.y,
      landing_position_m.z - launch_position_m.z,
  };
  const double gravity = Norm(gravity_mps2);
  const double distance = Norm(displacement);
  if (!FinitePositive(gravity) || !std::isfinite(distance)) {
    return {
        .reason_code = "HOPPER_BALLISTIC_NUMERICAL_INDETERMINATE"};
  }
  double center = std::sqrt(
      2.0 * std::max(distance, evidence_scale_m) / gravity);
  if (!FinitePositive(center)) {
    return {
        .reason_code = "HOPPER_BALLISTIC_NUMERICAL_INDETERMINATE"};
  }
  double left = 0.5 * center;
  double right = 2.0 * center;
  TimedArc left_arc = AtTime(
      launch_position_m, landing_position_m, gravity_mps2, left,
      capability);
  TimedArc center_arc = AtTime(
      launch_position_m, landing_position_m, gravity_mps2, center,
      capability);
  TimedArc right_arc = AtTime(
      launch_position_m, landing_position_m, gravity_mps2, right,
      capability);
  while (left_arc.required_delta_v_mps <
         center_arc.required_delta_v_mps) {
    right = center;
    right_arc = center_arc;
    center = left;
    center_arc = left_arc;
    const double next = 0.5 * left;
    if (!(next > 0.0 && next < left)) {
      return {
          .reason_code = "HOPPER_BALLISTIC_NUMERICAL_INDETERMINATE"};
    }
    left = next;
    left_arc = AtTime(
        launch_position_m, landing_position_m, gravity_mps2, left,
        capability);
  }
  while (right_arc.required_delta_v_mps <
         center_arc.required_delta_v_mps) {
    left = center;
    left_arc = center_arc;
    center = right;
    center_arc = right_arc;
    const double next = 2.0 * right;
    if (!std::isfinite(next) || !(next > right)) {
      return {
          .reason_code = "HOPPER_BALLISTIC_NUMERICAL_INDETERMINATE"};
    }
    right = next;
    right_arc = AtTime(
        launch_position_m, landing_position_m, gravity_mps2, right,
        capability);
  }

  constexpr double inverse_phi = 0.6180339887498948482;
  double x1 = right - inverse_phi * (right - left);
  double x2 = left + inverse_phi * (right - left);
  TimedArc arc1 = AtTime(
      launch_position_m, landing_position_m, gravity_mps2, x1,
      capability);
  TimedArc arc2 = AtTime(
      launch_position_m, landing_position_m, gravity_mps2, x2,
      capability);
  const double numerical_scale =
      std::sqrt(std::numeric_limits<double>::epsilon());
  while (right - left > numerical_scale *
         std::max({1.0, std::abs(left), std::abs(right)})) {
    if (arc1.required_delta_v_mps <= arc2.required_delta_v_mps) {
      right = x2;
      x2 = x1;
      arc2 = arc1;
      x1 = right - inverse_phi * (right - left);
      arc1 = AtTime(
          launch_position_m, landing_position_m, gravity_mps2, x1,
          capability);
    } else {
      left = x1;
      x1 = x2;
      arc1 = arc2;
      x2 = left + inverse_phi * (right - left);
      arc2 = AtTime(
          launch_position_m, landing_position_m, gravity_mps2, x2,
          capability);
    }
  }
  const TimedArc best = arc1.required_delta_v_mps <=
          arc2.required_delta_v_mps
      ? arc1
      : arc2;
  if (!std::isfinite(best.required_delta_v_mps) ||
      !FinitePositive(best.arc.flight_time_s)) {
    return {
        .reason_code = "HOPPER_BALLISTIC_NUMERICAL_INDETERMINATE"};
  }
  const SingleHopEnvelopeResult envelope =
      EvaluateSingleHopEnvelope(best.arc, capability);
  if (!envelope.ok()) {
    return {.reason_code = envelope.reason_code};
  }
  return {
      .evidence = MinimumSingleHopEnvelopeEvidence{
          .arc = best.arc,
          .envelope = *envelope.evidence,
      },
  };
}

}  // namespace lunar::pure_planning::hopper

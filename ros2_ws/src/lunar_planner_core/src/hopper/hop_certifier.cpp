#include "hopper/hop_certifier.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <optional>
#include <string>
#include <utility>

#include "hopper/ballistic_kinematics.hpp"
#include "hopper/flight_tube_certifier.hpp"

namespace lunar::planning::hopper {
namespace {

struct TimedArc final {
  BallisticArc arc;
  double certified_delta_v_mps{};
};

struct MinimumArcResult final {
  std::optional<TimedArc> timed_arc;
  std::string reason_code;
};

[[nodiscard]] bool Finite(const Vec3 value) noexcept {
  return std::isfinite(value.x) && std::isfinite(value.y) &&
      std::isfinite(value.z);
}

[[nodiscard]] double Norm(const Vec3 value) noexcept {
  return std::hypot(std::hypot(value.x, value.y), value.z);
}

[[nodiscard]] TimedArc AtTime(
    const Vec3 launch, const Vec3 landing, const Vec3 gravity,
    const double time_s, const HopperCapability& capability) noexcept {
  const auto solved = SolveBallisticArc(launch, landing, gravity, time_s);
  if (!solved.ok()) {
    return {
        .arc = BallisticArc{.flight_time_s =
            std::numeric_limits<double>::quiet_NaN()},
        .certified_delta_v_mps =
            std::numeric_limits<double>::infinity(),
    };
  }
  const double ideal = Norm(solved.arc->launch_velocity_mps) +
      Norm(solved.arc->landing_velocity_mps);
  return {
      .arc = *solved.arc,
      .certified_delta_v_mps =
          (1.0 + capability.reachability_delta_v_margin_ratio) * ideal,
  };
}

[[nodiscard]] MinimumArcResult MinimumDeltaVArc(
    const SingleHopCertificationProblem& problem) noexcept {
  const Vec3 displacement{
      problem.landing_position_m.x - problem.launch_position_m.x,
      problem.landing_position_m.y - problem.launch_position_m.y,
      problem.landing_position_m.z - problem.launch_position_m.z,
  };
  const double gravity = Norm(problem.gravity_mps2);
  const double distance = Norm(displacement);
  if (!std::isfinite(gravity) || gravity <= 0.0 ||
      !std::isfinite(distance)) {
    return {.reason_code = "HOPPER_BALLISTIC_NUMERICAL_INDETERMINATE"};
  }
  const double evidence_scale = problem.flight_map == nullptr
      ? 1.0
      : problem.flight_map->resolution_m();
  double center = std::sqrt(
      2.0 * std::max(distance, evidence_scale) / gravity);
  if (!std::isfinite(center) || center <= 0.0) {
    return {.reason_code = "HOPPER_BALLISTIC_NUMERICAL_INDETERMINATE"};
  }
  double left = 0.5 * center;
  double right = 2.0 * center;
  TimedArc left_arc = AtTime(
      problem.launch_position_m, problem.landing_position_m,
      problem.gravity_mps2, left, *problem.capability);
  TimedArc center_arc = AtTime(
      problem.launch_position_m, problem.landing_position_m,
      problem.gravity_mps2, center, *problem.capability);
  TimedArc right_arc = AtTime(
      problem.launch_position_m, problem.landing_position_m,
      problem.gravity_mps2, right, *problem.capability);
  while (left_arc.certified_delta_v_mps <
         center_arc.certified_delta_v_mps) {
    right = center;
    right_arc = center_arc;
    center = left;
    center_arc = left_arc;
    const double next = 0.5 * left;
    if (!(next > 0.0 && next < left)) {
      return {.reason_code = "HOPPER_BALLISTIC_NUMERICAL_INDETERMINATE"};
    }
    left = next;
    left_arc = AtTime(
        problem.launch_position_m, problem.landing_position_m,
        problem.gravity_mps2, left, *problem.capability);
  }
  while (right_arc.certified_delta_v_mps <
         center_arc.certified_delta_v_mps) {
    left = center;
    left_arc = center_arc;
    center = right;
    center_arc = right_arc;
    const double next = 2.0 * right;
    if (!std::isfinite(next) || !(next > right)) {
      return {.reason_code = "HOPPER_BALLISTIC_NUMERICAL_INDETERMINATE"};
    }
    right = next;
    right_arc = AtTime(
        problem.launch_position_m, problem.landing_position_m,
        problem.gravity_mps2, right, *problem.capability);
  }

  constexpr double inverse_phi = 0.6180339887498948482;
  double x1 = right - inverse_phi * (right - left);
  double x2 = left + inverse_phi * (right - left);
  TimedArc arc1 = AtTime(
      problem.launch_position_m, problem.landing_position_m,
      problem.gravity_mps2, x1, *problem.capability);
  TimedArc arc2 = AtTime(
      problem.launch_position_m, problem.landing_position_m,
      problem.gravity_mps2, x2, *problem.capability);
  const double numerical_scale = std::sqrt(
      std::numeric_limits<double>::epsilon());
  while (right - left > numerical_scale *
         std::max({1.0, std::abs(left), std::abs(right)})) {
    if (arc1.certified_delta_v_mps <= arc2.certified_delta_v_mps) {
      right = x2;
      x2 = x1;
      arc2 = arc1;
      x1 = right - inverse_phi * (right - left);
      arc1 = AtTime(
          problem.launch_position_m, problem.landing_position_m,
          problem.gravity_mps2, x1, *problem.capability);
    } else {
      left = x1;
      x1 = x2;
      arc1 = arc2;
      x2 = left + inverse_phi * (right - left);
      arc2 = AtTime(
          problem.launch_position_m, problem.landing_position_m,
          problem.gravity_mps2, x2, *problem.capability);
    }
  }
  TimedArc best = arc1.certified_delta_v_mps <=
          arc2.certified_delta_v_mps
      ? arc1
      : arc2;
  if (!std::isfinite(best.certified_delta_v_mps) ||
      !std::isfinite(best.arc.flight_time_s)) {
    return {.reason_code = "HOPPER_BALLISTIC_NUMERICAL_INDETERMINATE"};
  }
  return {.timed_arc = best};
}

[[nodiscard]] bool NumericalTubeFailure(
    const FlightTubeCertificationResult& tube) noexcept {
  return tube.reason_code == "HOPPER_FLIGHT_TUBE_INPUT_INVALID" ||
      tube.reason_code.find("NUMERICAL") != std::string::npos;
}

[[nodiscard]] SingleHopCertificationResult Certified(
    const BallisticArc& arc, const SingleHopEnvelopeEvidence& envelope,
    FlightTubeCertificationResult tube, const std::size_t examined) {
  return {
      .status = HopCertificationStatus::kCertified,
      .certification = CertifiedSingleHop{
          .arc = arc,
          .envelope = envelope,
          .flight_tube = std::move(tube),
      },
      .examined_intervals = examined,
  };
}

}  // namespace

SingleHopCertificationResult CertifySingleHop(
    const SingleHopCertificationProblem& problem) {
  if (problem.stop_token.stop_requested()) {
    return {
        .status = HopCertificationStatus::kCanceled,
        .reason_code = "REQUEST_CANCELED",
    };
  }
  if (problem.flight_map == nullptr || problem.capability == nullptr ||
      problem.map_safety == nullptr ||
      !Finite(problem.launch_position_m) ||
      !Finite(problem.landing_position_m) ||
      !Finite(problem.gravity_mps2)) {
    return {
        .status = HopCertificationStatus::kInvalid,
        .reason_code = "HOPPER_CAPABILITY_INVALID",
    };
  }
  const AvailableSingleHopDeltaVResult available =
      AvailableSingleHopDeltaV(*problem.capability);
  if (!available.ok()) {
    return {
        .status = HopCertificationStatus::kInvalid,
        .reason_code = available.reason_code,
    };
  }
  const MinimumArcResult minimum = MinimumDeltaVArc(problem);
  if (!minimum.timed_arc.has_value()) {
    return {
        .status = HopCertificationStatus::kNumericalIndeterminate,
        .reason_code = minimum.reason_code,
    };
  }
  if (minimum.timed_arc->certified_delta_v_mps >
      *available.delta_v_mps) {
    return {
        .status = HopCertificationStatus::kInfeasible,
        .reason_code = "HOPPER_SINGLE_HOP_ENVELOPE_EXCEEDED",
    };
  }

  const double minimum_time = minimum.timed_arc->arc.flight_time_s;
  double feasible_upper = minimum_time;
  double infeasible_upper = 2.0 * feasible_upper;
  while (true) {
    if (!std::isfinite(infeasible_upper) ||
        !(infeasible_upper > feasible_upper)) {
      return {
          .status = HopCertificationStatus::kNumericalIndeterminate,
          .reason_code = "HOPPER_BALLISTIC_NUMERICAL_INDETERMINATE",
      };
    }
    const TimedArc candidate = AtTime(
        problem.launch_position_m, problem.landing_position_m,
        problem.gravity_mps2, infeasible_upper, *problem.capability);
    if (candidate.certified_delta_v_mps > *available.delta_v_mps) {
      break;
    }
    feasible_upper = infeasible_upper;
    infeasible_upper *= 2.0;
  }
  const double numerical_scale = std::sqrt(
      std::numeric_limits<double>::epsilon());
  while (infeasible_upper - feasible_upper > numerical_scale *
         std::max({1.0, feasible_upper, infeasible_upper})) {
    const double midpoint = std::midpoint(feasible_upper, infeasible_upper);
    if (!(midpoint > feasible_upper && midpoint < infeasible_upper)) {
      break;
    }
    const TimedArc candidate = AtTime(
        problem.launch_position_m, problem.landing_position_m,
        problem.gravity_mps2, midpoint, *problem.capability);
    if (candidate.certified_delta_v_mps <= *available.delta_v_mps) {
      feasible_upper = midpoint;
    } else {
      infeasible_upper = midpoint;
    }
  }

  std::size_t examined = 1U;
  SingleHopEnvelopeResult minimum_envelope = EvaluateSingleHopEnvelope(
      minimum.timed_arc->arc, *problem.capability);
  if (!minimum_envelope.ok()) {
    return {
        .status = minimum_envelope.reason_code ==
                "HOPPER_SINGLE_HOP_ENVELOPE_EXCEEDED"
            ? HopCertificationStatus::kInfeasible
            : HopCertificationStatus::kNumericalIndeterminate,
        .examined_intervals = examined,
        .reason_code = minimum_envelope.reason_code,
    };
  }
  FlightTubeCertificationResult minimum_tube = CertifyFlightTube(
      minimum.timed_arc->arc, *problem.flight_map, *problem.capability,
      *problem.map_safety, problem.stop_token);
  if (minimum_tube.canceled) {
    return {
        .status = HopCertificationStatus::kCanceled,
        .examined_intervals = examined,
        .reason_code = "REQUEST_CANCELED",
    };
  }
  if (minimum_tube.certified) {
    return Certified(
        minimum.timed_arc->arc, *minimum_envelope.evidence,
        std::move(minimum_tube), examined);
  }
  if (NumericalTubeFailure(minimum_tube)) {
    return {
        .status = HopCertificationStatus::kNumericalIndeterminate,
        .examined_intervals = examined,
        .reason_code = minimum_tube.reason_code,
    };
  }

  const TimedArc upper_arc = AtTime(
      problem.launch_position_m, problem.landing_position_m,
      problem.gravity_mps2, feasible_upper, *problem.capability);
  SingleHopEnvelopeResult upper_envelope = EvaluateSingleHopEnvelope(
      upper_arc.arc, *problem.capability);
  if (!upper_envelope.ok()) {
    return {
        .status = HopCertificationStatus::kNumericalIndeterminate,
        .examined_intervals = examined,
        .reason_code = "HOPPER_BALLISTIC_NUMERICAL_INDETERMINATE",
    };
  }
  ++examined;
  FlightTubeCertificationResult upper_tube = CertifyFlightTube(
      upper_arc.arc, *problem.flight_map, *problem.capability,
      *problem.map_safety, problem.stop_token);
  if (upper_tube.canceled) {
    return {
        .status = HopCertificationStatus::kCanceled,
        .examined_intervals = examined,
        .reason_code = "REQUEST_CANCELED",
    };
  }
  if (!upper_tube.certified) {
    return {
        .status = NumericalTubeFailure(upper_tube)
            ? HopCertificationStatus::kNumericalIndeterminate
            : HopCertificationStatus::kInfeasible,
        .examined_intervals = examined,
        .reason_code = NumericalTubeFailure(upper_tube)
            ? upper_tube.reason_code
            : "HOPPER_ALL_FLIGHT_TUBES_BLOCKED",
    };
  }

  double blocked_time = minimum_time;
  double safe_time = feasible_upper;
  TimedArc safe_arc = upper_arc;
  SingleHopEnvelopeResult safe_envelope = std::move(upper_envelope);
  FlightTubeCertificationResult safe_tube = std::move(upper_tube);
  const double gravity_norm = Norm(problem.gravity_mps2);
  while (gravity_norm *
             std::abs(safe_time * safe_time - blocked_time * blocked_time) /
             8.0 >
         0.25 * problem.flight_map->resolution_m()) {
    if (problem.stop_token.stop_requested()) {
      return {
          .status = HopCertificationStatus::kCanceled,
          .examined_intervals = examined,
          .reason_code = "REQUEST_CANCELED",
      };
    }
    const double midpoint = std::midpoint(blocked_time, safe_time);
    if (!(midpoint > blocked_time && midpoint < safe_time)) {
      return {
          .status = HopCertificationStatus::kNumericalIndeterminate,
          .examined_intervals = examined,
          .reason_code = "HOPPER_FLIGHT_TUBE_NUMERICAL_INDETERMINATE",
      };
    }
    const TimedArc candidate = AtTime(
        problem.launch_position_m, problem.landing_position_m,
        problem.gravity_mps2, midpoint, *problem.capability);
    SingleHopEnvelopeResult candidate_envelope = EvaluateSingleHopEnvelope(
        candidate.arc, *problem.capability);
    if (!candidate_envelope.ok()) {
      return {
          .status = HopCertificationStatus::kNumericalIndeterminate,
          .examined_intervals = examined,
          .reason_code = "HOPPER_BALLISTIC_NUMERICAL_INDETERMINATE",
      };
    }
    ++examined;
    FlightTubeCertificationResult candidate_tube = CertifyFlightTube(
        candidate.arc, *problem.flight_map, *problem.capability,
        *problem.map_safety, problem.stop_token);
    if (candidate_tube.canceled) {
      return {
          .status = HopCertificationStatus::kCanceled,
          .examined_intervals = examined,
          .reason_code = "REQUEST_CANCELED",
      };
    }
    if (candidate_tube.certified) {
      safe_time = midpoint;
      safe_arc = candidate;
      safe_envelope = std::move(candidate_envelope);
      safe_tube = std::move(candidate_tube);
    } else if (NumericalTubeFailure(candidate_tube)) {
      return {
          .status = HopCertificationStatus::kNumericalIndeterminate,
          .examined_intervals = examined,
          .reason_code = candidate_tube.reason_code,
      };
    } else {
      blocked_time = midpoint;
    }
  }
  return Certified(
      safe_arc.arc, *safe_envelope.evidence, std::move(safe_tube), examined);
}

}  // namespace lunar::planning::hopper

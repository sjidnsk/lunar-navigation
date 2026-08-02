#include "lunar_path_planner/v3/hopper/ballistic_time_solver.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <format>
#include <limits>
#include <tuple>
#include <utility>

namespace lunar::planning::v3 {
namespace {

[[nodiscard]] double Seconds(const DurationNanoseconds duration) {
  return std::chrono::duration<double>(duration.value).count();
}

[[nodiscard]] bool HardFeasible(
    const NominalBallisticArc& arc,
    const HopperCapabilityView& capability) {
  const double launch_speed = arc.launch_velocity_mps.norm();
  const double landing_speed = arc.landing_velocity_mps.norm();
  const double impulse =
      capability.actuator_or_impulse_profile.platform_mass_kg *
      launch_speed;
  const double gravity_norm = arc.gravity_mps2.norm();
  if (!std::isfinite(launch_speed) ||
      !std::isfinite(landing_speed) ||
      !std::isfinite(impulse) || gravity_norm <= 0.0) {
    return false;
  }
  const Eigen::Vector3d up = -arc.gravity_mps2 / gravity_norm;
  const double upward_landing_speed =
      arc.landing_velocity_mps.dot(up);
  return launch_speed <=
             capability.launch_limits.maximum_launch_speed_mps +
                 1.0e-12 &&
         impulse <= capability.launch_limits
                            .maximum_launch_impulse_newton_seconds +
                        1.0e-12 &&
         landing_speed <=
             capability.launch_limits.maximum_landing_speed_mps +
                 1.0e-12 &&
         upward_landing_speed <=
             -capability.launch_limits
                  .minimum_downward_impact_speed_mps +
                 1.0e-12;
}

}  // namespace

bool ballistic_candidate_order(
    const BallisticCandidate& lhs,
    const BallisticCandidate& rhs) {
  return std::tuple{
             lhs.expected_execution_time_s,
             lhs.estimated_energy_j,
             -lhs.landing_margin_m,
             -lhs.nominal_minimum_clearance_m,
             lhs.candidate_id} <
         std::tuple{
             rhs.expected_execution_time_s,
             rhs.estimated_energy_j,
             -rhs.landing_margin_m,
             -rhs.nominal_minimum_clearance_m,
             rhs.candidate_id};
}

BallisticTimeSolver::BallisticTimeSolver(
    HopperCapabilityView capability,
    HopperPlannerLimits limits)
    : capability_(std::move(capability)),
      limits_(std::move(limits)) {}

BallisticSolveResult BallisticTimeSolver::solve(
    const FlightTimeSearchInput& input) const {
  BallisticSolveResult result{};
  result.diagnostics.aim_point_count = input.aim_points.size();
  if (input.aim_points.empty() ||
      !input.launch_position_m.allFinite() ||
      input.frame_id.empty() ||
      !std::isfinite(input.minimum_attitude_time_s) ||
      input.minimum_attitude_time_s < 0.0 ||
      !std::isfinite(input.launch_preparation_time_s) ||
      input.launch_preparation_time_s < 0.0 ||
      !std::isfinite(input.landing_settle_time_s) ||
      input.landing_settle_time_s < 0.0) {
    result.diagnostics.termination_reason = "invalid_input";
    return result;
  }

  const double minimum_time = std::max(
      Seconds(capability_.launch_limits.minimum_flight_time),
      input.minimum_attitude_time_s);
  const double maximum_time =
      Seconds(capability_.launch_limits.maximum_flight_time);
  if (!std::isfinite(minimum_time) ||
      !std::isfinite(maximum_time) ||
      minimum_time <= 0.0 || maximum_time < minimum_time) {
    result.diagnostics.termination_reason =
        "no_feasible_flight_time";
    return result;
  }

  const std::size_t subdivision_depth =
      std::min<std::size_t>(
          limits_.maximum_interval_subdivision_depth, 12U);
  const std::size_t interval_count =
      std::size_t{1U} << subdivision_depth;
  result.diagnostics.interval_count =
      input.aim_points.size() * interval_count;
  result.diagnostics.subdivision_count =
      input.aim_points.size() *
      (interval_count > 0U ? interval_count - 1U : 0U);

  for (const AimPointCandidate& aim : input.aim_points) {
    for (std::size_t index = 0U; index <= interval_count;
         ++index) {
      const double fraction =
          static_cast<double>(index) /
          static_cast<double>(interval_count);
      const double flight_time =
          minimum_time +
          fraction * (maximum_time - minimum_time);
      const auto arc_result = make_nominal_ballistic_arc(
          input.launch_position_m, input.frame_id, aim,
          input.gravity, flight_time);
      if (!IsOk(arc_result)) {
        continue;
      }
      const auto& arc =
          std::get<NominalBallisticArc>(arc_result);
      if (!HardFeasible(arc, capability_)) {
        continue;
      }
      const double energy =
          0.5 *
          capability_.actuator_or_impulse_profile
              .platform_mass_kg *
          arc.launch_velocity_mps.squaredNorm();
      result.candidates.push_back(
          {
              .candidate_id = std::format(
                  "{}-t{}", aim.aim_point_id,
                  std::llround(flight_time * 1.0e9)),
              .arc = arc,
              .expected_execution_time_s =
                  input.launch_preparation_time_s +
                  flight_time + input.landing_settle_time_s,
              .estimated_energy_j = energy,
              .landing_margin_m =
                  aim.minimum_boundary_distance_m,
              .nominal_minimum_clearance_m = 0.0,
          });
      break;
    }
  }
  std::sort(
      result.candidates.begin(), result.candidates.end(),
      ballistic_candidate_order);
  result.diagnostics.termination_reason =
      result.candidates.empty() ? "no_feasible_flight_time"
                                : "completed";
  return result;
}

}  // namespace lunar::planning::v3

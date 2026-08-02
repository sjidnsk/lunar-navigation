#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "lunar_path_planner/v3/hopper/ballistic_kinematics.hpp"

namespace lunar::planning::v3 {

struct FlightTimeSearchInput final {
  Eigen::Vector3d launch_position_m{Eigen::Vector3d::Zero()};
  FrameId frame_id;
  std::vector<AimPointCandidate> aim_points;
  GravityModelView gravity;
  double minimum_attitude_time_s{};
  double launch_preparation_time_s{};
  double landing_settle_time_s{};
};

struct BallisticCandidate final {
  std::string candidate_id;
  NominalBallisticArc arc;
  double expected_execution_time_s{};
  double estimated_energy_j{};
  double landing_margin_m{};
  double nominal_minimum_clearance_m{};
};

struct BallisticSolveDiagnostics final {
  std::size_t aim_point_count{};
  std::size_t interval_count{};
  std::size_t subdivision_count{};
  std::size_t root_iteration_count{};
  std::string termination_reason;
};

struct BallisticSolveResult final {
  std::vector<BallisticCandidate> candidates;
  BallisticSolveDiagnostics diagnostics;
};

[[nodiscard]] bool ballistic_candidate_order(
    const BallisticCandidate& lhs,
    const BallisticCandidate& rhs);

class BallisticTimeSolver final {
 public:
  BallisticTimeSolver(
      HopperCapabilityView capability,
      HopperPlannerLimits limits);

  [[nodiscard]] BallisticSolveResult solve(
      const FlightTimeSearchInput& input) const;

 private:
  HopperCapabilityView capability_;
  HopperPlannerLimits limits_;
};

}  // namespace lunar::planning::v3

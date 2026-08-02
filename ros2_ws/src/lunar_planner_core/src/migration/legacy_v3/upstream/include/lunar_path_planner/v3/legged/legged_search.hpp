#pragma once

#include <span>
#include <string>
#include <vector>

#include "lunar_path_planner/v3/legged/legged_lattice.hpp"
#include "lunar_path_planner/v3/search/ara_star.hpp"
#include "lunar_path_planner/v3/search/candidate_ranker.hpp"

namespace lunar::planning::v3 {

struct LeggedPlanningProblem final {
  const SafeProjection& projection;
  LeggedLatticeState start;
  std::span<const TerminalCandidate> terminals;
  GridConfig grid;
  DurationNanoseconds time_equivalence_tolerance;
};

struct LeggedDiscretePlan final {
  std::string stable_candidate_id;
  GridConfig grid;
  Vec2 grid_origin_m;
  std::vector<LeggedLatticeState> states;
  std::vector<LeggedLatticeEdge> edges;
  DurationNanoseconds estimated_execution_time;
  SecondaryCostVector secondary_costs;
  PoseXyzYaw terminal_pose;
  double target_linear_velocity_mps{};
  double target_yaw_rate_radps{};
  double achieved_epsilon{};
  bool resource_limit_hit{};
};

[[nodiscard]] Result<LeggedDiscretePlan> PlanLeggedDiscrete(
    const LeggedPlanningProblem& problem,
    const BodyMotionPrimitiveCatalog& catalog,
    const AraStarConfig& config);

}  // namespace lunar::planning::v3

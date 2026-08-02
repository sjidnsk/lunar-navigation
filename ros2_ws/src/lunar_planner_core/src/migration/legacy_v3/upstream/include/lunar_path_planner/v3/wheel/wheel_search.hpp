#pragma once

#include <string>
#include <vector>

#include "lunar_path_planner/v3/search/ara_star.hpp"
#include "lunar_path_planner/v3/search/candidate_ranker.hpp"
#include "lunar_path_planner/v3/wheel/wheel_lattice.hpp"

namespace lunar::planning::v3 {

struct WheelDiscreteSegment final {
  WheelMotionMode mode{WheelMotionMode::kStart};
  std::vector<WheelLatticeEdge> edges;
  DurationNanoseconds expected_time;
  SecondaryCostVector secondary_costs;
};

struct WheelSearchDiagnostics final {
  SearchStatus status{SearchStatus::kNoPath};
  double final_epsilon{};
  std::size_t expanded_states{};
  std::size_t generated_states{};
  std::size_t candidate_count{};
  bool resource_limit_hit{};
  std::string termination_reason;
};

struct WheelDiscretePlan final {
  std::string selected_candidate_id;
  std::vector<WheelDiscreteSegment> segments;
  DurationNanoseconds expected_time;
  SecondaryCostVector total_secondary_costs;
  SafeStopAnchor safe_stop_anchor;
  WheelSearchDiagnostics diagnostics;
};

struct WheelPlanningProblem final {
  WheelPlanningProblem(
      const SafeProjection& projection_value,
      const WheeledOrLeggedState& current_state_value,
      const ResolvedTerminalSet& terminals_value,
      const SafetyCapabilityProfile& capability_value,
      const PlannerAlgorithmConfig& algorithm_value) noexcept
      : projection(projection_value),
        current_state(current_state_value),
        terminals(terminals_value),
        capability(capability_value),
        algorithm(algorithm_value) {}

  const SafeProjection& projection;
  const WheeledOrLeggedState& current_state;
  const ResolvedTerminalSet& terminals;
  const SafetyCapabilityProfile& capability;
  const PlannerAlgorithmConfig& algorithm;
};

[[nodiscard]] Result<WheelDiscretePlan> PlanWheelDiscrete(
    const WheelPlanningProblem& problem,
    const WheelPrimitiveCatalog& primitives,
    const AraStarConfig& search_config);

}  // namespace lunar::planning::v3

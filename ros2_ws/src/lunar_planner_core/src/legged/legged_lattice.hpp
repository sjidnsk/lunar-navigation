#pragma once

#include <optional>
#include <stop_token>
#include <string>
#include <vector>

#include "hierarchical/local_planning_problem.hpp"
#include "legged/legged_types.hpp"
#include "lunar_planner_core/types/planner_io.hpp"
#include "shared/ara_star.hpp"
#include "shared/primitive_reachability_graph.hpp"
#include "shared/safe_projection.hpp"

namespace lunar::planning::legged {

struct LeggedLatticeGraph final {
  shared::AraStarProblem search_problem;
  std::vector<LeggedLatticeState> states;
  std::vector<LeggedPose> state_poses;
  std::vector<LeggedTransition> transitions;
  LeggedPose true_start_pose;
  Interval true_start_body_z_m;
};

enum class LeggedLatticeStatus {
  kReady,
  kCanceled,
  kResourceExhausted,
  kInvalidRequest,
  kNoPath,
};

struct LeggedLatticeBuildResult final {
  LeggedLatticeStatus status{LeggedLatticeStatus::kInvalidRequest};
  std::optional<LeggedLatticeGraph> graph;
  std::string reason_code;

  [[nodiscard]] bool ok() const noexcept {
    return status == LeggedLatticeStatus::kReady && graph.has_value();
  }
};

struct LeggedLatticeSearchResult final {
  LeggedLatticeStatus status{LeggedLatticeStatus::kInvalidRequest};
  std::optional<LeggedDiscretePlan> plan;
  std::string reason_code;

  [[nodiscard]] bool ok() const noexcept {
    return status == LeggedLatticeStatus::kReady && plan.has_value();
  }
};

[[nodiscard]] bool GoalContainsBodyPose(
    const GoalRegion& goal, const LeggedPose& pose) noexcept;

[[nodiscard]] LeggedLatticeBuildResult BuildLeggedLattice(
    const LeggedState& current_state,
    const GoalRegion& goal,
    const shared::SafeProjection& projection,
    const hierarchical::LocalSearchDomain& search_domain,
    const LeggedCapability& capability,
    const PlannerConfig& config,
    std::stop_token stop_token);

// Performs deterministic A* directly over lazily generated body primitives.
// Search state and terrain evidence are confined to this request.
[[nodiscard]] LeggedLatticeSearchResult SearchLeggedLattice(
    const LeggedState& current_state,
    const GoalRegion& goal,
    const shared::SafeProjection& projection,
    const hierarchical::LocalSearchDomain& search_domain,
    const LeggedCapability& capability,
    const PlannerConfig& config,
    std::stop_token stop_token);

[[nodiscard]] shared::PrimitiveGraphBuildResult BuildLeggedPrimitiveGraph(
    const LeggedState& current_state,
    const shared::SafeProjection& projection,
    const LeggedCapability& capability,
    const PlannerConfig& config,
    std::stop_token stop_token);

[[nodiscard]] std::optional<LeggedDiscretePlan> ResolveLeggedPlan(
    const LeggedLatticeGraph& graph,
    const shared::AraStarResult& search_result);

}  // namespace lunar::planning::legged

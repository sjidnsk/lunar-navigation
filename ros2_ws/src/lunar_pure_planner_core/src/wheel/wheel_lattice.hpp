#pragma once

#include <optional>
#include <span>
#include <stop_token>
#include <string>

#include "hierarchical/local_planning_problem.hpp"
#include "lunar_pure_planner_core/types/planner_io.hpp"
#include "shared/safe_projection.hpp"
#include "wheel/wheel_types.hpp"

namespace lunar::pure_planning::wheel {

enum class WheelLatticeStatus {
  kSolved,
  kNoPath,
  kCanceled,
  kResourceExhausted,
  kInvalidRequest,
};

struct WheelLatticeSearchResult final {
  WheelLatticeStatus status{WheelLatticeStatus::kInvalidRequest};
  std::optional<WheelDiscretePlan> plan;
  std::optional<std::size_t> selected_goal_index;
  std::string reason_code;

  [[nodiscard]] bool ok() const noexcept {
    return status == WheelLatticeStatus::kSolved && plan.has_value();
  }
};

[[nodiscard]] bool GoalContainsPose(
    const GoalRegion& goal, const WheelPose& pose) noexcept;

// Performs deterministic A* directly over lazily generated motion primitives.
// Continuous poses are retained in nodes; (cell, yaw bin, motion mode) is only
// the finite search key.
[[nodiscard]] WheelLatticeSearchResult SearchWheelLattice(
    const WheeledState& current_state,
    const GoalRegion& goal,
    const shared::SafeProjection& projection,
    const WheeledCapability& capability,
    const PlannerConfig& config,
    std::stop_token stop_token);

// Searches the farthest ranked goal once. If that goal is unreachable after
// the finite lattice is exhausted, the already explored tree is reused to
// connect the first reachable nearer goal. Goals must be ordered far-to-near.
[[nodiscard]] WheelLatticeSearchResult SearchWheelLatticeRanked(
    const WheeledState& current_state,
    std::span<const GoalRegion> ranked_goals,
    const shared::SafeProjection& physical_projection,
    const hierarchical::LocalSearchDomain& search_domain,
    const WheeledCapability& capability,
    const PlannerConfig& config,
    std::stop_token stop_token);

}  // namespace lunar::pure_planning::wheel

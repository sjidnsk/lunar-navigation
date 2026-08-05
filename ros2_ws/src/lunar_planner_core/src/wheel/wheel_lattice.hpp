#pragma once

#include <optional>
#include <stop_token>
#include <string>
#include <vector>

#include "lunar_planner_core/types/planner_io.hpp"
#include "shared/ara_star.hpp"
#include "shared/safe_projection.hpp"
#include "wheel/wheel_types.hpp"

namespace lunar::planning::wheel {

struct WheelLatticeGraph final {
  shared::AraStarProblem search_problem;
  std::vector<WheelLatticeState> states;
  std::vector<WheelTransition> transitions;
  WheelPose true_start_pose;
};

enum class WheelLatticeStatus {
  kReady,
  kCanceled,
  kResourceExhausted,
  kInvalidRequest,
};

struct WheelLatticeBuildResult final {
  WheelLatticeStatus status{WheelLatticeStatus::kInvalidRequest};
  std::optional<WheelLatticeGraph> graph;
  std::string reason_code;

  [[nodiscard]] bool ok() const noexcept {
    return status == WheelLatticeStatus::kReady && graph.has_value();
  }
};

[[nodiscard]] bool GoalContainsPose(
    const GoalRegion& goal, const WheelPose& pose) noexcept;

[[nodiscard]] WheelLatticeBuildResult BuildWheelLattice(
    const WheeledState& current_state,
    const GoalRegion& goal,
    const shared::SafeProjection& projection,
    const WheeledCapability& capability,
    const PlannerConfig& config,
    std::stop_token stop_token);

[[nodiscard]] std::optional<WheelDiscretePlan> ResolveWheelPlan(
    const WheelLatticeGraph& graph,
    const shared::AraStarResult& search_result);

}  // namespace lunar::planning::wheel

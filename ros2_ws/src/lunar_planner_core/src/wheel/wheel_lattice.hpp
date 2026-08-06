#pragma once

#include <optional>
#include <stop_token>
#include <string>

#include "lunar_planner_core/types/planner_io.hpp"
#include "shared/safe_projection.hpp"
#include "wheel/wheel_types.hpp"

namespace lunar::planning::wheel {

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

}  // namespace lunar::planning::wheel

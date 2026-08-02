#pragma once

#include <optional>
#include <string>
#include <vector>

#include "lunar_path_planner/v3/contracts/planning_request.hpp"
#include "lunar_path_planner/v3/map/safe_projection.hpp"

namespace lunar::planning::v3 {

enum class TerminalKind {
  kGoal,
  kSafeFrontier,
  kGoalInfeasible,
  kNoKnownSafeRoute,
};

struct TerminalCandidate final {
  std::string stable_id;
  Cell cell;
  Vec3 position_m;
  std::optional<CircularYawInterval> yaw_interval;
  bool requires_zero_speed{};
  double clearance_m{};
};

struct UnresolvedTailPreview final {
  std::vector<Vec3> intent_polyline_m;
  bool executable{false};
  std::string reason_code;
};

struct ResolvedTerminalSet final {
  TerminalKind kind{TerminalKind::kNoKnownSafeRoute};
  std::vector<TerminalCandidate> candidates;
  std::optional<UnresolvedTailPreview> unresolved_tail;
  std::string reason_code;
};

struct TerminalResolutionRequest final {
  const SafeProjection& projection;
  const GoalRegion& goal_region;
  Cell start_cell;
  const SafetyCapabilityProfile& capability;
  const PlannerAlgorithmConfig& algorithm_config;
};

[[nodiscard]] Result<ResolvedTerminalSet> ResolveTerminal(
    const TerminalResolutionRequest& request);

}  // namespace lunar::planning::v3

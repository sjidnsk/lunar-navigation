#pragma once

#include <chrono>
#include <cstdint>
#include <optional>
#include <stop_token>
#include <string>
#include <variant>
#include <vector>

#include "lunar_planner_core/types/execution_context.hpp"
#include "lunar_planner_core/types/goal.hpp"
#include "lunar_planner_core/types/motion_reference.hpp"
#include "lunar_planner_core/types/planner_config.hpp"
#include "lunar_planner_core/types/platform_capability.hpp"
#include "lunar_planner_core/types/world_snapshot.hpp"

namespace lunar::planning {

enum class PlanningOutcome : std::uint8_t {
  kNewReferenceAvailable = 0,
  kSafeFrontierReferenceAvailable = 1,
  kNoKnownSafeRoute = 2,
  kGoalInfeasible = 3,
  kInvalidRequest = 4,
  kStaleInput = 5,
  kNumericalFailure = 6,
  kResourceExhausted = 7,
  kActiveReferenceInvalidated = 8,
  kCanceled = 9,
};

enum class ExecutionDirective : std::uint8_t {
  kActivateNewReference = 0,
  kContinueActiveReference = 1,
  kHoldPosition = 2,
  kContinueCommittedHop = 3,
  kNoSafeReference = 4,
};

struct WheeledState final {
  Pose3 pose;
  Twist3 velocity;
};

struct LeggedState final {
  Pose3 body_pose;
  Twist3 body_velocity;
};

struct HopperState final {
  Pose3 pose;
  Twist3 velocity;
};

using PlatformState = std::variant<WheeledState, LeggedState, HopperState>;

struct PlannerInput final {
  std::string request_id;
  TimePoint state_time;
  PlatformState current_state;
  GoalRegion goal;
  WorldSnapshot world;
  PlatformCapability capability;
  PlannerConfig config;
  std::optional<ExecutionContext> previous_execution;
  std::stop_token stop_token;
};

struct PlannerDiagnostics final {
  std::string planner_name{"cpp_v3"};
  std::chrono::nanoseconds elapsed{};
  std::uint64_t expanded_states{};
  std::optional<double> best_cost;
  std::vector<std::string> warning_codes;
};

struct PlannerOutput final {
  PlanningOutcome outcome{PlanningOutcome::kInvalidRequest};
  ExecutionDirective directive{ExecutionDirective::kNoSafeReference};
  std::string reason_code;
  std::optional<MotionReference> reference;
  PlannerDiagnostics diagnostics;
};

}  // namespace lunar::planning

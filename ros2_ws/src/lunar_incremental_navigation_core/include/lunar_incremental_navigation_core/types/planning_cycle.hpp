#pragma once

#include <array>
#include <cmath>
#include <compare>
#include <cstdint>
#include <string>

namespace lunar::incremental_navigation {

struct SessionId final {
  std::array<std::uint8_t, 16U> bytes{};

  auto operator<=>(const SessionId&) const = default;
};

enum class SessionState : std::uint8_t {
  kPlanning = 0,
  kExecuting = 1,
  kReplanning = 2,
};

enum class SessionOutcome : std::uint8_t {
  kGoalReached = 0,
  kNoPath = 1,
  kInvalidGoal = 2,
  kMapUnavailable = 3,
  kTimeout = 4,
  kCanceled = 5,
  kInternalError = 6,
};

enum class GuidanceStatus : std::uint8_t {
  kAvailable = 0,
  kUnavailable = 1,
  kNoRoute = 2,
  kTimeout = 3,
};

struct NavigateToPoseGoal final {
  double target_x_m{};
  double target_y_m{};
  bool has_target_yaw{false};
  double target_yaw_rad{};
};

using FinalGoal = NavigateToPoseGoal;

struct NavigateToPoseFeedback final {
  SessionState session_state{SessionState::kPlanning};
  std::uint64_t planning_cycle{};
  std::uint64_t active_segment_revision{};
  std::string reason_code;
};

struct NavigateToPoseResult final {
  SessionOutcome outcome{SessionOutcome::kInternalError};
  std::string reason_code;
  std::uint64_t last_segment_revision{};
};

[[nodiscard]] inline bool IsValidFinalGoal(const FinalGoal& goal) noexcept {
  return std::isfinite(goal.target_x_m) && std::isfinite(goal.target_y_m) &&
         (!goal.has_target_yaw || std::isfinite(goal.target_yaw_rad));
}

}  // namespace lunar::incremental_navigation

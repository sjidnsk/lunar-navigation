#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include <lunar_planner_core/types/world_snapshot.hpp>

namespace lunar::goal_coordinator {

inline constexpr double kDefaultPositionToleranceM = 0.5;
inline constexpr double kDefaultYawToleranceRad = 0.2617993877991494;

enum class CoordinatorState : std::uint8_t {
  kIdle,
  kWaitingForMap,
  kReady,
  kPlanning,
  kExecuting,
  kWaitingForSnapshot,
  kCanceling,
  kSucceeded,
  kHold,
};

struct GoalPose final {
  std::string frame_id;
  std::int64_t stamp_ns{};
  double x_m{};
  double y_m{};
  double z_m{};
  double qx{};
  double qy{};
  double qz{};
  double qw{1.0};
};

struct RobotState final {
  std::int64_t stamp_ns{};
  double x_m{};
  double y_m{};
  double yaw_rad{};
};

struct PlanningSnapshot final {
  std::shared_ptr<const lunar::planning::GridMap> global_map;
  std::int64_t local_map_stamp_ns{};
  std::int64_t transform_stamp_ns{};
  RobotState robot;
};

struct GoalAcceptance final {
  bool accepted{};
  std::string reason_code;
  std::array<double, 4U> normalized_quaternion{};
  double yaw_rad{};
};

struct PlanningCommand final {
  std::string request_id;
  std::string mission_id;
  std::uint64_t mission_revision{};
  std::int64_t stamp_ns{};
  double goal_x_m{};
  double goal_y_m{};
  double goal_z_m{};
  double goal_yaw_rad{};
  double position_tolerance_m{kDefaultPositionToleranceM};
  double yaw_tolerance_rad{kDefaultYawToleranceRad};
  double roi_min_x_m{};
  double roi_min_y_m{};
  double roi_max_x_m{};
  double roi_max_y_m{};
  bool publish_mission{};
};

struct PlannerReply final {
  std::string request_id;
  std::string mission_id;
  std::uint64_t mission_revision{};
  std::uint8_t planning_outcome{};
  std::uint8_t execution_directive{};
  std::string reason_code;
  bool has_reference{};
  std::uint8_t reference_platform_type{};
  std::string plan_id;
};

enum class ExecutionState : std::uint8_t {
  kAccepted,
  kExecuting,
  kSegmentComplete,
  kFailed,
  kCanceled,
};

struct ExecutionEvent final {
  std::int64_t stamp_ns{};
  std::string plan_id;
  ExecutionState state{ExecutionState::kAccepted};
  std::string reason_code;
};

class GoalStateMachine final {
 public:
  GoalStateMachine() = default;

  void SetSession(std::string session_id);
  [[nodiscard]] bool UpdateSnapshot(PlanningSnapshot snapshot);
  [[nodiscard]] GoalAcceptance SubmitGoal(
      const GoalPose& goal, std::string target_uuid);
  [[nodiscard]] std::optional<PlanningCommand> TakePlanningCommand();
  [[nodiscard]] bool HandlePlannerReply(const PlannerReply& reply);
  void HandleExecutionEvent(const ExecutionEvent& event);
  [[nodiscard]] bool BeginCancel();
  void CompleteCancel(bool canceled);
  void ForceHold(std::string reason_code);
  [[nodiscard]] bool ConsumeHoldRequest() noexcept;

  [[nodiscard]] CoordinatorState state() const noexcept { return state_; }
  [[nodiscard]] const std::string& reason_code() const noexcept {
    return reason_code_;
  }
  [[nodiscard]] const std::string& session_id() const noexcept {
    return session_id_;
  }
  [[nodiscard]] const std::string& mission_id() const noexcept {
    return mission_id_;
  }
  [[nodiscard]] std::uint64_t mission_revision() const noexcept {
    return mission_revision_;
  }
  [[nodiscard]] const std::string& active_plan_id() const noexcept {
    return active_plan_id_;
  }

 private:
  struct NormalizedGoal final {
    GoalPose source;
    std::array<double, 4U> quaternion{};
    double yaw_rad{};
    double elevation_m{};
  };

  [[nodiscard]] GoalAcceptance ValidateAndNormalizeGoal(
      const GoalPose& goal) const;
  [[nodiscard]] std::string ValidateGoalAgainstSnapshot(
      NormalizedGoal& goal) const;
  void EvaluateWaitingGoal();
  void EvaluateCompletedSegment();
  void ClearGoal();
  void EnterHold(std::string reason_code);

  CoordinatorState state_{CoordinatorState::kIdle};
  std::string reason_code_{"IDLE"};
  std::string session_id_;
  std::optional<PlanningSnapshot> snapshot_;
  std::optional<NormalizedGoal> goal_;
  std::string target_uuid_;
  std::string mission_id_;
  std::uint64_t mission_revision_{};
  std::uint64_t next_mission_revision_{1U};
  std::uint64_t next_request_sequence_{1U};
  std::string active_request_id_;
  std::string active_plan_id_;
  std::int64_t segment_complete_stamp_ns_{};
  bool mission_pending_{};
  bool hold_pending_{};
};

}  // namespace lunar::goal_coordinator

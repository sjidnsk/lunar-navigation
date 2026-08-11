#include "lunar_goal_coordinator/goal_state_machine.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <numbers>
#include <string>
#include <utility>
#include <vector>

namespace lunar::goal_coordinator {
namespace {

constexpr std::int64_t kMaximumSnapshotSkewNs = 200'000'000LL;
constexpr std::uint8_t kNewReferenceAvailable = 0U;
constexpr std::uint8_t kSafeFrontierReferenceAvailable = 1U;
constexpr std::uint8_t kActivateNewReference = 0U;
constexpr std::uint8_t kWheeled = 1U;

bool Finite(const GoalPose& goal) noexcept {
  return std::isfinite(goal.x_m) && std::isfinite(goal.y_m) &&
      std::isfinite(goal.z_m) && std::isfinite(goal.qx) &&
      std::isfinite(goal.qy) && std::isfinite(goal.qz) &&
      std::isfinite(goal.qw);
}

double WrapAngle(double angle) noexcept {
  angle = std::remainder(angle, 2.0 * std::numbers::pi);
  if (angle <= -std::numbers::pi) {
    angle += 2.0 * std::numbers::pi;
  }
  return angle;
}

std::optional<std::size_t> CellIndex(
    const lunar::planning::GridMap& map,
    const double x_m, const double y_m) noexcept {
  if (!std::isfinite(x_m) || !std::isfinite(y_m) ||
      !std::isfinite(map.resolution_m) || map.resolution_m <= 0.0 ||
      map.width == 0U || map.height == 0U) {
    return std::nullopt;
  }
  const double x_cell = std::floor((x_m - map.origin_m.x) / map.resolution_m);
  const double y_cell = std::floor((y_m - map.origin_m.y) / map.resolution_m);
  if (x_cell < 0.0 || y_cell < 0.0 ||
      x_cell >= static_cast<double>(map.width) ||
      y_cell >= static_cast<double>(map.height)) {
    return std::nullopt;
  }
  return static_cast<std::size_t>(y_cell) * map.width +
      static_cast<std::size_t>(x_cell);
}

const std::vector<std::uint8_t>* BinaryLayer(
    const lunar::planning::GridMap& map, const std::string& name) {
  const auto found = map.layers.find(name);
  if (found == map.layers.end()) {
    return nullptr;
  }
  return std::get_if<std::vector<std::uint8_t>>(&found->second.values);
}

const std::vector<float>* FloatLayer(
    const lunar::planning::GridMap& map, const std::string& name) {
  const auto found = map.layers.find(name);
  if (found == map.layers.end()) {
    return nullptr;
  }
  return std::get_if<std::vector<float>>(&found->second.values);
}

bool SnapshotStructureValid(const PlanningSnapshot& snapshot) noexcept {
  if (!snapshot.global_map || snapshot.global_map->frame_id != "map" ||
      snapshot.global_map->stamp.nanoseconds_since_epoch <= 0 ||
      snapshot.local_map_stamp_ns !=
          snapshot.global_map->stamp.nanoseconds_since_epoch ||
      snapshot.transform_stamp_ns <= 0 || snapshot.robot.stamp_ns <= 0 ||
      !std::isfinite(snapshot.robot.x_m) ||
      !std::isfinite(snapshot.robot.y_m) ||
      !std::isfinite(snapshot.robot.yaw_rad)) {
    return false;
  }
  const std::int64_t map_stamp =
      snapshot.global_map->stamp.nanoseconds_since_epoch;
  if (std::abs(snapshot.robot.stamp_ns - map_stamp) > kMaximumSnapshotSkewNs ||
      std::abs(snapshot.transform_stamp_ns - map_stamp) >
          kMaximumSnapshotSkewNs) {
    return false;
  }
  const std::size_t count = snapshot.global_map->CellCount();
  const auto* valid = BinaryLayer(*snapshot.global_map, "valid_mask");
  const auto* obstacle = BinaryLayer(*snapshot.global_map, "obstacle");
  const auto* forbidden = BinaryLayer(*snapshot.global_map, "forbidden");
  const auto* elevation = FloatLayer(*snapshot.global_map, "elevation");
  return count > 0U && valid && obstacle && forbidden && elevation &&
      valid->size() == count && obstacle->size() == count &&
      forbidden->size() == count && elevation->size() == count;
}

}  // namespace

void GoalStateMachine::SetSession(std::string session_id) {
  if (session_id == session_id_) {
    return;
  }
  const bool had_active_goal = goal_.has_value();
  ClearGoal();
  snapshot_.reset();
  session_id_ = std::move(session_id);
  next_request_sequence_ = 1U;
  if (session_id_.empty()) {
    state_ = had_active_goal ? CoordinatorState::kHold
                             : CoordinatorState::kIdle;
    reason_code_ = "SESSION_NOT_READY";
    hold_pending_ = had_active_goal;
  } else {
    state_ = CoordinatorState::kIdle;
    reason_code_ = "SESSION_READY";
    hold_pending_ = false;
  }
}

bool GoalStateMachine::UpdateSnapshot(PlanningSnapshot snapshot) {
  if (!SnapshotStructureValid(snapshot)) {
    if (goal_.has_value()) {
      EnterHold("SNAPSHOT_INCONSISTENT");
    } else {
      reason_code_ = "SNAPSHOT_INCONSISTENT";
    }
    return false;
  }
  snapshot_ = std::move(snapshot);
  if (state_ == CoordinatorState::kWaitingForMap) {
    EvaluateWaitingGoal();
  } else if (state_ == CoordinatorState::kWaitingForSnapshot) {
    EvaluateCompletedSegment();
  }
  return true;
}

GoalAcceptance GoalStateMachine::SubmitGoal(
    const GoalPose& goal, std::string target_uuid) {
  if (state_ != CoordinatorState::kIdle || goal_.has_value()) {
    return GoalAcceptance{
        .accepted = false,
        .reason_code = "ACTIVE_GOAL_REQUIRES_CANCEL",
    };
  }
  if (session_id_.empty()) {
    return GoalAcceptance{
        .accepted = false,
        .reason_code = "SESSION_NOT_READY",
    };
  }
  if (target_uuid.empty()) {
    return GoalAcceptance{
        .accepted = false,
        .reason_code = "TARGET_UUID_EMPTY",
    };
  }
  GoalAcceptance normalized = ValidateAndNormalizeGoal(goal);
  if (!normalized.accepted) {
    reason_code_ = normalized.reason_code;
    return normalized;
  }
  goal_ = NormalizedGoal{
      .source = goal,
      .quaternion = normalized.normalized_quaternion,
      .yaw_rad = normalized.yaw_rad,
      .elevation_m = goal.z_m,
  };
  target_uuid_ = std::move(target_uuid);
  mission_id_ = session_id_ + ":" + target_uuid_;
  mission_revision_ = next_mission_revision_++;
  mission_pending_ = true;
  active_request_id_.clear();
  active_plan_id_.clear();
  state_ = CoordinatorState::kWaitingForMap;
  reason_code_ = "WAITING_FOR_CONTAINING_GLOBAL_MAP";

  if (snapshot_ &&
      snapshot_->global_map->stamp.nanoseconds_since_epoch >= goal.stamp_ns) {
    const std::string map_reason = ValidateGoalAgainstSnapshot(*goal_);
    if (!map_reason.empty()) {
      ClearGoal();
      state_ = CoordinatorState::kIdle;
      reason_code_ = map_reason;
      normalized.accepted = false;
      normalized.reason_code = map_reason;
      return normalized;
    }
    state_ = CoordinatorState::kReady;
    reason_code_ = "GOAL_READY";
  }
  return normalized;
}

std::optional<PlanningCommand> GoalStateMachine::TakePlanningCommand() {
  if (state_ != CoordinatorState::kReady || !goal_ || !snapshot_) {
    return std::nullopt;
  }
  const auto& map = *snapshot_->global_map;
  active_request_id_ = mission_id_ + "/request/" +
      std::to_string(next_request_sequence_++);
  PlanningCommand command{
      .request_id = active_request_id_,
      .mission_id = mission_id_,
      .mission_revision = mission_revision_,
      .stamp_ns = map.stamp.nanoseconds_since_epoch,
      .goal_x_m = goal_->source.x_m,
      .goal_y_m = goal_->source.y_m,
      .goal_z_m = goal_->elevation_m,
      .goal_yaw_rad = goal_->yaw_rad,
      .position_tolerance_m = kDefaultPositionToleranceM,
      .yaw_tolerance_rad = kDefaultYawToleranceRad,
      .roi_min_x_m = map.origin_m.x,
      .roi_min_y_m = map.origin_m.y,
      .roi_max_x_m = map.origin_m.x +
          static_cast<double>(map.width) * map.resolution_m,
      .roi_max_y_m = map.origin_m.y +
          static_cast<double>(map.height) * map.resolution_m,
      .publish_mission = mission_pending_,
  };
  mission_pending_ = false;
  state_ = CoordinatorState::kPlanning;
  reason_code_ = "PLAN_REQUEST_PENDING";
  return command;
}

bool GoalStateMachine::HandlePlannerReply(const PlannerReply& reply) {
  if (state_ != CoordinatorState::kPlanning) {
    EnterHold("UNEXPECTED_PLANNER_RESULT");
    return false;
  }
  if (reply.request_id != active_request_id_ ||
      reply.mission_id != mission_id_ ||
      reply.mission_revision != mission_revision_) {
    EnterHold("PLANNER_RESULT_IDENTITY_MISMATCH");
    return false;
  }
  if (reply.planning_outcome != kNewReferenceAvailable &&
      reply.planning_outcome != kSafeFrontierReferenceAvailable) {
    EnterHold(
        "PLANNER_" + (reply.reason_code.empty()
                           ? std::string{"FAILED"}
                           : reply.reason_code));
    return false;
  }
  if (reply.execution_directive != kActivateNewReference) {
    EnterHold("PLANNER_RESULT_NOT_ACTIVATE_NEW_REFERENCE");
    return false;
  }
  if (!reply.has_reference) {
    EnterHold("PLANNER_RESULT_REFERENCE_MISSING");
    return false;
  }
  if (reply.reference_platform_type != kWheeled) {
    EnterHold("PLANNER_RESULT_PLATFORM_NOT_WHEELED");
    return false;
  }
  if (reply.plan_id.empty()) {
    EnterHold("PLANNER_RESULT_PLAN_ID_EMPTY");
    return false;
  }
  active_plan_id_ = reply.plan_id;
  state_ = CoordinatorState::kExecuting;
  reason_code_ = "REFERENCE_PUBLISHED";
  return true;
}

void GoalStateMachine::HandleExecutionEvent(const ExecutionEvent& event) {
  if (state_ == CoordinatorState::kCanceling &&
      event.state == ExecutionState::kCanceled) {
    if (!active_plan_id_.empty() && event.plan_id != active_plan_id_) {
      EnterHold("EXECUTION_FEEDBACK_IDENTITY_MISMATCH");
      return;
    }
    CompleteCancel(true);
    return;
  }
  if (state_ != CoordinatorState::kExecuting ||
      event.plan_id != active_plan_id_) {
    EnterHold("EXECUTION_FEEDBACK_IDENTITY_MISMATCH");
    return;
  }
  switch (event.state) {
    case ExecutionState::kAccepted:
      reason_code_ = "REFERENCE_ACCEPTED";
      return;
    case ExecutionState::kExecuting:
      reason_code_ = "REFERENCE_EXECUTING";
      return;
    case ExecutionState::kSegmentComplete:
      if (event.stamp_ns <= 0) {
        EnterHold("SEGMENT_COMPLETE_STAMP_INVALID");
        return;
      }
      segment_complete_stamp_ns_ = event.stamp_ns;
      state_ = CoordinatorState::kWaitingForSnapshot;
      reason_code_ = "WAITING_FOR_NEWER_SNAPSHOT";
      return;
    case ExecutionState::kFailed:
      EnterHold(
          "EXECUTOR_" + (event.reason_code.empty()
                              ? std::string{"FAILED"}
                              : event.reason_code));
      return;
    case ExecutionState::kCanceled:
      EnterHold(
          "EXECUTOR_" + (event.reason_code.empty()
                              ? std::string{"CANCELED"}
                              : event.reason_code));
      return;
  }
}

bool GoalStateMachine::BeginCancel() {
  if (!goal_ || state_ == CoordinatorState::kCanceling) {
    return false;
  }
  state_ = CoordinatorState::kCanceling;
  reason_code_ = "CANCEL_PENDING";
  hold_pending_ = true;
  return true;
}

void GoalStateMachine::CompleteCancel(const bool canceled) {
  if (state_ != CoordinatorState::kCanceling) {
    return;
  }
  if (!canceled) {
    EnterHold("CANCEL_FAILED");
    return;
  }
  ClearGoal();
  state_ = CoordinatorState::kIdle;
  reason_code_ = session_id_.empty() ? "SESSION_NOT_READY" : "CANCELED";
}

void GoalStateMachine::ForceHold(std::string reason_code) {
  EnterHold(std::move(reason_code));
}

bool GoalStateMachine::ConsumeHoldRequest() noexcept {
  const bool result = hold_pending_;
  hold_pending_ = false;
  return result;
}

GoalAcceptance GoalStateMachine::ValidateAndNormalizeGoal(
    const GoalPose& goal) const {
  if (goal.frame_id != "map") {
    return {.accepted = false, .reason_code = "GOAL_WRONG_FRAME"};
  }
  if (goal.stamp_ns <= 0 || !Finite(goal)) {
    return {.accepted = false, .reason_code = "GOAL_NONFINITE_OR_BAD_STAMP"};
  }
  const double norm = std::hypot(
      std::hypot(goal.qx, goal.qy), std::hypot(goal.qz, goal.qw));
  if (!std::isfinite(norm) || norm < 1.0e-12) {
    return {.accepted = false, .reason_code = "GOAL_QUATERNION_INVALID"};
  }
  const double qx = goal.qx / norm;
  const double qy = goal.qy / norm;
  const double qz = goal.qz / norm;
  const double qw = goal.qw / norm;
  const double yaw = std::atan2(
      2.0 * (qw * qz + qx * qy),
      1.0 - 2.0 * (qy * qy + qz * qz));
  return GoalAcceptance{
      .accepted = true,
      .reason_code = "GOAL_ACCEPTED",
      .normalized_quaternion = {qx, qy, qz, qw},
      .yaw_rad = WrapAngle(yaw),
  };
}

std::string GoalStateMachine::ValidateGoalAgainstSnapshot(
    NormalizedGoal& goal) const {
  if (!snapshot_ || !SnapshotStructureValid(*snapshot_)) {
    return "SNAPSHOT_NOT_READY";
  }
  const auto& map = *snapshot_->global_map;
  const auto goal_index = CellIndex(map, goal.source.x_m, goal.source.y_m);
  if (!goal_index) {
    return "GOAL_OUTSIDE_ACTIVE_GLOBAL_MAP";
  }
  const auto robot_index = CellIndex(
      map, snapshot_->robot.x_m, snapshot_->robot.y_m);
  if (!robot_index) {
    return "ROBOT_OUTSIDE_ACTIVE_GLOBAL_MAP";
  }
  const auto& valid = *BinaryLayer(map, "valid_mask");
  const auto& obstacle = *BinaryLayer(map, "obstacle");
  const auto& forbidden = *BinaryLayer(map, "forbidden");
  const auto& elevation = *FloatLayer(map, "elevation");
  if (valid[*goal_index] == 0U) {
    return "GOAL_UNKNOWN";
  }
  if (obstacle[*goal_index] != 0U) {
    return "GOAL_OBSTACLE";
  }
  if (forbidden[*goal_index] != 0U) {
    return "GOAL_FORBIDDEN";
  }
  const auto traversable = [&](const std::size_t index) {
    return valid[index] != 0U && obstacle[index] == 0U &&
        forbidden[index] == 0U;
  };
  if (!traversable(*robot_index)) {
    return "ROBOT_NOT_ON_KNOWN_FREE_CELL";
  }
  std::vector<std::uint8_t> visited(map.CellCount(), 0U);
  std::deque<std::size_t> queue;
  visited[*robot_index] = 1U;
  queue.push_back(*robot_index);
  while (!queue.empty() && visited[*goal_index] == 0U) {
    const std::size_t current = queue.front();
    queue.pop_front();
    const std::size_t x = current % map.width;
    const std::size_t y = current / map.width;
    const std::array<std::pair<std::ptrdiff_t, std::ptrdiff_t>, 4U> offsets{
        std::pair{-1, 0}, std::pair{1, 0},
        std::pair{0, -1}, std::pair{0, 1}};
    for (const auto& [dx, dy] : offsets) {
      const auto nx = static_cast<std::ptrdiff_t>(x) + dx;
      const auto ny = static_cast<std::ptrdiff_t>(y) + dy;
      if (nx < 0 || ny < 0 || nx >= static_cast<std::ptrdiff_t>(map.width) ||
          ny >= static_cast<std::ptrdiff_t>(map.height)) {
        continue;
      }
      const std::size_t next = static_cast<std::size_t>(ny) * map.width +
          static_cast<std::size_t>(nx);
      if (visited[next] == 0U && traversable(next)) {
        visited[next] = 1U;
        queue.push_back(next);
      }
    }
  }
  if (visited[*goal_index] == 0U) {
    return "GOAL_DISCONNECTED";
  }
  goal.elevation_m = static_cast<double>(elevation[*goal_index]);
  return {};
}

void GoalStateMachine::EvaluateWaitingGoal() {
  if (!goal_ || !snapshot_ ||
      snapshot_->global_map->stamp.nanoseconds_since_epoch <
          goal_->source.stamp_ns) {
    return;
  }
  const std::string validation = ValidateGoalAgainstSnapshot(*goal_);
  if (!validation.empty()) {
    ClearGoal();
    state_ = CoordinatorState::kIdle;
    reason_code_ = validation;
    return;
  }
  state_ = CoordinatorState::kReady;
  reason_code_ = "GOAL_READY";
}

void GoalStateMachine::EvaluateCompletedSegment() {
  if (!goal_ || !snapshot_) {
    EnterHold("ACTIVE_GOAL_CONTEXT_LOST");
    return;
  }
  const auto map_stamp =
      snapshot_->global_map->stamp.nanoseconds_since_epoch;
  if (map_stamp <= segment_complete_stamp_ns_ ||
      snapshot_->local_map_stamp_ns <= segment_complete_stamp_ns_ ||
      snapshot_->robot.stamp_ns <= segment_complete_stamp_ns_ ||
      snapshot_->transform_stamp_ns <= segment_complete_stamp_ns_) {
    return;
  }
  const double distance = std::hypot(
      snapshot_->robot.x_m - goal_->source.x_m,
      snapshot_->robot.y_m - goal_->source.y_m);
  const double yaw_error = std::abs(WrapAngle(
      snapshot_->robot.yaw_rad - goal_->yaw_rad));
  if (distance <= kDefaultPositionToleranceM &&
      yaw_error <= kDefaultYawToleranceRad) {
    state_ = CoordinatorState::kSucceeded;
    reason_code_ = "GOAL_SUCCEEDED";
    hold_pending_ = true;
    return;
  }
  const std::string validation = ValidateGoalAgainstSnapshot(*goal_);
  if (!validation.empty()) {
    EnterHold(validation);
    return;
  }
  active_plan_id_.clear();
  state_ = CoordinatorState::kReady;
  reason_code_ = "ROLLING_REPLAN_READY";
}

void GoalStateMachine::ClearGoal() {
  goal_.reset();
  target_uuid_.clear();
  mission_id_.clear();
  mission_revision_ = 0U;
  active_request_id_.clear();
  active_plan_id_.clear();
  segment_complete_stamp_ns_ = 0;
  mission_pending_ = false;
}

void GoalStateMachine::EnterHold(std::string reason_code) {
  if (state_ == CoordinatorState::kHold) {
    return;
  }
  state_ = CoordinatorState::kHold;
  reason_code_ = reason_code.empty() ? "HOLD_UNSPECIFIED" : std::move(reason_code);
  hold_pending_ = true;
}

}  // namespace lunar::goal_coordinator

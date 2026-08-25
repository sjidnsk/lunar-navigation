#include "lunar_pure_exploration_core/exploration_state_machine.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace lunar::pure_exploration {
namespace {

bool Finite(Pose2 pose) {
  return std::isfinite(pose.x) && std::isfinite(pose.y) &&
         std::isfinite(pose.yaw);
}

void RequireState(bool condition, const char* message) {
  if (!condition) {
    throw std::logic_error(message);
  }
}

const char* ReleaseCode(GoalReleaseReason reason) {
  switch (reason) {
    case GoalReleaseReason::kArrived:
      return "ARRIVED";
    case GoalReleaseReason::kNoPath:
      return "NO_PATH";
    case GoalReleaseReason::kCandidateInvalid:
      return "CANDIDATE_INVALID";
    case GoalReleaseReason::kFrontierDisappeared:
      return "FRONTIER_DISAPPEARED";
    case GoalReleaseReason::kInformationGainZero:
      return "INFORMATION_GAIN_ZERO";
    case GoalReleaseReason::kLocalSegmentCompleted:
      return "LOCAL_SEGMENT_COMPLETED";
  }
  throw std::invalid_argument("unknown goal release reason");
}

}  // namespace

ActiveGoal::ActiveGoal(std::uint64_t candidate_id, std::uint64_t frontier_id,
                       std::variant<TaskFrontierGoalIdentity,
                                    BoundaryApproachGoalIdentity> identity,
                       Pose2 target,
                       std::string request_id)
    : candidate_id_(candidate_id),
      frontier_id_(frontier_id),
      identity_(std::move(identity)),
      target_(target),
      request_id_(std::move(request_id)) {}

ActiveGoal::ActiveGoal(ActiveGoal&& other) noexcept
    : candidate_id_(other.candidate_id_),
      frontier_id_(other.frontier_id_),
      identity_(std::move(other.identity_)),
      target_(other.target_),
      request_id_(std::move(other.request_id_)),
      replan_count_(other.replan_count_),
      owns_payload_(other.owns_payload_) {
  other.owns_payload_ = false;
}

std::uint64_t ActiveGoal::candidate_id() const { return candidate_id_; }

std::uint64_t ActiveGoal::frontier_id() const { return frontier_id_; }

GoalKind ActiveGoal::kind() const {
  return std::holds_alternative<BoundaryApproachGoalIdentity>(identity_)
             ? GoalKind::kBoundaryApproach
             : GoalKind::kTaskFrontier;
}

std::span<const std::int64_t> ActiveGoal::frontier_canonical_key() const {
  if (const auto* identity =
          std::get_if<TaskFrontierGoalIdentity>(&identity_)) {
    return identity->frontier_canonical_key;
  }
  return {};
}

const CandidateKey& ActiveGoal::candidate_key() const {
  return std::visit(
      [](const auto& identity) -> const CandidateKey& {
        return identity.candidate_key;
      },
      identity_);
}

const Pose2& ActiveGoal::target() const { return target_; }

const std::string& ActiveGoal::request_id() const { return request_id_; }

const BoundaryApproachGoalIdentity* ActiveGoal::boundary_approach_identity()
    const {
  return std::get_if<BoundaryApproachGoalIdentity>(&identity_);
}

std::uint8_t ActiveGoal::replan_count() const { return replan_count_; }

bool ActiveGoal::MatchesAnyFrontier(
    std::span<const FrontierCluster> current_frontiers) const {
  if (kind() != GoalKind::kTaskFrontier) {
    return false;
  }
  const auto& goal_identity = std::get<TaskFrontierGoalIdentity>(identity_);
  return std::any_of(
      current_frontiers.begin(), current_frontiers.end(),
      [&goal_identity](const FrontierCluster& frontier) {
        return frontier.canonical_key == goal_identity.frontier_canonical_key;
      });
}

ActiveGoal MakeActiveGoal(
    const CandidateView& candidate,
    std::span<const FrontierCluster> frozen_frontiers,
    std::string request_id) {
  if (request_id.empty()) {
    throw std::invalid_argument("active goal request id must be nonempty");
  }
  if (candidate.frontier_index >= frozen_frontiers.size()) {
    throw std::invalid_argument("active goal frontier index is out of range");
  }
  if (!Finite(candidate.pose)) {
    throw std::invalid_argument("active goal pose must be finite");
  }
  if (!std::isfinite(candidate.frontier_distance_m) ||
      candidate.frontier_distance_m < 0.0) {
    throw std::invalid_argument(
        "active goal frontier distance must be finite and nonnegative");
  }
  if (!candidate.frontier_canonical_key ||
      candidate.frontier_canonical_key->empty()) {
    throw std::invalid_argument("active goal shared frontier key is empty");
  }
  const FrontierCluster& frontier = frozen_frontiers[candidate.frontier_index];
  if (candidate.frontier_id != frontier.id) {
    throw std::invalid_argument("active goal frontier display id mismatch");
  }
  if (*candidate.frontier_canonical_key != frontier.canonical_key) {
    throw std::invalid_argument("active goal full frontier key mismatch");
  }
  return ActiveGoal(candidate.id, candidate.frontier_id,
                    TaskFrontierGoalIdentity{
                        *candidate.frontier_canonical_key, candidate.key},
                    candidate.pose, std::move(request_id));
}

ActiveGoal MakeBoundaryApproachActiveGoal(
    std::uint64_t display_id, BoundaryApproachGoalIdentity identity,
    Pose2 target, std::string request_id) {
  if (request_id.empty()) {
    throw std::invalid_argument("active goal request id must be nonempty");
  }
  if (!Finite(target)) {
    throw std::invalid_argument("active goal pose must be finite");
  }
  return ActiveGoal(display_id, 0U, std::move(identity), target,
                    std::move(request_id));
}

ExplorationStateMachine::ExplorationStateMachine(
    std::uint8_t maximum_replans)
    : maximum_replans_(maximum_replans) {}

void ExplorationStateMachine::Start(std::string task_id) {
  if (task_id.empty()) {
    throw std::invalid_argument("exploration task id must be nonempty");
  }
  active_goal_.reset();
  task_id_ = std::move(task_id);
  reason_code_ = "STARTED";
  state_ = ExplorationState::kWaitingForInput;
}

void ExplorationStateMachine::WaitForInput() {
  RequireState(state_ == ExplorationState::kWaitingForInput ||
                   state_ == ExplorationState::kSelectingFrontier,
               "WaitForInput is illegal in the current state");
  state_ = ExplorationState::kWaitingForInput;
  reason_code_ = "WAITING_FOR_INPUT";
}

void ExplorationStateMachine::BeginSelection() {
  RequireState(state_ == ExplorationState::kWaitingForInput,
               "BeginSelection is illegal in the current state");
  state_ = ExplorationState::kSelectingFrontier;
  reason_code_ = "SELECTING_FRONTIER";
}

void ExplorationStateMachine::BeginPlanning() {
  RequireState(state_ == ExplorationState::kSelectingFrontier,
               "BeginPlanning is illegal in the current state");
  state_ = ExplorationState::kPlanning;
  reason_code_ = "PLANNING";
}

void ExplorationStateMachine::Pause() {
  RequireState(state_ == ExplorationState::kWaitingForInput ||
                   state_ == ExplorationState::kSelectingFrontier ||
                   state_ == ExplorationState::kPlanning ||
                   state_ == ExplorationState::kExecuting ||
                   state_ == ExplorationState::kReplanning,
               "Pause is illegal in the current state");
  active_goal_.reset();
  state_ = ExplorationState::kPaused;
  reason_code_ = "PAUSED";
}

void ExplorationStateMachine::Resume() {
  RequireState(state_ == ExplorationState::kPaused,
               "Resume is illegal in the current state");
  state_ = ExplorationState::kWaitingForInput;
  reason_code_ = "RESUMED";
}

void ExplorationStateMachine::Cancel() {
  active_goal_.reset();
  task_id_.clear();
  state_ = ExplorationState::kIdle;
  reason_code_ = "CANCELED";
}

void ExplorationStateMachine::CommitGoal(ActiveGoal&& goal) {
  RequireState(state_ == ExplorationState::kPlanning,
               "CommitGoal is illegal in the current state");
  if (!goal.owns_payload_) {
    throw std::invalid_argument("active goal payload was already consumed");
  }
  active_goal_.emplace(std::move(goal));
  state_ = ExplorationState::kExecuting;
  reason_code_ = "EXECUTING";
}

ReplanResult ExplorationStateMachine::BeginReplanning(ReplanCause cause) {
  RequireState(state_ == ExplorationState::kExecuting &&
                   active_goal_.has_value(),
               "BeginReplanning requires an executing goal");
  switch (cause) {
    case ReplanCause::kRollingSegment:
      state_ = ExplorationState::kReplanning;
      reason_code_ = "ROLLING_SEGMENT";
      return ReplanResult::kStarted;
    case ReplanCause::kStuckRecovery:
      if (active_goal_->replan_count_ == maximum_replans_) {
        active_goal_.reset();
        state_ = ExplorationState::kSelectingFrontier;
        reason_code_ = "STUCK_RETRIES_EXHAUSTED";
        return ReplanResult::kExhausted;
      }
      ++active_goal_->replan_count_;
      state_ = ExplorationState::kReplanning;
      reason_code_ = "STUCK_RETRY";
      return ReplanResult::kStarted;
  }
  throw std::invalid_argument("unknown replan cause");
}

void ExplorationStateMachine::ResumeExecution() {
  RequireState(state_ == ExplorationState::kReplanning &&
                   active_goal_.has_value(),
               "ResumeExecution requires a replanning goal");
  state_ = ExplorationState::kExecuting;
  reason_code_ = "EXECUTING";
}

void ExplorationStateMachine::ReleaseGoal(GoalReleaseReason reason) {
  RequireState((state_ == ExplorationState::kExecuting ||
                state_ == ExplorationState::kReplanning) &&
                   active_goal_.has_value(),
               "ReleaseGoal requires an active execution goal");
  const char* reason_code = ReleaseCode(reason);
  active_goal_.reset();
  state_ = ExplorationState::kSelectingFrontier;
  reason_code_ = reason_code;
}

void ExplorationStateMachine::CompleteNoReachableFrontier() {
  RequireState(state_ == ExplorationState::kSelectingFrontier &&
                   !active_goal_.has_value(),
               "completion requires selection without a goal");
  state_ = ExplorationState::kCompleted;
  reason_code_ = "COMPLETED_NO_REACHABLE_FRONTIER";
}

void ExplorationStateMachine::Fail(std::string reason_code) {
  if (reason_code.empty()) {
    throw std::invalid_argument("exploration failure reason must be nonempty");
  }
  active_goal_.reset();
  reason_code_ = std::move(reason_code);
  state_ = ExplorationState::kError;
}

bool ExplorationStateMachine::MayReplaceGoalForMapUpdate() const {
  return !active_goal_.has_value();
}

ExplorationState ExplorationStateMachine::state() const { return state_; }

const std::string& ExplorationStateMachine::task_id() const { return task_id_; }

const std::string& ExplorationStateMachine::reason_code() const {
  return reason_code_;
}

const std::optional<ActiveGoal>& ExplorationStateMachine::active_goal() const {
  return active_goal_;
}

}  // namespace lunar::pure_exploration

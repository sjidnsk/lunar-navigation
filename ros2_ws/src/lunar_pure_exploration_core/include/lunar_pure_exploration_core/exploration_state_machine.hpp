#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "lunar_pure_exploration_core/candidate_generator.hpp"

namespace lunar::pure_exploration {

enum class ExplorationState : std::uint8_t {
  kIdle,
  kWaitingForInput,
  kSelectingFrontier,
  kPlanning,
  kExecuting,
  kReplanning,
  kPaused,
  kCompleted,
  kError,
};

enum class GoalReleaseReason : std::uint8_t {
  kArrived,
  kNoPath,
  kCandidateInvalid,
  kFrontierDisappeared,
  kInformationGainZero,
};

enum class ReplanCause : std::uint8_t {
  kRollingSegment,
  kStuckRecovery,
};

enum class ReplanResult : std::uint8_t {
  kStarted,
  kExhausted,
};

class ActiveGoal {
 public:
  ActiveGoal(ActiveGoal&& other) noexcept;
  ActiveGoal& operator=(ActiveGoal&&) = delete;
  ActiveGoal(const ActiveGoal&) = delete;
  ActiveGoal& operator=(const ActiveGoal&) = delete;

  std::uint64_t candidate_id() const;
  std::uint64_t frontier_id() const;
  std::span<const std::int64_t> frontier_canonical_key() const;
  const CandidateKey& candidate_key() const;
  const Pose2& target() const;
  const std::string& request_id() const;
  std::uint8_t replan_count() const;
  bool MatchesAnyFrontier(
      std::span<const FrontierCluster> current_frontiers) const;

 private:
  friend ActiveGoal MakeActiveGoal(const CandidateView& candidate,
                                   std::span<const FrontierCluster> frozen_frontiers,
                                   std::string request_id);
  friend class ExplorationStateMachine;

  ActiveGoal(std::uint64_t candidate_id, std::uint64_t frontier_id,
             std::vector<std::int64_t> frontier_canonical_key,
             CandidateKey candidate_key, Pose2 target,
             std::string request_id);

  std::uint64_t candidate_id_;
  std::uint64_t frontier_id_;
  std::vector<std::int64_t> frontier_canonical_key_;
  CandidateKey candidate_key_;
  Pose2 target_;
  std::string request_id_;
  std::uint8_t replan_count_{0U};
  bool owns_payload_{true};
};

ActiveGoal MakeActiveGoal(const CandidateView& candidate,
                          std::span<const FrontierCluster> frozen_frontiers,
                          std::string request_id);

class ExplorationStateMachine {
 public:
  explicit ExplorationStateMachine(std::uint8_t maximum_replans);

  void Start(std::string task_id);
  void WaitForInput();
  void BeginSelection();
  void BeginPlanning();
  void Pause();
  void Resume();
  void Cancel();
  void CommitGoal(ActiveGoal&& goal);
  ReplanResult BeginReplanning(ReplanCause cause);
  void ResumeExecution();
  void ReleaseGoal(GoalReleaseReason reason);
  void CompleteNoReachableFrontier();
  void Fail(std::string reason_code);

  bool MayReplaceGoalForMapUpdate() const;
  ExplorationState state() const;
  const std::string& task_id() const;
  const std::string& reason_code() const;
  const std::optional<ActiveGoal>& active_goal() const;

 private:
  std::uint8_t maximum_replans_;
  ExplorationState state_{ExplorationState::kIdle};
  std::string task_id_;
  std::string reason_code_;
  std::optional<ActiveGoal> active_goal_;
};

}  // namespace lunar::pure_exploration

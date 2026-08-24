#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "lunar_pure_exploration_core/exploration_state_machine.hpp"

namespace lunar::pure_exploration {
namespace {

constexpr std::array<ExplorationState, 9> kAllStates{
    ExplorationState::kIdle,
    ExplorationState::kWaitingForInput,
    ExplorationState::kSelectingFrontier,
    ExplorationState::kPlanning,
    ExplorationState::kExecuting,
    ExplorationState::kReplanning,
    ExplorationState::kPaused,
    ExplorationState::kCompleted,
    ExplorationState::kError,
};

FrontierCluster Frontier(std::uint64_t id = 77U,
                         std::vector<std::int64_t> key = {4, 5, 6}) {
  return FrontierCluster{
      .id = id,
      .cells = {{1, 2}},
      .interface_edges = {},
      .canonical_key = std::move(key),
      .centroid = {3.0, 4.0},
      .length_m = 1.0,
  };
}

CandidateView Candidate(
    std::shared_ptr<const std::vector<std::int64_t>> owner =
        std::make_shared<const std::vector<std::int64_t>>(
            std::initializer_list<std::int64_t>{4, 5, 6})) {
  return CandidateView{
      .id = 42U,
      .frontier_id = 77U,
      .frontier_index = 0U,
      .key = {1000, -2000, 135},
      .pose = {1.25, -2.5, 0.75},
      .frontier_distance_m = 3.5,
      .frontier_canonical_key = std::move(owner),
  };
}

ActiveGoal Goal(std::string request_id = "request-9") {
  const std::array<FrontierCluster, 1> frontiers{Frontier()};
  return MakeActiveGoal(Candidate(), frontiers, std::move(request_id));
}

struct GoalSnapshot {
  std::uint64_t candidate_id;
  std::uint64_t frontier_id;
  std::vector<std::int64_t> frontier_key;
  CandidateKey candidate_key;
  Pose2 target;
  std::string request_id;
  std::uint8_t replan_count;
};

struct MachineSnapshot {
  ExplorationState state;
  std::string task_id;
  std::string reason;
  std::optional<GoalSnapshot> goal;
};

GoalSnapshot Snapshot(const ActiveGoal& goal) {
  return GoalSnapshot{
      goal.candidate_id(),
      goal.frontier_id(),
      {goal.frontier_canonical_key().begin(),
       goal.frontier_canonical_key().end()},
      goal.candidate_key(),
      goal.target(),
      goal.request_id(),
      goal.replan_count(),
  };
}

MachineSnapshot Snapshot(const ExplorationStateMachine& machine) {
  std::optional<GoalSnapshot> goal;
  if (machine.active_goal().has_value()) {
    goal.emplace(Snapshot(*machine.active_goal()));
  }
  return MachineSnapshot{machine.state(), machine.task_id(),
                         machine.reason_code(), std::move(goal)};
}

void ExpectPose(const Pose2& actual, const Pose2& expected) {
  EXPECT_DOUBLE_EQ(actual.x, expected.x);
  EXPECT_DOUBLE_EQ(actual.y, expected.y);
  EXPECT_DOUBLE_EQ(actual.yaw, expected.yaw);
}

void ExpectGoal(const ActiveGoal& actual, const GoalSnapshot& expected) {
  EXPECT_EQ(actual.candidate_id(), expected.candidate_id);
  EXPECT_EQ(actual.frontier_id(), expected.frontier_id);
  EXPECT_EQ(std::vector<std::int64_t>(actual.frontier_canonical_key().begin(),
                                      actual.frontier_canonical_key().end()),
            expected.frontier_key);
  EXPECT_EQ(actual.candidate_key(), expected.candidate_key);
  ExpectPose(actual.target(), expected.target);
  EXPECT_EQ(actual.request_id(), expected.request_id);
  EXPECT_EQ(actual.replan_count(), expected.replan_count);
}

void ExpectSnapshot(const ExplorationStateMachine& actual,
                    const MachineSnapshot& expected) {
  EXPECT_EQ(actual.state(), expected.state);
  EXPECT_EQ(actual.task_id(), expected.task_id);
  EXPECT_EQ(actual.reason_code(), expected.reason);
  ASSERT_EQ(actual.active_goal().has_value(), expected.goal.has_value());
  if (expected.goal.has_value()) {
    ExpectGoal(*actual.active_goal(), *expected.goal);
  }
}

template <typename Callable>
void ExpectExactLogicError(Callable&& call) {
  try {
    std::forward<Callable>(call)();
    FAIL() << "expected exact std::logic_error";
  } catch (const std::invalid_argument&) {
    FAIL() << "std::invalid_argument must not outrank illegal source state";
  } catch (const std::logic_error&) {
  } catch (...) {
    FAIL() << "expected exact std::logic_error";
  }
}

std::unique_ptr<ExplorationStateMachine> MachineIn(
    ExplorationState wanted, std::uint8_t maximum_replans = 2U) {
  auto machine = std::make_unique<ExplorationStateMachine>(maximum_replans);
  if (wanted == ExplorationState::kIdle) {
    return machine;
  }
  machine->Start("task-A");
  if (wanted == ExplorationState::kWaitingForInput) {
    return machine;
  }
  if (wanted == ExplorationState::kPaused) {
    machine->Pause();
    return machine;
  }
  if (wanted == ExplorationState::kError) {
    machine->Fail("BROKEN_INPUT");
    return machine;
  }
  machine->BeginSelection();
  if (wanted == ExplorationState::kSelectingFrontier) {
    return machine;
  }
  if (wanted == ExplorationState::kCompleted) {
    machine->CompleteNoReachableFrontier();
    return machine;
  }
  machine->BeginPlanning();
  if (wanted == ExplorationState::kPlanning) {
    return machine;
  }
  auto goal = Goal();
  machine->CommitGoal(std::move(goal));
  if (wanted == ExplorationState::kExecuting) {
    return machine;
  }
  EXPECT_EQ(machine->BeginReplanning(ReplanCause::kRollingSegment),
            ReplanResult::kStarted);
  EXPECT_EQ(wanted, ExplorationState::kReplanning);
  return machine;
}

TEST(ExplorationStateMachineTest, ConstructionFreezesEmptyIdleState) {
  const ExplorationStateMachine machine(2U);
  EXPECT_EQ(machine.state(), ExplorationState::kIdle);
  EXPECT_TRUE(machine.task_id().empty());
  EXPECT_TRUE(machine.reason_code().empty());
  EXPECT_FALSE(machine.active_goal().has_value());
  EXPECT_TRUE(machine.MayReplaceGoalForMapUpdate());
}

TEST(ExplorationStateMachineTest, StartMatrixReplacesAtomicallyFromEveryState) {
  for (const ExplorationState source : kAllStates) {
    auto machine = MachineIn(source);
    machine->Start("replacement");
    EXPECT_EQ(machine->state(), ExplorationState::kWaitingForInput);
    EXPECT_EQ(machine->task_id(), "replacement");
    EXPECT_EQ(machine->reason_code(), "STARTED");
    EXPECT_FALSE(machine->active_goal().has_value());
    EXPECT_TRUE(machine->MayReplaceGoalForMapUpdate());
  }
  for (const ExplorationState source : kAllStates) {
    auto machine = MachineIn(source);
    const auto before = Snapshot(*machine);
    EXPECT_THROW(machine->Start(""), std::invalid_argument);
    ExpectSnapshot(*machine, before);
  }
}

TEST(ExplorationStateMachineTest, CancelMatrixIsIdempotentAndClearsTaskAndGoal) {
  for (const ExplorationState source : kAllStates) {
    auto machine = MachineIn(source);
    machine->Cancel();
    EXPECT_EQ(machine->state(), ExplorationState::kIdle);
    EXPECT_TRUE(machine->task_id().empty());
    EXPECT_EQ(machine->reason_code(), "CANCELED");
    EXPECT_FALSE(machine->active_goal().has_value());
    machine->Cancel();
    EXPECT_EQ(machine->reason_code(), "CANCELED");
  }
}

TEST(ExplorationStateMachineTest, FailMatrixRetainsTaskAndRejectsEmptyReason) {
  for (const ExplorationState source : kAllStates) {
    auto machine = MachineIn(source);
    const std::string task = machine->task_id();
    machine->Fail("NUMERIC_ERROR");
    EXPECT_EQ(machine->state(), ExplorationState::kError);
    EXPECT_EQ(machine->task_id(), task);
    EXPECT_EQ(machine->reason_code(), "NUMERIC_ERROR");
    EXPECT_FALSE(machine->active_goal().has_value());
  }
  for (const ExplorationState source : kAllStates) {
    auto machine = MachineIn(source);
    const auto before = Snapshot(*machine);
    EXPECT_THROW(machine->Fail(""), std::invalid_argument);
    ExpectSnapshot(*machine, before);
  }
}

TEST(ExplorationStateMachineTest, WaitForInputMatrixHasTwoLegalSources) {
  for (const ExplorationState source : kAllStates) {
    auto machine = MachineIn(source);
    if (source == ExplorationState::kWaitingForInput ||
        source == ExplorationState::kSelectingFrontier) {
      machine->WaitForInput();
      EXPECT_EQ(machine->state(), ExplorationState::kWaitingForInput);
      EXPECT_EQ(machine->task_id(), "task-A");
      EXPECT_EQ(machine->reason_code(), "WAITING_FOR_INPUT");
      EXPECT_FALSE(machine->active_goal().has_value());
    } else {
      const auto before = Snapshot(*machine);
      EXPECT_THROW(machine->WaitForInput(), std::logic_error);
      ExpectSnapshot(*machine, before);
    }
  }
}

TEST(ExplorationStateMachineTest, BeginSelectionMatrixOnlyAcceptsWaiting) {
  for (const ExplorationState source : kAllStates) {
    auto machine = MachineIn(source);
    if (source == ExplorationState::kWaitingForInput) {
      machine->BeginSelection();
      EXPECT_EQ(machine->state(), ExplorationState::kSelectingFrontier);
      EXPECT_EQ(machine->task_id(), "task-A");
      EXPECT_EQ(machine->reason_code(), "SELECTING_FRONTIER");
      EXPECT_FALSE(machine->active_goal().has_value());
    } else {
      const auto before = Snapshot(*machine);
      EXPECT_THROW(machine->BeginSelection(), std::logic_error);
      ExpectSnapshot(*machine, before);
    }
  }
}

TEST(ExplorationStateMachineTest, BeginPlanningMatrixOnlyAcceptsSelecting) {
  for (const ExplorationState source : kAllStates) {
    auto machine = MachineIn(source);
    if (source == ExplorationState::kSelectingFrontier) {
      machine->BeginPlanning();
      EXPECT_EQ(machine->state(), ExplorationState::kPlanning);
      EXPECT_EQ(machine->task_id(), "task-A");
      EXPECT_EQ(machine->reason_code(), "PLANNING");
      EXPECT_FALSE(machine->active_goal().has_value());
    } else {
      const auto before = Snapshot(*machine);
      EXPECT_THROW(machine->BeginPlanning(), std::logic_error);
      ExpectSnapshot(*machine, before);
    }
  }
}

TEST(ExplorationStateMachineTest, PauseMatrixAcceptsOnlyActiveTaskStatesAndClearsGoal) {
  for (const ExplorationState source : kAllStates) {
    auto machine = MachineIn(source);
    const bool legal = source == ExplorationState::kWaitingForInput ||
                       source == ExplorationState::kSelectingFrontier ||
                       source == ExplorationState::kPlanning ||
                       source == ExplorationState::kExecuting ||
                       source == ExplorationState::kReplanning;
    if (legal) {
      machine->Pause();
      EXPECT_EQ(machine->state(), ExplorationState::kPaused);
      EXPECT_EQ(machine->task_id(), "task-A");
      EXPECT_EQ(machine->reason_code(), "PAUSED");
      EXPECT_FALSE(machine->active_goal().has_value());
    } else {
      const auto before = Snapshot(*machine);
      EXPECT_THROW(machine->Pause(), std::logic_error);
      ExpectSnapshot(*machine, before);
    }
  }
}

TEST(ExplorationStateMachineTest, ResumeMatrixOnlyAcceptsPausedAndReturnsToWaiting) {
  for (const ExplorationState source : kAllStates) {
    auto machine = MachineIn(source);
    if (source == ExplorationState::kPaused) {
      machine->Resume();
      EXPECT_EQ(machine->state(), ExplorationState::kWaitingForInput);
      EXPECT_EQ(machine->task_id(), "task-A");
      EXPECT_EQ(machine->reason_code(), "RESUMED");
      EXPECT_FALSE(machine->active_goal().has_value());
    } else {
      const auto before = Snapshot(*machine);
      EXPECT_THROW(machine->Resume(), std::logic_error);
      ExpectSnapshot(*machine, before);
    }
  }
}

TEST(ExplorationStateMachineTest, CommitGoalMatrixOnlyAcceptsPlanning) {
  for (const ExplorationState source : kAllStates) {
    auto machine = MachineIn(source);
    auto goal = Goal("commit-request");
    const auto expected = Snapshot(goal);
    if (source == ExplorationState::kPlanning) {
      machine->CommitGoal(std::move(goal));
      EXPECT_EQ(machine->state(), ExplorationState::kExecuting);
      EXPECT_EQ(machine->task_id(), "task-A");
      EXPECT_EQ(machine->reason_code(), "EXECUTING");
      ASSERT_TRUE(machine->active_goal().has_value());
      ExpectGoal(*machine->active_goal(), expected);
      EXPECT_FALSE(machine->MayReplaceGoalForMapUpdate());
    } else {
      const auto before = Snapshot(*machine);
      EXPECT_THROW(machine->CommitGoal(std::move(goal)), std::logic_error);
      ExpectSnapshot(*machine, before);
    }
  }
}

TEST(ExplorationStateMachineTest,
     IllegalSourcesTakePriorityOverConsumedAndForgedArguments) {
  auto consumed_goal = Goal("consumed-priority");
  ActiveGoal retained_payload(std::move(consumed_goal));
  EXPECT_EQ(retained_payload.request_id(), "consumed-priority");
  for (const ExplorationState source : kAllStates) {
    if (source == ExplorationState::kPlanning) {
      continue;
    }
    auto machine = MachineIn(source);
    const auto before = Snapshot(*machine);
    ExpectExactLogicError(
        [&] { machine->CommitGoal(std::move(consumed_goal)); });
    ExpectSnapshot(*machine, before);
  }

  for (const ExplorationState source : kAllStates) {
    if (source == ExplorationState::kExecuting ||
        source == ExplorationState::kReplanning) {
      continue;
    }
    auto machine = MachineIn(source);
    const auto before = Snapshot(*machine);
    ExpectExactLogicError([&] {
      machine->ReleaseGoal(static_cast<GoalReleaseReason>(255U));
    });
    ExpectSnapshot(*machine, before);
  }

  for (const ExplorationState source : kAllStates) {
    if (source == ExplorationState::kExecuting) {
      continue;
    }
    auto machine = MachineIn(source);
    const auto before = Snapshot(*machine);
    ExpectExactLogicError([&] {
      (void)machine->BeginReplanning(static_cast<ReplanCause>(255U));
    });
    ExpectSnapshot(*machine, before);
  }
}

TEST(ExplorationStateMachineTest, BeginReplanningMatrixOnlyAcceptsExecutingGoal) {
  for (const ExplorationState source : kAllStates) {
    auto machine = MachineIn(source);
    const auto before = Snapshot(*machine);
    if (source == ExplorationState::kExecuting) {
      EXPECT_EQ(machine->BeginReplanning(ReplanCause::kRollingSegment),
                ReplanResult::kStarted);
      EXPECT_EQ(machine->state(), ExplorationState::kReplanning);
      EXPECT_EQ(machine->task_id(), "task-A");
      EXPECT_EQ(machine->reason_code(), "ROLLING_SEGMENT");
      ASSERT_TRUE(machine->active_goal().has_value());
      ExpectGoal(*machine->active_goal(), *before.goal);
    } else {
      EXPECT_THROW(machine->BeginReplanning(ReplanCause::kRollingSegment),
                   std::logic_error);
      ExpectSnapshot(*machine, before);
    }
  }
}

TEST(ExplorationStateMachineTest, ResumeExecutionMatrixOnlyAcceptsReplanningGoal) {
  for (const ExplorationState source : kAllStates) {
    auto machine = MachineIn(source);
    const auto before = Snapshot(*machine);
    if (source == ExplorationState::kReplanning) {
      machine->ResumeExecution();
      EXPECT_EQ(machine->state(), ExplorationState::kExecuting);
      EXPECT_EQ(machine->task_id(), "task-A");
      EXPECT_EQ(machine->reason_code(), "EXECUTING");
      ASSERT_TRUE(machine->active_goal().has_value());
      ExpectGoal(*machine->active_goal(), *before.goal);
    } else {
      EXPECT_THROW(machine->ResumeExecution(), std::logic_error);
      ExpectSnapshot(*machine, before);
    }
  }
}

TEST(ExplorationStateMachineTest, ReleaseGoalMatrixAcceptsExecutingAndReplanning) {
  for (const ExplorationState source : kAllStates) {
    auto machine = MachineIn(source);
    const auto before = Snapshot(*machine);
    if (source == ExplorationState::kExecuting ||
        source == ExplorationState::kReplanning) {
      machine->ReleaseGoal(GoalReleaseReason::kArrived);
      EXPECT_EQ(machine->state(), ExplorationState::kSelectingFrontier);
      EXPECT_EQ(machine->task_id(), "task-A");
      EXPECT_EQ(machine->reason_code(), "ARRIVED");
      EXPECT_FALSE(machine->active_goal().has_value());
      EXPECT_TRUE(machine->MayReplaceGoalForMapUpdate());
    } else {
      EXPECT_THROW(machine->ReleaseGoal(GoalReleaseReason::kArrived),
                   std::logic_error);
      ExpectSnapshot(*machine, before);
    }
  }
}

TEST(ExplorationStateMachineTest,
     LocalSegmentCompletionReleasesGoalWithoutFailureSideEffect) {
  auto machine = MachineIn(ExplorationState::kExecuting);
  machine->ReleaseGoal(GoalReleaseReason::kLocalSegmentCompleted);
  EXPECT_EQ(machine->state(), ExplorationState::kSelectingFrontier);
  EXPECT_EQ(machine->reason_code(), "LOCAL_SEGMENT_COMPLETED");
  EXPECT_FALSE(machine->active_goal().has_value());
}

TEST(ExplorationStateMachineTest, CompletionMatrixOnlyAcceptsSelectingWithoutGoal) {
  for (const ExplorationState source : kAllStates) {
    auto machine = MachineIn(source);
    if (source == ExplorationState::kSelectingFrontier) {
      machine->CompleteNoReachableFrontier();
      EXPECT_EQ(machine->state(), ExplorationState::kCompleted);
      EXPECT_EQ(machine->task_id(), "task-A");
      EXPECT_EQ(machine->reason_code(),
                "COMPLETED_NO_REACHABLE_FRONTIER");
      EXPECT_FALSE(machine->active_goal().has_value());
    } else {
      const auto before = Snapshot(*machine);
      EXPECT_THROW(machine->CompleteNoReachableFrontier(), std::logic_error);
      ExpectSnapshot(*machine, before);
    }
  }
}

TEST(ExplorationStateMachineTest, GoalFactoryRejectsEveryInvalidProvenanceField) {
  const std::array<FrontierCluster, 1> frontiers{Frontier()};
  const auto expect_invalid = [&](const CandidateView& candidate,
                                  std::string request = "request") {
    EXPECT_THROW((void)MakeActiveGoal(candidate, frontiers, std::move(request)),
                 std::invalid_argument);
  };

  expect_invalid(Candidate(), "");
  auto candidate = Candidate();
  candidate.frontier_index = 1U;
  expect_invalid(candidate);
  for (double Pose2::*member : {&Pose2::x, &Pose2::y, &Pose2::yaw}) {
    candidate = Candidate();
    candidate.pose.*member = std::numeric_limits<double>::quiet_NaN();
    expect_invalid(candidate);
    candidate = Candidate();
    candidate.pose.*member = std::numeric_limits<double>::infinity();
    expect_invalid(candidate);
  }
  for (const double invalid : {
           -1.0, std::numeric_limits<double>::quiet_NaN(),
           std::numeric_limits<double>::infinity()}) {
    candidate = Candidate();
    candidate.frontier_distance_m = invalid;
    expect_invalid(candidate);
  }
  candidate = Candidate(nullptr);
  expect_invalid(candidate);
  candidate = Candidate(
      std::make_shared<const std::vector<std::int64_t>>());
  expect_invalid(candidate);
  candidate = Candidate();
  candidate.frontier_id = 78U;
  expect_invalid(candidate);
  candidate = Candidate(
      std::make_shared<const std::vector<std::int64_t>>(
          std::initializer_list<std::int64_t>{4, 5, 7}));
  expect_invalid(candidate);

  auto collision = Frontier(77U, {9, 9, 9});
  const std::array<FrontierCluster, 1> other_span{std::move(collision)};
  EXPECT_THROW((void)MakeActiveGoal(Candidate(), other_span, "request"),
               std::invalid_argument);
}

TEST(ExplorationStateMachineTest, GoalFactoryDeepCopiesAllPayloadAndMatchesOnlyFullKey) {
  auto mutable_owner =
      std::make_shared<std::vector<std::int64_t>>(
          std::initializer_list<std::int64_t>{4, 5, 6});
  CandidateView candidate = Candidate(mutable_owner);
  std::vector<FrontierCluster> frontiers{Frontier()};
  std::string request = "owned-request";
  ActiveGoal goal = MakeActiveGoal(candidate, frontiers, request);

  request.assign("mutated-request");
  candidate.key = {9, 9, 9};
  candidate.pose = {9.0, 9.0, 9.0};
  mutable_owner->assign({8, 8, 8});
  frontiers.front().canonical_key.assign({7, 7, 7});
  frontiers.clear();
  mutable_owner.reset();

  EXPECT_EQ(goal.candidate_id(), 42U);
  EXPECT_EQ(goal.frontier_id(), 77U);
  EXPECT_EQ(goal.kind(), GoalKind::kTaskFrontier);
  EXPECT_EQ(goal.boundary_approach_identity(), nullptr);
  EXPECT_EQ(std::vector<std::int64_t>(goal.frontier_canonical_key().begin(),
                                      goal.frontier_canonical_key().end()),
            (std::vector<std::int64_t>{4, 5, 6}));
  EXPECT_EQ(goal.candidate_key(), (CandidateKey{1000, -2000, 135}));
  ExpectPose(goal.target(), Pose2{1.25, -2.5, 0.75});
  EXPECT_EQ(goal.request_id(), "owned-request");
  EXPECT_EQ(goal.replan_count(), 0U);

  const std::array<FrontierCluster, 2> snapshots{
      Frontier(42U, {4, 5, 6}), Frontier(77U, {1, 2, 3})};
  EXPECT_TRUE(goal.MatchesAnyFrontier(snapshots));
  const std::array<FrontierCluster, 1> same_display_wrong_key{
      Frontier(77U, {4, 5, 7})};
  EXPECT_FALSE(goal.MatchesAnyFrontier(same_display_wrong_key));

  const std::array<FrontierCluster, 1> key_a{Frontier(5U, {1, 2})};
  const std::array<FrontierCluster, 1> key_b{Frontier(5U, {1, 3})};
  auto view_a = Candidate(std::make_shared<const std::vector<std::int64_t>>(
      std::initializer_list<std::int64_t>{1, 2}));
  auto view_b = Candidate(std::make_shared<const std::vector<std::int64_t>>(
      std::initializer_list<std::int64_t>{1, 3}));
  view_a.id = view_b.id = 5U;
  view_a.frontier_id = view_b.frontier_id = 5U;
  const ActiveGoal goal_a = MakeActiveGoal(view_a, key_a, "a");
  const ActiveGoal goal_b = MakeActiveGoal(view_b, key_b, "b");
  EXPECT_TRUE(goal_a.MatchesAnyFrontier(key_a));
  EXPECT_FALSE(goal_a.MatchesAnyFrontier(key_b));
  EXPECT_FALSE(goal_b.MatchesAnyFrontier(key_a));
  EXPECT_TRUE(goal_b.MatchesAnyFrontier(key_b));
}

TEST(ExplorationStateMachineTest,
     BoundaryApproachGoalOwnsTypedIdentityWithoutFrontierMembership) {
  ActiveGoal goal = [] {
    auto identity = BoundaryApproachGoalIdentity{
        .intent_cell = {17, -9},
        .candidate_key = {250, -500, 90},
        .candidate_kind = ApproachCandidateKind::kRotation,
    };
    return MakeBoundaryApproachActiveGoal(314U, std::move(identity),
                                          {2.5, -5.0, 1.57}, "approach-1");
  }();

  EXPECT_EQ(goal.kind(), GoalKind::kBoundaryApproach);
  EXPECT_TRUE(goal.frontier_canonical_key().empty());
  const auto* identity = goal.boundary_approach_identity();
  ASSERT_NE(identity, nullptr);
  EXPECT_EQ(identity->intent_cell, (GridIndex{17, -9}));
  EXPECT_EQ(identity->candidate_key, (CandidateKey{250, -500, 90}));
  EXPECT_EQ(identity->candidate_kind, ApproachCandidateKind::kRotation);
  EXPECT_EQ(goal.candidate_key(), identity->candidate_key);
  ExpectPose(goal.target(), Pose2{2.5, -5.0, 1.57});
  EXPECT_EQ(goal.request_id(), "approach-1");

  const std::array<FrontierCluster, 1> frontiers{Frontier(314U, {17, -9})};
  EXPECT_FALSE(goal.MatchesAnyFrontier(frontiers));
}

TEST(ExplorationStateMachineTest, ActiveGoalIsOneShotAndCommitRejectsConsumedSource) {
  static_assert(std::is_move_constructible_v<ActiveGoal>);
  static_assert(!std::is_copy_constructible_v<ActiveGoal>);
  static_assert(!std::is_copy_assignable_v<ActiveGoal>);
  static_assert(!std::is_move_assignable_v<ActiveGoal>);
  static_assert(!std::is_aggregate_v<ActiveGoal>);
  static_assert(!std::is_default_constructible_v<ActiveGoal>);
  static_assert(std::is_same_v<decltype(&ExplorationStateMachine::CommitGoal),
                               void (ExplorationStateMachine::*)(ActiveGoal&&)>);

  auto machine = MachineIn(ExplorationState::kPlanning);
  auto one_shot = Goal("one-shot");
  machine->CommitGoal(std::move(one_shot));
  machine->ReleaseGoal(GoalReleaseReason::kArrived);
  machine->BeginPlanning();
  const auto before = Snapshot(*machine);
  EXPECT_THROW(machine->CommitGoal(std::move(one_shot)), std::invalid_argument);
  ExpectSnapshot(*machine, before);

  auto fresh = Goal("through-intermediate");
  ActiveGoal intermediate(std::move(fresh));
  machine->CommitGoal(std::move(intermediate));
  EXPECT_EQ(machine->state(), ExplorationState::kExecuting);
  ASSERT_TRUE(machine->active_goal().has_value());
  EXPECT_EQ(machine->active_goal()->request_id(), "through-intermediate");
}

TEST(ExplorationStateMachineTest, ReleaseReasonsAreTypedAndMappedLiterally) {
  struct Case {
    GoalReleaseReason reason;
    const char* code;
  };
  constexpr std::array<Case, 6> cases{{
      {GoalReleaseReason::kArrived, "ARRIVED"},
      {GoalReleaseReason::kNoPath, "NO_PATH"},
      {GoalReleaseReason::kCandidateInvalid, "CANDIDATE_INVALID"},
      {GoalReleaseReason::kFrontierDisappeared, "FRONTIER_DISAPPEARED"},
      {GoalReleaseReason::kInformationGainZero, "INFORMATION_GAIN_ZERO"},
      {GoalReleaseReason::kLocalSegmentCompleted, "LOCAL_SEGMENT_COMPLETED"},
  }};
  for (const auto& test_case : cases) {
    auto machine = MachineIn(ExplorationState::kExecuting);
    machine->ReleaseGoal(test_case.reason);
    EXPECT_EQ(machine->state(), ExplorationState::kSelectingFrontier);
    EXPECT_EQ(machine->reason_code(), test_case.code);
    EXPECT_FALSE(machine->active_goal().has_value());
  }

  auto machine = MachineIn(ExplorationState::kExecuting);
  const auto before = Snapshot(*machine);
  EXPECT_THROW(machine->ReleaseGoal(static_cast<GoalReleaseReason>(255U)),
               std::invalid_argument);
  ExpectSnapshot(*machine, before);
}

TEST(ExplorationStateMachineTest, TypedReplanningPreservesPayloadAndSeparatesRetryAuthority) {
  auto machine = MachineIn(ExplorationState::kExecuting, 2U);
  const auto original = Snapshot(*machine).goal.value();
  EXPECT_EQ(machine->BeginReplanning(ReplanCause::kRollingSegment),
            ReplanResult::kStarted);
  EXPECT_EQ(machine->reason_code(), "ROLLING_SEGMENT");
  ExpectGoal(*machine->active_goal(), original);
  machine->ResumeExecution();
  ExpectGoal(*machine->active_goal(), original);

  auto after_first_stuck = original;
  after_first_stuck.replan_count = 1U;
  EXPECT_EQ(machine->BeginReplanning(ReplanCause::kStuckRecovery),
            ReplanResult::kStarted);
  EXPECT_EQ(machine->reason_code(), "STUCK_RETRY");
  ExpectGoal(*machine->active_goal(), after_first_stuck);
  machine->ResumeExecution();
  ExpectGoal(*machine->active_goal(), after_first_stuck);

  EXPECT_EQ(machine->BeginReplanning(ReplanCause::kRollingSegment),
            ReplanResult::kStarted);
  ExpectGoal(*machine->active_goal(), after_first_stuck);
  machine->ResumeExecution();
  ExpectGoal(*machine->active_goal(), after_first_stuck);

  auto after_second_stuck = original;
  after_second_stuck.replan_count = 2U;
  EXPECT_EQ(machine->BeginReplanning(ReplanCause::kStuckRecovery),
            ReplanResult::kStarted);
  ExpectGoal(*machine->active_goal(), after_second_stuck);
  machine->ResumeExecution();
  ExpectGoal(*machine->active_goal(), after_second_stuck);
  EXPECT_EQ(machine->BeginReplanning(ReplanCause::kStuckRecovery),
            ReplanResult::kExhausted);
  EXPECT_EQ(machine->state(), ExplorationState::kSelectingFrontier);
  EXPECT_EQ(machine->reason_code(), "STUCK_RETRIES_EXHAUSTED");
  EXPECT_FALSE(machine->active_goal().has_value());

  auto zero = MachineIn(ExplorationState::kExecuting, 0U);
  EXPECT_EQ(zero->BeginReplanning(ReplanCause::kStuckRecovery),
            ReplanResult::kExhausted);
  EXPECT_EQ(zero->state(), ExplorationState::kSelectingFrontier);
  EXPECT_FALSE(zero->active_goal().has_value());
}

TEST(ExplorationStateMachineTest, MaximumUint8RetryCountNeverWraps) {
  auto machine = MachineIn(ExplorationState::kExecuting,
                           std::numeric_limits<std::uint8_t>::max());
  const auto stable = Snapshot(*machine).goal.value();
  for (std::uint16_t count = 1U; count <= 253U; ++count) {
    ASSERT_EQ(machine->BeginReplanning(ReplanCause::kStuckRecovery),
              ReplanResult::kStarted);
    ASSERT_TRUE(machine->active_goal().has_value());
    EXPECT_EQ(machine->active_goal()->replan_count(), count);
    machine->ResumeExecution();
  }

  auto at_254 = stable;
  at_254.replan_count = 254U;
  EXPECT_EQ(machine->BeginReplanning(ReplanCause::kStuckRecovery),
            ReplanResult::kStarted);
  ExpectGoal(*machine->active_goal(), at_254);
  machine->ResumeExecution();
  ExpectGoal(*machine->active_goal(), at_254);

  auto at_255 = stable;
  at_255.replan_count = 255U;
  EXPECT_EQ(machine->BeginReplanning(ReplanCause::kStuckRecovery),
            ReplanResult::kStarted);
  ExpectGoal(*machine->active_goal(), at_255);
  machine->ResumeExecution();
  ExpectGoal(*machine->active_goal(), at_255);
  EXPECT_EQ(machine->BeginReplanning(ReplanCause::kStuckRecovery),
            ReplanResult::kExhausted);
  EXPECT_FALSE(machine->active_goal().has_value());
}

TEST(ExplorationStateMachineTest, ForgedReplanCauseHasStrongGuarantee) {
  auto machine = MachineIn(ExplorationState::kExecuting);
  const auto before = Snapshot(*machine);
  EXPECT_THROW(machine->BeginReplanning(static_cast<ReplanCause>(255U)),
               std::invalid_argument);
  ExpectSnapshot(*machine, before);
}

}  // namespace
}  // namespace lunar::pure_exploration

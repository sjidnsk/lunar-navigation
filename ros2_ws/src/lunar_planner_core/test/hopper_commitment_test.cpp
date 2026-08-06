#include <optional>
#include <stop_token>
#include <string>

#include <gtest/gtest.h>

#include "hopper/commitment_state_machine.hpp"
#include "lunar_planner_core/planner.hpp"
#include "test_fixtures.hpp"

namespace lunar::planning {
namespace {

TEST(HopperCommitment, FollowsReadyCommittedFlightAndLandingLifecycle) {
  hopper::CommitmentStateMachine machine;
  HopperExecutionContext state{};

  auto transition = machine.Transition(
      state,
      hopper::CommitmentEvent{
          .type = hopper::CommitmentEventType::kPublishCertifiedHop,
          .plan_id = "plan-1",
          .segment_id = "hop-1",
      });
  ASSERT_TRUE(transition.accepted) << transition.reason_code;
  EXPECT_EQ(transition.next.state, HopperExecutionState::kJumpReady);
  EXPECT_TRUE(machine.MayPublishNewHop(transition.next));

  transition = machine.Transition(
      transition.next,
      hopper::CommitmentEvent{
          .type = hopper::CommitmentEventType::kLockJumpBoundary,
          .plan_id = "plan-1",
          .segment_id = "hop-1",
      });
  ASSERT_TRUE(transition.accepted) << transition.reason_code;
  EXPECT_EQ(transition.next.state, HopperExecutionState::kJumpCommitted);
  EXPECT_FALSE(machine.MayPublishNewHop(transition.next));

  transition = machine.Transition(
      transition.next,
      hopper::CommitmentEvent{
          .type = hopper::CommitmentEventType::kDetectLaunch,
          .plan_id = std::nullopt,
          .segment_id = "hop-1",
      });
  ASSERT_TRUE(transition.accepted) << transition.reason_code;
  EXPECT_EQ(transition.next.state, HopperExecutionState::kInFlight);

  transition = machine.Transition(
      transition.next,
      hopper::CommitmentEvent{
          .type = hopper::CommitmentEventType::kDetectStableLanding,
          .plan_id = std::nullopt,
          .segment_id = "hop-1",
      });
  ASSERT_TRUE(transition.accepted) << transition.reason_code;
  EXPECT_EQ(transition.next.state, HopperExecutionState::kLandedHold);
  EXPECT_EQ(transition.next.active_plan_id, "plan-1");
  EXPECT_EQ(transition.next.active_segment_id, "hop-1");
  EXPECT_TRUE(machine.MayPublishNewHop(transition.next));
}

class ProtectedHopperStateTest
    : public testing::TestWithParam<HopperExecutionState> {};

TEST_P(ProtectedHopperStateTest, ContinuesCommittedBoundaryWithoutNewHop) {
  Planner planner;
  auto input = test::MakeValidHopperInput();
  input.previous_execution = ExecutionContext{HopperExecutionContext{
      .state = GetParam(),
      .active_plan_id = "active-plan",
      .active_segment_id = "active-hop",
  }};

  const PlannerOutput output = planner.Plan(input);

  EXPECT_EQ(output.outcome, PlanningOutcome::kSafeFrontierReferenceAvailable);
  EXPECT_EQ(output.directive, ExecutionDirective::kContinueCommittedHop);
  EXPECT_EQ(output.reason_code, "COMMITTED_HOP_CONTINUES");
  EXPECT_FALSE(output.reference.has_value());
}

INSTANTIATE_TEST_SUITE_P(
    CommittedAndInFlight,
    ProtectedHopperStateTest,
    testing::Values(
        HopperExecutionState::kJumpCommitted,
        HopperExecutionState::kInFlight));

TEST(HopperCommitment, InvalidatesIncompleteProtectedContextFailClosed) {
  Planner planner;
  auto input = test::MakeValidHopperInput();
  input.previous_execution = ExecutionContext{HopperExecutionContext{
      .state = HopperExecutionState::kJumpCommitted,
      .active_plan_id = "active-plan",
      .active_segment_id = std::nullopt,
  }};

  const PlannerOutput output = planner.Plan(input);

  EXPECT_EQ(output.outcome, PlanningOutcome::kActiveReferenceInvalidated);
  EXPECT_EQ(output.directive, ExecutionDirective::kNoSafeReference);
  EXPECT_EQ(output.reason_code, "COMMITTED_HOP_CONTEXT_INVALID");
  EXPECT_FALSE(output.reference.has_value());
}

TEST(HopperCommitment, OrdinaryCancellationCannotReplaceCommittedHop) {
  Planner planner;
  auto input = test::MakeValidHopperInput();
  input.previous_execution = ExecutionContext{HopperExecutionContext{
      .state = HopperExecutionState::kJumpCommitted,
      .active_plan_id = "active-plan",
      .active_segment_id = "active-hop",
  }};
  std::stop_source stop_source;
  stop_source.request_stop();
  input.stop_token = stop_source.get_token();

  const PlannerOutput output = planner.Plan(input);

  EXPECT_EQ(output.outcome, PlanningOutcome::kSafeFrontierReferenceAvailable);
  EXPECT_EQ(output.directive, ExecutionDirective::kContinueCommittedHop);
  EXPECT_EQ(output.reason_code, "COMMITTED_HOP_CONTINUES");
  EXPECT_FALSE(output.reference.has_value());
}

TEST(HopperCommitment, RejectsReplacementAfterBoundaryLock) {
  hopper::CommitmentStateMachine machine;
  const HopperExecutionContext committed{
      .state = HopperExecutionState::kJumpCommitted,
      .active_plan_id = "plan-1",
      .active_segment_id = "hop-1",
  };

  const auto transition = machine.Transition(
      committed,
      hopper::CommitmentEvent{
          .type = hopper::CommitmentEventType::kPublishCertifiedHop,
          .plan_id = "plan-2",
          .segment_id = "hop-2",
      });

  EXPECT_FALSE(transition.accepted);
  EXPECT_EQ(transition.reason_code, "COMMITTED_HOP_CANNOT_BE_REPLACED");
  EXPECT_EQ(transition.next.state, HopperExecutionState::kJumpCommitted);
  EXPECT_EQ(transition.next.active_segment_id, "hop-1");
}

}  // namespace
}  // namespace lunar::planning

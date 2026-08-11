#include "lunar_goal_coordinator/goal_state_machine.hpp"

#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

namespace lunar::goal_coordinator {
namespace {

using lunar::planning::GridLayer;
using lunar::planning::GridMap;
using lunar::planning::TimePoint;
using lunar::planning::Vec3;

std::shared_ptr<GridMap> MakeMap(const std::int64_t stamp_ns = 20) {
  auto map = std::make_shared<GridMap>();
  map->frame_id = "map";
  map->stamp = TimePoint{.nanoseconds_since_epoch = stamp_ns};
  map->width = 5U;
  map->height = 5U;
  map->resolution_m = 1.0;
  map->origin_m = Vec3{.x = 0.0, .y = 0.0, .z = 0.0};
  map->layers.emplace(
      "valid_mask", GridLayer{.values = std::vector<std::uint8_t>(25U, 1U)});
  map->layers.emplace(
      "obstacle", GridLayer{.values = std::vector<std::uint8_t>(25U, 0U)});
  map->layers.emplace(
      "forbidden", GridLayer{.values = std::vector<std::uint8_t>(25U, 0U)});
  map->layers.emplace(
      "elevation", GridLayer{.values = std::vector<float>(25U, 3.0F)});
  return map;
}

PlanningSnapshot MakeSnapshot(
    std::shared_ptr<const GridMap> map, const std::int64_t stamp_ns = 20,
    const double x_m = 0.5, const double y_m = 0.5,
    const double yaw_rad = 0.0) {
  return PlanningSnapshot{
      .global_map = std::move(map),
      .local_map_stamp_ns = stamp_ns,
      .transform_stamp_ns = stamp_ns,
      .robot = RobotState{
          .stamp_ns = stamp_ns,
          .x_m = x_m,
          .y_m = y_m,
          .yaw_rad = yaw_rad,
      },
  };
}

GoalPose MakeGoal(
    const double x_m = 4.5, const double y_m = 4.5,
    const std::int64_t stamp_ns = 10) {
  return GoalPose{
      .frame_id = "map",
      .stamp_ns = stamp_ns,
      .x_m = x_m,
      .y_m = y_m,
      .z_m = 99.0,
      .qx = 0.0,
      .qy = 0.0,
      .qz = 0.0,
      .qw = 2.0,
  };
}

GoalStateMachine ReadyMachine() {
  GoalStateMachine machine;
  machine.SetSession("session-a");
  EXPECT_TRUE(machine.UpdateSnapshot(MakeSnapshot(MakeMap())));
  EXPECT_TRUE(machine.SubmitGoal(MakeGoal(), "target-a").accepted);
  EXPECT_EQ(machine.state(), CoordinatorState::kReady);
  return machine;
}

PlannerReply SuccessfulReply(const PlanningCommand& command) {
  return PlannerReply{
      .request_id = command.request_id,
      .mission_id = command.mission_id,
      .mission_revision = command.mission_revision,
      .planning_outcome = 0U,
      .execution_directive = 0U,
      .reason_code = "REFERENCE_READY",
      .has_reference = true,
      .reference_platform_type = 1U,
      .plan_id = "plan-a",
  };
}

ExecutionEvent SegmentComplete(
    const PlanningCommand& command, const std::int64_t stamp_ns = 30) {
  static_cast<void>(command);
  return ExecutionEvent{
      .stamp_ns = stamp_ns,
      .plan_id = "plan-a",
      .state = ExecutionState::kSegmentComplete,
      .reason_code = "SEGMENT_COMPLETE",
  };
}

TEST(GoalStateMachineTest, AcceptsFiniteMapGoalAndNormalizesQuaternion) {
  GoalStateMachine machine;
  machine.SetSession("session-a");
  ASSERT_TRUE(machine.UpdateSnapshot(MakeSnapshot(MakeMap())));
  GoalPose goal = MakeGoal();
  goal.qz = 1.0;
  goal.qw = 1.0;

  const GoalAcceptance accepted = machine.SubmitGoal(goal, "target-a");

  ASSERT_TRUE(accepted.accepted) << accepted.reason_code;
  const auto& q = accepted.normalized_quaternion;
  EXPECT_NEAR(
      std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]),
      1.0, 1.0e-12);
  EXPECT_NEAR(accepted.yaw_rad, M_PI_2, 1.0e-12);
  EXPECT_EQ(machine.state(), CoordinatorState::kReady);
}

TEST(
    GoalStateMachineTest,
    RejectsWrongFrameUnknownObstacleForbiddenOrDisconnectedGoal) {
  {
    GoalStateMachine machine;
    machine.SetSession("session-a");
    ASSERT_TRUE(machine.UpdateSnapshot(MakeSnapshot(MakeMap())));
    GoalPose goal = MakeGoal();
    goal.frame_id = "odom";
    EXPECT_EQ(machine.SubmitGoal(goal, "a").reason_code, "GOAL_WRONG_FRAME");
  }
  for (const std::string kind : {"unknown", "obstacle", "forbidden"}) {
    auto map = MakeMap();
    const std::size_t goal_index = 24U;
    if (kind == "unknown") {
      std::get<std::vector<std::uint8_t>>(map->layers.at("valid_mask").values)
          [goal_index] = 0U;
    } else {
      std::get<std::vector<std::uint8_t>>(map->layers.at(kind).values)
          [goal_index] = 1U;
    }
    GoalStateMachine machine;
    machine.SetSession("session-a");
    ASSERT_TRUE(machine.UpdateSnapshot(MakeSnapshot(map)));
    const auto result = machine.SubmitGoal(MakeGoal(), kind);
    EXPECT_FALSE(result.accepted);
    EXPECT_EQ(
        result.reason_code,
        kind == "unknown" ? "GOAL_UNKNOWN"
                           : (kind == "obstacle" ? "GOAL_OBSTACLE"
                                                 : "GOAL_FORBIDDEN"));
  }
  {
    auto map = MakeMap();
    auto& valid = std::get<std::vector<std::uint8_t>>(
        map->layers.at("valid_mask").values);
    for (std::size_t x = 0U; x < 5U; ++x) {
      valid[2U * 5U + x] = 0U;
    }
    GoalStateMachine machine;
    machine.SetSession("session-a");
    ASSERT_TRUE(machine.UpdateSnapshot(MakeSnapshot(map)));
    EXPECT_EQ(
        machine.SubmitGoal(MakeGoal(), "disconnected").reason_code,
        "GOAL_DISCONNECTED");
  }
}

TEST(GoalStateMachineTest, RejectsImplicitReplacementUntilCanceled) {
  auto machine = ReadyMachine();
  EXPECT_EQ(
      machine.SubmitGoal(MakeGoal(3.5, 3.5), "target-b").reason_code,
      "ACTIVE_GOAL_REQUIRES_CANCEL");
  EXPECT_TRUE(machine.BeginCancel());
  EXPECT_EQ(machine.state(), CoordinatorState::kCanceling);
  machine.CompleteCancel(true);
  EXPECT_TRUE(machine.SubmitGoal(MakeGoal(3.5, 3.5), "target-b").accepted);
}

TEST(GoalStateMachineTest, PublishesOnlyActivateNewWheeledReference) {
  {
    auto machine = ReadyMachine();
    const auto command = machine.TakePlanningCommand();
    ASSERT_TRUE(command);
    EXPECT_TRUE(machine.HandlePlannerReply(SuccessfulReply(*command)));
    EXPECT_EQ(machine.state(), CoordinatorState::kExecuting);
  }
  for (const int invalid_case : {0, 1, 2, 3}) {
    auto machine = ReadyMachine();
    const auto command = machine.TakePlanningCommand();
    ASSERT_TRUE(command);
    PlannerReply reply = SuccessfulReply(*command);
    if (invalid_case == 0) {
      reply.execution_directive = 1U;
    } else if (invalid_case == 1) {
      reply.has_reference = false;
    } else if (invalid_case == 2) {
      reply.reference_platform_type = 2U;
    } else {
      reply.plan_id.clear();
    }
    EXPECT_FALSE(machine.HandlePlannerReply(reply));
    EXPECT_EQ(machine.state(), CoordinatorState::kHold);
  }
}

TEST(GoalStateMachineTest, SegmentCompleteWaitsForStrictlyNewerStateAndMap) {
  auto machine = ReadyMachine();
  const auto command = machine.TakePlanningCommand();
  ASSERT_TRUE(command);
  ASSERT_TRUE(machine.HandlePlannerReply(SuccessfulReply(*command)));
  machine.HandleExecutionEvent(SegmentComplete(*command));
  EXPECT_EQ(machine.state(), CoordinatorState::kWaitingForSnapshot);

  EXPECT_TRUE(machine.UpdateSnapshot(MakeSnapshot(MakeMap(30), 30)));
  EXPECT_EQ(machine.state(), CoordinatorState::kWaitingForSnapshot);
  EXPECT_TRUE(machine.UpdateSnapshot(MakeSnapshot(MakeMap(31), 31)));
  EXPECT_EQ(machine.state(), CoordinatorState::kReady);
}

TEST(GoalStateMachineTest, RollsAgainOutsideTolerance) {
  auto machine = ReadyMachine();
  const auto first = machine.TakePlanningCommand();
  ASSERT_TRUE(first);
  ASSERT_TRUE(machine.HandlePlannerReply(SuccessfulReply(*first)));
  machine.HandleExecutionEvent(SegmentComplete(*first));
  ASSERT_TRUE(machine.UpdateSnapshot(
      MakeSnapshot(MakeMap(31), 31, 2.5, 2.5, 0.0)));

  const auto second = machine.TakePlanningCommand();
  ASSERT_TRUE(second);
  EXPECT_EQ(second->mission_id, first->mission_id);
  EXPECT_EQ(second->mission_revision, first->mission_revision);
  EXPECT_NE(second->request_id, first->request_id);
  EXPECT_FALSE(second->publish_mission);
}

TEST(GoalStateMachineTest, SucceedsInsideHalfMeterAndFifteenDegrees) {
  auto machine = ReadyMachine();
  const auto command = machine.TakePlanningCommand();
  ASSERT_TRUE(command);
  ASSERT_TRUE(machine.HandlePlannerReply(SuccessfulReply(*command)));
  machine.HandleExecutionEvent(SegmentComplete(*command));

  ASSERT_TRUE(machine.UpdateSnapshot(
      MakeSnapshot(MakeMap(31), 31, 4.2, 4.3, 0.1)));
  EXPECT_EQ(machine.state(), CoordinatorState::kSucceeded);
  EXPECT_TRUE(machine.ConsumeHoldRequest());
}

TEST(GoalStateMachineTest, PlannerOrExecutorFailureEntersHoldWithStableReason) {
  auto machine = ReadyMachine();
  const auto command = machine.TakePlanningCommand();
  ASSERT_TRUE(command);
  PlannerReply reply = SuccessfulReply(*command);
  reply.planning_outcome = 2U;
  reply.reason_code = "NO_KNOWN_SAFE_ROUTE";
  EXPECT_FALSE(machine.HandlePlannerReply(reply));
  EXPECT_EQ(machine.state(), CoordinatorState::kHold);
  EXPECT_EQ(machine.reason_code(), "PLANNER_NO_KNOWN_SAFE_ROUTE");
  machine.ForceHold("LATER_FAILURE");
  EXPECT_EQ(machine.reason_code(), "PLANNER_NO_KNOWN_SAFE_ROUTE");
  EXPECT_TRUE(machine.ConsumeHoldRequest());
}

}  // namespace
}  // namespace lunar::goal_coordinator

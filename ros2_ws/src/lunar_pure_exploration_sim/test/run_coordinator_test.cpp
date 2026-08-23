#include <array>
#include <cstdint>
#include <string>

#include <gtest/gtest.h>
#include <rclcpp/time.hpp>

#include "lunar_pure_exploration_sim/run_coordinator.hpp"

namespace lunar::pure_exploration_sim {
namespace {

CoordinatorReadiness FullyReady() {
  CoordinatorReadiness readiness;
  readiness.global_map_received = true;
  readiness.local_map_received = true;
  readiness.odometry_received = true;
  readiness.tf_chain_received = true;
  readiness.initial_status_received = true;
  readiness.planner_action_ready = true;
  return readiness;
}

TEST(CoordinatorReadinessTest, EveryRequiredInputIndependentlyBlocksStart) {
  using Member = bool CoordinatorReadiness::*;
  constexpr std::array<Member, 6> required_inputs{
      &CoordinatorReadiness::global_map_received,
      &CoordinatorReadiness::local_map_received,
      &CoordinatorReadiness::odometry_received,
      &CoordinatorReadiness::tf_chain_received,
      &CoordinatorReadiness::initial_status_received,
      &CoordinatorReadiness::planner_action_ready,
  };

  for (const Member missing : required_inputs) {
    auto readiness = FullyReady();
    readiness.*missing = false;
    EXPECT_FALSE(readiness.ShouldStart());
  }
}

TEST(CoordinatorReadinessTest, StartDecisionIsOneShotAfterMarkStarted) {
  auto readiness = FullyReady();
  ASSERT_TRUE(readiness.ShouldStart());

  readiness.MarkStarted();

  EXPECT_FALSE(readiness.ShouldStart());
  readiness.planner_action_ready = false;
  readiness.planner_action_ready = true;
  EXPECT_FALSE(readiness.ShouldStart());
}

TEST(CoordinatorReadinessTest, BuildsExactApprovedStartTask) {
  const auto task = MakeExplorationStartTask(20260824U, rclcpp::Time{12, 34});

  EXPECT_EQ(task.header.frame_id, "map");
  EXPECT_EQ(task.header.stamp.sec, 12);
  EXPECT_EQ(task.header.stamp.nanosec, 34U);
  EXPECT_EQ(task.task_id, "jazzy-300m-20260824");
  EXPECT_EQ(task.command, decltype(task)::START);
  ASSERT_EQ(task.boundary.points.size(), 4U);
  EXPECT_FLOAT_EQ(task.boundary.points[0].x, -145.0F);
  EXPECT_FLOAT_EQ(task.boundary.points[0].y, -145.0F);
  EXPECT_FLOAT_EQ(task.boundary.points[1].x, 145.0F);
  EXPECT_FLOAT_EQ(task.boundary.points[1].y, -145.0F);
  EXPECT_FLOAT_EQ(task.boundary.points[2].x, 145.0F);
  EXPECT_FLOAT_EQ(task.boundary.points[2].y, 145.0F);
  EXPECT_FLOAT_EQ(task.boundary.points[3].x, -145.0F);
  EXPECT_FLOAT_EQ(task.boundary.points[3].y, 145.0F);
}

TEST(CoordinatorReadinessTest, AcceptsOnlyTheExplorersInitialIdleStatus) {
  lunar_pure_exploration_msgs::msg::PureExplorationStatus status;
  status.state = decltype(status)::IDLE;
  EXPECT_TRUE(IsInitialExplorationStatus(status));

  status.task_id = "old-task";
  EXPECT_FALSE(IsInitialExplorationStatus(status));
  status.task_id.clear();
  status.state = decltype(status)::COMPLETED;
  EXPECT_FALSE(IsInitialExplorationStatus(status));
}

TEST(CoordinatorReadinessTest, TfChainCanArriveAcrossMultipleMessages) {
  RequiredTfChain chain;
  tf2_msgs::msg::TFMessage first;
  first.transforms.resize(1U);
  first.transforms[0].header.frame_id = "map";
  first.transforms[0].child_frame_id = "odom";
  chain.Observe(first);
  EXPECT_FALSE(chain.complete());

  tf2_msgs::msg::TFMessage second;
  second.transforms.resize(1U);
  second.transforms[0].header.frame_id = "odom";
  second.transforms[0].child_frame_id = "base_link";
  chain.Observe(second);
  EXPECT_TRUE(chain.complete());
}

}  // namespace
}  // namespace lunar::pure_exploration_sim

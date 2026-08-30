#include <array>
#include <chrono>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>
#include <geometry_msgs/msg/twist.hpp>
#include <rclcpp/time.hpp>
#include <rclcpp_action/create_server.hpp>

#include "lunar_pure_exploration_sim/run_coordinator.hpp"

namespace lunar::pure_exploration_sim {
namespace {

using Action = lunar_planning_msgs::action::PlanMotion;
using Task = lunar_pure_exploration_msgs::msg::PureExplorationTask;
using namespace std::chrono_literals;

class RosContextTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    if (!rclcpp::ok()) {
      rclcpp::init(0, nullptr);
    }
  }

  static void TearDownTestSuite() {
    if (rclcpp::ok()) {
      rclcpp::shutdown();
    }
  }
};

template <typename Predicate>
bool SpinUntil(rclcpp::executors::SingleThreadedExecutor& executor,
               Predicate&& predicate,
               const std::chrono::milliseconds timeout = 3s) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    executor.spin_some();
    if (predicate()) {
      return true;
    }
    std::this_thread::sleep_for(2ms);
  }
  executor.spin_some();
  return predicate();
}

CoordinatorReadiness FullyReady() {
  CoordinatorReadiness readiness;
  readiness.global_map_received = true;
  readiness.local_map_received = true;
  readiness.odometry_received = true;
  readiness.tf_chain_received = true;
  readiness.initial_status_received = true;
  readiness.planner_action_ready = true;
  readiness.controller_publisher_unique = true;
  readiness.controller_command_received = true;
  for (int poll = 0; poll < 20; ++poll) {
    readiness.ObservePoll();
  }
  return readiness;
}

TEST(CoordinatorReadinessTest, RequiresTwentyConsecutiveStablePollsBeforeStart) {
  CoordinatorReadiness readiness;
  readiness.global_map_received = true;
  readiness.local_map_received = true;
  readiness.odometry_received = true;
  readiness.tf_chain_received = true;
  readiness.initial_status_received = true;
  readiness.planner_action_ready = true;
  readiness.controller_publisher_unique = true;
  readiness.controller_command_received = true;

  for (int poll = 0; poll < 19; ++poll) {
    readiness.ObservePoll();
    EXPECT_FALSE(readiness.ShouldStart());
  }
  readiness.planner_action_ready = false;
  readiness.ObservePoll();
  readiness.planner_action_ready = true;
  for (int poll = 0; poll < 19; ++poll) {
    readiness.ObservePoll();
    EXPECT_FALSE(readiness.ShouldStart());
  }
  readiness.ObservePoll();
  EXPECT_TRUE(readiness.ShouldStart());
}

TEST(CoordinatorReadinessTest, EveryRequiredInputIndependentlyBlocksStart) {
  using Member = bool CoordinatorReadiness::*;
  constexpr std::array<Member, 8> required_inputs{
      &CoordinatorReadiness::global_map_received,
      &CoordinatorReadiness::local_map_received,
      &CoordinatorReadiness::odometry_received,
      &CoordinatorReadiness::tf_chain_received,
      &CoordinatorReadiness::initial_status_received,
      &CoordinatorReadiness::planner_action_ready,
      &CoordinatorReadiness::controller_publisher_unique,
      &CoordinatorReadiness::controller_command_received,
  };

  for (const Member missing : required_inputs) {
    auto readiness = FullyReady();
    readiness.*missing = false;
    EXPECT_FALSE(readiness.ShouldStart());
  }
}

TEST(CoordinatorReadinessTest, AcceptsOnlyFullyFiniteControllerCommands) {
  geometry_msgs::msg::Twist command;
  EXPECT_TRUE(IsFiniteControllerCommand(command));

  command.linear.x = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(IsFiniteControllerCommand(command));
  command.linear.x = 0.0;
  command.angular.z = std::numeric_limits<double>::infinity();
  EXPECT_FALSE(IsFiniteControllerCommand(command));
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
  for (const auto& point : task.boundary.points) {
    EXPECT_FLOAT_EQ(point.z, 0.0F);
  }
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

TEST(CoordinatorReadinessTest, ReverseTfEdgesDoNotSatisfyRequiredChain) {
  RequiredTfChain chain;
  tf2_msgs::msg::TFMessage reverse;
  reverse.transforms.resize(2U);
  reverse.transforms[0].header.frame_id = "odom";
  reverse.transforms[0].child_frame_id = "map";
  reverse.transforms[1].header.frame_id = "base_link";
  reverse.transforms[1].child_frame_id = "odom";

  chain.Observe(reverse);

  EXPECT_FALSE(chain.complete());
}

TEST_F(RosContextTest, RejectsNegativeSeedBeforeCreatingInterfaces) {
  rclcpp::NodeOptions options;
  options.parameter_overrides(
      {rclcpp::Parameter("seed", std::int64_t{-1})});
  EXPECT_THROW(
      { [[maybe_unused]] auto node =
            std::make_shared<RunCoordinator>(options); },
      std::invalid_argument);
}

TEST_F(RosContextTest, DeclaresExactControllerCommandTopicDefault) {
  const auto coordinator = std::make_shared<RunCoordinator>();
  EXPECT_EQ(coordinator->get_parameter("controller_command_topic").as_string(),
            "/Car/T5/Car_Cmd_Vel");
}

TEST_F(RosContextTest, WaitsForVolatileTaskSubscriberThenPublishesOnce) {
  const std::string prefix = "/coordinator_delivery_test";
  rclcpp::NodeOptions coordinator_options;
  coordinator_options.parameter_overrides({
      rclcpp::Parameter("seed", 41),
      rclcpp::Parameter("global_overview_topic", prefix + "/global"),
      rclcpp::Parameter("local_grid_map_topic", prefix + "/local"),
      rclcpp::Parameter("odometry_topic", prefix + "/odometry"),
      rclcpp::Parameter("tf_topic", prefix + "/tf"),
      rclcpp::Parameter("exploration_status_topic", prefix + "/status"),
      rclcpp::Parameter("exploration_task_topic", prefix + "/task"),
      rclcpp::Parameter("planner_action", prefix + "/plan_motion"),
      rclcpp::Parameter("controller_command_topic", prefix + "/command"),
  });
  auto coordinator = std::make_shared<RunCoordinator>(coordinator_options);
  auto harness = std::make_shared<rclcpp::Node>("coordinator_delivery_harness");
  const auto reliable = rclcpp::QoS{10}.reliable();
  const auto global_pub = harness->create_publisher<nav_msgs::msg::OccupancyGrid>(
      prefix + "/global", reliable);
  const auto local_pub = harness->create_publisher<grid_map_msgs::msg::GridMap>(
      prefix + "/local", reliable);
  const auto odometry_pub = harness->create_publisher<nav_msgs::msg::Odometry>(
      prefix + "/odometry", reliable);
  const auto tf_pub = harness->create_publisher<tf2_msgs::msg::TFMessage>(
      prefix + "/tf", reliable);
  const auto status_pub = harness->create_publisher<
      lunar_pure_exploration_msgs::msg::PureExplorationStatus>(
      prefix + "/status", rclcpp::QoS{1}.reliable().transient_local());
  const auto command_pub = harness->create_publisher<geometry_msgs::msg::Twist>(
      prefix + "/command", reliable);
  const auto make_action_server = [&] {
    return rclcpp_action::create_server<Action>(
        harness, prefix + "/plan_motion",
        [](const rclcpp_action::GoalUUID&,
           std::shared_ptr<const Action::Goal>) {
          return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
        },
        [](std::shared_ptr<rclcpp_action::ServerGoalHandle<Action>>) {
          return rclcpp_action::CancelResponse::ACCEPT;
        },
        [](std::shared_ptr<rclcpp_action::ServerGoalHandle<Action>>) {});
  };
  auto action_server = make_action_server();

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(coordinator);
  executor.add_node(harness);
  ASSERT_TRUE(SpinUntil(executor, [&] {
    return global_pub->get_subscription_count() == 1U &&
           local_pub->get_subscription_count() == 1U &&
           odometry_pub->get_subscription_count() == 1U &&
           tf_pub->get_subscription_count() == 1U &&
           status_pub->get_subscription_count() == 1U &&
           command_pub->get_subscription_count() == 1U;
  }));

  nav_msgs::msg::OccupancyGrid global;
  grid_map_msgs::msg::GridMap local;
  nav_msgs::msg::Odometry odometry;
  tf2_msgs::msg::TFMessage transforms;
  transforms.transforms.resize(2U);
  transforms.transforms[0].header.frame_id = "map";
  transforms.transforms[0].child_frame_id = "odom";
  transforms.transforms[1].header.frame_id = "odom";
  transforms.transforms[1].child_frame_id = "base_link";
  lunar_pure_exploration_msgs::msg::PureExplorationStatus status;
  status.state = decltype(status)::IDLE;
  global_pub->publish(global);
  local_pub->publish(local);
  odometry_pub->publish(odometry);
  tf_pub->publish(transforms);
  status_pub->publish(status);

  std::vector<Task> received_tasks;
  const auto task_sub = harness->create_subscription<Task>(
      prefix + "/task", rclcpp::QoS{1}.reliable(),
      [&received_tasks](Task::ConstSharedPtr message) {
        received_tasks.push_back(*message);
      });
  ASSERT_TRUE(SpinUntil(executor, [&] {
    return task_sub->get_publisher_count() == 1U;
  }));
  EXPECT_FALSE(SpinUntil(executor, [&] { return !received_tasks.empty(); }, 250ms));

  geometry_msgs::msg::Twist invalid_command;
  invalid_command.linear.x = std::numeric_limits<double>::quiet_NaN();
  command_pub->publish(invalid_command);
  EXPECT_FALSE(SpinUntil(executor, [&] { return !received_tasks.empty(); }, 250ms));

  auto duplicate_command_pub =
      harness->create_publisher<geometry_msgs::msg::Twist>(prefix + "/command",
                                                           reliable);
  ASSERT_TRUE(SpinUntil(executor, [&] {
    return harness->count_publishers(prefix + "/command") == 2U;
  }));
  geometry_msgs::msg::Twist finite_command;
  command_pub->publish(finite_command);
  EXPECT_FALSE(SpinUntil(executor, [&] { return !received_tasks.empty(); }, 250ms));
  duplicate_command_pub.reset();
  ASSERT_TRUE(SpinUntil(executor, [&] {
    return harness->count_publishers(prefix + "/command") == 1U;
  }));
  EXPECT_FALSE(SpinUntil(executor, [&] { return !received_tasks.empty(); }, 250ms));
  command_pub->publish(finite_command);

  ASSERT_TRUE(SpinUntil(executor, [&] {
    return harness->count_publishers(prefix + "/task") == 1U;
  }));
  std::this_thread::sleep_for(250ms);
  executor.spin_some();

  ASSERT_TRUE(SpinUntil(executor, [&] { return received_tasks.size() == 1U; }));
  ASSERT_EQ(received_tasks.size(), 1U);
  const auto& received = received_tasks.front();
  EXPECT_EQ(received.header.frame_id, "map");
  EXPECT_EQ(received.task_id, "jazzy-300m-41");
  EXPECT_EQ(received.command, Task::START);
  ASSERT_EQ(received.boundary.points.size(), 4U);
  EXPECT_EQ(received.boundary.points[0].x, -145.0F);
  EXPECT_EQ(received.boundary.points[0].y, -145.0F);
  EXPECT_EQ(received.boundary.points[1].x, 145.0F);
  EXPECT_EQ(received.boundary.points[1].y, -145.0F);
  EXPECT_EQ(received.boundary.points[2].x, 145.0F);
  EXPECT_EQ(received.boundary.points[2].y, 145.0F);
  EXPECT_EQ(received.boundary.points[3].x, -145.0F);
  EXPECT_EQ(received.boundary.points[3].y, 145.0F);
  for (const auto& point : received.boundary.points) {
    EXPECT_EQ(point.z, 0.0F);
  }

  action_server.reset();
  for (int iteration = 0; iteration < 3; ++iteration) {
    executor.spin_some();
    std::this_thread::sleep_for(110ms);
  }
  action_server = make_action_server();

  for (int iteration = 0; iteration < 5; ++iteration) {
    global_pub->publish(global);
    local_pub->publish(local);
    odometry_pub->publish(odometry);
    tf_pub->publish(transforms);
    status_pub->publish(status);
    executor.spin_some();
    std::this_thread::sleep_for(110ms);
  }
  executor.spin_some();
  EXPECT_EQ(received_tasks.size(), 1U);

  executor.remove_node(harness);
  executor.remove_node(coordinator);
}

}  // namespace
}  // namespace lunar::pure_exploration_sim

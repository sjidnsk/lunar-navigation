#include <atomic>
#include <chrono>
#include <cmath>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <gtest/gtest.h>
#include <lunar_planning_msgs/action/plan_motion.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>

#include "lunar_pure_planner_ros/rviz_goal_bridge.hpp"

namespace lunar::pure_planner_ros {
namespace {

using Action = lunar_planning_msgs::action::PlanMotion;
using GoalHandle = rclcpp_action::ServerGoalHandle<Action>;
using namespace std::chrono_literals;

class RosEnvironment final : public ::testing::Environment {
 public:
  void SetUp() override {
    if (!rclcpp::ok()) {
      int argc = 0;
      rclcpp::init(argc, nullptr);
    }
  }

  void TearDown() override {
    if (rclcpp::ok()) {
      rclcpp::shutdown();
    }
  }
};

const auto* const kRosEnvironment =
    ::testing::AddGlobalTestEnvironment(new RosEnvironment{});

template <typename Predicate>
bool WaitFor(Predicate&& predicate, const std::chrono::milliseconds timeout = 3s) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (predicate()) {
      return true;
    }
    std::this_thread::sleep_for(2ms);
  }
  return predicate();
}

class CapturingActionServer final {
 public:
  explicit CapturingActionServer(const bool hold_goal = false)
      : node_(std::make_shared<rclcpp::Node>("rviz_goal_bridge_action_test")),
        hold_goal_(hold_goal) {
    server_ = rclcpp_action::create_server<Action>(
        node_, "/test/plan_motion",
        [](const rclcpp_action::GoalUUID&,
           std::shared_ptr<const Action::Goal>) {
          return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
        },
        [this](const std::shared_ptr<GoalHandle>) {
          cancel_requests_.fetch_add(1U);
          return rclcpp_action::CancelResponse::ACCEPT;
        },
        [this](const std::shared_ptr<GoalHandle> handle) {
          {
            std::scoped_lock lock(mutex_);
            received_goal_ = *handle->get_goal();
          }
          if (hold_goal_) {
            held_goal_ = handle;
          } else {
            handle->succeed(std::make_shared<Action::Result>());
          }
        });
  }

  [[nodiscard]] rclcpp::Node::SharedPtr node() const { return node_; }

  [[nodiscard]] std::optional<Action::Goal> received_goal() const {
    std::scoped_lock lock(mutex_);
    return received_goal_;
  }

  [[nodiscard]] std::uint64_t cancel_requests() const noexcept {
    return cancel_requests_.load();
  }

 private:
  rclcpp::Node::SharedPtr node_;
  rclcpp_action::Server<Action>::SharedPtr server_;
  mutable std::mutex mutex_;
  std::optional<Action::Goal> received_goal_;
  bool hold_goal_{};
  std::atomic<std::uint64_t> cancel_requests_{};
  std::shared_ptr<GoalHandle> held_goal_;
};

rclcpp::NodeOptions BridgeOptions() {
  rclcpp::NodeOptions options;
  options.arguments({"--ros-args", "-r", "__node:=rviz_goal_bridge_test"});
  options.parameter_overrides({
      rclcpp::Parameter{"action_name", "/test/plan_motion"},
      rclcpp::Parameter{"environment_mode", 2},
      rclcpp::Parameter{"mission_id", "rviz-test"},
      rclcpp::Parameter{"mission_revision", 7},
      rclcpp::Parameter{"position_tolerance_m", 0.25},
      rclcpp::Parameter{"yaw_tolerance_rad", 0.15},
      rclcpp::Parameter{"start_topic", "/test/start_pose"},
  });
  return options;
}

TEST(RvizGoalBridge, ConvertsRvizPoseIntoPlanMotionGoal) {
  CapturingActionServer action_server;
  auto bridge = std::make_shared<RvizGoalBridge>(BridgeOptions());
  auto publisher_node = std::make_shared<rclcpp::Node>("rviz_goal_publisher_test");
  auto publisher = publisher_node->create_publisher<geometry_msgs::msg::PoseStamped>(
      "/Car/T4/rviz_goal", rclcpp::QoS{10}.reliable());

  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(action_server.node());
  executor.add_node(bridge);
  executor.add_node(publisher_node);
  std::jthread spinner([&executor] { executor.spin(); });

  ASSERT_TRUE(WaitFor([&publisher] {
    return publisher->get_subscription_count() == 1U;
  }));
  std::this_thread::sleep_for(100ms);

  geometry_msgs::msg::PoseStamped rviz_goal;
  rviz_goal.header.frame_id = "odom";
  rviz_goal.header.stamp.sec = 123;
  rviz_goal.header.stamp.nanosec = 456;
  rviz_goal.pose.position.x = 1.25;
  rviz_goal.pose.position.y = -2.5;
  rviz_goal.pose.position.z = 0.75;
  rviz_goal.pose.orientation.z = std::sin(0.5 * 0.8);
  rviz_goal.pose.orientation.w = std::cos(0.5 * 0.8);
  publisher->publish(rviz_goal);

  ASSERT_TRUE(WaitFor([&action_server] {
    return action_server.received_goal().has_value();
  }));
  const Action::Goal goal = *action_server.received_goal();
  EXPECT_EQ(goal.environment_mode, Action::Goal::LAVA_TUBE);
  EXPECT_EQ(goal.mission_id, "rviz-test");
  EXPECT_EQ(goal.mission_revision, 7U);
  EXPECT_TRUE(goal.replace_active_request);
  EXPECT_EQ(goal.goal.header.frame_id, "odom");
  EXPECT_EQ(goal.goal.header.stamp.sec, 123);
  EXPECT_EQ(goal.goal.header.stamp.nanosec, 456U);
  EXPECT_EQ(goal.goal.goal_type, goal.goal.POINT);
  EXPECT_DOUBLE_EQ(goal.goal.point.x, 1.25);
  EXPECT_DOUBLE_EQ(goal.goal.point.y, -2.5);
  EXPECT_DOUBLE_EQ(goal.goal.point.z, 0.75);
  EXPECT_DOUBLE_EQ(goal.goal.position_tolerance_m, 0.25);
  EXPECT_TRUE(goal.goal.has_yaw_constraint);
  EXPECT_NEAR(goal.goal.yaw_rad, 0.8, 1e-12);
  EXPECT_DOUBLE_EQ(goal.goal.yaw_tolerance_rad, 0.15);

  executor.cancel();
  spinner.join();
}

TEST(RvizGoalBridge, CancelsTheActiveRequestWhenRvizStartChanges) {
  CapturingActionServer action_server{true};
  auto bridge = std::make_shared<RvizGoalBridge>(BridgeOptions());
  auto publisher_node = std::make_shared<rclcpp::Node>("rviz_start_publisher_test");
  auto goal_publisher =
      publisher_node->create_publisher<geometry_msgs::msg::PoseStamped>(
          "/Car/T4/rviz_goal", rclcpp::QoS{10}.reliable());
  auto start_publisher =
      publisher_node->create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>(
          "/test/start_pose", rclcpp::QoS{10}.reliable());

  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(action_server.node());
  executor.add_node(bridge);
  executor.add_node(publisher_node);
  std::jthread spinner([&executor] { executor.spin(); });

  const bool subscriptions_ready = WaitFor([&] {
    return goal_publisher->get_subscription_count() == 1U &&
           start_publisher->get_subscription_count() == 1U;
  });
  if (!subscriptions_ready) {
    executor.cancel();
    spinner.join();
  }
  ASSERT_TRUE(subscriptions_ready);
  geometry_msgs::msg::PoseStamped goal;
  goal.header.frame_id = "map";
  goal.pose.orientation.w = 1.0;
  goal_publisher->publish(goal);
  const bool goal_received = WaitFor([&action_server] {
    return action_server.received_goal().has_value();
  });
  if (!goal_received) {
    executor.cancel();
    spinner.join();
  }
  ASSERT_TRUE(goal_received);

  geometry_msgs::msg::PoseWithCovarianceStamped start;
  start.header.frame_id = "map";
  start.pose.pose.orientation.w = 1.0;
  start_publisher->publish(start);

  EXPECT_TRUE(WaitFor([&action_server] {
    return action_server.cancel_requests() == 1U;
  }));
  executor.cancel();
  spinner.join();
}

}  // namespace
}  // namespace lunar::pure_planner_ros

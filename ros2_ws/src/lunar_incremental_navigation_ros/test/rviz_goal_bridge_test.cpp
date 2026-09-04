#include "lunar_incremental_navigation_ros/rviz_goal_bridge.hpp"

#include <cmath>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>

#include <gtest/gtest.h>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>

using namespace std::chrono_literals;

namespace lunar::incremental_navigation_ros {
namespace {

TEST(RvizGoalBridge, ConvertsMapPoseToNavigateToPoseGoalWithTerminalYaw) {
  geometry_msgs::msg::PoseStamped pose;
  pose.header.frame_id = "map";
  pose.pose.position.x = 2.3;
  pose.pose.position.y = 0.3;
  pose.pose.orientation.z = std::sin(0.75);
  pose.pose.orientation.w = std::cos(0.75);

  const auto goal = ConvertRvizGoal(pose, "map");

  ASSERT_TRUE(goal.has_value());
  EXPECT_DOUBLE_EQ(goal->target_x_m, 2.3);
  EXPECT_DOUBLE_EQ(goal->target_y_m, 0.3);
  EXPECT_TRUE(goal->has_target_yaw);
  EXPECT_NEAR(goal->target_yaw_rad, 1.5, 1e-12);
}

TEST(RvizGoalBridge, RejectsPoseOutsideConfiguredMapFrame) {
  geometry_msgs::msg::PoseStamped pose;
  pose.header.frame_id = "odom";
  pose.pose.position.x = 2.3;
  pose.pose.position.y = 0.3;
  pose.pose.orientation.w = 1.0;

  EXPECT_FALSE(ConvertRvizGoal(pose, "map").has_value());
}

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
bool WaitFor(Predicate&& predicate,
             const std::chrono::milliseconds timeout = 2s) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (predicate()) {
      return true;
    }
    std::this_thread::sleep_for(5ms);
  }
  return predicate();
}

class CapturingActionServer final {
 public:
  using Action = lunar_planning_msgs::action::NavigateToPose;
  using GoalHandle = rclcpp_action::ServerGoalHandle<Action>;

  explicit CapturingActionServer(const std::string& action_name)
      : node_(std::make_shared<rclcpp::Node>("rviz_goal_bridge_action_server")) {
    server_ = rclcpp_action::create_server<Action>(
        node_, action_name,
        [](const rclcpp_action::GoalUUID&,
           const std::shared_ptr<const Action::Goal>) {
          return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
        },
        [](const std::shared_ptr<GoalHandle>) {
          return rclcpp_action::CancelResponse::ACCEPT;
        },
        [this](const std::shared_ptr<GoalHandle> handle) {
          std::scoped_lock lock{mutex_};
          received_goal_ = *handle->get_goal();
          received_ = true;
          condition_.notify_all();
        });
  }

  [[nodiscard]] std::shared_ptr<rclcpp::Node> node() const { return node_; }

  [[nodiscard]] bool received() const {
    std::scoped_lock lock{mutex_};
    return received_;
  }

  [[nodiscard]] Action::Goal goal() const {
    std::scoped_lock lock{mutex_};
    return received_goal_;
  }

 private:
  std::shared_ptr<rclcpp::Node> node_;
  rclcpp_action::Server<Action>::SharedPtr server_;
  mutable std::mutex mutex_;
  std::condition_variable condition_;
  Action::Goal received_goal_;
  bool received_{};
};

TEST(RvizGoalBridge, ForwardsRvizMapGoalToNavigateToPoseAction) {
  const std::string goal_topic = "/test/rviz_goal";
  const std::string action_name = "/test/navigate_to_pose";
  CapturingActionServer server{action_name};

  rclcpp::NodeOptions options;
  options.parameter_overrides({
      rclcpp::Parameter{"goal_topic", goal_topic},
      rclcpp::Parameter{"action_name", action_name},
      rclcpp::Parameter{"expected_frame", "map"},
  });
  auto bridge = std::make_shared<RvizGoalBridge>(options);
  auto publisher = std::make_shared<rclcpp::Node>("rviz_goal_bridge_publisher");
  auto goals = publisher->create_publisher<geometry_msgs::msg::PoseStamped>(
      goal_topic, rclcpp::QoS{1}.reliable());

  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(server.node());
  executor.add_node(bridge);
  executor.add_node(publisher);
  std::thread spin_thread([&executor] { executor.spin(); });

  ASSERT_TRUE(WaitFor([&goals] { return goals->get_subscription_count() == 1U; }));
  geometry_msgs::msg::PoseStamped pose;
  pose.header.frame_id = "map";
  pose.pose.position.x = 2.3;
  pose.pose.position.y = 0.3;
  pose.pose.orientation.w = 1.0;
  goals->publish(pose);

  ASSERT_TRUE(WaitFor([&server] { return server.received(); }));
  const auto goal = server.goal();
  EXPECT_DOUBLE_EQ(goal.target_x_m, 2.3);
  EXPECT_DOUBLE_EQ(goal.target_y_m, 0.3);
  EXPECT_TRUE(goal.has_target_yaw);
  EXPECT_DOUBLE_EQ(goal.target_yaw_rad, 0.0);

  executor.cancel();
  spin_thread.join();
  executor.remove_node(publisher);
  executor.remove_node(bridge);
  executor.remove_node(server.node());
}

}  // namespace
}  // namespace lunar::incremental_navigation_ros

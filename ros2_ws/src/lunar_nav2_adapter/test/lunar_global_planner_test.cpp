#include "lunar_nav2_adapter/lunar_global_planner.hpp"

#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

#include <gtest/gtest.h>
#include <nav2_core/exceptions.hpp>
#include <pluginlib/class_loader.hpp>
#include <rclcpp/executors/multi_threaded_executor.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/create_server.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>
#include <tf2_ros/buffer.h>

namespace lunar::planning::nav2 {
namespace {

using Action = lunar_planning_msgs::action::PlanMotion;
using Reference = lunar_planning_msgs::msg::MotionReference;
using GoalHandle = rclcpp_action::ServerGoalHandle<Action>;
using namespace std::chrono_literals;

template<typename Predicate>
bool WaitFor(Predicate predicate, const std::chrono::milliseconds timeout = 2s) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (predicate()) {
      return true;
    }
    std::this_thread::sleep_for(5ms);
  }
  return predicate();
}

class ExecutorSpinGuard final {
 public:
  explicit ExecutorSpinGuard(rclcpp::Executor& executor)
      : executor_(executor), thread_([this] { executor_.spin(); }) {}

  ~ExecutorSpinGuard() {
    Stop();
  }

  ExecutorSpinGuard(const ExecutorSpinGuard&) = delete;
  ExecutorSpinGuard& operator=(const ExecutorSpinGuard&) = delete;

  void Stop() {
    executor_.cancel();
    if (thread_.joinable()) {
      thread_.join();
    }
  }

 private:
  rclcpp::Executor& executor_;
  std::jthread thread_;
};

Action::Result MakeResult(const std::uint8_t platform_type) {
  Action::Result result;
  result.has_reference = true;
  result.reference.platform_type = platform_type;
  result.reference.header.frame_id = "map";
  result.reference.path_preview.header.frame_id = "map";
  result.reference.path_preview.poses.resize(2U);
  for (auto& pose : result.reference.path_preview.poses) {
    pose.header = result.reference.path_preview.header;
  }
  return result;
}

TEST(LunarGlobalPlanner, RejectsMissingReference) {
  LunarGlobalPlanner adapter;
  Action::Result result;
  result.has_reference = false;
  EXPECT_THROW(adapter.ConvertResult(result), nav2_core::PlannerException);
}

TEST(LunarGlobalPlanner, RejectsLeggedReference) {
  LunarGlobalPlanner adapter;
  const auto result = MakeResult(Reference::LEGGED);
  EXPECT_THROW(adapter.ConvertResult(result), nav2_core::PlannerException);
}

TEST(LunarGlobalPlanner, RejectsHopperReference) {
  LunarGlobalPlanner adapter;
  const auto result = MakeResult(Reference::HOPPER);
  EXPECT_THROW(adapter.ConvertResult(result), nav2_core::PlannerException);
}

TEST(LunarGlobalPlanner, RejectsOdomEmptyAndInconsistentPreviews) {
  LunarGlobalPlanner adapter;

  auto result = MakeResult(Reference::WHEELED);
  result.reference.path_preview.header.frame_id = "odom";
  for (auto& pose : result.reference.path_preview.poses) {
    pose.header = result.reference.path_preview.header;
  }
  EXPECT_THROW(adapter.ConvertResult(result), nav2_core::PlannerException);

  result = MakeResult(Reference::WHEELED);
  result.reference.path_preview.poses.clear();
  EXPECT_THROW(adapter.ConvertResult(result), nav2_core::PlannerException);

  result = MakeResult(Reference::WHEELED);
  result.reference.path_preview.poses.back().header.frame_id = "odom";
  EXPECT_THROW(adapter.ConvertResult(result), nav2_core::PlannerException);
}

TEST(LunarGlobalPlanner, ReturnsFarMapPreviewWithoutReplanning) {
  LunarGlobalPlanner adapter;
  auto result = MakeResult(Reference::WHEELED);
  result.reference.path_preview.poses[1].pose.position.x = 900.0;

  const auto path = adapter.ConvertResult(result);

  ASSERT_EQ(path.poses.size(), 2U);
  EXPECT_EQ(path.header.frame_id, "map");
  EXPECT_DOUBLE_EQ(path.poses[1].pose.position.x, 900.0);
}

TEST(LunarGlobalPlanner, ExportsNav2GlobalPlannerPlugin) {
  pluginlib::ClassLoader<nav2_core::GlobalPlanner> loader{
      "nav2_core", "nav2_core::GlobalPlanner"};
  ASSERT_TRUE(loader.isClassAvailable(
      "lunar_nav2_adapter/LunarGlobalPlanner"));
  EXPECT_NE(
      loader.createSharedInstance("lunar_nav2_adapter/LunarGlobalPlanner"),
      nullptr);
}

class LunarGlobalPlannerRosTest : public ::testing::Test {
 protected:
  void SetUp() override {
    int argc = 0;
    char** argv = nullptr;
    rclcpp::init(argc, argv);
  }

  void TearDown() override {
    rclcpp::shutdown();
  }

  static rclcpp::NodeOptions ValidOptions() {
    rclcpp::NodeOptions options;
    options.parameter_overrides({
        rclcpp::Parameter{"Lunar.mission_id", "mission-nav2"},
        rclcpp::Parameter{"Lunar.mission_revision", std::int64_t{17}},
        rclcpp::Parameter{"Lunar.start_tolerance_m", 0.25},
        rclcpp::Parameter{"Lunar.goal_tolerance_m", 0.40},
        rclcpp::Parameter{"Lunar.yaw_tolerance_rad", 0.15},
        rclcpp::Parameter{"Lunar.action_server_timeout_s", 1.0},
        rclcpp::Parameter{"Lunar.action_result_timeout_s", 2.0},
    });
    return options;
  }

  static nav_msgs::msg::Odometry Odometry(
      const double x = 1.0,
      const double y = 2.0) {
    nav_msgs::msg::Odometry odometry;
    odometry.header.frame_id = "map";
    odometry.pose.pose.position.x = x;
    odometry.pose.pose.position.y = y;
    odometry.pose.pose.orientation.w = 1.0;
    return odometry;
  }

  static geometry_msgs::msg::PoseStamped Pose(
      const double x,
      const double y) {
    geometry_msgs::msg::PoseStamped pose;
    pose.header.frame_id = "map";
    pose.pose.position.x = x;
    pose.pose.position.y = y;
    pose.pose.orientation.w = 1.0;
    return pose;
  }
};

TEST_F(LunarGlobalPlannerRosTest, RejectsPlanBeforeActivation) {
  auto parent = std::make_shared<rclcpp_lifecycle::LifecycleNode>(
      "nav2_adapter_inactive_test", ValidOptions());
  auto tf = std::make_shared<tf2_ros::Buffer>(parent->get_clock());
  LunarGlobalPlanner adapter;
  adapter.configure(parent, "Lunar", tf, nullptr);
  adapter.SetLatestOdometryForTesting(Odometry());

  EXPECT_THROW(
      adapter.createPlan(Pose(1.0, 2.0), Pose(3.0, 4.0)),
      nav2_core::PlannerException);
  adapter.cleanup();
}

TEST_F(LunarGlobalPlannerRosTest, RejectsStartOutsideOdometryTolerance) {
  auto parent = std::make_shared<rclcpp_lifecycle::LifecycleNode>(
      "nav2_adapter_start_test", ValidOptions());
  auto tf = std::make_shared<tf2_ros::Buffer>(parent->get_clock());
  LunarGlobalPlanner adapter;
  adapter.configure(parent, "Lunar", tf, nullptr);
  adapter.activate();
  adapter.SetLatestOdometryForTesting(Odometry());

  EXPECT_THROW(
      adapter.createPlan(Pose(1.4, 2.0), Pose(3.0, 4.0)),
      nav2_core::PlannerException);
  adapter.deactivate();
  adapter.cleanup();
}

TEST_F(LunarGlobalPlannerRosTest, SendsPointGoalAndReturnsWheeledActionPath) {
  auto parent = std::make_shared<rclcpp_lifecycle::LifecycleNode>(
      "nav2_adapter_round_trip_test", ValidOptions());
  auto action_node = std::make_shared<rclcpp::Node>(
      "nav2_adapter_action_server_test");
  auto tf = std::make_shared<tf2_ros::Buffer>(parent->get_clock());
  std::mutex captured_mutex;
  std::optional<Action::Goal> captured_goal;

  auto action_server = rclcpp_action::create_server<Action>(
      action_node,
      "/plan_motion",
      [](const rclcpp_action::GoalUUID&,
         const std::shared_ptr<const Action::Goal>) {
        return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
      },
      [](const std::shared_ptr<GoalHandle>) {
        return rclcpp_action::CancelResponse::ACCEPT;
      },
      [&](const std::shared_ptr<GoalHandle> handle) {
        {
          std::scoped_lock lock{captured_mutex};
          captured_goal = *handle->get_goal();
        }
        auto result = std::make_shared<Action::Result>(
            MakeResult(Reference::WHEELED));
        result->reference.plan_id = "wheel-plan";
        result->reference.path_preview.poses[1].pose.position.x = 8.0;
        handle->succeed(result);
      });
  auto odometry_publisher =
      action_node->create_publisher<nav_msgs::msg::Odometry>(
          "/localization/odometry", rclcpp::SensorDataQoS{});

  LunarGlobalPlanner adapter;
  adapter.configure(parent, "Lunar", tf, nullptr);
  adapter.activate();

  rclcpp::executors::MultiThreadedExecutor executor{
      rclcpp::ExecutorOptions{}, 3U};
  executor.add_node(parent->get_node_base_interface());
  executor.add_node(action_node);
  ExecutorSpinGuard spin_guard{executor};
  ASSERT_TRUE(WaitFor([&] {
    return odometry_publisher->get_subscription_count() == 1U;
  }));
  odometry_publisher->publish(Odometry());
  std::this_thread::sleep_for(20ms);

  auto goal = Pose(6.0, 7.0);
  goal.pose.position.z = 0.5;
  const auto path = adapter.createPlan(Pose(1.1, 2.0), goal);

  ASSERT_EQ(path.poses.size(), 2U);
  EXPECT_DOUBLE_EQ(path.poses[1].pose.position.x, 8.0);
  {
    std::scoped_lock lock{captured_mutex};
    ASSERT_TRUE(captured_goal.has_value());
    EXPECT_EQ(captured_goal->mission_id, "mission-nav2");
    EXPECT_EQ(captured_goal->mission_revision, 17U);
    EXPECT_FALSE(captured_goal->replace_active_request);
    EXPECT_EQ(captured_goal->goal.goal_type, captured_goal->goal.POINT);
    EXPECT_DOUBLE_EQ(captured_goal->goal.point.x, 6.0);
    EXPECT_DOUBLE_EQ(captured_goal->goal.point.y, 7.0);
    EXPECT_DOUBLE_EQ(captured_goal->goal.point.z, 0.5);
    EXPECT_DOUBLE_EQ(captured_goal->goal.position_tolerance_m, 0.40);
    EXPECT_TRUE(captured_goal->goal.has_yaw_constraint);
    EXPECT_DOUBLE_EQ(captured_goal->goal.yaw_rad, 0.0);
    EXPECT_DOUBLE_EQ(captured_goal->goal.yaw_tolerance_rad, 0.15);
  }

  adapter.deactivate();
  adapter.cleanup();
  spin_guard.Stop();
  executor.remove_node(action_node);
  executor.remove_node(parent->get_node_base_interface());
  action_server.reset();
}

}  // namespace
}  // namespace lunar::planning::nav2

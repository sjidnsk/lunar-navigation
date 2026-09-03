#include "lunar_pure_exploration_ros/navigation_client.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

#include <lunar_planning_msgs/action/navigate_to_pose.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>

namespace lunar::pure_exploration_ros {
namespace {

using Action = lunar_planning_msgs::action::NavigateToPose;
using ServerGoalHandle = rclcpp_action::ServerGoalHandle<Action>;
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
bool WaitFor(Predicate&& predicate,
             const std::chrono::milliseconds timeout = 3s) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (predicate()) {
      return true;
    }
    std::this_thread::sleep_for(2ms);
  }
  return predicate();
}

class FakeNavigationServer final {
 public:
  FakeNavigationServer(rclcpp::Node& node, const std::string& action_name) {
    server_ = rclcpp_action::create_server<Action>(
        &node, action_name,
        [this](const rclcpp_action::GoalUUID&,
               std::shared_ptr<const Action::Goal>) {
          ++goal_count_;
          return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
        },
        [this](const std::shared_ptr<ServerGoalHandle>) {
          ++cancel_count_;
          return rclcpp_action::CancelResponse::ACCEPT;
        },
        [this](const std::shared_ptr<ServerGoalHandle> handle) {
          std::scoped_lock lock{mutex_};
          active_ = handle;
        });
  }

  std::size_t goal_count() const { return goal_count_.load(); }
  std::size_t cancel_count() const { return cancel_count_.load(); }

  void PublishFeedback(std::uint8_t state) {
    ASSERT_TRUE(WaitFor([this] {
      std::scoped_lock lock{mutex_};
      return active_ != nullptr;
    }));
    std::shared_ptr<ServerGoalHandle> active;
    {
      std::scoped_lock lock{mutex_};
      active = active_;
    }
    ASSERT_NE(active, nullptr);
    auto feedback = std::make_shared<Action::Feedback>();
    feedback->session_state = state;
    feedback->reason_code = "NAVIGATION_ACTIVE";
    active->publish_feedback(feedback);
  }

  void Finish(std::uint8_t outcome, std::string reason_code) {
    ASSERT_TRUE(WaitFor([this] {
      std::scoped_lock lock{mutex_};
      return active_ != nullptr;
    }));
    std::shared_ptr<ServerGoalHandle> active;
    {
      std::scoped_lock lock{mutex_};
      active = std::move(active_);
    }
    ASSERT_NE(active, nullptr);
    auto result = std::make_shared<Action::Result>();
    result->outcome = outcome;
    result->reason_code = std::move(reason_code);
    if (outcome == Action::Result::GOAL_REACHED) {
      active->succeed(result);
    } else if (outcome == Action::Result::CANCELED) {
      active->canceled(result);
    } else {
      active->abort(result);
    }
  }

 private:
  std::atomic<std::size_t> goal_count_{0U};
  std::atomic<std::size_t> cancel_count_{0U};
  std::mutex mutex_;
  std::shared_ptr<ServerGoalHandle> active_;
  rclcpp_action::Server<Action>::SharedPtr server_;
};

TEST(NavigationClient, OwnsOneGoalMapsFeedbackAndWaitsForTerminalResult) {
  auto server_node = std::make_shared<rclcpp::Node>("navigation_server");
  auto client_node = std::make_shared<rclcpp::Node>("navigation_client");
  FakeNavigationServer server(*server_node, "/test/navigation");
  std::mutex mutex;
  std::vector<NavigationFeedbackState> feedback;
  std::vector<NavigationTerminal> terminals;
  NavigationClient client(
      *client_node, "/test/navigation",
      [&](const NavigationFeedbackState state) {
        std::scoped_lock lock{mutex};
        feedback.push_back(state);
      },
      [&](NavigationTerminal terminal) {
        std::scoped_lock lock{mutex};
        terminals.push_back(std::move(terminal));
      });

  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(server_node);
  executor.add_node(client_node);
  std::jthread spin([&executor] { executor.spin(); });

  client.Navigate({.pose = {1.0, 2.0, 0.3}});
  EXPECT_THROW(client.Navigate({.pose = {3.0, 4.0, 0.0}}),
               std::logic_error);
  ASSERT_TRUE(WaitFor([&] { return server.goal_count() == 1U; }));
  server.PublishFeedback(Action::Feedback::PLANNING);
  server.PublishFeedback(Action::Feedback::EXECUTING);
  server.PublishFeedback(Action::Feedback::REPLANNING);
  ASSERT_TRUE(WaitFor([&] {
    std::scoped_lock lock{mutex};
    return feedback.size() == 3U;
  }));
  std::this_thread::sleep_for(100ms);
  {
    std::scoped_lock lock{mutex};
    EXPECT_TRUE(terminals.empty());
  }

  server.Finish(Action::Result::GOAL_REACHED, "GOAL_REACHED");
  ASSERT_TRUE(WaitFor([&] {
    std::scoped_lock lock{mutex};
    return terminals.size() == 1U;
  }));
  {
    std::scoped_lock lock{mutex};
    EXPECT_EQ(feedback,
              (std::vector<NavigationFeedbackState>{
                  NavigationFeedbackState::kPlanning,
                  NavigationFeedbackState::kExecuting,
                  NavigationFeedbackState::kReplanning}));
    EXPECT_EQ(terminals.front().outcome, Action::Result::GOAL_REACHED);
    EXPECT_EQ(terminals.front().reason_code, "GOAL_REACHED");
  }
  executor.cancel();
}

TEST(NavigationClient, CancelActiveCancelsTheOwnedActionGoal) {
  auto server_node = std::make_shared<rclcpp::Node>("cancel_server");
  auto client_node = std::make_shared<rclcpp::Node>("cancel_client");
  FakeNavigationServer server(*server_node, "/test/cancel_navigation");
  NavigationClient client(*client_node, "/test/cancel_navigation",
                          [](NavigationFeedbackState) {},
                          [](NavigationTerminal) {});
  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(server_node);
  executor.add_node(client_node);
  std::jthread spin([&executor] { executor.spin(); });

  client.Navigate({.pose = {1.0, 2.0, 0.0}});
  ASSERT_TRUE(WaitFor([&] { return server.goal_count() == 1U; }));
  client.CancelActive();
  EXPECT_TRUE(WaitFor([&] { return server.cancel_count() == 1U; }));
  server.Finish(Action::Result::CANCELED, "CANCELED");
  executor.cancel();
}

}  // namespace
}  // namespace lunar::pure_exploration_ros

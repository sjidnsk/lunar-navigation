#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>

#include <lunar_pure_exploration_core/types.hpp>
#include <rclcpp/node.hpp>

namespace lunar::pure_exploration_ros {

enum class NavigationFeedbackState : std::uint8_t {
  kPlanning,
  kExecuting,
  kReplanning,
  kWaitingForMap,
};

struct NavigationTarget final {
  lunar::pure_exploration::Pose2 pose;
  bool has_target_yaw{true};
};

struct NavigationTerminal final {
  std::uint8_t outcome;
  std::string reason_code;
};

class NavigationClient final {
 public:
  using FeedbackCallback = std::function<void(NavigationFeedbackState)>;
  using TerminalCallback = std::function<void(NavigationTerminal)>;

  NavigationClient(rclcpp::Node& node, std::string action_name,
                   FeedbackCallback feedback_callback,
                   TerminalCallback terminal_callback);
  ~NavigationClient() noexcept;

  NavigationClient(const NavigationClient&) = delete;
  NavigationClient& operator=(const NavigationClient&) = delete;
  NavigationClient(NavigationClient&&) = delete;
  NavigationClient& operator=(NavigationClient&&) = delete;

  void Navigate(NavigationTarget target);
  void CancelActive();

 private:
  struct CallbackState;
  std::shared_ptr<CallbackState> state_;
};

}  // namespace lunar::pure_exploration_ros

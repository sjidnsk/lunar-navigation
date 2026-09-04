#pragma once

#include <optional>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <lunar_planning_msgs/action/navigate_to_pose.hpp>
#include <rclcpp/node.hpp>
#include <rclcpp/node_options.hpp>
#include <rclcpp_action/client.hpp>

namespace lunar::incremental_navigation_ros {

[[nodiscard]] std::optional<lunar_planning_msgs::action::NavigateToPose::Goal>
ConvertRvizGoal(const geometry_msgs::msg::PoseStamped& pose,
                std::string_view expected_frame);

class RvizGoalBridge final : public rclcpp::Node {
 public:
  explicit RvizGoalBridge(
      const rclcpp::NodeOptions& options = rclcpp::NodeOptions{});

  RvizGoalBridge(const RvizGoalBridge&) = delete;
  RvizGoalBridge& operator=(const RvizGoalBridge&) = delete;

 private:
  using Action = lunar_planning_msgs::action::NavigateToPose;
  using ClientGoalHandle = rclcpp_action::ClientGoalHandle<Action>;

  void OnGoalPose(geometry_msgs::msg::PoseStamped::SharedPtr pose);
  void TrySendPendingGoal();
  void RequestCancelActiveGoal();

  std::string expected_frame_;
  std::shared_ptr<rclcpp_action::Client<Action>> action_client_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr
      goal_subscription_;
  rclcpp::TimerBase::SharedPtr retry_timer_;
  std::mutex mutex_;
  std::optional<Action::Goal> pending_goal_;
  ClientGoalHandle::SharedPtr active_goal_;
  bool request_in_flight_{};
  bool cancel_in_flight_{};
};

}  // namespace lunar::incremental_navigation_ros

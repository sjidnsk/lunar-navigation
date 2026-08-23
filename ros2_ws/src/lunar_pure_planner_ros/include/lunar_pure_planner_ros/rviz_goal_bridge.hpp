#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <lunar_planning_msgs/action/plan_motion.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>

namespace lunar::pure_planner_ros {

class RvizGoalBridge final : public rclcpp::Node {
 public:
  explicit RvizGoalBridge(
      const rclcpp::NodeOptions& options = rclcpp::NodeOptions{});

 private:
  void ForwardGoal(geometry_msgs::msg::PoseStamped::ConstSharedPtr goal_pose);

  using Action = lunar_planning_msgs::action::PlanMotion;

  std::uint8_t environment_mode_{};
  std::string mission_id_;
  std::uint64_t mission_revision_{};
  bool replace_active_request_{};
  double position_tolerance_m_{};
  double yaw_tolerance_rad_{};
  std::atomic<std::uint64_t> request_sequence_{1U};
  rclcpp_action::Client<Action>::SharedPtr action_client_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr subscription_;
};

}  // namespace lunar::pure_planner_ros

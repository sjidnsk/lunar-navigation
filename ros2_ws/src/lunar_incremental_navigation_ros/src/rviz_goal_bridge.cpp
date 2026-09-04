#include "lunar_incremental_navigation_ros/rviz_goal_bridge.hpp"

#include <chrono>
#include <cmath>
#include <utility>

#include <rclcpp/qos.hpp>
#include <rclcpp_action/create_client.hpp>

using namespace std::chrono_literals;

namespace lunar::incremental_navigation_ros {

std::optional<lunar_planning_msgs::action::NavigateToPose::Goal>
ConvertRvizGoal(const geometry_msgs::msg::PoseStamped& pose,
                const std::string_view expected_frame) {
  if (pose.header.frame_id != expected_frame ||
      !std::isfinite(pose.pose.position.x) ||
      !std::isfinite(pose.pose.position.y) ||
      !std::isfinite(pose.pose.orientation.x) ||
      !std::isfinite(pose.pose.orientation.y) ||
      !std::isfinite(pose.pose.orientation.z) ||
      !std::isfinite(pose.pose.orientation.w)) {
    return std::nullopt;
  }

  const auto& orientation = pose.pose.orientation;
  const double norm_squared = orientation.x * orientation.x +
                              orientation.y * orientation.y +
                              orientation.z * orientation.z +
                              orientation.w * orientation.w;
  if (!std::isfinite(norm_squared) || norm_squared <= 0.0) {
    return std::nullopt;
  }

  const double yaw = std::atan2(
      2.0 * (orientation.w * orientation.z + orientation.x * orientation.y),
      1.0 - 2.0 * (orientation.y * orientation.y + orientation.z * orientation.z));
  lunar_planning_msgs::action::NavigateToPose::Goal goal;
  goal.target_x_m = pose.pose.position.x;
  goal.target_y_m = pose.pose.position.y;
  goal.has_target_yaw = true;
  goal.target_yaw_rad = yaw;
  return goal;
}

RvizGoalBridge::RvizGoalBridge(const rclcpp::NodeOptions& options)
    : Node("incremental_rviz_goal_bridge", options),
      expected_frame_(declare_parameter<std::string>("expected_frame", "map")) {
  const std::string goal_topic =
      declare_parameter<std::string>("goal_topic", "/Car/T4/rviz_goal");
  const std::string action_name = declare_parameter<std::string>(
      "action_name", "/Car/T4/navigation/navigate_to_pose");
  if (expected_frame_.empty() || goal_topic.empty() || action_name.empty()) {
    throw std::runtime_error(
        "incremental RViz goal bridge requires non-empty frame and topics");
  }
  action_client_ = rclcpp_action::create_client<Action>(this, action_name);
  goal_subscription_ = create_subscription<geometry_msgs::msg::PoseStamped>(
      goal_topic, rclcpp::QoS{1}.reliable(),
      [this](geometry_msgs::msg::PoseStamped::SharedPtr pose) {
        OnGoalPose(std::move(pose));
      });
  retry_timer_ = create_wall_timer(100ms, [this] { TrySendPendingGoal(); });
}

void RvizGoalBridge::OnGoalPose(
    geometry_msgs::msg::PoseStamped::SharedPtr pose) {
  const auto goal = ConvertRvizGoal(*pose, expected_frame_);
  if (!goal) {
    RCLCPP_WARN(get_logger(), "ignored RViz goal outside '%s' or with invalid pose",
                expected_frame_.c_str());
    return;
  }
  {
    std::scoped_lock lock{mutex_};
    pending_goal_ = *goal;
  }
  RequestCancelActiveGoal();
  TrySendPendingGoal();
}

void RvizGoalBridge::TrySendPendingGoal() {
  Action::Goal goal;
  {
    std::scoped_lock lock{mutex_};
    if (!pending_goal_ || active_goal_ || request_in_flight_ ||
        cancel_in_flight_) {
      return;
    }
    if (!action_client_->action_server_is_ready()) {
      return;
    }
    goal = *pending_goal_;
    pending_goal_.reset();
    request_in_flight_ = true;
  }

  rclcpp_action::Client<Action>::SendGoalOptions send_options;
  send_options.goal_response_callback = [this](ClientGoalHandle::SharedPtr handle) {
    bool should_cancel{};
    {
      std::scoped_lock lock{mutex_};
      request_in_flight_ = false;
      if (handle) {
        active_goal_ = std::move(handle);
        should_cancel = pending_goal_.has_value();
      }
    }
    if (should_cancel) {
      RequestCancelActiveGoal();
    } else {
      TrySendPendingGoal();
    }
  };
  send_options.result_callback = [this](const ClientGoalHandle::WrappedResult&) {
    {
      std::scoped_lock lock{mutex_};
      active_goal_.reset();
      cancel_in_flight_ = false;
    }
    TrySendPendingGoal();
  };
  action_client_->async_send_goal(goal, send_options);
}

void RvizGoalBridge::RequestCancelActiveGoal() {
  ClientGoalHandle::SharedPtr active_goal;
  {
    std::scoped_lock lock{mutex_};
    if (!active_goal_ || cancel_in_flight_) {
      return;
    }
    active_goal = active_goal_;
    cancel_in_flight_ = true;
  }
  action_client_->async_cancel_goal(active_goal);
}

}  // namespace lunar::incremental_navigation_ros

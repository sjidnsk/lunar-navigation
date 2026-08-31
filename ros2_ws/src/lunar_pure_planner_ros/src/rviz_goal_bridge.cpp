#include "lunar_pure_planner_ros/rviz_goal_bridge.hpp"

#include <cmath>
#include <utility>

#include "lunar_pure_planner_ros/message_conversion.hpp"

namespace lunar::pure_planner_ros {
namespace {

[[nodiscard]] double YawFromQuaternion(
    const geometry_msgs::msg::Quaternion& orientation) noexcept {
  const double sine = 2.0 *
      (orientation.w * orientation.z + orientation.x * orientation.y);
  const double cosine = 1.0 - 2.0 *
      (orientation.y * orientation.y + orientation.z * orientation.z);
  return std::atan2(sine, cosine);
}

}  // namespace

RvizGoalBridge::RvizGoalBridge(const rclcpp::NodeOptions& options)
    : rclcpp::Node("rviz_goal_bridge", options),
      environment_mode_(static_cast<std::uint8_t>(
          declare_parameter<std::int64_t>("environment_mode", 2))),
      mission_id_(declare_parameter<std::string>("mission_id", "rviz")),
      mission_revision_(static_cast<std::uint64_t>(
          declare_parameter<std::int64_t>("mission_revision", 0))),
      replace_active_request_(
          declare_parameter<bool>("replace_active_request", true)),
      position_tolerance_m_(
          declare_parameter<double>("position_tolerance_m", 0.2)),
      yaw_tolerance_rad_(declare_parameter<double>("yaw_tolerance_rad", 0.1)) {
  const std::string action_name =
      declare_parameter<std::string>("action_name", "/Car/T4/plan_motion");
  const std::string goal_topic =
      declare_parameter<std::string>("goal_topic", "/Car/T4/rviz_goal");
  const std::string start_topic =
      declare_parameter<std::string>("start_topic", "/Car/T4/rviz_start");
  const std::string legged_global_path_topic =
      declare_parameter<std::string>("legged_global_path_topic", "");
  const std::string legged_local_path_topic =
      declare_parameter<std::string>("legged_local_path_topic", "");
  action_client_ = rclcpp_action::create_client<Action>(this, action_name);
  if (!legged_global_path_topic.empty()) {
    legged_global_path_publisher_ = create_publisher<nav_msgs::msg::Path>(
        legged_global_path_topic, rclcpp::QoS{10}.reliable());
  }
  if (!legged_local_path_topic.empty()) {
    legged_local_path_publisher_ = create_publisher<nav_msgs::msg::Path>(
        legged_local_path_topic, rclcpp::QoS{10}.reliable());
  }
  subscription_ = create_subscription<geometry_msgs::msg::PoseStamped>(
      goal_topic, rclcpp::QoS{10}.reliable(),
      [this](geometry_msgs::msg::PoseStamped::ConstSharedPtr goal_pose) {
        ForwardGoal(std::move(goal_pose));
      });
  start_subscription_ =
      create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
          start_topic, rclcpp::QoS{10}.reliable(),
          [this](geometry_msgs::msg::PoseWithCovarianceStamped::ConstSharedPtr) {
            action_client_->async_cancel_all_goals();
          });
}

void RvizGoalBridge::ForwardGoal(
    geometry_msgs::msg::PoseStamped::ConstSharedPtr goal_pose) {
  Action::Goal action_goal;
  const std::string request_id =
      "rviz-" + std::to_string(request_sequence_.fetch_add(1U));
  action_goal.environment_mode = environment_mode_;
  action_goal.request_id = request_id;
  action_goal.mission_id = mission_id_;
  action_goal.mission_revision = mission_revision_;
  action_goal.replace_active_request = replace_active_request_;
  action_goal.goal.header = goal_pose->header;
  action_goal.goal.goal_id = request_id;
  action_goal.goal.goal_type = action_goal.goal.POINT;
  action_goal.goal.point = goal_pose->pose.position;
  action_goal.goal.position_tolerance_m = position_tolerance_m_;
  action_goal.goal.has_yaw_constraint = true;
  action_goal.goal.yaw_rad = YawFromQuaternion(goal_pose->pose.orientation);
  action_goal.goal.yaw_tolerance_rad = yaw_tolerance_rad_;

  rclcpp_action::Client<Action>::SendGoalOptions send_options;
  send_options.goal_response_callback =
      [this, request_id](const rclcpp_action::ClientGoalHandle<Action>::SharedPtr& handle) {
        if (!handle) {
          RCLCPP_WARN(get_logger(), "RViz goal %s was rejected", request_id.c_str());
        }
      };
  send_options.result_callback =
      [this](const rclcpp_action::ClientGoalHandle<Action>::WrappedResult& result) {
        PublishLeggedResultPaths(result);
      };
  action_client_->async_send_goal(action_goal, send_options);
}

void RvizGoalBridge::PublishLeggedResultPaths(
    const rclcpp_action::ClientGoalHandle<Action>::WrappedResult& result) {
  nav_msgs::msg::Path global_path;
  nav_msgs::msg::Path local_path;
  global_path.header.frame_id = "map";
  local_path.header.frame_id = "map";
  if (result.code == rclcpp_action::ResultCode::SUCCEEDED && result.result &&
      result.result->planning_outcome ==
          result.result->NEW_REFERENCE_AVAILABLE &&
      result.result->reason_code == "PLAN_FOUND" &&
      result.result->has_reference &&
      result.result->reference.platform_type ==
          result.result->reference.LEGGED) {
    global_path = result.result->reference.path_preview;
    local_path = ConvertTrajectoryPath(result.result->reference);
  }
  if (legged_global_path_publisher_) {
    legged_global_path_publisher_->publish(global_path);
  }
  if (legged_local_path_publisher_) {
    legged_local_path_publisher_->publish(local_path);
  }
}

}  // namespace lunar::pure_planner_ros

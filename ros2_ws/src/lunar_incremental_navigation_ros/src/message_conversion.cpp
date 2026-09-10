#include "lunar_incremental_navigation_ros/message_conversion.hpp"

#include <utility>

#include <geometry_msgs/msg/pose_stamped.hpp>

namespace lunar::incremental_navigation_ros {
namespace {

[[nodiscard]] std_msgs::msg::Header MapHeader(std::string_view frame) {
  std_msgs::msg::Header header;
  header.frame_id = frame;
  header.stamp.sec = 0;
  header.stamp.nanosec = 0U;
  return header;
}

[[nodiscard]] geometry_msgs::msg::Pose RosPose(
    const lunar::incremental_navigation::Pose3& pose) {
  geometry_msgs::msg::Pose converted;
  converted.position.x = pose.position_m.x;
  converted.position.y = pose.position_m.y;
  converted.position.z = pose.position_m.z;
  converted.orientation.w = pose.orientation.w;
  converted.orientation.x = pose.orientation.x;
  converted.orientation.y = pose.orientation.y;
  converted.orientation.z = pose.orientation.z;
  return converted;
}

[[nodiscard]] nav_msgs::msg::Path ConvertPoses(
    const std::vector<lunar::incremental_navigation::Pose3>& poses, std::string_view map_frame) {
  nav_msgs::msg::Path path;
  path.header = MapHeader(map_frame);
  path.poses.reserve(poses.size());
  for (const auto& pose : poses) {
    geometry_msgs::msg::PoseStamped converted;
    converted.header = path.header;
    converted.pose = RosPose(pose);
    path.poses.push_back(std::move(converted));
  }
  return path;
}

}  // namespace

GoalConversionResult ConvertGoal(
    const lunar_planning_msgs::action::NavigateToPose::Goal& request) {
  lunar::incremental_navigation::FinalGoal goal{
      .target_x_m = request.target_x_m,
      .target_y_m = request.target_y_m,
      .has_target_yaw = request.has_target_yaw,
      .target_yaw_rad = request.target_yaw_rad,
  };
  if (!lunar::incremental_navigation::IsValidFinalGoal(goal)) {
    return {.reason_code = "INVALID_GOAL"};
  }
  return {.goal = std::move(goal)};
}

lunar_planning_msgs::action::NavigateToPose::Feedback ConvertFeedback(
    const lunar::incremental_navigation::NavigateToPoseFeedback& feedback) {
  lunar_planning_msgs::action::NavigateToPose::Feedback converted;
  converted.session_state = static_cast<std::uint8_t>(feedback.session_state);
  converted.planning_cycle = feedback.planning_cycle;
  converted.active_segment_revision = feedback.active_segment_revision;
  converted.reason_code = feedback.reason_code;
  return converted;
}

lunar_planning_msgs::action::NavigateToPose::Result ConvertResult(
    const lunar::incremental_navigation::NavigateToPoseResult& result) {
  lunar_planning_msgs::action::NavigateToPose::Result converted;
  converted.outcome = static_cast<std::uint8_t>(result.outcome);
  converted.reason_code = result.reason_code;
  converted.last_segment_revision = result.last_segment_revision;
  return converted;
}

lunar_planning_msgs::msg::PathReference ConvertPathReference(
    const lunar::incremental_navigation::PathReference& reference, std::string_view map_frame) {
  lunar_planning_msgs::msg::PathReference converted;
  converted.session_id.uuid = reference.session_id.bytes;
  converted.segment_revision = reference.segment_revision;
  converted.traversability_revision = reference.traversability_revision;
  converted.state = static_cast<std::uint8_t>(reference.state);
  converted.reaches_final_goal = reference.reaches_final_goal;
  converted.path = ConvertPoses(reference.path.poses, map_frame);
  return converted;
}

nav_msgs::msg::Path ConvertGlobalRoute(
    const std::optional<lunar::incremental_navigation::GlobalRoute>& route, std::string_view map_frame) {
  if (!route) {
    return ConvertPoses({}, map_frame);
  }
  return ConvertPoses(route->poses_map, map_frame);
}

}  // namespace lunar::incremental_navigation_ros

#pragma once

#include <optional>
#include <string>

#include <lunar_planning_msgs/action/navigate_to_pose.hpp>
#include <lunar_planning_msgs/msg/path_reference.hpp>
#include <nav_msgs/msg/path.hpp>

#include "lunar_incremental_navigation_core/planning_session_coordinator.hpp"

namespace lunar::incremental_navigation_ros {

struct GoalConversionResult final {
  std::optional<lunar::incremental_navigation::FinalGoal> goal;
  std::string reason_code;

  [[nodiscard]] bool ok() const noexcept {
    return goal.has_value() && reason_code.empty();
  }
};

[[nodiscard]] GoalConversionResult ConvertGoal(
    const lunar_planning_msgs::action::NavigateToPose::Goal& request);

[[nodiscard]] lunar_planning_msgs::action::NavigateToPose::Feedback ConvertFeedback(
    const lunar::incremental_navigation::NavigateToPoseFeedback& feedback);

[[nodiscard]] lunar_planning_msgs::action::NavigateToPose::Result ConvertResult(
    const lunar::incremental_navigation::NavigateToPoseResult& result);

[[nodiscard]] lunar_planning_msgs::msg::PathReference ConvertPathReference(
    const lunar::incremental_navigation::PathReference& reference);

[[nodiscard]] nav_msgs::msg::Path ConvertGlobalRoute(
    const std::optional<lunar::incremental_navigation::GlobalRoute>& route);

}  // namespace lunar::incremental_navigation_ros

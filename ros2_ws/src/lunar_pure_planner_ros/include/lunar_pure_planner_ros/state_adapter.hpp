#pragma once

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>

#include "lunar_pure_planner_core/types/planning_request.hpp"
#include "lunar_pure_planner_ros/input_store.hpp"

namespace lunar::pure_planner_ros {

[[nodiscard]] AdapterResult<lunar::pure_planning::RigidTransform>
AdaptDirectMapFromOdom(const geometry_msgs::msg::TransformStamped& transform);

[[nodiscard]] AdapterResult<lunar::pure_planning::PlatformState> AdaptOdometry(
    const nav_msgs::msg::Odometry& odometry,
    lunar::pure_planning::PlatformType platform);

[[nodiscard]] AdapterResult<lunar::pure_planning::Pose3>
ComposeMapFromOdomAndOdometry(
    const lunar::pure_planning::RigidTransform& map_from_odom,
    const lunar::pure_planning::PlatformState& odom_from_base);

}  // namespace lunar::pure_planner_ros

#pragma once

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>

#include "lunar_incremental_navigation_core/planning_session_coordinator.hpp"
#include "lunar_incremental_navigation_ros/input_store.hpp"

namespace lunar::incremental_navigation_ros {

[[nodiscard]] AdapterResult<lunar::incremental_navigation::RigidTransform>
AdaptDirectMapFromOdom(const geometry_msgs::msg::TransformStamped& transform);

[[nodiscard]] AdapterResult<lunar::incremental_navigation::StateInput>
AdaptStateInput(
    const lunar::incremental_navigation::RigidTransform& map_from_odom,
    const nav_msgs::msg::Odometry& odometry);

}  // namespace lunar::incremental_navigation_ros

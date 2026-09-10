#pragma once

#include <string_view>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>

#include "lunar_incremental_navigation_core/planning_session_coordinator.hpp"
#include "lunar_incremental_navigation_ros/input_store.hpp"

namespace lunar::incremental_navigation_ros {

[[nodiscard]] AdapterResult<lunar::incremental_navigation::RigidTransform>
AdaptDirectMapFromOdom(const geometry_msgs::msg::TransformStamped& transform,
    std::string_view map_frame = "map", std::string_view odom_frame = "odom");

[[nodiscard]] AdapterResult<lunar::incremental_navigation::StateInput>
AdaptStateInput(
    const lunar::incremental_navigation::RigidTransform& map_from_odom,
    const nav_msgs::msg::Odometry& odometry,
    std::string_view map_frame = "map", std::string_view odom_frame = "odom",
    std::string_view base_frame = "base_link");

}  // namespace lunar::incremental_navigation_ros

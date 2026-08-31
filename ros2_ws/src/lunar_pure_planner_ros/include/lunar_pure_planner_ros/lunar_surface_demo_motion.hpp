#pragma once

#include <cstddef>

#include "nav_msgs/msg/path.hpp"

namespace lunar::pure_planner_ros {

void AdvanceAlongDemoPath(const nav_msgs::msg::Path& path,
                          std::size_t& next_pose, double& x_m, double& y_m,
                          double step_m);

}  // namespace lunar::pure_planner_ros

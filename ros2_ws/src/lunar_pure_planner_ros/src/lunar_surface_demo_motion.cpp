#include "lunar_pure_planner_ros/lunar_surface_demo_motion.hpp"

#include <algorithm>
#include <cmath>

namespace lunar::pure_planner_ros {

void AdvanceAlongDemoPath(const nav_msgs::msg::Path& path,
                          std::size_t& next_pose, double& x_m, double& y_m,
                          const double step_m) {
  double remaining_m = std::max(0.0, step_m);
  while (remaining_m > 0.0 && next_pose < path.poses.size()) {
    const auto& target = path.poses[next_pose].pose.position;
    const double dx = target.x - x_m;
    const double dy = target.y - y_m;
    const double distance_m = std::hypot(dx, dy);
    if (distance_m <= remaining_m) {
      x_m = target.x;
      y_m = target.y;
      remaining_m -= distance_m;
      ++next_pose;
      continue;
    }
    x_m += remaining_m * dx / distance_m;
    y_m += remaining_m * dy / distance_m;
    remaining_m = 0.0;
  }
}

}  // namespace lunar::pure_planner_ros

#include "lunar_pure_planner_ros/lunar_surface_demo_state.hpp"

#include <cmath>
#include <utility>

namespace lunar::pure_planner_ros {

void LunarSurfaceDemoState::Reset(const double x_m, const double y_m,
                                  const double yaw_rad) noexcept {
  x_m_ = x_m;
  y_m_ = y_m;
  yaw_rad_ = yaw_rad;
  active_path_ = nav_msgs::msg::Path{};
  next_path_pose_ = 0U;
  last_local_x_m_.reset();
  last_local_y_m_.reset();
}

void LunarSurfaceDemoState::AcceptPath(const nav_msgs::msg::Path& path) {
  active_path_ = path;
  next_path_pose_ = active_path_.poses.size() > 1U ? 1U : 0U;
}

void LunarSurfaceDemoState::Advance(const double step_m) noexcept {
  if (!std::isfinite(step_m) || step_m <= 0.0) {
    return;
  }
  while (next_path_pose_ < active_path_.poses.size()) {
    const auto& target = active_path_.poses[next_path_pose_].pose.position;
    const double dx = target.x - x_m_;
    const double dy = target.y - y_m_;
    const double distance = std::hypot(dx, dy);
    if (distance <= step_m) {
      x_m_ = target.x;
      y_m_ = target.y;
      ++next_path_pose_;
      continue;
    }
    x_m_ += step_m * dx / distance;
    y_m_ += step_m * dy / distance;
    break;
  }
}

bool LunarSurfaceDemoState::LocalMapDue(
    const double update_distance_m) const noexcept {
  if (!last_local_x_m_.has_value() || !last_local_y_m_.has_value()) {
    return true;
  }
  return std::isfinite(update_distance_m) && update_distance_m > 0.0 &&
         std::hypot(x_m_ - *last_local_x_m_, y_m_ - *last_local_y_m_) >=
             update_distance_m;
}

void LunarSurfaceDemoState::MarkLocalMapPublished() noexcept {
  last_local_x_m_ = x_m_;
  last_local_y_m_ = y_m_;
}

bool LunarSurfaceDemoState::has_active_path() const noexcept {
  return next_path_pose_ < active_path_.poses.size();
}

}  // namespace lunar::pure_planner_ros

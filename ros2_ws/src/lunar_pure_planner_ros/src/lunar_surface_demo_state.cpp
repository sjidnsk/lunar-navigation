#include "lunar_pure_planner_ros/lunar_surface_demo_state.hpp"

#include <cmath>
#include <utility>

#include "lunar_pure_planner_ros/lunar_surface_demo_motion.hpp"

namespace lunar::pure_planner_ros {

bool LocalMapPublicationReady(const bool local_map_due,
                              const bool startup_delivery_active,
                              const std::size_t subscriber_count) noexcept {
  // The isolated launch has two required consumers: the planner and the
  // local traversability visualizer. Waiting for both prevents the visualizer
  // from consuming the only startup sample before the planner is discovered.
  return (local_map_due || startup_delivery_active) &&
         subscriber_count >= 2U;
}

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
  const double previous_x_m = x_m_;
  const double previous_y_m = y_m_;
  AdvanceAlongDemoPath(active_path_, next_path_pose_, x_m_, y_m_, step_m);
  const double dx = x_m_ - previous_x_m;
  const double dy = y_m_ - previous_y_m;
  if (std::hypot(dx, dy) > 1.0e-9) {
    yaw_rad_ = std::atan2(dy, dx);
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

#pragma once

#include <cstddef>
#include <optional>

#include <nav_msgs/msg/path.hpp>

namespace lunar::pure_planner_ros {

[[nodiscard]] bool LocalMapPublicationReady(
    bool local_map_due, bool startup_delivery_active,
    std::size_t subscriber_count) noexcept;

class LunarSurfaceDemoState final {
 public:
  void Reset(double x_m, double y_m, double yaw_rad = 0.0) noexcept;
  void AcceptPath(const nav_msgs::msg::Path& path);
  void Advance(double step_m) noexcept;

  [[nodiscard]] bool LocalMapDue(double update_distance_m) const noexcept;
  void MarkLocalMapPublished() noexcept;

  [[nodiscard]] double x_m() const noexcept { return x_m_; }
  [[nodiscard]] double y_m() const noexcept { return y_m_; }
  [[nodiscard]] double yaw_rad() const noexcept { return yaw_rad_; }
  [[nodiscard]] bool has_active_path() const noexcept;

 private:
  double x_m_{};
  double y_m_{};
  double yaw_rad_{};
  nav_msgs::msg::Path active_path_;
  std::size_t next_path_pose_{};
  std::optional<double> last_local_x_m_;
  std::optional<double> last_local_y_m_;
};

}  // namespace lunar::pure_planner_ros

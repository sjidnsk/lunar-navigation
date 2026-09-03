#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

#include <lunar_planning_msgs/msg/path_reference.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>

namespace lunar::incremental_navigation_ros {
namespace {

using namespace std::chrono_literals;

class IncrementalDemoMotionEmulator final : public rclcpp::Node {
 public:
  IncrementalDemoMotionEmulator()
      : Node("incremental_demo_motion_emulator"),
        platform_type_(declare_parameter<std::string>("platform_type", "wheel")),
        speed_mps_(declare_parameter<double>("speed_mps", 1.2)) {
    if ((platform_type_ != "wheel" && platform_type_ != "legged") ||
        !std::isfinite(speed_mps_) || speed_mps_ <= 0.0) {
      throw std::invalid_argument{"invalid motion-emulator parameters"};
    }
    odometry_publisher_ = create_publisher<nav_msgs::msg::Odometry>(
        "/planning_demo/odometry", rclcpp::QoS{10}.reliable());
    path_subscription_ =
        create_subscription<lunar_planning_msgs::msg::PathReference>(
            "/planning_demo/planning/path_reference",
            rclcpp::QoS{rclcpp::KeepLast{1}}.reliable().transient_local(),
            [this](const lunar_planning_msgs::msg::PathReference::SharedPtr path) {
              HandlePath(*path);
            });
    timer_ = create_wall_timer(50ms, [this] { Tick(); });
  }

 private:
  void HandlePath(const lunar_planning_msgs::msg::PathReference& message) {
    std::scoped_lock lock{mutex_};
    if (message.state == lunar_planning_msgs::msg::PathReference::ACTIVE &&
        !message.path.poses.empty()) {
      active_path_ = message.path;
      active_path_.header.frame_id = "map";
      next_pose_ = 0U;
      return;
    }
    active_path_.poses.clear();
    next_pose_ = 0U;
  }

  void Advance(const double distance_m) {
    double remaining = distance_m;
    while (remaining > 0.0 && next_pose_ < active_path_.poses.size()) {
      const auto& target = active_path_.poses[next_pose_].pose;
      const double dx = target.position.x - x_m_;
      const double dy = target.position.y - y_m_;
      const double distance = std::hypot(dx, dy);
      if (distance <= 1.0e-6) {
        yaw_rad_ = std::atan2(2.0 * target.orientation.w * target.orientation.z,
                             1.0 - 2.0 * target.orientation.z *
                                       target.orientation.z);
        ++next_pose_;
        continue;
      }
      const double step = std::min(distance, remaining);
      x_m_ += step * dx / distance;
      y_m_ += step * dy / distance;
      yaw_rad_ = std::atan2(dy, dx);
      remaining -= step;
      if (step >= distance - 1.0e-9) {
        ++next_pose_;
      }
    }
  }

  void Tick() {
    const auto stamp = now();
    nav_msgs::msg::Odometry odometry;
    {
      std::scoped_lock lock{mutex_};
      Advance(speed_mps_ * 0.05);
      odometry.header.stamp = stamp;
      odometry.header.frame_id = "odom";
      // The shared incremental explorer resolves odometry strictly as
      // odom -> base_link; the navigator accepts this frame for both profiles.
      odometry.child_frame_id = "base_link";
      odometry.pose.pose.position.x = x_m_;
      odometry.pose.pose.position.y = y_m_;
      odometry.pose.pose.position.z = platform_type_ == "legged" ? 0.33 : 0.0;
      odometry.pose.pose.orientation.z = std::sin(0.5 * yaw_rad_);
      odometry.pose.pose.orientation.w = std::cos(0.5 * yaw_rad_);
    }
    odometry_publisher_->publish(odometry);
  }

  std::string platform_type_;
  double speed_mps_{};
  std::mutex mutex_;
  nav_msgs::msg::Path active_path_;
  std::size_t next_pose_{};
  double x_m_{};
  double y_m_{};
  double yaw_rad_{};
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odometry_publisher_;
  rclcpp::Subscription<lunar_planning_msgs::msg::PathReference>::SharedPtr
      path_subscription_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace
}  // namespace lunar::incremental_navigation_ros

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<lunar::incremental_navigation_ros::
                                     IncrementalDemoMotionEmulator>());
  } catch (const std::exception& error) {
    RCLCPP_FATAL(rclcpp::get_logger("incremental_demo_motion_emulator"), "%s",
                 error.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}

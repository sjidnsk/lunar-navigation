#include <chrono>
#include <cmath>
#include <memory>
#include <mutex>
#include <numbers>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <geometry_msgs/msg/quaternion.hpp>
#include <lunar_planning_msgs/msg/path_reference.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>

#include "lunar_incremental_navigation_ros/demo_motion_follower.hpp"

namespace lunar::incremental_navigation_ros {
namespace {

using namespace std::chrono_literals;

[[nodiscard]] double YawFromQuaternion(
    const geometry_msgs::msg::Quaternion& orientation) {
  return std::atan2(2.0 * orientation.w * orientation.z,
                    1.0 - 2.0 * orientation.z * orientation.z);
}

[[nodiscard]] std::vector<DemoMotionPose> InitialScanPath() {
  constexpr double kStepRad = std::numbers::pi / 3.0;
  return {
      {.x_m = 0.0, .y_m = 0.0, .yaw_rad = 0.0},
      {.x_m = 0.0, .y_m = 0.0, .yaw_rad = kStepRad},
      {.x_m = 0.0, .y_m = 0.0, .yaw_rad = 2.0 * kStepRad},
      {.x_m = 0.0, .y_m = 0.0, .yaw_rad = std::numbers::pi},
      {.x_m = 0.0, .y_m = 0.0, .yaw_rad = -2.0 * kStepRad},
      {.x_m = 0.0, .y_m = 0.0, .yaw_rad = -kStepRad},
      {.x_m = 0.0, .y_m = 0.0, .yaw_rad = 0.0},
  };
}

class IncrementalDemoMotionEmulator final : public rclcpp::Node {
 public:
  IncrementalDemoMotionEmulator()
      : Node("incremental_demo_motion_emulator"),
        platform_type_(declare_parameter<std::string>("platform_type", "wheel")),
        speed_mps_(declare_parameter<double>("speed_mps", 1.2)),
        angular_speed_radps_(
            declare_parameter<double>("angular_speed_radps", 1.0)),
        follower_(speed_mps_, angular_speed_radps_) {
    if ((platform_type_ != "wheel" && platform_type_ != "legged") ||
        !std::isfinite(speed_mps_) || speed_mps_ <= 0.0 ||
        !std::isfinite(angular_speed_radps_) || angular_speed_radps_ <= 0.0) {
      throw std::invalid_argument{"invalid motion-emulator parameters"};
    }
    follower_.SetPath(InitialScanPath());
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
      std::vector<DemoMotionPose> path;
      path.reserve(message.path.poses.size());
      for (const auto& stamped_pose : message.path.poses) {
        path.push_back({.x_m = stamped_pose.pose.position.x,
                        .y_m = stamped_pose.pose.position.y,
                        .yaw_rad = YawFromQuaternion(
                            stamped_pose.pose.orientation)});
      }
      follower_.SetPath(std::move(path));
      return;
    }
    follower_.SetPath({});
  }

  void Tick() {
    const auto stamp = now();
    nav_msgs::msg::Odometry odometry;
    {
      std::scoped_lock lock{mutex_};
      follower_.Advance(0.05);
      const DemoMotionPose pose = follower_.pose();
      odometry.header.stamp = stamp;
      odometry.header.frame_id = "odom";
      // The shared incremental explorer resolves odometry strictly as
      // odom -> base_link; the navigator accepts this frame for both profiles.
      odometry.child_frame_id = "base_link";
      odometry.pose.pose.position.x = pose.x_m;
      odometry.pose.pose.position.y = pose.y_m;
      odometry.pose.pose.position.z = platform_type_ == "legged" ? 0.33 : 0.0;
      odometry.pose.pose.orientation.z = std::sin(0.5 * pose.yaw_rad);
      odometry.pose.pose.orientation.w = std::cos(0.5 * pose.yaw_rad);
    }
    odometry_publisher_->publish(odometry);
  }

  std::string platform_type_;
  double speed_mps_{};
  double angular_speed_radps_{};
  DemoMotionFollower follower_;
  std::mutex mutex_;
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

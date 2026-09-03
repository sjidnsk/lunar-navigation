#include "lunar_incremental_navigation_ros/demo_motion_follower.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <utility>

namespace lunar::incremental_navigation_ros {
namespace {

constexpr double kPositionToleranceM = 1.0e-6;
constexpr double kYawToleranceRad = 1.0e-6;

[[nodiscard]] double NormalizeAngle(const double angle_rad) {
  return std::remainder(angle_rad, 2.0 * std::numbers::pi);
}

}  // namespace

DemoMotionFollower::DemoMotionFollower(const double linear_speed_mps,
                                       const double angular_speed_radps)
    : linear_speed_mps_(linear_speed_mps),
      angular_speed_radps_(angular_speed_radps) {}

void DemoMotionFollower::SetPath(std::vector<DemoMotionPose> path) {
  path_ = std::move(path);
  next_pose_ = 0U;
}

void DemoMotionFollower::Advance(const double elapsed_s) {
  double remaining_distance_m = linear_speed_mps_ * elapsed_s;
  double remaining_turn_rad = angular_speed_radps_ * elapsed_s;
  while (next_pose_ < path_.size()) {
    const DemoMotionPose& target = path_[next_pose_];
    const double dx = target.x_m - pose_.x_m;
    const double dy = target.y_m - pose_.y_m;
    const double distance_m = std::hypot(dx, dy);
    if (distance_m > kPositionToleranceM) {
      if (remaining_distance_m <= 0.0) {
        return;
      }
      const double step_m = std::min(distance_m, remaining_distance_m);
      pose_.x_m += step_m * dx / distance_m;
      pose_.y_m += step_m * dy / distance_m;
      pose_.yaw_rad = std::atan2(dy, dx);
      remaining_distance_m -= step_m;
      if (step_m >= distance_m - kPositionToleranceM) {
        pose_.x_m = target.x_m;
        pose_.y_m = target.y_m;
        ++next_pose_;
      }
      continue;
    }

    pose_.x_m = target.x_m;
    pose_.y_m = target.y_m;
    const double yaw_error_rad =
        NormalizeAngle(target.yaw_rad - pose_.yaw_rad);
    if (std::abs(yaw_error_rad) <= kYawToleranceRad) {
      pose_.yaw_rad = target.yaw_rad;
      ++next_pose_;
      continue;
    }
    if (remaining_turn_rad <= 0.0) {
      return;
    }
    const double turn_rad = std::clamp(yaw_error_rad, -remaining_turn_rad,
                                       remaining_turn_rad);
    pose_.yaw_rad = NormalizeAngle(pose_.yaw_rad + turn_rad);
    remaining_turn_rad -= std::abs(turn_rad);
    if (std::abs(NormalizeAngle(target.yaw_rad - pose_.yaw_rad)) <=
        kYawToleranceRad) {
      pose_.yaw_rad = target.yaw_rad;
      ++next_pose_;
      continue;
    }
    return;
  }
}

DemoMotionPose DemoMotionFollower::pose() const noexcept { return pose_; }

}  // namespace lunar::incremental_navigation_ros

#include "lunar_pure_exploration_ros/pose_resolver.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "geometry_msgs/msg/quaternion.hpp"

namespace lunar::pure_exploration_ros {
namespace {

struct PlanarValue {
  double x;
  double y;
  double yaw;
};

double QuaternionYaw(const geometry_msgs::msg::Quaternion& quaternion) {
  const double x = quaternion.x;
  const double y = quaternion.y;
  const double z = quaternion.z;
  const double w = quaternion.w;
  if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z) ||
      !std::isfinite(w)) {
    throw std::invalid_argument{"non-finite quaternion"};
  }

  const double scale =
      std::max({std::abs(x), std::abs(y), std::abs(z), std::abs(w)});
  if (scale == 0.0) {
    throw std::invalid_argument{"zero-norm quaternion"};
  }
  const double scaled_x = x / scale;
  const double scaled_y = y / scale;
  const double scaled_z = z / scale;
  const double scaled_w = w / scale;
  const double scaled_norm =
      std::hypot(std::hypot(scaled_x, scaled_y),
                 std::hypot(scaled_z, scaled_w));
  const double normalized_x = scaled_x / scaled_norm;
  const double normalized_y = scaled_y / scaled_norm;
  const double normalized_z = scaled_z / scaled_norm;
  const double normalized_w = scaled_w / scaled_norm;

  return std::atan2(
      2.0 * (normalized_w * normalized_z +
             normalized_x * normalized_y),
      1.0 - 2.0 * (normalized_y * normalized_y +
                   normalized_z * normalized_z));
}

void RequireFinitePosition(const double x, const double y, const double z) {
  if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
    throw std::invalid_argument{"non-finite transform position"};
  }
}

double CheckedFiniteDouble(const long double value) {
  constexpr long double kMaximumDouble =
      static_cast<long double>(std::numeric_limits<double>::max());
  if (!std::isfinite(value) || value > kMaximumDouble ||
      value < -kMaximumDouble) {
    throw std::invalid_argument{"composed map pose is not finite double"};
  }
  const double converted = static_cast<double>(value);
  if (!std::isfinite(converted)) {
    throw std::invalid_argument{"composed map pose is not finite double"};
  }
  return converted;
}

}  // namespace

void PoseResolver::UpdateOdometry(const nav_msgs::msg::Odometry& odometry) {
  if (odometry.header.frame_id != "odom" ||
      odometry.child_frame_id != "base_link") {
    throw std::invalid_argument{"odometry must be exactly odom to base_link"};
  }

  const auto& position = odometry.pose.pose.position;
  RequireFinitePosition(position.x, position.y, position.z);
  const PlanarValue candidate{position.x, position.y,
                              QuaternionYaw(odometry.pose.pose.orientation)};
  std::optional<lunar::pure_exploration::Pose2> composed;
  if (map_from_odom_.has_value()) {
    composed = CheckedCompose(
        *map_from_odom_,
        PlanarTransform{candidate.x, candidate.y, candidate.yaw});
  }
  odom_from_base_ = PlanarTransform{candidate.x, candidate.y, candidate.yaw};
  latest_pose_in_map_ = composed;
}

void PoseResolver::UpdateTransforms(
    const tf2_msgs::msg::TFMessage& transforms) {
  const geometry_msgs::msg::TransformStamped* selected = nullptr;
  for (const auto& transform : transforms.transforms) {
    if (transform.header.frame_id == "map" &&
        transform.child_frame_id == "odom") {
      selected = &transform;
    }
  }
  if (selected == nullptr) {
    return;
  }

  const auto& translation = selected->transform.translation;
  RequireFinitePosition(translation.x, translation.y, translation.z);
  const PlanarValue candidate{
      translation.x, translation.y,
      QuaternionYaw(selected->transform.rotation)};
  std::optional<lunar::pure_exploration::Pose2> composed;
  if (odom_from_base_.has_value()) {
    composed = CheckedCompose(
        PlanarTransform{candidate.x, candidate.y, candidate.yaw},
        *odom_from_base_);
  }
  map_from_odom_ = PlanarTransform{candidate.x, candidate.y, candidate.yaw};
  latest_pose_in_map_ = composed;
}

lunar::pure_exploration::Pose2 PoseResolver::CheckedCompose(
    const PlanarTransform& map_from_odom,
    const PlanarTransform& odom_from_base) {
  const long double map_yaw = static_cast<long double>(map_from_odom.yaw);
  const long double cosine = std::cos(map_yaw);
  const long double sine = std::sin(map_yaw);
  const long double odom_x = static_cast<long double>(odom_from_base.x);
  const long double odom_y = static_cast<long double>(odom_from_base.y);
  const long double composed_x =
      static_cast<long double>(map_from_odom.x) + cosine * odom_x -
      sine * odom_y;
  const long double composed_y =
      static_cast<long double>(map_from_odom.y) + sine * odom_x +
      cosine * odom_y;
  const long double raw_yaw =
      map_yaw + static_cast<long double>(odom_from_base.yaw);
  const long double normalized_yaw =
      std::atan2(std::sin(raw_yaw), std::cos(raw_yaw));

  return lunar::pure_exploration::Pose2{
      .x = CheckedFiniteDouble(composed_x),
      .y = CheckedFiniteDouble(composed_y),
      .yaw = CheckedFiniteDouble(normalized_yaw),
  };
}

std::optional<lunar::pure_exploration::Pose2>
PoseResolver::LatestPoseInMap() const {
  return latest_pose_in_map_;
}

}  // namespace lunar::pure_exploration_ros

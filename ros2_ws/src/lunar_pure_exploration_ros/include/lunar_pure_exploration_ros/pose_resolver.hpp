#pragma once

#include <optional>

#include "lunar_pure_exploration_core/types.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "tf2_msgs/msg/tf_message.hpp"

namespace lunar::pure_exploration_ros {

class PoseResolver {
 public:
  void UpdateOdometry(const nav_msgs::msg::Odometry& odometry);
  void UpdateTransforms(const tf2_msgs::msg::TFMessage& transforms);
  [[nodiscard]] std::optional<lunar::pure_exploration::Pose2>
  LatestPoseInMap() const;

 private:
  struct PlanarTransform {
    double x;
    double y;
    double yaw;
  };

  [[nodiscard]] static lunar::pure_exploration::Pose2 CheckedCompose(
      const PlanarTransform& map_from_odom,
      const PlanarTransform& odom_from_base);

  std::optional<PlanarTransform> odom_from_base_;
  std::optional<PlanarTransform> map_from_odom_;
  std::optional<lunar::pure_exploration::Pose2> latest_pose_in_map_;
};

}  // namespace lunar::pure_exploration_ros

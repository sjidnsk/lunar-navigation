#include "lunar_incremental_navigation_ros/input_store.hpp"

#include <utility>

namespace lunar::incremental_navigation_ros {

void InputStore::UpdateLocal(grid_map_msgs::msg::GridMap::ConstSharedPtr message) {
  std::scoped_lock lock{mutex_};
  latest_.local_map = std::move(message);
  if (latest_.local_map) {
    ++latest_.local_sequence;
  }
}

void InputStore::UpdateOdometry(nav_msgs::msg::Odometry::ConstSharedPtr message) {
  std::scoped_lock lock{mutex_};
  latest_.odometry = std::move(message);
  if (latest_.odometry) {
    ++latest_.odometry_sequence;
  }
}

void InputStore::UpdateTf(const tf2_msgs::msg::TFMessage& message) {
  std::scoped_lock lock{mutex_};
  for (const auto& transform : message.transforms) {
    if (transform.header.frame_id == map_frame_ && transform.child_frame_id == odom_frame_) {
      latest_.map_from_odom = transform;
      ++latest_.tf_sequence;
    }
  }
}

InputSnapshot InputStore::Capture() const {
  std::scoped_lock lock{mutex_};
  return latest_;
}

}  // namespace lunar::incremental_navigation_ros

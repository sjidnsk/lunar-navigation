#include "lunar_pure_planner_ros/input_store.hpp"

#include <utility>

namespace lunar::pure_planner_ros {

void InputStore::UpdateGlobal(nav_msgs::msg::OccupancyGrid::ConstSharedPtr message) {
  std::scoped_lock lock{mutex_};
  latest_.global_map = std::move(message);
  if (latest_.global_map) {
    ++latest_.global_sequence;
  }
}

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
    if (transform.header.frame_id == "map" && transform.child_frame_id == "odom") {
      latest_.map_from_odom = transform;
      ++latest_.tf_sequence;
    }
  }
}

InputSnapshot InputStore::Capture() const {
  std::scoped_lock lock{mutex_};
  return latest_;
}

}  // namespace lunar::pure_planner_ros

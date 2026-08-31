#include "lunar_pure_planner_ros/input_store.hpp"

#include <utility>

namespace lunar::pure_planner_ros {
namespace {

[[nodiscard]] bool SameStamp(const builtin_interfaces::msg::Time& left,
                             const builtin_interfaces::msg::Time& right) {
  return left.sec == right.sec && left.nanosec == right.nanosec;
}

}  // namespace

InputStore::InputStore(const bool token_idempotent)
    : token_idempotent_(token_idempotent) {}

void InputStore::UpdateGlobal(nav_msgs::msg::OccupancyGrid::ConstSharedPtr message) {
  std::scoped_lock lock{mutex_};
  const bool duplicate_token =
      token_idempotent_ && message && latest_.global_map &&
      SameStamp(message->header.stamp, latest_.global_map->header.stamp);
  latest_.global_map = std::move(message);
  if (latest_.global_map && !duplicate_token) {
    ++latest_.global_sequence;
  }
}

void InputStore::UpdateLocal(grid_map_msgs::msg::GridMap::ConstSharedPtr message) {
  std::scoped_lock lock{mutex_};
  const bool duplicate_token =
      token_idempotent_ && message && latest_.local_map &&
      SameStamp(message->header.stamp, latest_.local_map->header.stamp);
  latest_.local_map = std::move(message);
  if (latest_.local_map) {
    ++latest_.local_arrival_sequence;
    if (!duplicate_token) {
      ++latest_.local_sequence;
    }
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
      const bool duplicate_token =
          token_idempotent_ && latest_.map_from_odom.has_value() &&
          SameStamp(transform.header.stamp,
                    latest_.map_from_odom->header.stamp);
      latest_.map_from_odom = transform;
      if (!duplicate_token) {
        ++latest_.tf_sequence;
      }
    }
  }
}

InputSnapshot InputStore::Capture() const {
  std::scoped_lock lock{mutex_};
  return latest_;
}

std::optional<InputSnapshot> InputStore::CaptureSynchronized() const {
  std::scoped_lock lock{mutex_};
  if (!latest_.local_map || !latest_.odometry ||
      !SameStamp(latest_.local_map->header.stamp,
                 latest_.odometry->header.stamp)) {
    return std::nullopt;
  }
  return latest_;
}

}  // namespace lunar::pure_planner_ros

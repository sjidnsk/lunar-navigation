#pragma once

#include <string>
#include <utility>

#include <mutex>
#include <optional>
#include <string>
#include <cstdint>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <grid_map_msgs/msg/grid_map.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <tf2_msgs/msg/tf_message.hpp>

namespace lunar::incremental_navigation_ros {

template <typename T>
struct AdapterResult final {
  std::optional<T> value;
  std::string reason_code;
};

struct InputSnapshot final {
  grid_map_msgs::msg::GridMap::ConstSharedPtr local_map;
  nav_msgs::msg::Odometry::ConstSharedPtr odometry;
  std::optional<geometry_msgs::msg::TransformStamped> map_from_odom;
  std::uint64_t local_sequence{};
  std::uint64_t odometry_sequence{};
  std::uint64_t tf_sequence{};
};

class InputStore final {
 public:
  InputStore(std::string map_frame = "map", std::string odom_frame = "odom")
      : map_frame_(std::move(map_frame)), odom_frame_(std::move(odom_frame)) {}
  void UpdateLocal(grid_map_msgs::msg::GridMap::ConstSharedPtr message);
  void UpdateOdometry(nav_msgs::msg::Odometry::ConstSharedPtr message);
  void UpdateTf(const tf2_msgs::msg::TFMessage& message);

  [[nodiscard]] InputSnapshot Capture() const;

 private:
  mutable std::mutex mutex_;
  InputSnapshot latest_;
  std::string map_frame_;
  std::string odom_frame_;
};

}  // namespace lunar::incremental_navigation_ros

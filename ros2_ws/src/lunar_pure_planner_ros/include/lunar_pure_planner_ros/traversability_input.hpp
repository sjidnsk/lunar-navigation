#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>

#include <grid_map_msgs/msg/grid_map.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <tf2_msgs/msg/tf_message.hpp>

#include "lunar_pure_planner_core/traversability_map.hpp"
#include "lunar_pure_planner_ros/map_adapters.hpp"

namespace lunar::pure_planner_ros {

struct TraversabilityInputSnapshot final {
  std::shared_ptr<const lunar::pure_planning::TraversabilitySnapshot> snapshot;
  std::string reason_code;
};

class TraversabilityInput final {
 public:
  explicit TraversabilityInput(lunar::pure_planning::TraversabilityProfile profile);

  void UpdateGlobal(nav_msgs::msg::OccupancyGrid::ConstSharedPtr message);
  void UpdateLocal(grid_map_msgs::msg::GridMap::ConstSharedPtr message);
  void UpdateTf(const tf2_msgs::msg::TFMessage& message);

  [[nodiscard]] TraversabilityInputSnapshot Capture() const;

 private:
  void ApplyPendingLocalLocked();

  mutable std::mutex mutex_;
  lunar::pure_planning::PersistentTraversabilityMap map_;
  std::optional<lunar::pure_planning::GridMap> latest_global_;
  std::optional<lunar::pure_planning::GridMap> latest_local_;
  std::optional<lunar::pure_planning::RigidTransform> latest_map_from_odom_;
  std::uint64_t global_sequence_{};
  std::uint64_t local_sequence_{};
  std::uint64_t tf_sequence_{};
  std::uint64_t applied_local_sequence_{};
  bool direct_tf_error_{};
  std::string reason_code_;
};

}  // namespace lunar::pure_planner_ros

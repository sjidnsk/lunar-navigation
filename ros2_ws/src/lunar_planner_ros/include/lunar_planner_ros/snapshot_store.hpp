#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <vector>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <grid_map_msgs/msg/grid_map.hpp>
#include <lunar_navigation_msgs/msg/localization_status.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <tf2_msgs/msg/tf_message.hpp>

namespace lunar::planning::ros {

struct SnapshotStoreView final {
  std::optional<grid_map_msgs::msg::GridMap> global_map;
  std::optional<grid_map_msgs::msg::GridMap> local_map;
  std::optional<nav_msgs::msg::Odometry> odometry;
  std::optional<lunar_navigation_msgs::msg::LocalizationStatus>
      localization_status;
  std::vector<geometry_msgs::msg::TransformStamped> transforms;
};

struct SnapshotContentGenerations final {
  std::uint64_t global_map{};
  std::uint64_t local_map{};
  std::uint64_t map_from_odom{};
};

class SnapshotStore final {
 public:
  explicit SnapshotStore(std::size_t maximum_transform_samples = 256U);

  void UpdateGlobalMap(const grid_map_msgs::msg::GridMap& message);
  void UpdateLocalMap(const grid_map_msgs::msg::GridMap& message);
  void UpdateOdometry(const nav_msgs::msg::Odometry& message);
  void UpdateLocalizationStatus(
      const lunar_navigation_msgs::msg::LocalizationStatus& message);
  void UpdateTransforms(const tf2_msgs::msg::TFMessage& message);
  void ClearTransforms();

  [[nodiscard]] SnapshotContentGenerations ResolveContentGenerations(
      std::uint64_t global_map_identity,
      std::uint64_t local_map_identity,
      std::uint64_t map_from_odom_identity);

  [[nodiscard]] SnapshotStoreView Capture() const;

 private:
  std::size_t maximum_transform_samples_;
  mutable std::mutex mutex_;
  SnapshotStoreView view_;
  std::optional<std::uint64_t> global_map_identity_;
  std::optional<std::uint64_t> local_map_identity_;
  std::optional<std::uint64_t> map_from_odom_identity_;
  SnapshotContentGenerations generations_;
};

}  // namespace lunar::planning::ros

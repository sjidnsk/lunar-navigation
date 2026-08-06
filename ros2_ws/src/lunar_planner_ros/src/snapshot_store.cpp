#include "lunar_planner_ros/snapshot_store.hpp"

#include <algorithm>
#include <utility>

namespace lunar::planning::ros {

SnapshotStore::SnapshotStore(const std::size_t maximum_transform_samples)
    : maximum_transform_samples_(std::max<std::size_t>(
          maximum_transform_samples, 1U)) {}

void SnapshotStore::UpdateGlobalMap(
    const grid_map_msgs::msg::GridMap& message) {
  std::scoped_lock lock{mutex_};
  view_.global_map = message;
}

void SnapshotStore::UpdateLocalMap(
    const grid_map_msgs::msg::GridMap& message) {
  std::scoped_lock lock{mutex_};
  view_.local_map = message;
}

void SnapshotStore::UpdateOdometry(const nav_msgs::msg::Odometry& message) {
  std::scoped_lock lock{mutex_};
  view_.odometry = message;
}

void SnapshotStore::UpdateLocalizationStatus(
    const lunar_navigation_msgs::msg::LocalizationStatus& message) {
  std::scoped_lock lock{mutex_};
  view_.localization_status = message;
}

void SnapshotStore::UpdateHopperPropellantState(
    const lunar_navigation_msgs::msg::HopperPropellantState& message) {
  std::scoped_lock lock{mutex_};
  view_.hopper_propellant_state = message;
}

void SnapshotStore::UpdateTransforms(const tf2_msgs::msg::TFMessage& message) {
  std::scoped_lock lock{mutex_};
  for (const auto& transform : message.transforms) {
    view_.transforms.push_back(transform);
  }
  if (view_.transforms.size() > maximum_transform_samples_) {
    const auto excess = view_.transforms.size() - maximum_transform_samples_;
    view_.transforms.erase(
        view_.transforms.begin(), view_.transforms.begin() + excess);
  }
}

void SnapshotStore::ClearTransforms() {
  std::scoped_lock lock{mutex_};
  view_.transforms.clear();
}

SnapshotContentGenerations SnapshotStore::ResolveContentGenerations(
    const std::uint64_t global_map_identity,
    const std::uint64_t local_map_identity,
    const std::uint64_t map_from_odom_identity) {
  std::scoped_lock lock{mutex_};
  const auto update = [](const std::uint64_t identity,
                         std::optional<std::uint64_t>& previous,
                         std::uint64_t& generation) {
    if (!previous.has_value() || *previous != identity) {
      previous = identity;
      ++generation;
    }
  };
  update(global_map_identity, global_map_identity_, generations_.global_map);
  update(local_map_identity, local_map_identity_, generations_.local_map);
  update(
      map_from_odom_identity,
      map_from_odom_identity_,
      generations_.map_from_odom);
  return generations_;
}

SnapshotStoreView SnapshotStore::Capture() const {
  std::scoped_lock lock{mutex_};
  return view_;
}

}  // namespace lunar::planning::ros

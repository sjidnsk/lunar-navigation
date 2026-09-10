#include "lunar_incremental_navigation_ros/incremental_map_publisher.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

namespace lunar::incremental_navigation_ros {

rclcpp::QoS ExplorationMapQos() {
  return rclcpp::QoS{rclcpp::KeepLast{1}}.reliable().transient_local();
}

rclcpp::QoS ConfiguredExplorationMapQos(rclcpp::Node& node) {
  const auto reliability = node.declare_parameter<std::string>("exploration_map_qos_reliability", "reliable");
  const auto durability = node.declare_parameter<std::string>("exploration_map_qos_durability", "transient_local");
  const auto depth = node.declare_parameter<int>("exploration_map_qos_depth", 1);
  if (depth <= 0) throw std::invalid_argument("exploration_map_qos_depth must be positive");
  rclcpp::QoS qos{rclcpp::KeepLast{static_cast<std::size_t>(depth)}};
  if (reliability == "reliable") qos.reliable();
  else if (reliability == "best_effort") qos.best_effort();
  else throw std::invalid_argument("exploration_map_qos_reliability: expected reliable or best_effort");
  if (durability == "transient_local") qos.transient_local();
  else if (durability == "volatile") qos.durability_volatile();
  else throw std::invalid_argument("exploration_map_qos_durability: expected transient_local or volatile");
  return qos;
}

IncrementalMapPublisher::IncrementalMapPublisher(rclcpp::Node& node,
                                                 std::string topic)
    : clock_(node.get_clock()),
      publisher_(node.create_publisher<nav_msgs::msg::OccupancyGrid>(
          std::move(topic), ConfiguredExplorationMapQos(node))) {}

bool IncrementalMapPublisher::Publish(
    const lunar::incremental_navigation::SnapshotBundle& bundle) {
  if (!bundle.fine || !bundle.guidance ||
      bundle.guidance->source_raw_elevation_revision() !=
          bundle.fine->raw_elevation_revision() ||
      bundle.guidance->source_fine_traversability_revision() !=
          bundle.fine->fine_traversability_revision()) {
    return false;
  }
  const ExplorationMapProjection projection =
      projector_.Project(*bundle.fine, bundle.guidance->geometry());
  const auto minimum = projection.geometry.min_inclusive();
  const auto maximum = projection.geometry.max_exclusive();
  const std::uint64_t width = static_cast<std::uint64_t>(maximum.x - minimum.x);
  const std::uint64_t height = static_cast<std::uint64_t>(maximum.y - minimum.y);
  if (width > std::numeric_limits<std::uint32_t>::max() ||
      height > std::numeric_limits<std::uint32_t>::max()) {
    throw std::overflow_error("exploration projection exceeds OccupancyGrid");
  }

  nav_msgs::msg::OccupancyGrid message;
  message.header.frame_id = projection.geometry.frame_id();
  message.header.stamp = clock_->now();
  message.info.resolution =
      static_cast<float>(projection.geometry.resolution_m());
  message.info.width = static_cast<std::uint32_t>(width);
  message.info.height = static_cast<std::uint32_t>(height);
  const auto origin = projection.geometry.origin_m();
  message.info.origin.position.x = std::fma(
      static_cast<double>(minimum.x), projection.geometry.resolution_m(),
      origin.x);
  message.info.origin.position.y = std::fma(
      static_cast<double>(minimum.y), projection.geometry.resolution_m(),
      origin.y);
  message.info.origin.position.z = origin.z;
  message.info.origin.orientation.w = 1.0;
  message.data = projection.data;
  publisher_->publish(std::move(message));
  return true;
}

}  // namespace lunar::incremental_navigation_ros

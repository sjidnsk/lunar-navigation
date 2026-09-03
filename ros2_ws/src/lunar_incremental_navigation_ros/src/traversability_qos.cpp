#include "lunar_incremental_navigation_ros/traversability_qos.hpp"

namespace lunar::incremental_navigation_ros {

rclcpp::QoS MakeTask3LocalMapInputQos() {
  return rclcpp::QoS{rclcpp::KeepLast{1}}.reliable().transient_local();
}

std::optional<rclcpp::QoS> MakeTraversabilityInputQos(
    const std::string_view reliability, const std::string_view durability) {
  rclcpp::QoS qos{rclcpp::KeepLast{1}};
  if (reliability == "reliable") {
    qos.reliable();
  } else if (reliability == "best_effort") {
    qos.best_effort();
  } else {
    return std::nullopt;
  }
  if (durability == "transient_local") {
    qos.transient_local();
  } else if (durability == "volatile") {
    qos.durability_volatile();
  } else {
    return std::nullopt;
  }
  return qos;
}

}  // namespace lunar::incremental_navigation_ros

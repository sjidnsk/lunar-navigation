#include "lunar_pure_planner_ros/traversability_qos.hpp"

namespace lunar::pure_planner_ros {

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

}  // namespace lunar::pure_planner_ros

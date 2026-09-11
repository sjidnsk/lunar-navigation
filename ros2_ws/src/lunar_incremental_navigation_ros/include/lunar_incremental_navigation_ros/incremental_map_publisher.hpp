#pragma once

#include <string>
#include <optional>

#include <nav_msgs/msg/occupancy_grid.hpp>
#include <rclcpp/node.hpp>
#include <rclcpp/qos.hpp>

#include "lunar_incremental_navigation_core/local_planning.hpp"
#include "lunar_incremental_navigation_core/types/platform_capability.hpp"
#include "lunar_incremental_navigation_ros/exploration_map_projector.hpp"

namespace lunar::incremental_navigation_ros {

[[nodiscard]] rclcpp::QoS ExplorationMapQos();

class IncrementalMapPublisher final {
 public:
  IncrementalMapPublisher(rclcpp::Node& node, std::string topic);

  [[nodiscard]] bool Publish(
      const lunar::incremental_navigation::SnapshotBundle& bundle);

  void PublishLocalFine(
      const lunar::incremental_navigation::FineTraversabilitySnapshot& fine,
      lunar::incremental_navigation::Point2 center);

 private:
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr local_fine_publisher_;
  double local_fine_window_m_{28.0};
  std::optional<std::uint64_t> last_fine_revision_;
  std::optional<lunar::incremental_navigation::Point2> last_fine_center_;
  ExplorationMapProjector projector_;
  rclcpp::Clock::SharedPtr clock_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr publisher_;
};

}  // namespace lunar::incremental_navigation_ros

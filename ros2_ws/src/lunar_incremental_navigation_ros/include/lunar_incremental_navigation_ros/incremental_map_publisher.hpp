#pragma once

#include <string>

#include <nav_msgs/msg/occupancy_grid.hpp>
#include <rclcpp/node.hpp>
#include <rclcpp/qos.hpp>

#include "lunar_incremental_navigation_core/local_planning.hpp"
#include "lunar_incremental_navigation_ros/exploration_map_projector.hpp"

namespace lunar::incremental_navigation_ros {

[[nodiscard]] rclcpp::QoS ExplorationMapQos();

class IncrementalMapPublisher final {
 public:
  IncrementalMapPublisher(rclcpp::Node& node, std::string topic);

  [[nodiscard]] bool Publish(
      const lunar::incremental_navigation::SnapshotBundle& bundle);

 private:
  ExplorationMapProjector projector_;
  rclcpp::Clock::SharedPtr clock_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr publisher_;
};

}  // namespace lunar::incremental_navigation_ros

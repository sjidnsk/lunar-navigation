#include <memory>

#include <rclcpp/rclcpp.hpp>

#include "lunar_pure_exploration_ros/incremental_exploration_node.hpp"

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(
        std::make_shared<lunar::pure_exploration_ros::
                             IncrementalExplorationNode>());
  } catch (const std::exception& error) {
    RCLCPP_FATAL(rclcpp::get_logger("incremental_exploration_node"), "%s",
                 error.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}

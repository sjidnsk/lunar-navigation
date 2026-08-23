#include <exception>
#include <memory>

#include <rclcpp/rclcpp.hpp>

#include "lunar_pure_exploration_ros/exploration_node.hpp"

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  int exit_code = 0;
  try {
    rclcpp::spin(
        std::make_shared<lunar::pure_exploration_ros::ExplorationNode>());
  } catch (const std::exception& error) {
    RCLCPP_FATAL(rclcpp::get_logger("pure_exploration"), "%s",
                 error.what());
    exit_code = 1;
  }
  rclcpp::shutdown();
  return exit_code;
}

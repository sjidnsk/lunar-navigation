#include <exception>
#include <memory>

#include <rclcpp/rclcpp.hpp>

#include "lunar_incremental_navigation_ros/incremental_navigation_node.hpp"

int main(int argc, char* argv[]) {
  rclcpp::init(argc, argv);
  try {
    auto node =
        std::make_shared<lunar::incremental_navigation_ros::IncrementalNavigationNode>();
    auto executor = lunar::incremental_navigation_ros::MakeIncrementalNavigationExecutor();
    executor->add_node(node);
    executor->spin();
    executor->remove_node(node);
    rclcpp::shutdown();
    return 0;
  } catch (const std::exception& error) {
    RCLCPP_FATAL(rclcpp::get_logger("lunar_incremental_navigation_node"), "%s",
                 error.what());
    rclcpp::shutdown();
    return 1;
  }
}

#include <exception>
#include <memory>

#include <rclcpp/rclcpp.hpp>

#include "lunar_incremental_navigation_ros/rviz_goal_bridge.hpp"

int main(int argc, char* argv[]) {
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(
        std::make_shared<lunar::incremental_navigation_ros::RvizGoalBridge>());
    rclcpp::shutdown();
    return 0;
  } catch (const std::exception& error) {
    RCLCPP_FATAL(rclcpp::get_logger("lunar_incremental_rviz_goal_bridge"),
                 "%s", error.what());
    rclcpp::shutdown();
    return 1;
  }
}

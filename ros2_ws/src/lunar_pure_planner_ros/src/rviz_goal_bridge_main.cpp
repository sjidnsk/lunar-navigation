#include <memory>

#include <rclcpp/rclcpp.hpp>

#include "lunar_pure_planner_ros/rviz_goal_bridge.hpp"

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<lunar::pure_planner_ros::RvizGoalBridge>());
  rclcpp::shutdown();
  return 0;
}

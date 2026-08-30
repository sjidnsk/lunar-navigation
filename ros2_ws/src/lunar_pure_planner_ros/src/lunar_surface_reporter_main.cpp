#include <rclcpp/rclcpp.hpp>

#include "lunar_pure_planner_ros/lunar_surface_reporter.hpp"

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(lunar::pure_planner_ros::MakeLunarSurfaceReporterNode());
  rclcpp::shutdown();
  return 0;
}

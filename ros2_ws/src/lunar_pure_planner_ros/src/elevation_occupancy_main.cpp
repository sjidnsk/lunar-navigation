#include <exception>
#include <memory>

#include <rclcpp/rclcpp.hpp>

namespace lunar::pure_planner_ros {
std::shared_ptr<rclcpp::Node> MakeElevationOccupancyNode();
}

int main(int argc, char* argv[]) {
  rclcpp::init(argc, argv);
  try {
    auto node = lunar::pure_planner_ros::MakeElevationOccupancyNode();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
  } catch (const std::exception& error) {
    RCLCPP_FATAL(rclcpp::get_logger("lunar_elevation_occupancy"), "%s", error.what());
    rclcpp::shutdown();
    return 1;
  }
}

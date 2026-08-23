#include <exception>
#include <memory>

#include <rclcpp/rclcpp.hpp>

namespace lunar::pure_planner_ros {
std::shared_ptr<rclcpp::Node> MakeLocalTraversabilityNode();
}

int main(int argc, char* argv[]) {
  rclcpp::init(argc, argv);
  try {
    auto node = lunar::pure_planner_ros::MakeLocalTraversabilityNode();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
  } catch (const std::exception& error) {
    RCLCPP_FATAL(rclcpp::get_logger("lunar_local_traversability"), "%s", error.what());
    rclcpp::shutdown();
    return 1;
  }
}

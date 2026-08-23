#include <exception>
#include <memory>

#include <rclcpp/rclcpp.hpp>

#include "lunar_pure_planner_ros/pure_plan_motion_server.hpp"

int main(int argc, char* argv[]) {
  rclcpp::init(argc, argv);
  try {
    auto node =
        std::make_shared<lunar::pure_planner_ros::PurePlanMotionServer>();
    auto executor = lunar::pure_planner_ros::MakePurePlannerExecutor();
    executor->add_node(node);
    executor->spin();
    executor->remove_node(node);
    rclcpp::shutdown();
    return 0;
  } catch (const std::exception& error) {
    RCLCPP_FATAL(rclcpp::get_logger("lunar_pure_planner_node"), "%s",
                 error.what());
    rclcpp::shutdown();
    return 1;
  }
}

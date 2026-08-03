#include <memory>

#include <rclcpp/executors/multi_threaded_executor.hpp>
#include <rclcpp/rclcpp.hpp>

#include "lunar_planner_ros/plan_motion_server.hpp"

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<lunar::planning::ros::PlanMotionServer>();
  rclcpp::executors::MultiThreadedExecutor executor{
      rclcpp::ExecutorOptions{}, 4U};
  executor.add_node(node->get_node_base_interface());
  executor.spin();
  executor.remove_node(node->get_node_base_interface());
  node.reset();
  rclcpp::shutdown();
  return 0;
}

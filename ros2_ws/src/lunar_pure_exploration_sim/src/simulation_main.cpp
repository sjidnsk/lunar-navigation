#include <memory>

#include <rclcpp/rclcpp.hpp>

#include "lunar_pure_exploration_sim/simulation_node.hpp"

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(
      std::make_shared<lunar::pure_exploration_sim::SimulationNode>());
  rclcpp::shutdown();
  return 0;
}

#include <memory>

#include <rclcpp/rclcpp.hpp>

#include "lunar_pure_exploration_sim/run_coordinator.hpp"

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(
      std::make_shared<lunar::pure_exploration_sim::RunCoordinator>());
  rclcpp::shutdown();
  return 0;
}

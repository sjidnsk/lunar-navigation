#include <memory>

#include <rclcpp/rclcpp.hpp>

#include "lunar_pure_exploration_sim/run_recorder.hpp"

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node =
      std::make_shared<lunar::pure_exploration_sim::RunRecorderNode>();
  rclcpp::spin(node);
  node->FinalizeShutdown();
  rclcpp::shutdown();
  return 0;
}

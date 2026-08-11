#include <memory>

#include <rclcpp/rclcpp.hpp>

#include "lunar_observed_map/observed_map_node.hpp"

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<lunar::observed_map::ObservedMapNode>();
  rclcpp::spin(node->get_node_base_interface());
  rclcpp::shutdown();
  return 0;
}

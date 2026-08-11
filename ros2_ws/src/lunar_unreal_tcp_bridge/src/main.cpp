#include <memory>

#include <rclcpp/rclcpp.hpp>

#include "lunar_unreal_tcp_bridge/bridge_node.hpp"

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  const auto node = std::make_shared<lunar::unreal_tcp::BridgeNode>();
  rclcpp::spin(node->get_node_base_interface());
  rclcpp::shutdown();
  return 0;
}

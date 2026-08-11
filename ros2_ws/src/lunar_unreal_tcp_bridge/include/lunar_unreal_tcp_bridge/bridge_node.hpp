#pragma once

#include <chrono>
#include <functional>
#include <memory>
#include <string>

#include <lunar_planning_msgs/msg/motion_reference.hpp>
#include <rclcpp/node_options.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>

#include "lunar_unreal_tcp_bridge/session_protocol.hpp"
#include "lunar_unreal_tcp_bridge/tcp_client.hpp"

namespace lunar::unreal_tcp {

struct BridgeNodeDependencies final {
  std::shared_ptr<SessionTransport> transport;
  std::function<std::chrono::steady_clock::time_point()> steady_now;
};

class BridgeNode final : public rclcpp_lifecycle::LifecycleNode {
 public:
  explicit BridgeNode(
      const rclcpp::NodeOptions& options = rclcpp::NodeOptions{},
      BridgeNodeDependencies dependencies = {});
  ~BridgeNode() override;

  BridgeNode(const BridgeNode&) = delete;
  BridgeNode& operator=(const BridgeNode&) = delete;

  CallbackReturn on_configure(
      const rclcpp_lifecycle::State& state) override;
  CallbackReturn on_activate(
      const rclcpp_lifecycle::State& state) override;
  CallbackReturn on_deactivate(
      const rclcpp_lifecycle::State& state) override;
  CallbackReturn on_cleanup(
      const rclcpp_lifecycle::State& state) override;
  CallbackReturn on_shutdown(
      const rclcpp_lifecycle::State& state) override;
  CallbackReturn on_error(
      const rclcpp_lifecycle::State& state) override;

  void DrainEventsForTesting();
  void CheckWatchdogsForTesting();
  void ReceiveMotionReferenceForTesting(
      const lunar_planning_msgs::msg::MotionReference& message);
  [[nodiscard]] SessionState session_state_for_testing() const noexcept;
  [[nodiscard]] std::string last_reason_for_testing() const;
  [[nodiscard]] bool configured_for_testing() const noexcept;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace lunar::unreal_tcp

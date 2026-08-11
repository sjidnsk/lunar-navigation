#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <grid_map_msgs/msg/grid_map.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/node_options.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>

#include "lunar_observed_map/map_types.hpp"

namespace lunar::observed_map {

enum class ObservedMapNodeState : std::uint8_t {
  kUnconfigured,
  kInactive,
  kSyncing,
  kReady,
  kError,
};

class ObservedMapNode final : public rclcpp_lifecycle::LifecycleNode {
 public:
  explicit ObservedMapNode(
      const rclcpp::NodeOptions& options = rclcpp::NodeOptions{});
  ~ObservedMapNode() override;

  ObservedMapNode(const ObservedMapNode&) = delete;
  ObservedMapNode& operator=(const ObservedMapNode&) = delete;

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

  void ReceiveObservedMapForTesting(
      const grid_map_msgs::msg::GridMap& message);
  void ReceiveOdometryForTesting(const nav_msgs::msg::Odometry& message);
  void ReceiveGoalForTesting(const geometry_msgs::msg::PoseStamped& message);
  void ReceiveBridgeStatusForTesting(
      const diagnostic_msgs::msg::DiagnosticArray& message);
  [[nodiscard]] FuseResult FusePatchForTesting(const ObservedPatch& patch);
  void ResetSessionForTesting(std::string session_id);
  [[nodiscard]] std::optional<grid_map_msgs::msg::GridMap>
      last_local_map_for_testing() const;
  [[nodiscard]] std::optional<grid_map_msgs::msg::GridMap>
      last_global_map_for_testing() const;
  [[nodiscard]] std::uint64_t generation_for_testing() const noexcept;
  [[nodiscard]] ObservedMapNodeState state_for_testing() const noexcept;
  [[nodiscard]] std::string last_reason_for_testing() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace lunar::observed_map

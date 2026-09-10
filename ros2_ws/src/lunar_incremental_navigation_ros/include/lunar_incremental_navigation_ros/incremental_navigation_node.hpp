#pragma once

#include <functional>
#include <memory>
#include <optional>
#include <string>

#include <rclcpp/executor.hpp>
#include <rclcpp/node.hpp>
#include <rclcpp/node_options.hpp>
#include <rclcpp/qos.hpp>

#include "lunar_incremental_navigation_core/planning_session_coordinator.hpp"

namespace lunar::incremental_navigation_ros {

enum class ActionTerminalState {
  kSucceeded,
  kCanceled,
  kAborted,
};

enum class CancelHandlingDecision {
  kDefer,
  kFinalize,
  kIgnore,
};

[[nodiscard]] ActionTerminalState TerminalStateFor(
    lunar::incremental_navigation::SessionOutcome outcome) noexcept;
[[nodiscard]] CancelHandlingDecision ClassifyCancelHandling(
    bool active, bool canceling) noexcept;
[[nodiscard]] rclcpp::QoS PathReferenceQos();
[[nodiscard]] rclcpp::QoS StateInputQos();
[[nodiscard]] lunar::incremental_navigation::TraversabilityProfile PlatformProfileFor(
    const lunar::incremental_navigation::PlatformCapability& capability,
    double start_blind_zone_margin_m);

using SessionPortsFactory = std::function<
    lunar::incremental_navigation::PlanningSessionPorts(
        const lunar::incremental_navigation::PlatformCapability&)>;

struct IncrementalNavigationNodeDependencies final {
  SessionPortsFactory ports_factory;
  std::optional<lunar::incremental_navigation::SnapshotBundle> snapshots;
  std::optional<lunar::incremental_navigation::StateInput> state;
  std::function<std::optional<lunar::incremental_navigation::StateInput>()> state_source;
  std::function<std::optional<lunar::incremental_navigation::SnapshotBundle>()> snapshot_source;
  std::function<void(const std::string&)> event_sink;
  std::function<void()> before_goal_processing;
  std::function<void()> before_terminal_commit;
};

[[nodiscard]] std::unique_ptr<rclcpp::Executor> MakeIncrementalNavigationExecutor();
[[nodiscard]] lunar::incremental_navigation::PlanningSessionPorts RealSessionPorts(
    const lunar::incremental_navigation::PlatformCapability& capability);

class IncrementalNavigationNode final : public rclcpp::Node {
 public:
  explicit IncrementalNavigationNode(
      const rclcpp::NodeOptions& options = rclcpp::NodeOptions{},
      IncrementalNavigationNodeDependencies dependencies = {});
  ~IncrementalNavigationNode() override;

  IncrementalNavigationNode(const IncrementalNavigationNode&) = delete;
  IncrementalNavigationNode& operator=(const IncrementalNavigationNode&) = delete;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr startup_parameters_;
};

}  // namespace lunar::incremental_navigation_ros

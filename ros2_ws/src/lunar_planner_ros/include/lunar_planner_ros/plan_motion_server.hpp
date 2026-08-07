#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>

#include <rclcpp/node_options.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>

#include "lunar_planner_core/planner.hpp"
#include "lunar_planner_ros/capability_loader.hpp"

namespace lunar::planning::ros {

using PlannerFunction = std::function<lunar::planning::PlannerOutput(
    const lunar::planning::PlannerInput&)>;
using ObservingPlannerFunction = std::function<lunar::planning::PlannerOutput(
    const lunar::planning::PlannerInput&,
    const lunar::planning::ProvisionalRouteObserver&)>;

struct PlanMotionServerDependencies final {
  PlannerFunction planner;
  ObservingPlannerFunction observing_planner;
  std::optional<LoadedCapabilities> preloaded_capabilities;
};

class PlanMotionServer final : public rclcpp_lifecycle::LifecycleNode {
 public:
  explicit PlanMotionServer(
      const rclcpp::NodeOptions& options = rclcpp::NodeOptions{},
      PlanMotionServerDependencies dependencies = {});
  ~PlanMotionServer() override;

  PlanMotionServer(const PlanMotionServer&) = delete;
  PlanMotionServer& operator=(const PlanMotionServer&) = delete;

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

  [[nodiscard]] std::size_t callback_group_count_for_testing() const noexcept;
  [[nodiscard]] bool callback_groups_mutually_exclusive_for_testing() const;
  [[nodiscard]] bool worker_active_for_testing() const;
  [[nodiscard]] std::optional<std::uint64_t>
  mission_revision_for_testing() const;
  [[nodiscard]] std::string last_diagnostic_reason_for_testing() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace lunar::planning::ros

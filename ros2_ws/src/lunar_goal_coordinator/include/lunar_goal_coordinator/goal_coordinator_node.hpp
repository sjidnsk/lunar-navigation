#pragma once

#include <functional>
#include <memory>
#include <optional>
#include <string>

#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <grid_map_msgs/msg/grid_map.hpp>
#include <lunar_navigation_msgs/msg/exploration_task.hpp>
#include <lunar_navigation_msgs/msg/motion_execution_feedback.hpp>
#include <lunar_planning_msgs/action/plan_motion.hpp>
#include <lunar_planning_msgs/msg/motion_reference.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/node_options.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>
#include <tf2_msgs/msg/tf_message.hpp>

#include "lunar_goal_coordinator/goal_state_machine.hpp"

namespace lunar::goal_coordinator {

struct PlanCompletion final {
  bool action_succeeded{};
  std::string reason_code;
  lunar_planning_msgs::action::PlanMotion::Result result;
};

class PlanMotionTransport {
 public:
  using ResultCallback = std::function<void(PlanCompletion)>;
  using CancelCallback = std::function<void(bool)>;

  virtual ~PlanMotionTransport() = default;
  [[nodiscard]] virtual bool Available() const = 0;
  virtual void Send(
      const lunar_planning_msgs::action::PlanMotion::Goal& goal,
      ResultCallback callback) = 0;
  virtual void Cancel(CancelCallback callback) = 0;
};

struct GoalCoordinatorDependencies final {
  std::shared_ptr<PlanMotionTransport> planner;
  std::function<std::string()> target_uuid;
  std::function<void()> request_hold;
};

class GoalCoordinatorNode final : public rclcpp_lifecycle::LifecycleNode {
 public:
  explicit GoalCoordinatorNode(
      const rclcpp::NodeOptions& options = rclcpp::NodeOptions{},
      GoalCoordinatorDependencies dependencies = {});
  ~GoalCoordinatorNode() override;

  GoalCoordinatorNode(const GoalCoordinatorNode&) = delete;
  GoalCoordinatorNode& operator=(const GoalCoordinatorNode&) = delete;

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

  void ReceiveGoalForTesting(const geometry_msgs::msg::PoseStamped& message);
  void ReceiveGlobalMapForTesting(const grid_map_msgs::msg::GridMap& message);
  void ReceiveLocalMapForTesting(const grid_map_msgs::msg::GridMap& message);
  void ReceiveOdometryForTesting(const nav_msgs::msg::Odometry& message);
  void ReceiveTfForTesting(const tf2_msgs::msg::TFMessage& message);
  void ReceiveExecutionFeedbackForTesting(
      const lunar_navigation_msgs::msg::MotionExecutionFeedback& message);
  void ReceiveBridgeStatusForTesting(
      const diagnostic_msgs::msg::DiagnosticArray& message);
  [[nodiscard]] bool CancelForTesting();
  void DispatchForTesting();

  [[nodiscard]] CoordinatorState state_for_testing() const noexcept;
  [[nodiscard]] std::string reason_for_testing() const;
  [[nodiscard]] std::optional<lunar_navigation_msgs::msg::ExplorationTask>
      last_mission_for_testing() const;
  [[nodiscard]] std::optional<lunar_planning_msgs::msg::MotionReference>
      last_reference_for_testing() const;
  [[nodiscard]] std::optional<lunar_planning_msgs::action::PlanMotion::Goal>
      last_plan_goal_for_testing() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace lunar::goal_coordinator

#pragma once

#include <cstdint>
#include <memory>

#include <grid_map_msgs/msg/grid_map.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <lunar_planning_msgs/action/plan_motion.hpp>
#include <lunar_pure_exploration_msgs/msg/pure_exploration_status.hpp>
#include <lunar_pure_exploration_msgs/msg/pure_exploration_task.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <tf2_msgs/msg/tf_message.hpp>

namespace lunar::pure_exploration_sim {

struct CoordinatorReadiness {
  bool global_map_received{false};
  bool local_map_received{false};
  bool odometry_received{false};
  bool tf_chain_received{false};
  bool initial_status_received{false};
  bool planner_action_ready{false};
  bool controller_publisher_unique{false};
  bool controller_command_received{false};

  [[nodiscard]] bool ShouldStart() const noexcept;
  void MarkStarted() noexcept;

 private:
  bool started_{false};
};

class RequiredTfChain {
 public:
  void Observe(const tf2_msgs::msg::TFMessage& message) noexcept;
  [[nodiscard]] bool complete() const noexcept;

 private:
  bool map_to_odom_{false};
  bool odom_to_base_link_{false};
};

[[nodiscard]] bool IsInitialExplorationStatus(
    const lunar_pure_exploration_msgs::msg::PureExplorationStatus& status)
    noexcept;

[[nodiscard]] bool IsFiniteControllerCommand(
    const geometry_msgs::msg::Twist& command) noexcept;

[[nodiscard]] lunar_pure_exploration_msgs::msg::PureExplorationTask
MakeExplorationStartTask(std::uint64_t seed, const rclcpp::Time& stamp);

class RunCoordinator final : public rclcpp::Node {
 public:
  RunCoordinator();
  explicit RunCoordinator(const rclcpp::NodeOptions& options);

 private:
  void PollReadiness();

  std::uint64_t seed_{20260824U};
  CoordinatorReadiness readiness_;
  RequiredTfChain required_tf_chain_;

  rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr global_sub_;
  rclcpp::Subscription<grid_map_msgs::msg::GridMap>::SharedPtr local_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odometry_sub_;
  rclcpp::Subscription<tf2_msgs::msg::TFMessage>::SharedPtr tf_sub_;
  rclcpp::Subscription<
      lunar_pure_exploration_msgs::msg::PureExplorationStatus>::SharedPtr
      status_sub_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr
      controller_command_sub_;
  rclcpp_action::Client<lunar_planning_msgs::action::PlanMotion>::SharedPtr
      planner_client_;
  rclcpp::Publisher<
      lunar_pure_exploration_msgs::msg::PureExplorationTask>::SharedPtr
      task_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace lunar::pure_exploration_sim

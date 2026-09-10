#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include <lunar_pure_exploration_core/candidate_generator.hpp>
#include <lunar_pure_exploration_core/candidate_ranker.hpp>
#include <lunar_pure_exploration_core/failure_memory.hpp>
#include <lunar_pure_exploration_core/information_gain.hpp>
#include <lunar_pure_exploration_core/task_raster.hpp>
#include <lunar_pure_exploration_msgs/msg/pure_exploration_task.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/node.hpp>
#include <rclcpp/subscription.hpp>
#include <rclcpp/timer.hpp>
#include <tf2_msgs/msg/tf_message.hpp>

namespace lunar::pure_exploration_ros {

struct IncrementalExplorationNodeParameters final {
  lunar::pure_exploration::PlatformGeometry platform;
  lunar::pure_exploration::CandidateParameters candidate_parameters;
  lunar::pure_exploration::CandidateGenerator::Limits candidate_limits;
  lunar::pure_exploration::TaskRaster::Limits task_raster_limits;
  lunar::pure_exploration::SensorModel sensor_model;
  lunar::pure_exploration::InformationGainEvaluator::Limits
      information_gain_limits;
  lunar::pure_exploration::ScoreWeights score_weights;
  lunar::pure_exploration::FailureMemoryLimits failure_memory_limits;
  double minimum_frontier_length_m;
  double coverage_target;
  std::string exploration_map_topic;
  std::string odometry_topic;
  std::string tf_topic;
  std::string task_topic;
  std::string navigation_action;
  std::string status_topic;
  std::string task_boundary_topic;
  std::string task_map_markers_topic;
  std::string current_goal_topic;
  std::string frontiers_topic;
  std::string diagnostics_topic;
  double navigation_map_wait_timeout_s{30.0};
  std::string map_frame{"map"}, odom_frame{"odom"}, base_frame{"base_link"};
};

class IncrementalExplorationNode final : public rclcpp::Node {
 public:
  struct Runtime;

  explicit IncrementalExplorationNode(
      IncrementalExplorationNodeParameters parameters,
      const rclcpp::NodeOptions& options = rclcpp::NodeOptions{});
  explicit IncrementalExplorationNode(
      const rclcpp::NodeOptions& options = rclcpp::NodeOptions{});
  ~IncrementalExplorationNode() noexcept override;

  IncrementalExplorationNode(const IncrementalExplorationNode&) = delete;
  IncrementalExplorationNode& operator=(const IncrementalExplorationNode&) =
      delete;
  IncrementalExplorationNode(IncrementalExplorationNode&&) = delete;
  IncrementalExplorationNode& operator=(IncrementalExplorationNode&&) = delete;

 private:
  void Initialize(IncrementalExplorationNodeParameters parameters);

  std::shared_ptr<Runtime> runtime_;
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr startup_parameters_;
  rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr
      exploration_map_subscription_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr
      odometry_subscription_;
  rclcpp::Subscription<tf2_msgs::msg::TFMessage>::SharedPtr tf_subscription_;
  rclcpp::Subscription<
      lunar_pure_exploration_msgs::msg::PureExplorationTask>::SharedPtr
      task_subscription_;
  rclcpp::TimerBase::SharedPtr navigation_map_wait_timer_;
};

}  // namespace lunar::pure_exploration_ros

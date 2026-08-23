#include "lunar_pure_exploration_sim/run_coordinator.hpp"

#include <chrono>
#include <string>
#include <utility>

#include <geometry_msgs/msg/point32.hpp>

namespace lunar::pure_exploration_sim {
namespace {

using Action = lunar_planning_msgs::action::PlanMotion;
using Status = lunar_pure_exploration_msgs::msg::PureExplorationStatus;
using Task = lunar_pure_exploration_msgs::msg::PureExplorationTask;

geometry_msgs::msg::Point32 BoundaryPoint(const float x, const float y) {
  geometry_msgs::msg::Point32 point;
  point.x = x;
  point.y = y;
  return point;
}

}  // namespace

bool CoordinatorReadiness::ShouldStart() const noexcept {
  return !started_ && global_map_received && local_map_received &&
         odometry_received && tf_chain_received && initial_status_received &&
         planner_action_ready;
}

void CoordinatorReadiness::MarkStarted() noexcept { started_ = true; }

void RequiredTfChain::Observe(
    const tf2_msgs::msg::TFMessage& message) noexcept {
  for (const auto& transform : message.transforms) {
    if (transform.header.frame_id == "map" &&
        transform.child_frame_id == "odom") {
      map_to_odom_ = true;
    }
    if (transform.header.frame_id == "odom" &&
        transform.child_frame_id == "base_link") {
      odom_to_base_link_ = true;
    }
  }
}

bool RequiredTfChain::complete() const noexcept {
  return map_to_odom_ && odom_to_base_link_;
}

bool IsInitialExplorationStatus(const Status& status) noexcept {
  return status.state == Status::IDLE && status.task_id.empty();
}

Task MakeExplorationStartTask(const std::uint64_t seed,
                              const rclcpp::Time& stamp) {
  Task task;
  task.header.stamp = stamp;
  task.header.frame_id = "map";
  task.task_id = "jazzy-300m-" + std::to_string(seed);
  task.command = Task::START;
  task.boundary.points = {
      BoundaryPoint(-145.0F, -145.0F), BoundaryPoint(145.0F, -145.0F),
      BoundaryPoint(145.0F, 145.0F), BoundaryPoint(-145.0F, 145.0F)};
  return task;
}

RunCoordinator::RunCoordinator() : RunCoordinator(rclcpp::NodeOptions{}) {}

RunCoordinator::RunCoordinator(const rclcpp::NodeOptions& options)
    : rclcpp::Node("exploration_run_coordinator", options) {
  seed_ = static_cast<std::uint64_t>(
      declare_parameter<std::int64_t>("seed", 20260824));
  const auto global_topic = declare_parameter<std::string>(
      "global_overview_topic", "/Car/T3/mapping/global_overview");
  const auto local_topic = declare_parameter<std::string>(
      "local_grid_map_topic", "/Car/T3/mapping/grid_map");
  const auto odometry_topic = declare_parameter<std::string>(
      "odometry_topic", "/Car/T3/localization/odometry");
  const auto tf_topic = declare_parameter<std::string>("tf_topic", "/tf");
  const auto status_topic = declare_parameter<std::string>(
      "exploration_status_topic", "/Car/T4/exploration/status");
  const auto task_topic = declare_parameter<std::string>(
      "exploration_task_topic", "/Car/T4/exploration/task");
  const auto planner_action = declare_parameter<std::string>(
      "planner_action", "/Car/T4/plan_motion");

  const auto input_qos = rclcpp::QoS{rclcpp::KeepLast{10}}.reliable();
  global_sub_ = create_subscription<nav_msgs::msg::OccupancyGrid>(
      global_topic, input_qos,
      [this](nav_msgs::msg::OccupancyGrid::ConstSharedPtr) {
        readiness_.global_map_received = true;
      });
  local_sub_ = create_subscription<grid_map_msgs::msg::GridMap>(
      local_topic, input_qos,
      [this](grid_map_msgs::msg::GridMap::ConstSharedPtr) {
        readiness_.local_map_received = true;
      });
  odometry_sub_ = create_subscription<nav_msgs::msg::Odometry>(
      odometry_topic, input_qos,
      [this](nav_msgs::msg::Odometry::ConstSharedPtr) {
        readiness_.odometry_received = true;
      });
  tf_sub_ = create_subscription<tf2_msgs::msg::TFMessage>(
      tf_topic, input_qos,
      [this](tf2_msgs::msg::TFMessage::ConstSharedPtr message) {
        required_tf_chain_.Observe(*message);
        readiness_.tf_chain_received = required_tf_chain_.complete();
      });
  status_sub_ = create_subscription<Status>(
      status_topic, rclcpp::QoS{1}.reliable().transient_local(),
      [this](Status::ConstSharedPtr message) {
        if (IsInitialExplorationStatus(*message)) {
          readiness_.initial_status_received = true;
        }
      });

  task_pub_ = create_publisher<Task>(
      task_topic, rclcpp::QoS{1}.reliable().transient_local());
  planner_client_ = rclcpp_action::create_client<Action>(this, planner_action);
  timer_ = create_wall_timer(std::chrono::milliseconds{100},
                             [this] { PollReadiness(); });
}

void RunCoordinator::PollReadiness() {
  readiness_.planner_action_ready = planner_client_->action_server_is_ready();
  if (!readiness_.ShouldStart()) {
    return;
  }
  readiness_.MarkStarted();
  task_pub_->publish(MakeExplorationStartTask(seed_, now()));
  RCLCPP_INFO(get_logger(), "started exploration task jazzy-300m-%lu",
              static_cast<unsigned long>(seed_));
}

}  // namespace lunar::pure_exploration_sim

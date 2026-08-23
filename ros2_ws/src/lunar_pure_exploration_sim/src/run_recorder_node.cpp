#include "lunar_pure_exploration_sim/run_recorder.hpp"

#include <chrono>
#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace lunar::pure_exploration_sim {

RunRecorderNode::RunRecorderNode() : RunRecorderNode(rclcpp::NodeOptions{}) {}

RunRecorderNode::RunRecorderNode(const rclcpp::NodeOptions& options)
    : rclcpp::Node("exploration_run_recorder", options) {
  const auto output_dir =
      declare_parameter<std::string>("output_dir", std::string{});
  const auto roots = declare_parameter<std::vector<std::string>>(
      "repository_roots", std::vector<std::string>{});
  const auto seed = declare_parameter<std::int64_t>("seed", 20260824);
  const double wall_timeout_s =
      declare_parameter<double>("wall_timeout_s", 0.0);
  if (output_dir.empty() || seed < 0 || !std::isfinite(wall_timeout_s) ||
      wall_timeout_s < 0.0) {
    throw std::invalid_argument{"invalid recorder parameters"};
  }
  std::vector<std::filesystem::path> repository_roots;
  repository_roots.reserve(roots.size());
  for (const auto& root : roots) {
    repository_roots.emplace_back(root);
  }
  recorder_ = std::make_unique<RunRecorder>(RunRecorderConfig{
      .output_dir = output_dir,
      .repository_roots = std::move(repository_roots),
      .seed = static_cast<std::uint64_t>(seed)});

  const auto status_topic = declare_parameter<std::string>(
      "exploration_status_topic", "/Car/T4/exploration/status");
  const auto odometry_topic = declare_parameter<std::string>(
      "odometry_topic", "/Car/T3/localization/odometry");
  const auto sim_elapsed_topic = declare_parameter<std::string>(
      "sim_elapsed_topic", "/Car/T4/simulation/sim_elapsed");
  const auto planner_diagnostics_topic = declare_parameter<std::string>(
      "planner_diagnostics_topic", "/Car/T4/planning/diagnostics");
  const auto exploration_diagnostics_topic = declare_parameter<std::string>(
      "exploration_diagnostics_topic", "/Car/T4/exploration/diagnostics");

  status_sub_ = create_subscription<
      lunar_pure_exploration_msgs::msg::PureExplorationStatus>(
      status_topic, rclcpp::QoS{1}.reliable().transient_local(),
      [this](const lunar_pure_exploration_msgs::msg::PureExplorationStatus::ConstSharedPtr message) {
        recorder_->ObserveStatus(*message);
      });
  odometry_sub_ = create_subscription<nav_msgs::msg::Odometry>(
      odometry_topic, rclcpp::QoS{10}.reliable(),
      [this](const nav_msgs::msg::Odometry::ConstSharedPtr message) {
        recorder_->ObserveOdometry(*message);
      });
  sim_elapsed_sub_ = create_subscription<std_msgs::msg::Float64>(
      sim_elapsed_topic, rclcpp::QoS{10}.reliable(),
      [this](const std_msgs::msg::Float64::ConstSharedPtr message) {
        recorder_->ObserveSimElapsed(message->data);
      });
  planner_diagnostics_sub_ =
      create_subscription<diagnostic_msgs::msg::DiagnosticArray>(
          planner_diagnostics_topic, rclcpp::QoS{10}.reliable(),
          [this](const diagnostic_msgs::msg::DiagnosticArray::ConstSharedPtr message) {
            recorder_->ObservePlannerDiagnostics(*message);
          });
  exploration_diagnostics_sub_ =
      create_subscription<diagnostic_msgs::msg::DiagnosticArray>(
          exploration_diagnostics_topic,
          rclcpp::QoS{1}.reliable().transient_local(),
          [this](const diagnostic_msgs::msg::DiagnosticArray::ConstSharedPtr message) {
            recorder_->ObserveExplorationDiagnostics(*message);
          });
  if (wall_timeout_s > 0.0) {
    timeout_timer_ = create_wall_timer(std::chrono::milliseconds{100},
      [this, wall_timeout_s] {
        if (!recorder_->finalized() &&
            recorder_->snapshot().wall_elapsed_s >= wall_timeout_s) {
          recorder_->Finalize(TerminalKind::kTimeout, "WALL_TIMEOUT");
        }
      });
  }
}

void RunRecorderNode::FinalizeShutdown() noexcept {
  try {
    if (recorder_ && !recorder_->finalized()) {
      recorder_->Finalize(TerminalKind::kShutdown, "SHUTDOWN");
    }
  } catch (const std::exception& error) {
    RCLCPP_ERROR(get_logger(), "failed to write shutdown summary: %s",
                 error.what());
  }
}

}  // namespace lunar::pure_exploration_sim

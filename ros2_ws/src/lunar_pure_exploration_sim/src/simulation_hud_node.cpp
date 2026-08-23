#include "lunar_pure_exploration_sim/run_recorder.hpp"

#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>

#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <lunar_planning_msgs/msg/motion_reference.hpp>
#include <lunar_pure_exploration_msgs/msg/pure_exploration_status.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

namespace lunar::pure_exploration_sim {
namespace {

using Status = lunar_pure_exploration_msgs::msg::PureExplorationStatus;

std::optional<std::string> Field(
    const diagnostic_msgs::msg::DiagnosticArray& message,
    const std::string_view key) {
  if (message.status.size() != 1U) {
    return std::nullopt;
  }
  for (const auto& entry : message.status.front().values) {
    if (entry.key == key) {
      return entry.value;
    }
  }
  return std::nullopt;
}

std::optional<double> Number(const std::optional<std::string>& text) {
  if (!text) {
    return std::nullopt;
  }
  double value{};
  const auto [end, error] = std::from_chars(
      text->data(), text->data() + text->size(), value,
      std::chars_format::general);
  if (error != std::errc{} || end != text->data() + text->size() ||
      !std::isfinite(value) || value < 0.0) {
    return std::nullopt;
  }
  return value;
}

std::optional<std::uint64_t> Count(const std::optional<std::string>& text) {
  if (!text) {
    return std::nullopt;
  }
  std::uint64_t value{};
  const auto [end, error] =
      std::from_chars(text->data(), text->data() + text->size(), value);
  if (error != std::errc{} || end != text->data() + text->size()) {
    return std::nullopt;
  }
  return value;
}

class SimulationHudNode final : public rclcpp::Node {
 public:
  SimulationHudNode() : rclcpp::Node("exploration_simulation_hud") {
    const auto status_topic = declare_parameter<std::string>(
        "exploration_status_topic", "/Car/T4/exploration/status");
    const auto odometry_topic = declare_parameter<std::string>(
        "odometry_topic", "/Car/T3/localization/odometry");
    const auto sim_elapsed_topic = declare_parameter<std::string>(
        "sim_elapsed_topic", "/Car/T4/simulation/sim_elapsed");
    const auto planner_topic = declare_parameter<std::string>(
        "planner_diagnostics_topic", "/Car/T4/planning/diagnostics");
    const auto reference_topic = declare_parameter<std::string>(
        "motion_reference_topic", "/Car/T4/execution/motion_reference");
    const auto hud_topic = declare_parameter<std::string>(
        "hud_topic", "/Car/T4/simulation/hud");
    const auto path_topic = declare_parameter<std::string>(
        "planned_path_topic", "/Car/T4/simulation/planned_path");

    const auto persistent = rclcpp::QoS{1}.reliable().transient_local();
    hud_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
        hud_topic, persistent);
    path_pub_ = create_publisher<nav_msgs::msg::Path>(path_topic, persistent);
    status_sub_ = create_subscription<Status>(
        status_topic, persistent,
        [this](const Status::ConstSharedPtr message) { status_ = *message; });
    odometry_sub_ = create_subscription<nav_msgs::msg::Odometry>(
        odometry_topic, rclcpp::QoS{10}.reliable(),
        [this](const nav_msgs::msg::Odometry::ConstSharedPtr message) {
          const double x = message->pose.pose.position.x;
          const double y = message->pose.pose.position.y;
          if (!std::isfinite(x) || !std::isfinite(y)) {
            return;
          }
          if (previous_position_) {
            distance_m_ += std::hypot(x - previous_position_->first,
                                      y - previous_position_->second);
          }
          previous_position_ = std::pair{x, y};
        });
    sim_elapsed_sub_ = create_subscription<std_msgs::msg::Float64>(
        sim_elapsed_topic, rclcpp::QoS{10}.reliable(),
        [this](const std_msgs::msg::Float64::ConstSharedPtr message) {
          if (std::isfinite(message->data) && message->data >= 0.0) {
            sim_elapsed_s_ = message->data;
          }
        });
    planner_sub_ = create_subscription<diagnostic_msgs::msg::DiagnosticArray>(
        planner_topic, rclcpp::QoS{10}.reliable(),
        [this](const diagnostic_msgs::msg::DiagnosticArray::ConstSharedPtr message) {
          ObservePlanner(*message);
        });
    reference_sub_ =
        create_subscription<lunar_planning_msgs::msg::MotionReference>(
            reference_topic, rclcpp::QoS{10}.reliable(),
            [this](const lunar_planning_msgs::msg::MotionReference::ConstSharedPtr message) {
              path_pub_->publish(PlannedPath(*message));
            });
    started_ = std::chrono::steady_clock::now();
    timer_ = create_wall_timer(std::chrono::milliseconds{250},
                               [this] { PublishHud(); });
  }

 private:
  void ObservePlanner(const diagnostic_msgs::msg::DiagnosticArray& message) {
    const auto request = Field(message, "request_id");
    const auto global_calls = Count(Field(message, "global_call_count"));
    const auto local_calls = Count(Field(message, "local_call_count"));
    const auto global_elapsed = Number(Field(message, "global_elapsed_ms"));
    const auto local_elapsed = Number(Field(message, "local_elapsed_ms"));
    if (!request || request->empty() || !global_calls || !local_calls ||
        !global_elapsed || !local_elapsed || requests_.contains(*request)) {
      return;
    }
    requests_.insert(*request);
    global_calls_ += *global_calls;
    local_calls_ += *local_calls;
    global_elapsed_ms_ += *global_elapsed;
    local_elapsed_ms_ += *local_elapsed;
  }

  void PublishHud() {
    const double wall_elapsed_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started_).count();
    RunSnapshot snapshot;
    snapshot.status = status_;
    snapshot.distance_m = distance_m_;
    snapshot.wall_elapsed_s = wall_elapsed_s;
    snapshot.sim_elapsed_s = sim_elapsed_s_;
    snapshot.global_planner_call_count = global_calls_;
    snapshot.local_planner_call_count = local_calls_;
    snapshot.global_planner_total_elapsed_ms = global_elapsed_ms_;
    snapshot.local_planner_total_elapsed_ms = local_elapsed_ms_;
    hud_pub_->publish(MakeHudMarkers(snapshot, now()));
  }

  std::chrono::steady_clock::time_point started_;
  std::optional<Status> status_;
  std::optional<std::pair<double, double>> previous_position_;
  double distance_m_{0.0};
  double sim_elapsed_s_{0.0};
  std::set<std::string> requests_;
  std::uint64_t global_calls_{0U};
  std::uint64_t local_calls_{0U};
  double global_elapsed_ms_{0.0};
  double local_elapsed_ms_{0.0};
  rclcpp::Subscription<Status>::SharedPtr status_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odometry_sub_;
  rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr sim_elapsed_sub_;
  rclcpp::Subscription<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr
      planner_sub_;
  rclcpp::Subscription<lunar_planning_msgs::msg::MotionReference>::SharedPtr
      reference_sub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr hud_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace
}  // namespace lunar::pure_exploration_sim

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(
      std::make_shared<lunar::pure_exploration_sim::SimulationHudNode>());
  rclcpp::shutdown();
  return 0;
}

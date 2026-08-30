#include "lunar_pure_planner_ros/lunar_surface_reporter.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string_view>
#include <utility>

#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <lunar_planning_msgs/msg/timed_path.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>

namespace lunar::pure_planner_ros {
namespace {

std::optional<std::string> DiagnosticValue(
    const diagnostic_msgs::msg::DiagnosticArray& diagnostics,
    const std::string_view key) {
  for (const auto& status : diagnostics.status) {
    if (status.name != "lunar_pure_planner/request") {
      continue;
    }
    for (const auto& value : status.values) {
      if (value.key == key) {
        return value.value;
      }
    }
  }
  return std::nullopt;
}

std::optional<double> ParseDouble(const std::optional<std::string>& value) {
  if (!value.has_value()) {
    return std::nullopt;
  }
  try {
    const double parsed = std::stod(*value);
    return std::isfinite(parsed) ? std::optional<double>{parsed} : std::nullopt;
  } catch (...) {
    return std::nullopt;
  }
}

class LunarSurfaceReporterNode final : public rclcpp::Node {
 public:
  LunarSurfaceReporterNode() : Node("lunar_surface_reporter") {
    odometry_sub_ = create_subscription<nav_msgs::msg::Odometry>(
        "/lunar_demo/odometry", rclcpp::QoS{10}.reliable(),
        [this](nav_msgs::msg::Odometry::ConstSharedPtr message) {
          std::scoped_lock lock{mutex_};
          latest_start_x_m_ = message->pose.pose.position.x;
          latest_start_y_m_ = message->pose.pose.position.y;
          have_odometry_ = true;
        });
    goal_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
        "/lunar_demo/rviz_goal", rclcpp::QoS{10}.reliable(),
        [this](geometry_msgs::msg::PoseStamped::ConstSharedPtr message) {
          std::scoped_lock lock{mutex_};
          if (!have_odometry_) {
            RCLCPP_WARN(get_logger(),
                        "Ignoring goal summary until odometry is available");
            active_goal_ = false;
            return;
          }
          start_x_m_ = latest_start_x_m_;
          start_y_m_ = latest_start_y_m_;
          report_state_.Begin(start_x_m_, start_y_m_,
                              message->pose.position.x,
                              message->pose.position.y);
          fallback_planning_time_ms_.reset();
          active_goal_ = true;
        });
    global_path_sub_ = create_subscription<nav_msgs::msg::Path>(
        "/lunar_demo/global_path", rclcpp::QoS{10}.reliable(),
        [this](nav_msgs::msg::Path::ConstSharedPtr message) {
          std::scoped_lock lock{mutex_};
          if (active_goal_ && !message->poses.empty()) {
            report_state_.SetGlobalPath(*message);
            EmitReadySummary();
          }
        });
    legged_global_path_sub_ = create_subscription<nav_msgs::msg::Path>(
        "/lunar_demo/legged_global_path", rclcpp::QoS{10}.reliable(),
        [this](nav_msgs::msg::Path::ConstSharedPtr message) {
          std::scoped_lock lock{mutex_};
          if (active_goal_ && !message->poses.empty()) {
            report_state_.SetGlobalPath(*message);
            EmitReadySummary();
          }
        });
    local_path_sub_ = create_subscription<nav_msgs::msg::Path>(
        "/lunar_demo/wheeled_path", rclcpp::QoS{10}.reliable(),
        [this](nav_msgs::msg::Path::ConstSharedPtr message) {
          std::scoped_lock lock{mutex_};
          if (active_goal_ && !message->poses.empty()) {
            report_state_.SetLocalPath(*message);
            EmitReadySummary();
          }
        });
    legged_local_path_sub_ = create_subscription<nav_msgs::msg::Path>(
        "/lunar_demo/legged_path", rclcpp::QoS{10}.reliable(),
        [this](nav_msgs::msg::Path::ConstSharedPtr message) {
          std::scoped_lock lock{mutex_};
          if (active_goal_ && !message->poses.empty()) {
            report_state_.SetLocalPath(*message);
            EmitReadySummary();
          }
        });
    timed_path_sub_ =
        create_subscription<lunar_planning_msgs::msg::TimedPath>(
            "/lunar_demo/wheeled_path_timing", rclcpp::QoS{10}.reliable(),
            [this](lunar_planning_msgs::msg::TimedPath::ConstSharedPtr message) {
              std::scoped_lock lock{mutex_};
              if (!active_goal_) {
                return;
              }
              fallback_planning_time_ms_ =
                  static_cast<double>(message->planning_time.sec) * 1000.0 +
                  static_cast<double>(message->planning_time.nanosec) / 1.0e6;
            });
    diagnostics_sub_ =
        create_subscription<diagnostic_msgs::msg::DiagnosticArray>(
            "/lunar_demo/diagnostics", rclcpp::QoS{10}.reliable(),
            [this](diagnostic_msgs::msg::DiagnosticArray::ConstSharedPtr message) {
              HandleDiagnostics(*message);
            });
  }

 private:
  void HandleDiagnostics(
      const diagnostic_msgs::msg::DiagnosticArray& diagnostics) {
    std::scoped_lock lock{mutex_};
    if (!active_goal_) {
      return;
    }
    const auto outcome = DiagnosticValue(diagnostics, "planning_outcome");
    const auto has_reference = DiagnosticValue(diagnostics, "has_reference");
    const auto reason = DiagnosticValue(diagnostics, "reason_code");
    const auto diagnostic_time =
        ParseDouble(DiagnosticValue(diagnostics, "total_elapsed_ms"));
    if (!outcome.has_value() || !has_reference.has_value() ||
        !reason.has_value() || (!diagnostic_time.has_value() &&
                                !fallback_planning_time_ms_.has_value())) {
      return;
    }
    const bool success = *outcome == "0" && *has_reference == "true" &&
        (*reason == "PLAN_FOUND" || *reason == "PLAN_FOUND_LATE");
    report_state_.SetResult(
        success, *reason,
        std::chrono::duration<double, std::milli>{
            diagnostic_time.has_value() ? *diagnostic_time
                                        : *fallback_planning_time_ms_});
    EmitReadySummary();
  }

  void EmitReadySummary() {
    const auto summary = report_state_.TakeReadySummary();
    if (!summary.has_value()) {
      return;
    }
    std::cout << FormatPlanningSummary(*summary) << std::endl;
    fallback_planning_time_ms_.reset();
  }

  std::mutex mutex_;
  bool have_odometry_{};
  bool active_goal_{};
  double latest_start_x_m_{};
  double latest_start_y_m_{};
  double start_x_m_{};
  double start_y_m_{};
  PlanningReportState report_state_;
  std::optional<double> fallback_planning_time_ms_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odometry_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr goal_sub_;
  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr global_path_sub_;
  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr legged_global_path_sub_;
  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr local_path_sub_;
  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr legged_local_path_sub_;
  rclcpp::Subscription<lunar_planning_msgs::msg::TimedPath>::SharedPtr
      timed_path_sub_;
  rclcpp::Subscription<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr
      diagnostics_sub_;
};

}  // namespace

void PlanningReportState::Begin(const double start_x_m,
                                const double start_y_m,
                                const double goal_x_m,
                                const double goal_y_m) noexcept {
  active_ = true;
  result_ready_ = false;
  start_x_m_ = start_x_m;
  start_y_m_ = start_y_m;
  goal_x_m_ = goal_x_m;
  goal_y_m_ = goal_y_m;
  global_path_length_m_.reset();
  local_path_length_m_.reset();
  reason_code_.clear();
}

void PlanningReportState::SetGlobalPath(
    const nav_msgs::msg::Path& path) noexcept {
  if (active_ && !path.poses.empty()) {
    global_path_length_m_ = PathLengthMetres(path);
  }
}

void PlanningReportState::SetLocalPath(
    const nav_msgs::msg::Path& path) noexcept {
  if (active_ && !path.poses.empty()) {
    local_path_length_m_ = PathLengthMetres(path);
  }
}

void PlanningReportState::SetResult(
    const bool success, std::string reason_code,
    const std::chrono::duration<double, std::milli> planning_time) {
  if (!active_) {
    return;
  }
  success_ = success;
  reason_code_ = std::move(reason_code);
  planning_time_ = planning_time;
  result_ready_ = true;
}

std::optional<PlanningSummary> PlanningReportState::TakeReadySummary() {
  if (!active_ || !result_ready_ ||
      (success_ && (!global_path_length_m_.has_value() ||
                    !local_path_length_m_.has_value()))) {
    return std::nullopt;
  }
  PlanningSummary summary;
  summary.sequence = ++sequence_;
  summary.start_x_m = start_x_m_;
  summary.start_y_m = start_y_m_;
  summary.goal_x_m = goal_x_m_;
  summary.goal_y_m = goal_y_m_;
  summary.planning_time = planning_time_;
  summary.reason_code = reason_code_;
  summary.success = success_;
  if (success_) {
    summary.global_path_length_m = global_path_length_m_;
    summary.local_path_length_m = local_path_length_m_;
  }
  result_ready_ = false;
  global_path_length_m_.reset();
  local_path_length_m_.reset();
  return summary;
}

double PathLengthMetres(const nav_msgs::msg::Path& path) noexcept {
  double length_m = 0.0;
  for (std::size_t index = 1U; index < path.poses.size(); ++index) {
    const auto& previous = path.poses[index - 1U].pose.position;
    const auto& current = path.poses[index].pose.position;
    length_m += std::hypot(current.x - previous.x, current.y - previous.y);
  }
  return length_m;
}

std::string FormatPlanningSummary(const PlanningSummary& summary) {
  std::ostringstream stream;
  stream << "[PLAN " << std::setw(3) << std::setfill('0') << summary.sequence
         << "] " << (summary.success ? "OK" : "FAIL") << std::setfill(' ')
         << std::fixed << std::setprecision(2)
         << " start=(" << summary.start_x_m << ',' << summary.start_y_m << ')'
         << " goal=(" << summary.goal_x_m << ',' << summary.goal_y_m << ')'
         << std::setprecision(1) << " time=" << summary.planning_time.count()
         << " ms";
  if (summary.success && summary.global_path_length_m.has_value() &&
      summary.local_path_length_m.has_value()) {
    stream << " path=" << *summary.global_path_length_m << " m"
           << " local=" << *summary.local_path_length_m << " m";
  } else {
    stream << " path=N/A local=N/A";
  }
  stream << " reason=" << summary.reason_code;
  return stream.str();
}

std::shared_ptr<rclcpp::Node> MakeLunarSurfaceReporterNode() {
  return std::make_shared<LunarSurfaceReporterNode>();
}

}  // namespace lunar::pure_planner_ros

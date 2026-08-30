#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include <nav_msgs/msg/path.hpp>
#include <rclcpp/node.hpp>

namespace lunar::pure_planner_ros {

struct PlanningSummary final {
  std::uint64_t sequence{};
  double start_x_m{};
  double start_y_m{};
  double goal_x_m{};
  double goal_y_m{};
  std::chrono::duration<double, std::milli> planning_time{};
  std::optional<double> global_path_length_m;
  std::optional<double> local_path_length_m;
  std::string reason_code;
  bool success{};
};

class PlanningReportState final {
 public:
  void Begin(double start_x_m, double start_y_m, double goal_x_m,
             double goal_y_m) noexcept;
  void SetGlobalPath(const nav_msgs::msg::Path& path) noexcept;
  void SetLocalPath(const nav_msgs::msg::Path& path) noexcept;
  void SetResult(bool success, std::string reason_code,
                 std::chrono::duration<double, std::milli> planning_time);
  [[nodiscard]] std::optional<PlanningSummary> TakeReadySummary();

 private:
  bool active_{};
  bool result_ready_{};
  bool success_{};
  std::uint64_t sequence_{};
  double start_x_m_{};
  double start_y_m_{};
  double goal_x_m_{};
  double goal_y_m_{};
  std::chrono::duration<double, std::milli> planning_time_{};
  std::optional<double> global_path_length_m_;
  std::optional<double> local_path_length_m_;
  std::string reason_code_;
};

[[nodiscard]] double PathLengthMetres(const nav_msgs::msg::Path& path) noexcept;
[[nodiscard]] std::string FormatPlanningSummary(const PlanningSummary& summary);
[[nodiscard]] std::shared_ptr<rclcpp::Node> MakeLunarSurfaceReporterNode();

}  // namespace lunar::pure_planner_ros

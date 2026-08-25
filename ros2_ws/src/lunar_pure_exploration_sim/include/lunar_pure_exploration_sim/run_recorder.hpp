#pragma once

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <lunar_planning_msgs/msg/motion_reference.hpp>
#include <lunar_pure_exploration_msgs/msg/pure_exploration_status.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

namespace lunar::pure_exploration_sim {

enum class TerminalKind : std::uint8_t {
  kTimeout,
  kShutdown,
};

struct RunRecorderConfig {
  std::filesystem::path output_dir;
  std::vector<std::filesystem::path> repository_roots;
  std::uint64_t seed{20260824U};
  std::chrono::steady_clock::duration terminal_diagnostic_drain{
      std::chrono::milliseconds{200}};
};

struct RunSnapshot {
  std::optional<lunar_pure_exploration_msgs::msg::PureExplorationStatus>
      status;
  double distance_m{0.0};
  double wall_elapsed_s{0.0};
  double sim_elapsed_s{0.0};
  std::uint64_t global_planner_call_count{0U};
  std::uint64_t local_planner_call_count{0U};
  double global_planner_last_elapsed_ms{0.0};
  double local_planner_last_elapsed_ms{0.0};
  double global_planner_total_elapsed_ms{0.0};
  double local_planner_total_elapsed_ms{0.0};
  std::map<std::string, double> exploration_timing;
};

class RunRecorder final {
 public:
  using SteadyNow = std::function<std::chrono::steady_clock::time_point()>;

  explicit RunRecorder(RunRecorderConfig config,
                       SteadyNow steady_now = [] {
                         return std::chrono::steady_clock::now();
                       });
  ~RunRecorder() = default;
  RunRecorder(const RunRecorder&) = delete;
  RunRecorder& operator=(const RunRecorder&) = delete;

  void ObserveStatus(
      const lunar_pure_exploration_msgs::msg::PureExplorationStatus& status);
  void ObserveOdometry(const nav_msgs::msg::Odometry& odometry);
  void ObserveSimElapsed(double seconds) noexcept;
  void ObservePlannerDiagnostics(
      const diagnostic_msgs::msg::DiagnosticArray& diagnostics);
  void ObserveExplorationDiagnostics(
      const diagnostic_msgs::msg::DiagnosticArray& diagnostics);
  void PollTerminal();
  void FlushPendingTerminal();
  void Finalize(TerminalKind kind);

  [[nodiscard]] RunSnapshot snapshot() const;
  [[nodiscard]] bool finalized() const noexcept { return finalized_; }
  [[nodiscard]] bool terminal_pending() const noexcept {
    return pending_terminal_since_.has_value();
  }

 private:
  [[nodiscard]] double WallElapsed() const;
  void WriteCoverageRow(
      const lunar_pure_exploration_msgs::msg::PureExplorationStatus& status);
  void WriteSummary(std::optional<TerminalKind> external_kind);

  RunRecorderConfig config_;
  SteadyNow steady_now_;
  std::chrono::steady_clock::time_point started_;
  std::ofstream coverage_;
  std::ofstream trajectory_;
  RunSnapshot snapshot_;
  std::optional<std::pair<double, double>> previous_position_;
  std::set<std::string> planner_request_ids_;
  std::optional<std::chrono::steady_clock::time_point> pending_terminal_since_;
  bool finalized_{false};
};

[[nodiscard]] nav_msgs::msg::Path PlannedPath(
    const lunar_planning_msgs::msg::MotionReference& reference);
[[nodiscard]] std::string ExplorationStateName(std::uint8_t state);
[[nodiscard]] visualization_msgs::msg::MarkerArray MakeHudMarkers(
    const RunSnapshot& snapshot, const rclcpp::Time& stamp);

class RunRecorderNode final : public rclcpp::Node {
 public:
  RunRecorderNode();
  explicit RunRecorderNode(const rclcpp::NodeOptions& options);
  void FinalizeShutdown() noexcept;

 private:
  std::unique_ptr<RunRecorder> recorder_;
  rclcpp::Subscription<
      lunar_pure_exploration_msgs::msg::PureExplorationStatus>::SharedPtr
      status_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odometry_sub_;
  rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr sim_elapsed_sub_;
  rclcpp::Subscription<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr
      planner_diagnostics_sub_;
  rclcpp::Subscription<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr
      exploration_diagnostics_sub_;
  rclcpp::TimerBase::SharedPtr timeout_timer_;
};

}  // namespace lunar::pure_exploration_sim

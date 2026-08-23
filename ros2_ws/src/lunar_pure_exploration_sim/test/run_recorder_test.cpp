#include "lunar_pure_exploration_sim/run_recorder.hpp"

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <diagnostic_msgs/msg/key_value.hpp>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

namespace lunar::pure_exploration_sim {
namespace {

using namespace std::chrono_literals;
using Status = lunar_pure_exploration_msgs::msg::PureExplorationStatus;

class ScopedTempDirectory final {
 public:
  explicit ScopedTempDirectory(const std::string& name) {
    const auto root = std::filesystem::temp_directory_path();
    const auto nonce =
        std::chrono::steady_clock::now().time_since_epoch().count();
    path_ = root / ("lunar-pure-exploration-" + name + "-" +
                    std::to_string(nonce) + "-" +
                    std::to_string(sequence_++));
    std::filesystem::create_directories(path_);
  }
  ~ScopedTempDirectory() {
    if (!Cleanup()) {
      ADD_FAILURE() << cleanup_error_;
    }
  }
  ScopedTempDirectory(const ScopedTempDirectory&) = delete;
  ScopedTempDirectory& operator=(const ScopedTempDirectory&) = delete;

  [[nodiscard]] const std::filesystem::path& path() const noexcept {
    return path_;
  }

  [[nodiscard]] bool Cleanup() noexcept {
    if (cleaned_) {
      return true;
    }
    const auto remove_known = [this](const std::filesystem::path& target) {
      std::error_code status_error;
      const auto status = std::filesystem::symlink_status(target, status_error);
      if (status_error == std::errc::no_such_file_or_directory ||
          status.type() == std::filesystem::file_type::not_found) {
        return true;
      }
      if (status_error) {
        cleanup_error_ = "cannot inspect " + target.string() + ": " +
                         status_error.message();
        return false;
      }
      std::error_code remove_error;
      const bool removed = std::filesystem::remove(target, remove_error);
      if (remove_error || !removed) {
        cleanup_error_ = "cannot remove " + target.string() + ": " +
                         (remove_error ? remove_error.message()
                                       : std::string{"path not removed"});
        return false;
      }
      return true;
    };
    const auto remove_result_files = [&remove_known](
                                         const std::filesystem::path& dir) {
      return remove_known(dir / "coverage.csv") &&
             remove_known(dir / "trajectory.csv") &&
             remove_known(dir / "summary.json") &&
             remove_known(dir / "summary.json.tmp");
    };
    const auto results = path_ / "results";
    const auto repository_runtime = path_ / "repository" / "runtime";
    if (!remove_result_files(results) ||
        !remove_result_files(repository_runtime) ||
        !remove_known(path_ / "repository-link") ||
        !remove_known(repository_runtime) || !remove_known(results) ||
        !remove_known(path_ / "repository") || !remove_known(path_)) {
      return false;
    }
    cleaned_ = true;
    return true;
  }

 private:
  static std::uint64_t sequence_;
  std::filesystem::path path_;
  std::string cleanup_error_;
  bool cleaned_{false};
};

std::uint64_t ScopedTempDirectory::sequence_ = 0U;

std::vector<std::string> Lines(const std::filesystem::path& path) {
  std::ifstream input{path};
  std::vector<std::string> lines;
  for (std::string line; std::getline(input, line);) {
    lines.push_back(std::move(line));
  }
  return lines;
}

Status SampleStatus(const std::uint8_t state, const double coverage,
                    std::string reason = {}) {
  Status status;
  status.task_id = "jazzy-300m-20260824";
  status.state = state;
  status.reason_code = std::move(reason);
  status.polygon_area_m2 = 84100.0;
  status.task_raster_area_m2 = 84100.0;
  status.known_free_area_m2 = coverage * 84100.0;
  status.known_occupied_area_m2 = 0.0;
  status.unknown_area_m2 = (1.0 - coverage) * 84100.0;
  status.outside_map_area_m2 = 0.0;
  status.coverage_ratio = coverage;
  status.frontier_cluster_count = 7U;
  status.candidate_count = 9U;
  status.reachable_candidate_count = 6U;
  status.failed_candidate_count = 2U;
  status.completed_goal_count = 3U;
  status.replan_count = 4U;
  status.current_plan_id = "plan-3";
  status.current_goal.position.x = 12.0;
  status.current_goal.position.y = -8.0;
  status.active_elapsed_s = 14.0;
  return status;
}

nav_msgs::msg::Odometry Pose(const double x, const double y) {
  nav_msgs::msg::Odometry message;
  message.pose.pose.position.x = x;
  message.pose.pose.position.y = y;
  return message;
}

diagnostic_msgs::msg::DiagnosticArray Diagnostics(
    std::initializer_list<std::pair<std::string, std::string>> values) {
  diagnostic_msgs::msg::DiagnosticArray message;
  message.status.resize(1U);
  for (const auto& [key, value] : values) {
    message.status.front().values.push_back(
        diagnostic_msgs::msg::KeyValue{}.set__key(key).set__value(value));
  }
  return message;
}

RunRecorderConfig Config(const std::filesystem::path& output) {
  return {.output_dir = output,
          .repository_roots = {output.parent_path() / "repository-root"},
          .seed = 20260824U,
          .terminal_diagnostic_drain = 200ms};
}

TEST(RunRecorderTest, RejectsRelativeAndRepositoryOutputDirectories) {
  ScopedTempDirectory temporary{"path-validation"};
  auto relative = Config("relative-output");
  EXPECT_THROW((void)RunRecorder{relative}, std::invalid_argument);

  const auto repository = temporary.path() / "repository";
  std::filesystem::create_directories(repository);
  auto inside = Config(repository / "runtime");
  inside.repository_roots = {repository};
  EXPECT_THROW((void)RunRecorder{inside}, std::invalid_argument);
}

TEST(RunRecorderTest, RejectsSymlinkThatResolvesInsideRepositoryRoot) {
  ScopedTempDirectory temporary{"symlink-root"};
  const auto sandbox = temporary.path();
  const auto repository = sandbox / "repository";
  const auto link = sandbox / "repository-link";
  std::filesystem::create_directories(repository);
  std::filesystem::create_directory_symlink(repository, link);
  auto config = Config(link / "runtime");
  config.repository_roots = {repository};
  EXPECT_THROW((void)RunRecorder{config}, std::invalid_argument);
}

TEST(RunRecorderTest, RecordsTwoMetresAndFlushesCsvRows) {
  ScopedTempDirectory temporary{"two-metres"};
  const auto output = temporary.path() / "results";
  auto now = std::chrono::steady_clock::time_point{};
  RunRecorder recorder(Config(output), [&now] { return now; });

  recorder.ObserveStatus(SampleStatus(Status::EXECUTING, 0.10));
  recorder.ObserveOdometry(Pose(0.0, 0.0));
  recorder.ObserveOdometry(Pose(1.0, 0.0));
  recorder.ObserveOdometry(Pose(1.0, 1.0));
  recorder.ObserveOdometry(
      Pose(std::numeric_limits<double>::quiet_NaN(), 4.0));
  recorder.ObserveSimElapsed(12.5);

  EXPECT_DOUBLE_EQ(recorder.snapshot().distance_m, 2.0);
  const auto coverage = Lines(output / "coverage.csv");
  const auto trajectory = Lines(output / "trajectory.csv");
  ASSERT_EQ(coverage.size(), 2U);
  EXPECT_EQ(coverage.front(),
            "wall_elapsed_s,sim_elapsed_s,state,reason_code,coverage_ratio,known_free_area_m2,known_occupied_area_m2,unknown_area_m2,outside_map_area_m2,frontier_cluster_count,candidate_count,reachable_candidate_count,failed_candidate_count,completed_goal_count,replan_count,current_plan_id,current_goal_x,current_goal_y");
  EXPECT_NE(coverage.back().find(",0.10000000000000001,"),
            std::string::npos);
  ASSERT_EQ(trajectory.size(), 4U);
  EXPECT_EQ(trajectory.front(), "wall_elapsed_s,sim_elapsed_s,x_m,y_m,distance_m");
  EXPECT_NE(trajectory.back().find(",2"), std::string::npos);
}

TEST(RunRecorderTest, AggregatesOnlyRealPlannerAndExplorationTimingKeys) {
  ScopedTempDirectory temporary{"timings"};
  const auto output = temporary.path() / "results";
  RunRecorder recorder(Config(output));
  recorder.ObservePlannerDiagnostics(Diagnostics({
      {"request_id", "req-1"}, {"platform_type", "WHEELED"},
      {"environment_mode", "1"}, {"planning_outcome", "0"},
      {"reason_code", "PLAN_FOUND"}, {"global_elapsed_ms", "11.5"},
      {"global_call_count", "2"}, {"local_elapsed_ms", "7.25"},
      {"local_call_count", "3"}, {"total_elapsed_ms", "18.75"}}));
  recorder.ObservePlannerDiagnostics(Diagnostics({
      {"request_id", "req-1"}, {"global_elapsed_ms", "999"},
      {"global_call_count", "999"}, {"local_elapsed_ms", "999"},
      {"local_call_count", "999"}}));
  recorder.ObservePlannerDiagnostics(Diagnostics({
      {"request_id", "req-2"}, {"global_elapsed_ms", "4.5"},
      {"global_call_count", "1"}, {"local_elapsed_ms", "2.75"},
      {"local_call_count", "2"}}));
  recorder.ObserveExplorationDiagnostics(Diagnostics({
      {"frontier_detection_call_count", "4"},
      {"frontier_detection_last_elapsed_ms", "1.5"},
      {"frontier_detection_accumulated_elapsed_ms", "5.5"},
      {"candidate_global_elapsed_ms", "11.5"},
      {"candidate_global_call_count", "2"},
      {"invented_elapsed_ms", "999"},
      {"rolling_local_elapsed_ms", "unavailable"}}));

  const auto snapshot = recorder.snapshot();
  EXPECT_EQ(snapshot.global_planner_call_count, 3U);
  EXPECT_EQ(snapshot.local_planner_call_count, 5U);
  EXPECT_DOUBLE_EQ(snapshot.global_planner_total_elapsed_ms, 16.0);
  EXPECT_DOUBLE_EQ(snapshot.local_planner_total_elapsed_ms, 10.0);
  EXPECT_DOUBLE_EQ(snapshot.global_planner_last_elapsed_ms, 4.5);
  EXPECT_DOUBLE_EQ(snapshot.local_planner_last_elapsed_ms, 2.75);
  EXPECT_EQ(snapshot.exploration_timing.at("frontier_detection_call_count"), 4.0);
  EXPECT_EQ(snapshot.exploration_timing.at("candidate_global_call_count"), 2.0);
  EXPECT_FALSE(snapshot.exploration_timing.contains("invented_elapsed_ms"));
  EXPECT_FALSE(snapshot.exploration_timing.contains("rolling_local_elapsed_ms"));
}

TEST(RunRecorderTest, AtomicallyFinalizesOnlyExactSuccessfulCompletion) {
  ScopedTempDirectory temporary{"success"};
  const auto output = temporary.path() / "results";
  auto now = std::chrono::steady_clock::time_point{};
  RunRecorder recorder(Config(output), [&now] { return now; });
  recorder.ObserveStatus(SampleStatus(Status::EXECUTING, 0.25));
  now += 3s;
  recorder.ObserveSimElapsed(60.0);
  recorder.ObserveStatus(SampleStatus(
      Status::COMPLETED, 0.625, "COMPLETED_NO_REACHABLE_FRONTIER"));

  EXPECT_TRUE(recorder.terminal_pending());
  EXPECT_FALSE(std::filesystem::exists(output / "summary.json"));
  now += 200ms;
  recorder.PollTerminal();
  EXPECT_FALSE(std::filesystem::exists(output / "summary.json.tmp"));
  ASSERT_TRUE(std::filesystem::exists(output / "summary.json"));
  const auto summary = nlohmann::json::parse(std::ifstream{output / "summary.json"});
  EXPECT_TRUE(summary.at("success").get<bool>());
  EXPECT_EQ(summary.at("terminal_state"), "COMPLETED");
  EXPECT_EQ(summary.at("reason_code"), "COMPLETED_NO_REACHABLE_FRONTIER");
  EXPECT_DOUBLE_EQ(summary.at("coverage_ratio").get<double>(), 0.625);
  EXPECT_DOUBLE_EQ(summary.at("status_coverage_ratio").get<double>(), 0.625);
  EXPECT_DOUBLE_EQ(summary.at("wall_elapsed_s").get<double>(), 3.2);
  EXPECT_DOUBLE_EQ(summary.at("sim_elapsed_s").get<double>(), 60.0);
  EXPECT_NE(Lines(output / "coverage.csv").back().find(",0.625,"),
            std::string::npos);
}

TEST(RunRecorderTest, StatusBeforeDiagnosticsDrainsLastRealTimingValues) {
  ScopedTempDirectory temporary{"status-before-diagnostics"};
  const auto output = temporary.path() / "results";
  auto now = std::chrono::steady_clock::time_point{};
  RunRecorder recorder(Config(output), [&now] { return now; });
  auto terminal = SampleStatus(
      Status::COMPLETED, 0.7, "COMPLETED_NO_REACHABLE_FRONTIER");
  terminal.header.stamp.sec = 42;
  recorder.ObserveStatus(terminal);
  recorder.ObservePlannerDiagnostics(Diagnostics({
      {"request_id", "terminal-request"}, {"global_elapsed_ms", "8.5"},
      {"global_call_count", "1"}, {"local_elapsed_ms", "4.25"},
      {"local_call_count", "2"}}));
  auto exploration = Diagnostics({
      {"frontier_detection_call_count", "9"},
      {"frontier_detection_accumulated_elapsed_ms", "12.5"}});
  exploration.header.stamp = terminal.header.stamp;
  recorder.ObserveExplorationDiagnostics(exploration);
  EXPECT_FALSE(std::filesystem::exists(output / "summary.json"));
  now += 200ms;
  recorder.PollTerminal();

  const auto summary =
      nlohmann::json::parse(std::ifstream{output / "summary.json"});
  EXPECT_EQ(summary["planner"]["global"]["call_count"], 1U);
  EXPECT_DOUBLE_EQ(summary["planner"]["local"]["total_elapsed_ms"], 4.25);
  EXPECT_DOUBLE_EQ(
      summary["exploration_timing"]["frontier_detection_accumulated_elapsed_ms"],
      12.5);
}

TEST(RunRecorderTest, DiagnosticsBeforeStatusRemainInTerminalSummary) {
  ScopedTempDirectory temporary{"diagnostics-before-status"};
  const auto output = temporary.path() / "results";
  auto now = std::chrono::steady_clock::time_point{};
  RunRecorder recorder(Config(output), [&now] { return now; });
  recorder.ObservePlannerDiagnostics(Diagnostics({
      {"request_id", "already-arrived"}, {"global_elapsed_ms", "3.0"},
      {"global_call_count", "2"}, {"local_elapsed_ms", "6.0"},
      {"local_call_count", "4"}}));
  recorder.ObserveExplorationDiagnostics(Diagnostics({
      {"snapshot_to_goal_call_count", "5"},
      {"snapshot_to_goal_accumulated_elapsed_ms", "7.5"}}));
  recorder.ObserveStatus(SampleStatus(Status::ERROR, 0.4, "PLANNER_ERROR"));
  now += 200ms;
  recorder.PollTerminal();

  const auto summary =
      nlohmann::json::parse(std::ifstream{output / "summary.json"});
  EXPECT_FALSE(summary["success"].get<bool>());
  EXPECT_EQ(summary["planner"]["local"]["call_count"], 4U);
  EXPECT_DOUBLE_EQ(
      summary["exploration_timing"]["snapshot_to_goal_accumulated_elapsed_ms"],
      7.5);
}

TEST(RunRecorderTest, RecognizesIdleCanceledAsUnsuccessfulTerminalStatus) {
  ScopedTempDirectory temporary{"actual-cancel"};
  const auto output = temporary.path() / "results";
  auto now = std::chrono::steady_clock::time_point{};
  RunRecorder recorder(Config(output), [&now] { return now; });
  recorder.ObserveStatus(SampleStatus(Status::IDLE, 0.4, "CANCELED"));
  EXPECT_TRUE(recorder.terminal_pending());
  now += 200ms;
  recorder.PollTerminal();
  const auto summary =
      nlohmann::json::parse(std::ifstream{output / "summary.json"});
  EXPECT_FALSE(summary["success"].get<bool>());
  EXPECT_EQ(summary["terminal_state"], "CANCELED");
  EXPECT_EQ(summary["reason_code"], "CANCELED");
}

TEST(RunRecorderTest, ShutdownFlushesPendingStatusWithoutReclassification) {
  ScopedTempDirectory temporary{"shutdown-pending"};
  const auto output = temporary.path() / "results";
  RunRecorder recorder(Config(output));
  recorder.ObserveStatus(SampleStatus(Status::ERROR, 0.4, "PLANNER_ERROR"));
  recorder.FlushPendingTerminal();
  const auto summary =
      nlohmann::json::parse(std::ifstream{output / "summary.json"});
  EXPECT_FALSE(summary["success"].get<bool>());
  EXPECT_EQ(summary["terminal_state"], "ERROR");
  EXPECT_EQ(summary["reason_code"], "PLANNER_ERROR");
}

TEST(RunRecorderTest, ExternalFinalizationCannotSpoofLatestStatusSuccess) {
  for (const auto kind : {TerminalKind::kTimeout, TerminalKind::kShutdown}) {
    ScopedTempDirectory temporary{"external-failure"};
    const auto output = temporary.path() / "results";
    RunRecorder recorder(Config(output));
    recorder.ObserveStatus(SampleStatus(
        Status::EXECUTING, 0.4, "COMPLETED_NO_REACHABLE_FRONTIER"));
    recorder.Finalize(kind);
    const auto summary =
        nlohmann::json::parse(std::ifstream{output / "summary.json"});
    EXPECT_FALSE(summary["success"].get<bool>());
  }

  ScopedTempDirectory temporary{"wrong-completed-reason"};
  const auto output = temporary.path() / "results";
  RunRecorder recorder(Config(output));
  recorder.ObserveStatus(
      SampleStatus(Status::COMPLETED, 0.4, "COVERAGE_REACHED"));
  recorder.Finalize(TerminalKind::kTimeout);
  const auto summary =
      nlohmann::json::parse(std::ifstream{output / "summary.json"});
  EXPECT_FALSE(summary["success"].get<bool>());
  EXPECT_EQ(summary["terminal_state"], "COMPLETED");
  EXPECT_EQ(summary["reason_code"], "COVERAGE_REACHED");
}

TEST(RunRecorderTest, SummaryJsonEscapesStatusStringsWithoutDataLoss) {
  ScopedTempDirectory temporary{"json-escaping"};
  const auto output = temporary.path() / "results";
  auto now = std::chrono::steady_clock::time_point{};
  RunRecorder recorder(Config(output), [&now] { return now; });
  auto status = SampleStatus(Status::ERROR, 0.4, "bad \"quote\"\nslash\\end");
  status.task_id = "task\n\"quoted\"";
  status.current_plan_id = "plan\\path";
  recorder.ObserveStatus(status);
  now += 200ms;
  recorder.PollTerminal();
  const auto summary =
      nlohmann::json::parse(std::ifstream{output / "summary.json"});
  EXPECT_EQ(summary["reason_code"], status.reason_code);
  EXPECT_EQ(summary["task_id"], status.task_id);
  EXPECT_EQ(summary["current_plan_id"], status.current_plan_id);
}

TEST(RunRecorderTest, ConvertsMotionReferencePreviewWithoutUsingTrajectory) {
  lunar_planning_msgs::msg::MotionReference reference;
  reference.path_preview.header.frame_id = "map";
  reference.path_preview.poses.resize(2U);
  reference.path_preview.poses[0].pose.position.x = 1.0;
  reference.path_preview.poses[1].pose.position.x = 2.0;
  reference.trajectory.points.resize(1U);
  reference.trajectory.points.front().transforms.resize(1U);
  reference.trajectory.points.front().transforms.front().translation.x = 99.0;

  const auto path = PlannedPath(reference);
  ASSERT_EQ(path.poses.size(), 2U);
  EXPECT_EQ(path.header.frame_id, "map");
  EXPECT_DOUBLE_EQ(path.poses.back().pose.position.x, 2.0);
}

TEST(RunRecorderTest, HudShowsStatusAreasPlannerTimingAndElapsedMetrics) {
  RunSnapshot snapshot;
  snapshot.status = SampleStatus(
      Status::COMPLETED, 0.625, "COMPLETED_NO_REACHABLE_FRONTIER");
  snapshot.distance_m = 2.0;
  snapshot.wall_elapsed_s = 3.0;
  snapshot.sim_elapsed_s = 60.0;
  snapshot.global_planner_call_count = 2U;
  snapshot.local_planner_call_count = 3U;
  snapshot.global_planner_total_elapsed_ms = 11.5;
  snapshot.local_planner_total_elapsed_ms = 7.25;

  const auto markers = MakeHudMarkers(snapshot, rclcpp::Time{123, 0});
  ASSERT_EQ(markers.markers.size(), 1U);
  const auto& marker = markers.markers.front();
  EXPECT_EQ(marker.header.frame_id, "map");
  EXPECT_EQ(marker.ns, "exploration_hud");
  EXPECT_EQ(marker.action, visualization_msgs::msg::Marker::ADD);
  EXPECT_EQ(marker.type, visualization_msgs::msg::Marker::TEXT_VIEW_FACING);
  EXPECT_NE(marker.text.find("COMPLETED_NO_REACHABLE_FRONTIER"),
            std::string::npos);
  EXPECT_NE(marker.text.find("Coverage: 0.625"), std::string::npos);
  EXPECT_NE(marker.text.find("Task/polygon area m2: 84100.000/84100.000"),
            std::string::npos);
  EXPECT_NE(marker.text.find("Planner global: 2 calls / 11.500 ms"),
            std::string::npos);
  EXPECT_NE(marker.text.find("Distance: 2.000 m  wall/sim: 3.000/60.000 s"),
            std::string::npos);
}

}  // namespace
}  // namespace lunar::pure_exploration_sim

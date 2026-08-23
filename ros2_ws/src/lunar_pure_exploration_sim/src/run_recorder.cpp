#include "lunar_pure_exploration_sim/run_recorder.hpp"

#include <array>
#include <charconv>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <system_error>
#include <utility>

#include <nlohmann/json.hpp>

namespace lunar::pure_exploration_sim {
namespace {

using Status = lunar_pure_exploration_msgs::msg::PureExplorationStatus;

std::filesystem::path Normal(const std::filesystem::path& path) {
  const auto absolute = std::filesystem::absolute(path).lexically_normal();
  std::error_code error;
  const auto canonical = std::filesystem::weakly_canonical(absolute, error);
  return error ? absolute : canonical;
}

bool IsWithin(const std::filesystem::path& child,
              const std::filesystem::path& parent) {
  const auto normalized_child = Normal(child);
  const auto normalized_parent = Normal(parent);
  auto child_it = normalized_child.begin();
  for (auto parent_it = normalized_parent.begin();
       parent_it != normalized_parent.end(); ++parent_it, ++child_it) {
    if (child_it == normalized_child.end() || *child_it != *parent_it) {
      return false;
    }
  }
  return true;
}

std::optional<std::string> Field(
    const diagnostic_msgs::msg::DiagnosticArray& message,
    const std::string_view key) {
  if (message.status.size() != 1U) {
    return std::nullopt;
  }
  for (const auto& value : message.status.front().values) {
    if (value.key == key) {
      return value.value;
    }
  }
  return std::nullopt;
}

std::optional<double> NonNegativeDouble(const std::string_view text) {
  double parsed{};
  const auto [end, error] = std::from_chars(
      text.data(), text.data() + text.size(), parsed,
      std::chars_format::general);
  if (error != std::errc{} || end != text.data() + text.size() ||
      !std::isfinite(parsed) || parsed < 0.0) {
    return std::nullopt;
  }
  return parsed;
}

std::optional<std::uint64_t> Unsigned(const std::string_view text) {
  std::uint64_t parsed{};
  const auto [end, error] =
      std::from_chars(text.data(), text.data() + text.size(), parsed);
  if (error != std::errc{} || end != text.data() + text.size()) {
    return std::nullopt;
  }
  return parsed;
}

std::string Csv(std::string value) {
  if (value.find_first_of(",\"\n") == std::string::npos) {
    return value;
  }
  std::string escaped{"\""};
  for (const char character : value) {
    if (character == '"') {
      escaped.push_back('"');
    }
    escaped.push_back(character);
  }
  escaped.push_back('"');
  return escaped;
}

std::string StateName(const std::uint8_t state) {
  switch (state) {
    case Status::IDLE: return "IDLE";
    case Status::WAITING_FOR_INPUT: return "WAITING_FOR_INPUT";
    case Status::SELECTING_FRONTIER: return "SELECTING_FRONTIER";
    case Status::PLANNING: return "PLANNING";
    case Status::EXECUTING: return "EXECUTING";
    case Status::REPLANNING: return "REPLANNING";
    case Status::PAUSED: return "PAUSED";
    case Status::COMPLETED: return "COMPLETED";
    case Status::ERROR: return "ERROR";
    default: return "UNKNOWN";
  }
}

bool IsExplorationTimingKey(const std::string_view key) {
  constexpr std::array<std::string_view, 5U> local_prefixes{
      "frontier_detection_", "information_gain_", "coarse_rank_",
      "final_rank_", "snapshot_to_goal_"};
  constexpr std::array<std::string_view, 3U> planner_prefixes{
      "candidate_", "rolling_", "stuck_"};
  const auto local_suffix = [](const std::string_view suffix) {
    return suffix == "call_count" || suffix == "last_elapsed_ms" ||
           suffix == "accumulated_elapsed_ms";
  };
  const auto planner_suffix = [](const std::string_view suffix) {
    return suffix == "call_count" || suffix == "record_count" ||
           suffix == "global_elapsed_ms" || suffix == "global_call_count" ||
           suffix == "local_elapsed_ms" || suffix == "local_call_count" ||
           suffix == "total_elapsed_ms";
  };
  for (const auto prefix : local_prefixes) {
    if (key.starts_with(prefix) && local_suffix(key.substr(prefix.size()))) {
      return true;
    }
  }
  for (const auto prefix : planner_prefixes) {
    if (key.starts_with(prefix) && planner_suffix(key.substr(prefix.size()))) {
      return true;
    }
  }
  return false;
}

bool IsTerminalStatus(const Status& status) {
  return status.state == Status::COMPLETED || status.state == Status::ERROR ||
         (status.state == Status::IDLE && status.reason_code == "CANCELED");
}

}  // namespace

RunRecorder::RunRecorder(RunRecorderConfig config, SteadyNow steady_now)
    : config_(std::move(config)), steady_now_(std::move(steady_now)) {
  if (!config_.output_dir.is_absolute()) {
    throw std::invalid_argument{"output_dir must be absolute"};
  }
  if (config_.repository_roots.empty()) {
    throw std::invalid_argument{"repository_roots must not be empty"};
  }
  if (config_.terminal_diagnostic_drain <=
      std::chrono::steady_clock::duration::zero()) {
    throw std::invalid_argument{
        "terminal diagnostic drain must be positive"};
  }
  for (const auto& root : config_.repository_roots) {
    if (!root.is_absolute()) {
      throw std::invalid_argument{"repository roots must be absolute"};
    }
    if (IsWithin(config_.output_dir, root)) {
      throw std::invalid_argument{"output_dir must be outside repositories"};
    }
  }
  std::filesystem::create_directories(config_.output_dir);
  coverage_.open(config_.output_dir / "coverage.csv", std::ios::trunc);
  trajectory_.open(config_.output_dir / "trajectory.csv", std::ios::trunc);
  if (!coverage_ || !trajectory_) {
    throw std::runtime_error{"failed to open run CSV files"};
  }
  coverage_ << "wall_elapsed_s,sim_elapsed_s,state,reason_code,coverage_ratio,known_free_area_m2,known_occupied_area_m2,unknown_area_m2,outside_map_area_m2,frontier_cluster_count,candidate_count,reachable_candidate_count,failed_candidate_count,completed_goal_count,replan_count,current_plan_id,current_goal_x,current_goal_y\n"
            << std::flush;
  trajectory_ << "wall_elapsed_s,sim_elapsed_s,x_m,y_m,distance_m\n"
              << std::flush;
  started_ = steady_now_();
}

double RunRecorder::WallElapsed() const {
  return std::chrono::duration<double>(steady_now_() - started_).count();
}

void RunRecorder::ObserveStatus(const Status& status) {
  if (finalized_ || pending_terminal_since_) {
    return;
  }
  snapshot_.status = status;
  snapshot_.wall_elapsed_s = WallElapsed();
  WriteCoverageRow(status);
  if (IsTerminalStatus(status)) {
    pending_terminal_since_ = steady_now_();
  }
}

void RunRecorder::WriteCoverageRow(const Status& status) {
  coverage_ << std::setprecision(17) << snapshot_.wall_elapsed_s << ','
            << snapshot_.sim_elapsed_s << ',' << StateName(status.state) << ','
            << Csv(status.reason_code) << ',' << status.coverage_ratio << ','
            << status.known_free_area_m2 << ',' << status.known_occupied_area_m2
            << ',' << status.unknown_area_m2 << ',' << status.outside_map_area_m2
            << ',' << status.frontier_cluster_count << ',' << status.candidate_count
            << ',' << status.reachable_candidate_count << ','
            << status.failed_candidate_count << ',' << status.completed_goal_count
            << ',' << status.replan_count << ',' << Csv(status.current_plan_id)
            << ',' << status.current_goal.position.x << ','
            << status.current_goal.position.y << '\n' << std::flush;
}

void RunRecorder::ObserveOdometry(const nav_msgs::msg::Odometry& odometry) {
  if (finalized_) {
    return;
  }
  const double x = odometry.pose.pose.position.x;
  const double y = odometry.pose.pose.position.y;
  if (!std::isfinite(x) || !std::isfinite(y)) {
    return;
  }
  if (previous_position_) {
    snapshot_.distance_m +=
        std::hypot(x - previous_position_->first, y - previous_position_->second);
  }
  previous_position_ = std::pair{x, y};
  snapshot_.wall_elapsed_s = WallElapsed();
  trajectory_ << std::setprecision(17) << snapshot_.wall_elapsed_s << ','
              << snapshot_.sim_elapsed_s << ',' << x << ',' << y << ','
              << snapshot_.distance_m << '\n' << std::flush;
}

void RunRecorder::ObserveSimElapsed(const double seconds) noexcept {
  if (std::isfinite(seconds) && seconds >= 0.0 && !finalized_) {
    snapshot_.sim_elapsed_s = seconds;
  }
}

void RunRecorder::ObservePlannerDiagnostics(
    const diagnostic_msgs::msg::DiagnosticArray& diagnostics) {
  if (finalized_) {
    return;
  }
  const auto request_id = Field(diagnostics, "request_id");
  const auto global_elapsed = Field(diagnostics, "global_elapsed_ms");
  const auto global_calls = Field(diagnostics, "global_call_count");
  const auto local_elapsed = Field(diagnostics, "local_elapsed_ms");
  const auto local_calls = Field(diagnostics, "local_call_count");
  if (!request_id || request_id->empty() || !global_elapsed || !global_calls ||
      !local_elapsed || !local_calls || planner_request_ids_.contains(*request_id)) {
    return;
  }
  const auto parsed_global_elapsed = NonNegativeDouble(*global_elapsed);
  const auto parsed_global_calls = Unsigned(*global_calls);
  const auto parsed_local_elapsed = NonNegativeDouble(*local_elapsed);
  const auto parsed_local_calls = Unsigned(*local_calls);
  if (!parsed_global_elapsed || !parsed_global_calls || !parsed_local_elapsed ||
      !parsed_local_calls) {
    return;
  }
  planner_request_ids_.insert(*request_id);
  snapshot_.global_planner_call_count += *parsed_global_calls;
  snapshot_.local_planner_call_count += *parsed_local_calls;
  snapshot_.global_planner_last_elapsed_ms = *parsed_global_elapsed;
  snapshot_.local_planner_last_elapsed_ms = *parsed_local_elapsed;
  snapshot_.global_planner_total_elapsed_ms += *parsed_global_elapsed;
  snapshot_.local_planner_total_elapsed_ms += *parsed_local_elapsed;
}

void RunRecorder::ObserveExplorationDiagnostics(
    const diagnostic_msgs::msg::DiagnosticArray& diagnostics) {
  if (finalized_ || diagnostics.status.size() != 1U) {
    return;
  }
  for (const auto& entry : diagnostics.status.front().values) {
    if (IsExplorationTimingKey(entry.key)) {
      if (const auto parsed = NonNegativeDouble(entry.value)) {
        snapshot_.exploration_timing[entry.key] = *parsed;
      }
    }
  }
}

void RunRecorder::PollTerminal() {
  if (!pending_terminal_since_ || finalized_) {
    return;
  }
  if (steady_now_() - *pending_terminal_since_ >=
      config_.terminal_diagnostic_drain) {
    FlushPendingTerminal();
  }
}

void RunRecorder::FlushPendingTerminal() {
  if (!pending_terminal_since_ || finalized_) {
    return;
  }
  snapshot_.wall_elapsed_s = WallElapsed();
  WriteSummary(std::nullopt);
  finalized_ = true;
  pending_terminal_since_.reset();
}

void RunRecorder::Finalize(const TerminalKind kind) {
  if (finalized_) {
    return;
  }
  if (pending_terminal_since_) {
    FlushPendingTerminal();
    return;
  }
  snapshot_.wall_elapsed_s = WallElapsed();
  WriteSummary(kind);
  finalized_ = true;
}

void RunRecorder::WriteSummary(
    const std::optional<TerminalKind> external_kind) {
  const Status status = snapshot_.status.value_or(Status{});
  const bool success = status.state == Status::COMPLETED &&
                       status.reason_code ==
                           "COMPLETED_NO_REACHABLE_FRONTIER";
  std::string terminal_state;
  std::string reason_code;
  if (IsTerminalStatus(status)) {
    terminal_state = status.state == Status::IDLE ? "CANCELED"
                                                   : StateName(status.state);
    reason_code = status.reason_code;
  } else if (external_kind == TerminalKind::kTimeout) {
    terminal_state = "TIMEOUT";
    reason_code = "WALL_TIMEOUT";
  } else {
    terminal_state = "SHUTDOWN";
    reason_code = "SHUTDOWN";
  }
  nlohmann::json summary{
      {"seed", config_.seed},
      {"map", {{"length_x_m", 300.0}, {"length_y_m", 300.0},
               {"global_resolution_m", 1.0}, {"local_resolution_m", 0.2}}},
      {"sensor", {{"range_m", 10.0}, {"field_of_view_deg", 90.0}}},
      {"success", success},
      {"terminal_state", terminal_state},
      {"reason_code", reason_code},
      {"task_id", status.task_id},
      {"coverage_ratio", status.coverage_ratio},
      {"status_coverage_ratio", status.coverage_ratio},
      {"polygon_area_m2", status.polygon_area_m2},
      {"task_raster_area_m2", status.task_raster_area_m2},
      {"known_free_area_m2", status.known_free_area_m2},
      {"known_occupied_area_m2", status.known_occupied_area_m2},
      {"unknown_area_m2", status.unknown_area_m2},
      {"outside_map_area_m2", status.outside_map_area_m2},
      {"frontier_cluster_count", status.frontier_cluster_count},
      {"candidate_count", status.candidate_count},
      {"reachable_candidate_count", status.reachable_candidate_count},
      {"failed_candidate_count", status.failed_candidate_count},
      {"completed_goal_count", status.completed_goal_count},
      {"replan_count", status.replan_count},
      {"current_plan_id", status.current_plan_id},
      {"current_goal", {{"x", status.current_goal.position.x},
                         {"y", status.current_goal.position.y}}},
      {"active_elapsed_s", status.active_elapsed_s},
      {"distance_m", snapshot_.distance_m},
      {"wall_elapsed_s", snapshot_.wall_elapsed_s},
      {"sim_elapsed_s", snapshot_.sim_elapsed_s},
      {"planner", {
          {"global", {{"call_count", snapshot_.global_planner_call_count},
                       {"last_elapsed_ms", snapshot_.global_planner_last_elapsed_ms},
                       {"total_elapsed_ms", snapshot_.global_planner_total_elapsed_ms}}},
          {"local", {{"call_count", snapshot_.local_planner_call_count},
                      {"last_elapsed_ms", snapshot_.local_planner_last_elapsed_ms},
                      {"total_elapsed_ms", snapshot_.local_planner_total_elapsed_ms}}}}},
      {"exploration_timing", snapshot_.exploration_timing}};
  const auto temporary = config_.output_dir / "summary.json.tmp";
  const auto final = config_.output_dir / "summary.json";
  {
    std::ofstream output{temporary, std::ios::trunc};
    if (!output) {
      throw std::runtime_error{"failed to open terminal summary"};
    }
    output << std::setw(2) << summary << '\n' << std::flush;
    if (!output) {
      throw std::runtime_error{"failed to flush terminal summary"};
    }
  }
  std::filesystem::rename(temporary, final);
}

RunSnapshot RunRecorder::snapshot() const {
  RunSnapshot copy = snapshot_;
  if (!finalized_) {
    copy.wall_elapsed_s = WallElapsed();
  }
  return copy;
}

nav_msgs::msg::Path PlannedPath(
    const lunar_planning_msgs::msg::MotionReference& reference) {
  return reference.path_preview;
}

std::string ExplorationStateName(const std::uint8_t state) {
  return StateName(state);
}

visualization_msgs::msg::MarkerArray MakeHudMarkers(
    const RunSnapshot& snapshot, const rclcpp::Time& stamp) {
  visualization_msgs::msg::Marker marker;
  marker.header.frame_id = "base_link";
  marker.header.stamp = stamp;
  marker.ns = "exploration_hud";
  marker.id = 0;
  marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
  marker.action = visualization_msgs::msg::Marker::ADD;
  marker.pose.position.x = 2.0;
  marker.pose.position.z = 2.5;
  marker.pose.orientation.w = 1.0;
  marker.scale.z = 0.6;
  marker.color.r = 1.0F;
  marker.color.g = 1.0F;
  marker.color.b = 1.0F;
  marker.color.a = 1.0F;
  std::ostringstream text;
  text << std::fixed << std::setprecision(3);
  if (snapshot.status) {
    const auto& status = *snapshot.status;
    text << "State: " << StateName(status.state)
         << "  Reason: " << status.reason_code << '\n'
         << "Coverage: " << status.coverage_ratio << '\n'
         << "Task/polygon area m2: " << status.task_raster_area_m2 << '/'
         << status.polygon_area_m2 << '\n'
         << "Free/occupied/unknown/outside m2: " << status.known_free_area_m2
         << '/' << status.known_occupied_area_m2 << '/' << status.unknown_area_m2
         << '/' << status.outside_map_area_m2 << '\n'
         << "Frontiers/candidates/reachable/failed: "
         << status.frontier_cluster_count << '/' << status.candidate_count << '/'
         << status.reachable_candidate_count << '/'
         << status.failed_candidate_count << '\n'
         << "Completed goals/replans: " << status.completed_goal_count << '/'
         << status.replan_count << '\n';
  } else {
    text << "State: waiting for exploration status\n";
  }
  text << "Planner global: " << snapshot.global_planner_call_count
       << " calls / " << snapshot.global_planner_total_elapsed_ms
       << " ms  local: " << snapshot.local_planner_call_count << " calls / "
       << snapshot.local_planner_total_elapsed_ms << " ms\n"
       << "Distance: " << snapshot.distance_m << " m  wall/sim: "
       << snapshot.wall_elapsed_s << '/' << snapshot.sim_elapsed_s << " s";
  marker.text = text.str();
  visualization_msgs::msg::MarkerArray markers;
  markers.markers.push_back(std::move(marker));
  return markers;
}

}  // namespace lunar::pure_exploration_sim

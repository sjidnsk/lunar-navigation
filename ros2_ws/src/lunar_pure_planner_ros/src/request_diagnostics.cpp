#include "lunar_pure_planner_ros/request_diagnostics.hpp"

#include <chrono>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <string>
#include <utility>

#include <lunar_planning_msgs/action/plan_motion.hpp>

namespace lunar::pure_planner_ros {
namespace {

[[nodiscard]] std::string Decimal(const double value) {
  std::ostringstream stream;
  stream << std::setprecision(15) << value;
  return stream.str();
}

[[nodiscard]] std::string Milliseconds(const std::chrono::nanoseconds value) {
  return Decimal(std::chrono::duration<double, std::milli>(value).count());
}

[[nodiscard]] std::string PlatformType(const lunar::pure_planning::PlatformType value) {
  switch (value) {
    case lunar::pure_planning::PlatformType::kWheeled:
      return "WHEELED";
    case lunar::pure_planning::PlatformType::kLegged:
      return "LEGGED";
    case lunar::pure_planning::PlatformType::kHopper:
      return "HOPPER";
  }
  return "UNKNOWN";
}

struct StatusFields final {
  std::uint8_t outcome;
  std::uint8_t level;
  const char* reason;
};

[[nodiscard]] StatusFields StatusFor(const lunar::pure_planning::PlanningStatus status) {
  using PlanningStatus = lunar::pure_planning::PlanningStatus;
  using DiagnosticStatus = diagnostic_msgs::msg::DiagnosticStatus;
  using Action = lunar_planning_msgs::action::PlanMotion;
  switch (status) {
    case PlanningStatus::kSuccess:
      return {Action::Result::NEW_REFERENCE_AVAILABLE, DiagnosticStatus::OK, "PLAN_FOUND"};
    case PlanningStatus::kInvalidInput:
      return {Action::Result::INVALID_REQUEST, DiagnosticStatus::ERROR, "INVALID_INPUT"};
    case PlanningStatus::kGoalOutsideLocalMap:
      return {Action::Result::GOAL_INFEASIBLE, DiagnosticStatus::WARN, "GOAL_OUTSIDE_LOCAL_MAP"};
    case PlanningStatus::kNoPath:
      return {Action::Result::GOAL_INFEASIBLE, DiagnosticStatus::WARN, "NO_PATH"};
    case PlanningStatus::kTimedOut:
      return {Action::Result::RESOURCE_EXHAUSTED, DiagnosticStatus::WARN, "TIMEOUT"};
    case PlanningStatus::kCanceled:
      return {Action::Result::CANCELED, DiagnosticStatus::WARN, "REQUEST_CANCELED"};
    case PlanningStatus::kPlannerError:
      return {Action::Result::NUMERICAL_FAILURE, DiagnosticStatus::ERROR, "PLANNER_ERROR"};
  }
  return {Action::Result::NUMERICAL_FAILURE, DiagnosticStatus::ERROR, "PLANNER_ERROR"};
}

}  // namespace

diagnostic_msgs::msg::DiagnosticArray MakeRequestDiagnostics(
    const std::string_view request_id,
    const lunar::pure_planning::PlatformType platform_type,
    const lunar::pure_planning::EnvironmentMode environment_mode,
    const lunar::pure_planning::PlanningResult& result) {
  diagnostic_msgs::msg::DiagnosticStatus status;
  status.name = "lunar_pure_planner/request";
  status.hardware_id = "lunar_pure_planner";
  const StatusFields fields = StatusFor(result.status);
  status.level = fields.level;
  const auto add_value = [&status](std::string key, std::string value) {
    diagnostic_msgs::msg::KeyValue entry;
    entry.key = std::move(key);
    entry.value = std::move(value);
    status.values.push_back(std::move(entry));
  };
  add_value("request_id", std::string{request_id});
  add_value("platform_type", PlatformType(platform_type));
  add_value("environment_mode", std::to_string(static_cast<std::uint8_t>(environment_mode)));
  add_value("planning_outcome", std::to_string(fields.outcome));
  add_value("reason_code",
            result.reason_code.empty() ? fields.reason : result.reason_code);
  add_value("expanded_states", std::to_string(result.expanded_states));
  add_value("has_best_cost", result.best_cost.has_value() ? "true" : "false");
  add_value("best_cost",
            result.best_cost.has_value() ? Decimal(*result.best_cost) : "0");
  add_value("has_reference", result.reference.has_value() ? "true" : "false");
  add_value("latency_class", std::string{
      lunar::pure_planning::RequestLatencyClassName(
          lunar::pure_planning::ClassifyRequestLatency(
              result.timing.total_elapsed))});
  add_value("snapshot_projection_elapsed_ms",
            Milliseconds(result.timing.snapshot_projection_elapsed));
  add_value("global_elapsed_ms", Milliseconds(result.timing.global_elapsed));
  add_value("global_call_count", std::to_string(result.timing.global_call_count));
  add_value("local_goal_elapsed_ms",
            Milliseconds(result.timing.local_goal_elapsed));
  add_value("local_search_elapsed_ms",
            Milliseconds(result.timing.local_search_elapsed));
  add_value("local_elapsed_ms", Milliseconds(result.timing.local_elapsed));
  add_value("local_call_count", std::to_string(result.timing.local_call_count));
  add_value("certification_elapsed_ms",
            Milliseconds(result.timing.certification_elapsed));
  add_value("output_elapsed_ms", Milliseconds(result.timing.output_elapsed));
  add_value("total_elapsed_ms", Milliseconds(result.timing.total_elapsed));
  if (result.grid_v1.active) {
    const auto& grid_v1 = result.grid_v1;
    add_value("grid_v1_active", "true");
    add_value("global_input_sequence", std::to_string(grid_v1.global_input_sequence));
    add_value("local_input_sequence", std::to_string(grid_v1.local_input_sequence));
    add_value("odometry_input_sequence", std::to_string(grid_v1.odometry_input_sequence));
    add_value("traversability_revision", std::to_string(grid_v1.traversability_revision));
    add_value("publish_check_revision", std::to_string(grid_v1.publish_check_revision));
    add_value("profile_hash", std::to_string(grid_v1.profile_hash));
    add_value("canonical_resolution_m", Decimal(grid_v1.canonical_resolution_m));
    add_value("map_origin_x_m", Decimal(grid_v1.map_origin_m.x));
    add_value("map_origin_y_m", Decimal(grid_v1.map_origin_m.y));
    add_value("map_origin_z_m", Decimal(grid_v1.map_origin_m.z));
    add_value("allocated_tiles", std::to_string(grid_v1.allocated_tiles));
    add_value("estimated_map_bytes", std::to_string(grid_v1.estimated_map_bytes));
    add_value("updated_cells", std::to_string(grid_v1.updated_cells));
    add_value("dirty_tiles", std::to_string(grid_v1.dirty_tiles));
    add_value("halo_recomputed_cells", std::to_string(grid_v1.halo_recomputed_cells));
    add_value("free_cells", std::to_string(grid_v1.free_cells));
    add_value("blocked_cells", std::to_string(grid_v1.blocked_cells));
    add_value("unknown_cells", std::to_string(grid_v1.unknown_cells));
    add_value("prior_conflicts", std::to_string(grid_v1.prior_conflicts));
    add_value("global_route_reused", grid_v1.global_route_reused ? "true" : "false");
    add_value("global_expanded_states", std::to_string(grid_v1.global_expanded_states));
    add_value("global_open_peak", std::to_string(grid_v1.global_open_peak));
    add_value("local_expanded_states", std::to_string(grid_v1.local_expanded_states));
    add_value("local_open_peak", std::to_string(grid_v1.local_open_peak));
    add_value("local_candidate_count", std::to_string(grid_v1.local_candidate_count));
    add_value("local_attempt_count", std::to_string(grid_v1.local_attempt_count));
    add_value("selected_candidate_index", std::to_string(grid_v1.selected_candidate_index));
    add_value("raw_path_points", std::to_string(grid_v1.raw_path_points));
    add_value("shortcut_path_points", std::to_string(grid_v1.shortcut_path_points));
    add_value("resampled_path_points", std::to_string(grid_v1.resampled_path_points));
    add_value("final_trajectory_points", std::to_string(grid_v1.final_trajectory_points));
    add_value("direction", grid_v1.direction);
    add_value("forward_cost", Decimal(grid_v1.forward_cost));
    add_value("reverse_cost", Decimal(grid_v1.reverse_cost));
    add_value("final_supercover_cells", std::to_string(grid_v1.final_supercover_cells));
    add_value("postprocess_mode", grid_v1.postprocess_mode);
    add_value("map_fusion_elapsed_ms", Milliseconds(grid_v1.map_fusion_elapsed));
    add_value("traversability_elapsed_ms", Milliseconds(grid_v1.traversability_elapsed));
    add_value("postprocess_elapsed_ms", Milliseconds(grid_v1.postprocess_elapsed));
  }
  diagnostic_msgs::msg::DiagnosticArray diagnostics;
  diagnostics.status.push_back(std::move(status));
  return diagnostics;
}

std::string FindDiagnosticValue(const diagnostic_msgs::msg::DiagnosticArray& diagnostics,
                                const std::string& key) {
  if (diagnostics.status.size() != 1U) {
    return {};
  }
  for (const auto& value : diagnostics.status.front().values) {
    if (value.key == key) {
      return value.value;
    }
  }
  return {};
}

}  // namespace lunar::pure_planner_ros

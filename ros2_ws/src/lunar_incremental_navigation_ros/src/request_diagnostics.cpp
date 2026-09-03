#include "lunar_incremental_navigation_ros/request_diagnostics.hpp"

#include <iomanip>
#include <sstream>
#include <string>
#include <utility>

namespace lunar::incremental_navigation_ros {
namespace {

[[nodiscard]] std::string Decimal(const double value) {
  std::ostringstream stream;
  stream << std::setprecision(15) << value;
  return stream.str();
}

void Add(diagnostic_msgs::msg::DiagnosticStatus& status, std::string key,
         std::string value) {
  diagnostic_msgs::msg::KeyValue field;
  field.key = std::move(key);
  field.value = std::move(value);
  status.values.push_back(std::move(field));
}

[[nodiscard]] diagnostic_msgs::msg::DiagnosticArray Wrap(
    diagnostic_msgs::msg::DiagnosticStatus status) {
  status.hardware_id = "lunar_incremental_navigation";
  diagnostic_msgs::msg::DiagnosticArray message;
  message.header.stamp.sec = 0;
  message.header.stamp.nanosec = 0U;
  message.status.push_back(std::move(status));
  return message;
}

}  // namespace

PlanningCycleLatencyClass ClassifyPlanningCycleLatency(
    const std::chrono::nanoseconds elapsed,
    const std::chrono::nanoseconds sla,
    const std::chrono::nanoseconds hard_timeout) noexcept {
  if (elapsed >= hard_timeout) {
    return PlanningCycleLatencyClass::kHardTimeout;
  }
  if (elapsed >= sla) {
    return PlanningCycleLatencyClass::kSlaMissed;
  }
  return PlanningCycleLatencyClass::kSlaMet;
}

std::string_view PlanningCycleLatencyClassName(
    const PlanningCycleLatencyClass latency_class) noexcept {
  switch (latency_class) {
    case PlanningCycleLatencyClass::kSlaMet:
      return "SLA_MET";
    case PlanningCycleLatencyClass::kSlaMissed:
      return "SLA_MISSED";
    case PlanningCycleLatencyClass::kHardTimeout:
      return "HARD_TIMEOUT";
  }
  return "HARD_TIMEOUT";
}

diagnostic_msgs::msg::DiagnosticArray MakeMapDiagnostics(
    const MapDiagnosticsRecord& record) {
  diagnostic_msgs::msg::DiagnosticStatus status;
  status.name = "lunar_incremental_navigation/map";
  status.level = record.incompatible_qos_count == 0U
                     ? diagnostic_msgs::msg::DiagnosticStatus::OK
                     : diagnostic_msgs::msg::DiagnosticStatus::ERROR;
  Add(status, "platform", record.platform);
  Add(status, "raw_elevation_revision",
      std::to_string(record.raw_elevation_revision));
  Add(status, "fine_traversability_revision",
      std::to_string(record.fine_traversability_revision));
  Add(status, "global_guidance_revision",
      std::to_string(record.global_guidance_revision));
  Add(status, "received_map_count", std::to_string(record.received_map_count));
  Add(status, "applied_map_count", std::to_string(record.applied_map_count));
  Add(status, "duplicate_map_count", std::to_string(record.duplicate_map_count));
  Add(status, "rejected_map_count", std::to_string(record.rejected_map_count));
  Add(status, "middleware_lost_count",
      std::to_string(record.middleware_lost_count));
  Add(status, "incompatible_qos_count",
      std::to_string(record.incompatible_qos_count));
  Add(status, "updated_cells", std::to_string(record.updated_cells));
  Add(status, "dirty_tiles", std::to_string(record.dirty_tiles));
  Add(status, "derivation_lag", std::to_string(record.derivation_lag));
  Add(status, "allocated_tiles", std::to_string(record.allocated_tiles));
  Add(status, "raw_update_ms", Decimal(record.raw_update_ms));
  Add(status, "fine_derivation_ms", Decimal(record.fine_derivation_ms));
  Add(status, "guidance_derivation_ms", Decimal(record.guidance_derivation_ms));
  return Wrap(std::move(status));
}

diagnostic_msgs::msg::DiagnosticArray MakePlanningCycleDiagnostics(
    const PlanningCycleDiagnosticsRecord& record) {
  diagnostic_msgs::msg::DiagnosticStatus status;
  status.name = "lunar_incremental_navigation/planning_cycle";
  const bool plan_found_within_sla =
      record.cycle_result == "PLAN_FOUND" &&
      record.latency_class == PlanningCycleLatencyClass::kSlaMet;
  status.level = plan_found_within_sla
                     ? diagnostic_msgs::msg::DiagnosticStatus::OK
                     : diagnostic_msgs::msg::DiagnosticStatus::WARN;
  Add(status, "session_id", record.session_id);
  Add(status, "planning_cycle", std::to_string(record.planning_cycle));
  Add(status, "segment_revision", std::to_string(record.segment_revision));
  Add(status, "cycle_result", record.cycle_result);
  Add(status, "reason_code", record.reason_code);
  Add(status, "path_state", record.path_state);
  Add(status, "reaches_final_goal", record.reaches_final_goal ? "true" : "false");
  Add(status, "fine_traversability_revision",
      std::to_string(record.fine_traversability_revision));
  Add(status, "global_guidance_revision",
      std::to_string(record.global_guidance_revision));
  Add(status, "global_guidance_status", record.global_guidance_status);
  Add(status, "global_route_reused", record.global_route_reused ? "true" : "false");
  Add(status, "global_elapsed_ms", Decimal(record.global_elapsed_ms));
  Add(status, "global_expanded_states",
      std::to_string(record.global_expanded_states));
  Add(status, "global_open_peak", std::to_string(record.global_open_peak));
  Add(status, "local_elapsed_ms", Decimal(record.local_elapsed_ms));
  Add(status, "local_expanded_states", std::to_string(record.local_expanded_states));
  Add(status, "local_open_peak", std::to_string(record.local_open_peak));
  Add(status, "start_patch_used", record.start_patch_used ? "true" : "false");
  Add(status, "start_patch_radius_m", Decimal(record.start_patch_radius_m));
  Add(status, "start_patch_assumed_cells",
      std::to_string(record.start_patch_assumed_cells));
  Add(status, "start_patch_elapsed_ms", Decimal(record.start_patch_elapsed_ms));
  Add(status, "postprocess_elapsed_ms", Decimal(record.postprocess_elapsed_ms));
  Add(status, "total_elapsed_ms", Decimal(record.total_elapsed_ms));
  Add(status, "latency_class",
      std::string{PlanningCycleLatencyClassName(record.latency_class)});
  Add(status, "path_points", std::to_string(record.path_points));
  Add(status, "path_length_m", Decimal(record.path_length_m));
  return Wrap(std::move(status));
}

std::string FindDiagnosticValue(
    const diagnostic_msgs::msg::DiagnosticArray& diagnostics,
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

}  // namespace lunar::incremental_navigation_ros

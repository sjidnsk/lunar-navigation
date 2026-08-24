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
  add_value("wheel_metrics_available",
            result.wheel_metrics.has_value() ? "true" : "false");
  if (result.wheel_metrics.has_value()) {
    const auto& metrics = *result.wheel_metrics;
    add_value("wheel_expanded_states", std::to_string(metrics.expanded_states));
    add_value("wheel_edge_validation_evaluations",
              std::to_string(metrics.edge_validation_evaluations));
    add_value("wheel_edge_validation_cache_hits",
              std::to_string(metrics.edge_validation_cache_hits));
    add_value("wheel_broad_phase_rejects", std::to_string(metrics.broad_phase_rejects));
    add_value("wheel_full_certifications", std::to_string(metrics.full_certifications));
    add_value("wheel_full_invalidations", std::to_string(metrics.full_invalidations));
    add_value("wheel_sweep_cell_checks", std::to_string(metrics.sweep_cell_checks));
    add_value("wheel_quantization_alias_states",
              std::to_string(metrics.quantization_alias_states));
    add_value("wheel_quantized_state_reuses",
              std::to_string(metrics.quantized_state_reuses));
    add_value("wheel_quantized_endpoint_aliases",
              std::to_string(metrics.quantized_endpoint_aliases));
    add_value("wheel_quantized_state_count",
              std::to_string(metrics.quantized_state_count));
    add_value("wheel_maximum_active_labels_per_key",
              std::to_string(metrics.maximum_active_labels_per_key));
    add_value("wheel_used_narrow_resolution",
              metrics.used_narrow_resolution ? "true" : "false");
    add_value("wheel_finest_xy_key_resolution_m",
              Decimal(metrics.finest_xy_key_resolution_m));
    add_value("wheel_maximum_yaw_bins", std::to_string(metrics.maximum_yaw_bins));
    add_value("wheel_ara_search_invocations",
              std::to_string(metrics.ara_search_invocations));
    add_value("wheel_returned_edge_certificate_confirmations",
              std::to_string(metrics.returned_edge_certificate_confirmations));
    add_value("wheel_mode_switch_edge_count",
              std::to_string(metrics.mode_switch_edge_count));
    add_value("wheel_reverse_edge_count", std::to_string(metrics.reverse_edge_count));
    add_value("wheel_start_heuristic_lower_bound",
              Decimal(metrics.start_heuristic_lower_bound));
    add_value("wheel_has_certified_preferred_candidate",
              metrics.has_certified_preferred_candidate ? "true" : "false");
    add_value("wheel_preferred_candidate_full_primitive_edge_count",
              std::to_string(metrics.preferred_candidate_full_primitive_edge_count));
    add_value("wheel_preferred_candidate_terminal_connector_edge_count",
              std::to_string(metrics.preferred_candidate_terminal_connector_edge_count));
    add_value("wheel_preferred_candidate_certified_edge_count",
              std::to_string(metrics.preferred_candidate_certified_edge_count));
    add_value("wheel_preferred_candidate_cost",
              Decimal(metrics.preferred_candidate_cost));
    add_value("wheel_preferred_builder_invocations",
              std::to_string(metrics.preferred_builder_invocations));
    for (std::size_t index = 0; index < metrics.cost_components.size(); ++index) {
      add_value("wheel_cost_component_" + std::to_string(index),
                Decimal(metrics.cost_components[index]));
    }
    for (std::size_t index = 0; index < metrics.cost_scales.size(); ++index) {
      add_value("wheel_cost_scale_" + std::to_string(index),
                Decimal(metrics.cost_scales[index]));
    }
    add_value("wheel_direct_unknown_or_unsupported_footprint_rejects",
              std::to_string(metrics.direct_unknown_or_unsupported_footprint_rejects));
    add_value("wheel_measured_obstacle_clearance_rejects",
              std::to_string(metrics.measured_obstacle_clearance_rejects));
    add_value("wheel_slope_or_roughness_rejects",
              std::to_string(metrics.slope_or_roughness_rejects));
    add_value("wheel_relief_or_underbody_rejects",
              std::to_string(metrics.relief_or_underbody_rejects));
    add_value("wheel_dynamics_or_primitive_shape_rejects",
              std::to_string(metrics.dynamics_or_primitive_shape_rejects));
    add_value("wheel_deadline_or_cancellation_interruptions",
              std::to_string(metrics.deadline_or_cancellation_interruptions));
    add_value("wheel_far_clearance_scan_skips",
              std::to_string(metrics.far_clearance_scan_skips));
    add_value("wheel_occupied_clearance_cell_checks",
              std::to_string(metrics.occupied_clearance_cell_checks));
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

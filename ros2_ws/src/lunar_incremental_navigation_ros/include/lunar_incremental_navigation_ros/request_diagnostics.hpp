#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

#include <diagnostic_msgs/msg/diagnostic_array.hpp>

namespace lunar::incremental_navigation_ros {

enum class PlanningCycleLatencyClass : std::uint8_t {
  kSlaMet,
  kSlaMissed,
  kHardTimeout,
};

[[nodiscard]] PlanningCycleLatencyClass ClassifyPlanningCycleLatency(
    std::chrono::nanoseconds elapsed, std::chrono::nanoseconds sla,
    std::chrono::nanoseconds hard_timeout) noexcept;
[[nodiscard]] std::string_view PlanningCycleLatencyClassName(
    PlanningCycleLatencyClass latency_class) noexcept;

struct MapDiagnosticsRecord final {
  std::string platform;
  std::uint64_t raw_elevation_revision{};
  std::uint64_t fine_traversability_revision{};
  std::uint64_t global_guidance_revision{};
  std::uint64_t received_map_count{};
  std::uint64_t applied_map_count{};
  std::uint64_t duplicate_map_count{};
  std::uint64_t rejected_map_count{};
  std::uint64_t middleware_lost_count{};
  std::uint64_t incompatible_qos_count{};
  std::size_t updated_cells{};
  std::size_t dirty_tiles{};
  std::uint64_t derivation_lag{};
  std::size_t allocated_tiles{};
  double raw_update_ms{};
  double fine_derivation_ms{};
  double guidance_derivation_ms{};
};

struct PlanningCycleDiagnosticsRecord final {
  std::string session_id;
  std::uint64_t planning_cycle{};
  std::uint64_t segment_revision{};
  std::string cycle_result;
  std::string reason_code;
  std::string path_state;
  bool reaches_final_goal{};
  std::uint64_t fine_traversability_revision{};
  std::uint64_t global_guidance_revision{};
  std::string global_guidance_status;
  bool global_route_reused{};
  double global_elapsed_ms{};
  std::uint64_t global_expanded_states{};
  std::size_t global_open_peak{};
  double local_elapsed_ms{};
  std::uint64_t local_expanded_states{};
  std::size_t local_open_peak{};
  bool start_patch_used{};
  double start_patch_radius_m{};
  std::size_t start_patch_assumed_cells{};
  double start_patch_elapsed_ms{};
  double postprocess_elapsed_ms{};
  double total_elapsed_ms{};
  PlanningCycleLatencyClass latency_class{PlanningCycleLatencyClass::kSlaMet};
  std::size_t path_points{};
  double path_length_m{};
};

[[nodiscard]] diagnostic_msgs::msg::DiagnosticArray MakeMapDiagnostics(
    const MapDiagnosticsRecord& record);
[[nodiscard]] diagnostic_msgs::msg::DiagnosticArray
MakePlanningCycleDiagnostics(const PlanningCycleDiagnosticsRecord& record);

[[nodiscard]] std::string FindDiagnosticValue(
    const diagnostic_msgs::msg::DiagnosticArray& diagnostics,
    const std::string& key);

}  // namespace lunar::incremental_navigation_ros

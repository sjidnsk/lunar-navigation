#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <optional>
#include <vector>

#include "lunar_pure_planner_core/types/planning_request.hpp"
#include "shared/local_terrain_projection.hpp"

namespace lunar::pure_planning {

struct LocalPlanMetrics final {
  std::uint64_t expanded_states{};
  std::size_t edge_validation_evaluations{};
  bool used_narrow_resolution{};
};

namespace wheel {

struct WheelPlanRequest final {
  WheeledState start;
  LocalGoalSet goals_odom;
  const shared::LocalTerrainProjection* terrain{};
  const WheeledCapability* capability{};
  std::uint64_t local_source_sequence{};
  std::uint64_t capability_fingerprint{};
  SearchControl control;
  AnytimeSearchConfig search;
};

struct WheelPlanResult final {
  LocalPlanStatus status{LocalPlanStatus::kInvalidInput};
  std::string reason_code;
  std::vector<TrajectoryPoint> trajectory;
  LocalPlanMetrics metrics;
  std::size_t edge_validation_cache_hits{};
  std::size_t quantization_alias_states{};
  std::size_t quantized_state_reuses{};
  std::size_t quantized_endpoint_aliases{};
  std::size_t quantized_state_count{};
  std::size_t maximum_active_labels_per_key{};
  std::size_t sweep_cell_checks{};
  std::size_t mode_switch_edge_count{};
  std::size_t reverse_edge_count{};
  double finest_xy_key_resolution_m{};
  std::size_t maximum_yaw_bins{};
  double start_heuristic_lower_bound{};
  bool has_certified_preferred_candidate{};
  std::size_t preferred_candidate_full_primitive_edge_count{};
  std::size_t preferred_candidate_terminal_connector_edge_count{};
  std::size_t preferred_candidate_certified_edge_count{};
  double preferred_candidate_cost{};
  std::size_t preferred_builder_invocations{};
  std::size_t ara_search_invocations{};
  std::optional<std::size_t> selected_goal_index;
  std::array<double, 5U> cost_components{};
  std::array<double, 5U> cost_scales{};
  double cost{};

  [[nodiscard]] bool ok() const noexcept {
    return status == LocalPlanStatus::kSolved && !trajectory.empty();
  }
};

[[nodiscard]] WheelPlanResult PlanWheel(const WheelPlanRequest& request);

}  // namespace wheel
}  // namespace lunar::pure_planning

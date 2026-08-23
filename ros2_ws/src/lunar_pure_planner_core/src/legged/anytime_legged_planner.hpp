#pragma once

#include <array>
#include <cstddef>
#include <limits>
#include <string>
#include <vector>

#include "legged/legged_types.hpp"
#include "lunar_pure_planner_core/search_control.hpp"
#include "lunar_pure_planner_core/types/planning_request.hpp"
#include "shared/local_terrain_projection.hpp"
#include "wheel/anytime_wheel_planner.hpp"

namespace lunar::pure_planning::legged {

struct LeggedPlanRequest final {
  LeggedState start;
  GoalRegion goal_odom;
  const shared::LocalTerrainProjection* terrain{};
  const LeggedCapability* capability{};
  SearchControl control;
  AnytimeSearchConfig search;
};

struct LeggedPlanResult final {
  LocalPlanStatus status{LocalPlanStatus::kInvalidInput};
  std::string reason_code;
  std::vector<LeggedTransition> trajectory;
  double cost{std::numeric_limits<double>::infinity()};
  LocalPlanMetrics metrics;
  std::size_t edge_validation_cache_hits{};
  std::size_t maximum_edge_sweep_evaluations{};
  std::size_t sweep_cell_checks{};
  std::size_t quantized_endpoint_aliases{};
  std::size_t quantized_state_count{};
  std::size_t mode_change_edge_count{};
  double finest_xy_key_resolution_m{};
  std::size_t maximum_yaw_bin_count{};
  double maximum_sweep_translation_step_m{};
  std::array<double, 5U> cost_components{};
  std::array<double, 5U> cost_scales{};

  [[nodiscard]] bool ok() const noexcept {
    return status == LocalPlanStatus::kSolved;
  }
};

[[nodiscard]] LeggedPlanResult PlanLegged(const LeggedPlanRequest& request);

}  // namespace lunar::pure_planning::legged

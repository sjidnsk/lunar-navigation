#pragma once

#include <cstddef>
#include <stop_token>
#include <string>
#include <vector>

#include "lunar_planner_core/types/platform_capability.hpp"
#include "shared/map_snapshot.hpp"
#include "shared/safe_projection.hpp"
#include "legged/legged_types.hpp"

namespace lunar::planning::legged {

struct LeggedTerrainEvaluation final {
  bool hard_feasible{};
  bool canceled{};
  Interval body_height_m;
  double elevation_m{};
  double slope_rad{};
  double roughness_m{};
  double maximum_neighbor_step_m{};
  double body_clearance_m{};
  std::vector<std::string> rejection_reasons;
};

struct LeggedSweepResult final {
  bool valid{};
  bool canceled{};
  Interval reachable_body_z_m;
  std::size_t sample_count{};
  std::string reason_code;
};

[[nodiscard]] LeggedTerrainEvaluation EvaluateLeggedTerrainCell(
    const shared::SafeProjection& projection,
    const LeggedCapability& capability,
    shared::GridCell cell,
    std::stop_token stop_token);

[[nodiscard]] LeggedSweepResult ValidateLeggedBodySweep(
    const LeggedPose& source,
    const LeggedPose& target,
    const Interval& source_body_z_m,
    const shared::SafeProjection& projection,
    const LeggedCapability& capability,
    std::stop_token stop_token);

}  // namespace lunar::planning::legged

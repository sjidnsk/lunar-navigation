#pragma once

#include <cstddef>
#include <stop_token>
#include <string>
#include <vector>

#include "hierarchical/local_planning_problem.hpp"
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

class LeggedTerrainGrid final {
 public:
  LeggedTerrainGrid(
      const shared::SafeProjection& projection,
      const LeggedCapability& capability,
      std::stop_token stop_token);

  [[nodiscard]] bool ok() const noexcept;
  [[nodiscard]] bool canceled() const noexcept;
  [[nodiscard]] bool Matches(
      const shared::SafeProjection& projection,
      const LeggedCapability& capability) const noexcept;
  [[nodiscard]] const LeggedTerrainEvaluation* Find(
      shared::GridCell cell) const noexcept;
  [[nodiscard]] bool AllCellsHardFeasible(
      shared::GridCell minimum, shared::GridCell maximum) const noexcept;
  [[nodiscard]] std::size_t exact_rectangle_test_count() const noexcept;
  void RecordExactRectangleTest() const noexcept;

 private:
  const shared::SafeProjection* projection_{};
  const shared::MapSnapshot* source_map_{};
  const LeggedCapability* capability_{};
  std::vector<LeggedTerrainEvaluation> cells_;
  std::vector<std::size_t> hard_infeasible_prefix_sum_;
  bool canceled_{};
  mutable std::size_t exact_rectangle_test_count_{};
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
    const hierarchical::LocalSearchDomain& search_domain,
    std::stop_token stop_token,
    const LeggedTerrainGrid* terrain_grid = nullptr);

}  // namespace lunar::planning::legged

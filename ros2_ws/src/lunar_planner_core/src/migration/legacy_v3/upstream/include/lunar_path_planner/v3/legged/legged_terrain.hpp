#pragma once

#include <optional>
#include <span>
#include <string>
#include <vector>

#include "lunar_path_planner/v3/legged/body_motion_primitive.hpp"
#include "lunar_path_planner/v3/legged/height_interval.hpp"
#include "lunar_path_planner/v3/map/safe_projection.hpp"

namespace lunar::planning::v3 {

struct LeggedTerrainCellSample final {
  bool known{};
  bool hard_obstacle{};
  double confidence{};
  double elevation_m{};
  Vec3 normal;
  double roughness_m{};
  double maximum_neighbor_step_m{};
  double unsupported_gap_width_m{};
  double body_clearance_m{};
  double analytic_time_cost_s{};
  double resolved_energy{};
  double resolved_nonfatal_risk{};
};

struct LeggedTerrainEvaluation final {
  bool hard_feasible{};
  HeightInterval body_height_interval;
  Vec3 fitted_normal;
  double plane_residual_m{};
  DurationNanoseconds terrain_scaled_duration;
  SecondaryCostVector secondary_costs;
  std::vector<std::string> rejection_reasons;
};

[[nodiscard]] LeggedTerrainEvaluation EvaluateLeggedTerrainSample(
    const LeggedTerrainCellSample& sample,
    const LeggedCapabilityView& capability);

[[nodiscard]] std::optional<HeightInterval>
PropagateLeggedEdgeHeightInterval(
    const HeightInterval& source,
    const BodyMotionPrimitive& edge,
    std::span<const LeggedTerrainEvaluation> samples);

class LeggedTerrainEvaluator final {
 public:
  LeggedTerrainEvaluator(const SafeProjection& projection,
                         const LeggedCapabilityView& capability) noexcept;

  [[nodiscard]] LeggedTerrainEvaluation EvaluatePose(
      const PoseXyzYaw& pose,
      const BodyConvexPolytope& body_envelope) const;

  [[nodiscard]] std::optional<HeightInterval> PropagateEdgeInterval(
      const HeightInterval& source,
      const BodyMotionPrimitive& edge,
      std::span<const LeggedTerrainEvaluation> samples) const;

 private:
  const SafeProjection* projection_;
  const LeggedCapabilityView* capability_;
};

}  // namespace lunar::planning::v3

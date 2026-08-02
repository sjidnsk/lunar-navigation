#include "lunar_path_planner/v3/legged/legged_terrain.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numbers>

namespace lunar::planning::v3 {
namespace {

[[nodiscard]] bool IsFinite(const Vec3& value) noexcept {
  return std::isfinite(value.x) && std::isfinite(value.y) &&
         std::isfinite(value.z);
}

[[nodiscard]] bool IsFiniteNonNegative(double value) noexcept {
  return std::isfinite(value) && value >= 0.0;
}

[[nodiscard]] DurationNanoseconds SecondsToDuration(
    double seconds) noexcept {
  if (!IsFiniteNonNegative(seconds)) {
    return {};
  }
  constexpr long double kNanosecondsPerSecond = 1.0e9L;
  const long double nanoseconds =
      static_cast<long double>(seconds) * kNanosecondsPerSecond;
  if (nanoseconds >=
      static_cast<long double>(
          std::numeric_limits<std::int64_t>::max())) {
    return DurationNanoseconds{
        std::chrono::nanoseconds{
            std::numeric_limits<std::int64_t>::max()}};
  }
  return DurationNanoseconds{
      std::chrono::nanoseconds{
          static_cast<std::int64_t>(std::llround(nanoseconds))}};
}

[[nodiscard]] double SlopeRadians(const Vec3& normal) noexcept {
  const double norm = std::hypot(normal.x, normal.y, normal.z);
  if (!std::isfinite(norm) || norm <= 0.0) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return std::acos(std::clamp(normal.z / norm, -1.0, 1.0));
}

[[nodiscard]] bool ValidBodyEnvelope(
    const BodyConvexPolytope& envelope) noexcept {
  return !envelope.body_frame_halfspaces.halfspaces.empty();
}

[[nodiscard]] std::optional<Cell> PositionToCell(
    const GridGeometry& geometry, const Vec3& position) noexcept {
  if (!IsFinite(position) || !(geometry.resolution_m > 0.0)) {
    return std::nullopt;
  }
  const double x =
      (position.x - geometry.origin_m.x) / geometry.resolution_m;
  const double y =
      (position.y - geometry.origin_m.y) / geometry.resolution_m;
  if (!std::isfinite(x) || !std::isfinite(y) || x < 0.0 ||
      y < 0.0 || x >= static_cast<double>(geometry.width) ||
      y >= static_cast<double>(geometry.height)) {
    return std::nullopt;
  }
  return Cell{
      .x = static_cast<std::int32_t>(std::floor(x)),
      .y = static_cast<std::int32_t>(std::floor(y)),
  };
}

[[nodiscard]] std::size_t Index(const GridGeometry& geometry,
                                Cell cell) noexcept {
  return static_cast<std::size_t>(cell.y) * geometry.width +
         static_cast<std::size_t>(cell.x);
}

[[nodiscard]] double MaximumNeighborStep(
    const SafeProjection& projection, Cell cell) noexcept {
  constexpr std::int32_t kDx[] = {-1, 1, 0, 0};
  constexpr std::int32_t kDy[] = {0, 0, -1, 1};
  const auto elevation = projection.ElevationMeters();
  const std::size_t center = Index(projection.geometry, cell);
  double maximum = 0.0;
  for (std::size_t direction = 0U; direction < 4U; ++direction) {
    const Cell neighbor{
        .x = cell.x + kDx[direction],
        .y = cell.y + kDy[direction],
    };
    if (!projection.InBounds(neighbor) || !projection.Known(neighbor)) {
      continue;
    }
    maximum =
        std::max(maximum,
                 std::abs(static_cast<double>(elevation[center]) -
                          static_cast<double>(
                              elevation[Index(projection.geometry,
                                              neighbor)])));
  }
  return maximum;
}

}  // namespace

LeggedTerrainEvaluation EvaluateLeggedTerrainSample(
    const LeggedTerrainCellSample& sample,
    const LeggedCapabilityView& capability) {
  LeggedTerrainEvaluation result;
  result.fitted_normal = sample.normal;
  result.plane_residual_m =
      std::isfinite(sample.roughness_m) ? sample.roughness_m : 0.0;
  const auto& limits = capability.terrain_thresholds;

  if (!sample.known || !std::isfinite(sample.confidence) ||
      sample.confidence < limits.minimum_confidence) {
    result.rejection_reasons.emplace_back(
        "known_and_confident");
  }
  if (sample.hard_obstacle ||
      !std::isfinite(sample.body_clearance_m) ||
      sample.body_clearance_m < limits.minimum_body_clearance_m) {
    result.rejection_reasons.emplace_back(
        "body_collision_and_clearance");
  }

  const double slope = SlopeRadians(sample.normal);
  if (!std::isfinite(slope) || slope > limits.maximum_slope_rad) {
    result.rejection_reasons.emplace_back("slope");
  }
  if (!std::isfinite(sample.roughness_m) ||
      sample.roughness_m < 0.0 ||
      sample.roughness_m > limits.maximum_roughness_m) {
    result.rejection_reasons.emplace_back(
        "roughness_and_plane_residual");
  }
  if (!std::isfinite(sample.maximum_neighbor_step_m) ||
      sample.maximum_neighbor_step_m < 0.0 ||
      sample.maximum_neighbor_step_m >
          limits.maximum_step_height_m) {
    result.rejection_reasons.emplace_back("step_height");
  }
  if (!std::isfinite(sample.unsupported_gap_width_m) ||
      sample.unsupported_gap_width_m < 0.0 ||
      sample.unsupported_gap_width_m > limits.maximum_gap_width_m) {
    result.rejection_reasons.emplace_back("gap_width");
  }
  if (!IsFinite(sample.normal)) {
    result.rejection_reasons.emplace_back("terrain_normal_invalid");
  }

  const HeightInterval body_height{
      .min_m = sample.elevation_m + limits.minimum_body_height_m,
      .max_m = sample.elevation_m + limits.maximum_body_height_m,
  };
  if (!std::isfinite(sample.elevation_m) ||
      !IsValidHeightInterval(body_height)) {
    result.rejection_reasons.emplace_back("body_height_interval");
  } else {
    result.body_height_interval = body_height;
  }

  result.hard_feasible = result.rejection_reasons.empty();
  if (!result.hard_feasible) {
    return result;
  }
  result.terrain_scaled_duration =
      SecondsToDuration(sample.analytic_time_cost_s);
  if (IsFiniteNonNegative(sample.resolved_energy) &&
      IsFiniteNonNegative(sample.resolved_nonfatal_risk)) {
    result.secondary_costs.energy = sample.resolved_energy;
    result.secondary_costs.risk = sample.resolved_nonfatal_risk;
  }
  return result;
}

std::optional<HeightInterval> PropagateLeggedEdgeHeightInterval(
    const HeightInterval& source,
    const BodyMotionPrimitive& edge,
    std::span<const LeggedTerrainEvaluation> samples) {
  if (!IsValidHeightInterval(source) || samples.empty() ||
      !std::isfinite(edge.maximum_up_delta_m) ||
      edge.maximum_up_delta_m < 0.0 ||
      !std::isfinite(edge.maximum_down_delta_m) ||
      edge.maximum_down_delta_m < 0.0) {
    return std::nullopt;
  }

  std::optional<HeightInterval> reachable =
      IntersectHeightIntervals(source, samples.front().body_height_interval);
  if (!samples.front().hard_feasible || !reachable.has_value()) {
    return std::nullopt;
  }
  const double denominator =
      static_cast<double>(std::max<std::size_t>(samples.size() - 1U, 1U));
  const double up_per_sample = edge.maximum_up_delta_m / denominator;
  const double down_per_sample =
      edge.maximum_down_delta_m / denominator;
  for (std::size_t index = 1U; index < samples.size(); ++index) {
    if (!samples[index].hard_feasible) {
      return std::nullopt;
    }
    const auto expanded =
        ExpandHeightInterval(*reachable, down_per_sample, up_per_sample);
    if (!expanded.has_value()) {
      return std::nullopt;
    }
    reachable = IntersectHeightIntervals(
        *expanded, samples[index].body_height_interval);
    if (!reachable.has_value()) {
      return std::nullopt;
    }
  }
  return reachable;
}

LeggedTerrainEvaluator::LeggedTerrainEvaluator(
    const SafeProjection& projection,
    const LeggedCapabilityView& capability) noexcept
    : projection_(&projection), capability_(&capability) {}

LeggedTerrainEvaluation LeggedTerrainEvaluator::EvaluatePose(
    const PoseXyzYaw& pose,
    const BodyConvexPolytope& body_envelope) const {
  if (projection_ == nullptr || capability_ == nullptr ||
      !ValidBodyEnvelope(body_envelope)) {
    LeggedTerrainEvaluation invalid;
    invalid.rejection_reasons.emplace_back(
        "body_collision_and_clearance");
    return invalid;
  }
  const auto cell = PositionToCell(projection_->geometry, pose.position_m);
  if (!cell.has_value()) {
    LeggedTerrainEvaluation invalid;
    invalid.rejection_reasons.emplace_back("known_and_confident");
    return invalid;
  }
  const std::size_t index = Index(projection_->geometry, *cell);
  const auto map = projection_->source_map();
  const auto normals = projection_->SurfaceNormals();
  const auto confidence = map->Confidence();
  const auto obstacles = map->HardObstacleMask();
  const LeggedTerrainCellSample sample{
      .known = projection_->Known(*cell),
      .hard_obstacle = obstacles[index] != 0U ||
                       !projection_->HardFeasible(*cell),
      .confidence = static_cast<double>(confidence[index]),
      .elevation_m =
          static_cast<double>(projection_->ElevationMeters()[index]),
      .normal =
          Vec3{
              static_cast<double>(normals.x[index]),
              static_cast<double>(normals.y[index]),
              static_cast<double>(normals.z[index]),
          },
      .roughness_m =
          static_cast<double>(projection_->RoughnessMeters()[index]),
      .maximum_neighbor_step_m =
          MaximumNeighborStep(*projection_, *cell),
      .unsupported_gap_width_m = 0.0,
      .body_clearance_m = projection_->ClearanceMeters(*cell),
      .analytic_time_cost_s =
          static_cast<double>(projection_->analytic_time_cost_s[index]),
      .resolved_energy =
          static_cast<double>(projection_->resolved_energy[index]),
      .resolved_nonfatal_risk =
          static_cast<double>(
              projection_->resolved_nonfatal_risk[index]),
  };
  return EvaluateLeggedTerrainSample(sample, *capability_);
}

std::optional<HeightInterval>
LeggedTerrainEvaluator::PropagateEdgeInterval(
    const HeightInterval& source,
    const BodyMotionPrimitive& edge,
    std::span<const LeggedTerrainEvaluation> samples) const {
  return PropagateLeggedEdgeHeightInterval(source, edge, samples);
}

}  // namespace lunar::planning::v3

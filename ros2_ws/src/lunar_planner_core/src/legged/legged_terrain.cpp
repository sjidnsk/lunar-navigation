#include "legged/legged_terrain.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "shared/terrain_checks.hpp"

namespace lunar::planning::legged {
namespace {

constexpr double kComparisonTolerance = 1.0e-9;

[[nodiscard]] bool IsFinite(const LeggedPose& pose) noexcept {
  return std::isfinite(pose.position_m.x) &&
      std::isfinite(pose.position_m.y) &&
      std::isfinite(pose.position_m.z) && std::isfinite(pose.yaw_rad);
}

void AddReason(
    LeggedTerrainEvaluation& result, std::string reason) {
  if (std::ranges::find(result.rejection_reasons, reason) ==
      result.rejection_reasons.end()) {
    result.rejection_reasons.push_back(std::move(reason));
  }
}

[[nodiscard]] LeggedSweepResult SweepFailure(
    std::string reason_code, const std::size_t sample_count = 0U) {
  return LeggedSweepResult{
      .valid = false,
      .canceled = false,
      .reachable_body_z_m = {},
      .sample_count = sample_count,
      .reason_code = std::move(reason_code),
  };
}

[[nodiscard]] bool ValidCapability(
    const LeggedCapability& capability) noexcept {
  const auto valid_velocity = [](const Interval& interval) {
    return ValidInterval(interval) && interval.lower <= 0.0 &&
           interval.upper >= 0.0 && interval.lower < interval.upper;
  };
  return std::isfinite(capability.body_half_extent_m.x) &&
      std::isfinite(capability.body_half_extent_m.y) &&
      std::isfinite(capability.body_half_extent_m.z) &&
      capability.body_half_extent_m.x > 0.0 &&
      capability.body_half_extent_m.y > 0.0 &&
      capability.body_half_extent_m.z > 0.0 &&
      std::isfinite(capability.maximum_slope_rad) &&
      capability.maximum_slope_rad > 0.0 &&
      std::isfinite(capability.maximum_roughness_m) &&
      capability.maximum_roughness_m >= 0.0 &&
      std::isfinite(capability.maximum_step_height_m) &&
      capability.maximum_step_height_m >= 0.0 &&
      std::isfinite(capability.maximum_gap_width_m) &&
      capability.maximum_gap_width_m >= 0.0 &&
      std::isfinite(capability.minimum_confidence) &&
      capability.minimum_confidence >= 0.0 &&
      capability.minimum_confidence <= 1.0 &&
      std::isfinite(capability.minimum_body_clearance_m) &&
      capability.minimum_body_clearance_m >= 0.0 &&
      ValidInterval(capability.body_height_m) &&
      valid_velocity(capability.forward_speed_mps) &&
      valid_velocity(capability.lateral_speed_mps) &&
      valid_velocity(capability.vertical_speed_mps) &&
      valid_velocity(capability.yaw_rate_radps) &&
      std::isfinite(capability.maximum_linear_acceleration_mps2) &&
      capability.maximum_linear_acceleration_mps2 > 0.0 &&
      std::isfinite(capability.maximum_yaw_acceleration_radps2) &&
      capability.maximum_yaw_acceleration_radps2 > 0.0;
}

}  // namespace

LeggedTerrainEvaluation EvaluateLeggedTerrainCell(
    const shared::SafeProjection& projection,
    const LeggedCapability& capability,
    const shared::GridCell cell,
    const std::stop_token stop_token) {
  LeggedTerrainEvaluation result;
  if (stop_token.stop_requested()) {
    result.canceled = true;
    result.rejection_reasons.emplace_back("REQUEST_CANCELED");
    return result;
  }
  if (projection.source_map() == nullptr ||
      !ValidCapability(capability) || !projection.InBounds(cell)) {
    result.rejection_reasons.emplace_back("LEGGED_TERRAIN_REQUEST_INVALID");
    return result;
  }
  const shared::MapSnapshot& map = *projection.source_map();
  const std::size_t index = map.Index(cell);
  result.elevation_m = static_cast<double>(map.FloatLayer("elevation")[index]);
  result.slope_rad = static_cast<double>(projection.SlopeRadians(cell));
  result.roughness_m = static_cast<double>(projection.RoughnessMeters(cell));
  result.maximum_neighbor_step_m =
      shared::ComputeMaximumNeighborStep(map, cell);
  result.body_clearance_m =
      static_cast<double>(projection.ClearanceMeters(cell));
  result.body_height_m = Interval{
      .lower = result.elevation_m + capability.body_height_m.lower,
      .upper = result.elevation_m + capability.body_height_m.upper,
  };

  if (!projection.Known(cell)) {
    AddReason(result, "LEGGED_TERRAIN_UNKNOWN");
  }
  if (map.ByteLayer("obstacle")[index] != 0U ||
      map.ByteLayer("forbidden")[index] != 0U) {
    AddReason(result, "LEGGED_HARD_OBSTACLE");
  }
  if (!std::isfinite(result.slope_rad) ||
      result.slope_rad > capability.maximum_slope_rad +
          kComparisonTolerance ||
      result.slope_rad > projection.maximum_slope_rad() +
          kComparisonTolerance) {
    AddReason(result, "LEGGED_SLOPE_LIMIT");
  }
  if (!std::isfinite(result.roughness_m) || result.roughness_m < 0.0 ||
      result.roughness_m > capability.maximum_roughness_m +
          kComparisonTolerance) {
    AddReason(result, "LEGGED_ROUGHNESS_LIMIT");
  }
  if (!std::isfinite(result.maximum_neighbor_step_m) ||
      result.maximum_neighbor_step_m > capability.maximum_step_height_m +
          kComparisonTolerance) {
    AddReason(result, "LEGGED_STEP_HEIGHT_LIMIT");
  }
  if (!std::isfinite(result.body_clearance_m) ||
      result.body_clearance_m + kComparisonTolerance <
          capability.minimum_body_clearance_m) {
    AddReason(result, "LEGGED_BODY_CLEARANCE_LIMIT");
  }
  if (!ValidInterval(result.body_height_m)) {
    AddReason(result, "LEGGED_BODY_HEIGHT_INTERVAL_INVALID");
  }
  if (!projection.HardFeasible(cell) &&
      result.rejection_reasons.empty()) {
    AddReason(result, "LEGGED_TERRAIN_HARD_INFEASIBLE");
  }
  result.hard_feasible = result.rejection_reasons.empty();
  return result;
}

LeggedSweepResult ValidateLeggedBodySweep(
    const LeggedPose& source,
    const LeggedPose& target,
    const Interval& source_body_z_m,
    const std::chrono::nanoseconds nominal_duration,
    const shared::SafeProjection& projection,
    const LeggedCapability& capability,
    const std::size_t maximum_subdivisions,
    const std::stop_token stop_token) {
  if (stop_token.stop_requested()) {
    return LeggedSweepResult{
        .valid = false,
        .canceled = true,
        .reachable_body_z_m = {},
        .sample_count = 0U,
        .reason_code = "REQUEST_CANCELED",
    };
  }
  if (!IsFinite(source) || !IsFinite(target) ||
      !ValidInterval(source_body_z_m) || nominal_duration.count() <= 0 ||
      projection.source_map() == nullptr ||
      !ValidCapability(capability) || maximum_subdivisions == 0U) {
    return SweepFailure("LEGGED_SWEEP_REQUEST_INVALID");
  }
  const double support_radius = std::hypot(
      capability.body_half_extent_m.x, capability.body_half_extent_m.y);
  const double translation = std::hypot(
      target.position_m.x - source.position_m.x,
      target.position_m.y - source.position_m.y);
  const double yaw_delta = ShortestYawDelta(source.yaw_rad, target.yaw_rad);
  const double swept_distance =
      translation + std::abs(yaw_delta) * support_radius;
  const double maximum_step =
      projection.source_map()->resolution_m() * 0.25;
  const std::size_t subdivisions = std::max<std::size_t>(
      1U, static_cast<std::size_t>(std::ceil(swept_distance / maximum_step)));
  if (subdivisions > maximum_subdivisions) {
    return SweepFailure("LEGGED_SWEEP_RESOLUTION_LIMIT");
  }
  const double duration_s =
      std::chrono::duration<double>(nominal_duration).count();
  const double sample_duration_s =
      duration_s / static_cast<double>(subdivisions);
  Interval reachable = source_body_z_m;
  std::size_t sample_count = 0U;
  const shared::MapSnapshot& map = *projection.source_map();
  for (std::size_t sample = 0U; sample <= subdivisions; ++sample) {
    if (stop_token.stop_requested()) {
      return LeggedSweepResult{
          .valid = false,
          .canceled = true,
          .reachable_body_z_m = {},
          .sample_count = sample_count,
          .reason_code = "REQUEST_CANCELED",
      };
    }
    ++sample_count;
    const double ratio = static_cast<double>(sample) /
        static_cast<double>(subdivisions);
    const double center_x = source.position_m.x +
        ratio * (target.position_m.x - source.position_m.x);
    const double center_y = source.position_m.y +
        ratio * (target.position_m.y - source.position_m.y);
    const double yaw = source.yaw_rad + ratio * yaw_delta;
    const double cosine = std::cos(yaw);
    const double sine = std::sin(yaw);
    const double extent_x =
        std::abs(cosine) * capability.body_half_extent_m.x +
        std::abs(sine) * capability.body_half_extent_m.y;
    const double extent_y =
        std::abs(sine) * capability.body_half_extent_m.x +
        std::abs(cosine) * capability.body_half_extent_m.y;
    const auto minimum_cell = map.PositionToCell(Vec2{
        .x = center_x - extent_x,
        .y = center_y - extent_y,
    });
    const auto maximum_cell = map.PositionToCell(Vec2{
        .x = center_x + extent_x,
        .y = center_y + extent_y,
    });
    if (!minimum_cell.has_value() || !maximum_cell.has_value()) {
      return SweepFailure("LEGGED_BODY_SWEEP_OUTSIDE_MAP", sample_count);
    }
    for (std::int32_t y = minimum_cell->y; y <= maximum_cell->y; ++y) {
      for (std::int32_t x = minimum_cell->x; x <= maximum_cell->x; ++x) {
        const LeggedTerrainEvaluation terrain = EvaluateLeggedTerrainCell(
            projection, capability, shared::GridCell{.x = x, .y = y},
            stop_token);
        if (terrain.canceled) {
          return LeggedSweepResult{
              .valid = false,
              .canceled = true,
              .reachable_body_z_m = {},
              .sample_count = sample_count,
              .reason_code = "REQUEST_CANCELED",
          };
        }
        if (!terrain.hard_feasible) {
          return SweepFailure("LEGGED_BODY_SWEEP_COLLISION", sample_count);
        }
      }
    }
    const auto center_cell = map.PositionToCell(
        Vec2{.x = center_x, .y = center_y});
    if (!center_cell.has_value()) {
      return SweepFailure("LEGGED_BODY_SWEEP_OUTSIDE_MAP", sample_count);
    }
    const LeggedTerrainEvaluation terrain = EvaluateLeggedTerrainCell(
        projection, capability, *center_cell, stop_token);
    if (!terrain.hard_feasible) {
      return SweepFailure("LEGGED_TERRAIN_SWEEP_INVALID", sample_count);
    }
    if (sample > 0U) {
      const Interval expanded{
          .lower = reachable.lower +
              capability.vertical_speed_mps.lower * sample_duration_s,
          .upper = reachable.upper +
              capability.vertical_speed_mps.upper * sample_duration_s,
      };
      const auto intersection =
          IntersectIntervals(expanded, terrain.body_height_m);
      if (!intersection.has_value()) {
        return SweepFailure("LEGGED_HEIGHT_INTERVAL_EMPTY", sample_count);
      }
      reachable = *intersection;
    } else {
      const auto intersection =
          IntersectIntervals(reachable, terrain.body_height_m);
      if (!intersection.has_value()) {
        return SweepFailure("LEGGED_START_HEIGHT_INTERVAL_EMPTY", sample_count);
      }
      reachable = *intersection;
    }
  }
  return LeggedSweepResult{
      .valid = true,
      .canceled = false,
      .reachable_body_z_m = reachable,
      .sample_count = sample_count,
      .reason_code = "LEGGED_SWEEP_VALID",
  };
}

}  // namespace lunar::planning::legged

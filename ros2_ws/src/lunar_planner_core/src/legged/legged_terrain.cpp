#include "legged/legged_terrain.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <ranges>
#include <string>
#include <utility>
#include <vector>

#include "shared/terrain_checks.hpp"

namespace lunar::planning::legged {
namespace {

constexpr double kComparisonTolerance = 1.0e-9;
constexpr double kAngularTolerance = 1.0e-6;

[[nodiscard]] bool IsFinite(const LeggedPose& pose) noexcept {
  return std::isfinite(pose.position_m.x) &&
      std::isfinite(pose.position_m.y) &&
      std::isfinite(pose.position_m.z) && std::isfinite(pose.yaw_rad);
}

[[nodiscard]] bool PositiveExtent(const Vec3 extent) noexcept {
  return std::isfinite(extent.x) && std::isfinite(extent.y) &&
      std::isfinite(extent.z) && extent.x > 0.0 && extent.y > 0.0 &&
      extent.z > 0.0;
}

void AddReason(LeggedTerrainEvaluation& result, std::string reason) {
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
  return PositiveExtent(capability.body_extent_m) &&
      std::isfinite(capability.maximum_slope_rad) &&
      capability.maximum_slope_rad > 0.0 &&
      std::isfinite(capability.maximum_step_height_m) &&
      capability.maximum_step_height_m >= 0.0 &&
      std::isfinite(capability.maximum_gap_width_m) &&
      capability.maximum_gap_width_m >= 0.0 &&
      std::isfinite(capability.minimum_body_clearance_m) &&
      capability.minimum_body_clearance_m >= 0.0 &&
      std::isfinite(capability.step_vertical_rate_mps) &&
      capability.step_vertical_rate_mps > 0.0 &&
      ValidInterval(capability.body_height_m) &&
      valid_velocity(capability.forward_speed_mps) &&
      valid_velocity(capability.lateral_speed_mps) &&
      valid_velocity(capability.yaw_rate_radps) &&
      std::isfinite(capability.maximum_linear_acceleration_mps2) &&
      capability.maximum_linear_acceleration_mps2 > 0.0 &&
      std::isfinite(capability.maximum_yaw_acceleration_radps2) &&
      capability.maximum_yaw_acceleration_radps2 > 0.0;
}

[[nodiscard]] bool RectangleIntersectsCell(
    const Vec2 center, const double yaw, const double half_length,
    const double half_width, const shared::MapSnapshot& map,
    const shared::GridCell cell) noexcept {
  const Vec3 cell_center_3 = map.CellCenter(cell);
  const Vec2 delta{
      .x = cell_center_3.x - center.x,
      .y = cell_center_3.y - center.y,
  };
  const Vec2 longitudinal{.x = std::cos(yaw), .y = std::sin(yaw)};
  const Vec2 lateral{.x = -longitudinal.y, .y = longitudinal.x};
  const double cell_half = map.resolution_m() / 2.0;
  const auto projection_radius = [cell_half](const Vec2 axis) {
    return cell_half * (std::abs(axis.x) + std::abs(axis.y));
  };
  if (std::abs(delta.x * longitudinal.x + delta.y * longitudinal.y) >
      half_length + projection_radius(longitudinal) + kComparisonTolerance) {
    return false;
  }
  if (std::abs(delta.x * lateral.x + delta.y * lateral.y) >
      half_width + projection_radius(lateral) + kComparisonTolerance) {
    return false;
  }
  if (std::abs(delta.x) >
      cell_half + half_length * std::abs(longitudinal.x) +
          half_width * std::abs(lateral.x) + kComparisonTolerance) {
    return false;
  }
  return std::abs(delta.y) <=
      cell_half + half_length * std::abs(longitudinal.y) +
          half_width * std::abs(lateral.y) + kComparisonTolerance;
}

[[nodiscard]] std::optional<std::pair<double, double>> SegmentCellInterval(
    const Vec2 source, const Vec2 target, const shared::MapSnapshot& map,
    const shared::GridCell cell) noexcept {
  const double resolution = map.resolution_m();
  const double inset = std::min(1.0e-10, resolution * 1.0e-8);
  const double minimum_x =
      map.origin_m().x + static_cast<double>(cell.x) * resolution + inset;
  const double maximum_x = minimum_x + resolution - 2.0 * inset;
  const double minimum_y =
      map.origin_m().y + static_cast<double>(cell.y) * resolution + inset;
  const double maximum_y = minimum_y + resolution - 2.0 * inset;
  const double dx = target.x - source.x;
  const double dy = target.y - source.y;
  double begin = 0.0;
  double end = 1.0;
  const auto clip = [&](const double p, const double q) {
    if (std::abs(p) <= kComparisonTolerance) {
      return q >= 0.0;
    }
    const double ratio = q / p;
    if (p < 0.0) {
      if (ratio > end) {
        return false;
      }
      begin = std::max(begin, ratio);
    } else {
      if (ratio < begin) {
        return false;
      }
      end = std::min(end, ratio);
    }
    return true;
  };
  if (!clip(-dx, source.x - minimum_x) ||
      !clip(dx, maximum_x - source.x) ||
      !clip(-dy, source.y - minimum_y) ||
      !clip(dy, maximum_y - source.y) || begin > end) {
    return std::nullopt;
  }
  return std::pair{std::clamp(begin, 0.0, 1.0),
                   std::clamp(end, 0.0, 1.0)};
}

[[nodiscard]] double MaximumUnsupportedSpan(
    const Vec2 source, const Vec2 target,
    const shared::MapSnapshot& map) {
  const double length = std::hypot(target.x - source.x, target.y - source.y);
  if (length <= kComparisonTolerance) {
    return 0.0;
  }
  const auto source_cell = map.PositionToCell(source);
  const auto target_cell = map.PositionToCell(target);
  if (!source_cell.has_value() || !target_cell.has_value()) {
    return std::numeric_limits<double>::infinity();
  }
  std::vector<std::pair<double, double>> intervals;
  const auto valid = map.ByteLayer("valid_mask");
  const std::int32_t minimum_x = std::max(
      std::int32_t{0}, std::min(source_cell->x, target_cell->x) - 1);
  const std::int32_t maximum_x = std::min(
      static_cast<std::int32_t>(map.width()) - 1,
      std::max(source_cell->x, target_cell->x) + 1);
  const std::int32_t minimum_y = std::max(
      std::int32_t{0}, std::min(source_cell->y, target_cell->y) - 1);
  const std::int32_t maximum_y = std::min(
      static_cast<std::int32_t>(map.height()) - 1,
      std::max(source_cell->y, target_cell->y) + 1);
  for (std::int32_t y = minimum_y; y <= maximum_y; ++y) {
    for (std::int32_t x = minimum_x; x <= maximum_x; ++x) {
      const shared::GridCell cell{.x = x, .y = y};
      if (valid[map.Index(cell)] != 0U) {
        continue;
      }
      if (const auto interval = SegmentCellInterval(source, target, map, cell)) {
        intervals.push_back(*interval);
      }
    }
  }
  if (intervals.empty()) {
    return 0.0;
  }
  std::ranges::sort(intervals);
  double maximum_span = 0.0;
  double begin = intervals.front().first;
  double end = intervals.front().second;
  for (std::size_t index = 1U; index < intervals.size(); ++index) {
    if (intervals[index].first <= end + kComparisonTolerance) {
      end = std::max(end, intervals[index].second);
      continue;
    }
    maximum_span = std::max(maximum_span, (end - begin) * length);
    begin = intervals[index].first;
    end = intervals[index].second;
  }
  return std::max(maximum_span, (end - begin) * length);
}

[[nodiscard]] bool SlopeOrOrdinaryStep(
    const LeggedTerrainEvaluation& terrain,
    const LeggedCapability& capability) noexcept {
  return terrain.rejection_reasons.size() == 1U &&
      terrain.rejection_reasons.front() == "LEGGED_SLOPE_LIMIT" &&
      std::isfinite(terrain.maximum_neighbor_step_m) &&
      terrain.maximum_neighbor_step_m <=
          capability.maximum_step_height_m + kComparisonTolerance;
}

}  // namespace

LeggedTerrainEvaluation EvaluateLeggedTerrainCell(
    const shared::SafeProjection& projection,
    const LeggedCapability& capability, const shared::GridCell cell,
    const std::stop_token stop_token) {
  LeggedTerrainEvaluation result;
  if (stop_token.stop_requested()) {
    result.canceled = true;
    result.rejection_reasons.emplace_back("REQUEST_CANCELED");
    return result;
  }
  if (projection.source_map() == nullptr || !ValidCapability(capability) ||
      !projection.InBounds(cell)) {
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
      result.slope_rad > capability.maximum_slope_rad + kAngularTolerance ||
      result.slope_rad >
          projection.maximum_slope_rad() + kAngularTolerance) {
    AddReason(result, "LEGGED_SLOPE_LIMIT");
  }
  if (!ValidInterval(result.body_height_m)) {
    AddReason(result, "LEGGED_BODY_HEIGHT_INTERVAL_INVALID");
  }
  if (!projection.IntrinsicFeasible(cell) &&
      result.rejection_reasons.empty()) {
    AddReason(result, "LEGGED_TERRAIN_HARD_INFEASIBLE");
  }
  result.hard_feasible = result.rejection_reasons.empty();
  return result;
}

LeggedTerrainGrid::LeggedTerrainGrid(
    const shared::SafeProjection& projection,
    const LeggedCapability& capability,
    const std::stop_token stop_token)
    : projection_(&projection), source_map_(projection.source_map().get()),
      capability_(&capability) {
  if (source_map_ == nullptr || !ValidCapability(capability)) {
    return;
  }
  cells_.reserve(source_map_->cell_count());
  for (std::size_t y = 0U; y < source_map_->height(); ++y) {
    for (std::size_t x = 0U; x < source_map_->width(); ++x) {
      if (stop_token.stop_requested()) {
        canceled_ = true;
        return;
      }
      LeggedTerrainEvaluation evaluation = EvaluateLeggedTerrainCell(
          projection, capability,
          shared::GridCell{.x = static_cast<std::int32_t>(x),
                           .y = static_cast<std::int32_t>(y)},
          stop_token);
      if (evaluation.canceled) {
        canceled_ = true;
        return;
      }
      cells_.push_back(std::move(evaluation));
    }
  }
}

bool LeggedTerrainGrid::ok() const noexcept {
  return !canceled_ && source_map_ != nullptr &&
      cells_.size() == source_map_->cell_count();
}

bool LeggedTerrainGrid::canceled() const noexcept { return canceled_; }

bool LeggedTerrainGrid::Matches(
    const shared::SafeProjection& projection,
    const LeggedCapability& capability) const noexcept {
  return ok() && projection_ == &projection &&
      source_map_ == projection.source_map().get() && capability_ == &capability;
}

const LeggedTerrainEvaluation* LeggedTerrainGrid::Find(
    const shared::GridCell cell) const noexcept {
  if (!ok() || !source_map_->InBounds(cell)) {
    return nullptr;
  }
  return &cells_[source_map_->Index(cell)];
}

LeggedSweepResult ValidateLeggedBodySweep(
    const LeggedPose& source, const LeggedPose& target,
    const Interval& source_body_z_m,
    const shared::SafeProjection& projection,
    const LeggedCapability& capability,
    const hierarchical::LocalSearchDomain& search_domain,
    const std::stop_token stop_token,
    const LeggedTerrainGrid* const terrain_grid) {
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
      !ValidInterval(source_body_z_m) || projection.source_map() == nullptr ||
      !ValidCapability(capability)) {
    return SweepFailure("LEGGED_SWEEP_REQUEST_INVALID");
  }

  const shared::MapSnapshot& map = *projection.source_map();
  if (search_domain.width() != map.width() ||
      search_domain.height() != map.height()) {
    return SweepFailure("LEGGED_SWEEP_REQUEST_INVALID");
  }
  if (terrain_grid != nullptr &&
      !terrain_grid->Matches(projection, capability)) {
    return SweepFailure("LEGGED_SWEEP_REQUEST_INVALID");
  }
  const Vec2 source_xy{.x = source.position_m.x, .y = source.position_m.y};
  const Vec2 target_xy{.x = target.position_m.x, .y = target.position_m.y};
  const auto source_cell = map.PositionToCell(source_xy);
  const auto target_cell = map.PositionToCell(target_xy);
  if (!source_cell.has_value() || !target_cell.has_value() ||
      !projection.Known(*source_cell) || !projection.Known(*target_cell)) {
    return SweepFailure("LEGGED_EDGE_ENDPOINT_UNKNOWN");
  }
  const double unsupported_span = MaximumUnsupportedSpan(source_xy, target_xy, map);
  if (!std::isfinite(unsupported_span) ||
      unsupported_span > capability.maximum_gap_width_m +
          kComparisonTolerance) {
    return SweepFailure("LEGGED_GAP_WIDTH_LIMIT");
  }

  const double half_length =
      capability.body_extent_m.x / 2.0 + capability.minimum_body_clearance_m;
  const double half_width =
      capability.body_extent_m.y / 2.0 + capability.minimum_body_clearance_m;
  const double support_radius = std::hypot(half_length, half_width);
  const double translation =
      std::hypot(target.position_m.x - source.position_m.x,
                 target.position_m.y - source.position_m.y);
  const double yaw_delta = ShortestYawDelta(source.yaw_rad, target.yaw_rad);
  const double swept_distance =
      translation + std::abs(yaw_delta) * support_radius;
  const double maximum_step =
      std::min(0.05, map.resolution_m() * 0.25);
  const std::size_t subdivisions = std::max<std::size_t>(
      1U, static_cast<std::size_t>(std::ceil(swept_distance / maximum_step)));

  Interval reachable = source_body_z_m;
  std::optional<double> previous_known_elevation;
  std::size_t sample_count = 0U;
  const auto obstacles = map.ByteLayer("obstacle");
  const auto forbidden = map.ByteLayer("forbidden");
  std::optional<LeggedTerrainEvaluation> direct_evaluation;
  const auto evaluate_terrain = [&](const shared::GridCell cell)
      -> const LeggedTerrainEvaluation* {
    if (terrain_grid != nullptr && !stop_token.stop_requested()) {
      return terrain_grid->Find(cell);
    }
    direct_evaluation = EvaluateLeggedTerrainCell(
        projection, capability, cell, stop_token);
    return &*direct_evaluation;
  };
  bool center_left_search_domain = false;
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
    const double ratio =
        static_cast<double>(sample) / static_cast<double>(subdivisions);
    const Vec2 center{
        .x = source.position_m.x +
            ratio * (target.position_m.x - source.position_m.x),
        .y = source.position_m.y +
            ratio * (target.position_m.y - source.position_m.y),
    };
    const double yaw = source.yaw_rad + ratio * yaw_delta;
    const double cosine = std::cos(yaw);
    const double sine = std::sin(yaw);
    const double aabb_x =
        std::abs(cosine) * half_length + std::abs(sine) * half_width;
    const double aabb_y =
        std::abs(sine) * half_length + std::abs(cosine) * half_width;
    const auto minimum_cell = map.PositionToCell(
        Vec2{.x = center.x - aabb_x, .y = center.y - aabb_y});
    const auto maximum_cell = map.PositionToCell(
        Vec2{.x = center.x + aabb_x, .y = center.y + aabb_y});
    if (!minimum_cell.has_value() || !maximum_cell.has_value()) {
      return SweepFailure("LEGGED_BODY_SWEEP_OUTSIDE_MAP", sample_count);
    }
    for (std::int32_t y = minimum_cell->y; y <= maximum_cell->y; ++y) {
      for (std::int32_t x = minimum_cell->x; x <= maximum_cell->x; ++x) {
        const shared::GridCell cell{.x = x, .y = y};
        if (!RectangleIntersectsCell(
                center, yaw, half_length, half_width, map, cell)) {
          continue;
        }
        const std::size_t index = map.Index(cell);
        if (obstacles[index] != 0U || forbidden[index] != 0U) {
          return SweepFailure("LEGGED_BODY_SWEEP_COLLISION", sample_count);
        }
        if (!projection.Known(cell)) {
          if (translation <= kComparisonTolerance ||
              unsupported_span > capability.maximum_gap_width_m +
                  kComparisonTolerance) {
            return SweepFailure("LEGGED_BODY_SWEEP_UNKNOWN", sample_count);
          }
          continue;
        }
        const LeggedTerrainEvaluation* const body_terrain =
            evaluate_terrain(cell);
        if (body_terrain == nullptr || body_terrain->canceled) {
          return LeggedSweepResult{
              .valid = false,
              .canceled = true,
              .reachable_body_z_m = {},
              .sample_count = sample_count,
              .reason_code = "REQUEST_CANCELED",
          };
        }
        if (!body_terrain->hard_feasible &&
            !SlopeOrOrdinaryStep(*body_terrain, capability)) {
          if (std::ranges::find(
                  body_terrain->rejection_reasons,
                  "LEGGED_SLOPE_LIMIT") !=
                  body_terrain->rejection_reasons.end() &&
              body_terrain->maximum_neighbor_step_m >
                  capability.maximum_step_height_m + kComparisonTolerance) {
            return SweepFailure("LEGGED_STEP_HEIGHT_LIMIT", sample_count);
          }
          return SweepFailure("LEGGED_BODY_SWEEP_COLLISION", sample_count);
        }
      }
    }

    const auto center_cell = map.PositionToCell(center);
    if (!center_cell.has_value()) {
      return SweepFailure("LEGGED_BODY_SWEEP_OUTSIDE_MAP", sample_count);
    }
    center_left_search_domain = center_left_search_domain ||
        !search_domain.Contains(*center_cell);
    if (!projection.Known(*center_cell)) {
      continue;
    }
    const LeggedTerrainEvaluation* const terrain =
        evaluate_terrain(*center_cell);
    if (terrain == nullptr || terrain->canceled) {
      return LeggedSweepResult{
          .valid = false,
          .canceled = true,
          .reachable_body_z_m = {},
          .sample_count = sample_count,
          .reason_code = "REQUEST_CANCELED",
      };
    }
    if (!terrain->hard_feasible &&
        !SlopeOrOrdinaryStep(*terrain, capability)) {
      if (std::ranges::find(
              terrain->rejection_reasons, "LEGGED_SLOPE_LIMIT") !=
              terrain->rejection_reasons.end() &&
          terrain->maximum_neighbor_step_m >
              capability.maximum_step_height_m + kComparisonTolerance) {
        return SweepFailure("LEGGED_STEP_HEIGHT_LIMIT", sample_count);
      }
      return SweepFailure("LEGGED_TERRAIN_SWEEP_INVALID", sample_count);
    }
    if (previous_known_elevation.has_value() &&
        std::abs(terrain->elevation_m - *previous_known_elevation) >
            capability.maximum_step_height_m + kComparisonTolerance) {
      return SweepFailure("LEGGED_STEP_HEIGHT_LIMIT", sample_count);
    }
    previous_known_elevation = terrain->elevation_m;
    if (sample == 0U) {
      const auto start = IntersectIntervals(reachable, terrain->body_height_m);
      if (!start.has_value()) {
        return SweepFailure("LEGGED_START_HEIGHT_INTERVAL_EMPTY", sample_count);
      }
      reachable = *start;
    } else {
      reachable = terrain->body_height_m;
    }
  }

  if (center_left_search_domain) {
    return SweepFailure(
        "LEGGED_BODY_SWEEP_OUTSIDE_SEARCH_DOMAIN", sample_count);
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

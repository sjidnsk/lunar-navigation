#include "shared/safe_projection.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <numbers>
#include <queue>
#include <utility>
#include <vector>

#include "shared/terrain_checks.hpp"

namespace lunar::planning::shared {
namespace {

[[nodiscard]] SafeProjectionBuildResult Failure(std::string reason_code) {
  return SafeProjectionBuildResult{
      .projection = std::nullopt,
      .reason_code = std::move(reason_code),
  };
}

[[nodiscard]] GridCell CellFromIndex(
    const MapSnapshot& map, const std::size_t index) noexcept {
  return GridCell{
      .x = static_cast<std::int32_t>(index % map.width()),
      .y = static_cast<std::int32_t>(index / map.width()),
  };
}

[[nodiscard]] bool ComputeClearance(
    const MapSnapshot& map,
    const std::vector<std::uint8_t>& hazard_mask,
    std::vector<float>& clearance_m,
    const SafeProjectionClearanceMargins clearance_margins,
    const std::stop_token stop_token) {
  using QueueEntry = std::pair<double, std::size_t>;
  std::priority_queue<
      QueueEntry, std::vector<QueueEntry>, std::greater<QueueEntry>>
      open;
  const double infinity = std::numeric_limits<double>::infinity();
  std::vector<double> distance(map.cell_count(), infinity);
  for (std::size_t index = 0U; index < map.cell_count(); ++index) {
    if (hazard_mask[index] != 0U) {
      distance[index] = 0.0;
      open.emplace(0.0, index);
    }
  }

  constexpr std::array<std::int32_t, 8> kDx{-1, 0, 1, -1, 1, -1, 0, 1};
  constexpr std::array<std::int32_t, 8> kDy{-1, -1, -1, 0, 0, 1, 1, 1};
  while (!open.empty()) {
    if (stop_token.stop_requested()) {
      return false;
    }
    const auto [current_distance, current_index] = open.top();
    open.pop();
    if (current_distance > distance[current_index]) {
      continue;
    }
    const GridCell current = CellFromIndex(map, current_index);
    for (std::size_t neighbor = 0U; neighbor < kDx.size(); ++neighbor) {
      const GridCell next{
          .x = current.x + kDx[neighbor],
          .y = current.y + kDy[neighbor],
      };
      if (!map.InBounds(next)) {
        continue;
      }
      const double edge_distance = map.resolution_m() *
          ((kDx[neighbor] != 0 && kDy[neighbor] != 0)
               ? std::numbers::sqrt2
               : 1.0);
      const std::size_t next_index = map.Index(next);
      const double candidate = current_distance + edge_distance;
      if (candidate < distance[next_index]) {
        distance[next_index] = candidate;
        open.emplace(candidate, next_index);
      }
    }
  }

  clearance_m.resize(map.cell_count());
  for (std::size_t index = 0U; index < map.cell_count(); ++index) {
    const GridCell cell = CellFromIndex(map, index);
    const double boundary_distance_cells = std::min(
        {static_cast<double>(cell.x) + 0.5,
         static_cast<double>(cell.y) + 0.5,
         static_cast<double>(map.width()) -
             static_cast<double>(cell.x) - 0.5,
         static_cast<double>(map.height()) -
             static_cast<double>(cell.y) - 0.5});
    const double boundary_distance_m =
        boundary_distance_cells * map.resolution_m();
    const double hazard_clearance_m =
        distance[index] - clearance_margins.hazard_m;
    const double boundary_clearance_m =
        boundary_distance_m - clearance_margins.boundary_m;
    clearance_m[index] = static_cast<float>(
        std::min(hazard_clearance_m, boundary_clearance_m));
  }
  return true;
}

[[nodiscard]] bool LabelConnectedComponents(
    const MapSnapshot& map,
    const std::vector<std::uint8_t>& hard_feasible_mask,
    std::vector<std::int32_t>& connected_component,
    const std::stop_token stop_token) {
  connected_component.assign(map.cell_count(), -1);
  constexpr std::array<std::int32_t, 4> kDx{-1, 0, 1, 0};
  constexpr std::array<std::int32_t, 4> kDy{0, -1, 0, 1};
  std::queue<GridCell> open;
  std::int32_t component = 0;
  for (std::size_t start_index = 0U;
       start_index < map.cell_count(); ++start_index) {
    if (stop_token.stop_requested()) {
      return false;
    }
    if (hard_feasible_mask[start_index] == 0U ||
        connected_component[start_index] >= 0) {
      continue;
    }
    connected_component[start_index] = component;
    open.push(CellFromIndex(map, start_index));
    while (!open.empty()) {
      if (stop_token.stop_requested()) {
        return false;
      }
      const GridCell current = open.front();
      open.pop();
      for (std::size_t neighbor = 0U; neighbor < kDx.size(); ++neighbor) {
        const GridCell next{
            .x = current.x + kDx[neighbor],
            .y = current.y + kDy[neighbor],
        };
        if (!map.InBounds(next)) {
          continue;
        }
        const std::size_t next_index = map.Index(next);
        if (hard_feasible_mask[next_index] == 0U ||
            connected_component[next_index] >= 0) {
          continue;
        }
        connected_component[next_index] = component;
        open.push(next);
      }
    }
    ++component;
  }
  return true;
}

template <class T>
[[nodiscard]] T ValueAt(
    const SafeProjection& projection, const GridCell cell,
    const std::vector<T>& values, const T fallback) noexcept {
  const auto& map = projection.source_map();
  if (map == nullptr || !map->InBounds(cell)) {
    return fallback;
  }
  const std::size_t index = map->Index(cell);
  return index < values.size() ? values[index] : fallback;
}

}  // namespace

SafeProjectionBuildResult BuildSafeProjection(
    std::shared_ptr<const MapSnapshot> map,
    const PlatformCapability& capability,
    const MapSafetyConfig& config,
    const std::stop_token stop_token,
    const SafeProjectionClearanceMargins clearance_margins) {
  if (stop_token.stop_requested()) {
    return Failure("REQUEST_CANCELED");
  }
  if (map == nullptr) {
    return Failure("MAP_SNAPSHOT_REQUIRED");
  }
  if (!std::isfinite(clearance_margins.hazard_m) ||
      clearance_margins.hazard_m < 0.0 ||
      !std::isfinite(clearance_margins.boundary_m) ||
      clearance_margins.boundary_m < 0.0) {
    return Failure("SAFE_PROJECTION_CLEARANCE_MARGIN_INVALID");
  }
  const TerrainLimitsResult resolved =
      ResolveTerrainLimits(capability, config);
  if (!resolved.ok()) {
    return Failure(resolved.reason_code);
  }
  const TerrainLimits& limits = *resolved.limits;

  SafeProjection projection;
  projection.source_map_ = std::move(map);
  projection.platform_type_ = limits.platform_type;
  projection.maximum_slope_rad_ = limits.maximum_slope_rad;
  const std::size_t count = projection.source_map_->cell_count();
  projection.known_mask_.assign(
      projection.source_map_->ByteLayer("valid_mask").begin(),
      projection.source_map_->ByteLayer("valid_mask").end());
  projection.hard_feasible_mask_.assign(count, 0U);
  projection.slope_rad_.assign(count, 0.0F);
  projection.roughness_m_.assign(count, 0.0F);
  projection.traversal_cost_.assign(
      count, std::numeric_limits<float>::infinity());

  std::vector<std::uint8_t> hazard_mask(count, 0U);
  for (std::size_t index = 0U; index < count; ++index) {
    if (stop_token.stop_requested()) {
      return Failure("REQUEST_CANCELED");
    }
    const GridCell cell = CellFromIndex(*projection.source_map_, index);
    const TerrainCellEvaluation intrinsic = EvaluateTerrainCell(
        *projection.source_map_, cell, limits, config,
        std::numeric_limits<double>::infinity());
    hazard_mask[index] = static_cast<std::uint8_t>(!intrinsic.hard_feasible);
  }
  if (!ComputeClearance(
          *projection.source_map_, hazard_mask,
          projection.clearance_m_, clearance_margins, stop_token)) {
    return Failure("REQUEST_CANCELED");
  }

  for (std::size_t index = 0U; index < count; ++index) {
    if (stop_token.stop_requested()) {
      return Failure("REQUEST_CANCELED");
    }
    const GridCell cell = CellFromIndex(*projection.source_map_, index);
    const TerrainCellEvaluation evaluation = EvaluateTerrainCell(
        *projection.source_map_, cell, limits, config,
        static_cast<double>(projection.clearance_m_[index]));
    projection.slope_rad_[index] =
        static_cast<float>(evaluation.slope_rad);
    projection.roughness_m_[index] =
        static_cast<float>(evaluation.roughness_m);
    if (!evaluation.hard_feasible) {
      continue;
    }
    projection.hard_feasible_mask_[index] = 1U;
    projection.traversal_cost_[index] = static_cast<float>(
        projection.source_map_->resolution_m() /
            evaluation.conservative_speed_mps +
        evaluation.roughness_m + evaluation.slope_rad * evaluation.slope_rad);
  }

  if (!LabelConnectedComponents(
          *projection.source_map_, projection.hard_feasible_mask_,
          projection.connected_component_, stop_token)) {
    return Failure("REQUEST_CANCELED");
  }
  return SafeProjectionBuildResult{
      .projection = std::move(projection),
      .reason_code = {},
  };
}

const std::shared_ptr<const MapSnapshot>& SafeProjection::source_map()
    const noexcept {
  return source_map_;
}

bool SafeProjection::InBounds(const GridCell cell) const noexcept {
  return source_map_ != nullptr && source_map_->InBounds(cell);
}

bool SafeProjection::Known(const GridCell cell) const noexcept {
  return ValueAt(*this, cell, known_mask_, std::uint8_t{0}) != 0U;
}

bool SafeProjection::HardFeasible(const GridCell cell) const noexcept {
  return ValueAt(*this, cell, hard_feasible_mask_, std::uint8_t{0}) != 0U;
}

float SafeProjection::ClearanceMeters(const GridCell cell) const noexcept {
  return ValueAt(*this, cell, clearance_m_, 0.0F);
}

float SafeProjection::SlopeRadians(const GridCell cell) const noexcept {
  return ValueAt(
      *this, cell, slope_rad_, std::numeric_limits<float>::infinity());
}

float SafeProjection::RoughnessMeters(const GridCell cell) const noexcept {
  return ValueAt(
      *this, cell, roughness_m_, std::numeric_limits<float>::infinity());
}

float SafeProjection::TraversalCost(const GridCell cell) const noexcept {
  return ValueAt(
      *this, cell, traversal_cost_, std::numeric_limits<float>::infinity());
}

std::int32_t SafeProjection::ConnectedComponent(
    const GridCell cell) const noexcept {
  return ValueAt(*this, cell, connected_component_, std::int32_t{-1});
}

double SafeProjection::maximum_slope_rad() const noexcept {
  return maximum_slope_rad_;
}

PlatformType SafeProjection::platform_type() const noexcept {
  return platform_type_;
}

}  // namespace lunar::planning::shared

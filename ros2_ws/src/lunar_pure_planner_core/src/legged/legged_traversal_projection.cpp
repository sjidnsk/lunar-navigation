#include "legged/legged_traversal_projection.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "shared/controlled_work.hpp"

namespace lunar::pure_planning::legged {
namespace {

constexpr double kTolerance = 1.0e-9;

[[nodiscard]] bool StepFeasible(
    const shared::GridCell cell,
    const shared::LocalTerrainProjection& terrain,
    const LeggedCapability& capability) noexcept {
  const auto elevation = terrain.map->FloatLayer("elevation");
  const double center = elevation[terrain.map->Index(cell)];
  if (!std::isfinite(center)) {
    return false;
  }
  for (std::int32_t dy = -1; dy <= 1; ++dy) {
    for (std::int32_t dx = -1; dx <= 1; ++dx) {
      const shared::GridCell neighbor{.x = cell.x + dx, .y = cell.y + dy};
      if (!terrain.map->InBounds(neighbor)) {
        continue;
      }
      const std::size_t neighbor_index = terrain.map->Index(neighbor);
      if (terrain.free_with_height[neighbor_index] == 0U) {
        continue;
      }
      const double neighbor_height = elevation[neighbor_index];
      if (!std::isfinite(neighbor_height) ||
          std::abs(neighbor_height - center) >
              capability.maximum_step_height_m + kTolerance) {
        return false;
      }
    }
  }
  return true;
}

[[nodiscard]] LeggedTraversalProjectionBuildResult Failure(
    std::string reason_code) {
  return {.reason_code = std::move(reason_code)};
}

}  // namespace

bool LeggedTraversalProjection::AllCellsTraversable(
    const shared::GridCell minimum,
    const shared::GridCell maximum) const noexcept {
  if (terrain == nullptr || terrain->map == nullptr ||
      !terrain->map->InBounds(minimum) ||
      !terrain->map->InBounds(maximum) || minimum.x > maximum.x ||
      minimum.y > maximum.y) {
    return false;
  }
  const std::size_t stride = terrain->map->width() + 1U;
  const std::size_t left = static_cast<std::size_t>(minimum.x);
  const std::size_t top = static_cast<std::size_t>(minimum.y);
  const std::size_t right = static_cast<std::size_t>(maximum.x) + 1U;
  const std::size_t bottom = static_cast<std::size_t>(maximum.y) + 1U;
  const std::size_t unsafe =
      hard_infeasible_prefix_sum[bottom * stride + right] -
      hard_infeasible_prefix_sum[top * stride + right] -
      hard_infeasible_prefix_sum[bottom * stride + left] +
      hard_infeasible_prefix_sum[top * stride + left];
  return unsafe == 0U;
}

LeggedTraversalProjectionBuildResult BuildLeggedTraversalProjection(
    std::shared_ptr<const shared::LocalTerrainProjection> terrain,
    const LeggedCapability& capability,
    const SearchControl& control) {
  if (const auto stopped = shared::StopReason(control); stopped.has_value()) {
    return Failure(std::string{*stopped});
  }
  if (terrain == nullptr || terrain->map == nullptr ||
      !std::isfinite(capability.maximum_slope_rad) ||
      capability.maximum_slope_rad < 0.0 ||
      !std::isfinite(capability.maximum_step_height_m) ||
      capability.maximum_step_height_m < 0.0) {
    return Failure("INVALID_INPUT");
  }
  const std::size_t count = terrain->map->cell_count();
  if (terrain->free_with_height.size() != count ||
      terrain->slope_rad.size() != count ||
      terrain->roughness_m.size() != count ||
      terrain->clearance_m.size() != count) {
    return Failure("INVALID_INPUT");
  }

  auto value = std::make_shared<LeggedTraversalProjection>();
  value->terrain = std::move(terrain);
  value->hard_feasible.resize(count, 0U);
  value->step_feasible.resize(count, 0U);
  value->slope_rad = value->terrain->slope_rad;
  value->roughness_m = value->terrain->roughness_m;
  value->clearance_m = value->terrain->clearance_m;

  for (std::size_t index = 0U; index < count; ++index) {
    if (shared::ControlCheckDue(index)) {
      if (const auto stopped = shared::StopReason(control);
          stopped.has_value()) {
        return Failure(std::string{*stopped});
      }
    }
    const double slope = value->slope_rad[index];
    value->hard_feasible[index] =
        value->terrain->free_with_height[index] != 0U &&
            std::isfinite(slope) &&
            slope <= capability.maximum_slope_rad + kTolerance
        ? 1U
        : 0U;
    const shared::GridCell cell{
        .x = static_cast<std::int32_t>(index % value->terrain->map->width()),
        .y = static_cast<std::int32_t>(index / value->terrain->map->width()),
    };
    value->step_feasible[index] =
        StepFeasible(cell, *value->terrain, capability)
        ? 1U
        : 0U;
  }

  const std::size_t width = value->terrain->map->width();
  const std::size_t height = value->terrain->map->height();
  const std::size_t stride = width + 1U;
  value->hard_infeasible_prefix_sum.assign(
      stride * (height + 1U), 0U);
  for (std::size_t y = 0U; y < height; ++y) {
    for (std::size_t x = 0U; x < width; ++x) {
      const std::size_t index = y * width + x;
      const std::size_t prefix = (y + 1U) * stride + x + 1U;
      const std::size_t unsafe =
          value->hard_feasible[index] != 0U &&
                  value->step_feasible[index] != 0U
              ? 0U
              : 1U;
      value->hard_infeasible_prefix_sum[prefix] =
          unsafe + value->hard_infeasible_prefix_sum[prefix - 1U] +
          value->hard_infeasible_prefix_sum[prefix - stride] -
          value->hard_infeasible_prefix_sum[prefix - stride - 1U];
    }
  }
  if (const auto stopped = shared::StopReason(control); stopped.has_value()) {
    return Failure(std::string{*stopped});
  }
  return {.value = std::move(value)};
}

}  // namespace lunar::pure_planning::legged

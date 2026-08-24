#include "shared/global_occupancy_projection.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "shared/cell_area_distance_transform.hpp"
#include "shared/controlled_work.hpp"

namespace lunar::pure_planning::shared {
namespace {

[[nodiscard]] bool IsHardHazard(const std::int8_t raw_value,
                                const std::int32_t threshold) noexcept {
  const auto value = static_cast<std::int32_t>(raw_value);
  return value < 0 || value > 100 || value >= threshold;
}

// Unknown and malformed cells are individually non-traversable, but neither
// represents a measured obstacle.  Only valid occupied cells consume vehicle
// clearance around their boundary.
[[nodiscard]] bool IsOccupiedInflationSource(
    const std::int8_t raw_value, const std::int32_t threshold) noexcept {
  const auto value = static_cast<std::int32_t>(raw_value);
  return value >= threshold && value <= 100;
}

}  // namespace

bool GlobalOccupancyProjectionView::Valid() const noexcept {
  return map != nullptr && hard_feasible.size() == map->cell_count() &&
         clearance_m.size() == map->cell_count();
}

bool GlobalOccupancyProjectionView::HardFeasible(
    const GridCell cell) const noexcept {
  return map != nullptr && map->InBounds(cell) &&
         hard_feasible[map->Index(cell)] != 0U;
}

float GlobalOccupancyProjectionView::ClearanceMeters(
    const GridCell cell) const noexcept {
  if (map == nullptr || !map->InBounds(cell)) {
    return 0.0F;
  }
  const std::size_t index = map->Index(cell);
  return index < clearance_m.size() ? clearance_m[index] : 0.0F;
}

GlobalOccupancyProjectionView
GlobalOccupancyProjection::View() const & noexcept {
  return GlobalOccupancyProjectionView{
    .map = source_map_.get(),
    .hard_feasible = hard_feasible_,
    .clearance_m = clearance_m_,
  };
}

GlobalOccupancyProjectionBuildResult BuildGlobalOccupancyProjection(
    std::shared_ptr<const MapSnapshot> map,
    const std::int32_t obstacle_threshold_percent, SearchControl control) {
  if (map == nullptr || obstacle_threshold_percent < 0 ||
      obstacle_threshold_percent > 100) {
    return {.reason_code = "GLOBAL_OCCUPANCY_PROJECTION_INVALID"};
  }
  const auto occupancy = map->Int8Layer("occupancy");
  if (occupancy.size() != map->cell_count()) {
    return {.reason_code = "GLOBAL_OCCUPANCY_PROJECTION_INVALID"};
  }
  if (const auto stopped = StopReason(control); stopped.has_value()) {
    return {.reason_code = std::string{*stopped}};
  }

  GlobalOccupancyProjection projection;
  projection.source_map_ = std::move(map);
  const std::size_t count = projection.source_map_->cell_count();
  if (const auto stopped = ControlledFill(
          &projection.hard_feasible_, count, std::uint8_t{0U}, control);
      stopped.has_value()) {
    return {.reason_code = std::string{*stopped}};
  }

  std::vector<std::uint8_t> occupied_mask(count, 0U);
  for (std::size_t index = 0U; index < count; ++index) {
    if (ControlCheckDue(index)) {
      if (const auto stopped = StopReason(control); stopped.has_value()) {
        return {.reason_code = std::string{*stopped}};
      }
    }
    const bool hard_hazard =
        IsHardHazard(occupancy[index], obstacle_threshold_percent);
    projection.hard_feasible_[index] = static_cast<std::uint8_t>(!hard_hazard);
    occupied_mask[index] = static_cast<std::uint8_t>(
        IsOccupiedInflationSource(occupancy[index],
                                  obstacle_threshold_percent));
  }
  auto clearance = BuildCellAreaClearance(
      projection.source_map_->width(), projection.source_map_->height(),
      projection.source_map_->resolution_m(), occupied_mask, control);
  if (!clearance.ok()) {
    return {.reason_code = std::move(clearance.reason_code)};
  }
  projection.clearance_m_ = std::move(clearance.clearance_m);
  if (const auto stopped = StopReason(control); stopped.has_value()) {
    return {.reason_code = std::string{*stopped}};
  }
  return {
      .projection = std::move(projection),
      .reason_code = {},
  };
}

GlobalOccupancyProjectionBuildResult BuildInflatedGlobalOccupancyProjection(
    std::shared_ptr<const MapSnapshot> map,
    const std::int32_t obstacle_threshold_percent, const double inflation_m,
    SearchControl control) {
  if (!std::isfinite(inflation_m) || inflation_m < 0.0) {
    return {.reason_code = "GLOBAL_OCCUPANCY_PROJECTION_INVALID"};
  }
  auto native = BuildGlobalOccupancyProjection(
      std::move(map), obstacle_threshold_percent, control);
  if (!native.ok()) {
    return native;
  }

  GlobalOccupancyProjection projection = std::move(*native.projection);
  const auto stencil = BuildCellAreaInflationStencil(
      projection.source_map_->resolution_m(), inflation_m);
  const auto occupancy = projection.source_map_->Int8Layer("occupancy");
  std::size_t work{};
  for (std::size_t index = 0U; index < projection.hard_feasible_.size();
       ++index) {
    if (!IsOccupiedInflationSource(occupancy[index],
                                   obstacle_threshold_percent)) {
      continue;
    }
    const std::int64_t hazard_x = static_cast<std::int64_t>(
        index % projection.source_map_->width());
    const std::int64_t hazard_y = static_cast<std::int64_t>(
        index / projection.source_map_->width());
    for (const CellAreaOffset offset : stencil) {
      if (ControlCheckDue(work++)) {
        if (const auto stopped = StopReason(control); stopped.has_value()) {
          return {.reason_code = std::string{*stopped}};
        }
      }
      const std::int64_t x = hazard_x + offset.dx;
      const std::int64_t y = hazard_y + offset.dy;
      if (x < 0 || y < 0 ||
          x >= static_cast<std::int64_t>(projection.source_map_->width()) ||
          y >= static_cast<std::int64_t>(projection.source_map_->height())) {
        continue;
      }
      projection.hard_feasible_[static_cast<std::size_t>(y) *
                                    projection.source_map_->width() +
                                static_cast<std::size_t>(x)] = 0U;
    }
    if (ControlCheckDue(work++)) {
      if (const auto stopped = StopReason(control); stopped.has_value()) {
        return {.reason_code = std::string{*stopped}};
      }
    }
  }
  if (const auto stopped = StopReason(control); stopped.has_value()) {
    return {.reason_code = std::string{*stopped}};
  }
  return {
      .projection = std::move(projection),
      .reason_code = {},
  };
}

}  // namespace lunar::pure_planning::shared

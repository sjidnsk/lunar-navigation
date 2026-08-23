#include "shared/global_occupancy_projection.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "shared/controlled_work.hpp"

namespace lunar::pure_planning::shared {
namespace {

[[nodiscard]] bool IsHazard(const std::int8_t raw_value,
                            const std::int32_t threshold) noexcept {
  const auto value = static_cast<std::int32_t>(raw_value);
  return value < 0 || value > 100 || value >= threshold;
}

// Felzenszwalb-Huttenlocher lower-envelope transform.  Input and output are
// squared distances in cell units; infinite inputs are non-obstacle cells.
void TransformLine(const std::vector<double>& input, std::vector<double>* output) {
  const std::size_t size = input.size();
  output->assign(size, std::numeric_limits<double>::infinity());
  std::vector<std::size_t> sites;
  sites.reserve(size);
  for (std::size_t index = 0U; index < size; ++index) {
    if (std::isfinite(input[index])) sites.push_back(index);
  }
  if (sites.empty()) return;
  std::vector<std::size_t> envelope(sites.size());
  std::vector<double> boundary(sites.size() + 1U);
  std::size_t count = 0U;
  envelope[0] = sites[0];
  boundary[0] = -std::numeric_limits<double>::infinity();
  boundary[1] = std::numeric_limits<double>::infinity();
  for (std::size_t source_index = 1U; source_index < sites.size(); ++source_index) {
    const std::size_t q = sites[source_index];
    double intersection{};
    do {
      const std::size_t v = envelope[count];
      intersection = ((input[q] + static_cast<double>(q) * q) -
                      (input[v] + static_cast<double>(v) * v)) /
                     (2.0 * static_cast<double>(q - v));
      if (intersection <= boundary[count] && count > 0U) --count;
      else break;
    } while (true);
    ++count;
    envelope[count] = q;
    boundary[count] = intersection;
    boundary[count + 1U] = std::numeric_limits<double>::infinity();
  }
  std::size_t segment{};
  for (std::size_t x = 0U; x < size; ++x) {
    while (segment + 1U <= count && boundary[segment + 1U] < static_cast<double>(x)) {
      ++segment;
    }
    const double delta = static_cast<double>(x) - envelope[segment];
    (*output)[x] = delta * delta + input[envelope[segment]];
  }
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

  const double infinity = std::numeric_limits<double>::infinity();
  std::vector<double> squared_distance(count, infinity);
  for (std::size_t index = 0U; index < count; ++index) {
    if (ControlCheckDue(index)) {
      if (const auto stopped = StopReason(control); stopped.has_value()) {
        return {.reason_code = std::string{*stopped}};
      }
    }
    const bool hazard = IsHazard(occupancy[index], obstacle_threshold_percent);
    projection.hard_feasible_[index] = static_cast<std::uint8_t>(!hazard);
    if (hazard) squared_distance[index] = 0.0;
  }
  std::vector<double> temporary(count, infinity);
  std::vector<double> line;
  std::vector<double> transformed;
  line.resize(projection.source_map_->width());
  for (std::size_t y = 0U; y < projection.source_map_->height(); ++y) {
    if (ControlCheckDue(y)) {
      if (const auto stopped = StopReason(control); stopped.has_value()) {
        return {.reason_code = std::string{*stopped}};
      }
    }
    const std::size_t begin = y * projection.source_map_->width();
    std::copy_n(squared_distance.begin() + begin, line.size(), line.begin());
    TransformLine(line, &transformed);
    std::copy(transformed.begin(), transformed.end(), temporary.begin() + begin);
  }
  line.resize(projection.source_map_->height());
  projection.clearance_m_.resize(count);
  for (std::size_t x = 0U; x < projection.source_map_->width(); ++x) {
    if (ControlCheckDue(x)) {
      if (const auto stopped = StopReason(control); stopped.has_value()) {
        return {.reason_code = std::string{*stopped}};
      }
    }
    for (std::size_t y = 0U; y < projection.source_map_->height(); ++y) {
      line[y] = temporary[y * projection.source_map_->width() + x];
    }
    TransformLine(line, &transformed);
    for (std::size_t y = 0U; y < projection.source_map_->height(); ++y) {
      projection.clearance_m_[y * projection.source_map_->width() + x] =
          static_cast<float>(std::sqrt(transformed[y]) *
                             projection.source_map_->resolution_m());
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
  for (std::size_t index = 0U; index < projection.hard_feasible_.size();
       ++index) {
    if (ControlCheckDue(index)) {
      if (const auto stopped = StopReason(control); stopped.has_value()) {
        return {.reason_code = std::string{*stopped}};
      }
    }
    if (projection.clearance_m_[index] < inflation_m) {
      projection.hard_feasible_[index] = 0U;
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

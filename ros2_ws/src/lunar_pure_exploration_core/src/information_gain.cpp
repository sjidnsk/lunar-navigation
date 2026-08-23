#include "lunar_pure_exploration_core/information_gain.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numbers>
#include <stdexcept>

#include "detail/visibility_traversal.hpp"

namespace lunar::pure_exploration {
namespace {

using Wide = long double;

std::size_t Remaining(const detail::VisibilityWorkBudget& budget) {
  if (budget.used > budget.limit) {
    throw std::overflow_error("visibility work budget is inconsistent");
  }
  return budget.limit - budget.used;
}

std::int32_t CheckedBound(Wide value) {
  if (!std::isfinite(value) ||
      value < static_cast<Wide>(std::numeric_limits<std::int32_t>::min()) ||
      value > static_cast<Wide>(std::numeric_limits<std::int32_t>::max())) {
    throw std::overflow_error("visibility bound exceeds int32");
  }
  return static_cast<std::int32_t>(value);
}

bool IsRepresentable(const detail::TraceCell& cell) {
  return cell.x >= std::numeric_limits<std::int32_t>::min() &&
         cell.x <= std::numeric_limits<std::int32_t>::max() &&
         cell.y >= std::numeric_limits<std::int32_t>::min() &&
         cell.y <= std::numeric_limits<std::int32_t>::max();
}

GridIndex ToGridIndex(const detail::TraceCell& cell) {
  if (!IsRepresentable(cell)) {
    throw std::overflow_error("virtual trace cell has no GridIndex");
  }
  return {static_cast<std::int32_t>(cell.x),
          static_cast<std::int32_t>(cell.y)};
}

}  // namespace

InformationGainEvaluator::InformationGainEvaluator(SensorModel model,
                                                   Limits limits)
    : model_(model), limits_(limits) {
  const double full_turn = 2.0 * std::numbers::pi;
  if (!std::isfinite(model_.range_m) || model_.range_m <= 0.0) {
    throw std::invalid_argument("sensor range must be finite and positive");
  }
  if (!std::isfinite(model_.field_of_view_rad) ||
      model_.field_of_view_rad <= 0.0 ||
      model_.field_of_view_rad > full_turn) {
    throw std::invalid_argument("sensor FOV must be in (0, 2*pi]");
  }
  if (limits_.maximum_visibility_work_units == 0U) {
    throw std::invalid_argument("visibility work limit must be positive");
  }
}

GainEvaluation InformationGainEvaluator::Evaluate(
    const TaskRaster& raster, const CandidateView& candidate) const {
  if (!std::isfinite(candidate.pose.x) ||
      !std::isfinite(candidate.pose.y) ||
      !std::isfinite(candidate.pose.yaw)) {
    throw std::invalid_argument("candidate pose must be finite");
  }
  const auto start = raster.WorldToGrid({candidate.pose.x, candidate.pose.y});
  if (!start.has_value()) {
    throw std::overflow_error("candidate cannot be represented in grid frame");
  }

  const GridGeometry& geometry = raster.geometry();
  const double range_cells_double = model_.range_m / geometry.resolution;
  if (!std::isfinite(range_cells_double)) {
    throw std::overflow_error("sensor range in cells overflowed");
  }
  const Wide range_cells = range_cells_double;
  const Wide start_x = start->x;
  const Wide start_y = start->y;
  const Wide lower_x = std::ceil(start_x - range_cells - 0.5L);
  const Wide upper_x = std::floor(start_x + range_cells - 0.5L);
  const Wide lower_y = std::ceil(start_y - range_cells - 0.5L);
  const Wide upper_y = std::floor(start_y + range_cells - 0.5L);

  detail::VisibilityWorkBudget budget{
      limits_.maximum_visibility_work_units, 0U};
  if (lower_x > upper_x || lower_y > upper_y) {
    return {0U, 0.0};
  }
  const Wide int32_min = std::numeric_limits<std::int32_t>::min();
  const Wide int32_max = std::numeric_limits<std::int32_t>::max();
  const Wide clipped_lower_x = std::max(lower_x, int32_min);
  const Wide clipped_upper_x = std::min(upper_x, int32_max);
  const Wide clipped_lower_y = std::max(lower_y, int32_min);
  const Wide clipped_upper_y = std::min(upper_y, int32_max);
  if (clipped_lower_x > clipped_upper_x ||
      clipped_lower_y > clipped_upper_y) {
    return {0U, 0.0};
  }
  const std::int32_t minimum_x = CheckedBound(clipped_lower_x);
  const std::int32_t maximum_x = CheckedBound(clipped_upper_x);
  const std::int32_t minimum_y = CheckedBound(clipped_lower_y);
  const std::int32_t maximum_y = CheckedBound(clipped_upper_y);
  const std::uint64_t clipped_width =
      static_cast<std::uint64_t>(static_cast<std::int64_t>(maximum_x) -
                                 minimum_x) +
      1U;
  const std::uint64_t clipped_height =
      static_cast<std::uint64_t>(static_cast<std::int64_t>(maximum_y) -
                                 minimum_y) +
      1U;
  const std::size_t remaining = Remaining(budget);
  if (clipped_height > remaining ||
      clipped_width > remaining / clipped_height) {
    throw std::length_error("visibility AABB exceeds remaining work budget");
  }

  const double yaw_delta = candidate.pose.yaw - geometry.origin_yaw;
  if (!std::isfinite(yaw_delta)) {
    throw std::overflow_error("candidate grid yaw overflowed");
  }
  const double full_turn = 2.0 * std::numbers::pi;
  const double yaw_grid = std::remainder(yaw_delta, full_turn);
  const double half_fov = model_.field_of_view_rad / 2.0;
  const bool omnidirectional = model_.field_of_view_rad == full_turn;
  const Wide range_squared = range_cells * range_cells;
  std::uint32_t visible_count = 0U;

  for (std::int64_t y = minimum_y; y <= maximum_y; ++y) {
    for (std::int64_t x = minimum_x; x <= maximum_x; ++x) {
      detail::ConsumeVisibilityWork(budget);
      const GridIndex target{static_cast<std::int32_t>(x),
                             static_cast<std::int32_t>(y)};
      const Wide wide_delta_x =
          static_cast<Wide>(x) + 0.5L - static_cast<Wide>(start->x);
      const Wide wide_delta_y =
          static_cast<Wide>(y) + 0.5L - static_cast<Wide>(start->y);
      const Wide distance_squared =
          wide_delta_x * wide_delta_x + wide_delta_y * wide_delta_y;
      if (distance_squared > range_squared) {
        continue;
      }
      if (!omnidirectional && distance_squared != 0.0L) {
        const double bearing =
            std::atan2(static_cast<double>(wide_delta_y),
                       static_cast<double>(wide_delta_x));
        const double bearing_difference =
            std::abs(std::remainder(bearing - yaw_grid, full_turn));
        if (bearing_difference > half_fov) {
          continue;
        }
      }
      const CellState target_state = raster.Classify(target);
      if (target_state != CellState::kUnknown &&
          target_state != CellState::kOutsideMap) {
        continue;
      }

      bool blocked = false;
      detail::TraceClosedSegment(
          *start, {static_cast<double>(x) + 0.5,
                   static_cast<double>(y) + 0.5},
          budget,
          [&](long double, std::span<const detail::TraceCell> group) {
            bool outside_task = false;
            bool occupied = false;
            for (const detail::TraceCell& traced : group) {
              if (!IsRepresentable(traced)) {
                outside_task = true;
                continue;
              }
              const GridIndex visited = ToGridIndex(traced);
              const detail::PreTargetContact contact =
                  detail::ClassifyPreTargetContact(raster.Classify(visited),
                                                   visited, target);
              outside_task = outside_task ||
                             contact == detail::PreTargetContact::kOutsideTask;
              occupied = occupied ||
                         contact == detail::PreTargetContact::kOccupied;
            }
            if (outside_task || occupied) {
              blocked = true;
              return detail::TraceControl::kStop;
            }
            return detail::TraceControl::kContinue;
          });
      if (!blocked) {
        detail::CheckedVisibleIncrement(visible_count);
      }
    }
  }

  const Wide area_preflight =
      static_cast<Wide>(visible_count) * geometry.resolution *
      geometry.resolution;
  if (!std::isfinite(area_preflight) ||
      area_preflight > std::numeric_limits<double>::max()) {
    throw std::overflow_error("visible unknown area overflow");
  }
  const double area = static_cast<double>(visible_count) *
                      (geometry.resolution * geometry.resolution);
  if (!std::isfinite(area)) {
    throw std::overflow_error("visible unknown area is not finite");
  }
  return {visible_count, area};
}

}  // namespace lunar::pure_exploration

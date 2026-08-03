#include "shared/convex_corridor.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace lunar::planning::shared {
namespace {

constexpr double kComparisonTolerance = 1.0e-12;

struct CellRectangle final {
  std::int32_t minimum_x{};
  std::int32_t maximum_x{};
  std::int32_t minimum_y{};
  std::int32_t maximum_y{};
};

struct MetricRectangle final {
  double minimum_x{};
  double maximum_x{};
  double minimum_y{};
  double maximum_y{};
};

[[nodiscard]] CorridorResult Invalid(std::string reason_code) {
  return CorridorResult{
      .status = CorridorStatus::kInvalidRequest,
      .fallback = CorridorFallback::kUseDiscreteValidatedPrimitives,
      .cells = {},
      .reason_code = std::move(reason_code),
  };
}

[[nodiscard]] CorridorResult Fallback(
    std::string reason_code, const std::size_t iterations) {
  return CorridorResult{
      .status = CorridorStatus::kFallbackRequired,
      .fallback = CorridorFallback::kUseDiscreteValidatedPrimitives,
      .cells = {},
      .reason_code = std::move(reason_code),
      .iterations = iterations,
  };
}

[[nodiscard]] CorridorResult Canceled(const std::size_t iterations) {
  return CorridorResult{
      .status = CorridorStatus::kCanceled,
      .fallback = CorridorFallback::kUseDiscreteValidatedPrimitives,
      .cells = {},
      .reason_code = "REQUEST_CANCELED",
      .iterations = iterations,
  };
}

[[nodiscard]] bool IsFinite(const Vec2& value) noexcept {
  return std::isfinite(value.x) && std::isfinite(value.y);
}

[[nodiscard]] bool CellPasses(
    const SafeProjection& projection, const GridCell cell,
    const double required_clearance_m) noexcept {
  return projection.HardFeasible(cell) &&
         static_cast<double>(projection.ClearanceMeters(cell)) +
                 kComparisonTolerance >=
             required_clearance_m;
}

[[nodiscard]] bool ExpansionStripPasses(
    const SafeProjection& projection, const CellRectangle& rectangle,
    const std::size_t direction,
    const double required_clearance_m) noexcept {
  const std::int32_t minimum_x = direction == 0U
      ? rectangle.minimum_x
      : (direction == 1U ? rectangle.maximum_x : rectangle.minimum_x);
  const std::int32_t maximum_x = direction <= 1U
      ? minimum_x
      : rectangle.maximum_x;
  const std::int32_t minimum_y = direction == 2U
      ? rectangle.minimum_y
      : (direction == 3U ? rectangle.maximum_y : rectangle.minimum_y);
  const std::int32_t maximum_y = direction >= 2U
      ? minimum_y
      : rectangle.maximum_y;
  for (std::int32_t y = minimum_y; y <= maximum_y; ++y) {
    for (std::int32_t x = minimum_x; x <= maximum_x; ++x) {
      if (!CellPasses(
              projection, GridCell{.x = x, .y = y},
              required_clearance_m)) {
        return false;
      }
    }
  }
  return true;
}

[[nodiscard]] CellRectangle Expanded(
    const CellRectangle& source, const std::size_t direction) noexcept {
  CellRectangle expanded = source;
  switch (direction) {
    case 0U:
      --expanded.minimum_x;
      break;
    case 1U:
      ++expanded.maximum_x;
      break;
    case 2U:
      --expanded.minimum_y;
      break;
    default:
      ++expanded.maximum_y;
      break;
  }
  return expanded;
}

[[nodiscard]] MetricRectangle ToMetricRectangle(
    const MapSnapshot& map, const CellRectangle& cells,
    const double margin_m) noexcept {
  return MetricRectangle{
      .minimum_x = map.origin_m().x +
          static_cast<double>(cells.minimum_x) * map.resolution_m() + margin_m,
      .maximum_x = map.origin_m().x +
          static_cast<double>(cells.maximum_x + 1) * map.resolution_m() -
          margin_m,
      .minimum_y = map.origin_m().y +
          static_cast<double>(cells.minimum_y) * map.resolution_m() + margin_m,
      .maximum_y = map.origin_m().y +
          static_cast<double>(cells.maximum_y + 1) * map.resolution_m() -
          margin_m,
  };
}

[[nodiscard]] bool SameRectangle(
    const MetricRectangle& lhs, const MetricRectangle& rhs) noexcept {
  return std::abs(lhs.minimum_x - rhs.minimum_x) <= kComparisonTolerance &&
         std::abs(lhs.maximum_x - rhs.maximum_x) <= kComparisonTolerance &&
         std::abs(lhs.minimum_y - rhs.minimum_y) <= kComparisonTolerance &&
         std::abs(lhs.maximum_y - rhs.maximum_y) <= kComparisonTolerance;
}

[[nodiscard]] bool HasRequiredOverlap(
    const MetricRectangle& lhs, const MetricRectangle& rhs,
    const double minimum_overlap_m) noexcept {
  const double overlap_x =
      std::min(lhs.maximum_x, rhs.maximum_x) -
      std::max(lhs.minimum_x, rhs.minimum_x);
  const double overlap_y =
      std::min(lhs.maximum_y, rhs.maximum_y) -
      std::max(lhs.minimum_y, rhs.minimum_y);
  return overlap_x + kComparisonTolerance >= minimum_overlap_m &&
         overlap_y + kComparisonTolerance >= minimum_overlap_m;
}

[[nodiscard]] std::vector<HalfPlane2> HalfPlanes(
    const MetricRectangle& rectangle) {
  return {
      HalfPlane2{
          .outward_unit_normal = Vec2{.x = 1.0, .y = 0.0},
          .upper_offset_m = rectangle.maximum_x,
      },
      HalfPlane2{
          .outward_unit_normal = Vec2{.x = -1.0, .y = 0.0},
          .upper_offset_m = -rectangle.minimum_x,
      },
      HalfPlane2{
          .outward_unit_normal = Vec2{.x = 0.0, .y = 1.0},
          .upper_offset_m = rectangle.maximum_y,
      },
      HalfPlane2{
          .outward_unit_normal = Vec2{.x = 0.0, .y = -1.0},
          .upper_offset_m = -rectangle.minimum_y,
      },
  };
}

[[nodiscard]] std::vector<double> ArcLengths(
    const std::vector<Vec2>& centerline) {
  std::vector<double> lengths(centerline.size(), 0.0);
  for (std::size_t index = 1U; index < centerline.size(); ++index) {
    lengths[index] = lengths[index - 1U] + std::hypot(
        centerline[index].x - centerline[index - 1U].x,
        centerline[index].y - centerline[index - 1U].y);
  }
  return lengths;
}

}  // namespace

CorridorResult BuildConvexCorridor(
    const SafeProjection& projection,
    const std::vector<Vec2>& validated_centerline,
    const CorridorTightening& tightening,
    const CorridorConfig& config,
    const std::stop_token stop_token) {
  if (stop_token.stop_requested()) {
    return Canceled(0U);
  }
  if (projection.source_map() == nullptr || validated_centerline.empty() ||
      !std::ranges::all_of(validated_centerline, IsFinite)) {
    return Invalid("CORRIDOR_CENTERLINE_INVALID");
  }
  if (!std::isfinite(tightening.footprint_support_radius_m) ||
      !std::isfinite(tightening.tracking_error_bound_m) ||
      !std::isfinite(tightening.additional_margin_m) ||
      tightening.footprint_support_radius_m < 0.0 ||
      tightening.tracking_error_bound_m < 0.0 ||
      tightening.additional_margin_m < 0.0 ||
      config.maximum_regions == 0U ||
      config.maximum_inflation_iterations == 0U ||
      config.maximum_halfplanes_per_region < 4U ||
      config.maximum_split_depth == 0U ||
      !std::isfinite(config.minimum_overlap_m) ||
      config.minimum_overlap_m < 0.0 ||
      !std::isfinite(config.sampling_spacing_m) ||
      config.sampling_spacing_m <= 0.0) {
    return Invalid("CORRIDOR_CONFIG_INVALID");
  }
  const double margin_m = tightening.footprint_support_radius_m +
      tightening.tracking_error_bound_m + tightening.additional_margin_m;
  if (!std::isfinite(margin_m)) {
    return Invalid("CORRIDOR_TIGHTENING_INVALID");
  }

  const MapSnapshot& map = *projection.source_map();
  const std::vector<double> arc_lengths = ArcLengths(validated_centerline);
  std::vector<MetricRectangle> rectangles;
  std::vector<ConvexCorridorCell> cells;
  std::size_t total_iterations = 0U;
  for (std::size_t point_index = 0U;
       point_index < validated_centerline.size(); ++point_index) {
    if (stop_token.stop_requested()) {
      return Canceled(total_iterations);
    }
    const auto center = map.PositionToCell(validated_centerline[point_index]);
    if (!center.has_value() ||
        !CellPasses(projection, *center, margin_m)) {
      return Fallback("CORRIDOR_CENTERLINE_NOT_SAFE", total_iterations);
    }
    CellRectangle rectangle{
        .minimum_x = center->x,
        .maximum_x = center->x,
        .minimum_y = center->y,
        .maximum_y = center->y,
    };
    bool changed = true;
    std::size_t point_iterations = 0U;
    while (changed && point_iterations < config.maximum_inflation_iterations) {
      changed = false;
      for (std::size_t direction = 0U; direction < 4U; ++direction) {
        if (stop_token.stop_requested()) {
          return Canceled(total_iterations);
        }
        if (point_iterations >= config.maximum_inflation_iterations) {
          break;
        }
        const CellRectangle candidate = Expanded(rectangle, direction);
        ++point_iterations;
        ++total_iterations;
        if (candidate.minimum_x < 0 || candidate.minimum_y < 0 ||
            candidate.maximum_x >= static_cast<std::int32_t>(map.width()) ||
            candidate.maximum_y >= static_cast<std::int32_t>(map.height()) ||
            !ExpansionStripPasses(
                projection, candidate, direction, margin_m)) {
          continue;
        }
        rectangle = candidate;
        changed = true;
      }
    }

    const MetricRectangle metric =
        ToMetricRectangle(map, rectangle, margin_m);
    if (metric.minimum_x >= metric.maximum_x ||
        metric.minimum_y >= metric.maximum_y) {
      return Fallback("CORRIDOR_EMPTY_AFTER_TIGHTENING", total_iterations);
    }
    const double begin_s = point_index == 0U
        ? 0.0
        : 0.5 * (arc_lengths[point_index - 1U] + arc_lengths[point_index]);
    const double end_s = point_index + 1U == validated_centerline.size()
        ? arc_lengths.back()
        : 0.5 * (arc_lengths[point_index] + arc_lengths[point_index + 1U]);
    if (!rectangles.empty() && SameRectangle(rectangles.back(), metric)) {
      cells.back().centerline_s_end_m = end_s;
      continue;
    }
    if (!rectangles.empty() &&
        !HasRequiredOverlap(rectangles.back(), metric, config.minimum_overlap_m)) {
      return Fallback("CORRIDOR_OVERLAP_INSUFFICIENT", total_iterations);
    }
    if (cells.size() >= config.maximum_regions) {
      return Fallback("CORRIDOR_REGION_LIMIT", total_iterations);
    }
    rectangles.push_back(metric);
    cells.push_back(ConvexCorridorCell{
        .stable_index = cells.size(),
        .centerline_s_begin_m = begin_s,
        .centerline_s_end_m = end_s,
        .half_planes = HalfPlanes(metric),
    });
  }

  return CorridorResult{
      .status = CorridorStatus::kCertified,
      .fallback = CorridorFallback::kNone,
      .cells = std::move(cells),
      .reason_code = "CORRIDOR_CERTIFIED",
      .iterations = total_iterations,
  };
}

}  // namespace lunar::planning::shared

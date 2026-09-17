#include "lunar_incremental_navigation_core/fine_traversability_builder.hpp"

#include <algorithm>
#include <array>
#include <bitset>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "map/grid_bounds.hpp"

namespace lunar::incremental_navigation {
namespace {

struct LocalElevationMeasurements final {
  bool center_known{};
  bool neighborhood_complete{};
  double slope_rad{};
  double relief_m{};
  double positive_rise_m{};
};

[[nodiscard]] bool FiniteNonnegative(const double value) noexcept {
  return std::isfinite(value) && value >= 0.0;
}

[[nodiscard]] Point2 CellCenter(const SparseGridGeometry& geometry,
                                const GridIndex index) noexcept {
  const Vec3 origin = geometry.origin_m();
  const double resolution_m = geometry.resolution_m();
  return Point2{
      .x = std::fma(static_cast<double>(index.x) + 0.5, resolution_m,
                    origin.x),
      .y = std::fma(static_cast<double>(index.y) + 0.5, resolution_m,
                    origin.y),
  };
}

[[nodiscard]] double DistanceToCellArea(
    const Point2 point, const SparseGridGeometry& geometry,
    const GridIndex cell) noexcept {
  const Vec3 origin = geometry.origin_m();
  const double resolution_m = geometry.resolution_m();
  const double point_x_cells = (point.x - origin.x) / resolution_m;
  const double point_y_cells = (point.y - origin.y) / resolution_m;
  const double minimum_x = static_cast<double>(cell.x);
  const double minimum_y = static_cast<double>(cell.y);
  const double dx = std::max(
      {minimum_x - point_x_cells, 0.0,
       point_x_cells - minimum_x - 1.0});
  const double dy = std::max(
      {minimum_y - point_y_cells, 0.0,
       point_y_cells - minimum_y - 1.0});
  return std::hypot(dx, dy) * resolution_m;
}

[[nodiscard]] bool CircleLeavesGeometry(
    const Point2 center, const double radius_m,
    const SparseGridGeometry& geometry) noexcept {
  const Vec3 origin = geometry.origin_m();
  const GridIndex minimum = geometry.min_inclusive();
  const GridIndex maximum = geometry.max_exclusive();
  const double minimum_x = std::fma(
      static_cast<double>(minimum.x), geometry.resolution_m(), origin.x);
  const double minimum_y = std::fma(
      static_cast<double>(minimum.y), geometry.resolution_m(), origin.y);
  const double maximum_x = std::fma(
      static_cast<double>(maximum.x), geometry.resolution_m(), origin.x);
  const double maximum_y = std::fma(
      static_cast<double>(maximum.y), geometry.resolution_m(), origin.y);
  return center.x - radius_m < minimum_x ||
         center.y - radius_m < minimum_y ||
         center.x + radius_m > maximum_x ||
         center.y + radius_m > maximum_y;
}

[[nodiscard]] double DistanceToGeometryBoundary(
    const Point2 center, const SparseGridGeometry& geometry) noexcept {
  const Vec3 origin = geometry.origin_m();
  const GridIndex minimum = geometry.min_inclusive();
  const GridIndex maximum = geometry.max_exclusive();
  const double minimum_x = std::fma(
      static_cast<double>(minimum.x), geometry.resolution_m(), origin.x);
  const double minimum_y = std::fma(
      static_cast<double>(minimum.y), geometry.resolution_m(), origin.y);
  const double maximum_x = std::fma(
      static_cast<double>(maximum.x), geometry.resolution_m(), origin.x);
  const double maximum_y = std::fma(
      static_cast<double>(maximum.y), geometry.resolution_m(), origin.y);
  return std::min({center.x - minimum_x, maximum_x - center.x,
                   center.y - minimum_y, maximum_y - center.y});
}

[[nodiscard]] double SlopeLimit(const PlatformCapability& capability) {
  return std::visit(
      [](const auto& typed_capability) {
        using Capability = std::decay_t<decltype(typed_capability)>;
        if constexpr (std::is_same_v<Capability, WheeledCapability> ||
                      std::is_same_v<Capability, LeggedCapability>) {
          return typed_capability.maximum_slope_rad;
        }
        return 0.0;
      },
      capability);
}

[[nodiscard]] double ReliefLimit(const PlatformCapability& capability) {
  return std::visit(
      [](const auto& typed_capability) {
        using Capability = std::decay_t<decltype(typed_capability)>;
        if constexpr (std::is_same_v<Capability, WheeledCapability>) {
          return typed_capability.maximum_local_obstacle_relief_m;
        } else if constexpr (std::is_same_v<Capability, LeggedCapability>) {
          return typed_capability.maximum_step_height_m;
        }
        return 0.0;
      },
      capability);
}

[[nodiscard]] double NormalizedSquared(const double value,
                                       const double limit) noexcept {
  if (limit <= 0.0) {
    return 0.0;
  }
  const double normalized = std::clamp(value / limit, 0.0, 1.0);
  return normalized * normalized;
}

[[nodiscard]] double SaturatingProduct(const double left,
                                       const double right) noexcept {
  if (left == 0.0 || right == 0.0) {
    return 0.0;
  }
  constexpr double kLargestFinite = std::numeric_limits<double>::max();
  if (left > kLargestFinite / right) {
    return kLargestFinite;
  }
  const double product = left * right;
  return std::isfinite(product) ? product : kLargestFinite;
}

[[nodiscard]] double SaturatingSum(const double left,
                                   const double right) noexcept {
  constexpr double kLargestFinite = std::numeric_limits<double>::max();
  if (left >= kLargestFinite - right) {
    return kLargestFinite;
  }
  const double sum = left + right;
  return std::isfinite(sum) ? sum : kLargestFinite;
}

[[nodiscard]] LocalElevationMeasurements MeasureLocalElevation(
    const ElevationRangeView& elevation, const GridIndex center) {
  const std::optional<ElevationRange> center_range =
      elevation.ElevationRangeAt(center);
  if (!center_range) {
    return {};
  }

  const double center_mid =
      0.5 * (static_cast<double>(center_range->min_m) +
             static_cast<double>(center_range->max_m));
  double minimum = center_range->min_m;
  double maximum = center_range->max_m;
  double maximum_slope = 0.0;
  double maximum_positive_rise = 0.0;
  bool complete = true;
  const double resolution_m = elevation.geometry().resolution_m();
  for (std::int64_t dy = -1; dy <= 1; ++dy) {
    for (std::int64_t dx = -1; dx <= 1; ++dx) {
      if (dx == 0 && dy == 0) {
        continue;
      }
      const std::optional<GridIndex> neighbor_index =
          detail::OffsetWithinGeometry(elevation.geometry(), center, dx, dy);
      const std::optional<ElevationRange> neighbor =
          neighbor_index ? elevation.ElevationRangeAt(*neighbor_index)
                         : std::nullopt;
      if (!neighbor_index || !neighbor) {
        complete = false;
        continue;
      }
      const double neighbor_mid =
          0.5 * (static_cast<double>(neighbor->min_m) +
                 static_cast<double>(neighbor->max_m));
      const double planar_distance_m =
          resolution_m * std::hypot(static_cast<double>(dx),
                                    static_cast<double>(dy));
      maximum_slope =
          std::max(maximum_slope,
                   std::atan2(std::abs(neighbor_mid - center_mid),
                              planar_distance_m));
      minimum = std::min(minimum, static_cast<double>(neighbor->min_m));
      maximum = std::max(maximum, static_cast<double>(neighbor->max_m));
      maximum_positive_rise = std::max(
          maximum_positive_rise,
          static_cast<double>(neighbor->max_m) - center_mid);
    }
  }
  return LocalElevationMeasurements{
      .center_known = true,
      .neighborhood_complete = complete,
      .slope_rad = maximum_slope,
      .relief_m = maximum - minimum,
      .positive_rise_m = std::max(0.0, maximum_positive_rise),
  };
}

// Basic wheel terrain model: a fixed 5x5 support window (60% coverage),
// robust background plane, then unsmoothed local residuals for steps/rocks.
// Legged evaluation below deliberately retains its existing 3x3 model.
struct WheelPatchPoint {
  int x{}, y{};
  double low{}, high{}, z{}, residual{};
};

template <std::size_t N>
double PatchMedian(std::array<double, N> values, std::size_t count) {
  std::sort(values.begin(), values.begin() + count);
  return values[count / 2];
}

IntrinsicTraversalEvaluation EvaluateWheelPatch(
    const ElevationRangeView& elevation, GridIndex center,
    const WheeledCapability& capability) {
  if (!elevation.ElevationRangeAt(center)) return {};
  std::array<WheelPatchPoint, 25> points{};
  std::array<double, 25> heights{};
  std::array<int, 25> indices{};
  indices.fill(-1);
  std::size_t count = 0;
  int min_x = 2, max_x = -2, min_y = 2, max_y = -2;
  const double resolution = elevation.geometry().resolution_m();
  for (int y = -2; y <= 2; ++y)
    for (int x = -2; x <= 2; ++x) {
      const auto cell =
          detail::OffsetWithinGeometry(elevation.geometry(), center, x, y);
      const auto range =
          cell ? elevation.ElevationRangeAt(*cell) : std::nullopt;
      if (!range) continue;
      indices[(y + 2) * 5 + x + 2] = static_cast<int>(count);
      const double z = .5 * (double(range->min_m) + double(range->max_m));
      points[count++] = {x, y, range->min_m, range->max_m, z, 0.};
      min_x = std::min(min_x, x);
      max_x = std::max(max_x, x);
      min_y = std::min(min_y, y);
      max_y = std::max(max_y, y);
    }
  if (count < 15 || min_x >= 0 || max_x <= 0 || min_y >= 0 || max_y <= 0)
    return {};

  // Median adjacent derivatives initialize the background: a sharp step must
  // not tilt the initial fit into an artificial ramp joining its two surfaces.
  std::array<double, 20> gx{}, gy{};
  std::size_t nx = 0, ny = 0;
  for (int y = 0; y < 5; ++y)
    for (int x = 0; x < 5; ++x) {
      const int i = indices[y * 5 + x];
      if (i < 0) continue;
      if (x < 4 && indices[y * 5 + x + 1] >= 0)
        gx[nx++] =
            (points[indices[y * 5 + x + 1]].z - points[i].z) / resolution;
      if (y < 4 && indices[(y + 1) * 5 + x] >= 0)
        gy[ny++] =
            (points[indices[(y + 1) * 5 + x]].z - points[i].z) / resolution;
    }
  if (nx < 4 || ny < 4) return {};
  double a = PatchMedian(gx, nx), b = PatchMedian(gy, ny);
  for (std::size_t i = 0; i < count; ++i)
    heights[i] = points[i].z - resolution * (a * points[i].x + b * points[i].y);
  double c = PatchMedian(heights, count);
  const double background_a = a, background_b = b, background_c = c;
  for (std::size_t i = 0; i < count; ++i) heights[i] = std::abs(heights[i] - c);
  // A 5 mm robust scale floor prevents nearly exact flats from producing
  // singular weights. It is a numerical/noise scale, not an obstacle threshold.
  const double cutoff =
      2.5 * std::max(.005, 1.4826 * PatchMedian(heights, count));
  double sw = 0., sx = 0., sy = 0., sz = 0., sxx = 0., syy = 0., sxy = 0.,
         sxz = 0., syz = 0.;
  for (std::size_t i = 0; i < count; ++i) {
    const auto& p = points[i];
    const double x = p.x * resolution, y = p.y * resolution;
    const double residual = p.z - (a * x + b * y + c);
    const double w = std::min(1., cutoff / std::max(std::abs(residual), 1e-12));
    sw += w;
    sx += w * x;
    sy += w * y;
    sz += w * p.z;
    sxx += w * x * x;
    syy += w * y * y;
    sxy += w * x * y;
    sxz += w * x * p.z;
    syz += w * y * p.z;
  }
  sxx -= sx * sx / sw;
  syy -= sy * sy / sw;
  sxy -= sx * sy / sw;
  sxz -= sx * sz / sw;
  syz -= sy * sz / sw;
  const double det = sxx * syy - sxy * sxy;
  if (!(sxx > 0. && syy > 0. && det > 1e-8 * sxx * syy)) return {};
  a = (sxz * syy - syz * sxy) / det;
  b = (syz * sxx - sxz * sxy) / det;
  c = (sz - a * sx - b * sy) / sw;
  const double slope = std::atan(std::hypot(a, b));
  if (!std::isfinite(slope) || !std::isfinite(c)) return {};
  double low = std::numeric_limits<double>::infinity(), high = -low,
         center_residual = 0.;
  for (std::size_t i = 0; i < count; ++i) {
    auto& p = points[i];
    // Use the robust background before least-squares refinement for
    // discontinuities: even a small fitted tilt must not shave a 21 cm
    // step below the existing 20 cm limit.
    const double plane =
        resolution * (background_a * p.x + background_b * p.y) + background_c;
    p.residual = p.z - plane;
    if (p.x == 0 && p.y == 0) center_residual = p.residual;
    if (std::abs(p.x) <= 1 && std::abs(p.y) <= 1) {
      low = std::min(low, p.low - plane);
      high = std::max(high, p.high - plane);
    }
  }
  const double relief = high - low;
  const double rise = std::max(0., high - center_residual);
  const bool discontinuity =
      relief > capability.maximum_local_obstacle_relief_m ||
      rise > capability.minimum_underbody_clearance_m;
  if (discontinuity) {
    // Both surfaces need adjacent support. One tall/low cell stays UNKNOWN,
    // never FREE; small real obstacles are not silently discarded as outliers.
    const double band =
        .25 * std::min(capability.maximum_local_obstacle_relief_m,
                       capability.minimum_underbody_clearance_m);
    bool low_supported = false, high_supported = false;
    for (std::size_t i = 0; i < count; ++i)
      for (std::size_t j = i + 1; j < count; ++j) {
        const auto& p = points[i];
        const auto& q = points[j];
        if (std::max(std::abs(p.x - q.x), std::abs(p.y - q.y)) != 1) continue;
        low_supported |= p.residual <= low + band && q.residual <= low + band;
        high_supported |=
            p.residual >= high - band && q.residual >= high - band;
      }
    if (!low_supported || !high_supported)
      return {.state = IntrinsicCellState::kUnknown,
              .slope_rad = slope,
              .relief_m = relief};
  }
  return {.state = (discontinuity || slope > capability.maximum_slope_rad)
                       ? IntrinsicCellState::kBlocked
                       : IntrinsicCellState::kFree,
          .slope_rad = slope,
          .relief_m = relief};
}

class WheelElevationEvaluator final : public PlatformElevationEvaluator {
 public:
  explicit WheelElevationEvaluator(WheeledCapability capability)
      : capability_(std::move(capability)) {
    if (!FiniteNonnegative(capability_.maximum_slope_rad) ||
        !FiniteNonnegative(capability_.maximum_local_obstacle_relief_m) ||
        !FiniteNonnegative(capability_.minimum_underbody_clearance_m)) {
      throw std::invalid_argument(
          "wheel elevation limits must be finite and nonnegative");
    }
  }

  [[nodiscard]] IntrinsicTraversalEvaluation Evaluate(
      const ElevationRangeView& elevation,
      const GridIndex index) const override {
    return EvaluateWheelPatch(elevation, index, capability_);
  }

 private:
  WheeledCapability capability_;
};

class LeggedElevationEvaluator final : public PlatformElevationEvaluator {
 public:
  explicit LeggedElevationEvaluator(LeggedCapability capability)
      : capability_(std::move(capability)) {
    if (!FiniteNonnegative(capability_.maximum_slope_rad) ||
        !FiniteNonnegative(capability_.maximum_step_height_m) ||
        !FiniteNonnegative(capability_.maximum_gap_width_m)) {
      throw std::invalid_argument(
          "legged elevation limits must be finite and nonnegative");
    }
  }

  [[nodiscard]] IntrinsicTraversalEvaluation Evaluate(
      const ElevationRangeView& elevation,
      const GridIndex index) const override {
    const LocalElevationMeasurements measured =
        MeasureLocalElevation(elevation, index);
    if (!measured.center_known) {
      return {};
    }
    const bool exceeds_physical_limit =
        measured.slope_rad > capability_.maximum_slope_rad ||
        measured.relief_m > capability_.maximum_step_height_m;
    if (exceeds_physical_limit) {
      return IntrinsicTraversalEvaluation{
          .state = IntrinsicCellState::kBlocked,
          .slope_rad = measured.slope_rad,
          .relief_m = measured.relief_m,
      };
    }
    return IntrinsicTraversalEvaluation{
        .state = measured.neighborhood_complete
                     ? IntrinsicCellState::kFree
                     : IntrinsicCellState::kUnknown,
        .slope_rad = measured.slope_rad,
        .relief_m = measured.relief_m,
    };
  }

 private:
  LeggedCapability capability_;
};

}  // namespace

std::unique_ptr<const PlatformElevationEvaluator>
MakePlatformElevationEvaluator(const PlatformCapability& capability) {
  return std::visit(
      [](const auto& typed_capability)
          -> std::unique_ptr<const PlatformElevationEvaluator> {
        using Capability = std::decay_t<decltype(typed_capability)>;
        if constexpr (std::is_same_v<Capability, WheeledCapability>) {
          return std::make_unique<const WheelElevationEvaluator>(
              typed_capability);
        } else if constexpr (std::is_same_v<Capability, LeggedCapability>) {
          return std::make_unique<const LeggedElevationEvaluator>(
              typed_capability);
        } else {
          throw std::invalid_argument(
              "fine traversability supports wheel or legged capability");
        }
      },
      capability);
}

struct FineCellEvaluator::Impl final {
  struct IntrinsicCacheTile final {
    std::array<IntrinsicTraversalEvaluation, kGridTileCellCount> evaluations;
    std::bitset<kGridTileCellCount> populated;
  };

  Impl(const ElevationRangeView& elevation_value,
       PlatformCapability capability_value,
       TraversabilityProfile profile_value)
      : elevation(elevation_value),
        capability(std::move(capability_value)),
        profile(std::move(profile_value)),
        intrinsic_evaluator(MakePlatformElevationEvaluator(capability)),
        hard_radius_m(CircumscribedRadius(profile.planar_envelope_xy_m)) {
    if (!FiniteNonnegative(profile.preferred_clearance_m) ||
        !FiniteNonnegative(profile.slope_weight) ||
        !FiniteNonnegative(profile.relief_weight) ||
        !FiniteNonnegative(profile.clearance_weight)) {
      throw std::invalid_argument(
          "fine cell soft costs must be finite and nonnegative");
    }
  }

  [[nodiscard]] const IntrinsicTraversalEvaluation& IntrinsicAt(
      const GridIndex index) {
    const TileIndex tile_index = TileForCell(index);
    if (last_cache_tile == nullptr || tile_index != last_cache_tile_index) {
      auto [found, inserted] = intrinsic_cache.try_emplace(tile_index);
      if (inserted) {
        found->second = std::make_unique<IntrinsicCacheTile>();
      }
      last_cache_tile_index = tile_index;
      last_cache_tile = found->second.get();
    }
    const std::size_t offset = TileCellOffset(index);
    if (!last_cache_tile->populated.test(offset)) {
      last_cache_tile->evaluations[offset] =
          intrinsic_evaluator->Evaluate(elevation, index);
      last_cache_tile->populated.set(offset);
      ++evaluated_elevation_cell_count;
    }
    return last_cache_tile->evaluations[offset];
  }

  [[nodiscard]] FineCellEvaluation Evaluate(const GridIndex candidate) {
    const SparseGridGeometry& geometry = elevation.geometry();
    if (!geometry.Contains(candidate)) {
      return {};
    }
    const Point2 center = CellCenter(geometry, candidate);
    bool intersects_unknown =
        CircleLeavesGeometry(center, hard_radius_m, geometry);
    const auto [minimum, maximum] =
        detail::CellBoundsForRadius(geometry, candidate, hard_radius_m);
    for (std::int64_t y = minimum.y; y < maximum.y; ++y) {
      for (std::int64_t x = minimum.x; x < maximum.x; ++x) {
        const GridIndex seed{.x = x, .y = y};
        if (!CircleIntersectsCellArea(center, hard_radius_m, geometry, seed)) {
          continue;
        }
        const IntrinsicCellState state = IntrinsicAt(seed).state;
        if (state == IntrinsicCellState::kBlocked) {
          return {.state = FineCellState::kBlocked};
        }
        intersects_unknown =
            intersects_unknown || state == IntrinsicCellState::kUnknown;
      }
    }
    if (intersects_unknown) {
      return {.state = FineCellState::kUnknown};
    }

    const IntrinsicTraversalEvaluation& intrinsic = IntrinsicAt(candidate);
    const double slope_cost = SaturatingProduct(
        profile.slope_weight,
        NormalizedSquared(intrinsic.slope_rad, SlopeLimit(capability)));
    const double relief_cost = SaturatingProduct(
        profile.relief_weight,
        NormalizedSquared(intrinsic.relief_m, ReliefLimit(capability)));
    double traversal_cost = SaturatingSum(slope_cost, relief_cost);
    if (profile.clearance_weight == 0.0 ||
        profile.preferred_clearance_m == 0.0) {
      return {.state = FineCellState::kFree,
              .traversal_cost = traversal_cost};
    }

    const double influence_m =
        hard_radius_m + profile.preferred_clearance_m;
    double nearest_hazard_m = DistanceToGeometryBoundary(center, geometry);
    const auto [clearance_minimum, clearance_maximum] =
        detail::CellBoundsForRadius(geometry, candidate, influence_m);
    for (std::int64_t y = clearance_minimum.y; y < clearance_maximum.y; ++y) {
      for (std::int64_t x = clearance_minimum.x;
           x < clearance_maximum.x; ++x) {
        const GridIndex seed{.x = x, .y = y};
        if (IntrinsicAt(seed).state == IntrinsicCellState::kFree) {
          continue;
        }
        nearest_hazard_m = std::min(
            nearest_hazard_m, DistanceToCellArea(center, geometry, seed));
      }
    }
    const double remaining_clearance_m =
        std::max(0.0, nearest_hazard_m - hard_radius_m);
    const double clearance_penalty = std::clamp(
        1.0 - remaining_clearance_m / profile.preferred_clearance_m, 0.0,
        1.0);
    traversal_cost = SaturatingSum(
        traversal_cost,
        SaturatingProduct(profile.clearance_weight, clearance_penalty));
    return {.state = FineCellState::kFree,
            .traversal_cost = traversal_cost};
  }

  const ElevationRangeView& elevation;
  PlatformCapability capability;
  TraversabilityProfile profile;
  std::unique_ptr<const PlatformElevationEvaluator> intrinsic_evaluator;
  double hard_radius_m{};
  std::map<TileIndex, std::unique_ptr<IntrinsicCacheTile>> intrinsic_cache;
  TileIndex last_cache_tile_index;
  IntrinsicCacheTile* last_cache_tile{};
  std::size_t evaluated_elevation_cell_count{};
};

FineCellEvaluator::FineCellEvaluator(
    const ElevationRangeView& elevation,
    const PlatformCapability& capability,
    const TraversabilityProfile& profile)
    : impl_(std::make_unique<Impl>(elevation, capability, profile)) {}

FineCellEvaluator::~FineCellEvaluator() = default;
FineCellEvaluator::FineCellEvaluator(FineCellEvaluator&&) noexcept = default;
FineCellEvaluator& FineCellEvaluator::operator=(FineCellEvaluator&&) noexcept =
    default;

FineCellEvaluation FineCellEvaluator::Evaluate(const GridIndex index) {
  return impl_->Evaluate(index);
}

double FineCellEvaluator::hard_inflation_radius_m() const noexcept {
  return impl_->hard_radius_m;
}

std::size_t FineCellEvaluator::evaluated_elevation_cells() const noexcept {
  return impl_->evaluated_elevation_cell_count;
}

std::size_t FineCellEvaluator::cached_elevation_tiles() const noexcept {
  return impl_->intrinsic_cache.size();
}

}  // namespace lunar::incremental_navigation

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
    const LocalElevationMeasurements measured =
        MeasureLocalElevation(elevation, index);
    if (!measured.center_known) {
      return {};
    }
    const bool exceeds_physical_limit =
        measured.slope_rad > capability_.maximum_slope_rad ||
        measured.relief_m > capability_.maximum_local_obstacle_relief_m ||
        measured.positive_rise_m >
            capability_.minimum_underbody_clearance_m;
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

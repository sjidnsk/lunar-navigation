#include "lunar_path_planner/v3/corridor/convex_corridor.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <ranges>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace lunar::planning::v3 {
namespace {

constexpr double kGeometryTolerance = 1.0e-9;
constexpr double kSafetyTolerance = 1.0e-7;
constexpr double kUnitNormalTolerance = 1.0e-9;
constexpr std::uint64_t kFnvOffsetBasis =
    UINT64_C(14695981039346656037);
constexpr std::uint64_t kFnvPrime = UINT64_C(1099511628211);

struct BuiltCell final {
  ConvexCorridorCell published;
  std::vector<Vec2> polygon;
  std::size_t centerline_begin{};
  std::size_t centerline_end{};
};

struct ForbiddenCandidate final {
  double squared_distance_m2{};
  std::size_t y{};
  std::size_t x{};
  HalfPlane2 separating_plane;
};

enum class AddPlaneStatus {
  kAddedOrRedundant,
  kPlaneLimit,
  kEmpty,
};

[[nodiscard]] double CanonicalZero(double value) noexcept {
  return value == 0.0 ? 0.0 : value;
}

[[nodiscard]] bool HasNegativeZero(double value) noexcept {
  return value == 0.0 && std::signbit(value);
}

[[nodiscard]] bool IsFinite(const Vec2& value) noexcept {
  return std::isfinite(value.x) && std::isfinite(value.y);
}

[[nodiscard]] bool IsNonNegativeFinite(double value) noexcept {
  return std::isfinite(value) && value >= 0.0;
}

[[nodiscard]] CorridorResult InvalidRequest(
    std::string reason_code) {
  return {
      .status = CorridorStatus::kInvalidRequest,
      .fallback =
          CorridorFallback::kUseDiscreteValidatedPrimitives,
      .cells = {},
      .reason_code = std::move(reason_code),
      .iterations = 0U,
  };
}

[[nodiscard]] CorridorResult Fallback(
    std::string reason_code, std::size_t iterations) {
  return {
      .status = CorridorStatus::kFallbackRequired,
      .fallback =
          CorridorFallback::kUseDiscreteValidatedPrimitives,
      .cells = {},
      .reason_code = std::move(reason_code),
      .iterations = iterations,
  };
}

[[nodiscard]] bool ValidProjection(
    const SafeProjection& projection) noexcept {
  const auto& geometry = projection.geometry;
  const std::size_t count = geometry.CellCount();
  if (count == 0U || geometry.width >
                         static_cast<std::size_t>(
                             std::numeric_limits<std::int32_t>::max()) ||
      geometry.height >
          static_cast<std::size_t>(
              std::numeric_limits<std::int32_t>::max()) ||
      !std::isfinite(geometry.resolution_m) ||
      geometry.resolution_m <= 0.0 ||
      !IsFinite(geometry.origin_m) || geometry.frame_id.empty() ||
      projection.known_mask.size() != count ||
      projection.hard_feasible_mask.size() != count ||
      projection.esdf_clearance_m.size() != count) {
    return false;
  }

  const double width_m =
      static_cast<double>(geometry.width) * geometry.resolution_m;
  const double height_m =
      static_cast<double>(geometry.height) * geometry.resolution_m;
  if (!std::isfinite(width_m) || !std::isfinite(height_m) ||
      !std::isfinite(geometry.origin_m.x + width_m) ||
      !std::isfinite(geometry.origin_m.y + height_m)) {
    return false;
  }

  for (std::size_t index = 0U; index < count; ++index) {
    const auto known = projection.known_mask[index];
    const auto feasible = projection.hard_feasible_mask[index];
    const double clearance =
        static_cast<double>(projection.esdf_clearance_m[index]);
    if (known > 1U || feasible > 1U ||
        (feasible != 0U && known == 0U) ||
        !std::isfinite(clearance) || clearance < 0.0) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool ValidCanonicalUnitPlane(
    const HalfPlane2& plane) noexcept {
  if (!IsFinite(plane.outward_unit_normal) ||
      !std::isfinite(plane.upper_offset_m) ||
      HasNegativeZero(plane.outward_unit_normal.x) ||
      HasNegativeZero(plane.outward_unit_normal.y) ||
      HasNegativeZero(plane.upper_offset_m)) {
    return false;
  }
  const double norm =
      std::hypot(plane.outward_unit_normal.x,
                 plane.outward_unit_normal.y);
  return std::isfinite(norm) &&
         std::abs(norm - 1.0) <= kUnitNormalTolerance;
}

[[nodiscard]] HalfPlane2 NormalizePlane(
    HalfPlane2 plane) noexcept {
  const double norm =
      std::hypot(plane.outward_unit_normal.x,
                 plane.outward_unit_normal.y);
  plane.outward_unit_normal.x =
      CanonicalZero(plane.outward_unit_normal.x / norm);
  plane.outward_unit_normal.y =
      CanonicalZero(plane.outward_unit_normal.y / norm);
  plane.upper_offset_m =
      CanonicalZero(plane.upper_offset_m / norm);
  return plane;
}

[[nodiscard]] bool PlaneLess(const HalfPlane2& first,
                             const HalfPlane2& second) noexcept {
  return std::tie(first.outward_unit_normal.x,
                  first.outward_unit_normal.y,
                  first.upper_offset_m) <
         std::tie(second.outward_unit_normal.x,
                  second.outward_unit_normal.y,
                  second.upper_offset_m);
}

[[nodiscard]] bool SameNormal(const HalfPlane2& first,
                              const HalfPlane2& second) noexcept {
  return std::abs(first.outward_unit_normal.x -
                  second.outward_unit_normal.x) <=
             kGeometryTolerance &&
         std::abs(first.outward_unit_normal.y -
                  second.outward_unit_normal.y) <=
             kGeometryTolerance;
}

[[nodiscard]] double Evaluate(const HalfPlane2& plane,
                              const Vec2& point) noexcept {
  return plane.outward_unit_normal.x * point.x +
         plane.outward_unit_normal.y * point.y -
         plane.upper_offset_m;
}

[[nodiscard]] bool Contains(
    std::span<const HalfPlane2> planes,
    const Vec2& point) noexcept {
  return std::ranges::all_of(
      planes, [&](const HalfPlane2& plane) {
        return Evaluate(plane, point) <= kGeometryTolerance;
      });
}

[[nodiscard]] double PolygonArea(
    std::span<const Vec2> polygon) noexcept {
  if (polygon.size() < 3U) {
    return 0.0;
  }
  double doubled_area = 0.0;
  for (std::size_t index = 0U; index < polygon.size(); ++index) {
    const auto& current = polygon[index];
    const auto& next = polygon[(index + 1U) % polygon.size()];
    doubled_area += current.x * next.y - current.y * next.x;
  }
  return std::abs(doubled_area) * 0.5;
}

void AppendDistinct(std::vector<Vec2>& points,
                    Vec2 point) {
  point.x = CanonicalZero(point.x);
  point.y = CanonicalZero(point.y);
  if (!points.empty() &&
      std::hypot(points.back().x - point.x,
                 points.back().y - point.y) <=
          kGeometryTolerance) {
    return;
  }
  points.push_back(point);
}

[[nodiscard]] std::vector<Vec2> ClipPolygon(
    std::span<const Vec2> polygon,
    const HalfPlane2& plane) {
  std::vector<Vec2> clipped;
  if (polygon.empty()) {
    return clipped;
  }
  clipped.reserve(polygon.size() + 1U);
  for (std::size_t index = 0U; index < polygon.size(); ++index) {
    const Vec2& start = polygon[index];
    const Vec2& finish =
        polygon[(index + 1U) % polygon.size()];
    const double start_value = Evaluate(plane, start);
    const double finish_value = Evaluate(plane, finish);
    const bool start_inside =
        start_value <= kGeometryTolerance;
    const bool finish_inside =
        finish_value <= kGeometryTolerance;

    if (start_inside) {
      AppendDistinct(clipped, start);
    }
    if (start_inside == finish_inside) {
      continue;
    }
    const double denominator = start_value - finish_value;
    if (std::abs(denominator) <=
        std::numeric_limits<double>::epsilon()) {
      continue;
    }
    const double fraction =
        std::clamp(start_value / denominator, 0.0, 1.0);
    AppendDistinct(
        clipped,
        {
            .x = start.x + fraction * (finish.x - start.x),
            .y = start.y + fraction * (finish.y - start.y),
        });
  }
  if (clipped.size() > 1U &&
      std::hypot(clipped.front().x - clipped.back().x,
                 clipped.front().y - clipped.back().y) <=
          kGeometryTolerance) {
    clipped.pop_back();
  }
  return clipped;
}

[[nodiscard]] AddPlaneStatus AddPlane(
    HalfPlane2 plane, std::size_t max_planes,
    std::vector<HalfPlane2>& planes,
    std::vector<Vec2>& polygon) {
  plane = NormalizePlane(plane);
  for (auto& existing : planes) {
    if (!SameNormal(existing, plane)) {
      continue;
    }
    if (plane.upper_offset_m >=
        existing.upper_offset_m - kGeometryTolerance) {
      return AddPlaneStatus::kAddedOrRedundant;
    }
    existing.upper_offset_m = plane.upper_offset_m;
    polygon = ClipPolygon(polygon, plane);
    if (polygon.size() < 3U ||
        PolygonArea(polygon) <= kGeometryTolerance) {
      return AddPlaneStatus::kEmpty;
    }
    return AddPlaneStatus::kAddedOrRedundant;
  }

  if (planes.size() >= max_planes) {
    return AddPlaneStatus::kPlaneLimit;
  }
  planes.push_back(plane);
  polygon = ClipPolygon(polygon, plane);
  if (polygon.size() < 3U ||
      PolygonArea(polygon) <= kGeometryTolerance) {
    return AddPlaneStatus::kEmpty;
  }
  return AddPlaneStatus::kAddedOrRedundant;
}

[[nodiscard]] double MaximumProjection(
    std::span<const Vec2> polygon,
    const Vec2& direction) noexcept {
  double maximum =
      -std::numeric_limits<double>::infinity();
  for (const auto& point : polygon) {
    maximum = std::max(
        maximum,
        direction.x * point.x + direction.y * point.y);
  }
  return maximum;
}

[[nodiscard]] std::size_t CellIndex(
    const GridGeometry& geometry, std::size_t x,
    std::size_t y) noexcept {
  return y * geometry.width + x;
}

[[nodiscard]] bool PositionCell(
    const GridGeometry& geometry, const Vec2& position,
    std::size_t& x, std::size_t& y) noexcept {
  if (!IsFinite(position)) {
    return false;
  }
  const double grid_x =
      (position.x - geometry.origin_m.x) /
      geometry.resolution_m;
  const double grid_y =
      (position.y - geometry.origin_m.y) /
      geometry.resolution_m;
  if (!std::isfinite(grid_x) || !std::isfinite(grid_y) ||
      grid_x < 0.0 || grid_y < 0.0 ||
      grid_x >= static_cast<double>(geometry.width) ||
      grid_y >= static_cast<double>(geometry.height)) {
    return false;
  }
  x = static_cast<std::size_t>(std::floor(grid_x));
  y = static_cast<std::size_t>(std::floor(grid_y));
  return true;
}

[[nodiscard]] Vec2 GridCellCenter(
    const GridGeometry& geometry, std::size_t x,
    std::size_t y) noexcept {
  return {
      .x = geometry.origin_m.x +
           (static_cast<double>(x) + 0.5) *
               geometry.resolution_m,
      .y = geometry.origin_m.y +
           (static_cast<double>(y) + 0.5) *
               geometry.resolution_m,
  };
}

[[nodiscard]] bool SafeWithMargin(
    const SafeProjection& projection, std::size_t x,
    std::size_t y, double margin_m) noexcept {
  const std::size_t index =
      CellIndex(projection.geometry, x, y);
  return projection.known_mask[index] != 0U &&
         projection.hard_feasible_mask[index] != 0U &&
         static_cast<double>(
             projection.esdf_clearance_m[index]) >
             margin_m + kSafetyTolerance;
}

[[nodiscard]] std::vector<ForbiddenCandidate>
EnumerateForbiddenCandidates(
    const SafeProjection& projection, const Vec2& seed,
    double margin_m, std::span<const Vec2> seed_polygon) {
  std::vector<ForbiddenCandidate> candidates;
  const auto& geometry = projection.geometry;
  for (std::size_t y = 0U; y < geometry.height; ++y) {
    for (std::size_t x = 0U; x < geometry.width; ++x) {
      if (SafeWithMargin(projection, x, y, margin_m)) {
        continue;
      }
      const Vec2 center = GridCellCenter(geometry, x, y);
      const double delta_x = center.x - seed.x;
      const double delta_y = center.y - seed.y;
      const double distance = std::hypot(delta_x, delta_y);
      if (!std::isfinite(distance) ||
          distance <= kGeometryTolerance) {
        candidates.push_back(
            {
                .squared_distance_m2 = 0.0,
                .y = y,
                .x = x,
                .separating_plane =
                    {{1.0, 0.0},
                     -std::numeric_limits<double>::infinity()},
            });
        continue;
      }
      const Vec2 normal{
          .x = CanonicalZero(delta_x / distance),
          .y = CanonicalZero(delta_y / distance),
      };
      const double inflated_support =
          0.5 * geometry.resolution_m *
              (std::abs(normal.x) + std::abs(normal.y)) +
          margin_m;
      const double offset =
          normal.x * center.x + normal.y * center.y -
          inflated_support;
      if (MaximumProjection(seed_polygon, normal) <=
          offset + kGeometryTolerance) {
        continue;
      }
      candidates.push_back(
          {
              .squared_distance_m2 =
                  delta_x * delta_x + delta_y * delta_y,
              .y = y,
              .x = x,
              .separating_plane =
                  {
                      .outward_unit_normal = normal,
                      .upper_offset_m = CanonicalZero(offset),
                  },
          });
    }
  }
  std::ranges::sort(
      candidates,
      [](const ForbiddenCandidate& first,
         const ForbiddenCandidate& second) {
        return std::tie(first.squared_distance_m2, first.y,
                        first.x) <
               std::tie(second.squared_distance_m2, second.y,
                        second.x);
      });
  return candidates;
}

[[nodiscard]] bool HalfPlaneIntersectionNonempty(
    std::span<const HalfPlane2> first,
    std::span<const HalfPlane2> second) noexcept {
  std::vector<HalfPlane2> combined;
  combined.reserve(first.size() + second.size());
  combined.insert(combined.end(), first.begin(), first.end());
  combined.insert(combined.end(), second.begin(), second.end());
  for (std::size_t a = 0U; a < combined.size(); ++a) {
    for (std::size_t b = a + 1U; b < combined.size(); ++b) {
      const double determinant =
          combined[a].outward_unit_normal.x *
              combined[b].outward_unit_normal.y -
          combined[a].outward_unit_normal.y *
              combined[b].outward_unit_normal.x;
      if (std::abs(determinant) <= kGeometryTolerance) {
        continue;
      }
      const Vec2 candidate{
          .x =
              (combined[a].upper_offset_m *
                   combined[b].outward_unit_normal.y -
               combined[a].outward_unit_normal.y *
                   combined[b].upper_offset_m) /
              determinant,
          .y =
              (combined[a].outward_unit_normal.x *
                   combined[b].upper_offset_m -
               combined[a].upper_offset_m *
                   combined[b].outward_unit_normal.x) /
              determinant,
      };
      if (IsFinite(candidate) && Contains(combined, candidate)) {
        return true;
      }
    }
  }
  return false;
}

void HashByte(std::uint64_t& hash, std::uint8_t byte) noexcept {
  hash ^= byte;
  hash *= kFnvPrime;
}

void HashU64(std::uint64_t& hash, std::uint64_t value) noexcept {
  for (int shift = 56; shift >= 0; shift -= 8) {
    HashByte(hash, static_cast<std::uint8_t>(
                       value >> static_cast<unsigned>(shift)));
  }
}

[[nodiscard]] std::string HexU64(std::uint64_t value) {
  constexpr char kDigits[] = "0123456789abcdef";
  std::string output(16U, '0');
  for (std::size_t index = output.size(); index > 0U; --index) {
    output[index - 1U] = kDigits[value & 0xFU];
    value >>= 4U;
  }
  return output;
}

[[nodiscard]] std::string StableCellId(
    const ConvexCorridorCell& cell,
    std::size_t ordinal) {
  std::uint64_t hash = kFnvOffsetBasis;
  HashU64(hash, static_cast<std::uint64_t>(ordinal));
  HashU64(hash, std::bit_cast<std::uint64_t>(
                    CanonicalZero(cell.centerline_s_begin)));
  HashU64(hash, std::bit_cast<std::uint64_t>(
                    CanonicalZero(cell.centerline_s_end)));
  for (const auto& plane : cell.half_planes) {
    HashU64(hash, std::bit_cast<std::uint64_t>(
                      CanonicalZero(
                          plane.outward_unit_normal.x)));
    HashU64(hash, std::bit_cast<std::uint64_t>(
                      CanonicalZero(
                          plane.outward_unit_normal.y)));
    HashU64(hash, std::bit_cast<std::uint64_t>(
                      CanonicalZero(plane.upper_offset_m)));
  }
  return "corridor-cell-" + std::to_string(ordinal) + "-" +
         HexU64(hash);
}

[[nodiscard]] bool CellCertified(
    const BuiltCell& cell, const CorridorRequest& request,
    double margin_m) noexcept {
  if (cell.polygon.size() < 3U ||
      PolygonArea(cell.polygon) <= kGeometryTolerance ||
      !std::ranges::all_of(cell.polygon, IsFinite) ||
      cell.published.half_planes.size() < 4U ||
      cell.published.half_planes.size() >
          request.max_planes ||
      !std::ranges::all_of(cell.published.half_planes,
                           ValidCanonicalUnitPlane)) {
    return false;
  }

  for (std::size_t index = cell.centerline_begin;
       index <= cell.centerline_end; ++index) {
    if (!Contains(cell.published.half_planes,
                  request.validated_centerline[index]
                      .position_m)) {
      return false;
    }
  }

  const auto& geometry = request.projection.geometry;
  bool covered_a_grid_cell = false;
  for (std::size_t y = 0U; y < geometry.height; ++y) {
    for (std::size_t x = 0U; x < geometry.width; ++x) {
      const Vec2 center = GridCellCenter(geometry, x, y);
      if (!Contains(cell.published.half_planes, center)) {
        continue;
      }
      covered_a_grid_cell = true;
      if (!SafeWithMargin(request.projection, x, y,
                          margin_m)) {
        return false;
      }
    }
  }
  return covered_a_grid_cell;
}

}  // namespace

CorridorResult BuildConvexCorridor(
    const CorridorRequest& request) {
  if (!ValidProjection(request.projection)) {
    return InvalidRequest("invalid_projection");
  }
  if (request.projection.platform_type() ==
      PlatformType::kHopper) {
    return InvalidRequest("hopper_not_supported");
  }
  if (request.max_planes == 0U ||
      request.max_iterations == 0U) {
    return InvalidRequest("invalid_resource_bounds");
  }
  if (!IsNonNegativeFinite(
          request.tightening.footprint_support_radius_m) ||
      !IsNonNegativeFinite(
          request.tightening.tracking_error_bound_m) ||
      !IsNonNegativeFinite(
          request.tightening.additional_margin_m)) {
    return InvalidRequest("invalid_tightening");
  }
  const double margin_m =
      request.tightening.footprint_support_radius_m +
      request.tightening.tracking_error_bound_m +
      request.tightening.additional_margin_m;
  if (!std::isfinite(margin_m)) {
    return InvalidRequest("invalid_tightening");
  }
  if (request.validated_centerline.empty()) {
    return InvalidRequest("invalid_centerline");
  }
  for (std::size_t index = 0U;
       index < request.validated_centerline.size(); ++index) {
    const auto& point = request.validated_centerline[index];
    if (!std::isfinite(point.s_m) ||
        !IsFinite(point.position_m) ||
        (index > 0U &&
         !(request.validated_centerline[index - 1U].s_m <
           point.s_m))) {
      return InvalidRequest("invalid_centerline");
    }
  }

  std::vector<HalfPlane2> platform_planes =
      request.tightening.platform_half_planes;
  if (!std::ranges::all_of(platform_planes,
                           ValidCanonicalUnitPlane)) {
    return InvalidRequest("invalid_platform_half_plane");
  }
  for (auto& plane : platform_planes) {
    plane = NormalizePlane(plane);
  }
  std::ranges::sort(platform_planes, PlaneLess);

  const auto& projection = request.projection;
  const auto& geometry = projection.geometry;
  std::vector<std::pair<std::size_t, std::size_t>>
      centerline_cells;
  centerline_cells.reserve(
      request.validated_centerline.size());
  for (const auto& point : request.validated_centerline) {
    std::size_t x = 0U;
    std::size_t y = 0U;
    if (!PositionCell(geometry, point.position_m, x, y) ||
        !projection.Known(
            {static_cast<std::int32_t>(x),
             static_cast<std::int32_t>(y)}) ||
        !projection.HardFeasible(
            {static_cast<std::int32_t>(x),
             static_cast<std::int32_t>(y)})) {
      return Fallback("centerline_not_hard_feasible", 0U);
    }
    if (!SafeWithMargin(projection, x, y, margin_m)) {
      return Fallback("insufficient_clearance", 0U);
    }
    centerline_cells.emplace_back(x, y);
  }

  const double map_min_x =
      geometry.origin_m.x + margin_m;
  const double map_min_y =
      geometry.origin_m.y + margin_m;
  const double map_max_x =
      geometry.origin_m.x +
      static_cast<double>(geometry.width) *
          geometry.resolution_m -
      margin_m;
  const double map_max_y =
      geometry.origin_m.y +
      static_cast<double>(geometry.height) *
          geometry.resolution_m -
      margin_m;
  if (!std::isfinite(map_min_x) ||
      !std::isfinite(map_min_y) ||
      !std::isfinite(map_max_x) ||
      !std::isfinite(map_max_y) ||
      map_min_x >= map_max_x ||
      map_min_y >= map_max_y) {
    return Fallback("insufficient_clearance", 0U);
  }

  std::vector<BuiltCell> built_cells;
  std::size_t next_centerline = 0U;
  std::size_t iterations = 0U;
  while (next_centerline <
         request.validated_centerline.size()) {
    const std::size_t begin_index = next_centerline;
    const auto [seed_x, seed_y] =
        centerline_cells[begin_index];
    const auto& seed =
        request.validated_centerline[begin_index];
    const double clearance_m = static_cast<double>(
        projection.esdf_clearance_m[
            CellIndex(geometry, seed_x, seed_y)]);
    const double half_width_m =
        clearance_m - margin_m;
    if (!std::isfinite(half_width_m) ||
        half_width_m <= kSafetyTolerance) {
      return Fallback("insufficient_clearance", iterations);
    }

    const double minimum_x =
        std::max(seed.position_m.x - half_width_m,
                 map_min_x);
    const double maximum_x =
        std::min(seed.position_m.x + half_width_m,
                 map_max_x);
    const double minimum_y =
        std::max(seed.position_m.y - half_width_m,
                 map_min_y);
    const double maximum_y =
        std::min(seed.position_m.y + half_width_m,
                 map_max_y);
    if (!std::isfinite(minimum_x) ||
        !std::isfinite(maximum_x) ||
        !std::isfinite(minimum_y) ||
        !std::isfinite(maximum_y) ||
        minimum_x >= maximum_x ||
        minimum_y >= maximum_y ||
        seed.position_m.x < minimum_x -
                                kGeometryTolerance ||
        seed.position_m.x > maximum_x +
                                kGeometryTolerance ||
        seed.position_m.y < minimum_y -
                                kGeometryTolerance ||
        seed.position_m.y > maximum_y +
                                kGeometryTolerance) {
      return Fallback("empty_cell", iterations);
    }

    std::vector<Vec2> polygon{
        {minimum_x, minimum_y},
        {maximum_x, minimum_y},
        {maximum_x, maximum_y},
        {minimum_x, maximum_y},
    };
    std::vector<HalfPlane2> planes{
        {{-1.0, 0.0}, CanonicalZero(-minimum_x)},
        {{0.0, -1.0}, CanonicalZero(-minimum_y)},
        {{0.0, 1.0}, CanonicalZero(maximum_y)},
        {{1.0, 0.0}, CanonicalZero(maximum_x)},
    };
    if (planes.size() > request.max_planes) {
      return Fallback("plane_limit_exceeded", iterations);
    }

    const auto candidates = EnumerateForbiddenCandidates(
        projection, seed.position_m, margin_m, polygon);
    for (const auto& candidate : candidates) {
      if (iterations >= request.max_iterations) {
        return Fallback("iteration_limit_exceeded",
                        iterations);
      }
      ++iterations;
      if (!std::isfinite(
              candidate.separating_plane.upper_offset_m) ||
          Evaluate(candidate.separating_plane,
                   seed.position_m) >
              -kGeometryTolerance) {
        return Fallback("forbidden_cell_not_separable",
                        iterations);
      }
      if (MaximumProjection(
              polygon,
              candidate.separating_plane
                  .outward_unit_normal) <=
          candidate.separating_plane.upper_offset_m +
              kGeometryTolerance) {
        continue;
      }
      const auto status = AddPlane(
          candidate.separating_plane, request.max_planes,
          planes, polygon);
      if (status == AddPlaneStatus::kPlaneLimit) {
        return Fallback("plane_limit_exceeded", iterations);
      }
      if (status == AddPlaneStatus::kEmpty) {
        return Fallback("empty_cell", iterations);
      }
    }

    for (const auto& plane : platform_planes) {
      const auto status = AddPlane(
          plane, request.max_planes, planes, polygon);
      if (status == AddPlaneStatus::kPlaneLimit) {
        return Fallback("plane_limit_exceeded", iterations);
      }
      if (status == AddPlaneStatus::kEmpty) {
        return Fallback("empty_cell", iterations);
      }
    }
    if (!Contains(planes, seed.position_m)) {
      return Fallback("centerline_not_contained", iterations);
    }

    std::size_t end_index = begin_index;
    while (end_index + 1U <
               request.validated_centerline.size() &&
           Contains(
               planes,
               request.validated_centerline[end_index + 1U]
                   .position_m)) {
      ++end_index;
    }
    std::ranges::sort(planes, PlaneLess);
    ConvexCorridorCell published{
        .stable_cell_id = {},
        .centerline_s_begin =
            request.validated_centerline[begin_index].s_m,
        .centerline_s_end =
            request.validated_centerline[end_index].s_m,
        .half_planes = std::move(planes),
    };
    published.stable_cell_id =
        StableCellId(published, built_cells.size());
    built_cells.push_back(
        {
            .published = std::move(published),
            .polygon = std::move(polygon),
            .centerline_begin = begin_index,
            .centerline_end = end_index,
        });
    next_centerline = end_index + 1U;
  }

  for (const auto& cell : built_cells) {
    if (!CellCertified(cell, request, margin_m)) {
      return Fallback("cell_certification_failed",
                      iterations);
    }
  }
  for (std::size_t index = 1U; index < built_cells.size();
       ++index) {
    if (!HalfPlaneIntersectionNonempty(
            built_cells[index - 1U].published.half_planes,
            built_cells[index].published.half_planes)) {
      return Fallback("adjacent_cells_do_not_overlap",
                      iterations);
    }
  }

  CorridorResult result{
      .status = CorridorStatus::kCertified,
      .fallback = CorridorFallback::kNone,
      .cells = {},
      .reason_code = "certified",
      .iterations = iterations,
  };
  result.cells.reserve(built_cells.size());
  for (auto& cell : built_cells) {
    result.cells.push_back(std::move(cell.published));
  }
  return result;
}

}  // namespace lunar::planning::v3

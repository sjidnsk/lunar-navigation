#include "lunar_pure_planner_core/traversability_map.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

namespace lunar::pure_planning {
namespace {

constexpr std::int64_t kTileWidth = 256;
constexpr std::size_t kTileCellCount =
    static_cast<std::size_t>(kTileWidth * kTileWidth);
constexpr double kGeometryEpsilon = 1.0e-12;

struct TileCoordinate final {
  std::int64_t x{};
  std::int64_t y{};

  auto operator<=>(const TileCoordinate&) const = default;
};

struct Tile final {
  std::array<TraversabilityState, kTileCellCount> states{};

  Tile() { states.fill(TraversabilityState::kUnknown); }
};

struct SnapshotState final {
  TraversabilityProfile profile;
  std::optional<GridMap> global_map;
  std::map<TileCoordinate, std::shared_ptr<const Tile>> tiles;
  double resolution_m{};
  Vec3 origin_m;
  std::uint64_t revision{};
  std::uint64_t profile_hash{};
  std::size_t prior_conflicts{};
};

[[nodiscard]] bool Finite(const double value) noexcept {
  return std::isfinite(value);
}

[[nodiscard]] bool Finite(const Vec3 value) noexcept {
  return Finite(value.x) && Finite(value.y) && Finite(value.z);
}

[[nodiscard]] bool ProfileValid(const TraversabilityProfile& profile) noexcept {
  return Finite(profile.local_occupancy_threshold) &&
         Finite(profile.maximum_slope_rad) &&
         Finite(profile.inflation_radius_m) &&
         profile.local_occupancy_threshold >= 0.0 &&
         profile.local_occupancy_threshold <= 1.0 &&
         profile.maximum_slope_rad >= 0.0 &&
         profile.inflation_radius_m >= 0.0;
}

[[nodiscard]] std::uint64_t ProfileHash(
    const TraversabilityProfile& profile) noexcept {
  std::uint64_t hash = 1469598103934665603ULL;
  const auto append = [&hash](const std::uint64_t value) {
    hash ^= value;
    hash *= 1099511628211ULL;
  };
  append(static_cast<std::uint64_t>(
      static_cast<std::int64_t>(profile.global_occupancy_threshold)));
  append(std::bit_cast<std::uint64_t>(profile.local_occupancy_threshold));
  append(std::bit_cast<std::uint64_t>(profile.maximum_slope_rad));
  append(std::bit_cast<std::uint64_t>(profile.inflation_radius_m));
  return hash;
}

[[nodiscard]] bool GridGeometryValid(const GridMap& map) noexcept {
  return map.width != 0U && map.height != 0U && Finite(map.resolution_m) &&
         map.resolution_m > 0.0 && Finite(map.origin_m) &&
         map.HasConsistentLayerSizes();
}

[[nodiscard]] bool GlobalMapValid(const GridMap& map) noexcept {
  if (!GridGeometryValid(map) || map.frame_id != "map") {
    return false;
  }
  const auto found = map.layers.find("occupancy");
  return found != map.layers.end() &&
         std::holds_alternative<std::vector<std::int8_t>>(found->second.values);
}

[[nodiscard]] bool LocalMapValid(const GridMap& map) noexcept {
  if (!GridGeometryValid(map) || map.frame_id != "odom") {
    return false;
  }
  const auto occupancy = map.layers.find("occupancy");
  const auto elevation = map.layers.find("elevation");
  return occupancy != map.layers.end() && elevation != map.layers.end() &&
         std::holds_alternative<std::vector<float>>(occupancy->second.values) &&
         std::holds_alternative<std::vector<float>>(elevation->second.values);
}

[[nodiscard]] bool GridMapsEqual(const GridMap& left, const GridMap& right) {
  if (left.frame_id != right.frame_id || left.width != right.width ||
      left.height != right.height ||
      left.resolution_m != right.resolution_m || left.origin_m != right.origin_m ||
      left.layers.size() != right.layers.size()) {
    return false;
  }
  auto left_it = left.layers.begin();
  auto right_it = right.layers.begin();
  for (; left_it != left.layers.end(); ++left_it, ++right_it) {
    if (left_it->first != right_it->first ||
        left_it->second.values != right_it->second.values) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] std::int64_t FloorDivide(const std::int64_t value,
                                       const std::int64_t divisor) noexcept {
  const std::int64_t quotient = value / divisor;
  const std::int64_t remainder = value % divisor;
  return remainder < 0 ? quotient - 1 : quotient;
}

[[nodiscard]] std::int64_t PositiveModulo(const std::int64_t value,
                                           const std::int64_t divisor) noexcept {
  const std::int64_t remainder = value % divisor;
  return remainder < 0 ? remainder + divisor : remainder;
}

[[nodiscard]] std::optional<std::int64_t> WorldCellIndex(
    const double coordinate_m, const double origin_m,
    const double resolution_m) noexcept {
  if (!Finite(coordinate_m) || !Finite(origin_m) || !Finite(resolution_m) ||
      resolution_m <= 0.0) {
    return std::nullopt;
  }
  const double index = std::floor((coordinate_m - origin_m) / resolution_m);
  if (!Finite(index) ||
      index < static_cast<double>(std::numeric_limits<std::int64_t>::min()) ||
      index > static_cast<double>(std::numeric_limits<std::int64_t>::max())) {
    return std::nullopt;
  }
  return static_cast<std::int64_t>(index);
}

[[nodiscard]] TileCoordinate TileForCell(const std::int64_t x,
                                          const std::int64_t y) noexcept {
  return TileCoordinate{.x = FloorDivide(x, kTileWidth),
                        .y = FloorDivide(y, kTileWidth)};
}

[[nodiscard]] std::size_t TileOffset(const std::int64_t x,
                                      const std::int64_t y) noexcept {
  return static_cast<std::size_t>(PositiveModulo(y, kTileWidth)) *
             static_cast<std::size_t>(kTileWidth) +
         static_cast<std::size_t>(PositiveModulo(x, kTileWidth));
}

[[nodiscard]] TraversabilityState RawStateAt(const SnapshotState& state,
                                              const std::int64_t x,
                                              const std::int64_t y) noexcept {
  const auto found = state.tiles.find(TileForCell(x, y));
  if (found == state.tiles.end()) {
    return TraversabilityState::kUnknown;
  }
  return found->second->states[TileOffset(x, y)];
}

struct Rectangle final {
  double min_x{};
  double max_x{};
  double min_y{};
  double max_y{};
};

[[nodiscard]] Rectangle CellRectangle(const SnapshotState& state,
                                      const std::int64_t x,
                                      const std::int64_t y) noexcept {
  const double min_x = state.origin_m.x + static_cast<double>(x) * state.resolution_m;
  const double min_y = state.origin_m.y + static_cast<double>(y) * state.resolution_m;
  return Rectangle{.min_x = min_x,
                   .max_x = min_x + state.resolution_m,
                   .min_y = min_y,
                   .max_y = min_y + state.resolution_m};
}

[[nodiscard]] double RectangleDistance(const Rectangle& left,
                                       const Rectangle& right) noexcept {
  const double gap_x = std::max(
      {0.0, left.min_x - right.max_x, right.min_x - left.max_x});
  const double gap_y = std::max(
      {0.0, left.min_y - right.max_y, right.min_y - left.max_y});
  return std::hypot(gap_x, gap_y);
}

[[nodiscard]] bool InflatedByBlockedRawCell(const SnapshotState& state,
                                            const std::int64_t query_x,
                                            const std::int64_t query_y) {
  if (state.profile.inflation_radius_m <= 0.0) {
    return false;
  }
  const Rectangle query = CellRectangle(state, query_x, query_y);
  const std::int64_t halo = static_cast<std::int64_t>(std::ceil(
      state.profile.inflation_radius_m / state.resolution_m)) + 1;
  for (std::int64_t y = query_y - halo; y <= query_y + halo; ++y) {
    for (std::int64_t x = query_x - halo; x <= query_x + halo; ++x) {
      if (RawStateAt(state, x, y) == TraversabilityState::kBlocked &&
          RectangleDistance(query, CellRectangle(state, x, y)) <=
              state.profile.inflation_radius_m + kGeometryEpsilon) {
        return true;
      }
    }
  }
  return false;
}

[[nodiscard]] bool InflatedByGlobalBlockedCell(const SnapshotState& state,
                                                const std::int64_t query_x,
                                                const std::int64_t query_y) {
  if (!state.global_map || state.profile.inflation_radius_m <= 0.0) {
    return false;
  }
  const GridMap& global = *state.global_map;
  const auto* occupancy = std::get_if<std::vector<std::int8_t>>(
      &global.layers.at("occupancy").values);
  const Rectangle query = CellRectangle(state, query_x, query_y);
  const double radius = state.profile.inflation_radius_m;
  const auto first_x = WorldCellIndex(query.min_x - radius, global.origin_m.x,
                                      global.resolution_m);
  const auto last_x = WorldCellIndex(query.max_x + radius, global.origin_m.x,
                                     global.resolution_m);
  const auto first_y = WorldCellIndex(query.min_y - radius, global.origin_m.y,
                                      global.resolution_m);
  const auto last_y = WorldCellIndex(query.max_y + radius, global.origin_m.y,
                                     global.resolution_m);
  if (!first_x || !last_x || !first_y || !last_y) {
    return false;
  }
  const std::int64_t min_x = std::max<std::int64_t>(0, *first_x);
  const std::int64_t max_x = std::min<std::int64_t>(
      static_cast<std::int64_t>(global.width) - 1, *last_x);
  const std::int64_t min_y = std::max<std::int64_t>(0, *first_y);
  const std::int64_t max_y = std::min<std::int64_t>(
      static_cast<std::int64_t>(global.height) - 1, *last_y);
  for (std::int64_t y = min_y; y <= max_y; ++y) {
    for (std::int64_t x = min_x; x <= max_x; ++x) {
      const std::size_t index = static_cast<std::size_t>(y) * global.width +
                                static_cast<std::size_t>(x);
      if ((*occupancy)[index] < state.profile.global_occupancy_threshold) {
        continue;
      }
      const Rectangle source{
          .min_x = global.origin_m.x + static_cast<double>(x) * global.resolution_m,
          .max_x = global.origin_m.x + static_cast<double>(x + 1) * global.resolution_m,
          .min_y = global.origin_m.y + static_cast<double>(y) * global.resolution_m,
          .max_y = global.origin_m.y + static_cast<double>(y + 1) * global.resolution_m,
      };
      if (RectangleDistance(query, source) <= radius + kGeometryEpsilon) {
        return true;
      }
    }
  }
  return false;
}

[[nodiscard]] TraversabilityState GlobalPriorAt(const SnapshotState& state,
                                                 const std::int64_t x,
                                                 const std::int64_t y) noexcept {
  if (!state.global_map) {
    return TraversabilityState::kUnknown;
  }
  const GridMap& global = *state.global_map;
  const double center_x = state.origin_m.x +
                          (static_cast<double>(x) + 0.5) * state.resolution_m;
  const double center_y = state.origin_m.y +
                          (static_cast<double>(y) + 0.5) * state.resolution_m;
  const auto global_x = WorldCellIndex(center_x, global.origin_m.x,
                                       global.resolution_m);
  const auto global_y = WorldCellIndex(center_y, global.origin_m.y,
                                       global.resolution_m);
  if (!global_x || !global_y || *global_x < 0 || *global_y < 0 ||
      *global_x >= static_cast<std::int64_t>(global.width) ||
      *global_y >= static_cast<std::int64_t>(global.height)) {
    return TraversabilityState::kUnknown;
  }
  const auto* occupancy = std::get_if<std::vector<std::int8_t>>(
      &global.layers.at("occupancy").values);
  const std::size_t index = static_cast<std::size_t>(*global_y) * global.width +
                            static_cast<std::size_t>(*global_x);
  const std::int8_t value = (*occupancy)[index];
  if (value < 0) {
    return TraversabilityState::kUnknown;
  }
  return value >= state.profile.global_occupancy_threshold
             ? TraversabilityState::kBlocked
             : TraversabilityState::kFree;
}

[[nodiscard]] TraversabilityState StateAt(const SnapshotState& state,
                                          const double x_m,
                                          const double y_m) {
  if (state.resolution_m <= 0.0) {
    return TraversabilityState::kUnknown;
  }
  const auto x = WorldCellIndex(x_m, state.origin_m.x, state.resolution_m);
  const auto y = WorldCellIndex(y_m, state.origin_m.y, state.resolution_m);
  if (!x || !y) {
    return TraversabilityState::kUnknown;
  }
  const TraversabilityState raw = RawStateAt(state, *x, *y);
  if (raw == TraversabilityState::kBlocked ||
      InflatedByBlockedRawCell(state, *x, *y)) {
    return TraversabilityState::kBlocked;
  }
  if (raw == TraversabilityState::kFree) {
    return TraversabilityState::kFree;
  }
  return InflatedByGlobalBlockedCell(state, *x, *y)
             ? TraversabilityState::kBlocked
             : GlobalPriorAt(state, *x, *y);
}

[[nodiscard]] TraversabilityMetrics Metrics(const SnapshotState& state,
                                            const std::size_t updated_cells = 0U,
                                            const std::size_t dirty_tiles = 0U) {
  TraversabilityMetrics metrics{
      .revision = state.revision,
      .profile_hash = state.profile_hash,
      .allocated_tiles = state.tiles.size(),
      .estimated_bytes = state.tiles.size() * sizeof(Tile),
      .updated_cells = updated_cells,
      .dirty_tiles = dirty_tiles,
      .halo_recomputed_cells = 0U,
      .free_cells = 0U,
      .blocked_cells = 0U,
      .unknown_cells = 0U,
      .prior_conflicts = state.prior_conflicts,
  };
  for (const auto& [coordinate, tile] : state.tiles) {
    static_cast<void>(coordinate);
    for (const TraversabilityState value : tile->states) {
      switch (value) {
        case TraversabilityState::kFree:
          ++metrics.free_cells;
          break;
        case TraversabilityState::kBlocked:
          ++metrics.blocked_cells;
          break;
        case TraversabilityState::kUnknown:
          ++metrics.unknown_cells;
          break;
      }
    }
  }
  return metrics;
}

[[nodiscard]] TraversabilityUpdateResult Reject(const SnapshotState& state,
                                                std::string reason_code) {
  return TraversabilityUpdateResult{
      .accepted = false,
      .changed = false,
      .reason_code = std::move(reason_code),
      .metrics = Metrics(state),
  };
}

[[nodiscard]] std::optional<Quaternion> Normalize(const Quaternion& value) noexcept {
  if (!Finite(value.w) || !Finite(value.x) || !Finite(value.y) ||
      !Finite(value.z)) {
    return std::nullopt;
  }
  const double squared = value.w * value.w + value.x * value.x +
                         value.y * value.y + value.z * value.z;
  if (!Finite(squared) || squared <= kGeometryEpsilon) {
    return std::nullopt;
  }
  const double inverse = 1.0 / std::sqrt(squared);
  return Quaternion{.w = value.w * inverse,
                    .x = value.x * inverse,
                    .y = value.y * inverse,
                    .z = value.z * inverse};
}

[[nodiscard]] Vec3 Rotate(const Quaternion& rotation, const Vec3& point) noexcept {
  const double xx = rotation.x * rotation.x;
  const double yy = rotation.y * rotation.y;
  const double zz = rotation.z * rotation.z;
  const double xy = rotation.x * rotation.y;
  const double xz = rotation.x * rotation.z;
  const double yz = rotation.y * rotation.z;
  const double wx = rotation.w * rotation.x;
  const double wy = rotation.w * rotation.y;
  const double wz = rotation.w * rotation.z;
  return Vec3{
      .x = (1.0 - 2.0 * (yy + zz)) * point.x + 2.0 * (xy - wz) * point.y +
           2.0 * (xz + wy) * point.z,
      .y = 2.0 * (xy + wz) * point.x + (1.0 - 2.0 * (xx + zz)) * point.y +
           2.0 * (yz - wx) * point.z,
      .z = 2.0 * (xz - wy) * point.x + 2.0 * (yz + wx) * point.y +
           (1.0 - 2.0 * (xx + yy)) * point.z,
  };
}

[[nodiscard]] std::optional<Vec3> TransformPoint(const Vec3& point,
                                                  const RigidTransform& transform) {
  const auto rotation = Normalize(transform.rotation);
  if (!rotation || !Finite(point) || !Finite(transform.translation_m) ||
      transform.parent_frame != "map" || transform.child_frame != "odom") {
    return std::nullopt;
  }
  const Vec3 rotated = Rotate(*rotation, point);
  const Vec3 transformed{.x = rotated.x + transform.translation_m.x,
                         .y = rotated.y + transform.translation_m.y,
                         .z = rotated.z + transform.translation_m.z};
  return Finite(transformed) ? std::optional<Vec3>{transformed} : std::nullopt;
}

[[nodiscard]] double Dot(const Vec2& left, const Vec2& right) noexcept {
  return left.x * right.x + left.y * right.y;
}

[[nodiscard]] bool OverlapsOnAxis(const std::array<Vec2, 4U>& left,
                                  const std::array<Vec2, 4U>& right,
                                  const Vec2 axis) noexcept {
  if (std::hypot(axis.x, axis.y) <= kGeometryEpsilon) {
    return true;
  }
  double left_min = Dot(left.front(), axis);
  double left_max = left_min;
  double right_min = Dot(right.front(), axis);
  double right_max = right_min;
  for (const Vec2 point : left) {
    left_min = std::min(left_min, Dot(point, axis));
    left_max = std::max(left_max, Dot(point, axis));
  }
  for (const Vec2 point : right) {
    right_min = std::min(right_min, Dot(point, axis));
    right_max = std::max(right_max, Dot(point, axis));
  }
  return left_max + kGeometryEpsilon >= right_min &&
         right_max + kGeometryEpsilon >= left_min;
}

[[nodiscard]] bool IntersectsCellRectangle(const std::array<Vec2, 4U>& polygon,
                                           const Rectangle& rectangle) noexcept {
  const std::array<Vec2, 4U> cell{{
      {rectangle.min_x, rectangle.min_y},
      {rectangle.max_x, rectangle.min_y},
      {rectangle.max_x, rectangle.max_y},
      {rectangle.min_x, rectangle.max_y},
  }};
  for (const auto& shape : {polygon, cell}) {
    for (std::size_t index = 0U; index < shape.size(); ++index) {
      const Vec2 start = shape[index];
      const Vec2 end = shape[(index + 1U) % shape.size()];
      const Vec2 normal{.x = -(end.y - start.y), .y = end.x - start.x};
      if (!OverlapsOnAxis(polygon, cell, normal)) {
        return false;
      }
    }
  }
  return true;
}

[[nodiscard]] bool ContainsPoint(const std::array<Vec2, 4U>& polygon,
                                 const Vec2 point) noexcept {
  bool has_positive{};
  bool has_negative{};
  for (std::size_t index = 0U; index < polygon.size(); ++index) {
    const Vec2 start = polygon[index];
    const Vec2 end = polygon[(index + 1U) % polygon.size()];
    const double cross = (end.x - start.x) * (point.y - start.y) -
                         (end.y - start.y) * (point.x - start.x);
    has_positive = has_positive || cross > kGeometryEpsilon;
    has_negative = has_negative || cross < -kGeometryEpsilon;
    if (has_positive && has_negative) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] TraversabilityState LocalStateAt(const GridMap& local,
                                               const std::size_t x,
                                               const std::size_t y,
                                               const TraversabilityProfile& profile) {
  const auto& occupancy = std::get<std::vector<float>>(
      local.layers.at("occupancy").values);
  const auto& elevation = std::get<std::vector<float>>(
      local.layers.at("elevation").values);
  const std::size_t index = y * local.width + x;
  if (!Finite(occupancy[index]) || occupancy[index] < 0.0F ||
      occupancy[index] > 1.0F || !Finite(elevation[index])) {
    return TraversabilityState::kUnknown;
  }
  double maximum_gradient{};
  const auto consider_neighbor = [&](const std::size_t neighbor_x,
                                     const std::size_t neighbor_y) {
    const float neighbor = elevation[neighbor_y * local.width + neighbor_x];
    if (Finite(neighbor)) {
      maximum_gradient = std::max(
          maximum_gradient,
          std::abs(static_cast<double>(neighbor) - elevation[index]) /
              local.resolution_m);
    }
  };
  if (x > 0U) {
    consider_neighbor(x - 1U, y);
  }
  if (x + 1U < local.width) {
    consider_neighbor(x + 1U, y);
  }
  if (y > 0U) {
    consider_neighbor(x, y - 1U);
  }
  if (y + 1U < local.height) {
    consider_neighbor(x, y + 1U);
  }
  return occupancy[index] >= profile.local_occupancy_threshold ||
                 std::atan(maximum_gradient) > profile.maximum_slope_rad
             ? TraversabilityState::kBlocked
             : TraversabilityState::kFree;
}

[[nodiscard]] std::array<Vec2, 4U> TransformedCellPolygon(
    const GridMap& local, const std::size_t x, const std::size_t y,
    const RigidTransform& map_from_odom) {
  const double min_x = local.origin_m.x + static_cast<double>(x) * local.resolution_m;
  const double min_y = local.origin_m.y + static_cast<double>(y) * local.resolution_m;
  const double max_x = min_x + local.resolution_m;
  const double max_y = min_y + local.resolution_m;
  const std::array<Vec3, 4U> corners{{
      {min_x, min_y, local.origin_m.z},
      {max_x, min_y, local.origin_m.z},
      {max_x, max_y, local.origin_m.z},
      {min_x, max_y, local.origin_m.z},
  }};
  std::array<Vec2, 4U> polygon{};
  for (std::size_t index = 0U; index < corners.size(); ++index) {
    const Vec3 transformed = *TransformPoint(corners[index], map_from_odom);
    polygon[index] = Vec2{.x = transformed.x, .y = transformed.y};
  }
  return polygon;
}

[[nodiscard]] bool WriteRawCell(
    SnapshotState& next, const std::int64_t x, const std::int64_t y,
    const TraversabilityState value,
    std::map<TileCoordinate, std::shared_ptr<Tile>>& writable_tiles,
    std::size_t& updated_cells) {
  const TileCoordinate coordinate = TileForCell(x, y);
  auto writable = writable_tiles.find(coordinate);
  if (writable == writable_tiles.end()) {
    const auto existing = next.tiles.find(coordinate);
    auto clone = existing == next.tiles.end()
                     ? std::make_shared<Tile>()
                     : std::make_shared<Tile>(*existing->second);
    next.tiles[coordinate] = clone;
    writable = writable_tiles.emplace(coordinate, std::move(clone)).first;
  }
  TraversabilityState& old = writable->second->states[TileOffset(x, y)];
  if (old == value) {
    return false;
  }
  old = value;
  ++updated_cells;
  return true;
}

[[nodiscard]] std::size_t CountPriorConflicts(const SnapshotState& state) {
  if (!state.global_map) {
    return 0U;
  }
  std::size_t conflicts{};
  for (const auto& [tile_coordinate, tile] : state.tiles) {
    for (std::size_t offset = 0U; offset < tile->states.size(); ++offset) {
      const TraversabilityState local = tile->states[offset];
      if (local == TraversabilityState::kUnknown) {
        continue;
      }
      const std::int64_t x = tile_coordinate.x * kTileWidth +
                             static_cast<std::int64_t>(offset % kTileWidth);
      const std::int64_t y = tile_coordinate.y * kTileWidth +
                             static_cast<std::int64_t>(offset / kTileWidth);
      const TraversabilityState prior = GlobalPriorAt(state, x, y);
      if (prior != TraversabilityState::kUnknown && prior != local) {
        ++conflicts;
      }
    }
  }
  return conflicts;
}

void ReanchorRawTiles(const SnapshotState& previous, SnapshotState& next) {
  if (previous.resolution_m <= 0.0 ||
      previous.origin_m == next.origin_m) {
    return;
  }
  next.tiles.clear();
  std::map<TileCoordinate, std::shared_ptr<Tile>> writable_tiles;
  std::size_t ignored_updates{};
  for (const auto& [tile_coordinate, tile] : previous.tiles) {
    for (std::size_t offset = 0U; offset < tile->states.size(); ++offset) {
      const TraversabilityState source = tile->states[offset];
      if (source == TraversabilityState::kUnknown) {
        continue;
      }
      const std::int64_t source_x = tile_coordinate.x * kTileWidth +
                                    static_cast<std::int64_t>(offset % kTileWidth);
      const std::int64_t source_y = tile_coordinate.y * kTileWidth +
                                    static_cast<std::int64_t>(offset / kTileWidth);
      const Rectangle source_rectangle =
          CellRectangle(previous, source_x, source_y);
      const auto first_x = WorldCellIndex(source_rectangle.min_x, next.origin_m.x,
                                          next.resolution_m);
      const auto last_x = WorldCellIndex(
          std::nextafter(source_rectangle.max_x,
                         -std::numeric_limits<double>::infinity()),
          next.origin_m.x, next.resolution_m);
      const auto first_y = WorldCellIndex(source_rectangle.min_y, next.origin_m.y,
                                          next.resolution_m);
      const auto last_y = WorldCellIndex(
          std::nextafter(source_rectangle.max_y,
                         -std::numeric_limits<double>::infinity()),
          next.origin_m.y, next.resolution_m);
      if (!first_x || !last_x || !first_y || !last_y) {
        continue;
      }
      for (std::int64_t target_y = *first_y; target_y <= *last_y; ++target_y) {
        for (std::int64_t target_x = *first_x; target_x <= *last_x; ++target_x) {
          const Rectangle target = CellRectangle(next, target_x, target_y);
          const bool covered =
              IntersectsCellRectangle(
                  std::array<Vec2, 4U>{{
                      {source_rectangle.min_x, source_rectangle.min_y},
                      {source_rectangle.max_x, source_rectangle.min_y},
                      {source_rectangle.max_x, source_rectangle.max_y},
                      {source_rectangle.min_x, source_rectangle.max_y},
                  }},
                  target);
          const double target_center_x = (target.min_x + target.max_x) / 2.0;
          const double target_center_y = (target.min_y + target.max_y) / 2.0;
          const bool free_center_covered =
              source_rectangle.min_x <= target_center_x &&
              target_center_x <= source_rectangle.max_x &&
              source_rectangle.min_y <= target_center_y &&
              target_center_y <= source_rectangle.max_y;
          if (covered && (source == TraversabilityState::kBlocked ||
                          free_center_covered)) {
            static_cast<void>(WriteRawCell(next, target_x, target_y, source,
                                            writable_tiles, ignored_updates));
          }
        }
      }
    }
  }
}

}  // namespace

struct TraversabilitySnapshot::Impl final {
  SnapshotState state;
};

struct PersistentTraversabilityMap::Impl final {
  std::shared_ptr<const TraversabilitySnapshot::Impl> snapshot;
};

TraversabilitySnapshot::TraversabilitySnapshot(
    std::shared_ptr<const Impl> impl) noexcept
    : impl_(std::move(impl)) {}

bool TraversabilitySnapshot::valid() const noexcept {
  return impl_ != nullptr && Finite(impl_->state.resolution_m) &&
         impl_->state.resolution_m > 0.0;
}

double TraversabilitySnapshot::resolution_m() const noexcept {
  return impl_ ? impl_->state.resolution_m : 0.0;
}

Vec3 TraversabilitySnapshot::origin_m() const noexcept {
  return impl_ ? impl_->state.origin_m : Vec3{};
}

std::uint64_t TraversabilitySnapshot::revision() const noexcept {
  return impl_ ? impl_->state.revision : 0U;
}

std::uint64_t TraversabilitySnapshot::profile_hash() const noexcept {
  return impl_ ? impl_->state.profile_hash : 0U;
}

TraversabilityState TraversabilitySnapshot::StateAtWorld(const double x_m,
                                                          const double y_m) const {
  return impl_ ? StateAt(impl_->state, x_m, y_m) : TraversabilityState::kUnknown;
}

TraversabilityMetrics TraversabilitySnapshot::metrics() const noexcept {
  return impl_ ? Metrics(impl_->state) : TraversabilityMetrics{};
}

PersistentTraversabilityMap::PersistentTraversabilityMap(
    TraversabilityProfile profile)
    : impl_(std::make_unique<Impl>()) {
  auto initial = std::make_shared<TraversabilitySnapshot::Impl>();
  initial->state.profile = profile;
  initial->state.profile_hash = ProfileHash(profile);
  impl_->snapshot = std::move(initial);
}

PersistentTraversabilityMap::~PersistentTraversabilityMap() = default;
PersistentTraversabilityMap::PersistentTraversabilityMap(
    PersistentTraversabilityMap&&) noexcept = default;
PersistentTraversabilityMap& PersistentTraversabilityMap::operator=(
    PersistentTraversabilityMap&&) noexcept = default;

TraversabilityUpdateResult PersistentTraversabilityMap::UpdateGlobal(
    const GridMap& global) {
  const SnapshotState& previous = impl_->snapshot->state;
  if (!ProfileValid(previous.profile)) {
    return Reject(previous, "INVALID_INPUT");
  }
  if (!GlobalMapValid(global)) {
    return Reject(previous, "GLOBAL_MAP_GEOMETRY_INVALID");
  }
  if (previous.global_map && GridMapsEqual(*previous.global_map, global)) {
    return TraversabilityUpdateResult{
        .accepted = true,
        .changed = false,
        .reason_code = "",
        .metrics = Metrics(previous),
    };
  }
  auto next = std::make_shared<TraversabilitySnapshot::Impl>();
  next->state = previous;
  next->state.global_map = global;
  if (next->state.resolution_m > 0.0 &&
      next->state.origin_m != global.origin_m) {
    next->state.origin_m = global.origin_m;
    ReanchorRawTiles(previous, next->state);
  } else if (next->state.resolution_m == 0.0) {
    next->state.origin_m = global.origin_m;
  }
  next->state.prior_conflicts = CountPriorConflicts(next->state);
  ++next->state.revision;
  impl_->snapshot = next;
  return TraversabilityUpdateResult{
      .accepted = true,
      .changed = true,
      .reason_code = "",
      .metrics = Metrics(next->state),
  };
}

TraversabilityUpdateResult PersistentTraversabilityMap::UpdateLocal(
    const GridMap& local, const RigidTransform& map_from_odom,
    const std::uint64_t local_sequence) {
  static_cast<void>(local_sequence);
  const SnapshotState& previous = impl_->snapshot->state;
  if (!ProfileValid(previous.profile) || !LocalMapValid(local) ||
      !TransformPoint(local.origin_m, map_from_odom)) {
    return Reject(previous, "INVALID_INPUT");
  }
  if (previous.resolution_m > 0.0 &&
      std::abs(previous.resolution_m - local.resolution_m) >
          std::max(1.0e-9, previous.resolution_m * 1.0e-6)) {
    return Reject(previous, "MAP_RESOLUTION_MISMATCH");
  }

  auto next = std::make_shared<TraversabilitySnapshot::Impl>();
  next->state = previous;
  bool changed = false;
  if (next->state.resolution_m == 0.0) {
    next->state.resolution_m = local.resolution_m;
    next->state.origin_m = next->state.global_map
                                ? next->state.global_map->origin_m
                                : *TransformPoint(local.origin_m, map_from_odom);
    changed = true;
  }

  std::map<TileCoordinate, std::shared_ptr<Tile>> writable_tiles;
  std::size_t updated_cells{};
  for (std::size_t source_y = 0U; source_y < local.height; ++source_y) {
    for (std::size_t source_x = 0U; source_x < local.width; ++source_x) {
      const TraversabilityState source =
          LocalStateAt(local, source_x, source_y, previous.profile);
      if (source == TraversabilityState::kUnknown) {
        continue;
      }
      const auto polygon =
          TransformedCellPolygon(local, source_x, source_y, map_from_odom);
      double min_x = polygon.front().x;
      double max_x = min_x;
      double min_y = polygon.front().y;
      double max_y = min_y;
      for (const Vec2 point : polygon) {
        min_x = std::min(min_x, point.x);
        max_x = std::max(max_x, point.x);
        min_y = std::min(min_y, point.y);
        max_y = std::max(max_y, point.y);
      }
      const auto first_x = WorldCellIndex(min_x, next->state.origin_m.x,
                                          next->state.resolution_m);
      const auto last_x = WorldCellIndex(
          std::nextafter(max_x, -std::numeric_limits<double>::infinity()),
          next->state.origin_m.x, next->state.resolution_m);
      const auto first_y = WorldCellIndex(min_y, next->state.origin_m.y,
                                          next->state.resolution_m);
      const auto last_y = WorldCellIndex(
          std::nextafter(max_y, -std::numeric_limits<double>::infinity()),
          next->state.origin_m.y, next->state.resolution_m);
      if (!first_x || !last_x || !first_y || !last_y ||
          *last_x - *first_x > 1'000'000 || *last_y - *first_y > 1'000'000) {
        return Reject(previous, "INVALID_INPUT");
      }
      for (std::int64_t target_y = *first_y; target_y <= *last_y; ++target_y) {
        for (std::int64_t target_x = *first_x; target_x <= *last_x; ++target_x) {
          const Rectangle target =
              CellRectangle(next->state, target_x, target_y);
          const bool covered = IntersectsCellRectangle(polygon, target);
          const bool free_center_covered = ContainsPoint(
              polygon, Vec2{.x = (target.min_x + target.max_x) / 2.0,
                            .y = (target.min_y + target.max_y) / 2.0});
          if (covered && (source == TraversabilityState::kBlocked ||
                          free_center_covered)) {
            const bool wrote = WriteRawCell(next->state, target_x, target_y,
                                             source, writable_tiles, updated_cells);
            changed = wrote || changed;
          } else if (covered && source == TraversabilityState::kFree &&
                     RawStateAt(next->state, target_x, target_y) !=
                         TraversabilityState::kUnknown) {
            const bool cleared = WriteRawCell(
                next->state, target_x, target_y, TraversabilityState::kUnknown,
                writable_tiles, updated_cells);
            changed = cleared || changed;
          }
        }
      }
    }
  }
  next->state.prior_conflicts = CountPriorConflicts(next->state);
  if (!changed) {
    return TraversabilityUpdateResult{
        .accepted = true,
        .changed = false,
        .reason_code = "",
        .metrics = Metrics(previous),
    };
  }
  ++next->state.revision;
  impl_->snapshot = next;
  return TraversabilityUpdateResult{
      .accepted = true,
      .changed = true,
      .reason_code = "",
      .metrics = Metrics(next->state, updated_cells, writable_tiles.size()),
  };
}

std::shared_ptr<const TraversabilitySnapshot> PersistentTraversabilityMap::Capture() const {
  return std::shared_ptr<const TraversabilitySnapshot>(
      new TraversabilitySnapshot(impl_->snapshot));
}

}  // namespace lunar::pure_planning

#include "lunar_incremental_navigation_core/elevation_map.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

namespace lunar::incremental_navigation {
namespace {

struct ElevationTile final {
  std::array<ElevationRange, kGridTileCellCount> values;

  ElevationTile() {
    const float unknown = std::numeric_limits<float>::quiet_NaN();
    values.fill(ElevationRange{.min_m = unknown, .max_m = unknown});
  }
};

struct ElevationLineage final {};

[[nodiscard]] bool Finite(const double value) noexcept {
  return std::isfinite(value);
}

[[nodiscard]] bool Finite(const Vec3 value) noexcept {
  return Finite(value.x) && Finite(value.y) && Finite(value.z);
}

[[nodiscard]] bool Finite(const Quaternion value) noexcept {
  return Finite(value.w) && Finite(value.x) && Finite(value.y) &&
         Finite(value.z);
}

[[nodiscard]] bool ValidGeometry(const ElevationEvidence& evidence) noexcept {
  const GridGeometry& geometry = evidence.geometry;
  if (geometry.width == 0U || geometry.height == 0U ||
      geometry.width > std::numeric_limits<std::size_t>::max() /
                           geometry.height ||
      evidence.elevation_m.size() != geometry.width * geometry.height ||
      geometry.frame_id.empty() || !Finite(geometry.resolution_m) ||
      geometry.resolution_m <= 0.0 || !Finite(geometry.origin_m)) {
    return false;
  }
  const RigidTransform& transform = evidence.map_from_source;
  if (transform.parent_frame.empty() ||
      transform.child_frame != geometry.frame_id ||
      !Finite(transform.translation_m) || !Finite(transform.rotation)) {
    return false;
  }
  const Quaternion q = transform.rotation;
  const double norm_squared =
      q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z;
  return Finite(norm_squared) && norm_squared > 1.0e-20;
}

[[nodiscard]] Quaternion Normalized(const Quaternion value) noexcept {
  const double norm = std::sqrt(value.w * value.w + value.x * value.x +
                                value.y * value.y + value.z * value.z);
  return Quaternion{.w = value.w / norm,
                    .x = value.x / norm,
                    .y = value.y / norm,
                    .z = value.z / norm};
}

[[nodiscard]] Vec3 Rotate(const Quaternion q, const Vec3 point) noexcept {
  const Vec3 twice_cross{
      .x = 2.0 * (q.y * point.z - q.z * point.y),
      .y = 2.0 * (q.z * point.x - q.x * point.z),
      .z = 2.0 * (q.x * point.y - q.y * point.x),
  };
  return Vec3{
      .x = point.x + q.w * twice_cross.x +
           (q.y * twice_cross.z - q.z * twice_cross.y),
      .y = point.y + q.w * twice_cross.y +
           (q.z * twice_cross.x - q.x * twice_cross.z),
      .z = point.z + q.w * twice_cross.z +
           (q.x * twice_cross.y - q.y * twice_cross.x),
  };
}

[[nodiscard]] Vec3 TransformPoint(const RigidTransform& transform,
                                  const Vec3 point) noexcept {
  const Vec3 rotated = Rotate(Normalized(transform.rotation), point);
  return Vec3{.x = rotated.x + transform.translation_m.x,
              .y = rotated.y + transform.translation_m.y,
              .z = rotated.z + transform.translation_m.z};
}

[[nodiscard]] std::optional<std::int64_t> CellCoordinate(
    const double world_m, const double origin_m,
    const double resolution_m) noexcept {
  const double value = std::floor((world_m - origin_m) / resolution_m);
  if (!Finite(value) ||
      value < static_cast<double>(std::numeric_limits<std::int64_t>::min()) ||
      value >= static_cast<double>(std::numeric_limits<std::int64_t>::max())) {
    return std::nullopt;
  }
  return static_cast<std::int64_t>(value);
}

struct InputBounds final {
  GridIndex minimum;
  GridIndex maximum;
};

// Preserve the source rectangle independently of its finite height samples.
// Project its reference-plane corners; actual 3D samples can expand it further.
[[nodiscard]] std::optional<InputBounds> ProjectInputBounds(
    const ElevationEvidence& evidence, const Vec3 canonical_origin) {
  const auto& geometry = evidence.geometry;
  const double resolution = geometry.resolution_m;
  double min_x = std::numeric_limits<double>::infinity();
  double min_y = min_x;
  double max_x = -min_x;
  double max_y = -min_y;
  for (const std::size_t y : {std::size_t{0}, geometry.height}) {
    for (const std::size_t x : {std::size_t{0}, geometry.width}) {
      const Vec3 corner =
          TransformPoint(evidence.map_from_source,
                         {.x = std::fma(static_cast<double>(x), resolution,
                                        geometry.origin_m.x),
                          .y = std::fma(static_cast<double>(y), resolution,
                                        geometry.origin_m.y),
                          .z = geometry.origin_m.z});
      if (!Finite(corner)) return std::nullopt;
      const double grid_x = (corner.x - canonical_origin.x) / resolution;
      const double grid_y = (corner.y - canonical_origin.y) / resolution;
      min_x = std::min(min_x, grid_x);
      max_x = std::max(max_x, grid_x);
      min_y = std::min(min_y, grid_y);
      max_y = std::max(max_y, grid_y);
    }
  }
  const auto boundary = [](double value,
                           bool upper) -> std::optional<std::int64_t> {
    if (!Finite(value)) return std::nullopt;
    // Avoid an extra UNKNOWN row for roundoff at exact grid-aligned corners.
    const double nearest = std::round(value);
    if (std::abs(value - nearest) <= 1.0e-9) value = nearest;
    value = upper ? std::ceil(value) : std::floor(value);
    if (value < static_cast<double>(std::numeric_limits<std::int64_t>::min()) ||
        value >=
            static_cast<double>(std::numeric_limits<std::int64_t>::max())) {
      return std::nullopt;
    }
    return static_cast<std::int64_t>(value);
  };
  const auto x0 = boundary(min_x, false), y0 = boundary(min_y, false);
  const auto x1 = boundary(max_x, true), y1 = boundary(max_y, true);
  if (!x0 || !y0 || !x1 || !y1) return std::nullopt;
  return InputBounds{
      .minimum = {*x0, *y0},
      .maximum = {std::max(*x1, *x0 + 1), std::max(*y1, *y0 + 1)}};
}

[[nodiscard]] std::size_t IndexSpan(const std::int64_t min_inclusive,
                                    const std::int64_t max_exclusive) noexcept {
  if (max_exclusive <= min_inclusive) {
    return 0U;
  }
  constexpr std::uint64_t kSignBit = std::uint64_t{1} << 63U;
  const std::uint64_t ordered_min =
      static_cast<std::uint64_t>(min_inclusive) ^ kSignBit;
  const std::uint64_t ordered_max =
      static_cast<std::uint64_t>(max_exclusive) ^ kSignBit;
  const std::uint64_t span = ordered_max - ordered_min;
  if (span > std::numeric_limits<std::size_t>::max()) {
    return 0U;
  }
  return static_cast<std::size_t>(span);
}

}  // namespace

struct ElevationSnapshot::Impl final {
  std::shared_ptr<const ElevationLineage> lineage;
  SparseGridGeometry geometry;
  std::uint64_t revision{};
  std::map<TileIndex, std::shared_ptr<const ElevationTile>> tiles;
  std::weak_ptr<const Impl> previous;
  std::vector<GridIndex> changed_cells;
  std::vector<TileIndex> changed_tiles;

  [[nodiscard]] std::optional<ElevationRange> ElevationRangeAt(
      const GridIndex index) const noexcept {
    const auto found = tiles.find(TileForCell(index));
    if (found == tiles.end()) {
      return std::nullopt;
    }
    const ElevationRange value = found->second->values[TileCellOffset(index)];
    return std::isfinite(value.min_m) && std::isfinite(value.max_m) &&
                   value.min_m <= value.max_m
               ? std::optional<ElevationRange>(value)
               : std::nullopt;
  }

  [[nodiscard]] std::size_t EstimatedBytes() const noexcept {
    return tiles.size() * sizeof(ElevationTile);
  }
};

struct PersistentElevationMap::Impl final {
  mutable std::mutex mutex;
  std::shared_ptr<const ElevationLineage> lineage =
      std::make_shared<const ElevationLineage>();
  std::shared_ptr<const ElevationSnapshot::Impl> state;
  std::shared_ptr<const ElevationSnapshot> snapshot;
  ElevationMapCounters counters;
};

SparseGridGeometry::SparseGridGeometry(std::string frame_id,
                                       const double resolution_m,
                                       const Vec3 origin_m,
                                       const GridIndex min_inclusive,
                                       const GridIndex max_exclusive) noexcept
    : frame_id_(std::move(frame_id)),
      resolution_m_(resolution_m),
      origin_m_(origin_m),
      min_inclusive_(min_inclusive),
      max_exclusive_(max_exclusive) {}

bool SparseGridGeometry::valid() const noexcept {
  if (frame_id_.empty() || !std::isfinite(resolution_m_) ||
      resolution_m_ <= 0.0 || !std::isfinite(origin_m_.x) ||
      !std::isfinite(origin_m_.y) || !std::isfinite(origin_m_.z) || empty() ||
      CellCount() == 0U) {
    return false;
  }
  return std::isfinite(std::fma(static_cast<double>(min_inclusive_.x),
                                resolution_m_, origin_m_.x)) &&
         std::isfinite(std::fma(static_cast<double>(max_exclusive_.x),
                                resolution_m_, origin_m_.x)) &&
         std::isfinite(std::fma(static_cast<double>(min_inclusive_.y),
                                resolution_m_, origin_m_.y)) &&
         std::isfinite(std::fma(static_cast<double>(max_exclusive_.y),
                                resolution_m_, origin_m_.y));
}

const std::string& SparseGridGeometry::frame_id() const noexcept {
  return frame_id_;
}

double SparseGridGeometry::resolution_m() const noexcept {
  return resolution_m_;
}

Vec3 SparseGridGeometry::origin_m() const noexcept { return origin_m_; }

GridIndex SparseGridGeometry::min_inclusive() const noexcept {
  return min_inclusive_;
}

GridIndex SparseGridGeometry::max_exclusive() const noexcept {
  return max_exclusive_;
}

bool SparseGridGeometry::empty() const noexcept {
  return max_exclusive_.x <= min_inclusive_.x ||
         max_exclusive_.y <= min_inclusive_.y;
}

std::size_t SparseGridGeometry::width() const noexcept {
  return IndexSpan(min_inclusive_.x, max_exclusive_.x);
}

std::size_t SparseGridGeometry::height() const noexcept {
  return IndexSpan(min_inclusive_.y, max_exclusive_.y);
}

std::size_t SparseGridGeometry::CellCount() const noexcept {
  const std::size_t span_width = width();
  const std::size_t span_height = height();
  if (span_width == 0U || span_height == 0U ||
      span_width > std::numeric_limits<std::size_t>::max() / span_height) {
    return 0U;
  }
  return span_width * span_height;
}

bool SparseGridGeometry::Contains(const GridIndex index) const noexcept {
  return valid() && index.x >= min_inclusive_.x &&
         index.x < max_exclusive_.x && index.y >= min_inclusive_.y &&
         index.y < max_exclusive_.y;
}

bool SparseGridGeometry::Contains(const TileIndex index) const noexcept {
  if (!valid()) {
    return false;
  }
  const TileIndex min_tile = TileForCell(min_inclusive_);
  const TileIndex max_tile = TileForCell(
      GridIndex{.x = max_exclusive_.x - 1, .y = max_exclusive_.y - 1});
  return index.x >= min_tile.x && index.x <= max_tile.x &&
         index.y >= min_tile.y && index.y <= max_tile.y;
}

ElevationSnapshot::ElevationSnapshot(std::shared_ptr<const Impl> impl) noexcept
    : impl_(std::move(impl)) {}

bool ElevationSnapshot::valid() const noexcept {
  return impl_ && impl_->geometry.valid();
}

const SparseGridGeometry& ElevationSnapshot::geometry() const noexcept {
  static const SparseGridGeometry empty;
  return impl_ ? impl_->geometry : empty;
}

std::uint64_t ElevationSnapshot::raw_elevation_revision() const noexcept {
  return impl_ ? impl_->revision : 0U;
}

std::optional<ElevationRange> ElevationSnapshot::ElevationRangeAt(
    const GridIndex index) const noexcept {
  return impl_ && impl_->geometry.Contains(index)
             ? impl_->ElevationRangeAt(index)
             : std::nullopt;
}

std::optional<ElevationRange> ElevationSnapshot::ElevationRangeAtWorld(
    const double x_m, const double y_m) const noexcept {
  if (!valid() || !Finite(x_m) || !Finite(y_m)) {
    return std::nullopt;
  }
  const Vec3 origin_m = impl_->geometry.origin_m();
  const auto x =
      CellCoordinate(x_m, origin_m.x, impl_->geometry.resolution_m());
  const auto y =
      CellCoordinate(y_m, origin_m.y, impl_->geometry.resolution_m());
  return x && y ? ElevationRangeAt(GridIndex{.x = *x, .y = *y})
                : std::nullopt;
}

std::optional<float> ElevationSnapshot::ElevationAt(
    const GridIndex index) const noexcept {
  const auto range = ElevationRangeAt(index);
  if (!range) {
    return std::nullopt;
  }
  return static_cast<float>(
      (static_cast<double>(range->min_m) +
       static_cast<double>(range->max_m)) /
      2.0);
}

std::optional<float> ElevationSnapshot::ElevationAtWorld(
    const double x_m, const double y_m) const noexcept {
  const auto range = ElevationRangeAtWorld(x_m, y_m);
  if (!range) {
    return std::nullopt;
  }
  return static_cast<float>(
      (static_cast<double>(range->min_m) +
       static_cast<double>(range->max_m)) /
      2.0);
}

std::span<const GridIndex> ElevationSnapshot::changed_cells() const noexcept {
  return impl_ ? std::span<const GridIndex>(impl_->changed_cells)
               : std::span<const GridIndex>();
}

std::span<const TileIndex> ElevationSnapshot::changed_tiles() const noexcept {
  return impl_ ? std::span<const TileIndex>(impl_->changed_tiles)
               : std::span<const TileIndex>();
}

bool ElevationSnapshot::IsDirectSuccessorOf(
    const ElevationSnapshot& previous) const noexcept {
  return impl_ && previous.impl_ && impl_->previous.lock() == previous.impl_;
}

bool ElevationSnapshot::SharesLineageWith(
    const ElevationSnapshot& other) const noexcept {
  return impl_ && other.impl_ && impl_->lineage &&
         impl_->lineage == other.impl_->lineage;
}

std::vector<TileIndex> ElevationSnapshot::tile_indices() const {
  std::vector<TileIndex> result;
  if (!impl_) {
    return result;
  }
  result.reserve(impl_->tiles.size());
  for (const auto& [index, tile] : impl_->tiles) {
    static_cast<void>(tile);
    result.push_back(index);
  }
  return result;
}

std::size_t ElevationSnapshot::allocated_tiles() const noexcept {
  return impl_ ? impl_->tiles.size() : 0U;
}

PersistentElevationMap::PersistentElevationMap()
    : impl_(std::make_unique<Impl>()) {}

PersistentElevationMap::~PersistentElevationMap() = default;
PersistentElevationMap::PersistentElevationMap(PersistentElevationMap&&) noexcept =
    default;
PersistentElevationMap& PersistentElevationMap::operator=(
    PersistentElevationMap&&) noexcept = default;

ElevationUpdateResult PersistentElevationMap::Apply(
    const ElevationEvidence& evidence) {
  std::lock_guard lock(impl_->mutex);
  const auto reject = [this]() {
    ++impl_->counters.rejected_updates;
    return ElevationUpdateResult{
        .status = ElevationUpdateResult::Status::kRejected,
        .raw_elevation_revision = impl_->state ? impl_->state->revision : 0U,
    };
  };

  if (!ValidGeometry(evidence) ||
      (impl_->state &&
       impl_->state->geometry.resolution_m() !=
           evidence.geometry.resolution_m)) {
    return reject();
  }

  double canonical_resolution_m{};
  Vec3 canonical_origin_m;
  if (impl_->state) {
    if (impl_->state->geometry.frame_id() != evidence.map_from_source.parent_frame) {
      return reject();
    }
    canonical_resolution_m = impl_->state->geometry.resolution_m();
    canonical_origin_m = impl_->state->geometry.origin_m();
  } else {
    const Vec3 canonical_origin =
        TransformPoint(evidence.map_from_source, evidence.geometry.origin_m);
    if (!Finite(canonical_origin)) {
      return reject();
    }
    canonical_resolution_m = evidence.geometry.resolution_m;
    canonical_origin_m = canonical_origin;
  }

  const auto input_bounds = ProjectInputBounds(evidence, canonical_origin_m);
  if (!input_bounds) return reject();
  GridIndex min_inclusive = input_bounds->minimum;
  GridIndex max_exclusive = input_bounds->maximum;
  if (impl_->state) {
    const auto& old = impl_->state->geometry;
    min_inclusive.x = std::min(min_inclusive.x, old.min_inclusive().x);
    min_inclusive.y = std::min(min_inclusive.y, old.min_inclusive().y);
    max_exclusive.x = std::max(max_exclusive.x, old.max_exclusive().x);
    max_exclusive.y = std::max(max_exclusive.y, old.max_exclusive().y);
  }

  std::map<GridIndex, ElevationRange> projected;
  for (std::size_t source_y = 0U; source_y < evidence.geometry.height;
       ++source_y) {
    for (std::size_t source_x = 0U; source_x < evidence.geometry.width;
         ++source_x) {
      const std::size_t offset = source_y * evidence.geometry.width + source_x;
      const float elevation_m = evidence.elevation_m[offset];
      if (!std::isfinite(elevation_m)) {
        continue;
      }
      const Vec3 source_point{
          .x = evidence.geometry.origin_m.x +
               (static_cast<double>(source_x) + 0.5) *
                   evidence.geometry.resolution_m,
          .y = evidence.geometry.origin_m.y +
               (static_cast<double>(source_y) + 0.5) *
                   evidence.geometry.resolution_m,
          .z = static_cast<double>(elevation_m),
      };
      const Vec3 canonical_point =
          TransformPoint(evidence.map_from_source, source_point);
      const auto target_x = CellCoordinate(
          canonical_point.x, canonical_origin_m.x, canonical_resolution_m);
      const auto target_y = CellCoordinate(
          canonical_point.y, canonical_origin_m.y, canonical_resolution_m);
      if (!Finite(canonical_point) || !target_x || !target_y ||
          std::abs(canonical_point.z) >
              static_cast<double>(std::numeric_limits<float>::max())) {
        return reject();
      }
      const float canonical_elevation_m =
          static_cast<float>(canonical_point.z);
      const GridIndex target{.x = *target_x, .y = *target_y};
      min_inclusive.x = std::min(min_inclusive.x, target.x);
      min_inclusive.y = std::min(min_inclusive.y, target.y);
      max_exclusive.x = std::max(max_exclusive.x, target.x + 1);
      max_exclusive.y = std::max(max_exclusive.y, target.y + 1);
      const auto [found, inserted] = projected.try_emplace(
          target, ElevationRange{.min_m = canonical_elevation_m,
                                 .max_m = canonical_elevation_m});
      if (!inserted) {
        found->second.min_m =
            std::min(found->second.min_m, canonical_elevation_m);
        found->second.max_m =
            std::max(found->second.max_m, canonical_elevation_m);
      }
    }
  }

  std::map<TileIndex, std::vector<std::pair<GridIndex, ElevationRange>>> changed;
  for (const auto& [index, elevation_range] : projected) {
    const std::optional<ElevationRange> old_value =
        impl_->state ? impl_->state->ElevationRangeAt(index) : std::nullopt;
    if (!old_value || *old_value != elevation_range) {
      changed[TileForCell(index)].emplace_back(index, elevation_range);
    }
  }

  const SparseGridGeometry next_geometry(
      evidence.map_from_source.parent_frame, canonical_resolution_m,
      canonical_origin_m, min_inclusive, max_exclusive);
  if (!next_geometry.valid()) return reject();
  const bool same_bounds =
      impl_->state && impl_->state->geometry.min_inclusive() == min_inclusive &&
      impl_->state->geometry.max_exclusive() == max_exclusive;
  if (changed.empty() && same_bounds) {
    ++impl_->counters.duplicate_updates;
    return ElevationUpdateResult{
        .status = ElevationUpdateResult::Status::kDuplicate,
        .raw_elevation_revision = impl_->state ? impl_->state->revision : 0U,
    };
  }

  auto next = std::make_shared<ElevationSnapshot::Impl>();
  next->lineage = impl_->lineage;
  if (impl_->state) {
    next->previous = impl_->state;
    next->tiles = impl_->state->tiles;
    next->revision = impl_->state->revision + 1U;
  } else {
    next->revision = 1U;
  }

  std::size_t updated_cells = 0U;
  std::vector<TileIndex> dirty_tiles;
  dirty_tiles.reserve(changed.size());
  for (const auto& [tile_index, values] : changed) {
    std::shared_ptr<ElevationTile> mutable_tile;
    const auto old_tile = next->tiles.find(tile_index);
    if (old_tile == next->tiles.end()) {
      mutable_tile = std::make_shared<ElevationTile>();
      ++impl_->counters.tile_allocations;
    } else {
      mutable_tile = std::make_shared<ElevationTile>(*old_tile->second);
      ++impl_->counters.tile_copies;
    }
    for (const auto& [index, elevation_range] : values) {
      mutable_tile->values[TileCellOffset(index)] = elevation_range;
      next->changed_cells.push_back(index);
      min_inclusive.x = std::min(min_inclusive.x, index.x);
      min_inclusive.y = std::min(min_inclusive.y, index.y);
      max_exclusive.x = std::max(max_exclusive.x, index.x + 1);
      max_exclusive.y = std::max(max_exclusive.y, index.y + 1);
      ++updated_cells;
    }
    next->tiles[tile_index] = std::move(mutable_tile);
    dirty_tiles.push_back(tile_index);
  }
  next->changed_tiles = dirty_tiles;
  next->geometry = next_geometry;

  impl_->state = next;
  impl_->snapshot = std::shared_ptr<const ElevationSnapshot>(
      new ElevationSnapshot(next));
  ++impl_->counters.applied_updates;
  impl_->counters.updated_cells += updated_cells;
  impl_->counters.allocated_tiles = next->tiles.size();
  impl_->counters.estimated_bytes = next->EstimatedBytes();
  return ElevationUpdateResult{
      .status = ElevationUpdateResult::Status::kApplied,
      .raw_elevation_revision = next->revision,
      .updated_cells = updated_cells,
      .dirty_tiles = std::move(dirty_tiles),
  };
}

std::shared_ptr<const ElevationSnapshot> PersistentElevationMap::Snapshot()
    const {
  std::lock_guard lock(impl_->mutex);
  return impl_->snapshot;
}

ElevationMapCounters PersistentElevationMap::Counters() const {
  std::lock_guard lock(impl_->mutex);
  return impl_->counters;
}

}  // namespace lunar::incremental_navigation

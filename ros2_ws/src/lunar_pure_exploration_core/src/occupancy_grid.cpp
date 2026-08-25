#include "lunar_pure_exploration_core/occupancy_grid.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace lunar::pure_exploration {
namespace {

std::size_t CheckedCellCount(const GridGeometry& geometry) {
  constexpr auto kMaxLogicalExtent =
      static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()) + 1U;
  if (geometry.width > kMaxLogicalExtent ||
      geometry.height > kMaxLogicalExtent) {
    throw std::overflow_error("map dimensions exceed GridIndex range");
  }

  const auto width = static_cast<std::size_t>(geometry.width);
  const auto height = static_cast<std::size_t>(geometry.height);
  if (height != 0U && width > std::numeric_limits<std::size_t>::max() / height) {
    throw std::overflow_error("map cell count overflows size_t");
  }
  const std::size_t count = width * height;
  if (count > std::vector<std::int8_t>{}.max_size()) {
    throw std::overflow_error("map cell count exceeds storage capacity");
  }
  return count;
}

bool IsFiniteGeometry(const GridGeometry& geometry) {
  if (!std::isfinite(geometry.resolution) || geometry.resolution <= 0.0 ||
      !std::isfinite(geometry.origin_x) ||
      !std::isfinite(geometry.origin_y) ||
      !std::isfinite(geometry.origin_yaw)) {
    return false;
  }
  return std::isfinite(static_cast<double>(geometry.width) *
                       geometry.resolution) &&
         std::isfinite(static_cast<double>(geometry.height) *
                       geometry.resolution);
}

}  // namespace

OccupancyGridView::OccupancyGridView(
    GridGeometry geometry, std::span<const std::int8_t> data,
    std::int8_t occupied_threshold)
    : geometry_(geometry), occupied_threshold_(occupied_threshold) {
  if (!IsFiniteGeometry(geometry_)) {
    throw std::invalid_argument("occupancy grid geometry must be finite and positive");
  }
  if (occupied_threshold_ < 0 || occupied_threshold_ > 100) {
    throw std::invalid_argument("occupied threshold must be in [0, 100]");
  }

  const std::size_t expected_size = CheckedCellCount(geometry_);
  if (data.size() != expected_size) {
    throw std::invalid_argument("occupancy grid dimensions do not match data size");
  }
  data_.assign(data.begin(), data.end());
}

CellState OccupancyGridView::Classify(GridIndex index) const {
  if (!Contains(index)) {
    return CellState::kOutsideMap;
  }
  const auto offset = static_cast<std::size_t>(index.y) * geometry_.width +
                      static_cast<std::size_t>(index.x);
  const int value = data_[offset];
  if (value < 0 || value > 100) {
    return CellState::kUnknown;
  }
  return value >= occupied_threshold_ ? CellState::kOccupied
                                      : CellState::kFree;
}

std::optional<std::int8_t> OccupancyGridView::RawValue(
    GridIndex index) const {
  if (!Contains(index)) {
    return std::nullopt;
  }
  const auto offset = static_cast<std::size_t>(index.y) * geometry_.width +
                      static_cast<std::size_t>(index.x);
  return data_[offset];
}

bool OccupancyGridView::Contains(GridIndex index) const {
  return index.x >= 0 && index.y >= 0 &&
         static_cast<std::uint32_t>(index.x) < geometry_.width &&
         static_cast<std::uint32_t>(index.y) < geometry_.height;
}

std::optional<Vec2> OccupancyGridView::WorldToGrid(
    const GridGeometry& geometry, Vec2 world_point) {
  if (!std::isfinite(world_point.x) || !std::isfinite(world_point.y) ||
      !IsFiniteGeometry(geometry)) {
    return std::nullopt;
  }
  const double cosine = std::cos(geometry.origin_yaw);
  const double sine = std::sin(geometry.origin_yaw);
  const double dx = world_point.x - geometry.origin_x;
  const double dy = world_point.y - geometry.origin_y;
  const double grid_x = (cosine * dx + sine * dy) / geometry.resolution;
  const double grid_y = (-sine * dx + cosine * dy) / geometry.resolution;
  if (!std::isfinite(grid_x) || !std::isfinite(grid_y)) {
    return std::nullopt;
  }
  return Vec2{grid_x, grid_y};
}

std::optional<Vec2> OccupancyGridView::GridToWorld(
    const GridGeometry& geometry, Vec2 grid_point) {
  if (!std::isfinite(grid_point.x) || !std::isfinite(grid_point.y) ||
      !IsFiniteGeometry(geometry)) {
    return std::nullopt;
  }
  const double local_x = grid_point.x * geometry.resolution;
  const double local_y = grid_point.y * geometry.resolution;
  const double cosine = std::cos(geometry.origin_yaw);
  const double sine = std::sin(geometry.origin_yaw);
  const double world_x = geometry.origin_x + cosine * local_x - sine * local_y;
  const double world_y = geometry.origin_y + sine * local_x + cosine * local_y;
  if (!std::isfinite(world_x) || !std::isfinite(world_y)) {
    return std::nullopt;
  }
  return Vec2{world_x, world_y};
}

std::optional<GridIndex> OccupancyGridView::WorldToCell(
    const GridGeometry& geometry, Vec2 world_point) {
  const auto grid = WorldToGrid(geometry, world_point);
  if (!grid.has_value()) {
    return std::nullopt;
  }

  const double cell_x = std::floor(grid->x);
  const double cell_y = std::floor(grid->y);
  constexpr double kMin =
      static_cast<double>(std::numeric_limits<std::int32_t>::min());
  constexpr double kMax =
      static_cast<double>(std::numeric_limits<std::int32_t>::max());
  if (cell_x < kMin || cell_x > kMax || cell_y < kMin || cell_y > kMax) {
    return std::nullopt;
  }
  return GridIndex{static_cast<std::int32_t>(cell_x),
                   static_cast<std::int32_t>(cell_y)};
}

std::array<Vec2, 4> OccupancyGridView::CellCornersInGrid(
    GridIndex logical_index) {
  const double x = static_cast<double>(logical_index.x);
  const double y = static_cast<double>(logical_index.y);
  return {Vec2{x, y}, Vec2{x + 1.0, y}, Vec2{x + 1.0, y + 1.0},
          Vec2{x, y + 1.0}};
}

std::optional<Vec2> OccupancyGridView::WorldToGrid(Vec2 point) const {
  return OccupancyGridView::WorldToGrid(geometry_, point);
}

std::optional<Vec2> OccupancyGridView::GridToWorld(Vec2 point) const {
  return OccupancyGridView::GridToWorld(geometry_, point);
}

std::optional<GridIndex> OccupancyGridView::WorldToCell(Vec2 point) const {
  return OccupancyGridView::WorldToCell(geometry_, point);
}

Vec2 OccupancyGridView::CellCenter(GridIndex index) const {
  const Vec2 grid_point{static_cast<double>(index.x) + 0.5,
                        static_cast<double>(index.y) + 0.5};
  const auto world = OccupancyGridView::GridToWorld(geometry_, grid_point);
  if (!world.has_value()) {
    throw std::overflow_error("cell center exceeds finite world coordinates");
  }
  return *world;
}

const GridGeometry& OccupancyGridView::geometry() const { return geometry_; }

}  // namespace lunar::pure_exploration

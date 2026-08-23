#include "lunar_pure_exploration_core/task_raster.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

namespace lunar::pure_exploration {
namespace {

using Wide = long double;

bool SamePoint(const Vec2& left, const Vec2& right) {
  return left.x == right.x && left.y == right.y;
}

Wide Cross(const Vec2& a, const Vec2& b, const Vec2& c) {
  return (static_cast<Wide>(b.x) - a.x) *
             (static_cast<Wide>(c.y) - a.y) -
         (static_cast<Wide>(b.y) - a.y) *
             (static_cast<Wide>(c.x) - a.x);
}

Wide CrossTolerance(const Vec2& a, const Vec2& b, const Vec2& c) {
  const Wide first = std::abs((static_cast<Wide>(b.x) - a.x) *
                              (static_cast<Wide>(c.y) - a.y));
  const Wide second = std::abs((static_cast<Wide>(b.y) - a.y) *
                               (static_cast<Wide>(c.x) - a.x));
  return 64.0L * std::numeric_limits<double>::epsilon() *
         std::max<Wide>(1.0L, first + second);
}

bool IsCollinear(const Vec2& a, const Vec2& b, const Vec2& c) {
  return std::abs(Cross(a, b, c)) <= CrossTolerance(a, b, c);
}

bool PointOnSegment(const Vec2& point, const Vec2& a, const Vec2& b) {
  if (!IsCollinear(a, b, point)) {
    return false;
  }
  const Wide edge_x = static_cast<Wide>(b.x) - a.x;
  const Wide edge_y = static_cast<Wide>(b.y) - a.y;
  const Wide point_x = static_cast<Wide>(point.x) - a.x;
  const Wide point_y = static_cast<Wide>(point.y) - a.y;
  const Wide edge_length_squared = edge_x * edge_x + edge_y * edge_y;
  const Wide projection = point_x * edge_x + point_y * edge_y;
  const Wide tolerance =
      64.0L * std::numeric_limits<double>::epsilon() *
      std::max<Wide>(1.0L, std::abs(point_x * edge_x) +
                               std::abs(point_y * edge_y) +
                               edge_length_squared);
  return projection >= -tolerance &&
         projection <= edge_length_squared + tolerance;
}

int Orientation(const Vec2& a, const Vec2& b, const Vec2& c) {
  const Wide cross = Cross(a, b, c);
  const Wide tolerance = CrossTolerance(a, b, c);
  if (cross > tolerance) {
    return 1;
  }
  if (cross < -tolerance) {
    return -1;
  }
  return 0;
}

bool SegmentsIntersect(const Vec2& a, const Vec2& b, const Vec2& c,
                       const Vec2& d) {
  const int abc = Orientation(a, b, c);
  const int abd = Orientation(a, b, d);
  const int cda = Orientation(c, d, a);
  const int cdb = Orientation(c, d, b);
  if (abc != abd && cda != cdb) {
    return true;
  }
  return (abc == 0 && PointOnSegment(c, a, b)) ||
         (abd == 0 && PointOnSegment(d, a, b)) ||
         (cda == 0 && PointOnSegment(a, c, d)) ||
         (cdb == 0 && PointOnSegment(b, c, d));
}

double ValidateAndArea(const Polygon2& polygon) {
  const auto& vertices = polygon.vertices;
  if (vertices.size() < 3U) {
    throw std::invalid_argument("task polygon needs at least three vertices");
  }
  for (const Vec2& vertex : vertices) {
    if (!std::isfinite(vertex.x) || !std::isfinite(vertex.y)) {
      throw std::invalid_argument("task polygon coordinates must be finite");
    }
  }
  for (std::size_t i = 0U; i < vertices.size(); ++i) {
    for (std::size_t j = i + 1U; j < vertices.size(); ++j) {
      if (SamePoint(vertices[i], vertices[j])) {
        throw std::invalid_argument("task polygon vertices must be distinct");
      }
    }
  }

  const std::size_t count = vertices.size();
  for (std::size_t i = 0U; i < count; ++i) {
    const Vec2& previous = vertices[(i + count - 1U) % count];
    const Vec2& current = vertices[i];
    const Vec2& next = vertices[(i + 1U) % count];
    if (IsCollinear(previous, current, next)) {
      const Wide incoming_x = static_cast<Wide>(current.x) - previous.x;
      const Wide incoming_y = static_cast<Wide>(current.y) - previous.y;
      const Wide outgoing_x = static_cast<Wide>(next.x) - current.x;
      const Wide outgoing_y = static_cast<Wide>(next.y) - current.y;
      if (incoming_x * outgoing_x + incoming_y * outgoing_y < 0.0L) {
        throw std::invalid_argument("task polygon has collinear overlap");
      }
    }
  }

  for (std::size_t i = 0U; i < count; ++i) {
    const std::size_t i_next = (i + 1U) % count;
    for (std::size_t j = i + 1U; j < count; ++j) {
      const std::size_t j_next = (j + 1U) % count;
      const bool adjacent = i == j || i_next == j || j_next == i;
      if (!adjacent && SegmentsIntersect(vertices[i], vertices[i_next],
                                         vertices[j], vertices[j_next])) {
        throw std::invalid_argument("task polygon self-intersects");
      }
    }
  }

  Wide twice_signed_area = 0.0L;
  const Vec2& anchor = vertices.front();
  for (std::size_t i = 1U; i + 1U < count; ++i) {
    twice_signed_area += Cross(anchor, vertices[i], vertices[i + 1U]);
  }
  const Wide area = std::abs(twice_signed_area) / 2.0L;
  if (area == 0.0L) {
    throw std::invalid_argument("task polygon area must be nonzero");
  }
  if (!std::isfinite(area) ||
      area > std::numeric_limits<double>::max()) {
    throw std::overflow_error("task polygon area exceeds double range");
  }
  return static_cast<double>(area);
}

bool ContainsPointInclusive(const Polygon2& polygon, Vec2 point) {
  const auto& vertices = polygon.vertices;
  for (std::size_t i = 0U; i < vertices.size(); ++i) {
    if (PointOnSegment(point, vertices[i],
                       vertices[(i + 1U) % vertices.size()])) {
      return true;
    }
  }

  bool inside = false;
  for (std::size_t i = 0U, j = vertices.size() - 1U; i < vertices.size();
       j = i++) {
    const Vec2& a = vertices[j];
    const Vec2& b = vertices[i];
    if ((a.y > point.y) != (b.y > point.y)) {
      const Wide intersection_x =
          static_cast<Wide>(a.x) +
          (static_cast<Wide>(point.y) - a.y) *
              (static_cast<Wide>(b.x) - a.x) /
              (static_cast<Wide>(b.y) - a.y);
      if (static_cast<Wide>(point.x) < intersection_x) {
        inside = !inside;
      }
    }
  }
  return inside;
}

std::pair<Wide, Wide> WorldToContinuousGrid(const GridGeometry& geometry,
                                             Vec2 point) {
  const auto grid = OccupancyGridView::WorldToGrid(geometry, point);
  if (!grid.has_value()) {
    throw std::overflow_error("task polygon grid coordinates overflow");
  }
  return {static_cast<Wide>(grid->x), static_cast<Wide>(grid->y)};
}

std::int32_t CheckedGridBound(Wide value) {
  if (!std::isfinite(value) ||
      value < std::numeric_limits<std::int32_t>::min() ||
      value > std::numeric_limits<std::int32_t>::max()) {
    throw std::overflow_error("task raster bound exceeds GridIndex range");
  }
  return static_cast<std::int32_t>(value);
}

std::size_t CheckedExtent(std::int32_t lower, std::int32_t upper) {
  const auto extent = static_cast<std::uint64_t>(
      static_cast<std::int64_t>(upper) - lower + 1);
  if (extent > std::numeric_limits<std::size_t>::max()) {
    throw std::overflow_error("task raster extent overflows size_t");
  }
  return static_cast<std::size_t>(extent);
}

}  // namespace

TaskRaster TaskRaster::Build(const OccupancyGridView& map,
                             const Polygon2& polygon,
                             TaskRaster::Limits limits) {
  if (limits.maximum_raster_cell_count == 0U) {
    throw std::invalid_argument("task raster cell limit must be positive");
  }
  TaskRaster raster;
  raster.geometry_ = map.geometry();
  raster.polygon_area_m2_ = ValidateAndArea(polygon);

  Wide min_x = std::numeric_limits<Wide>::infinity();
  Wide min_y = std::numeric_limits<Wide>::infinity();
  Wide max_x = -std::numeric_limits<Wide>::infinity();
  Wide max_y = -std::numeric_limits<Wide>::infinity();
  for (const Vec2& vertex : polygon.vertices) {
    const auto [grid_x, grid_y] =
        WorldToContinuousGrid(raster.geometry_, vertex);
    min_x = std::min(min_x, grid_x);
    min_y = std::min(min_y, grid_y);
    max_x = std::max(max_x, grid_x);
    max_y = std::max(max_y, grid_y);
  }

  const Wide lower_x = std::ceil(min_x - 0.5L);
  const Wide lower_y = std::ceil(min_y - 0.5L);
  const Wide upper_x = std::floor(max_x - 0.5L);
  const Wide upper_y = std::floor(max_y - 0.5L);
  if (lower_x > upper_x || lower_y > upper_y) {
    return raster;
  }

  raster.raster_min_ =
      GridIndex{CheckedGridBound(lower_x), CheckedGridBound(lower_y)};
  const std::int32_t max_index_x = CheckedGridBound(upper_x);
  const std::int32_t max_index_y = CheckedGridBound(upper_y);
  raster.raster_width_ = CheckedExtent(raster.raster_min_.x, max_index_x);
  raster.raster_height_ = CheckedExtent(raster.raster_min_.y, max_index_y);
  if (raster.raster_height_ != 0U &&
      raster.raster_width_ >
          std::numeric_limits<std::size_t>::max() / raster.raster_height_) {
    throw std::overflow_error("task raster cell count overflows size_t");
  }
  const std::size_t cell_count =
      raster.raster_width_ * raster.raster_height_;
  if (cell_count > raster.raster_states_.max_size() ||
      cell_count > raster.task_cells_.max_size() ||
      cell_count > raster.map_backed_cells_.max_size()) {
    throw std::overflow_error("task raster exceeds storage capacity");
  }
  if (cell_count > limits.maximum_raster_cell_count) {
    throw std::length_error("task raster exceeds configured cell limit");
  }
  raster.raster_states_.assign(cell_count, CellState::kOutsideTask);

  for (std::int64_t y = raster.raster_min_.y; y <= max_index_y; ++y) {
    for (std::int64_t x = raster.raster_min_.x; x <= max_index_x; ++x) {
      const GridIndex index{static_cast<std::int32_t>(x),
                            static_cast<std::int32_t>(y)};
      if (!ContainsPointInclusive(polygon, map.CellCenter(index))) {
        continue;
      }
      const std::size_t offset =
          static_cast<std::size_t>(y - raster.raster_min_.y) *
              raster.raster_width_ +
          static_cast<std::size_t>(x - raster.raster_min_.x);
      raster.task_cells_.push_back(index);
      if (!map.Contains(index)) {
        raster.raster_states_[offset] = CellState::kOutsideMap;
        continue;
      }
      raster.raster_states_[offset] = map.Classify(index);
      raster.map_backed_cells_.push_back(index);
    }
  }
  return raster;
}

CellState TaskRaster::Classify(GridIndex logical_index) const {
  if (raster_width_ == 0U || raster_height_ == 0U) {
    return CellState::kOutsideTask;
  }
  const std::int64_t dx =
      static_cast<std::int64_t>(logical_index.x) - raster_min_.x;
  const std::int64_t dy =
      static_cast<std::int64_t>(logical_index.y) - raster_min_.y;
  if (dx < 0 || dy < 0 || static_cast<std::uint64_t>(dx) >= raster_width_ ||
      static_cast<std::uint64_t>(dy) >= raster_height_) {
    return CellState::kOutsideTask;
  }
  const std::size_t offset =
      static_cast<std::size_t>(dy) * raster_width_ +
      static_cast<std::size_t>(dx);
  return raster_states_[offset];
}

bool TaskRaster::ContainsCellCenter(GridIndex logical_index) const {
  return Classify(logical_index) != CellState::kOutsideTask;
}

bool TaskRaster::IsMapBacked(GridIndex logical_index) const {
  const CellState state = Classify(logical_index);
  return state != CellState::kOutsideTask && state != CellState::kOutsideMap;
}

Vec2 TaskRaster::CellCenter(GridIndex logical_index) const {
  const Vec2 grid_point{static_cast<double>(logical_index.x) + 0.5,
                        static_cast<double>(logical_index.y) + 0.5};
  const auto world = OccupancyGridView::GridToWorld(geometry_, grid_point);
  if (!world.has_value()) {
    throw std::overflow_error("cell center exceeds finite world coordinates");
  }
  return *world;
}

std::optional<Vec2> TaskRaster::WorldToGrid(Vec2 world_point) const {
  return OccupancyGridView::WorldToGrid(geometry_, world_point);
}

std::optional<Vec2> TaskRaster::GridToWorld(Vec2 grid_point) const {
  return OccupancyGridView::GridToWorld(geometry_, grid_point);
}

std::optional<GridIndex> TaskRaster::WorldToCell(Vec2 world_point) const {
  return OccupancyGridView::WorldToCell(geometry_, world_point);
}

std::array<Vec2, 4> TaskRaster::CellCornersInGrid(
    GridIndex logical_index) const {
  return OccupancyGridView::CellCornersInGrid(logical_index);
}

const GridGeometry& TaskRaster::geometry() const { return geometry_; }

double TaskRaster::polygon_area_m2() const { return polygon_area_m2_; }

std::span<const GridIndex> TaskRaster::task_cells() const {
  return task_cells_;
}

std::span<const GridIndex> TaskRaster::map_backed_cells() const {
  return map_backed_cells_;
}

}  // namespace lunar::pure_exploration

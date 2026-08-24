#include "lunar_pure_exploration_core/safe_pose_validator.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace lunar::pure_exploration {
namespace {

using Wide = long double;

bool Finite(Vec2 point) {
  return std::isfinite(point.x) && std::isfinite(point.y);
}

Wide Cross(Vec2 a, Vec2 b, Vec2 c) {
  return (static_cast<Wide>(b.x) - a.x) *
             (static_cast<Wide>(c.y) - a.y) -
         (static_cast<Wide>(b.y) - a.y) *
             (static_cast<Wide>(c.x) - a.x);
}

Wide Dot(Vec2 a, Vec2 b) {
  return static_cast<Wide>(a.x) * b.x + static_cast<Wide>(a.y) * b.y;
}

Vec2 Subtract(Vec2 a, Vec2 b) { return Vec2{a.x - b.x, a.y - b.y}; }

bool SamePoint(Vec2 a, Vec2 b) { return a.x == b.x && a.y == b.y; }

bool OnSegment(Vec2 point, Vec2 a, Vec2 b) {
  if (Cross(a, b, point) != 0.0L) {
    return false;
  }
  const Vec2 edge = Subtract(b, a);
  const Vec2 relative = Subtract(point, a);
  const Wide projection = Dot(edge, relative);
  return projection >= 0.0L && projection <= Dot(edge, edge);
}

int Orientation(Vec2 a, Vec2 b, Vec2 c) {
  const Wide cross = Cross(a, b, c);
  return cross > 0.0L ? 1 : cross < 0.0L ? -1 : 0;
}

bool SegmentsIntersect(Vec2 a, Vec2 b, Vec2 c, Vec2 d) {
  const int abc = Orientation(a, b, c);
  const int abd = Orientation(a, b, d);
  const int cda = Orientation(c, d, a);
  const int cdb = Orientation(c, d, b);
  if (abc != abd && cda != cdb) {
    return true;
  }
  return (abc == 0 && OnSegment(c, a, b)) ||
         (abd == 0 && OnSegment(d, a, b)) ||
         (cda == 0 && OnSegment(a, c, d)) ||
         (cdb == 0 && OnSegment(a, c, d));
}

std::size_t ValidateAndNormalizeFootprint(PlatformGeometry& platform,
                                          std::size_t maximum_work_units) {
  if (platform.platform_type != "WHEELED" || platform.platform_id.empty() ||
      platform.base_frame_id.empty()) {
    throw std::invalid_argument("candidate platform must be identified WHEELED payload");
  }
  if (!std::isfinite(platform.minimum_clearance_m) ||
      platform.minimum_clearance_m < 0.0) {
    throw std::invalid_argument("candidate clearance must be finite and nonnegative");
  }
  auto& vertices = platform.footprint_vertices;
  if (vertices.size() < 3U) {
    throw std::invalid_argument("candidate footprint needs at least three vertices");
  }
  const std::size_t vertex_count = vertices.size();
  if (vertex_count - 1U >
      std::numeric_limits<std::size_t>::max() / vertex_count) {
    throw std::length_error("candidate footprint validation work overflows size_t");
  }
  const std::size_t validation_work = vertex_count * (vertex_count - 1U);
  if (validation_work > maximum_work_units) {
    throw std::length_error("candidate footprint validation work limit exceeded");
  }
  for (const Vec2 vertex : vertices) {
    if (!Finite(vertex)) {
      throw std::invalid_argument("candidate footprint must be finite");
    }
  }
  for (std::size_t i = 0U; i < vertices.size(); ++i) {
    for (std::size_t j = i + 1U; j < vertices.size(); ++j) {
      if (SamePoint(vertices[i], vertices[j])) {
        throw std::invalid_argument("candidate footprint vertices must be distinct");
      }
    }
  }
  const std::size_t count = vertices.size();
  for (std::size_t i = 0U; i < count; ++i) {
    const std::size_t next_i = (i + 1U) % count;
    for (std::size_t j = i + 1U; j < count; ++j) {
      const std::size_t next_j = (j + 1U) % count;
      if (next_i == j || next_j == i) {
        continue;
      }
      if (SegmentsIntersect(vertices[i], vertices[next_i], vertices[j],
                            vertices[next_j])) {
        throw std::invalid_argument("candidate footprint self-intersects");
      }
    }
  }

  int turn = 0;
  Wide twice_area = 0.0L;
  for (std::size_t i = 0U; i < count; ++i) {
    const Wide cross = Cross(vertices[i], vertices[(i + 1U) % count],
                             vertices[(i + 2U) % count]);
    if (cross == 0.0L) {
      throw std::invalid_argument("candidate footprint is degenerate");
    }
    const int current = cross > 0.0L ? 1 : -1;
    if (turn != 0 && current != turn) {
      throw std::invalid_argument("candidate footprint must be convex");
    }
    turn = current;
    const Vec2& a = vertices[i];
    const Vec2& b = vertices[(i + 1U) % count];
    twice_area += static_cast<Wide>(a.x) * b.y -
                  static_cast<Wide>(a.y) * b.x;
  }
  if (twice_area == 0.0L || !std::isfinite(twice_area)) {
    throw std::invalid_argument("candidate footprint area must be finite and nonzero");
  }
  if (twice_area < 0.0L) {
    std::reverse(vertices.begin(), vertices.end());
  }
  return validation_work;
}

double NormalizeYaw(double yaw) {
  if (!std::isfinite(yaw)) {
    throw std::invalid_argument("candidate yaw must be finite");
  }
  if (yaw >= -std::numbers::pi && yaw < std::numbers::pi) {
    return yaw == 0.0 ? 0.0 : yaw;
  }
  if (yaw == std::numbers::pi) {
    return -std::numbers::pi;
  }
  const double two_pi = 2.0 * std::numbers::pi;
  double normalized = std::fmod(yaw + std::numbers::pi, two_pi);
  if (normalized < 0.0) {
    normalized += two_pi;
  }
  normalized -= std::numbers::pi;
  if (normalized == 0.0) {
    normalized = 0.0;
  }
  return normalized;
}

Wide SquaredDistance(Vec2 a, Vec2 b) {
  const Wide dx = static_cast<Wide>(a.x) - b.x;
  const Wide dy = static_cast<Wide>(a.y) - b.y;
  return dx * dx + dy * dy;
}

Wide PointSegmentSquaredDistance(Vec2 point, Vec2 a, Vec2 b) {
  const Vec2 edge = Subtract(b, a);
  const Vec2 relative = Subtract(point, a);
  const Wide denominator = Dot(edge, edge);
  Wide t = denominator == 0.0L ? 0.0L : Dot(relative, edge) / denominator;
  t = std::clamp(t, 0.0L, 1.0L);
  const Vec2 closest{static_cast<double>(static_cast<Wide>(a.x) + t * edge.x),
                     static_cast<double>(static_cast<Wide>(a.y) + t * edge.y)};
  return SquaredDistance(point, closest);
}

bool PointInConvexClosed(std::span<const Vec2> polygon, Vec2 point) {
  int side = 0;
  for (std::size_t i = 0U; i < polygon.size(); ++i) {
    const Wide cross = Cross(polygon[i], polygon[(i + 1U) % polygon.size()], point);
    if (cross == 0.0L) {
      continue;
    }
    const int current = cross > 0.0L ? 1 : -1;
    if (side != 0 && side != current) {
      return false;
    }
    side = current;
  }
  return true;
}

Wide PolygonCellSquaredDistance(std::span<const Vec2> polygon,
                                GridIndex cell) {
  const auto corners = OccupancyGridView::CellCornersInGrid(cell);
  for (const Vec2 vertex : polygon) {
    if (vertex.x >= corners[0].x && vertex.x <= corners[2].x &&
        vertex.y >= corners[0].y && vertex.y <= corners[2].y) {
      return 0.0L;
    }
  }
  for (const Vec2 corner : corners) {
    if (PointInConvexClosed(polygon, corner)) {
      return 0.0L;
    }
  }
  Wide minimum = std::numeric_limits<Wide>::infinity();
  for (std::size_t i = 0U; i < polygon.size(); ++i) {
    const Vec2 a = polygon[i];
    const Vec2 b = polygon[(i + 1U) % polygon.size()];
    for (std::size_t j = 0U; j < corners.size(); ++j) {
      const Vec2 c = corners[j];
      const Vec2 d = corners[(j + 1U) % corners.size()];
      if (SegmentsIntersect(a, b, c, d)) {
        return 0.0L;
      }
      minimum = std::min(minimum, PointSegmentSquaredDistance(a, c, d));
      minimum = std::min(minimum, PointSegmentSquaredDistance(b, c, d));
      minimum = std::min(minimum, PointSegmentSquaredDistance(c, a, b));
      minimum = std::min(minimum, PointSegmentSquaredDistance(d, a, b));
    }
  }
  return minimum;
}

std::int32_t CheckedLowerCell(double value) {
  const double bound = std::ceil(value - 1.0);
  if (!std::isfinite(bound) ||
      bound < static_cast<double>(std::numeric_limits<std::int32_t>::min()) ||
      bound > static_cast<double>(std::numeric_limits<std::int32_t>::max())) {
    throw std::overflow_error("candidate collision lower bound exceeds GridIndex");
  }
  return static_cast<std::int32_t>(bound);
}

std::int32_t CheckedUpperCell(double value) {
  const double bound = std::floor(value);
  if (!std::isfinite(bound) ||
      bound < static_cast<double>(std::numeric_limits<std::int32_t>::min()) ||
      bound > static_cast<double>(std::numeric_limits<std::int32_t>::max())) {
    throw std::overflow_error("candidate collision upper bound exceeds GridIndex");
  }
  return static_cast<std::int32_t>(bound);
}

template <typename WorldToGrid, typename IsFree>
bool CollisionFree(const PlatformGeometry& platform, std::size_t maximum_work,
                   Pose2 pose, const GridGeometry& geometry,
                   WorldToGrid&& world_to_grid, IsFree&& is_free,
                   std::size_t& consumed_work) {
  if (consumed_work > maximum_work ||
      platform.footprint_vertices.size() > maximum_work - consumed_work) {
    throw std::length_error("candidate footprint transform work limit exceeded");
  }
  const auto center_grid = world_to_grid(Vec2{pose.x, pose.y});
  if (!center_grid.has_value()) {
    throw std::overflow_error("candidate center transform is not representable");
  }
  std::vector<Vec2> grid_polygon;
  grid_polygon.reserve(platform.footprint_vertices.size());
  const double relative_yaw = NormalizeYaw(pose.yaw - geometry.origin_yaw);
  const double cosine = std::cos(relative_yaw);
  const double sine = std::sin(relative_yaw);
  const double inverse_resolution = 1.0 / geometry.resolution;
  for (const Vec2 vertex : platform.footprint_vertices) {
    ++consumed_work;
    const Vec2 grid{
        center_grid->x +
            (cosine * vertex.x - sine * vertex.y) * inverse_resolution,
        center_grid->y +
            (sine * vertex.x + cosine * vertex.y) * inverse_resolution};
    if (!Finite(grid)) {
      throw std::overflow_error("candidate footprint transform is not representable");
    }
    grid_polygon.push_back(grid);
  }
  const double clearance = platform.minimum_clearance_m / geometry.resolution;
  double minimum_x = std::numeric_limits<double>::infinity();
  double minimum_y = std::numeric_limits<double>::infinity();
  double maximum_x = -std::numeric_limits<double>::infinity();
  double maximum_y = -std::numeric_limits<double>::infinity();
  for (const Vec2 point : grid_polygon) {
    minimum_x = std::min(minimum_x, point.x);
    minimum_y = std::min(minimum_y, point.y);
    maximum_x = std::max(maximum_x, point.x);
    maximum_y = std::max(maximum_y, point.y);
  }
  const std::int32_t lower_x = CheckedLowerCell(minimum_x - clearance);
  const std::int32_t lower_y = CheckedLowerCell(minimum_y - clearance);
  const std::int32_t upper_x = CheckedUpperCell(maximum_x + clearance);
  const std::int32_t upper_y = CheckedUpperCell(maximum_y + clearance);
  const Wide clearance_squared = static_cast<Wide>(clearance) * clearance;
  for (std::int64_t y = lower_y; y <= upper_y; ++y) {
    for (std::int64_t x = lower_x; x <= upper_x; ++x) {
      if (consumed_work == maximum_work) {
        throw std::length_error("candidate collision work limit exceeded");
      }
      ++consumed_work;
      const GridIndex cell{static_cast<std::int32_t>(x),
                           static_cast<std::int32_t>(y)};
      const Wide distance_squared = PolygonCellSquaredDistance(grid_polygon, cell);
      const Wide comparison_scale =
          std::max({1.0L, std::abs(distance_squared), std::abs(clearance_squared)});
      const Wide closed_tolerance =
          64.0L * std::numeric_limits<double>::epsilon() * comparison_scale;
      if (distance_squared <= clearance_squared + closed_tolerance &&
          !is_free(cell)) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace

SafePoseValidator::SafePoseValidator(PlatformGeometry platform,
                                     std::size_t maximum_collision_work_units)
    : platform_(std::move(platform)),
      maximum_collision_work_units_(maximum_collision_work_units),
      platform_length_m_(0.0),
      platform_width_m_(0.0),
      footprint_circumscribed_radius_m_(0.0) {
  (void)ValidateAndNormalizeFootprint(platform_, maximum_collision_work_units_);
  double minimum_x = std::numeric_limits<double>::infinity();
  double minimum_y = std::numeric_limits<double>::infinity();
  double maximum_x = -std::numeric_limits<double>::infinity();
  double maximum_y = -std::numeric_limits<double>::infinity();
  for (const Vec2 vertex : platform_.footprint_vertices) {
    minimum_x = std::min(minimum_x, vertex.x);
    minimum_y = std::min(minimum_y, vertex.y);
    maximum_x = std::max(maximum_x, vertex.x);
    maximum_y = std::max(maximum_y, vertex.y);
    footprint_circumscribed_radius_m_ =
        std::max(footprint_circumscribed_radius_m_, std::hypot(vertex.x, vertex.y));
  }
  platform_length_m_ = maximum_x - minimum_x;
  platform_width_m_ = maximum_y - minimum_y;
  if (!std::isfinite(platform_length_m_) || !std::isfinite(platform_width_m_) ||
      !std::isfinite(footprint_circumscribed_radius_m_) ||
      !std::isfinite(footprint_circumscribed_radius_m_ +
                     platform_.minimum_clearance_m) ||
      !std::isfinite(2.0 * platform_length_m_)) {
    throw std::overflow_error("candidate derived platform geometry exceeds double range");
  }
  if (!(platform_length_m_ > 0.0) || !(platform_width_m_ > 0.0) ||
      !(footprint_circumscribed_radius_m_ > 0.0)) {
    throw std::invalid_argument("candidate footprint dimensions must be positive");
  }
}

bool SafePoseValidator::IsMapFree(const OccupancyGridView& map, Pose2 pose,
                                  std::size_t& consumed_work) const {
  return CollisionFree(platform_, maximum_collision_work_units_, pose, map.geometry(),
                       [&map](Vec2 point) { return map.WorldToGrid(point); },
                       [&map](GridIndex cell) {
                         return map.Classify(cell) == CellState::kFree;
                       },
                       consumed_work);
}

bool SafePoseValidator::IsTaskFree(const TaskRaster& raster, Pose2 pose,
                                   std::size_t& consumed_work) const {
  return CollisionFree(platform_, maximum_collision_work_units_, pose,
                       raster.geometry(),
                       [&raster](Vec2 point) { return raster.WorldToGrid(point); },
                       [&raster](GridIndex cell) {
                         return raster.IsMapBacked(cell) &&
                                raster.Classify(cell) == CellState::kFree;
                       },
                       consumed_work);
}

double SafePoseValidator::platform_length_m() const { return platform_length_m_; }

double SafePoseValidator::platform_width_m() const { return platform_width_m_; }

double SafePoseValidator::footprint_circumscribed_radius_m() const {
  return footprint_circumscribed_radius_m_;
}

double SafePoseValidator::minimum_standoff_m() const {
  return footprint_circumscribed_radius_m_ + platform_.minimum_clearance_m;
}

}  // namespace lunar::pure_exploration

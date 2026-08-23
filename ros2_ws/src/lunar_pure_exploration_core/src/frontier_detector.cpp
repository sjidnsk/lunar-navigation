#include "lunar_pure_exploration_core/frontier_detector.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <set>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace lunar::pure_exploration {
namespace {

struct DirectionStep {
  std::int32_t dx;
  std::int32_t dy;
  std::uint8_t direction;
};

constexpr std::array<DirectionStep, 4> kCardinalSteps{{
    {1, 0, 0},
    {0, 1, 1},
    {-1, 0, 2},
    {0, -1, 3},
}};

constexpr std::array<std::pair<std::int32_t, std::int32_t>, 8>
    kClusterSteps{{
        {1, 0}, {1, 1}, {0, 1}, {-1, 1},
        {-1, 0}, {-1, -1}, {0, -1}, {1, -1},
    }};

constexpr std::uint64_t kFnvOffsetBasis = UINT64_C(14695981039346656037);
constexpr std::uint64_t kFnvPrime = UINT64_C(1099511628211);

GridIndex CheckedOffset(GridIndex index, std::int32_t dx, std::int32_t dy) {
  const std::int64_t x = static_cast<std::int64_t>(index.x) + dx;
  const std::int64_t y = static_cast<std::int64_t>(index.y) + dy;
  if (x < std::numeric_limits<std::int32_t>::min() ||
      x > std::numeric_limits<std::int32_t>::max() ||
      y < std::numeric_limits<std::int32_t>::min() ||
      y > std::numeric_limits<std::int32_t>::max()) {
    throw std::overflow_error("frontier neighbor exceeds GridIndex range");
  }
  return GridIndex{static_cast<std::int32_t>(x),
                   static_cast<std::int32_t>(y)};
}

Vec2 CheckedCellCenter(const TaskRaster& raster, GridIndex index) {
  const Vec2 center = raster.CellCenter(index);
  if (!std::isfinite(center.x) || !std::isfinite(center.y)) {
    throw std::overflow_error("frontier world coordinate is not finite");
  }
  return center;
}

bool WorldIndexLess(const TaskRaster& raster, GridIndex left,
                    GridIndex right) {
  const Vec2 left_center = CheckedCellCenter(raster, left);
  const Vec2 right_center = CheckedCellCenter(raster, right);
  if (left_center.x != right_center.x) {
    return left_center.x < right_center.x;
  }
  if (left_center.y != right_center.y) {
    return left_center.y < right_center.y;
  }
  return left < right;
}

Vec2 CheckedMidpoint(Vec2 first, Vec2 second) {
  const long double midpoint_x =
      (static_cast<long double>(first.x) + second.x) / 2.0L;
  const long double midpoint_y =
      (static_cast<long double>(first.y) + second.y) / 2.0L;
  if (!std::isfinite(midpoint_x) || !std::isfinite(midpoint_y) ||
      midpoint_x < -std::numeric_limits<double>::max() ||
      midpoint_x > std::numeric_limits<double>::max() ||
      midpoint_y < -std::numeric_limits<double>::max() ||
      midpoint_y > std::numeric_limits<double>::max()) {
    throw std::overflow_error("frontier midpoint exceeds double range");
  }
  return Vec2{static_cast<double>(midpoint_x),
              static_cast<double>(midpoint_y)};
}

std::int64_t QuantizeMillimeters(double value) {
  if (!std::isfinite(value)) {
    throw std::overflow_error("frontier midpoint is not finite");
  }
  const long double scaled = static_cast<long double>(value) * 1000.0L;
  constexpr long double kMinimum =
      static_cast<long double>(std::numeric_limits<std::int64_t>::min());
  constexpr long double kMaximum =
      static_cast<long double>(std::numeric_limits<std::int64_t>::max());
  if (!std::isfinite(scaled) || scaled <= kMinimum - 0.5L ||
      scaled >= kMaximum + 0.5L) {
    throw std::overflow_error("frontier millimeter quantization overflows int64");
  }
  return static_cast<std::int64_t>(std::llround(scaled));
}

void HashByte(std::uint64_t& hash, std::uint8_t byte) {
  hash ^= byte;
  hash *= kFnvPrime;
}

void HashInt64LittleEndian(std::uint64_t& hash, std::int64_t value) {
  const std::uint64_t bits = static_cast<std::uint64_t>(value);
  for (unsigned int shift = 0U; shift < 64U; shift += 8U) {
    HashByte(hash, static_cast<std::uint8_t>((bits >> shift) & 0xffU));
  }
}

std::uint64_t BuildIdentity(
    const std::vector<FrontierCluster::InterfaceEdge>& edges,
    std::vector<std::int64_t>& canonical_key) {
  if (edges.size() > canonical_key.max_size() / 3U) {
    throw std::overflow_error("frontier canonical key size overflows");
  }
  canonical_key.clear();
  canonical_key.reserve(edges.size() * 3U);
  std::uint64_t hash = kFnvOffsetBasis;
  for (const FrontierCluster::InterfaceEdge& edge : edges) {
    const std::int64_t x_mm = QuantizeMillimeters(edge.midpoint.x);
    const std::int64_t y_mm = QuantizeMillimeters(edge.midpoint.y);
    canonical_key.push_back(x_mm);
    canonical_key.push_back(y_mm);
    canonical_key.push_back(edge.direction);
    HashInt64LittleEndian(hash, x_mm);
    HashInt64LittleEndian(hash, y_mm);
    HashByte(hash, edge.direction);
  }
  return hash;
}

bool InterfaceEdgeLess(const TaskRaster& raster,
                       const FrontierCluster::InterfaceEdge& left,
                       const FrontierCluster::InterfaceEdge& right) {
  const Vec2 left_free = CheckedCellCenter(raster, left.free_cell);
  const Vec2 right_free = CheckedCellCenter(raster, right.free_cell);
  if (left_free.x != right_free.x) {
    return left_free.x < right_free.x;
  }
  if (left_free.y != right_free.y) {
    return left_free.y < right_free.y;
  }
  if (left.direction != right.direction) {
    return left.direction < right.direction;
  }
  const Vec2 left_unknown = CheckedCellCenter(raster, left.unknown_cell);
  const Vec2 right_unknown = CheckedCellCenter(raster, right.unknown_cell);
  if (left_unknown.x != right_unknown.x) {
    return left_unknown.x < right_unknown.x;
  }
  if (left_unknown.y != right_unknown.y) {
    return left_unknown.y < right_unknown.y;
  }
  return std::tie(left.free_cell, left.unknown_cell) <
         std::tie(right.free_cell, right.unknown_cell);
}

FrontierCluster BuildCluster(const TaskRaster& raster,
                             std::vector<GridIndex> cells) {
  std::sort(cells.begin(), cells.end(),
            [&raster](GridIndex left, GridIndex right) {
              return WorldIndexLess(raster, left, right);
            });

  std::vector<FrontierCluster::InterfaceEdge> edges;
  for (const GridIndex cell : cells) {
    for (const DirectionStep& step : kCardinalSteps) {
      const GridIndex neighbor = CheckedOffset(cell, step.dx, step.dy);
      if (raster.Classify(neighbor) != CellState::kUnknown) {
        continue;
      }
      edges.push_back(FrontierCluster::InterfaceEdge{
          .free_cell = cell,
          .unknown_cell = neighbor,
          .direction = step.direction,
          .midpoint = CheckedMidpoint(CheckedCellCenter(raster, cell),
                                      CheckedCellCenter(raster, neighbor)),
      });
    }
  }
  std::sort(edges.begin(), edges.end(),
            [&raster](const FrontierCluster::InterfaceEdge& left,
                      const FrontierCluster::InterfaceEdge& right) {
              return InterfaceEdgeLess(raster, left, right);
            });

  if (edges.empty()) {
    throw std::logic_error("frontier cluster has no interface edge");
  }
  long double midpoint_sum_x = 0.0L;
  long double midpoint_sum_y = 0.0L;
  for (const FrontierCluster::InterfaceEdge& edge : edges) {
    midpoint_sum_x += edge.midpoint.x;
    midpoint_sum_y += edge.midpoint.y;
  }
  const long double divisor = static_cast<long double>(edges.size());
  const long double centroid_x = midpoint_sum_x / divisor;
  const long double centroid_y = midpoint_sum_y / divisor;
  const long double length =
      divisor * static_cast<long double>(raster.geometry().resolution);
  if (!std::isfinite(centroid_x) || !std::isfinite(centroid_y) ||
      !std::isfinite(length) ||
      centroid_x < -std::numeric_limits<double>::max() ||
      centroid_x > std::numeric_limits<double>::max() ||
      centroid_y < -std::numeric_limits<double>::max() ||
      centroid_y > std::numeric_limits<double>::max() ||
      length > std::numeric_limits<double>::max()) {
    throw std::overflow_error("frontier aggregate exceeds double range");
  }

  FrontierCluster cluster{
      .id = 0U,
      .cells = std::move(cells),
      .interface_edges = std::move(edges),
      .canonical_key = {},
      .centroid = Vec2{static_cast<double>(centroid_x),
                       static_cast<double>(centroid_y)},
      .length_m = static_cast<double>(length),
  };
  cluster.id = BuildIdentity(cluster.interface_edges, cluster.canonical_key);
  return cluster;
}

std::vector<GridIndex> FindFrontierCells(const TaskRaster& raster,
                                         GridIndex start,
                                         std::uint32_t& reachable_count) {
  std::deque<GridIndex> queue;
  std::set<GridIndex> visited;
  std::vector<GridIndex> frontier_cells;
  queue.push_back(start);
  visited.insert(start);

  while (!queue.empty()) {
    const GridIndex current = queue.front();
    queue.pop_front();
    if (reachable_count == std::numeric_limits<std::uint32_t>::max()) {
      throw std::overflow_error("reachable free cell count overflows uint32");
    }
    ++reachable_count;

    bool is_frontier = false;
    for (const DirectionStep& step : kCardinalSteps) {
      const GridIndex neighbor = CheckedOffset(current, step.dx, step.dy);
      const CellState state = raster.Classify(neighbor);
      if (state == CellState::kUnknown) {
        is_frontier = true;
      }
      if (state == CellState::kFree && visited.insert(neighbor).second) {
        queue.push_back(neighbor);
      }
    }
    if (is_frontier) {
      frontier_cells.push_back(current);
    }
  }
  return frontier_cells;
}

std::vector<std::vector<GridIndex>> ClusterFrontierCells(
    const std::vector<GridIndex>& frontier_cells) {
  const std::set<GridIndex> frontier_set(frontier_cells.begin(),
                                         frontier_cells.end());
  std::set<GridIndex> assigned;
  std::vector<std::vector<GridIndex>> clusters;
  for (const GridIndex seed : frontier_set) {
    if (!assigned.insert(seed).second) {
      continue;
    }
    std::deque<GridIndex> queue{seed};
    std::vector<GridIndex> cluster;
    while (!queue.empty()) {
      const GridIndex current = queue.front();
      queue.pop_front();
      cluster.push_back(current);
      for (const auto& [dx, dy] : kClusterSteps) {
        const GridIndex neighbor = CheckedOffset(current, dx, dy);
        if (frontier_set.contains(neighbor) &&
            assigned.insert(neighbor).second) {
          queue.push_back(neighbor);
        }
      }
    }
    clusters.push_back(std::move(cluster));
  }
  return clusters;
}

GridIndex SelectStart(const TaskRaster& raster, GridIndex robot_cell,
                      bool& found) {
  if (raster.Classify(robot_cell) == CellState::kFree) {
    found = true;
    return robot_cell;
  }

  const Vec2 robot_center = CheckedCellCenter(raster, robot_cell);
  struct Candidate {
    GridIndex index;
    Vec2 center;
    long double squared_distance;
  };
  std::vector<Candidate> candidates;
  for (const DirectionStep& step : kCardinalSteps) {
    const GridIndex neighbor = CheckedOffset(robot_cell, step.dx, step.dy);
    if (raster.Classify(neighbor) != CellState::kFree) {
      continue;
    }
    const Vec2 center = CheckedCellCenter(raster, neighbor);
    const long double dx = static_cast<long double>(center.x) - robot_center.x;
    const long double dy = static_cast<long double>(center.y) - robot_center.y;
    const long double squared_distance = dx * dx + dy * dy;
    if (!std::isfinite(squared_distance)) {
      throw std::overflow_error("frontier start distance overflows");
    }
    candidates.push_back(Candidate{neighbor, center, squared_distance});
  }
  if (candidates.empty()) {
    found = false;
    return {};
  }
  std::sort(candidates.begin(), candidates.end(),
            [](const Candidate& left, const Candidate& right) {
              if (left.squared_distance != right.squared_distance) {
                return left.squared_distance < right.squared_distance;
              }
              if (left.center.x != right.center.x) {
                return left.center.x < right.center.x;
              }
              if (left.center.y != right.center.y) {
                return left.center.y < right.center.y;
              }
              return left.index < right.index;
            });
  found = true;
  return candidates.front().index;
}

}  // namespace

FrontierDetector::FrontierDetector(FrontierParameters parameters)
    : parameters_(parameters) {
  if (!std::isfinite(parameters_.minimum_cluster_length_m) ||
      parameters_.minimum_cluster_length_m < 0.0) {
    throw std::invalid_argument(
        "minimum frontier cluster length must be finite and nonnegative");
  }
}

FrontierDetection FrontierDetector::Detect(const TaskRaster& raster,
                                           GridIndex robot_cell) const {
  bool has_start = false;
  const GridIndex start = SelectStart(raster, robot_cell, has_start);
  if (!has_start) {
    return FrontierDetection{
        .clusters = {},
        .reachable_free_cell_count = 0U,
        .has_reachable_free_start = false,
        .reason = FrontierDetectionReason::kNoReachableFreeStart,
    };
  }

  std::uint32_t reachable_count = 0U;
  const std::vector<GridIndex> frontier_cells =
      FindFrontierCells(raster, start, reachable_count);
  std::vector<FrontierCluster> clusters;
  for (std::vector<GridIndex>& cells :
       ClusterFrontierCells(frontier_cells)) {
    FrontierCluster cluster = BuildCluster(raster, std::move(cells));
    if (cluster.length_m >= parameters_.minimum_cluster_length_m) {
      clusters.push_back(std::move(cluster));
    }
  }
  std::sort(clusters.begin(), clusters.end(),
            [](const FrontierCluster& left, const FrontierCluster& right) {
              return std::tie(left.id, left.canonical_key, left.cells) <
                     std::tie(right.id, right.canonical_key, right.cells);
            });

  return FrontierDetection{
      .clusters = std::move(clusters),
      .reachable_free_cell_count = reachable_count,
      .has_reachable_free_start = true,
      .reason = FrontierDetectionReason::kOk,
  };
}

}  // namespace lunar::pure_exploration

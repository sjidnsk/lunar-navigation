#include "lunar_pure_exploration_core/candidate_generator.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <numbers>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace lunar::pure_exploration {
namespace {

using Wide = long double;

constexpr std::array<std::pair<std::int32_t, std::int32_t>, 4>
    kCardinalSteps{{{1, 0}, {0, 1}, {-1, 0}, {0, -1}}};
constexpr std::uint64_t kFnvOffset = UINT64_C(14695981039346656037);
constexpr std::uint64_t kFnvPrime = UINT64_C(1099511628211);

bool Finite(Vec2 point) {
  return std::isfinite(point.x) && std::isfinite(point.y);
}

Wide Dot(Vec2 a, Vec2 b) {
  return static_cast<Wide>(a.x) * b.x + static_cast<Wide>(a.y) * b.y;
}

Vec2 Subtract(Vec2 a, Vec2 b) { return Vec2{a.x - b.x, a.y - b.y}; }

bool SamePoint(Vec2 a, Vec2 b) { return a.x == b.x && a.y == b.y; }

GridIndex CheckedOffset(GridIndex index, std::int32_t dx, std::int32_t dy) {
  const std::int64_t x = static_cast<std::int64_t>(index.x) + dx;
  const std::int64_t y = static_cast<std::int64_t>(index.y) + dy;
  if (x < std::numeric_limits<std::int32_t>::min() ||
      x > std::numeric_limits<std::int32_t>::max() ||
      y < std::numeric_limits<std::int32_t>::min() ||
      y > std::numeric_limits<std::int32_t>::max()) {
    throw std::overflow_error("candidate logical neighbor exceeds GridIndex");
  }
  return GridIndex{static_cast<std::int32_t>(x), static_cast<std::int32_t>(y)};
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

std::int64_t Quantize(double value, long double scale,
                      const char* description) {
  if (!std::isfinite(value)) {
    throw std::invalid_argument(std::string(description) + " must be finite");
  }
  const long double scaled = static_cast<long double>(value) * scale;
  if (!std::isfinite(scaled) ||
      scaled < static_cast<long double>(std::numeric_limits<std::int64_t>::min()) ||
      scaled > static_cast<long double>(std::numeric_limits<std::int64_t>::max())) {
    throw std::overflow_error(std::string(description) + " exceeds int64 quantization");
  }
  return static_cast<std::int64_t>(std::llround(scaled));
}

void HashByte(std::uint64_t& hash, std::uint8_t byte) {
  hash ^= byte;
  hash *= kFnvPrime;
}

void HashInt64(std::uint64_t& hash, std::int64_t value) {
  const std::uint64_t bits = static_cast<std::uint64_t>(value);
  for (unsigned shift = 0U; shift < 64U; shift += 8U) {
    HashByte(hash, static_cast<std::uint8_t>((bits >> shift) & UINT64_C(0xff)));
  }
}

std::uint64_t CandidateDisplayId(std::span<const std::int64_t> frontier_key,
                                 const CandidateKey& candidate_key) {
  std::uint64_t hash = kFnvOffset;
  for (const std::int64_t value : frontier_key) {
    HashInt64(hash, value);
  }
  HashInt64(hash, candidate_key.x_mm);
  HashInt64(hash, candidate_key.y_mm);
  HashInt64(hash, candidate_key.yaw_tenth_deg);
  return hash;
}

Wide SquaredDistance(Vec2 a, Vec2 b) {
  const Wide dx = static_cast<Wide>(a.x) - b.x;
  const Wide dy = static_cast<Wide>(a.y) - b.y;
  return dx * dx + dy * dy;
}

bool SameEdge(const FrontierCluster::InterfaceEdge& left,
              const FrontierCluster::InterfaceEdge& right) {
  return left.free_cell == right.free_cell &&
         left.unknown_cell == right.unknown_cell &&
         left.direction == right.direction &&
         SamePoint(left.midpoint, right.midpoint);
}

struct RepresentativeFrame {
  Vec2 midpoint_grid;
  Vec2 unknown_centroid_grid;
  Vec2 unknown_normal_grid;
};

RepresentativeFrame RepresentativeGeometry(
    const TaskRaster& raster, const FrontierCluster::InterfaceEdge& edge) {
  if (!Finite(edge.midpoint) || edge.direction > 3U) {
    throw std::invalid_argument("frontier interface edge must be finite and directed");
  }
  const std::int64_t edge_dx =
      static_cast<std::int64_t>(edge.unknown_cell.x) - edge.free_cell.x;
  const std::int64_t edge_dy =
      static_cast<std::int64_t>(edge.unknown_cell.y) - edge.free_cell.y;
  if (std::abs(edge_dx) + std::abs(edge_dy) != 1 ||
      (edge.direction == 0U && !(edge_dx == 1 && edge_dy == 0)) ||
      (edge.direction == 1U && !(edge_dx == 0 && edge_dy == 1)) ||
      (edge.direction == 2U && !(edge_dx == -1 && edge_dy == 0)) ||
      (edge.direction == 3U && !(edge_dx == 0 && edge_dy == -1))) {
    throw std::invalid_argument("frontier interface direction is inconsistent");
  }
  if (raster.Classify(edge.unknown_cell) != CellState::kUnknown ||
      raster.Classify(edge.free_cell) != CellState::kFree) {
    throw std::invalid_argument("frontier interface classifications are inconsistent");
  }

  std::int64_t sum_dx = 0;
  std::int64_t sum_dy = 0;
  std::size_t count = 0U;
  for (const auto& [dx, dy] : kCardinalSteps) {
    const GridIndex neighbor = CheckedOffset(edge.free_cell, dx, dy);
    if (raster.Classify(neighbor) == CellState::kUnknown) {
      sum_dx += dx;
      sum_dy += dy;
      ++count;
    }
  }
  if (count == 0U) {
    throw std::invalid_argument("selected frontier edge has no local unknown cell");
  }
  std::int64_t normal_dx = sum_dx;
  std::int64_t normal_dy = sum_dy;
  if (normal_dx == 0 && normal_dy == 0) {
    normal_dx = edge_dx;
    normal_dy = edge_dy;
  }
  const Wide norm_squared = static_cast<Wide>(normal_dx) * normal_dx +
                            static_cast<Wide>(normal_dy) * normal_dy;
  if (!(norm_squared > 0.0L) || !std::isfinite(norm_squared)) {
    throw std::invalid_argument("frontier unknown-side normal is invalid");
  }
  const double inverse_norm = 1.0 / std::sqrt(static_cast<double>(norm_squared));
  const double free_grid_x = static_cast<double>(edge.free_cell.x) + 0.5;
  const double free_grid_y = static_cast<double>(edge.free_cell.y) + 0.5;
  return RepresentativeFrame{
      .midpoint_grid =
          Vec2{free_grid_x + 0.5 * static_cast<double>(edge_dx),
               free_grid_y + 0.5 * static_cast<double>(edge_dy)},
      .unknown_centroid_grid =
          Vec2{free_grid_x + static_cast<double>(sum_dx) / count,
               free_grid_y + static_cast<double>(sum_dy) / count},
      .unknown_normal_grid =
          Vec2{static_cast<double>(normal_dx) * inverse_norm,
               static_cast<double>(normal_dy) * inverse_norm},
  };
}

Vec2 RotateGridVectorToWorld(const GridGeometry& geometry,
                             Vec2 grid_vector) {
  GridGeometry direction_geometry = geometry;
  direction_geometry.width = 0U;
  direction_geometry.height = 0U;
  direction_geometry.resolution = 1.0;
  direction_geometry.origin_x = 0.0;
  direction_geometry.origin_y = 0.0;
  const auto world =
      OccupancyGridView::GridToWorld(direction_geometry, grid_vector);
  if (!world.has_value()) {
    throw std::overflow_error("candidate direction transform is not finite");
  }
  return *world;
}

struct BucketKey {
  std::int64_t x;
  std::int64_t y;
  auto operator<=>(const BucketKey&) const = default;
};

std::int64_t BucketCoordinate(double value, double spacing) {
  const long double bucket =
      std::floor(static_cast<long double>(value) / spacing);
  if (!std::isfinite(bucket) ||
      bucket < std::numeric_limits<std::int64_t>::min() ||
      bucket > std::numeric_limits<std::int64_t>::max()) {
    throw std::overflow_error("candidate position bucket exceeds int64");
  }
  return static_cast<std::int64_t>(bucket);
}

bool HasClosePosition(const std::map<BucketKey, std::vector<Vec2>>& buckets,
                      Vec2 point, double spacing) {
  const BucketKey key{BucketCoordinate(point.x, spacing),
                      BucketCoordinate(point.y, spacing)};
  const Wide spacing_squared = static_cast<Wide>(spacing) * spacing;
  for (std::int64_t dy = -1; dy <= 1; ++dy) {
    for (std::int64_t dx = -1; dx <= 1; ++dx) {
      if ((dx < 0 && key.x == std::numeric_limits<std::int64_t>::min()) ||
          (dx > 0 && key.x == std::numeric_limits<std::int64_t>::max()) ||
          (dy < 0 && key.y == std::numeric_limits<std::int64_t>::min()) ||
          (dy > 0 && key.y == std::numeric_limits<std::int64_t>::max())) {
        continue;
      }
      const auto found = buckets.find(BucketKey{key.x + dx, key.y + dy});
      if (found == buckets.end()) {
        continue;
      }
      for (const Vec2 accepted : found->second) {
        if (SquaredDistance(point, accepted) < spacing_squared) {
          return true;
        }
      }
    }
  }
  return false;
}

void AddPosition(std::map<BucketKey, std::vector<Vec2>>& buckets,
                 Vec2 point, double spacing) {
  buckets[BucketKey{BucketCoordinate(point.x, spacing),
                    BucketCoordinate(point.y, spacing)}]
      .push_back(point);
}

std::size_t ValidateLimits(CandidateGenerator::Limits limits) {
  if (limits.maximum_position_probes == 0U ||
      limits.maximum_candidate_views == 0U ||
      limits.maximum_collision_work_units == 0U) {
    throw std::invalid_argument("candidate resource limits must be positive");
  }
  return limits.maximum_collision_work_units;
}

}  // namespace

CandidateKey MakeCandidateKey(Pose2 pose) {
  pose.yaw = NormalizeYaw(pose.yaw);
  return CandidateKey{
      Quantize(pose.x, 1000.0L, "candidate x"),
      Quantize(pose.y, 1000.0L, "candidate y"),
      Quantize(pose.yaw, 1800.0L / std::numbers::pi_v<long double>,
               "candidate yaw"),
  };
}

CandidateGenerator::CandidateGenerator(PlatformGeometry platform,
                                       CandidateParameters parameters,
                                       Limits limits)
    : validator_(std::move(platform), ValidateLimits(limits)),
      parameters_(parameters),
      limits_(limits) {
  for (std::size_t i = 0U; i < parameters_.yaw_offsets_rad.size(); ++i) {
    if (!std::isfinite(parameters_.yaw_offsets_rad[i]) ||
        (i > 0U && !(parameters_.yaw_offsets_rad[i - 1U] <
                     parameters_.yaw_offsets_rad[i]))) {
      throw std::invalid_argument("candidate yaw offsets must be finite and strictly ascending");
    }
  }
}

double CandidateGenerator::platform_length_m() const {
  return validator_.platform_length_m();
}

double CandidateGenerator::platform_width_m() const {
  return validator_.platform_width_m();
}

double CandidateGenerator::footprint_circumscribed_radius_m() const {
  return validator_.footprint_circumscribed_radius_m();
}

double CandidateGenerator::minimum_spacing_m(double resolution_m) const {
  if (!std::isfinite(resolution_m) || resolution_m <= 0.0) {
    throw std::invalid_argument("candidate resolution must be finite and positive");
  }
  return std::max(validator_.platform_length_m(), 2.0 * resolution_m);
}

double CandidateGenerator::minimum_standoff_m() const {
  return validator_.minimum_standoff_m();
}

double CandidateGenerator::maximum_extra_search_m() const {
  return 2.0 * validator_.platform_length_m();
}

std::size_t CandidateGenerator::maximum_search_step(double resolution_m) const {
  if (!std::isfinite(resolution_m) || resolution_m <= 0.0) {
    throw std::invalid_argument("candidate resolution must be finite and positive");
  }
  const double maximum_extra = maximum_extra_search_m();
  const double ratio = maximum_extra / resolution_m;
  const long double exact_size_max =
      static_cast<long double>(std::numeric_limits<std::size_t>::max());
  if (!std::isfinite(ratio) ||
      static_cast<long double>(ratio) >= exact_size_max) {
    throw std::overflow_error("candidate search step count exceeds size_t");
  }
  std::size_t step = static_cast<std::size_t>(std::floor(ratio));
  while (step > 0U && static_cast<double>(step) * resolution_m > maximum_extra) {
    --step;
  }
  while (step < std::numeric_limits<std::size_t>::max() &&
         static_cast<double>(step + 1U) * resolution_m <= maximum_extra) {
    ++step;
  }
  return step;
}

std::vector<CandidateView> CandidateGenerator::Generate(
    const TaskRaster& raster,
    std::span<const FrontierCluster> frontiers,
    const CandidatePositionAcceptance& accept_position) const {
  const double resolution = raster.geometry().resolution;
  const double spacing = minimum_spacing_m(resolution);
  const double spacing_grid = spacing / resolution;
  if (!std::isfinite(spacing_grid) || spacing_grid <= 0.0) {
    throw std::overflow_error("candidate grid spacing is not finite and positive");
  }
  const std::size_t maximum_step = maximum_search_step(resolution);
  const double standoff = minimum_standoff_m();

  std::vector<std::size_t> order(frontiers.size());
  for (std::size_t index = 0U; index < frontiers.size(); ++index) {
    order[index] = index;
    const FrontierCluster& cluster = frontiers[index];
    if (cluster.canonical_key.empty() || cluster.interface_edges.empty() ||
        !Finite(cluster.centroid) || !std::isfinite(cluster.length_m) ||
        cluster.length_m < 0.0) {
      throw std::invalid_argument("frontier cluster structure must be finite and nonempty");
    }
    for (const auto& edge : cluster.interface_edges) {
      if (!Finite(edge.midpoint) || edge.direction > 3U) {
        throw std::invalid_argument("frontier interface edge is invalid");
      }
    }
  }
  std::sort(order.begin(), order.end(), [&](std::size_t left, std::size_t right) {
    return frontiers[left].canonical_key < frontiers[right].canonical_key;
  });
  for (std::size_t index = 1U; index < order.size(); ++index) {
    if (frontiers[order[index - 1U]].canonical_key ==
        frontiers[order[index]].canonical_key) {
      throw std::invalid_argument("frontier canonical keys must be unique");
    }
  }

  std::vector<CandidateView> output;
  std::map<BucketKey, std::vector<Vec2>> position_buckets;
  std::size_t position_probes = 0U;
  std::size_t candidate_views = 0U;
  std::size_t collision_work = 0U;

  for (const std::size_t original_index : order) {
    const FrontierCluster& frontier = frontiers[original_index];
    std::shared_ptr<const std::vector<std::int64_t>> frontier_canonical_key;
    const std::size_t count = frontier.interface_edges.size();
    const std::array<std::size_t, 3> representative_indices{
        (count - 1U) / 4U,
        (count - 1U) / 2U,
        (count - 1U) - count / 4U,
    };
    std::vector<FrontierCluster::InterfaceEdge> representatives;
    for (const std::size_t index : representative_indices) {
      const auto& edge = frontier.interface_edges[index];
      if (std::none_of(representatives.begin(), representatives.end(),
                       [&](const auto& existing) {
                         return SameEdge(existing, edge);
                       })) {
        representatives.push_back(edge);
      }
    }

    for (const auto& edge : representatives) {
      const RepresentativeFrame frame = RepresentativeGeometry(raster, edge);
      const std::size_t search_step_count = maximum_step + 1U;
      for (std::size_t step = 0U; step < search_step_count; ++step) {
        if (position_probes == limits_.maximum_position_probes) {
          throw std::length_error("candidate position probe limit exceeded");
        }
        ++position_probes;
        const double distance =
            standoff + static_cast<double>(step) * resolution;
        const Vec2 center_grid{
            frame.midpoint_grid.x -
                frame.unknown_normal_grid.x * distance / resolution,
            frame.midpoint_grid.y -
                frame.unknown_normal_grid.y * distance / resolution,
        };
        const auto center_world = raster.GridToWorld(center_grid);
        if (!center_world.has_value()) {
          throw std::overflow_error("candidate center exceeds finite coordinates");
        }
        const Vec2 center = *center_world;
        const auto center_cell = raster.WorldToCell(center);
        if (!center_cell.has_value() || !raster.IsMapBacked(*center_cell) ||
            raster.Classify(*center_cell) != CellState::kFree ||
            HasClosePosition(position_buckets, center_grid, spacing_grid)) {
          continue;
        }
        Vec2 yaw_vector =
            Subtract(frame.unknown_centroid_grid, center_grid);
        if (Dot(yaw_vector, yaw_vector) == 0.0L) {
          yaw_vector = frame.unknown_normal_grid;
        }
        const Vec2 world_yaw_vector =
            RotateGridVectorToWorld(raster.geometry(), yaw_vector);
        const double standard_yaw =
            std::atan2(world_yaw_vector.y, world_yaw_vector.x);
        std::vector<CandidateView> group;
        group.reserve(parameters_.yaw_offsets_rad.size());
        for (const double yaw_offset : parameters_.yaw_offsets_rad) {
          const Pose2 pose{center.x, center.y,
                           NormalizeYaw(standard_yaw + yaw_offset)};
          if (!validator_.IsTaskFree(raster, pose, collision_work)) {
            continue;
          }
          if (candidate_views == limits_.maximum_candidate_views) {
            throw std::length_error("candidate view limit exceeded");
          }
          ++candidate_views;
          const CandidateKey key = MakeCandidateKey(pose);
          const double frontier_distance =
              std::hypot(center.x - edge.midpoint.x,
                         center.y - edge.midpoint.y);
          if (!std::isfinite(frontier_distance)) {
            throw std::overflow_error("candidate frontier distance exceeds double");
          }
          if (!frontier_canonical_key) {
            frontier_canonical_key =
                std::make_shared<const std::vector<std::int64_t>>(
                    frontier.canonical_key);
          }
          group.push_back(CandidateView{
              .id = CandidateDisplayId(frontier.canonical_key, key),
              .frontier_id = frontier.id,
              .frontier_index = original_index,
              .key = key,
              .pose = pose,
              .frontier_distance_m = frontier_distance,
              .frontier_canonical_key = frontier_canonical_key,
          });
        }
        if (!group.empty()) {
          // The caller can supply an exact downstream feasibility predicate
          // (for example the planner's inflated global projection).  A
          // rejection means this free, body-safe position is not actionable,
          // so continue searching farther into known space.  Without a
          // predicate, retain the original one-position-per-edge budget.
          if (accept_position && !accept_position(group.front().pose)) {
            continue;
          }
          if (output.size() > output.max_size() - group.size()) {
            throw std::length_error("candidate output exceeds storage capacity");
          }
          AddPosition(position_buckets, center_grid, spacing_grid);
          output.insert(output.end(), group.begin(), group.end());
          break;
        }
      }
    }
  }
  return output;
}

}  // namespace lunar::pure_exploration

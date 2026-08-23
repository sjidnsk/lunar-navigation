#include "hopper/anytime_hopper_planner.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <numbers>
#include <numeric>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "hopper/ballistic_envelope.hpp"
#include "hopper/ballistic_kinematics.hpp"
#include "shared/anytime_ara_star.hpp"
#include "shared/controlled_work.hpp"
#include "shared/edge_validation_cache.hpp"

namespace lunar::pure_planning::hopper {
namespace {

constexpr double kTolerance = 1.0e-9;
constexpr double kMinimumEdgeCost = 1.0e-6;

[[nodiscard]] HopperPlanResult Failure(
    const LocalPlanStatus status, std::string reason_code,
    const LocalPlanMetrics metrics = {}) {
  return HopperPlanResult{
      .status = status,
      .reason_code = std::move(reason_code),
      .metrics = metrics,
  };
}

[[nodiscard]] bool Finite(const Vec3& value) noexcept {
  return std::isfinite(value.x) && std::isfinite(value.y) &&
         std::isfinite(value.z);
}

[[nodiscard]] bool Finite(const Quaternion& value) noexcept {
  return std::isfinite(value.w) && std::isfinite(value.x) &&
         std::isfinite(value.y) && std::isfinite(value.z);
}

[[nodiscard]] bool FiniteXY(const Vec3& value) noexcept {
  return std::isfinite(value.x) && std::isfinite(value.y);
}

[[nodiscard]] std::optional<double> YawFromQuaternion(
    const Quaternion& quaternion) noexcept {
  if (!Finite(quaternion)) {
    return std::nullopt;
  }
  const double norm = std::sqrt(
      quaternion.w * quaternion.w + quaternion.x * quaternion.x +
      quaternion.y * quaternion.y + quaternion.z * quaternion.z);
  if (!std::isfinite(norm) || std::abs(norm - 1.0) > 1.0e-6) {
    return std::nullopt;
  }
  const double yaw = std::atan2(
      2.0 * (quaternion.w * quaternion.z +
             quaternion.x * quaternion.y),
      1.0 - 2.0 * (quaternion.y * quaternion.y +
                   quaternion.z * quaternion.z));
  return std::isfinite(yaw) ? std::optional<double>{yaw} : std::nullopt;
}

[[nodiscard]] bool GoalYawSatisfied(
    const double start_yaw, const GoalRegion& goal) noexcept {
  if (!goal.yaw_rad.has_value()) {
    return true;
  }
  return std::abs(std::remainder(*goal.yaw_rad - start_yaw,
                                 2.0 * std::numbers::pi)) <=
         goal.yaw_tolerance_rad + kTolerance;
}

[[nodiscard]] bool ValidCapability(
    const HopperCapability& capability) noexcept {
  return AvailableSingleHopDeltaV(capability).ok() &&
         Finite(capability.gravity_mps2) &&
         std::isfinite(capability.reference_horizontal_range_m) &&
         capability.reference_horizontal_range_m > 0.0 &&
         std::isfinite(capability.landing_support_radius_m) &&
         capability.landing_support_radius_m >= 0.0 &&
         std::isfinite(capability.flight_collision_radius_m) &&
         capability.flight_collision_radius_m >= 0.0 &&
         std::isfinite(capability.flight_map_margin_m) &&
         capability.flight_map_margin_m >= 0.0 &&
         std::isfinite(capability.maximum_landing_slope_rad) &&
         capability.maximum_landing_slope_rad >= 0.0;
}

[[nodiscard]] bool ValidTerrain(
    const shared::LocalTerrainProjection& terrain) noexcept {
  return terrain.map != nullptr && terrain.map->cell_count() != 0U &&
         terrain.map->cell_count() <=
             std::numeric_limits<std::size_t>::max() - 2U &&
         terrain.free_with_height.size() == terrain.map->cell_count() &&
         terrain.slope_rad.size() == terrain.map->cell_count();
}

[[nodiscard]] bool DistanceIndexable(
    const double distance_m, const shared::MapSnapshot& map) noexcept {
  const double radius_cells = distance_m / map.resolution_m();
  return std::isfinite(radius_cells) && radius_cells > 0.0 &&
         radius_cells <=
             static_cast<double>(
                 std::numeric_limits<std::int32_t>::max() - 1);
}

struct EdgeKey final {
  std::size_t source{};
  std::size_t target{};

  bool operator==(const EdgeKey&) const = default;
};

struct EdgeKeyHash final {
  [[nodiscard]] std::size_t operator()(const EdgeKey& key) const noexcept {
    return key.source ^
           (key.target + 0x9e3779b97f4a7c15ULL + (key.source << 6U) +
            (key.source >> 2U));
  }
};

struct EdgeEvaluation final {
  bool valid{};
  MinimumSingleHopEnvelopeEvidence hop;
  double cost{};
};

struct TimeSection final {
  double begin_s{};
  double end_s{};
};

[[nodiscard]] double Distance(const Vec3 first, const Vec3 second) noexcept {
  return std::hypot(std::hypot(first.x - second.x, first.y - second.y),
                    first.z - second.z);
}

[[nodiscard]] double Norm(const Vec3 value) noexcept {
  return std::hypot(std::hypot(value.x, value.y), value.z);
}

[[nodiscard]] double PointRectangleDistanceSquared(
    const Vec3 point, const double x0, const double x1, const double y0,
    const double y1) noexcept {
  const double nearest_x = std::clamp(point.x, x0, x1);
  const double nearest_y = std::clamp(point.y, y0, y1);
  const double offset_x = point.x - nearest_x;
  const double offset_y = point.y - nearest_y;
  return offset_x * offset_x + offset_y * offset_y;
}

[[nodiscard]] bool SegmentIntersectsRectangle(
    const Vec3 begin, const Vec3 end, const double x0, const double x1,
    const double y0, const double y1) noexcept {
  double lower = 0.0;
  double upper = 1.0;
  const double dx = end.x - begin.x;
  const double dy = end.y - begin.y;
  const auto clip = [&](const double p, const double q) {
    if (std::abs(p) <= std::numeric_limits<double>::epsilon()) {
      return q >= 0.0;
    }
    const double ratio = q / p;
    if (p < 0.0) {
      lower = std::max(lower, ratio);
    } else {
      upper = std::min(upper, ratio);
    }
    return lower <= upper;
  };
  return clip(-dx, begin.x - x0) && clip(dx, x1 - begin.x) &&
         clip(-dy, begin.y - y0) && clip(dy, y1 - begin.y);
}

[[nodiscard]] double SegmentRectangleDistanceSquared(
    const Vec3 begin, const Vec3 end, const double x0, const double x1,
    const double y0, const double y1) noexcept {
  const double dx = end.x - begin.x;
  const double dy = end.y - begin.y;
  const double squared_length = dx * dx + dy * dy;
  if (squared_length <= std::numeric_limits<double>::epsilon()) {
    return PointRectangleDistanceSquared(begin, x0, x1, y0, y1);
  }
  if (SegmentIntersectsRectangle(begin, end, x0, x1, y0, y1)) {
    return 0.0;
  }
  double squared_distance = std::min(
      PointRectangleDistanceSquared(begin, x0, x1, y0, y1),
      PointRectangleDistanceSquared(end, x0, x1, y0, y1));
  const double inverse_squared_length = 1.0 / squared_length;
  for (const Vec3 corner : {Vec3{x0, y0, 0.0}, Vec3{x1, y0, 0.0},
                            Vec3{x1, y1, 0.0}, Vec3{x0, y1, 0.0}}) {
    const double projection = std::clamp(
        ((corner.x - begin.x) * dx + (corner.y - begin.y) * dy) *
            inverse_squared_length,
        0.0, 1.0);
    const double offset_x = corner.x - (begin.x + projection * dx);
    const double offset_y = corner.y - (begin.y + projection * dy);
    squared_distance = std::min(
        squared_distance, offset_x * offset_x + offset_y * offset_y);
  }
  return squared_distance;
}

[[nodiscard]] std::pair<double, double> CoordinateRange(
    const BallisticArc& arc, const std::size_t axis, const double begin_s,
    const double end_s) noexcept {
  const auto coordinate = [axis](const Vec3 value) {
    return axis == 0U ? value.x : (axis == 1U ? value.y : value.z);
  };
  const double initial = coordinate(arc.launch_position_m);
  const double velocity = coordinate(arc.launch_velocity_mps);
  const double acceleration = coordinate(arc.gravity_mps2);
  const auto value = [&](const double time_s) {
    return initial + velocity * time_s +
           0.5 * acceleration * time_s * time_s;
  };
  double minimum = std::min(value(begin_s), value(end_s));
  double maximum = std::max(value(begin_s), value(end_s));
  if (std::abs(acceleration) > std::numeric_limits<double>::epsilon()) {
    const double stationary = -velocity / acceleration;
    if (stationary > begin_s && stationary < end_s) {
      minimum = std::min(minimum, value(stationary));
      maximum = std::max(maximum, value(stationary));
    }
  }
  return {minimum, maximum};
}

[[nodiscard]] std::size_t StableEdgeIndex(
    const EdgeKey key) noexcept {
  std::uint64_t hash = 1469598103934665603ULL;
  for (std::uint64_t value : {static_cast<std::uint64_t>(key.source),
                              static_cast<std::uint64_t>(key.target)}) {
    for (std::size_t byte = 0U; byte < sizeof(value); ++byte) {
      hash ^= value & 0xffU;
      hash *= 1099511628211ULL;
      value >>= 8U;
    }
  }
  return static_cast<std::size_t>(hash);
}

class HopperSearchGraph final {
  struct LandingSite final {
    std::size_t state{};
    Vec3 position;
  };

 public:
  HopperSearchGraph(const HopperPlanRequest& request, PointGoal goal,
                    const Vec3 exact_goal)
      : request_(request),
        terrain_(*request.terrain),
        capability_(*request.capability),
        map_(*request.terrain->map),
        goal_(std::move(goal)),
        exact_goal_(exact_goal),
        goal_state_(map_.cell_count() + 1U) {}

  [[nodiscard]] std::size_t state_count() const noexcept {
    return map_.cell_count() + 2U;
  }

  [[nodiscard]] std::size_t goal_state() const noexcept {
    return goal_state_;
  }

  [[nodiscard]] bool IsGoalState(const std::size_t state) const {
    if (state >= state_count()) {
      return false;
    }
    if (state == goal_state_) {
      return true;
    }
    const Vec3 position = PositionForState(state);
    return std::hypot(position.x - goal_.position_m.x,
                      position.y - goal_.position_m.y) <=
           goal_.tolerance_m + kTolerance;
  }

  [[nodiscard]] bool ValidateStart() const noexcept {
    return LandingAllowed(request_.start.pose.position_m);
  }

  [[nodiscard]] std::size_t validation_count() const noexcept {
    return edge_cache_.evaluation_count();
  }

  [[nodiscard]] const EdgeEvaluation* Edge(
      const std::size_t source, const std::size_t target) const noexcept {
    const auto found = accepted_edges_.find(EdgeKey{source, target});
    return found == accepted_edges_.end() ? nullptr : found->second;
  }

  [[nodiscard]] Vec3 PositionForState(const std::size_t state) const {
    if (state == 0U) {
      return request_.start.pose.position_m;
    }
    if (state == goal_state_) {
      return exact_goal_;
    }
    const std::size_t index = state - 1U;
    return map_.CellCenter(shared::GridCell{
        .x = static_cast<std::int32_t>(index % map_.width()),
        .y = static_cast<std::int32_t>(index / map_.width()),
    });
  }

  void Expand(const std::size_t state,
              std::vector<shared::GraphEdge>& edges) {
    if (state >= state_count() || state == goal_state_ ||
        request_.control.canceled() || request_.control.expired()) {
      return;
    }
    const Vec3 source = PositionForState(state);
    if (AppendEdge(state, goal_state_, source, exact_goal_, edges)) {
      return;
    }
    const auto source_cell = map_.PositionToCell(
        Vec2{.x = source.x, .y = source.y});
    if (!source_cell.has_value()) {
      return;
    }
    std::vector<std::uint8_t> emitted_states(map_.cell_count(), 0U);
    const auto append_cell = [&](const std::int64_t target_x,
                                 const std::int64_t target_y) {
      if (target_x < 0 || target_y < 0 ||
          target_x >= static_cast<std::int64_t>(map_.width()) ||
          target_y >= static_cast<std::int64_t>(map_.height())) {
        return false;
      }
      const shared::GridCell target_cell{
          .x = static_cast<std::int32_t>(target_x),
          .y = static_cast<std::int32_t>(target_y),
      };
      const std::size_t target = map_.Index(target_cell) + 1U;
      const std::size_t target_index = target - 1U;
      if (target == state || emitted_states[target_index] != 0U) {
        return false;
      }
      emitted_states[target_index] = 1U;
      const Vec3 landing = map_.CellCenter(target_cell);
      return LandingAllowed(landing) &&
             AppendEdge(state, target, source, landing, edges);
    };

    // Goal-region cells are visited in outward rings and edge certification
    // stops at the first usable connector. The radius is capped by the map, so
    // the complete in-map XY goal disk is covered without eagerly certifying
    // every landing in it.
    const auto goal_cell = map_.PositionToCell(
        Vec2{.x = goal_.position_m.x, .y = goal_.position_m.y});
    if (goal_cell.has_value()) {
      const double map_radius_cells = static_cast<double>(
          std::max(map_.width(), map_.height()));
      const auto maximum_goal_ring = static_cast<std::int64_t>(std::ceil(
          std::min(goal_.tolerance_m / map_.resolution_m(),
                   map_radius_cells)));
      const auto append_goal_offset = [&](const std::int64_t dx,
                                          const std::int64_t dy) {
        const std::int64_t target_x =
            static_cast<std::int64_t>(goal_cell->x) + dx;
        const std::int64_t target_y =
            static_cast<std::int64_t>(goal_cell->y) + dy;
        if (target_x < 0 || target_y < 0 ||
            target_x >= static_cast<std::int64_t>(map_.width()) ||
            target_y >= static_cast<std::int64_t>(map_.height())) {
          return false;
        }
        const Vec3 landing = map_.CellCenter(shared::GridCell{
            .x = static_cast<std::int32_t>(target_x),
            .y = static_cast<std::int32_t>(target_y),
        });
        return std::hypot(landing.x - goal_.position_m.x,
                          landing.y - goal_.position_m.y) <=
                   goal_.tolerance_m + kTolerance &&
               append_cell(target_x, target_y);
      };
      for (std::int64_t ring = 0; ring <= maximum_goal_ring; ++ring) {
        if (request_.control.canceled() || request_.control.expired()) {
          return;
        }
        if (ring == 0) {
          if (append_goal_offset(0, 0)) {
            return;
          }
          continue;
        }
        for (std::int64_t dx = -ring; dx <= ring; ++dx) {
          if (request_.control.canceled() || request_.control.expired()) {
            return;
          }
          if (append_goal_offset(dx, -ring) ||
              append_goal_offset(dx, ring)) {
            return;
          }
        }
        for (std::int64_t dy = -ring + 1; dy < ring; ++dy) {
          if (request_.control.canceled() || request_.control.expired()) {
            return;
          }
          if (append_goal_offset(-ring, dy) ||
              append_goal_offset(ring, dy)) {
            return;
          }
        }
      }
    }

    struct LandingCandidate final {
      std::size_t state{};
      Vec3 position;
      double goal_distance_m{};
    };
    std::vector<LandingCandidate> candidates;
    const double maximum_range = capability_.reference_horizontal_range_m;
    if (!EnsureTerrainIndex()) {
      return;
    }

    // The index contains every legal landing in the frozen map. Each expansion
    // still enumerates the complete capability disk, but avoids rescanning
    // cells that can never be landings before deterministic goal-first edge
    // certification.
    for (const LandingSite& site : landing_sites_) {
      if (request_.control.canceled() || request_.control.expired()) {
        return;
      }
      const std::size_t target = site.state;
      const std::size_t index = target - 1U;
      if (target == state || emitted_states[index] != 0U) {
        continue;
      }
      emitted_states[index] = 1U;
      const Vec3 landing = site.position;
      const double distance = std::hypot(landing.x - source.x,
                                         landing.y - source.y);
      if (!Finite(landing) || !std::isfinite(distance) ||
          distance <= kTolerance ||
          distance > maximum_range + kTolerance) {
        continue;
      }
      candidates.push_back(LandingCandidate{
          .state = target,
          .position = landing,
          .goal_distance_m = std::max(
              0.0,
              std::hypot(landing.x - goal_.position_m.x,
                         landing.y - goal_.position_m.y) -
                  goal_.tolerance_m),
      });
    }
    std::ranges::sort(candidates, [](const LandingCandidate& lhs,
                                     const LandingCandidate& rhs) {
      return std::tie(lhs.goal_distance_m, lhs.state) <
             std::tie(rhs.goal_distance_m, rhs.state);
    });
    for (const LandingCandidate& candidate : candidates) {
      if (request_.control.canceled() || request_.control.expired()) {
        return;
      }
      static_cast<void>(AppendEdge(
          state, candidate.state, source, candidate.position, edges));
    }
  }

  [[nodiscard]] double Heuristic(const std::size_t state) const noexcept {
    if (state == goal_state_) {
      return 0.0;
    }
    const Vec3 position = PositionForState(state);
    return std::max(
        0.0,
        std::hypot(position.x - goal_.position_m.x,
                   position.y - goal_.position_m.y) - goal_.tolerance_m);
  }

 private:
  [[nodiscard]] bool EnsureTerrainIndex() {
    if (terrain_index_ready_) {
      return true;
    }
    const std::span<const float> occupancy = map_.FloatLayer("occupancy");
    const std::span<const float> elevation = map_.FloatLayer("elevation");
    if (occupancy.size() != map_.cell_count() ||
        elevation.size() != map_.cell_count()) {
      return false;
    }
    landing_sites_.reserve(map_.cell_count());
    height_cache_.resize(map_.cell_count());
    height_cached_.assign(map_.cell_count(), std::uint8_t{0U});
    for (std::size_t index = 0U; index < map_.cell_count(); ++index) {
      if (index != 0U && index % 1024U == 0U &&
          (request_.control.canceled() || request_.control.expired())) {
        return false;
      }
      const shared::GridCell cell{
          .x = static_cast<std::int32_t>(index % map_.width()),
          .y = static_cast<std::int32_t>(index / map_.width()),
      };
      if (terrain_.free_with_height[index] != 0U &&
          std::isfinite(terrain_.slope_rad[index]) &&
          terrain_.slope_rad[index] <=
              capability_.maximum_landing_slope_rad) {
        landing_sites_.push_back(LandingSite{
            .state = index + 1U,
            .position = map_.CellCenter(cell),
        });
      }
    }
    terrain_index_ready_ = true;
    return true;
  }

  [[nodiscard]] std::optional<double> ValidateStraightFlightColumns(
      const BallisticArc& arc, const std::span<const float> occupancy,
      const std::span<const float> elevation, const double radius,
      const double tolerated_radius_squared) {
    const auto cell_index = [&](const double coordinate,
                                const double origin) {
      return static_cast<long long>(
          std::floor((coordinate - origin) / map_.resolution_m()));
    };
    const long long x0 = cell_index(
        std::min(arc.launch_position_m.x, arc.landing_position_m.x) - radius,
        map_.origin_m().x);
    const long long x1 = cell_index(
        std::max(arc.launch_position_m.x, arc.landing_position_m.x) + radius,
        map_.origin_m().x);
    const long long y0 = cell_index(
        std::min(arc.launch_position_m.y, arc.landing_position_m.y) - radius,
        map_.origin_m().y);
    const long long y1 = cell_index(
        std::max(arc.launch_position_m.y, arc.landing_position_m.y) + radius,
        map_.origin_m().y);
    if (x0 < 0 || y0 < 0 || x1 >= static_cast<long long>(map_.width()) ||
        y1 >= static_cast<long long>(map_.height())) {
      return std::nullopt;
    }

    double maximum_elevation = -std::numeric_limits<double>::infinity();
    for (long long y = y0; y <= y1; ++y) {
      if (request_.control.canceled() || request_.control.expired()) {
        return std::nullopt;
      }
      for (long long x = x0; x <= x1; ++x) {
        const shared::GridCell cell{
            .x = static_cast<std::int32_t>(x),
            .y = static_cast<std::int32_t>(y),
        };
        const double cell_x0 = map_.origin_m().x +
                               static_cast<double>(cell.x) *
                                   map_.resolution_m();
        const double cell_y0 = map_.origin_m().y +
                               static_cast<double>(cell.y) *
                                   map_.resolution_m();
        if (SegmentRectangleDistanceSquared(
                arc.launch_position_m, arc.landing_position_m, cell_x0,
                cell_x0 + map_.resolution_m(), cell_y0,
                cell_y0 + map_.resolution_m()) > tolerated_radius_squared) {
          continue;
        }
        const std::size_t index = map_.Index(cell);
        const float occupied = occupancy[index];
        const float cell_elevation = elevation[index];
        if (!std::isfinite(occupied) || occupied < 0.0F ||
            occupied > 1.0F || !std::isfinite(cell_elevation)) {
          return std::nullopt;
        }
        maximum_elevation =
            std::max(maximum_elevation,
                     static_cast<double>(cell_elevation));
        if (occupied < 0.5F) {
          continue;
        }
        if (height_cached_[index] == 0U) {
          height_cache_[index] = request_.estimate_height(map_, cell);
          height_cached_[index] = 1U;
        }
        const shared::ObstacleHeight& obstacle = height_cache_[index];
        if (!obstacle.flyover_allowed ||
            !std::isfinite(obstacle.height_m) || obstacle.height_m <= 0.0) {
          return std::nullopt;
        }
      }
    }
    return maximum_elevation;
  }

  [[nodiscard]] bool LandingAllowed(const Vec3 position) const noexcept {
    const auto cell = map_.PositionToCell(
        Vec2{.x = position.x, .y = position.y});
    if (!cell.has_value()) {
      return false;
    }
    const std::size_t index = map_.Index(*cell);
    return terrain_.free_with_height[index] != 0U &&
           std::isfinite(terrain_.slope_rad[index]) &&
           terrain_.slope_rad[index] <=
               capability_.maximum_landing_slope_rad;
  }

  [[nodiscard]] bool ValidateFlightTube(const BallisticArc& arc) {
    const std::span<const float> occupancy = map_.FloatLayer("occupancy");
    const std::span<const float> elevation = map_.FloatLayer("elevation");
    if (!request_.estimate_height || occupancy.size() != map_.cell_count() ||
        elevation.size() != map_.cell_count()) {
      return false;
    }
    const double radius = capability_.flight_collision_radius_m +
                          capability_.flight_map_margin_m;
    if (!std::isfinite(radius) || radius <= 0.0 ||
        !std::isfinite(arc.flight_time_s) || arc.flight_time_s <= 0.0) {
      return false;
    }
    if (!EnsureTerrainIndex()) {
      return false;
    }
    const double radius_squared = radius * radius;
    const double tolerated_radius_squared =
        (radius + kTolerance) * (radius + kTolerance);
    // With zero horizontal gravity the arc is exactly a straight line in XY.
    // Certify every column in that planar capsule once: invalid map columns and
    // non-flyover obstacles are unbounded blockers in the existing semantics.
    // Curved XY arcs retain the full section-by-section path below.
    std::optional<double> straight_corridor_maximum_elevation;
    if (arc.gravity_mps2.x == 0.0 && arc.gravity_mps2.y == 0.0) {
      straight_corridor_maximum_elevation = ValidateStraightFlightColumns(
          arc, occupancy, elevation, radius, tolerated_radius_squared);
      if (!straight_corridor_maximum_elevation.has_value()) {
        return false;
      }
    }

    const double spatial_tolerance = 0.25 * map_.resolution_m();
    const double peak_time = arc.gravity_mps2.z < 0.0
                                 ? std::clamp(-arc.launch_velocity_mps.z /
                                                  arc.gravity_mps2.z,
                                              0.0, arc.flight_time_s)
                                 : 0.0;
    const double peak_z =
        EvaluateBallisticState(arc, peak_time).position_m.z;
    const bool clears_launch_contact =
        std::isfinite(peak_z) &&
        peak_z > arc.launch_position_m.z + radius + kTolerance;
    const bool clears_landing_contact =
        std::isfinite(peak_z) &&
        peak_z > arc.landing_position_m.z + radius + kTolerance;
    std::vector<TimeSection> pending{
        TimeSection{.begin_s = 0.0, .end_s = arc.flight_time_s}};
    std::vector<TimeSection> sections;
    while (!pending.empty()) {
      if (request_.control.canceled() || request_.control.expired()) {
        return false;
      }
      const TimeSection section = pending.back();
      pending.pop_back();
      const double midpoint = std::midpoint(section.begin_s, section.end_s);
      if (!(midpoint > section.begin_s && midpoint < section.end_s)) {
        return false;
      }
      const Vec3 begin =
          EvaluateBallisticState(arc, section.begin_s).position_m;
      const Vec3 end = EvaluateBallisticState(arc, section.end_s).position_m;
      const double duration = section.end_s - section.begin_s;
      const double chord_length = Distance(begin, end);
      const double curvature_deviation =
          Norm(arc.gravity_mps2) * duration * duration / 8.0;
      if (!std::isfinite(chord_length) ||
          !std::isfinite(curvature_deviation)) {
        return false;
      }
      if (std::max(chord_length, curvature_deviation) > spatial_tolerance) {
        pending.push_back(TimeSection{.begin_s = midpoint,
                                      .end_s = section.end_s});
        pending.push_back(TimeSection{.begin_s = section.begin_s,
                                      .end_s = midpoint});
        continue;
      }
      sections.push_back(section);
    }
    std::ranges::sort(sections, {}, &TimeSection::begin_s);

    for (const TimeSection& section : sections) {
      if (request_.control.canceled() || request_.control.expired()) {
        return false;
      }
      const Vec3 section_begin =
          EvaluateBallisticState(arc, section.begin_s).position_m;
      const Vec3 section_end =
          EvaluateBallisticState(arc, section.end_s).position_m;
      const Vec3 section_midpoint{
          .x = std::midpoint(section_begin.x, section_end.x),
          .y = std::midpoint(section_begin.y, section_end.y),
          .z = 0.0,
      };
      const double half_planar_chord =
          0.5 * std::hypot(section_end.x - section_begin.x,
                           section_end.y - section_begin.y);
      const double broad_radius_squared =
          (radius + half_planar_chord + kTolerance) *
          (radius + half_planar_chord + kTolerance);
      const auto [minimum_x, maximum_x] =
          CoordinateRange(arc, 0U, section.begin_s, section.end_s);
      const auto [minimum_y, maximum_y] =
          CoordinateRange(arc, 1U, section.begin_s, section.end_s);
      const auto [minimum_z, maximum_z] =
          CoordinateRange(arc, 2U, section.begin_s, section.end_s);
      if (!std::isfinite(minimum_x) || !std::isfinite(maximum_x) ||
          !std::isfinite(minimum_y) || !std::isfinite(maximum_y) ||
          !std::isfinite(minimum_z) || !std::isfinite(maximum_z)) {
        return false;
      }
      if (straight_corridor_maximum_elevation.has_value() &&
          minimum_z - radius > *straight_corridor_maximum_elevation) {
        continue;
      }
      const auto cell_index = [&](const double coordinate,
                                  const double origin) {
        return static_cast<long long>(
            std::floor((coordinate - origin) / map_.resolution_m()));
      };
      const long long x0 = cell_index(minimum_x - radius, map_.origin_m().x);
      const long long x1 = cell_index(maximum_x + radius, map_.origin_m().x);
      const long long y0 = cell_index(minimum_y - radius, map_.origin_m().y);
      const long long y1 = cell_index(maximum_y + radius, map_.origin_m().y);
      if (x0 < 0 || y0 < 0 || x1 >= static_cast<long long>(map_.width()) ||
          y1 >= static_cast<long long>(map_.height())) {
        return false;
      }
      for (long long y = y0; y <= y1; ++y) {
        for (long long x = x0; x <= x1; ++x) {
          const shared::GridCell cell{
              .x = static_cast<std::int32_t>(x),
              .y = static_cast<std::int32_t>(y),
          };
          const double cell_x0 = map_.origin_m().x +
                                 static_cast<double>(cell.x) *
                                     map_.resolution_m();
          const double cell_y0 = map_.origin_m().y +
                                 static_cast<double>(cell.y) *
                                     map_.resolution_m();
          if (PointRectangleDistanceSquared(
                  section_midpoint, cell_x0,
                  cell_x0 + map_.resolution_m(), cell_y0,
                  cell_y0 + map_.resolution_m()) > broad_radius_squared) {
            continue;
          }
          const double planar_distance_squared =
              SegmentRectangleDistanceSquared(
                  section_begin, section_end, cell_x0,
                  cell_x0 + map_.resolution_m(), cell_y0,
                  cell_y0 + map_.resolution_m());
          if (planar_distance_squared > tolerated_radius_squared) {
            continue;
          }
          const std::size_t index = map_.Index(cell);
          const float occupied = occupancy[index];
          if (!std::isfinite(occupied) || occupied < 0.0F ||
              occupied > 1.0F || !std::isfinite(elevation[index])) {
            return false;
          }
          const double vertical_radius = std::sqrt(
              std::max(0.0, radius_squared - planar_distance_squared));
          const double lower_tube_z = minimum_z - vertical_radius;
          if (occupied < 0.5F) {
            // Free terrain is exempt only during the two connected contact
            // phases needed for the tube to leave and re-enter its endpoint
            // plane. A higher cell is never part of endpoint contact, and an
            // arc that never clears a full radius receives no exemption.
            const bool launch_contact =
                clears_launch_contact && section.begin_s <= peak_time &&
                minimum_z <= arc.launch_position_m.z + radius + kTolerance &&
                static_cast<double>(elevation[index]) <=
                    arc.launch_position_m.z + kTolerance;
            const bool landing_contact =
                clears_landing_contact && section.end_s >= peak_time &&
                minimum_z <= arc.landing_position_m.z + radius + kTolerance &&
                static_cast<double>(elevation[index]) <=
                    arc.landing_position_m.z + kTolerance;
            if (lower_tube_z + kTolerance <
                    static_cast<double>(elevation[index]) &&
                !launch_contact && !landing_contact) {
              return false;
            }
            continue;
          }
          if (height_cached_[index] == 0U) {
            height_cache_[index] = request_.estimate_height(map_, cell);
            height_cached_[index] = 1U;
          }
          const shared::ObstacleHeight& obstacle = height_cache_[index];
          if (!obstacle.flyover_allowed ||
              !std::isfinite(obstacle.height_m) ||
              obstacle.height_m <= 0.0) {
            return false;
          }
          const double local_ground =
              static_cast<double>(elevation[index]) - obstacle.height_m;
          const double required_clearance = obstacle.height_m;
          if (!(lower_tube_z - local_ground > required_clearance)) {
            return false;
          }
        }
      }
    }
    return true;
  }

  [[nodiscard]] EdgeEvaluation EvaluateEdge(
      const Vec3 source, const Vec3 target) {
    const double distance = std::hypot(target.x - source.x,
                                       target.y - source.y);
    if (!std::isfinite(distance) || distance <= kTolerance ||
        distance > capability_.reference_horizontal_range_m + kTolerance ||
        !LandingAllowed(target)) {
      return {};
    }
    const MinimumSingleHopEnvelopeResult envelope =
        EvaluateMinimumSingleHopEnvelope(
            source, target, capability_.gravity_mps2, map_.resolution_m(),
            capability_);
    if (!envelope.ok() || !ValidateFlightTube(envelope.evidence->arc)) {
      return {};
    }
    const double cost =
        distance + envelope.evidence->arc.flight_time_s +
        0.1 * envelope.evidence->envelope.required_delta_v_mps;
    if (!std::isfinite(cost) || cost <= 0.0) {
      return {};
    }
    return EdgeEvaluation{
        .valid = true,
        .hop = *envelope.evidence,
        .cost = std::max(kMinimumEdgeCost, cost),
    };
  }

  [[nodiscard]] bool AppendEdge(
      const std::size_t source_state, const std::size_t target_state,
      const Vec3 source, const Vec3 target,
      std::vector<shared::GraphEdge>& edges) {
    const EdgeKey key{.source = source_state, .target = target_state};
    const EdgeEvaluation& evaluation = edge_cache_.GetOrEvaluate(
        key, [&] { return EvaluateEdge(source, target); });
    if (!evaluation.valid) {
      return false;
    }
    accepted_edges_[key] = &evaluation;
    edges.push_back(shared::GraphEdge{
        .target_state = target_state,
        .cost = evaluation.cost,
        .stable_index = StableEdgeIndex(key),
    });
    return true;
  }

  const HopperPlanRequest& request_;
  const shared::LocalTerrainProjection& terrain_;
  const HopperCapability& capability_;
  const shared::MapSnapshot& map_;
  PointGoal goal_;
  Vec3 exact_goal_;
  std::size_t goal_state_{};
  std::vector<shared::ObstacleHeight> height_cache_;
  std::vector<std::uint8_t> height_cached_;
  shared::EdgeValidationCache<EdgeKey, EdgeEvaluation, EdgeKeyHash>
      edge_cache_;
  std::unordered_map<EdgeKey, const EdgeEvaluation*, EdgeKeyHash>
      accepted_edges_;
  std::vector<LandingSite> landing_sites_;
  bool terrain_index_ready_{};
};

}  // namespace

HopperPlanResult PlanHopper(const HopperPlanRequest& request) try {
  if (request.control.canceled()) {
    return Failure(LocalPlanStatus::kCanceled, "REQUEST_CANCELED");
  }
  if (request.control.expired()) {
    if (request.control.canceled()) {
      return Failure(LocalPlanStatus::kCanceled, "REQUEST_CANCELED");
    }
    return Failure(LocalPlanStatus::kTimedOut, "TIMEOUT");
  }
  const auto* goal = std::get_if<PointGoal>(&request.goal_odom.target);
  const auto start_yaw = YawFromQuaternion(request.start.pose.orientation);
  if (request.terrain == nullptr || request.capability == nullptr ||
      !ValidTerrain(*request.terrain) ||
      !ValidCapability(*request.capability) || goal == nullptr ||
      !DistanceIndexable(request.capability->reference_horizontal_range_m,
                         *request.terrain->map) ||
      !DistanceIndexable(request.capability->flight_collision_radius_m +
                             request.capability->flight_map_margin_m,
                         *request.terrain->map) ||
      !Finite(request.start.pose.position_m) ||
      !start_yaw.has_value() || !FiniteXY(goal->position_m) ||
      !std::isfinite(goal->tolerance_m) || goal->tolerance_m < 0.0 ||
      (request.goal_odom.yaw_rad.has_value() &&
       !std::isfinite(*request.goal_odom.yaw_rad)) ||
      !std::isfinite(request.goal_odom.yaw_tolerance_rad) ||
      request.goal_odom.yaw_tolerance_rad < 0.0 ||
      request.search.epsilon_schedule !=
          std::array<double, 4>{2.5, 2.0, 1.5, 1.0}) {
    return Failure(LocalPlanStatus::kInvalidInput, "HOPPER_INPUT_INVALID");
  }
  if (!GoalYawSatisfied(*start_yaw, request.goal_odom)) {
    return Failure(LocalPlanStatus::kNoPath,
                   "HOPPER_GOAL_YAW_UNREACHABLE");
  }
  const auto goal_cell = request.terrain->map->PositionToCell(
      Vec2{.x = goal->position_m.x, .y = goal->position_m.y});
  if (!goal_cell.has_value()) {
    return Failure(LocalPlanStatus::kNoPath,
                   "HOPPER_GOAL_OUTSIDE_LOCAL_MAP");
  }
  const Vec3 goal_on_elevation{
      .x = goal->position_m.x,
      .y = goal->position_m.y,
      .z = request.terrain->map->CellCenter(*goal_cell).z,
  };
  HopperSearchGraph graph{request, *goal, goal_on_elevation};
  if (!graph.ValidateStart()) {
    return Failure(LocalPlanStatus::kNoPath, "HOPPER_START_INFEASIBLE");
  }

  const shared::anytime::AraStarResult search =
      shared::anytime::SearchAnytimeAraStar(
          shared::anytime::AraStarProblem{
              .state_count = graph.state_count(),
              .start_state = 0U,
              .expand = [&](const std::size_t state,
                            std::vector<shared::GraphEdge>& edges) {
                graph.Expand(state, edges);
              },
              .heuristic = [&](const std::size_t state) {
                return graph.Heuristic(state);
              },
              .is_goal = [&](const std::size_t state) {
                return graph.IsGoalState(state);
              },
              .config = request.search,
              .control = request.control,
          });
  const LocalPlanMetrics metrics{
      .expanded_states = search.expanded_states,
      .edge_validation_evaluations = graph.validation_count(),
  };
  switch (search.status) {
    case shared::anytime::AraStarStatus::kCanceled:
      return Failure(LocalPlanStatus::kCanceled, "REQUEST_CANCELED", metrics);
    case shared::anytime::AraStarStatus::kTimedOut:
      return Failure(LocalPlanStatus::kTimedOut, "TIMEOUT", metrics);
    case shared::anytime::AraStarStatus::kNoPath:
      return Failure(LocalPlanStatus::kNoPath, "NO_PATH", metrics);
    case shared::anytime::AraStarStatus::kResourceExhausted:
      return Failure(LocalPlanStatus::kPlannerError,
                     "HOPPER_SEARCH_RESOURCE_EXHAUSTED", metrics);
    case shared::anytime::AraStarStatus::kInvalidProblem:
      return Failure(LocalPlanStatus::kPlannerError,
                     search.reason_code.empty() ? "HOPPER_SEARCH_INVALID"
                                                : search.reason_code,
                     metrics);
    case shared::anytime::AraStarStatus::kSolved:
      break;
  }
  if (search.candidates.empty()) {
    return Failure(LocalPlanStatus::kPlannerError,
                   "HOPPER_INCUMBENT_MISSING", metrics);
  }
  const shared::SearchCandidate& candidate = search.candidates.back();
  const auto control_failure = [&]() -> std::optional<HopperPlanResult> {
    const auto stopped = shared::StopReason(request.control);
    if (!stopped.has_value()) {
      return std::nullopt;
    }
    return Failure(
        *stopped == "REQUEST_CANCELED" ? LocalPlanStatus::kCanceled
                                        : LocalPlanStatus::kTimedOut,
        std::string{*stopped}, metrics);
  };
  if (const auto stopped = control_failure(); stopped.has_value()) {
    return *stopped;
  }
  std::vector<HopSegment> hops;
  hops.reserve(candidate.states.size() - 1U);
  if (const auto stopped = control_failure(); stopped.has_value()) {
    return *stopped;
  }
  for (std::size_t index = 1U; index < candidate.states.size(); ++index) {
    if (const auto stopped = control_failure(); stopped.has_value()) {
      return *stopped;
    }
    const std::size_t source_state = candidate.states[index - 1U];
    const std::size_t target_state = candidate.states[index];
    const EdgeEvaluation* edge = graph.Edge(source_state, target_state);
    if (edge == nullptr || !edge->valid) {
      return Failure(LocalPlanStatus::kPlannerError,
                     "HOPPER_EDGE_CACHE_MISSING", metrics);
    }
    const Vec3 source = graph.PositionForState(source_state);
    const Vec3 target = graph.PositionForState(target_state);
    hops.push_back(HopSegment{
        .segment_id = "hop-" + std::to_string(index - 1U),
        .launch_pose = Pose3{
            .position_m = source,
            .orientation = request.start.pose.orientation,
        },
        .flight_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::duration<double>{edge->hop.arc.flight_time_s}),
        .launch_velocity_mps = edge->hop.arc.launch_velocity_mps,
        .flight_tube_radius_m =
            request.capability->flight_collision_radius_m +
            request.capability->flight_map_margin_m,
        .nominal_landing_point_m = target,
        .required_delta_v_mps = edge->hop.envelope.required_delta_v_mps,
        .available_delta_v_mps = edge->hop.envelope.available_delta_v_mps,
    });
  }
  const bool stationary = candidate.states.size() == 1U &&
                          candidate.states.front() == 0U &&
                          graph.IsGoalState(0U) &&
                          candidate.cost == 0.0;
  if ((!stationary && hops.empty()) || !std::isfinite(candidate.cost) ||
      candidate.cost < 0.0 || (stationary && !hops.empty()) ||
      (!stationary && candidate.cost <= 0.0)) {
    return Failure(LocalPlanStatus::kPlannerError,
                   "HOPPER_INCUMBENT_INVALID", metrics);
  }
  if (const auto stopped = control_failure(); stopped.has_value()) {
    return *stopped;
  }
  return HopperPlanResult{
      .status = LocalPlanStatus::kSolved,
      .reason_code = stationary ? "HOPPER_STATIONARY_GOAL_SATISFIED"
                                : "HOPPER_PLAN_AVAILABLE",
      .hops = std::move(hops),
      .cost = candidate.cost,
      .metrics = metrics,
  };
} catch (const std::bad_alloc&) {
  return Failure(LocalPlanStatus::kPlannerError,
                 "HOPPER_SEARCH_RESOURCE_EXHAUSTED");
} catch (...) {
  return Failure(LocalPlanStatus::kPlannerError, "PLANNER_ERROR");
}

}  // namespace lunar::pure_planning::hopper

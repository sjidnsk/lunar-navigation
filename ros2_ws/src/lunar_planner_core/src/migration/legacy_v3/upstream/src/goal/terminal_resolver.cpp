#include "lunar_path_planner/v3/goal/terminal_resolver.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "lunar_path_planner/v3/crypto/sha256.hpp"

namespace lunar::planning::v3 {
namespace {

constexpr double kGeometryTolerance = 1.0e-12;
constexpr std::int32_t kNeighborDx[] = {-1, 1, 0, 0};
constexpr std::int32_t kNeighborDy[] = {0, 0, 1, -1};

[[nodiscard]] Error Invalid(std::string field_path,
                            std::string message) {
  return {
      .code = ErrorCode::kInvalidArgument,
      .field_path = std::move(field_path),
      .message = std::move(message),
  };
}

[[nodiscard]] bool IsFinite(const Vec2& value) noexcept {
  return std::isfinite(value.x) && std::isfinite(value.y);
}

[[nodiscard]] bool IsFinite(const Vec3& value) noexcept {
  return std::isfinite(value.x) && std::isfinite(value.y) &&
         std::isfinite(value.z);
}

[[nodiscard]] double Dot(const Vec3& lhs, const Vec3& rhs) noexcept {
  return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
}

[[nodiscard]] Vec3 Cross(const Vec3& lhs, const Vec3& rhs) noexcept {
  return {
      .x = lhs.y * rhs.z - lhs.z * rhs.y,
      .y = lhs.z * rhs.x - lhs.x * rhs.z,
      .z = lhs.x * rhs.y - lhs.y * rhs.x,
  };
}

[[nodiscard]] double Norm(const Vec3& value) noexcept {
  return std::hypot(value.x, value.y, value.z);
}

[[nodiscard]] std::size_t CellIndex(const GridGeometry& geometry,
                                    Cell cell) noexcept {
  return static_cast<std::size_t>(cell.y) * geometry.width +
         static_cast<std::size_t>(cell.x);
}

[[nodiscard]] bool SameGeometry(const GridGeometry& lhs,
                                const GridGeometry& rhs) noexcept {
  return lhs.width == rhs.width && lhs.height == rhs.height &&
         lhs.resolution_m == rhs.resolution_m &&
         lhs.origin_m.x == rhs.origin_m.x &&
         lhs.origin_m.y == rhs.origin_m.y &&
         lhs.frame_id == rhs.frame_id;
}

[[nodiscard]] Vec3 SurfacePoint(const SafeProjection& projection,
                                Cell cell) {
  const std::size_t index = CellIndex(projection.geometry, cell);
  return {
      .x = projection.geometry.origin_m.x +
           (static_cast<double>(cell.x) + 0.5) *
               projection.geometry.resolution_m,
      .y = projection.geometry.origin_m.y +
           (static_cast<double>(cell.y) + 0.5) *
               projection.geometry.resolution_m,
      .z = static_cast<double>(
          projection.ElevationMeters()[index]),
  };
}

[[nodiscard]] bool ValidConvexCcwPolygon(
    const ConvexPolygonUv& polygon) noexcept {
  if (polygon.vertices_uv.size() < 3U) {
    return false;
  }
  for (std::size_t index = 0U;
       index < polygon.vertices_uv.size(); ++index) {
    const Vec2& first = polygon.vertices_uv[index];
    const Vec2& second =
        polygon.vertices_uv[
            (index + 1U) % polygon.vertices_uv.size()];
    const Vec2& third =
        polygon.vertices_uv[
            (index + 2U) % polygon.vertices_uv.size()];
    if (!IsFinite(first) || !IsFinite(second) ||
        !IsFinite(third)) {
      return false;
    }
    const double cross =
        (second.x - first.x) * (third.y - second.y) -
        (second.y - first.y) * (third.x - second.x);
    if (!std::isfinite(cross) ||
        !(cross > kGeometryTolerance)) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] std::optional<Error> ValidateGoal(
    const GoalRegion& goal) {
  if (goal.goal_id.empty()) {
    return Invalid("terminal_resolution.goal_region.goal_id",
                   "goal ID must not be empty");
  }
  const auto target_error = std::visit(
      [](const auto& target) -> std::optional<Error> {
        using Goal = std::decay_t<decltype(target)>;
        if constexpr (std::is_same_v<Goal, PointGoal>) {
          if (!IsFinite(target.position_m) ||
              !std::isfinite(target.position_tolerance_m) ||
              target.position_tolerance_m < 0.0) {
            return Invalid(
                "terminal_resolution.goal_region.target",
                "point goal and tolerance must be finite and "
                "non-negative");
          }
        } else {
          const LandingPlane& plane = target.plane;
          if (!IsFinite(plane.origin_m) ||
              !IsFinite(plane.normal) ||
              !IsFinite(plane.basis_u) ||
              !IsFinite(plane.basis_v) ||
              !std::isfinite(plane.residual_bound_m) ||
              plane.residual_bound_m < 0.0 ||
              !std::isfinite(target.normal_tolerance_m) ||
              target.normal_tolerance_m < 0.0) {
            return Invalid(
                "terminal_resolution.goal_region.target",
                "planar goal values and tolerances must be finite "
                "and non-negative");
          }
          const double normal_norm = Norm(plane.normal);
          const double basis_u_norm = Norm(plane.basis_u);
          const double basis_v_norm = Norm(plane.basis_v);
          if (std::abs(normal_norm - 1.0) >
                  kGeometryTolerance ||
              std::abs(basis_u_norm - 1.0) >
                  kGeometryTolerance ||
              std::abs(basis_v_norm - 1.0) >
                  kGeometryTolerance ||
              std::abs(Dot(plane.normal, plane.basis_u)) >
                  kGeometryTolerance ||
              std::abs(Dot(plane.normal, plane.basis_v)) >
                  kGeometryTolerance ||
              std::abs(Dot(plane.basis_u, plane.basis_v)) >
                  kGeometryTolerance ||
              Dot(Cross(plane.basis_u, plane.basis_v),
                  plane.normal) <
                  1.0 - kGeometryTolerance) {
            return Invalid(
                "terminal_resolution.goal_region.target.plane",
                "landing-plane basis must be finite, orthonormal, "
                "and right handed");
          }
          if (!ValidConvexCcwPolygon(target.polygon)) {
            return Invalid(
                "terminal_resolution.goal_region.target.polygon",
                "planar goal polygon must be strictly convex CCW");
          }
        }
        return std::nullopt;
      },
      goal.target);
  if (target_error.has_value()) {
    return target_error;
  }

  if (goal.optional_yaw_interval.has_value()) {
    const CircularYawInterval& yaw =
        *goal.optional_yaw_interval;
    if (!std::isfinite(yaw.start_rad) ||
        yaw.start_rad < -std::numbers::pi_v<double> ||
        yaw.start_rad >= std::numbers::pi_v<double> ||
        !std::isfinite(yaw.span_rad) || yaw.span_rad < 0.0 ||
        yaw.span_rad >
            2.0 * std::numbers::pi_v<double> ||
        !yaw.closed) {
      return Invalid(
          "terminal_resolution.goal_region.optional_yaw_interval",
          "yaw interval must use the closed canonical CCW form");
    }
  }
  if (goal.mission_direction_hint.has_value() &&
      !IsFinite(*goal.mission_direction_hint)) {
    return Invalid(
        "terminal_resolution.goal_region.mission_direction_hint",
        "mission direction hint must be finite");
  }
  std::set<std::string> metadata_keys;
  for (const MetadataEntry& entry : goal.task_metadata) {
    if (entry.key.empty() ||
        !metadata_keys.insert(entry.key).second) {
      return Invalid(
          "terminal_resolution.goal_region.task_metadata",
          "goal metadata keys must be non-empty and unique");
    }
    if (std::holds_alternative<double>(entry.value) &&
        !std::isfinite(std::get<double>(entry.value))) {
      return Invalid(
          "terminal_resolution.goal_region.task_metadata",
          "numeric goal metadata must be finite");
    }
  }
  return std::nullopt;
}

[[nodiscard]] Result<double> HorizontalPositionErrorRadius(
    const SafetyCapabilityProfile& capability) {
  const DeterministicVectorSet3& position_error =
      std::visit(
          [](const auto& concrete)
              -> const DeterministicVectorSet3& {
            return concrete.certified_state_error_bounds
                .position_bound_m;
          },
          capability.content);
  return std::visit(
      [](const auto& bound) -> Result<double> {
        using Bound = std::decay_t<decltype(bound)>;
        if constexpr (std::is_same_v<Bound, AxisAlignedBox3>) {
          if (!IsFinite(bound.center) ||
              !IsFinite(bound.half_extent) ||
              bound.half_extent.x < 0.0 ||
              bound.half_extent.y < 0.0 ||
              bound.half_extent.z < 0.0) {
            return Invalid(
                "terminal_resolution.capability."
                "certified_state_error_bounds.position_bound_m",
                "AABB position-error bound must be finite with "
                "non-negative half extents");
          }
          const double radius = std::hypot(
              std::abs(bound.center.x) + bound.half_extent.x,
              std::abs(bound.center.y) + bound.half_extent.y);
          if (!std::isfinite(radius)) {
            return Invalid(
                "terminal_resolution.capability."
                "certified_state_error_bounds.position_bound_m",
                "horizontal AABB error radius must be finite");
          }
          return radius;
        } else {
          if (!IsFinite(bound.center) ||
              !std::isfinite(bound.radius) ||
              bound.radius < 0.0) {
            return Invalid(
                "terminal_resolution.capability."
                "certified_state_error_bounds.position_bound_m",
                "ball position-error bound must be finite with "
                "non-negative radius");
          }
          const double radius =
              std::hypot(bound.center.x, bound.center.y) +
              bound.radius;
          if (!std::isfinite(radius)) {
            return Invalid(
                "terminal_resolution.capability."
                "certified_state_error_bounds.position_bound_m",
                "horizontal ball error radius must be finite");
          }
          return radius;
        }
      },
      position_error);
}

[[nodiscard]] std::vector<std::int32_t>
CanonicalConnectedComponents(const SafeProjection& projection) {
  const std::size_t count = projection.geometry.CellCount();
  std::vector<std::int32_t> components(count, -1);
  std::vector<Cell> queue;
  queue.reserve(count);
  std::int32_t component = 0;
  for (std::size_t y = 0U; y < projection.geometry.height; ++y) {
    for (std::size_t x = 0U; x < projection.geometry.width; ++x) {
      const Cell seed{
          .x = static_cast<std::int32_t>(x),
          .y = static_cast<std::int32_t>(y),
      };
      const std::size_t seed_index =
          CellIndex(projection.geometry, seed);
      if (projection.hard_feasible_mask[seed_index] == 0U ||
          components[seed_index] >= 0) {
        continue;
      }
      queue.clear();
      queue.push_back(seed);
      components[seed_index] = component;
      std::size_t head = 0U;
      while (head < queue.size()) {
        const Cell current = queue[head++];
        for (std::size_t neighbor = 0U; neighbor < 4U;
             ++neighbor) {
          const Cell next{
              .x = current.x + kNeighborDx[neighbor],
              .y = current.y + kNeighborDy[neighbor],
          };
          if (!projection.InBounds(next)) {
            continue;
          }
          const std::size_t next_index =
              CellIndex(projection.geometry, next);
          if (projection.hard_feasible_mask[next_index] == 0U ||
              components[next_index] >= 0) {
            continue;
          }
          components[next_index] = component;
          queue.push_back(next);
        }
      }
      ++component;
    }
  }
  return components;
}

[[nodiscard]] std::optional<Error> ValidateProjection(
    const TerminalResolutionRequest& request,
    double& required_frontier_clearance_m) {
  const SafeProjection& projection = request.projection;
  if (!projection.source_map()) {
    return Invalid(
        "terminal_resolution.projection.source_map",
        "safe projection must retain its immutable map snapshot");
  }
  const GridGeometry& geometry = projection.geometry;
  const std::size_t count = geometry.CellCount();
  if (count == 0U ||
      count > static_cast<std::size_t>(
                  std::numeric_limits<std::int32_t>::max()) ||
      geometry.width >
          static_cast<std::size_t>(
              std::numeric_limits<std::int32_t>::max()) ||
      geometry.height >
          static_cast<std::size_t>(
              std::numeric_limits<std::int32_t>::max()) ||
      !std::isfinite(geometry.resolution_m) ||
      geometry.resolution_m <= 0.0 ||
      !IsFinite(geometry.origin_m) ||
      geometry.frame_id.empty()) {
    return Invalid("terminal_resolution.projection.geometry",
                   "projection geometry must be finite and bounded");
  }
  if (!SameGeometry(geometry,
                    projection.source_map()->geometry())) {
    return Invalid(
        "terminal_resolution.projection.geometry",
        "projection geometry must exactly match the source map");
  }
  if (projection.ElevationMeters().size() != count) {
    return Invalid(
        "terminal_resolution.projection.elevation_m",
        "source-map elevation shape must match projection geometry");
  }

  const auto require_shape =
      [count](std::size_t size, std::string field)
      -> std::optional<Error> {
    if (size != count) {
      return Invalid(std::move(field),
                     "projection layer shape must match geometry");
    }
    return std::nullopt;
  };
  if (auto error = require_shape(
          projection.known_mask.size(),
          "terminal_resolution.projection.known_mask");
      error.has_value()) {
    return error;
  }
  if (auto error = require_shape(
          projection.hard_feasible_mask.size(),
          "terminal_resolution.projection.hard_feasible_mask");
      error.has_value()) {
    return error;
  }
  if (auto error = require_shape(
          projection.esdf_clearance_m.size(),
          "terminal_resolution.projection.esdf_clearance_m");
      error.has_value()) {
    return error;
  }
  if (auto error = require_shape(
          projection.connected_component.size(),
          "terminal_resolution.projection.connected_component");
      error.has_value()) {
    return error;
  }
  if (auto error = require_shape(
          projection.safe_stop_candidate_mask.size(),
          "terminal_resolution.projection.safe_stop_candidate_mask");
      error.has_value()) {
    return error;
  }

  const auto map_known = projection.source_map()->KnownMask();
  if (map_known.size() != count ||
      !std::equal(map_known.begin(), map_known.end(),
                  projection.known_mask.begin())) {
    return Invalid(
        "terminal_resolution.projection.known_mask",
        "projection known mask must exactly match the source map");
  }
  for (std::size_t index = 0U; index < count; ++index) {
    const std::uint8_t known = projection.known_mask[index];
    const std::uint8_t feasible =
        projection.hard_feasible_mask[index];
    const std::uint8_t stop =
        projection.safe_stop_candidate_mask[index];
    const float clearance = projection.esdf_clearance_m[index];
    if (known > 1U || feasible > 1U || stop > 1U ||
        (feasible != 0U && known == 0U) ||
        (stop != 0U && feasible == 0U) ||
        !std::isfinite(clearance) || clearance < 0.0F) {
      return Invalid(
          "terminal_resolution.projection",
          "projection masks and clearance must preserve hard-safety "
          "invariants");
    }
  }
  if (CanonicalConnectedComponents(projection) !=
      projection.connected_component) {
    return Invalid(
        "terminal_resolution.projection.connected_component",
        "connected-component labels must use canonical row-major "
        "four-connectivity");
  }

  auto limits = ResolveProjectionLimits(request.capability);
  if (!IsOk(limits)) {
    const Error& cause = std::get<Error>(limits);
    return Invalid("terminal_resolution.capability",
                   "capability cannot define terminal clearance: " +
                       cause.message);
  }
  const SafetyProjectionLimits& resolved_limits =
      std::get<SafetyProjectionLimits>(limits);
  if (resolved_limits.platform_type !=
          projection.platform_type() ||
      request.capability.content_ref !=
          projection.capability_ref()) {
    return Invalid(
        "terminal_resolution.capability",
        "capability must exactly match the safe projection");
  }
  auto error_radius =
      HorizontalPositionErrorRadius(request.capability);
  if (!IsOk(error_radius)) {
    return std::get<Error>(std::move(error_radius));
  }
  required_frontier_clearance_m =
      resolved_limits.minimum_clearance_m +
      std::get<double>(error_radius);
  if (!std::isfinite(required_frontier_clearance_m) ||
      required_frontier_clearance_m < 0.0) {
    return Invalid(
        "terminal_resolution.capability",
        "combined footprint and position-error clearance must be "
        "finite and non-negative");
  }
  if (request.algorithm_config.ara_star.resource_caps
          .maximum_generated_candidates == 0U) {
    return Invalid(
        "terminal_resolution.algorithm_config.ara_star."
        "resource_caps.maximum_generated_candidates",
        "at least one generated terminal candidate is required");
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<Error> ValidateRequest(
    const TerminalResolutionRequest& request,
    double& required_frontier_clearance_m) {
  if (auto error = ValidateGoal(request.goal_region);
      error.has_value()) {
    return error;
  }
  return ValidateProjection(request,
                            required_frontier_clearance_m);
}

[[nodiscard]] Vec3 GoalCentroid(const GoalRegion& goal) {
  return std::visit(
      [](const auto& target) -> Vec3 {
        using Goal = std::decay_t<decltype(target)>;
        if constexpr (std::is_same_v<Goal, PointGoal>) {
          return target.position_m;
        } else {
          const auto& vertices = target.polygon.vertices_uv;
          double twice_area = 0.0;
          double weighted_u = 0.0;
          double weighted_v = 0.0;
          for (std::size_t index = 0U;
               index < vertices.size(); ++index) {
            const Vec2& current = vertices[index];
            const Vec2& next =
                vertices[(index + 1U) % vertices.size()];
            const double cross =
                current.x * next.y - next.x * current.y;
            twice_area += cross;
            weighted_u += (current.x + next.x) * cross;
            weighted_v += (current.y + next.y) * cross;
          }
          const double centroid_u =
              weighted_u / (3.0 * twice_area);
          const double centroid_v =
              weighted_v / (3.0 * twice_area);
          return {
              .x = target.plane.origin_m.x +
                   centroid_u * target.plane.basis_u.x +
                   centroid_v * target.plane.basis_v.x,
              .y = target.plane.origin_m.y +
                   centroid_u * target.plane.basis_u.y +
                   centroid_v * target.plane.basis_v.y,
              .z = target.plane.origin_m.z +
                   centroid_u * target.plane.basis_u.z +
                   centroid_v * target.plane.basis_v.z,
          };
        }
      },
      goal.target);
}

[[nodiscard]] bool CellInsideGoal(const Vec3& surface,
                                  const GoalRegion& goal) {
  return std::visit(
      [&](const auto& target) {
        using Goal = std::decay_t<decltype(target)>;
        if constexpr (std::is_same_v<Goal, PointGoal>) {
          return std::hypot(
                     surface.x - target.position_m.x,
                     surface.y - target.position_m.y,
                     surface.z - target.position_m.z) <=
                 target.position_tolerance_m +
                     kGeometryTolerance;
        } else {
          const Vec3 delta{
              .x = surface.x - target.plane.origin_m.x,
              .y = surface.y - target.plane.origin_m.y,
              .z = surface.z - target.plane.origin_m.z,
          };
          if (std::abs(Dot(delta, target.plane.normal)) >
              target.normal_tolerance_m +
                  kGeometryTolerance) {
            return false;
          }
          const Vec2 point{
              .x = Dot(delta, target.plane.basis_u),
              .y = Dot(delta, target.plane.basis_v),
          };
          const auto& vertices = target.polygon.vertices_uv;
          for (std::size_t index = 0U; index < vertices.size();
               ++index) {
            const Vec2& start = vertices[index];
            const Vec2& end =
                vertices[(index + 1U) % vertices.size()];
            const double halfspace =
                (end.x - start.x) * (point.y - start.y) -
                (end.y - start.y) * (point.x - start.x);
            if (halfspace < -kGeometryTolerance) {
              return false;
            }
          }
          return true;
        }
      },
      goal.target);
}

struct RasterizedGoal final {
  std::vector<Cell> in_bounds;
  std::vector<Cell> known;
  std::vector<Cell> hard_feasible;
};

[[nodiscard]] RasterizedGoal RasterizeGoal(
    const SafeProjection& projection, const GoalRegion& goal) {
  RasterizedGoal cells;
  for (std::size_t y = 0U; y < projection.geometry.height; ++y) {
    for (std::size_t x = 0U; x < projection.geometry.width; ++x) {
      const Cell cell{
          .x = static_cast<std::int32_t>(x),
          .y = static_cast<std::int32_t>(y),
      };
      if (!CellInsideGoal(SurfacePoint(projection, cell), goal)) {
        continue;
      }
      cells.in_bounds.push_back(cell);
      if (projection.Known(cell)) {
        cells.known.push_back(cell);
      }
      if (projection.HardFeasible(cell)) {
        cells.hard_feasible.push_back(cell);
      }
    }
  }
  return cells;
}

[[nodiscard]] std::vector<std::uint8_t> ReachableFromStart(
    const SafeProjection& projection, Cell start) {
  const std::size_t count = projection.geometry.CellCount();
  std::vector<std::uint8_t> reachable(count, 0U);
  std::vector<Cell> queue;
  queue.reserve(count);
  reachable[CellIndex(projection.geometry, start)] = 1U;
  queue.push_back(start);
  std::size_t head = 0U;
  while (head < queue.size()) {
    const Cell current = queue[head++];
    for (std::size_t neighbor = 0U; neighbor < 4U;
         ++neighbor) {
      const Cell next{
          .x = current.x + kNeighborDx[neighbor],
          .y = current.y + kNeighborDy[neighbor],
      };
      if (!projection.HardFeasible(next)) {
        continue;
      }
      const std::size_t next_index =
          CellIndex(projection.geometry, next);
      if (reachable[next_index] != 0U) {
        continue;
      }
      reachable[next_index] = 1U;
      queue.push_back(next);
    }
  }
  return reachable;
}

void AppendUint32(std::string& output, std::uint32_t value) {
  for (int shift = 24; shift >= 0; shift -= 8) {
    output.push_back(static_cast<char>(
        (value >> static_cast<unsigned int>(shift)) & 0xffU));
  }
}

void AppendUint64(std::string& output, std::uint64_t value) {
  for (int shift = 56; shift >= 0; shift -= 8) {
    output.push_back(static_cast<char>(
        (value >> static_cast<unsigned int>(shift)) & 0xffU));
  }
}

void AppendString(std::string& output, std::string_view value) {
  AppendUint64(output, static_cast<std::uint64_t>(value.size()));
  output.append(value.data(), value.size());
}

[[nodiscard]] Result<std::string> StableTerminalId(
    const ImmutableMapSnapshot& map, Cell cell) {
  std::string canonical;
  canonical.reserve(256U);
  AppendString(canonical, "lpp-v3-terminal-cell/v1");
  AppendString(canonical, map.snapshot_ref().id);
  AppendUint32(canonical, map.snapshot_ref().revision);
  AppendString(canonical, map.snapshot_ref().content_hash);
  AppendUint32(canonical, map.map_revision());
  AppendString(canonical, map.immutable_data_handle());
  AppendString(canonical, map.LayerManifestHash());
  AppendString(canonical, map.frame_id());
  const ClockStamp source_time = map.source_time();
  AppendString(canonical, source_time.clock_id);
  AppendUint64(
      canonical,
      static_cast<std::uint64_t>(source_time.tick.count()));
  AppendUint32(canonical,
               static_cast<std::uint32_t>(cell.y));
  AppendUint32(canonical,
               static_cast<std::uint32_t>(cell.x));

  auto digest = Sha256Hex(canonical);
  if (!IsOk(digest)) {
    return std::get<Error>(std::move(digest));
  }
  return "terminal-cell/v1/" +
         std::get<Sha256Digest>(std::move(digest));
}

[[nodiscard]] Result<TerminalCandidate> MakeCandidate(
    const SafeProjection& projection, Cell cell,
    std::optional<CircularYawInterval> yaw_interval) {
  auto stable_id =
      StableTerminalId(*projection.source_map(), cell);
  if (!IsOk(stable_id)) {
    return std::get<Error>(std::move(stable_id));
  }
  return TerminalCandidate{
      .stable_id = std::get<std::string>(std::move(stable_id)),
      .cell = cell,
      .position_m = SurfacePoint(projection, cell),
      .yaw_interval = std::move(yaw_interval),
      .requires_zero_speed = true,
      .clearance_m = static_cast<double>(
          projection.ClearanceMeters(cell)),
  };
}

[[nodiscard]] Result<std::vector<TerminalCandidate>>
MakeGoalCandidates(const SafeProjection& projection,
                   const std::vector<Cell>& cells,
                   const GoalRegion& goal) {
  std::vector<TerminalCandidate> candidates;
  candidates.reserve(cells.size());
  for (const Cell cell : cells) {
    auto candidate = MakeCandidate(
        projection, cell, goal.optional_yaw_interval);
    if (!IsOk(candidate)) {
      return std::get<Error>(std::move(candidate));
    }
    candidates.push_back(
        std::get<TerminalCandidate>(std::move(candidate)));
  }
  return candidates;
}

[[nodiscard]] double DistanceToGoal(
    const SafeProjection& projection, Cell cell,
    const Vec3& goal_centroid) {
  const Vec3 point = SurfacePoint(projection, cell);
  return std::hypot(point.x - goal_centroid.x,
                    point.y - goal_centroid.y,
                    point.z - goal_centroid.z);
}

struct RankedCell final {
  Cell cell;
  double distance_to_goal_m{};
  double clearance_m{};
};

[[nodiscard]] auto RankTuple(const RankedCell& ranked) {
  return std::tuple{
      ranked.distance_to_goal_m,
      -ranked.clearance_m,
      ranked.cell.y,
      ranked.cell.x,
  };
}

[[nodiscard]] Result<std::vector<TerminalCandidate>>
ExtractFrontierCandidates(
    const TerminalResolutionRequest& request,
    const std::vector<std::uint8_t>& start_component,
    double required_clearance_m, const Vec3& goal_centroid) {
  const SafeProjection& projection = request.projection;
  const std::size_t count = projection.geometry.CellCount();
  std::vector<std::uint8_t> frontier(count, 0U);
  for (std::size_t y = 0U; y < projection.geometry.height; ++y) {
    for (std::size_t x = 0U; x < projection.geometry.width; ++x) {
      const Cell cell{
          .x = static_cast<std::int32_t>(x),
          .y = static_cast<std::int32_t>(y),
      };
      const std::size_t index =
          CellIndex(projection.geometry, cell);
      if (start_component[index] == 0U ||
          projection.safe_stop_candidate_mask[index] == 0U ||
          static_cast<double>(
              projection.esdf_clearance_m[index]) +
                  kGeometryTolerance <
              required_clearance_m) {
        continue;
      }
      bool adjacent_to_unknown = false;
      for (std::size_t neighbor = 0U; neighbor < 4U;
           ++neighbor) {
        const Cell next{
            .x = cell.x + kNeighborDx[neighbor],
            .y = cell.y + kNeighborDy[neighbor],
        };
        if (projection.InBounds(next) &&
            !projection.Known(next)) {
          adjacent_to_unknown = true;
          break;
        }
      }
      frontier[index] =
          static_cast<std::uint8_t>(adjacent_to_unknown);
    }
  }

  std::vector<std::uint8_t> visited(count, 0U);
  std::vector<Cell> queue;
  queue.reserve(count);
  std::vector<RankedCell> representatives;
  for (std::size_t y = 0U; y < projection.geometry.height; ++y) {
    for (std::size_t x = 0U; x < projection.geometry.width; ++x) {
      const Cell seed{
          .x = static_cast<std::int32_t>(x),
          .y = static_cast<std::int32_t>(y),
      };
      const std::size_t seed_index =
          CellIndex(projection.geometry, seed);
      if (frontier[seed_index] == 0U ||
          visited[seed_index] != 0U) {
        continue;
      }
      queue.clear();
      queue.push_back(seed);
      visited[seed_index] = 1U;
      std::size_t head = 0U;
      std::optional<RankedCell> representative;
      while (head < queue.size()) {
        const Cell current = queue[head++];
        const RankedCell ranked{
            .cell = current,
            .distance_to_goal_m =
                DistanceToGoal(projection, current,
                               goal_centroid),
            .clearance_m = static_cast<double>(
                projection.ClearanceMeters(current)),
        };
        if (!representative.has_value() ||
            RankTuple(ranked) < RankTuple(*representative)) {
          representative = ranked;
        }
        for (std::size_t neighbor = 0U; neighbor < 4U;
             ++neighbor) {
          const Cell next{
              .x = current.x + kNeighborDx[neighbor],
              .y = current.y + kNeighborDy[neighbor],
          };
          if (!projection.InBounds(next)) {
            continue;
          }
          const std::size_t next_index =
              CellIndex(projection.geometry, next);
          if (frontier[next_index] == 0U ||
              visited[next_index] != 0U) {
            continue;
          }
          visited[next_index] = 1U;
          queue.push_back(next);
        }
      }
      if (representative.has_value()) {
        representatives.push_back(*representative);
      }
    }
  }

  std::sort(
      representatives.begin(), representatives.end(),
      [](const RankedCell& lhs, const RankedCell& rhs) {
        return RankTuple(lhs) < RankTuple(rhs);
      });
  const std::size_t cap =
      request.algorithm_config.ara_star.resource_caps
          .maximum_generated_candidates;
  if (representatives.size() > cap) {
    representatives.resize(cap);
  }

  std::vector<TerminalCandidate> candidates;
  candidates.reserve(representatives.size());
  for (const RankedCell& representative : representatives) {
    auto candidate =
        MakeCandidate(projection, representative.cell, std::nullopt);
    if (!IsOk(candidate)) {
      return std::get<Error>(std::move(candidate));
    }
    candidates.push_back(
        std::get<TerminalCandidate>(std::move(candidate)));
  }
  return candidates;
}

struct ClippedIntent final {
  Vec3 endpoint;
  bool clipped{};
};

[[nodiscard]] Vec3 Interpolate(const Vec3& start,
                               const Vec3& end,
                               double parameter) noexcept {
  return {
      .x = start.x + parameter * (end.x - start.x),
      .y = start.y + parameter * (end.y - start.y),
      .z = start.z + parameter * (end.z - start.z),
  };
}

[[nodiscard]] ClippedIntent ClipAtKnownBoundary(
    const SafeProjection& projection, Cell start_cell,
    const Vec3& start, const Vec3& goal) {
  const double delta_x = goal.x - start.x;
  const double delta_y = goal.y - start.y;
  if (std::abs(delta_x) <= kGeometryTolerance &&
      std::abs(delta_y) <= kGeometryTolerance) {
    return {.endpoint = goal, .clipped = false};
  }

  const double infinity =
      std::numeric_limits<double>::infinity();
  std::int32_t step_x = 0;
  std::int32_t step_y = 0;
  double maximum_x = infinity;
  double maximum_y = infinity;
  double delta_parameter_x = infinity;
  double delta_parameter_y = infinity;
  if (delta_x > kGeometryTolerance) {
    step_x = 1;
    const double boundary =
        projection.geometry.origin_m.x +
        (static_cast<double>(start_cell.x) + 1.0) *
            projection.geometry.resolution_m;
    maximum_x = (boundary - start.x) / delta_x;
    delta_parameter_x =
        projection.geometry.resolution_m / delta_x;
  } else if (delta_x < -kGeometryTolerance) {
    step_x = -1;
    const double boundary =
        projection.geometry.origin_m.x +
        static_cast<double>(start_cell.x) *
            projection.geometry.resolution_m;
    maximum_x = (boundary - start.x) / delta_x;
    delta_parameter_x =
        -projection.geometry.resolution_m / delta_x;
  }
  if (delta_y > kGeometryTolerance) {
    step_y = 1;
    const double boundary =
        projection.geometry.origin_m.y +
        (static_cast<double>(start_cell.y) + 1.0) *
            projection.geometry.resolution_m;
    maximum_y = (boundary - start.y) / delta_y;
    delta_parameter_y =
        projection.geometry.resolution_m / delta_y;
  } else if (delta_y < -kGeometryTolerance) {
    step_y = -1;
    const double boundary =
        projection.geometry.origin_m.y +
        static_cast<double>(start_cell.y) *
            projection.geometry.resolution_m;
    maximum_y = (boundary - start.y) / delta_y;
    delta_parameter_y =
        -projection.geometry.resolution_m / delta_y;
  }

  Cell current = start_cell;
  const std::size_t maximum_crossings =
      projection.geometry.width + projection.geometry.height + 2U;
  for (std::size_t crossing = 0U;
       crossing < maximum_crossings; ++crossing) {
    const double parameter = std::min(maximum_x, maximum_y);
    if (!std::isfinite(parameter) ||
        parameter > 1.0 + kGeometryTolerance) {
      break;
    }
    const double clipped_parameter =
        std::clamp(parameter, 0.0, 1.0);
    const bool crosses_x =
        std::abs(maximum_x - maximum_y) <=
        kGeometryTolerance;
    if (crosses_x) {
      const Cell x_neighbor{
          .x = current.x + step_x,
          .y = current.y,
      };
      const Cell y_neighbor{
          .x = current.x,
          .y = current.y + step_y,
      };
      const Cell diagonal{
          .x = current.x + step_x,
          .y = current.y + step_y,
      };
      if (!projection.Known(x_neighbor) ||
          !projection.Known(y_neighbor) ||
          !projection.Known(diagonal)) {
        return {
            .endpoint =
                Interpolate(start, goal, clipped_parameter),
            .clipped = true,
        };
      }
      current = diagonal;
      maximum_x += delta_parameter_x;
      maximum_y += delta_parameter_y;
    } else if (maximum_x < maximum_y) {
      const Cell next{
          .x = current.x + step_x,
          .y = current.y,
      };
      if (!projection.Known(next)) {
        return {
            .endpoint =
                Interpolate(start, goal, clipped_parameter),
            .clipped = true,
        };
      }
      current = next;
      maximum_x += delta_parameter_x;
    } else {
      const Cell next{
          .x = current.x,
          .y = current.y + step_y,
      };
      if (!projection.Known(next)) {
        return {
            .endpoint =
                Interpolate(start, goal, clipped_parameter),
            .clipped = true,
        };
      }
      current = next;
      maximum_y += delta_parameter_y;
    }
  }
  return {.endpoint = goal, .clipped = false};
}

[[nodiscard]] UnresolvedTailPreview MakeUnresolvedTail(
    const SafeProjection& projection,
    const TerminalCandidate& first_candidate,
    const Vec3& goal_centroid) {
  const ClippedIntent clipped = ClipAtKnownBoundary(
      projection, first_candidate.cell,
      first_candidate.position_m, goal_centroid);
  return {
      .intent_polyline_m =
          {first_candidate.position_m, clipped.endpoint},
      .executable = false,
      .reason_code =
          clipped.clipped
              ? "mission_intent_clipped_at_known_boundary"
              : "mission_intent_did_not_cross_unknown_space",
  };
}

[[nodiscard]] ResolvedTerminalSet Outcome(
    TerminalKind kind, std::string reason_code) {
  return {
      .kind = kind,
      .candidates = {},
      .unresolved_tail = std::nullopt,
      .reason_code = std::move(reason_code),
  };
}

}  // namespace

Result<ResolvedTerminalSet> ResolveTerminal(
    const TerminalResolutionRequest& request) {
  double required_frontier_clearance_m = 0.0;
  if (auto error =
          ValidateRequest(request,
                          required_frontier_clearance_m);
      error.has_value()) {
    return std::move(*error);
  }

  if (!request.projection.InBounds(request.start_cell) ||
      !request.projection.HardFeasible(request.start_cell)) {
    return Outcome(TerminalKind::kNoKnownSafeRoute,
                   "start_not_hard_feasible");
  }
  const std::vector<std::uint8_t> start_component =
      ReachableFromStart(request.projection, request.start_cell);
  const RasterizedGoal goal_cells =
      RasterizeGoal(request.projection, request.goal_region);
  std::vector<Cell> reachable_goal_cells;
  reachable_goal_cells.reserve(goal_cells.hard_feasible.size());
  for (const Cell cell : goal_cells.hard_feasible) {
    if (start_component[
            CellIndex(request.projection.geometry, cell)] != 0U) {
      reachable_goal_cells.push_back(cell);
    }
  }

  if (!reachable_goal_cells.empty()) {
    auto candidates = MakeGoalCandidates(
        request.projection, reachable_goal_cells,
        request.goal_region);
    if (!IsOk(candidates)) {
      return std::get<Error>(std::move(candidates));
    }
    return ResolvedTerminalSet{
        .kind = TerminalKind::kGoal,
        .candidates =
            std::get<std::vector<TerminalCandidate>>(
                std::move(candidates)),
        .unresolved_tail = std::nullopt,
        .reason_code = "known_safe_goal_reachable",
    };
  }
  if (!goal_cells.in_bounds.empty() &&
      goal_cells.known.size() == goal_cells.in_bounds.size() &&
      goal_cells.hard_feasible.empty()) {
    return Outcome(
        TerminalKind::kGoalInfeasible,
        "goal_fully_known_and_infeasible");
  }
  if (!goal_cells.hard_feasible.empty()) {
    return Outcome(
        TerminalKind::kNoKnownSafeRoute,
        "goal_in_disconnected_known_component");
  }

  const Vec3 goal_centroid = GoalCentroid(request.goal_region);
  auto candidates = ExtractFrontierCandidates(
      request, start_component, required_frontier_clearance_m,
      goal_centroid);
  if (!IsOk(candidates)) {
    return std::get<Error>(std::move(candidates));
  }
  auto resolved_candidates =
      std::get<std::vector<TerminalCandidate>>(
          std::move(candidates));
  if (resolved_candidates.empty()) {
    return Outcome(TerminalKind::kNoKnownSafeRoute,
                   "no_certified_safe_frontier");
  }
  UnresolvedTailPreview unresolved_tail = MakeUnresolvedTail(
      request.projection, resolved_candidates.front(),
      goal_centroid);
  return ResolvedTerminalSet{
      .kind = TerminalKind::kSafeFrontier,
      .candidates = std::move(resolved_candidates),
      .unresolved_tail = std::move(unresolved_tail),
      .reason_code =
          "known_safe_frontier_before_unknown_space",
  };
}

}  // namespace lunar::planning::v3

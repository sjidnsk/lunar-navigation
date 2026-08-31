#include "hierarchical/surface_portal_set.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

#include "hierarchical/frame_transform.hpp"
#include "legged/legged_traversal_projection.hpp"
#include "shared/controlled_work.hpp"
#include "shared/global_occupancy_projection.hpp"
#include "shared/local_terrain_projection.hpp"

namespace lunar::pure_planning::hierarchical {
namespace {

constexpr std::size_t kMaximumPortalCandidates = 32U;
constexpr std::size_t kMaximumLongitudinalSamples = 32U;
constexpr double kMaximumPortalForwardProgressM = 12.0;
constexpr double kMinimumPortalForwardProgressM = 1.0;
constexpr std::array<std::int32_t, 9> kLateralCells{0, -1, 1, -2, 2,
                                                    -3, 3, -4, 4};

struct RouteSample final {
  Vec3 position_map;
  Vec2 tangent;
};

[[nodiscard]] SurfacePortalSetResult Failure(std::string reason_code) {
  return {.reason_code = std::move(reason_code)};
}

[[nodiscard]] double DistanceXY(const Vec3& left, const Vec3& right) noexcept {
  return std::hypot(left.x - right.x, left.y - right.y);
}

[[nodiscard]] double RouteLength(const GlobalRoute& route) noexcept {
  double length{};
  for (std::size_t index = 1U; index < route.poses_map.size(); ++index) {
    length += DistanceXY(route.poses_map[index - 1U].position_m,
                         route.poses_map[index].position_m);
  }
  return length;
}

[[nodiscard]] std::optional<RouteSample> SampleRoute(
    const GlobalRoute& route, const double progress_m) noexcept {
  double consumed{};
  std::optional<RouteSample> last;
  for (std::size_t index = 1U; index < route.poses_map.size(); ++index) {
    const Vec3& start = route.poses_map[index - 1U].position_m;
    const Vec3& finish = route.poses_map[index].position_m;
    const double dx = finish.x - start.x;
    const double dy = finish.y - start.y;
    const double length = std::hypot(dx, dy);
    if (length <= std::numeric_limits<double>::epsilon()) {
      continue;
    }
    const Vec2 tangent{.x = dx / length, .y = dy / length};
    const double fraction = std::clamp((progress_m - consumed) / length,
                                       0.0, 1.0);
    last = RouteSample{
        .position_map = {
            .x = start.x + fraction * dx,
            .y = start.y + fraction * dy,
            .z = start.z + fraction * (finish.z - start.z),
        },
        .tangent = tangent,
    };
    if (progress_m <= consumed + length) {
      return last;
    }
    consumed += length;
  }
  return last;
}

[[nodiscard]] std::optional<double> GlobalInflation(
    const PlatformCapability& capability) noexcept {
  const auto* wheel = std::get_if<WheeledCapability>(&capability);
  if (wheel == nullptr) {
    return 0.0;
  }
  if (!std::isfinite(wheel->minimum_clearance_m) ||
      wheel->minimum_clearance_m < 0.0) {
    return std::nullopt;
  }
  double radius{};
  for (const Vec2 vertex : wheel->footprint_xy_m) {
    if (!std::isfinite(vertex.x) || !std::isfinite(vertex.y)) {
      return std::nullopt;
    }
    radius = std::max(radius, std::hypot(vertex.x, vertex.y));
  }
  return radius + wheel->minimum_clearance_m;
}

[[nodiscard]] bool CandidateLess(const SurfacePortalCandidate& left,
                                 const SurfacePortalCandidate& right) noexcept {
  if (left.route_progress_m != right.route_progress_m) {
    return left.route_progress_m > right.route_progress_m;
  }
  if (std::abs(left.lateral_offset_cells) !=
      std::abs(right.lateral_offset_cells)) {
    return std::abs(left.lateral_offset_cells) <
           std::abs(right.lateral_offset_cells);
  }
  if (left.global_clearance_m != right.global_clearance_m) {
    return left.global_clearance_m > right.global_clearance_m;
  }
  if (left.local_clearance_m != right.local_clearance_m) {
    return left.local_clearance_m > right.local_clearance_m;
  }
  return left.stable_rank < right.stable_rank;
}

[[nodiscard]] bool LocalCellSafe(
    const shared::LocalTerrainProjection& local,
    const shared::GridCell cell,
    const legged::LeggedTraversalProjection* legged_local = nullptr) noexcept {
  if (local.map == nullptr || !local.map->InBounds(cell)) {
    return false;
  }
  const std::size_t index = local.map->Index(cell);
  return index < local.free_with_height.size() &&
         local.free_with_height[index] != 0U &&
         index < local.clearance_m.size() &&
         (legged_local == nullptr ||
          (index < legged_local->body_center_feasible.size() &&
           legged_local->body_center_feasible[index] != 0U));
}

[[nodiscard]] std::vector<double> LongitudinalProgresses(
    const SurfaceRollingDecision& decision,
    const double resolution_m) {
  const double maximum_progress_m = std::min(
      decision.desired_horizon_progress_m,
      decision.projected_route_progress_m + kMaximumPortalForwardProgressM);
  const double minimum_progress_m = decision.projected_route_progress_m +
                                    kMinimumPortalForwardProgressM;
  if (maximum_progress_m + 1.0e-9 < minimum_progress_m) {
    return {};
  }
  const double progress_span_m = maximum_progress_m - minimum_progress_m;
  const auto available_backoff_cells = static_cast<std::size_t>(
      std::max(0.0, std::floor(progress_span_m / resolution_m + 1.0e-9)));
  const std::size_t sample_count = std::clamp(
      available_backoff_cells + 1U, std::size_t{1U},
      kMaximumLongitudinalSamples);

  std::vector<double> progresses;
  progresses.reserve(sample_count);
  for (std::size_t sample = 0U; sample < sample_count; ++sample) {
    std::size_t backoff_cells = sample;
    if (available_backoff_cells + 1U > kMaximumLongitudinalSamples &&
        sample_count > 1U) {
      backoff_cells = static_cast<std::size_t>(std::llround(
          static_cast<double>(sample) *
          static_cast<double>(available_backoff_cells) /
          static_cast<double>(sample_count - 1U)));
    }
    progresses.push_back(maximum_progress_m -
                         static_cast<double>(backoff_cells) * resolution_m);
  }
  return progresses;
}

}  // namespace

SurfacePortalSetResult BuildSurfacePortalSet(
    const PlanningRequest& input, const GlobalRoute& route,
    const SurfaceRollingDecision& decision, const std::size_t max_candidates,
    SearchControl control) {
  const double route_length_m = RouteLength(route);
  if (decision.kind != SurfaceRollingDecision::Kind::kNextPortalSet ||
      route.poses_map.size() < 2U || !std::isfinite(route_length_m) ||
      route_length_m <= 0.0 ||
      !std::isfinite(decision.projected_route_progress_m) ||
      !std::isfinite(decision.desired_horizon_progress_m) ||
      decision.projected_route_progress_m < 0.0 ||
      decision.desired_horizon_progress_m <
          decision.projected_route_progress_m ||
      decision.desired_horizon_progress_m > route_length_m + 1.0e-9 ||
      !input.world.global_map.has_value() ||
      input.world.global_map->frame_id != input.world.map_from_odom.parent_frame ||
      input.world.local_map.frame_id != input.world.map_from_odom.child_frame) {
    return Failure("INVALID_INPUT");
  }
  if (const auto stopped = shared::StopReason(control); stopped.has_value()) {
    return Failure(std::string{*stopped});
  }

  const auto global_map = shared::MapSnapshot::Create(
      *input.world.global_map, shared::MapContract::kGlobalOccupancy, control);
  if (!global_map.ok()) {
    return Failure(global_map.reason_code == "TIMEOUT" ||
                           global_map.reason_code == "REQUEST_CANCELED"
                       ? global_map.reason_code
                       : "INVALID_INPUT");
  }
  const auto inflation = GlobalInflation(input.capability);
  if (!inflation.has_value()) {
    return Failure("INVALID_INPUT");
  }
  const auto global = shared::BuildInflatedGlobalOccupancyProjection(
      global_map.snapshot, input.config.global_occupancy_threshold, *inflation,
      control);
  if (!global.ok()) {
    return Failure(global.reason_code == "TIMEOUT" ||
                           global.reason_code == "REQUEST_CANCELED"
                       ? global.reason_code
                       : "INVALID_INPUT");
  }

  const auto local_map = shared::MapSnapshot::Create(
      input.world.local_map, shared::MapContract::kLocalElevation, control);
  if (!local_map.ok()) {
    return Failure(local_map.reason_code == "TIMEOUT" ||
                           local_map.reason_code == "REQUEST_CANCELED"
                       ? local_map.reason_code
                       : "INVALID_INPUT");
  }
  const auto local = shared::BuildLocalTerrainProjection(
      local_map.snapshot,
      static_cast<float>(input.config.local_occupancy_threshold), control);
  if (!local.ok()) {
    return Failure(local.reason_code == "TIMEOUT" ||
                           local.reason_code == "REQUEST_CANCELED"
                       ? local.reason_code
                       : "INVALID_INPUT");
  }
  std::shared_ptr<const shared::LocalTerrainProjection> local_projection =
      std::make_shared<const shared::LocalTerrainProjection>(
          std::move(*local.value));
  std::shared_ptr<const legged::LeggedTraversalProjection> legged_local;
  if (!decision.targets_final_goal &&
      input.config.legged_global_mode ==
          LeggedGlobalMode::kGridTraversabilityV1) {
    if (const auto* capability =
            std::get_if<LeggedCapability>(&input.capability);
        capability != nullptr) {
      auto built = legged::BuildLeggedTraversalProjection(
          local_projection, *capability, control);
      if (!built.ok()) {
        return Failure(built.reason_code == "TIMEOUT" ||
                               built.reason_code == "REQUEST_CANCELED"
                           ? std::move(built.reason_code)
                           : "INVALID_INPUT");
      }
      legged_local = std::move(built.value);
    }
  }

  const auto global_view = global.projection->View();
  const std::size_t bounded_max =
      std::min(max_candidates, kMaximumPortalCandidates);
  if (bounded_max == 0U) {
    return Failure("NO_PATH");
  }

  if (decision.targets_final_goal) {
    const auto final_odom = TransformGoal(
        input.goal_map, input.world.map_from_odom,
        TransformDirection::kParentToChild);
    const auto* final_map_point = std::get_if<PointGoal>(&input.goal_map.target);
    const auto* final_odom_point =
        final_odom ? std::get_if<PointGoal>(&final_odom->target) : nullptr;
    if (final_map_point == nullptr || final_odom_point == nullptr) {
      return Failure("INVALID_INPUT");
    }
    const auto global_cell = global_map.snapshot->PositionToCell(
        {.x = final_map_point->position_m.x,
         .y = final_map_point->position_m.y});
    const auto local_cell = local_map.snapshot->PositionToCell(
        {.x = final_odom_point->position_m.x,
         .y = final_odom_point->position_m.y});
    if (!global_cell.has_value() || !local_cell.has_value() ||
        !global_view.HardFeasible(*global_cell) ||
        !LocalCellSafe(*local_projection, *local_cell)) {
      return Failure("NO_PATH");
    }
    const std::size_t local_index = local_map.snapshot->Index(*local_cell);
    return SurfacePortalSetResult{
        .candidates = {SurfacePortalCandidate{
            .goal_odom = *final_odom,
            .route_progress_m = decision.desired_horizon_progress_m,
            .global_cell = *global_cell,
            .local_cell = *local_cell,
            .global_clearance_m = global_view.ClearanceMeters(*global_cell),
            .local_clearance_m = local_projection->clearance_m[local_index],
            .lateral_offset_cells = 0,
            .stable_rank = 0U,
        }},
    };
  }

  const auto longitudinal_progresses = LongitudinalProgresses(
      decision, global_map.snapshot->resolution_m());
  std::vector<SurfacePortalCandidate> candidates;
  candidates.reserve(longitudinal_progresses.size() * kLateralCells.size());
  std::set<std::tuple<std::int32_t, std::int32_t, std::int32_t, std::int32_t>>
      seen_cells;
  std::size_t creation_rank{};
  for (const double progress_m : longitudinal_progresses) {
    const auto sample = SampleRoute(route, progress_m);
    if (!sample.has_value()) {
      return Failure("INVALID_INPUT");
    }
    for (const std::int32_t lateral_cells : kLateralCells) {
      if (const auto stopped = shared::StopReason(control);
          stopped.has_value()) {
        return Failure(std::string{*stopped});
      }
      const double lateral_m = static_cast<double>(lateral_cells) *
                               global_map.snapshot->resolution_m();
      const Vec2 position_map{
          .x = sample->position_map.x - sample->tangent.y * lateral_m,
          .y = sample->position_map.y + sample->tangent.x * lateral_m,
      };
      const auto global_cell =
          global_map.snapshot->PositionToCell(position_map);
      if (!global_cell.has_value() ||
          !global_view.HardFeasible(*global_cell)) {
        ++creation_rank;
        continue;
      }
      const Vec3 global_center = global_map.snapshot->CellCenter(*global_cell);
      const auto center_odom = TransformPoint(
          {.x = global_center.x, .y = global_center.y, .z = 0.0},
          input.world.map_from_odom, TransformDirection::kParentToChild);
      if (!center_odom.has_value()) {
        return Failure("INVALID_INPUT");
      }
      const auto local_cell = local_map.snapshot->PositionToCell(
          {.x = center_odom->x, .y = center_odom->y});
      if (!local_cell.has_value() ||
          !LocalCellSafe(*local_projection, *local_cell,
                         legged_local.get())) {
        ++creation_rank;
        continue;
      }
      const auto key = std::tuple{global_cell->x, global_cell->y,
                                  local_cell->x, local_cell->y};
      if (!seen_cells.insert(key).second) {
        ++creation_rank;
        continue;
      }
      const std::size_t local_index = local_map.snapshot->Index(*local_cell);
      candidates.push_back(SurfacePortalCandidate{
          .goal_odom = GoalRegion{
              .goal_id = input.goal_map.goal_id + "/portal",
              .target = PointGoal{
                  .position_m = {.x = center_odom->x,
                                 .y = center_odom->y,
                                 .z = 0.0},
                  .tolerance_m = 0.0,
              },
              .yaw_rad = std::nullopt,
              .yaw_tolerance_rad = 0.0,
          },
          .route_progress_m = progress_m,
          .global_cell = *global_cell,
          .local_cell = *local_cell,
          .global_clearance_m = global_view.ClearanceMeters(*global_cell),
          .local_clearance_m = local_projection->clearance_m[local_index],
          .lateral_offset_cells = lateral_cells,
          .stable_rank = creation_rank,
      });
      ++creation_rank;
    }
  }

  if (candidates.empty()) {
    return Failure("NO_PATH");
  }
  std::sort(candidates.begin(), candidates.end(), CandidateLess);
  if (candidates.size() > bounded_max) {
    candidates.resize(bounded_max);
  }
  return {.candidates = std::move(candidates)};
}

}  // namespace lunar::pure_planning::hierarchical

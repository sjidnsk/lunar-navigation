#include "hierarchical/surface_rolling_session.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <utility>

namespace lunar::pure_planning::hierarchical {
namespace {

[[nodiscard]] bool IsFinite(const Vec3& value) {
  return std::isfinite(value.x) && std::isfinite(value.y) &&
         std::isfinite(value.z);
}

[[nodiscard]] bool IsFinite(const Quaternion& value) {
  return std::isfinite(value.w) && std::isfinite(value.x) &&
         std::isfinite(value.y) && std::isfinite(value.z);
}

[[nodiscard]] bool IsFinite(const Pose3& pose) {
  return IsFinite(pose.position_m) && IsFinite(pose.orientation);
}

[[nodiscard]] double DistanceXY(const Vec3& left, const Vec3& right) {
  return std::hypot(left.x - right.x, left.y - right.y);
}

[[nodiscard]] Vec3 Interpolate(const Vec3& start, const Vec3& end,
                                 const double fraction) {
  return {
      .x = start.x + (end.x - start.x) * fraction,
      .y = start.y + (end.y - start.y) * fraction,
      .z = start.z + (end.z - start.z) * fraction,
  };
}

struct ClosestProjection final {
  std::size_t segment_index{};
  double fraction{};
  double lateral_deviation_m{std::numeric_limits<double>::infinity()};
};

[[nodiscard]] std::optional<ClosestProjection> FindClosestProjection(
    const std::vector<Pose3>& route, const Vec3& position) {
  std::optional<ClosestProjection> closest;
  for (std::size_t index = 0; index + 1U < route.size(); ++index) {
    const Vec3& start = route[index].position_m;
    const Vec3& end = route[index + 1U].position_m;
    const double dx = end.x - start.x;
    const double dy = end.y - start.y;
    const double length_squared = dx * dx + dy * dy;
    if (length_squared <= std::numeric_limits<double>::epsilon()) {
      continue;
    }
    const double raw_fraction =
        ((position.x - start.x) * dx + (position.y - start.y) * dy) /
        length_squared;
    const double fraction = std::clamp(raw_fraction, 0.0, 1.0);
    const Vec3 projected = Interpolate(start, end, fraction);
    const double lateral_deviation_m = DistanceXY(position, projected);
    // At an equal-distance corner, use the later segment so the rolling
    // target continues forward rather than consuming the preceding leg.
    if (!closest.has_value() ||
        lateral_deviation_m <= closest->lateral_deviation_m) {
      closest = ClosestProjection{
          .segment_index = index,
          .fraction = fraction,
          .lateral_deviation_m = lateral_deviation_m,
      };
    }
  }
  return closest;
}

[[nodiscard]] bool IsValidFinalGoal(const GoalRegion& goal) {
  const auto* point = std::get_if<PointGoal>(&goal.target);
  return point != nullptr && IsFinite(point->position_m) &&
         std::isfinite(point->tolerance_m) && point->tolerance_m >= 0.0;
}

[[nodiscard]] bool IsValidRoute(const GlobalRoute& route) {
  return route.poses_map.size() >= 2U &&
         std::all_of(route.poses_map.begin(), route.poses_map.end(),
                     [](const Pose3& pose) { return IsFinite(pose); });
}

}  // namespace

SurfaceRollingSession::SurfaceRollingSession(GlobalRoute route,
                                             GoalRegion final_goal,
                                             SurfaceRollingConfig config)
    : route_(std::move(route)),
      final_goal_(std::move(final_goal)),
      config_(config) {}

SurfaceRollingDecision SurfaceRollingSession::Decide(
    const Pose3& pose_map, const double minimum_route_progress_m) const {
  if (!IsValidRoute(route_) || !IsValidFinalGoal(final_goal_) ||
      !IsFinite(pose_map) || !std::isfinite(config_.horizon_m) ||
      config_.horizon_m <= 0.0 || !std::isfinite(config_.max_deviation_m) ||
      config_.max_deviation_m < 0.0 ||
      !std::isfinite(minimum_route_progress_m) ||
      minimum_route_progress_m < 0.0) {
    return {};
  }

  double route_length_m{};
  for (std::size_t index = 0U; index + 1U < route_.poses_map.size();
       ++index) {
    route_length_m += DistanceXY(route_.poses_map[index].position_m,
                                 route_.poses_map[index + 1U].position_m);
  }
  if (minimum_route_progress_m > route_length_m) {
    return {};
  }

  const auto* final_point = std::get_if<PointGoal>(&final_goal_.target);
  if (DistanceXY(pose_map.position_m, final_point->position_m) <=
      final_point->tolerance_m) {
    return {
        .kind = SurfaceRollingDecision::Kind::kFinalGoalReached,
        .lateral_deviation_m = 0.0,
    };
  }

  const auto closest = FindClosestProjection(route_.poses_map, pose_map.position_m);
  if (!closest.has_value()) {
    return {};
  }

  double projected_route_progress_m{};
  for (std::size_t index = 0U; index + 1U < route_.poses_map.size(); ++index) {
    const double segment_length_m =
        DistanceXY(route_.poses_map[index].position_m,
                   route_.poses_map[index + 1U].position_m);
    if (index < closest->segment_index) {
      projected_route_progress_m += segment_length_m;
    } else if (index == closest->segment_index) {
      projected_route_progress_m += closest->fraction * segment_length_m;
    }
  }
  projected_route_progress_m =
      std::max(projected_route_progress_m, minimum_route_progress_m);

  const double desired_horizon_progress_m =
      std::min(route_length_m,
               projected_route_progress_m + config_.horizon_m);
  return {
      .kind = SurfaceRollingDecision::Kind::kNextPortalSet,
      .projected_route_progress_m = projected_route_progress_m,
      .desired_horizon_progress_m = desired_horizon_progress_m,
      .targets_final_goal = desired_horizon_progress_m >= route_length_m,
      .lateral_deviation_m = closest->lateral_deviation_m,
  };
}

}  // namespace lunar::pure_planning::hierarchical

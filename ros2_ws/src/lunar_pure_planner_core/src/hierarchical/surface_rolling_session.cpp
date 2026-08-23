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

SurfaceRollingDecision SurfaceRollingSession::Decide(const Pose3& pose_map) const {
  if (!IsValidRoute(route_) || !IsValidFinalGoal(final_goal_) ||
      !IsFinite(pose_map) || !std::isfinite(config_.horizon_m) ||
      config_.horizon_m <= 0.0 || !std::isfinite(config_.max_deviation_m) ||
      config_.max_deviation_m < 0.0) {
    return {};
  }

  const auto* final_point = std::get_if<PointGoal>(&final_goal_.target);
  if (DistanceXY(pose_map.position_m, final_point->position_m) <=
      final_point->tolerance_m) {
    return {
        .kind = SurfaceRollingDecision::Kind::kFinalGoalReached,
        .goal = std::nullopt,
        .lateral_deviation_m = 0.0,
    };
  }

  const auto closest = FindClosestProjection(route_.poses_map, pose_map.position_m);
  if (!closest.has_value()) {
    return {};
  }

  double remaining_horizon_m = config_.horizon_m;
  for (std::size_t index = closest->segment_index;
       index + 1U < route_.poses_map.size(); ++index) {
    const Vec3& start = route_.poses_map[index].position_m;
    const Vec3& end = route_.poses_map[index + 1U].position_m;
    const double segment_length_m = DistanceXY(start, end);
    if (segment_length_m <= std::numeric_limits<double>::epsilon()) {
      continue;
    }
    const double start_fraction = index == closest->segment_index
                                      ? closest->fraction
                                      : 0.0;
    const double available_m = (1.0 - start_fraction) * segment_length_m;
    if (remaining_horizon_m <= available_m) {
      const double target_fraction =
          start_fraction + remaining_horizon_m / segment_length_m;
      GoalRegion goal = final_goal_;
      goal.target = PointGoal{
          .position_m = Interpolate(start, end, target_fraction),
          .tolerance_m = final_point->tolerance_m,
      };
      return {
          .kind = SurfaceRollingDecision::Kind::kNextGoal,
          .goal = std::move(goal),
          .lateral_deviation_m = closest->lateral_deviation_m,
      };
    }
    remaining_horizon_m -= available_m;
  }

  return {
      .kind = SurfaceRollingDecision::Kind::kNextGoal,
      .goal = final_goal_,
      .lateral_deviation_m = closest->lateral_deviation_m,
  };
}

}  // namespace lunar::pure_planning::hierarchical

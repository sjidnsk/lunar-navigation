#include "hierarchical/route_continuation.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <optional>
#include <string>
#include <utility>
#include <variant>

#include "hierarchical/frame_transform.hpp"

namespace lunar::planning {
namespace {

constexpr double kTolerance = 1.0e-9;

[[nodiscard]] bool SameGoal(const GoalRegion &lhs,
                            const GoalRegion &rhs) noexcept {
  if (lhs.goal_id != rhs.goal_id || lhs.yaw_rad != rhs.yaw_rad ||
      lhs.yaw_tolerance_rad != rhs.yaw_tolerance_rad ||
      lhs.target.index() != rhs.target.index()) {
    return false;
  }
  if (const auto *left = std::get_if<PointGoal>(&lhs.target)) {
    const auto &right = std::get<PointGoal>(rhs.target);
    return left->position_m == right.position_m &&
           left->tolerance_m == right.tolerance_m;
  }
  const auto &left = std::get<PlanarRegionGoal>(lhs.target);
  const auto &right = std::get<PlanarRegionGoal>(rhs.target);
  return left.boundary_m == right.boundary_m &&
         left.normal_tolerance_m == right.normal_tolerance_m;
}

[[nodiscard]] std::optional<Pose3> CurrentGroundPose(
    const PlannerInput &input) noexcept {
  if (const auto *state = std::get_if<WheeledState>(&input.current_state)) {
    return state->pose;
  }
  if (const auto *state = std::get_if<LeggedState>(&input.current_state)) {
    return state->body_pose;
  }
  return std::nullopt;
}

[[nodiscard]] double PlanarDistance(const Vec3 lhs, const Vec3 rhs) noexcept {
  return std::hypot(lhs.x - rhs.x, lhs.y - rhs.y);
}

[[nodiscard]] double Norm(const Vec3 value) noexcept {
  return std::hypot(std::hypot(value.x, value.y), value.z);
}

[[nodiscard]] std::optional<double> QuaternionDistance(
    const Quaternion lhs, const Quaternion rhs) noexcept {
  const double lhs_norm =
      std::hypot(std::hypot(lhs.w, lhs.x), std::hypot(lhs.y, lhs.z));
  const double rhs_norm =
      std::hypot(std::hypot(rhs.w, rhs.x), std::hypot(rhs.y, rhs.z));
  if (!std::isfinite(lhs_norm) || !std::isfinite(rhs_norm) ||
      lhs_norm <= kTolerance || rhs_norm <= kTolerance) {
    return std::nullopt;
  }
  const double dot = std::clamp(
      std::abs((lhs.w * rhs.w + lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z) /
               (lhs_norm * rhs_norm)),
      0.0, 1.0);
  return 2.0 * std::acos(dot);
}

[[nodiscard]] bool TransformDeltaWithin(
    const RigidTransform &current, const RigidTransform &issued,
    const double maximum_translation_m) noexcept {
  if (current.parent_frame != issued.parent_frame ||
      current.child_frame != issued.child_frame ||
      !std::isfinite(maximum_translation_m) || maximum_translation_m < 0.0) {
    return false;
  }
  const Vec3 delta{
      .x = current.translation_m.x - issued.translation_m.x,
      .y = current.translation_m.y - issued.translation_m.y,
      .z = current.translation_m.z - issued.translation_m.z,
  };
  const auto rotation_delta =
      QuaternionDistance(current.rotation, issued.rotation);
  return rotation_delta.has_value() && Norm(delta) <= maximum_translation_m &&
         *rotation_delta <= std::numbers::pi / 8.0;
}

[[nodiscard]] std::optional<double> Yaw(const Quaternion value) noexcept {
  const double norm =
      std::hypot(std::hypot(value.w, value.x), std::hypot(value.y, value.z));
  if (!std::isfinite(norm) || norm <= kTolerance) {
    return std::nullopt;
  }
  const double w = value.w / norm;
  const double x = value.x / norm;
  const double y = value.y / norm;
  const double z = value.z / norm;
  const double result =
      std::atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z));
  return std::isfinite(result) ? std::optional<double>{result} : std::nullopt;
}

[[nodiscard]] double AngleDifference(const double lhs,
                                     const double rhs) noexcept {
  return std::abs(std::remainder(lhs - rhs, 2.0 * std::numbers::pi));
}

[[nodiscard]] Quaternion YawQuaternion(const double yaw) noexcept {
  return Quaternion{.w = std::cos(yaw / 2.0), .z = std::sin(yaw / 2.0)};
}

[[nodiscard]] bool PointOnSegment(const Vec2 point, const Vec2 start,
                                  const Vec2 end) noexcept {
  const double cross = (point.x - start.x) * (end.y - start.y) -
                       (point.y - start.y) * (end.x - start.x);
  if (std::abs(cross) > kTolerance) {
    return false;
  }
  return (point.x - start.x) * (point.x - end.x) +
             (point.y - start.y) * (point.y - end.y) <=
         kTolerance;
}

[[nodiscard]] bool PointInPolygon(const Vec2 point,
                                  const std::vector<Vec3> &boundary) noexcept {
  if (boundary.size() < 3U) {
    return false;
  }
  bool inside = false;
  for (std::size_t current = 0U, previous = boundary.size() - 1U;
       current < boundary.size(); previous = current++) {
    const Vec2 start{boundary[previous].x, boundary[previous].y};
    const Vec2 end{boundary[current].x, boundary[current].y};
    if (PointOnSegment(point, start, end)) {
      return true;
    }
    if ((start.y > point.y) == (end.y > point.y)) {
      continue;
    }
    const double crossing =
        start.x + (point.y - start.y) * (end.x - start.x) / (end.y - start.y);
    if (point.x < crossing) {
      inside = !inside;
    }
  }
  return inside;
}

[[nodiscard]] bool FreshPromotionSnapshot(const PlannerInput &input) noexcept {
  const auto maximum_skew = input.config.maximum_input_skew.count();
  if (maximum_skew < 0 || input.state_time.nanoseconds_since_epoch <= 0) {
    return false;
  }
  const auto within = [&](const TimePoint stamp) {
    if (stamp.nanoseconds_since_epoch <= 0) {
      return false;
    }
    const long double difference = std::abs(
        static_cast<long double>(input.state_time.nanoseconds_since_epoch) -
        static_cast<long double>(stamp.nanoseconds_since_epoch));
    return difference <= static_cast<long double>(maximum_skew);
  };
  return within(input.world.global_map.stamp) &&
         within(input.world.local_map.stamp) &&
         within(input.world.map_from_odom.stamp);
}

[[nodiscard]] std::string IdentityMismatch(
    const PlannerInput &input, const RouteContinuation &continuation) noexcept {
  if (input.mission_id.empty() || continuation.mission_id().empty() ||
      input.mission_id != continuation.mission_id() ||
      input.mission_revision != continuation.mission_revision()) {
    return "ROUTE_CONTINUATION_MISSION_MISMATCH";
  }
  if (!SameGoal(input.goal_map, continuation.goal_map())) {
    return "ROUTE_CONTINUATION_GOAL_MISMATCH";
  }
  if (input.platform_id.empty() || continuation.platform_id().empty() ||
      input.platform_id != continuation.platform_id() ||
      CapabilityPlatform(input.capability) != continuation.platform_type()) {
    return "ROUTE_CONTINUATION_PLATFORM_MISMATCH";
  }
  if (input.capability_version.empty() ||
      continuation.capability_version().empty() ||
      input.capability_version != continuation.capability_version()) {
    return "ROUTE_CONTINUATION_CAPABILITY_MISMATCH";
  }
  if (input.global_map_generation == 0U ||
      continuation.global_map_generation() == 0U ||
      input.global_map_generation != continuation.global_map_generation()) {
    return "ROUTE_CONTINUATION_GLOBAL_MAP_MISMATCH";
  }
  return {};
}

}  // namespace

RouteContinuation::RouteContinuation(
    std::string route_id, std::string current_reference_plan_id,
    const PlannerInput &input, hierarchical::GlobalRoute global_route,
    std::vector<CertifiedHopPreview> certified_hops,
    const std::size_t route_cursor, const double corridor_half_width_m,
    const std::uint64_t rolling_request_count)
    : route_id_(std::move(route_id)),
      current_reference_plan_id_(std::move(current_reference_plan_id)),
      platform_type_(CapabilityPlatform(input.capability)),
      mission_id_(input.mission_id),
      mission_revision_(input.mission_revision),
      platform_id_(input.platform_id),
      goal_map_(input.goal_map),
      capability_version_(input.capability_version),
      global_map_generation_(input.global_map_generation),
      local_map_generation_at_issue_(input.local_map_generation),
      map_from_odom_generation_(input.map_from_odom_generation),
      map_from_odom_at_issue_(input.world.map_from_odom),
      global_route_(std::move(global_route)),
      certified_hops_(std::move(certified_hops)),
      route_cursor_(route_cursor),
      corridor_half_width_m_(corridor_half_width_m),
      rolling_request_count_(rolling_request_count) {}

const std::string &RouteContinuation::route_id() const noexcept {
  return route_id_;
}

const std::string &RouteContinuation::current_reference_plan_id()
    const noexcept {
  return current_reference_plan_id_;
}

PlatformType RouteContinuation::platform_type() const noexcept {
  return platform_type_;
}

const std::string &RouteContinuation::mission_id() const noexcept {
  return mission_id_;
}

std::uint64_t RouteContinuation::mission_revision() const noexcept {
  return mission_revision_;
}

const std::string &RouteContinuation::platform_id() const noexcept {
  return platform_id_;
}

const GoalRegion &RouteContinuation::goal_map() const noexcept {
  return goal_map_;
}

const std::string &RouteContinuation::capability_version() const noexcept {
  return capability_version_;
}

std::uint64_t RouteContinuation::global_map_generation() const noexcept {
  return global_map_generation_;
}

std::uint64_t RouteContinuation::local_map_generation_at_issue()
    const noexcept {
  return local_map_generation_at_issue_;
}

std::uint64_t RouteContinuation::map_from_odom_generation() const noexcept {
  return map_from_odom_generation_;
}

const RigidTransform &RouteContinuation::map_from_odom_at_issue()
    const noexcept {
  return map_from_odom_at_issue_;
}

const hierarchical::GlobalRoute &RouteContinuation::global_route()
    const noexcept {
  return global_route_;
}

const std::vector<CertifiedHopPreview> &RouteContinuation::certified_hops()
    const noexcept {
  return certified_hops_;
}

std::size_t RouteContinuation::route_cursor() const noexcept {
  return route_cursor_;
}

double RouteContinuation::corridor_half_width_m() const noexcept {
  return corridor_half_width_m_;
}

std::uint64_t RouteContinuation::rolling_request_count() const noexcept {
  return rolling_request_count_;
}

namespace hierarchical {
namespace {

[[nodiscard]] GroundRouteReuseResult GroundFailure(std::string reason_code) {
  return GroundRouteReuseResult{.reason_code = std::move(reason_code)};
}

[[nodiscard]] HopperHopPromotionResult PromotionFailure(
    std::string reason_code) {
  return HopperHopPromotionResult{.reason_code = std::move(reason_code)};
}

}  // namespace

GroundRouteReuseResult TryReuseGroundRoute(
    const PlannerInput &input, const RouteContinuation &continuation) {
  if (const std::string mismatch = IdentityMismatch(input, continuation);
      !mismatch.empty()) {
    return GroundFailure(mismatch);
  }
  const PlatformType platform = CapabilityPlatform(input.capability);
  if ((platform != PlatformType::kWheeled &&
       platform != PlatformType::kLegged) ||
      !continuation.certified_hops().empty() ||
      !std::isfinite(continuation.corridor_half_width_m()) ||
      continuation.corridor_half_width_m() <= 0.0) {
    return GroundFailure("GROUND_ROUTE_CONTINUATION_INVALID");
  }
  const double maximum_tf_translation =
      std::max(continuation.corridor_half_width_m(),
               input.world.global_map.resolution_m);
  if (!TransformDeltaWithin(input.world.map_from_odom,
                            continuation.map_from_odom_at_issue(),
                            maximum_tf_translation)) {
    return GroundFailure("ROUTE_TF_DELTA_REPLAN_REQUIRED");
  }
  const auto pose_odom = CurrentGroundPose(input);
  if (!pose_odom.has_value()) {
    return GroundFailure("GROUND_ROUTE_STATE_INVALID");
  }
  const auto pose_map = TransformPose(*pose_odom, input.world.map_from_odom,
                                      TransformDirection::kChildToParent);
  if (!pose_map.has_value()) {
    return GroundFailure("FRAME_TRANSFORM_INVALID");
  }
  const std::vector<Pose3> &poses = continuation.global_route().poses_map;
  if (poses.size() < 2U) {
    return GroundFailure("GROUND_ROUTE_CONTINUATION_INVALID");
  }
  const std::size_t first_segment =
      continuation.route_cursor() == 0U
          ? 0U
          : std::min(continuation.route_cursor() - 1U, poses.size() - 2U);
  double best_distance = std::numeric_limits<double>::infinity();
  double best_fraction = 0.0;
  std::size_t best_segment = first_segment;
  Pose3 projection{};
  for (std::size_t index = first_segment; index + 1U < poses.size(); ++index) {
    const Vec3 start = poses[index].position_m;
    const Vec3 finish = poses[index + 1U].position_m;
    const double dx = finish.x - start.x;
    const double dy = finish.y - start.y;
    const double squared_length = dx * dx + dy * dy;
    if (!std::isfinite(squared_length) || squared_length <= kTolerance) {
      continue;
    }
    const double fraction =
        std::clamp(((pose_map->position_m.x - start.x) * dx +
                    (pose_map->position_m.y - start.y) * dy) /
                       squared_length,
                   0.0, 1.0);
    const Pose3 candidate{
        .position_m =
            Vec3{
                .x = start.x + fraction * dx,
                .y = start.y + fraction * dy,
                .z = start.z + fraction * (finish.z - start.z),
            },
        .orientation = YawQuaternion(std::atan2(dy, dx)),
    };
    const double distance =
        PlanarDistance(pose_map->position_m, candidate.position_m);
    if (distance < best_distance - kTolerance ||
        (std::abs(distance - best_distance) <= kTolerance &&
         (index > best_segment || fraction > best_fraction))) {
      best_distance = distance;
      best_fraction = fraction;
      best_segment = index;
      projection = candidate;
    }
  }
  if (!std::isfinite(best_distance) ||
      best_distance > continuation.corridor_half_width_m() + kTolerance) {
    return GroundFailure("GROUND_ROUTE_CORRIDOR_DEVIATION_REPLAN_REQUIRED");
  }
  const auto actual_yaw = Yaw(pose_map->orientation);
  const auto route_yaw = Yaw(projection.orientation);
  if (!actual_yaw.has_value() || !route_yaw.has_value() ||
      AngleDifference(*actual_yaw, *route_yaw) >
          std::numbers::pi / 4.0 + kTolerance) {
    return GroundFailure("GROUND_ROUTE_HEADING_DEVIATION_REPLAN_REQUIRED");
  }

  GlobalRoute reused = continuation.global_route();
  reused.poses_map.clear();
  reused.poses_map.reserve(poses.size() - best_segment + 1U);
  reused.poses_map.push_back(*pose_map);
  if (PlanarDistance(pose_map->position_m, projection.position_m) >
      kTolerance) {
    reused.poses_map.push_back(projection);
  }
  for (std::size_t index = best_segment + 1U; index < poses.size(); ++index) {
    if (PlanarDistance(reused.poses_map.back().position_m,
                       poses[index].position_m) > kTolerance) {
      reused.poses_map.push_back(poses[index]);
    }
  }
  if (reused.poses_map.empty()) {
    return GroundFailure("GROUND_ROUTE_CONTINUATION_INVALID");
  }
  reused.expanded_states = 0U;
  reused.open_peak = 0U;
  reused.estimated_work_memory_bytes = 0U;
  return GroundRouteReuseResult{
      .route = std::move(reused),
      .route_cursor = best_segment + 1U,
      .reason_code = "GROUND_ROUTE_REUSED",
  };
}

HopperHopPromotionResult TryPromoteHopperHop(
    const PlannerInput &input, const RouteContinuation &continuation) {
  if (const std::string mismatch = IdentityMismatch(input, continuation);
      !mismatch.empty()) {
    return PromotionFailure(mismatch);
  }
  const auto *state = std::get_if<HopperState>(&input.current_state);
  const auto *capability = std::get_if<HopperCapability>(&input.capability);
  if (state == nullptr || capability == nullptr ||
      continuation.platform_type() != PlatformType::kHopper ||
      continuation.certified_hops().empty()) {
    return PromotionFailure("HOPPER_ROUTE_CONTINUATION_INVALID");
  }
  const double maximum_tf_translation =
      input.world.global_map.resolution_m + input.position_uncertainty_m;
  if (!TransformDeltaWithin(input.world.map_from_odom,
                            continuation.map_from_odom_at_issue(),
                            maximum_tf_translation)) {
    return PromotionFailure("ROUTE_TF_DELTA_REPLAN_REQUIRED");
  }
  const auto *execution =
      input.previous_execution.has_value()
          ? std::get_if<HopperExecutionContext>(&*input.previous_execution)
          : nullptr;
  if (execution == nullptr ||
      execution->state != HopperExecutionState::kLandedHold ||
      !execution->active_plan_id.has_value() ||
      !execution->active_segment_id.has_value()) {
    return PromotionFailure("HOP_LANDING_CONTEXT_INVALID");
  }
  if (!FreshPromotionSnapshot(input)) {
    return PromotionFailure("HOP_LANDING_STATE_STALE");
  }
  if (continuation.route_cursor() >= continuation.certified_hops().size()) {
    return PromotionFailure("HOPPER_ROUTE_CONTINUATION_INVALID");
  }
  const CertifiedHopPreview &completed =
      continuation.certified_hops()[continuation.route_cursor()];
  if (*execution->active_plan_id != continuation.current_reference_plan_id() ||
      *execution->active_segment_id != completed.segment_id) {
    return PromotionFailure("HOP_EXECUTION_ID_MISMATCH");
  }
  if (!std::isfinite(input.velocity_uncertainty_mps) ||
      input.velocity_uncertainty_mps < 0.0 ||
      Norm(state->velocity.linear_mps) >
          input.velocity_uncertainty_mps + kTolerance ||
      Norm(state->velocity.angular_radps) >
          capability->maximum_initial_angular_speed_radps + kTolerance) {
    return PromotionFailure("HOP_LANDING_NOT_STABLE");
  }
  if (input.position_uncertainty_m >
          completed.position_uncertainty_m + kTolerance ||
      input.velocity_uncertainty_mps >
          completed.velocity_uncertainty_mps + kTolerance) {
    return PromotionFailure("HOP_UNCERTAINTY_EXCEEDS_CERTIFICATE");
  }
  const auto pose_map = TransformPose(state->pose, input.world.map_from_odom,
                                      TransformDirection::kChildToParent);
  if (!pose_map.has_value()) {
    return PromotionFailure("FRAME_TRANSFORM_INVALID");
  }
  if (!PointInPolygon(Vec2{pose_map->position_m.x, pose_map->position_m.y},
                      completed.promotion_region_map)) {
    return PromotionFailure("HOP_LANDING_DEVIATION_REPLAN_REQUIRED");
  }
  const std::size_t next_cursor = continuation.route_cursor() + 1U;
  if (next_cursor >= continuation.certified_hops().size()) {
    return PromotionFailure("HOP_ROUTE_COMPLETE");
  }
  return HopperHopPromotionResult{
      .hop = continuation.certified_hops()[next_cursor],
      .route_cursor = next_cursor,
      .reason_code = "HOPPER_HOP_PROMOTED",
  };
}

}  // namespace hierarchical
}  // namespace lunar::planning

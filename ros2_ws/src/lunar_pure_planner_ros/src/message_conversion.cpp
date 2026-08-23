#include "lunar_pure_planner_ros/message_conversion.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>

namespace lunar::pure_planner_ros {
namespace {

using Action = lunar_planning_msgs::action::PlanMotion;
using lunar::pure_planning::Quaternion;
using lunar::pure_planning::RigidTransform;
using lunar::pure_planning::Vec3;

[[nodiscard]] bool Finite(const double value) noexcept {
  return std::isfinite(value);
}

[[nodiscard]] bool Finite(const Vec3& value) noexcept {
  return Finite(value.x) && Finite(value.y) && Finite(value.z);
}

[[nodiscard]] std::optional<Quaternion> Normalize(
    const Quaternion& value) noexcept {
  if (!Finite(value.w) || !Finite(value.x) || !Finite(value.y) ||
      !Finite(value.z)) {
    return std::nullopt;
  }
  const double squared_norm = value.w * value.w + value.x * value.x +
      value.y * value.y + value.z * value.z;
  if (!Finite(squared_norm) || squared_norm <= 1.0e-24) {
    return std::nullopt;
  }
  const double inverse_norm = 1.0 / std::sqrt(squared_norm);
  return Quaternion{.w = value.w * inverse_norm, .x = value.x * inverse_norm,
                    .y = value.y * inverse_norm, .z = value.z * inverse_norm};
}

[[nodiscard]] Vec3 Rotate(const Quaternion& rotation, const Vec3& point) noexcept {
  const Vec3 first{
      .x = rotation.y * point.z - rotation.z * point.y,
      .y = rotation.z * point.x - rotation.x * point.z,
      .z = rotation.x * point.y - rotation.y * point.x,
  };
  const Vec3 second{
      .x = rotation.y * first.z - rotation.z * first.y,
      .y = rotation.z * first.x - rotation.x * first.z,
      .z = rotation.x * first.y - rotation.y * first.x,
  };
  return {.x = point.x + 2.0 * (rotation.w * first.x + second.x),
          .y = point.y + 2.0 * (rotation.w * first.y + second.y),
          .z = point.z + 2.0 * (rotation.w * first.z + second.z)};
}

[[nodiscard]] std::optional<Quaternion> ValidMapFromOdomRotation(
    const RigidTransform& map_from_odom) noexcept {
  if (map_from_odom.parent_frame != "map" ||
      map_from_odom.child_frame != "odom" || !Finite(map_from_odom.translation_m)) {
    return std::nullopt;
  }
  return Normalize(map_from_odom.rotation);
}

[[nodiscard]] std::optional<Vec3> TransformPlanarPoint(
    const Vec3& point_odom, const RigidTransform& map_from_odom) noexcept {
  const auto rotation = ValidMapFromOdomRotation(map_from_odom);
  if (!rotation) {
    return std::nullopt;
  }
  const Vec3 rotated = Rotate(*rotation, point_odom);
  const Vec3 mapped{.x = rotated.x + map_from_odom.translation_m.x,
                    .y = rotated.y + map_from_odom.translation_m.y,
                    .z = 0.0};
  return Finite(mapped) ? std::optional<Vec3>{mapped} : std::nullopt;
}

[[nodiscard]] std::optional<double> MapYaw(
    const RigidTransform& map_from_odom) noexcept {
  const auto rotation = ValidMapFromOdomRotation(map_from_odom);
  if (!rotation) {
    return std::nullopt;
  }
  const double sine = 2.0 * (rotation->w * rotation->z + rotation->x * rotation->y);
  const double cosine = 1.0 - 2.0 * (rotation->y * rotation->y + rotation->z * rotation->z);
  const double yaw = std::atan2(sine, cosine);
  return Finite(yaw) ? std::optional<double>{yaw} : std::nullopt;
}

[[nodiscard]] double NormalizeYaw(const double yaw) noexcept {
  return std::atan2(std::sin(yaw), std::cos(yaw));
}

[[nodiscard]] builtin_interfaces::msg::Time RosTime(
    const lunar::pure_planning::TimePoint time) noexcept {
  constexpr std::int64_t kNanosecondsPerSecond = 1'000'000'000LL;
  constexpr std::int64_t kMinSeconds = std::numeric_limits<std::int32_t>::min();
  constexpr std::int64_t kMaxSeconds = std::numeric_limits<std::int32_t>::max();
  std::int64_t seconds = time.nanoseconds_since_epoch / kNanosecondsPerSecond;
  std::int64_t nanoseconds = time.nanoseconds_since_epoch % kNanosecondsPerSecond;
  if (nanoseconds < 0) {
    --seconds;
    nanoseconds += kNanosecondsPerSecond;
  }
  builtin_interfaces::msg::Time converted;
  converted.sec =
      static_cast<std::int32_t>(std::clamp(seconds, kMinSeconds, kMaxSeconds));
  converted.nanosec = static_cast<std::uint32_t>(nanoseconds);
  return converted;
}

[[nodiscard]] geometry_msgs::msg::Pose RosPose(
    const lunar::pure_planning::Pose3& pose) {
  geometry_msgs::msg::Pose converted;
  converted.position.x = pose.position_m.x;
  converted.position.y = pose.position_m.y;
  converted.position.z = pose.position_m.z;
  converted.orientation.w = pose.orientation.w;
  converted.orientation.x = pose.orientation.x;
  converted.orientation.y = pose.orientation.y;
  converted.orientation.z = pose.orientation.z;
  return converted;
}

[[nodiscard]] geometry_msgs::msg::Transform RosTransform(
    const lunar::pure_planning::Pose3& pose) {
  geometry_msgs::msg::Transform converted;
  converted.translation.x = pose.position_m.x;
  converted.translation.y = pose.position_m.y;
  converted.translation.z = pose.position_m.z;
  converted.rotation.w = pose.orientation.w;
  converted.rotation.x = pose.orientation.x;
  converted.rotation.y = pose.orientation.y;
  converted.rotation.z = pose.orientation.z;
  return converted;
}

[[nodiscard]] geometry_msgs::msg::Twist RosTwist(
    const lunar::pure_planning::Twist3& twist) {
  geometry_msgs::msg::Twist converted;
  converted.linear.x = twist.linear_mps.x;
  converted.linear.y = twist.linear_mps.y;
  converted.linear.z = twist.linear_mps.z;
  converted.angular.x = twist.angular_radps.x;
  converted.angular.y = twist.angular_radps.y;
  converted.angular.z = twist.angular_radps.z;
  return converted;
}

void SetReference(const lunar::pure_planning::MotionReference& source,
                  lunar_planning_msgs::msg::MotionReference& target) {
  const auto input_time = RosTime(source.input_time);
  target.header.frame_id = "map";
  target.header.stamp = input_time;
  target.plan_id = source.plan_id;
  target.input_time = input_time;
  target.path_preview.header = target.header;
  for (const auto& pose : source.preview.poses_map) {
    geometry_msgs::msg::PoseStamped converted;
    converted.header = target.header;
    converted.pose = RosPose(pose);
    target.path_preview.poses.push_back(std::move(converted));
  }
  std::visit(
      [&source, &target, &input_time](const auto& data) {
        using Data = std::decay_t<decltype(data)>;
        if constexpr (std::is_same_v<Data, lunar::pure_planning::TrajectoryReference>) {
          target.platform_type = source.platform_type == lunar::pure_planning::PlatformType::kWheeled
              ? target.WHEELED : target.LEGGED;
          target.trajectory.header = target.header;
          target.trajectory.joint_names = {"base_link"};
          for (const auto& point : data.points) {
            trajectory_msgs::msg::MultiDOFJointTrajectoryPoint converted;
            converted.transforms.push_back(RosTransform(point.pose));
            converted.velocities.push_back(RosTwist(point.velocity));
            const auto nanoseconds = point.time_from_start.count();
            converted.time_from_start.sec = static_cast<std::int32_t>(nanoseconds / 1'000'000'000LL);
            converted.time_from_start.nanosec = static_cast<std::uint32_t>(nanoseconds % 1'000'000'000LL);
            target.trajectory.points.push_back(std::move(converted));
          }
        } else {
          target.platform_type = target.HOPPER;
          for (const auto& segment : data.segments) {
            lunar_planning_msgs::msg::HopSegment converted;
            converted.header.frame_id = "map";
            converted.header.stamp = input_time;
            converted.segment_id = segment.segment_id;
            converted.launch_pose = RosPose(segment.launch_pose);
            const auto nanoseconds = segment.flight_time.count();
            converted.flight_time.sec = static_cast<std::int32_t>(nanoseconds / 1'000'000'000LL);
            converted.flight_time.nanosec = static_cast<std::uint32_t>(nanoseconds % 1'000'000'000LL);
            converted.launch_velocity.x = segment.launch_velocity_mps.x;
            converted.launch_velocity.y = segment.launch_velocity_mps.y;
            converted.launch_velocity.z = segment.launch_velocity_mps.z;
            converted.flight_tube_radius_m = segment.flight_tube_radius_m;
            converted.nominal_landing_point.x = segment.nominal_landing_point_m.x;
            converted.nominal_landing_point.y = segment.nominal_landing_point_m.y;
            converted.nominal_landing_point.z = segment.nominal_landing_point_m.z;
            converted.required_delta_v_mps = segment.required_delta_v_mps;
            converted.available_delta_v_mps = segment.available_delta_v_mps;
            converted.capability_version = segment.capability_version;
            converted.global_map_generation = segment.global_map_generation;
            converted.local_map_generation = segment.local_map_generation;
            for (const auto& point : segment.landing_region_boundary_m) {
              geometry_msgs::msg::Point32 boundary;
              boundary.x = static_cast<float>(point.x);
              boundary.y = static_cast<float>(point.y);
              boundary.z = static_cast<float>(point.z);
              converted.landing_region.points.push_back(std::move(boundary));
            }
            target.hops.push_back(std::move(converted));
          }
        }
      }, source.data);
}

void SetFailure(const lunar::pure_planning::PlanningStatus status,
                Action::Result& result) {
  result.execution_directive = Action::Result::NO_SAFE_REFERENCE;
  result.has_reference = false;
  switch (status) {
    case lunar::pure_planning::PlanningStatus::kInvalidInput:
      result.planning_outcome = Action::Result::INVALID_REQUEST;
      result.reason_code = "INVALID_INPUT";
      return;
    case lunar::pure_planning::PlanningStatus::kGoalOutsideLocalMap:
      result.planning_outcome = Action::Result::GOAL_INFEASIBLE;
      result.reason_code = "GOAL_OUTSIDE_LOCAL_MAP";
      return;
    case lunar::pure_planning::PlanningStatus::kNoPath:
      result.planning_outcome = Action::Result::GOAL_INFEASIBLE;
      result.reason_code = "NO_PATH";
      return;
    case lunar::pure_planning::PlanningStatus::kTimedOut:
      result.planning_outcome = Action::Result::RESOURCE_EXHAUSTED;
      result.reason_code = "TIMEOUT";
      return;
    case lunar::pure_planning::PlanningStatus::kCanceled:
      result.planning_outcome = Action::Result::CANCELED;
      result.reason_code = "REQUEST_CANCELED";
      return;
    case lunar::pure_planning::PlanningStatus::kPlannerError:
    case lunar::pure_planning::PlanningStatus::kSuccess:
      result.planning_outcome = Action::Result::NUMERICAL_FAILURE;
      result.reason_code = "PLANNER_ERROR";
      return;
  }
}

void SetLatencyWarnings(
    const lunar::pure_planning::RequestLatencyClass latency_class,
    lunar_planning_msgs::msg::PlannerDiagnostics& diagnostics) {
  if (latency_class !=
      lunar::pure_planning::RequestLatencyClass::kTargetMet) {
    diagnostics.warning_codes.emplace_back("TARGET_MISSED");
  }
  if (latency_class == lunar::pure_planning::RequestLatencyClass::kSlaMissed ||
      latency_class ==
          lunar::pure_planning::RequestLatencyClass::kHardTimeout) {
    diagnostics.warning_codes.emplace_back("PLANNING_SLA_MISSED");
  }
}

}  // namespace

GoalConversionResult ConvertGoal(const Action::Goal& request,
                                 const RigidTransform& map_from_odom) {
  const auto& message = request.goal;
  if ((request.environment_mode != request.LUNAR_SURFACE &&
       request.environment_mode != request.LAVA_TUBE) ||
      message.goal_type != message.POINT || message.goal_id.empty() ||
      (message.header.frame_id != "map" && message.header.frame_id != "odom") ||
      !Finite(message.point.x) || !Finite(message.point.y) ||
      !Finite(message.position_tolerance_m) || message.position_tolerance_m < 0.0 ||
      !Finite(message.yaw_tolerance_rad) || message.yaw_tolerance_rad < 0.0 ||
      (message.has_yaw_constraint && !Finite(message.yaw_rad))) {
    return {.goal = std::nullopt, .reason_code = "INVALID_INPUT"};
  }
  Vec3 point{.x = message.point.x, .y = message.point.y, .z = 0.0};
  std::optional<double> yaw = message.has_yaw_constraint
      ? std::optional<double>{message.yaw_rad} : std::nullopt;
  if (message.header.frame_id == "odom") {
    if (!ValidMapFromOdomRotation(map_from_odom)) {
      return {.goal = std::nullopt, .reason_code = "INVALID_INPUT"};
    }
    const auto transformed = TransformPlanarPoint(point, map_from_odom);
    if (!transformed) {
      return {.goal = std::nullopt, .reason_code = "INVALID_INPUT"};
    }
    point = *transformed;
    if (yaw) {
      const auto map_yaw = MapYaw(map_from_odom);
      if (!map_yaw) {
        return {.goal = std::nullopt, .reason_code = "INVALID_INPUT"};
      }
      yaw = NormalizeYaw(*yaw + *map_yaw);
    }
  }
  return {.goal = lunar::pure_planning::GoalRegion{
              .goal_id = message.goal_id,
              .target = lunar::pure_planning::PointGoal{
                  .position_m = point, .tolerance_m = message.position_tolerance_m},
              .yaw_rad = yaw,
              .yaw_tolerance_rad = message.yaw_tolerance_rad},
          .reason_code = {}};
}

Action::Result ConvertResult(const lunar::pure_planning::PlanningResult& source,
                             const std::uint64_t mission_revision) {
  Action::Result result;
  result.mission_revision = mission_revision;
  result.diagnostics.planner_name = "lunar_pure_planner";
  result.diagnostics.elapsed_s =
      std::chrono::duration<double>(source.timing.total_elapsed).count();
  result.diagnostics.expanded_states = source.expanded_states;
  result.diagnostics.has_best_cost = source.best_cost.has_value();
  result.diagnostics.best_cost = source.best_cost.value_or(0.0);
  SetLatencyWarnings(
      lunar::pure_planning::ClassifyRequestLatency(
          source.timing.total_elapsed),
      result.diagnostics);
  if (source.status == lunar::pure_planning::PlanningStatus::kSuccess &&
      source.reference.has_value()) {
    result.planning_outcome = Action::Result::NEW_REFERENCE_AVAILABLE;
    result.execution_directive = Action::Result::ACTIVATE_NEW_REFERENCE;
    result.reason_code = source.reason_code.empty() ? "PLAN_FOUND" : source.reason_code;
    result.has_reference = true;
    SetReference(*source.reference, result.reference);
    return result;
  }
  SetFailure(source.status, result);
  result.reference = lunar_planning_msgs::msg::MotionReference{};
  return result;
}

}  // namespace lunar::pure_planner_ros

#include "lunar_planner_ros/message_conversion.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>

#include <builtin_interfaces/msg/duration.hpp>
#include <builtin_interfaces/msg/time.hpp>
#include <geometry_msgs/msg/point32.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <trajectory_msgs/msg/multi_dof_joint_trajectory_point.hpp>

namespace lunar::planning::ros {
namespace {

constexpr std::int64_t kNanosecondsPerSecond = 1'000'000'000LL;

[[nodiscard]] bool Finite(const double value) noexcept {
  return std::isfinite(value);
}

[[nodiscard]] bool Finite(const lunar::planning::Vec3& value) noexcept {
  return Finite(value.x) && Finite(value.y) && Finite(value.z);
}

[[nodiscard]] bool Finite(const lunar::planning::Quaternion& value) noexcept {
  return Finite(value.w) && Finite(value.x) &&
      Finite(value.y) && Finite(value.z);
}

[[nodiscard]] bool Finite(const lunar::planning::Pose3& value) noexcept {
  const double quaternion_norm = std::sqrt(
      value.orientation.w * value.orientation.w +
      value.orientation.x * value.orientation.x +
      value.orientation.y * value.orientation.y +
      value.orientation.z * value.orientation.z);
  return Finite(value.position_m) && Finite(value.orientation) &&
      Finite(quaternion_norm) && quaternion_norm > 1.0e-12;
}

[[nodiscard]] bool Finite(const lunar::planning::Twist3& value) noexcept {
  return Finite(value.linear_mps) && Finite(value.angular_radps);
}

[[nodiscard]] bool FinitePoint(const geometry_msgs::msg::Point& point) noexcept {
  return Finite(point.x) && Finite(point.y) && Finite(point.z);
}

[[nodiscard]] bool FinitePoint(
    const geometry_msgs::msg::Point32& point) noexcept {
  return Finite(point.x) && Finite(point.y) && Finite(point.z);
}

[[nodiscard]] GoalMessageConversion GoalFailure(std::string reason_code) {
  return GoalMessageConversion{
      .goal = std::nullopt,
      .reason_code = std::move(reason_code),
  };
}

[[nodiscard]] ActionResultConversion ResultFailure(std::string reason_code) {
  return ActionResultConversion{
      .result = std::nullopt,
      .reason_code = std::move(reason_code),
  };
}

[[nodiscard]] std::optional<builtin_interfaces::msg::Time> RosTime(
    const lunar::planning::TimePoint time) noexcept {
  if (time.nanoseconds_since_epoch < 0) {
    return std::nullopt;
  }
  const std::int64_t seconds =
      time.nanoseconds_since_epoch / kNanosecondsPerSecond;
  if (seconds > std::numeric_limits<std::int32_t>::max()) {
    return std::nullopt;
  }
  builtin_interfaces::msg::Time result;
  result.sec = static_cast<std::int32_t>(seconds);
  result.nanosec = static_cast<std::uint32_t>(
      time.nanoseconds_since_epoch % kNanosecondsPerSecond);
  return result;
}

[[nodiscard]] std::optional<builtin_interfaces::msg::Duration> RosDuration(
    const std::chrono::nanoseconds duration) noexcept {
  if (duration.count() < 0) {
    return std::nullopt;
  }
  const std::int64_t seconds = duration.count() / kNanosecondsPerSecond;
  if (seconds > std::numeric_limits<std::int32_t>::max()) {
    return std::nullopt;
  }
  builtin_interfaces::msg::Duration result;
  result.sec = static_cast<std::int32_t>(seconds);
  result.nanosec = static_cast<std::uint32_t>(
      duration.count() % kNanosecondsPerSecond);
  return result;
}

[[nodiscard]] std::optional<builtin_interfaces::msg::Time> AddTime(
    const lunar::planning::TimePoint base,
    const std::chrono::nanoseconds offset) noexcept {
  if (base.nanoseconds_since_epoch < 0 || offset.count() < 0 ||
      offset.count() >
          std::numeric_limits<std::int64_t>::max() -
              base.nanoseconds_since_epoch) {
    return std::nullopt;
  }
  return RosTime(lunar::planning::TimePoint{
      base.nanoseconds_since_epoch + offset.count(),
  });
}

[[nodiscard]] geometry_msgs::msg::Pose RosPose(
    const lunar::planning::Pose3& pose) {
  geometry_msgs::msg::Pose result;
  result.position.x = pose.position_m.x;
  result.position.y = pose.position_m.y;
  result.position.z = pose.position_m.z;
  result.orientation.w = pose.orientation.w;
  result.orientation.x = pose.orientation.x;
  result.orientation.y = pose.orientation.y;
  result.orientation.z = pose.orientation.z;
  return result;
}

[[nodiscard]] geometry_msgs::msg::Transform RosTransform(
    const lunar::planning::Pose3& pose) {
  geometry_msgs::msg::Transform result;
  result.translation.x = pose.position_m.x;
  result.translation.y = pose.position_m.y;
  result.translation.z = pose.position_m.z;
  result.rotation.w = pose.orientation.w;
  result.rotation.x = pose.orientation.x;
  result.rotation.y = pose.orientation.y;
  result.rotation.z = pose.orientation.z;
  return result;
}

[[nodiscard]] geometry_msgs::msg::Twist RosTwist(
    const lunar::planning::Twist3& twist) {
  geometry_msgs::msg::Twist result;
  result.linear.x = twist.linear_mps.x;
  result.linear.y = twist.linear_mps.y;
  result.linear.z = twist.linear_mps.z;
  result.angular.x = twist.angular_radps.x;
  result.angular.y = twist.angular_radps.y;
  result.angular.z = twist.angular_radps.z;
  return result;
}

[[nodiscard]] bool ReferenceAllowed(
    const lunar::planning::PlannerOutput& output) noexcept {
  return output.outcome ==
             lunar::planning::PlanningOutcome::kNewReferenceAvailable ||
      output.outcome ==
             lunar::planning::PlanningOutcome::kSafeFrontierReferenceAvailable ||
      output.directive ==
             lunar::planning::ExecutionDirective::kContinueActiveReference ||
      output.directive ==
             lunar::planning::ExecutionDirective::kContinueCommittedHop;
}

[[nodiscard]] std::optional<std::string> ConvertTrajectory(
    const lunar::planning::MotionReference& reference,
    const lunar::planning::TrajectoryReference& trajectory,
    const builtin_interfaces::msg::Time& input_time,
    const std::string& frame,
    lunar_planning_msgs::msg::MotionReference& message) {
  const bool wheeled =
      reference.platform_type == lunar::planning::PlatformType::kWheeled;
  const bool legged =
      reference.platform_type == lunar::planning::PlatformType::kLegged;
  if ((!wheeled && !legged) ||
      (wheeled && trajectory.semantics !=
          lunar::planning::TrajectorySemantics::kWheeledBase) ||
      (legged && trajectory.semantics !=
          lunar::planning::TrajectorySemantics::kLeggedBodyReference)) {
    return "REFERENCE_PLATFORM_DATA_MISMATCH";
  }
  if (trajectory.points.empty()) {
    return "REFERENCE_TRAJECTORY_EMPTY";
  }

  message.platform_type = wheeled ? message.WHEELED : message.LEGGED;
  message.path_preview.header.frame_id = frame;
  message.path_preview.header.stamp = input_time;
  message.trajectory.header.frame_id = frame;
  message.trajectory.header.stamp = input_time;
  message.trajectory.joint_names = {"base_link"};

  std::chrono::nanoseconds previous_time{-1};
  for (const auto& point : trajectory.points) {
    if (point.time_from_start < std::chrono::nanoseconds::zero() ||
        point.time_from_start < previous_time || !Finite(point.pose) ||
        !Finite(point.velocity)) {
      return "REFERENCE_TRAJECTORY_POINT_INVALID";
    }
    const auto relative_time = RosDuration(point.time_from_start);
    const auto pose_time = AddTime(reference.input_time, point.time_from_start);
    if (!relative_time || !pose_time) {
      return "REFERENCE_TIME_INVALID";
    }

    geometry_msgs::msg::PoseStamped preview;
    preview.header.frame_id = frame;
    preview.header.stamp = *pose_time;
    preview.pose = RosPose(point.pose);
    message.path_preview.poses.push_back(std::move(preview));

    trajectory_msgs::msg::MultiDOFJointTrajectoryPoint trajectory_point;
    trajectory_point.transforms.push_back(RosTransform(point.pose));
    trajectory_point.velocities.push_back(RosTwist(point.velocity));
    trajectory_point.time_from_start = *relative_time;
    message.trajectory.points.push_back(std::move(trajectory_point));
    previous_time = point.time_from_start;
  }
  return std::nullopt;
}

[[nodiscard]] bool FitsPoint32(const lunar::planning::Vec3& point) noexcept {
  constexpr double kMaximum = std::numeric_limits<float>::max();
  return Finite(point) && std::abs(point.x) <= kMaximum &&
      std::abs(point.y) <= kMaximum && std::abs(point.z) <= kMaximum;
}

[[nodiscard]] std::optional<std::string> ConvertHops(
    const lunar::planning::MotionReference& reference,
    const lunar::planning::HopReference& hops,
    const builtin_interfaces::msg::Time& input_time,
    const std::string& frame,
    lunar_planning_msgs::msg::MotionReference& message) {
  if (reference.platform_type != lunar::planning::PlatformType::kHopper) {
    return "REFERENCE_PLATFORM_DATA_MISMATCH";
  }
  if (hops.segments.empty()) {
    return "REFERENCE_HOPS_EMPTY";
  }
  message.platform_type = message.HOPPER;
  message.path_preview.header.frame_id = frame;
  message.path_preview.header.stamp = input_time;

  std::chrono::nanoseconds elapsed{};
  for (const auto& segment : hops.segments) {
    if (segment.segment_id.empty() || !Finite(segment.launch_pose) ||
        segment.landing_region_boundary_m.size() < 3U ||
        segment.flight_time <= std::chrono::nanoseconds::zero() ||
        !Finite(segment.launch_velocity_mps) ||
        !Finite(segment.flight_tube_radius_m) ||
        segment.flight_tube_radius_m <= 0.0) {
      return "REFERENCE_HOP_INVALID";
    }
    if (segment.flight_time.count() >
        std::numeric_limits<std::int64_t>::max() - elapsed.count()) {
      return "REFERENCE_TIME_INVALID";
    }
    const std::chrono::nanoseconds landing_offset =
        elapsed + segment.flight_time;
    const auto start_time = AddTime(reference.input_time, elapsed);
    const auto landing_time = AddTime(reference.input_time, landing_offset);
    const auto flight_time = RosDuration(segment.flight_time);
    if (!start_time || !landing_time || !flight_time) {
      return "REFERENCE_TIME_INVALID";
    }

    lunar_planning_msgs::msg::HopSegment hop;
    hop.header.frame_id = frame;
    hop.header.stamp = *start_time;
    hop.segment_id = segment.segment_id;
    hop.launch_pose = RosPose(segment.launch_pose);
    hop.flight_time = *flight_time;
    hop.launch_velocity.x = segment.launch_velocity_mps.x;
    hop.launch_velocity.y = segment.launch_velocity_mps.y;
    hop.launch_velocity.z = segment.launch_velocity_mps.z;
    hop.flight_tube_radius_m = segment.flight_tube_radius_m;

    lunar::planning::Vec3 centroid{};
    for (const auto& point : segment.landing_region_boundary_m) {
      if (!FitsPoint32(point)) {
        return "REFERENCE_HOP_INVALID";
      }
      geometry_msgs::msg::Point32 converted;
      converted.x = static_cast<float>(point.x);
      converted.y = static_cast<float>(point.y);
      converted.z = static_cast<float>(point.z);
      hop.landing_region.points.push_back(converted);
      centroid.x += point.x;
      centroid.y += point.y;
      centroid.z += point.z;
    }
    const double count =
        static_cast<double>(segment.landing_region_boundary_m.size());
    centroid.x /= count;
    centroid.y /= count;
    centroid.z /= count;

    geometry_msgs::msg::PoseStamped launch_preview;
    launch_preview.header = hop.header;
    launch_preview.pose = hop.launch_pose;
    message.path_preview.poses.push_back(std::move(launch_preview));
    geometry_msgs::msg::PoseStamped landing_preview;
    landing_preview.header.frame_id = frame;
    landing_preview.header.stamp = *landing_time;
    landing_preview.pose.position.x = centroid.x;
    landing_preview.pose.position.y = centroid.y;
    landing_preview.pose.position.z = centroid.z;
    landing_preview.pose.orientation.w = 1.0;
    message.path_preview.poses.push_back(std::move(landing_preview));

    message.hops.push_back(std::move(hop));
    elapsed = landing_offset;
  }
  return std::nullopt;
}

}  // namespace

GoalMessageConversion ConvertGoalMessage(
    const lunar_planning_msgs::msg::GoalRegion& message) {
  if (message.goal_id.empty()) {
    return GoalFailure("GOAL_ID_EMPTY");
  }
  if (message.goal_type != message.POINT &&
      message.goal_type != message.PLANAR_REGION) {
    return GoalFailure("GOAL_TYPE_INVALID");
  }
  if (!Finite(message.position_tolerance_m) ||
      (message.goal_type == message.POINT &&
       message.position_tolerance_m <= 0.0) ||
      (message.goal_type == message.PLANAR_REGION &&
       message.position_tolerance_m < 0.0) ||
      !Finite(message.yaw_tolerance_rad) || message.yaw_tolerance_rad < 0.0 ||
      (message.has_yaw_constraint && !Finite(message.yaw_rad))) {
    return GoalFailure("GOAL_TOLERANCE_INVALID");
  }

  lunar::planning::GoalRegion goal;
  goal.goal_id = message.goal_id;
  goal.yaw_rad = message.has_yaw_constraint
      ? std::optional<double>{message.yaw_rad} : std::nullopt;
  goal.yaw_tolerance_rad = message.yaw_tolerance_rad;
  if (message.goal_type == message.POINT) {
    if (!FinitePoint(message.point)) {
      return GoalFailure("GOAL_POINT_INVALID");
    }
    goal.target = lunar::planning::PointGoal{
        .position_m = {message.point.x, message.point.y, message.point.z},
        .tolerance_m = message.position_tolerance_m,
    };
  } else {
    if (message.planar_region.points.size() < 3U) {
      return GoalFailure("GOAL_REGION_INVALID");
    }
    lunar::planning::PlanarRegionGoal target{
        .boundary_m = {},
        .normal_tolerance_m = message.position_tolerance_m,
    };
    target.boundary_m.reserve(message.planar_region.points.size());
    for (const auto& point : message.planar_region.points) {
      if (!FinitePoint(point)) {
        return GoalFailure("GOAL_REGION_INVALID");
      }
      target.boundary_m.push_back({point.x, point.y, point.z});
    }
    goal.target = std::move(target);
  }
  return GoalMessageConversion{
      .goal = std::move(goal),
      .reason_code = {},
  };
}

ActionResultConversion ConvertPlannerOutput(
    const lunar::planning::PlannerOutput& output,
    const PlannerResultContext& context) {
  const auto global_stamp = RosTime(context.global_map_stamp);
  const auto local_stamp = RosTime(context.local_map_stamp);
  const auto state_stamp = RosTime(context.state_stamp);
  if (!global_stamp || !local_stamp || !state_stamp ||
      context.global_map_stamp.nanoseconds_since_epoch <= 0 ||
      context.local_map_stamp.nanoseconds_since_epoch <= 0 ||
      context.state_stamp.nanoseconds_since_epoch <= 0 ||
      context.planning_frame != "odom") {
    return ResultFailure("RESULT_CONTEXT_INVALID");
  }
  if (static_cast<std::uint8_t>(output.outcome) >
          static_cast<std::uint8_t>(
              lunar::planning::PlanningOutcome::kCanceled) ||
      static_cast<std::uint8_t>(output.directive) >
          static_cast<std::uint8_t>(
              lunar::planning::ExecutionDirective::kNoSafeReference)) {
    return ResultFailure("RESULT_ENUM_INVALID");
  }
  if (output.reason_code.empty() || output.diagnostics.elapsed.count() < 0 ||
      (output.diagnostics.best_cost &&
       !Finite(*output.diagnostics.best_cost))) {
    return ResultFailure("RESULT_DIAGNOSTICS_INVALID");
  }

  lunar_planning_msgs::action::PlanMotion::Result result;
  result.planning_outcome = static_cast<std::uint8_t>(output.outcome);
  result.execution_directive = static_cast<std::uint8_t>(output.directive);
  result.reason_code = output.reason_code;
  result.global_map_stamp = *global_stamp;
  result.local_map_stamp = *local_stamp;
  result.state_stamp = *state_stamp;
  result.mission_revision = context.mission_revision;
  result.diagnostics.planner_name = output.diagnostics.planner_name;
  result.diagnostics.elapsed_s =
      std::chrono::duration<double>(output.diagnostics.elapsed).count();
  result.diagnostics.expanded_states = output.diagnostics.expanded_states;
  result.diagnostics.has_best_cost = output.diagnostics.best_cost.has_value();
  result.diagnostics.best_cost = output.diagnostics.best_cost.value_or(0.0);
  result.diagnostics.warning_codes = output.diagnostics.warning_codes;

  if (!output.reference) {
    if (ReferenceAllowed(output)) {
      return ResultFailure("REFERENCE_REQUIRED_BY_OUTCOME");
    }
    result.has_reference = false;
    result.reference = lunar_planning_msgs::msg::MotionReference{};
    return ActionResultConversion{
        .result = std::move(result),
        .reason_code = {},
    };
  }
  if (!ReferenceAllowed(output)) {
    return ResultFailure("REFERENCE_FORBIDDEN_BY_OUTCOME");
  }
  if (output.reference->plan_id.empty()) {
    return ResultFailure("REFERENCE_PLAN_ID_EMPTY");
  }
  const auto input_time = RosTime(output.reference->input_time);
  if (!input_time || output.reference->input_time.nanoseconds_since_epoch <= 0) {
    return ResultFailure("REFERENCE_TIME_INVALID");
  }

  lunar_planning_msgs::msg::MotionReference reference;
  reference.header.frame_id = context.planning_frame;
  reference.header.stamp = *input_time;
  reference.plan_id = output.reference->plan_id;
  const auto error = std::visit(
      [&](const auto& data) -> std::optional<std::string> {
        using Data = std::decay_t<decltype(data)>;
        if constexpr (std::is_same_v<Data, lunar::planning::TrajectoryReference>) {
          return ConvertTrajectory(
              *output.reference, data, *input_time,
              context.planning_frame, reference);
        } else {
          return ConvertHops(
              *output.reference, data, *input_time,
              context.planning_frame, reference);
        }
      },
      output.reference->data);
  if (error) {
    return ResultFailure(*error);
  }

  result.has_reference = true;
  result.reference = std::move(reference);
  return ActionResultConversion{
      .result = std::move(result),
      .reason_code = {},
  };
}

}  // namespace lunar::planning::ros

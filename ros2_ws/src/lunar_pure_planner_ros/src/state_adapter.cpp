#include "lunar_pure_planner_ros/state_adapter.hpp"

#include <cmath>
#include <optional>
#include <string>

namespace lunar::pure_planner_ros {
namespace {

constexpr char kInvalidInput[] = "INVALID_INPUT";

[[nodiscard]] bool Finite(const double value) noexcept { return std::isfinite(value); }

[[nodiscard]] std::optional<lunar::pure_planning::Quaternion> Normalize(
    const geometry_msgs::msg::Quaternion& value) noexcept {
  if (!Finite(value.w) || !Finite(value.x) || !Finite(value.y) || !Finite(value.z)) {
    return std::nullopt;
  }
  const double norm = std::sqrt(value.w * value.w + value.x * value.x +
                                value.y * value.y + value.z * value.z);
  if (!Finite(norm) || norm <= 0.0) {
    return std::nullopt;
  }
  return lunar::pure_planning::Quaternion{
      .w = value.w / norm, .x = value.x / norm, .y = value.y / norm, .z = value.z / norm};
}

[[nodiscard]] std::optional<lunar::pure_planning::Pose3> Pose(
    const geometry_msgs::msg::Pose& value) noexcept {
  const auto rotation = Normalize(value.orientation);
  if (!rotation.has_value() || !Finite(value.position.x) || !Finite(value.position.y) ||
      !Finite(value.position.z)) {
    return std::nullopt;
  }
  return lunar::pure_planning::Pose3{
      .position_m = {.x = value.position.x, .y = value.position.y, .z = value.position.z},
      .orientation = *rotation};
}

[[nodiscard]] std::optional<lunar::pure_planning::Twist3> Twist(
    const geometry_msgs::msg::Twist& value) noexcept {
  const auto finite = [](const geometry_msgs::msg::Vector3& vector) {
    return Finite(vector.x) && Finite(vector.y) && Finite(vector.z);
  };
  if (!finite(value.linear) || !finite(value.angular)) {
    return std::nullopt;
  }
  return lunar::pure_planning::Twist3{
      .linear_mps = {.x = value.linear.x, .y = value.linear.y, .z = value.linear.z},
      .angular_radps = {.x = value.angular.x, .y = value.angular.y, .z = value.angular.z}};
}

[[nodiscard]] lunar::pure_planning::TimePoint Stamp(
    const builtin_interfaces::msg::Time& stamp) noexcept {
  return {.nanoseconds_since_epoch =
              static_cast<std::int64_t>(stamp.sec) * 1'000'000'000LL + stamp.nanosec};
}

[[nodiscard]] lunar::pure_planning::Quaternion Multiply(
    const lunar::pure_planning::Quaternion& left,
    const lunar::pure_planning::Quaternion& right) noexcept {
  return {.w = left.w * right.w - left.x * right.x - left.y * right.y - left.z * right.z,
          .x = left.w * right.x + left.x * right.w + left.y * right.z - left.z * right.y,
          .y = left.w * right.y - left.x * right.z + left.y * right.w + left.z * right.x,
          .z = left.w * right.z + left.x * right.y - left.y * right.x + left.z * right.w};
}

[[nodiscard]] lunar::pure_planning::Vec3 Rotate(
    const lunar::pure_planning::Quaternion& rotation,
    const lunar::pure_planning::Vec3& point) noexcept {
  const lunar::pure_planning::Quaternion vector{
      .w = 0.0, .x = point.x, .y = point.y, .z = point.z};
  const lunar::pure_planning::Quaternion conjugate{
      .w = rotation.w, .x = -rotation.x, .y = -rotation.y, .z = -rotation.z};
  const auto output = Multiply(Multiply(rotation, vector), conjugate);
  return {.x = output.x, .y = output.y, .z = output.z};
}

[[nodiscard]] const lunar::pure_planning::Pose3* StatePose(
    const lunar::pure_planning::PlatformState& state) noexcept {
  if (const auto* wheel = std::get_if<lunar::pure_planning::WheeledState>(&state)) {
    return &wheel->pose;
  }
  if (const auto* legged = std::get_if<lunar::pure_planning::LeggedState>(&state)) {
    return &legged->body_pose;
  }
  return &std::get<lunar::pure_planning::HopperState>(state).pose;
}

template <typename T>
[[nodiscard]] AdapterResult<T> Invalid() {
  return {.value = std::nullopt, .reason_code = kInvalidInput};
}

}  // namespace

AdapterResult<lunar::pure_planning::RigidTransform> AdaptDirectMapFromOdom(
    const geometry_msgs::msg::TransformStamped& transform) {
  const auto rotation = Normalize(transform.transform.rotation);
  if (transform.header.frame_id != "map" || transform.child_frame_id != "odom" ||
      !rotation.has_value() || !Finite(transform.transform.translation.x) ||
      !Finite(transform.transform.translation.y) || !Finite(transform.transform.translation.z)) {
    return Invalid<lunar::pure_planning::RigidTransform>();
  }
  return {.value = lunar::pure_planning::RigidTransform{
              .parent_frame = transform.header.frame_id,
              .child_frame = transform.child_frame_id,
              .stamp = Stamp(transform.header.stamp),
              .translation_m = {.x = transform.transform.translation.x,
                                .y = transform.transform.translation.y,
                                .z = transform.transform.translation.z},
              .rotation = *rotation},
          .reason_code = {}};
}

AdapterResult<lunar::pure_planning::PlatformState> AdaptOdometry(
    const nav_msgs::msg::Odometry& odometry,
    const lunar::pure_planning::PlatformType platform) {
  if (odometry.header.frame_id != "odom" || odometry.child_frame_id != "base_link") {
    return Invalid<lunar::pure_planning::PlatformState>();
  }
  const auto pose = Pose(odometry.pose.pose);
  const auto twist = Twist(odometry.twist.twist);
  if (!pose.has_value() || !twist.has_value()) {
    return Invalid<lunar::pure_planning::PlatformState>();
  }
  switch (platform) {
    case lunar::pure_planning::PlatformType::kWheeled:
      return {.value = lunar::pure_planning::WheeledState{.pose = *pose, .velocity = *twist},
              .reason_code = {}};
    case lunar::pure_planning::PlatformType::kLegged:
      return {.value = lunar::pure_planning::LeggedState{.body_pose = *pose, .body_velocity = *twist},
              .reason_code = {}};
    case lunar::pure_planning::PlatformType::kHopper:
      return {.value = lunar::pure_planning::HopperState{.pose = *pose, .velocity = *twist},
              .reason_code = {}};
  }
  return Invalid<lunar::pure_planning::PlatformState>();
}

AdapterResult<lunar::pure_planning::Pose3> ComposeMapFromOdomAndOdometry(
    const lunar::pure_planning::RigidTransform& map_from_odom,
    const lunar::pure_planning::PlatformState& odom_from_base) {
  const auto* pose = StatePose(odom_from_base);
  const auto translated = Rotate(map_from_odom.rotation, pose->position_m);
  const auto orientation = Multiply(map_from_odom.rotation, pose->orientation);
  const double norm = std::sqrt(orientation.w * orientation.w + orientation.x * orientation.x +
                                orientation.y * orientation.y + orientation.z * orientation.z);
  if (!Finite(translated.x) || !Finite(translated.y) || !Finite(translated.z) ||
      !Finite(norm) || norm <= 0.0) {
    return Invalid<lunar::pure_planning::Pose3>();
  }
  return {.value = lunar::pure_planning::Pose3{
              .position_m = {.x = map_from_odom.translation_m.x + translated.x,
                             .y = map_from_odom.translation_m.y + translated.y,
                             .z = map_from_odom.translation_m.z + translated.z},
              .orientation = {.w = orientation.w / norm, .x = orientation.x / norm,
                              .y = orientation.y / norm, .z = orientation.z / norm}},
          .reason_code = {}};
}

}  // namespace lunar::pure_planner_ros

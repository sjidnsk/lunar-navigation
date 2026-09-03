#include "lunar_incremental_navigation_ros/state_adapter.hpp"

#include <cmath>
#include <optional>

namespace lunar::incremental_navigation_ros {
namespace {

constexpr char kInvalidInput[] = "INVALID_INPUT";

[[nodiscard]] bool Finite(const double value) noexcept {
  return std::isfinite(value);
}

[[nodiscard]] std::optional<lunar::incremental_navigation::Quaternion> Normalize(
    const geometry_msgs::msg::Quaternion& value) noexcept {
  if (!Finite(value.w) || !Finite(value.x) || !Finite(value.y) ||
      !Finite(value.z)) {
    return std::nullopt;
  }
  const double norm = std::sqrt(value.w * value.w + value.x * value.x +
                                value.y * value.y + value.z * value.z);
  if (!Finite(norm) || norm <= 0.0) {
    return std::nullopt;
  }
  return lunar::incremental_navigation::Quaternion{
      .w = value.w / norm,
      .x = value.x / norm,
      .y = value.y / norm,
      .z = value.z / norm};
}

[[nodiscard]] lunar::incremental_navigation::Quaternion Multiply(
    const lunar::incremental_navigation::Quaternion& left,
    const lunar::incremental_navigation::Quaternion& right) noexcept {
  return {
      .w = left.w * right.w - left.x * right.x - left.y * right.y -
           left.z * right.z,
      .x = left.w * right.x + left.x * right.w + left.y * right.z -
           left.z * right.y,
      .y = left.w * right.y - left.x * right.z + left.y * right.w +
           left.z * right.x,
      .z = left.w * right.z + left.x * right.y - left.y * right.x +
           left.z * right.w};
}

[[nodiscard]] lunar::incremental_navigation::Vec3 Rotate(
    const lunar::incremental_navigation::Quaternion& rotation,
    const lunar::incremental_navigation::Vec3& point) noexcept {
  const lunar::incremental_navigation::Quaternion vector{
      .w = 0.0, .x = point.x, .y = point.y, .z = point.z};
  const lunar::incremental_navigation::Quaternion conjugate{
      .w = rotation.w,
      .x = -rotation.x,
      .y = -rotation.y,
      .z = -rotation.z};
  const auto output = Multiply(Multiply(rotation, vector), conjugate);
  return {.x = output.x, .y = output.y, .z = output.z};
}

template <typename T>
[[nodiscard]] AdapterResult<T> Invalid() {
  return {.value = std::nullopt, .reason_code = kInvalidInput};
}

}  // namespace

AdapterResult<lunar::incremental_navigation::RigidTransform>
AdaptDirectMapFromOdom(
    const geometry_msgs::msg::TransformStamped& transform) {
  const auto rotation = Normalize(transform.transform.rotation);
  if (transform.header.frame_id != "map" ||
      transform.child_frame_id != "odom" || !rotation ||
      !Finite(transform.transform.translation.x) ||
      !Finite(transform.transform.translation.y) ||
      !Finite(transform.transform.translation.z)) {
    return Invalid<lunar::incremental_navigation::RigidTransform>();
  }
  return {
      .value = lunar::incremental_navigation::RigidTransform{
          .parent_frame = "map",
          .child_frame = "odom",
          .stamp = {},
          .translation_m = {.x = transform.transform.translation.x,
                            .y = transform.transform.translation.y,
                            .z = transform.transform.translation.z},
          .rotation = *rotation},
      .reason_code = {}};
}

AdapterResult<lunar::incremental_navigation::StateInput> AdaptStateInput(
    const lunar::incremental_navigation::RigidTransform& map_from_odom,
    const nav_msgs::msg::Odometry& odometry) {
  if (map_from_odom.parent_frame != "map" ||
      map_from_odom.child_frame != "odom" ||
      odometry.header.frame_id != "odom" ||
      odometry.child_frame_id != "base_link" ||
      !Finite(odometry.pose.pose.position.x) ||
      !Finite(odometry.pose.pose.position.y) ||
      !Finite(odometry.pose.pose.position.z)) {
    return Invalid<lunar::incremental_navigation::StateInput>();
  }
  const auto orientation = Normalize(odometry.pose.pose.orientation);
  if (!orientation) {
    return Invalid<lunar::incremental_navigation::StateInput>();
  }
  const auto translated = Rotate(
      map_from_odom.rotation,
      {.x = odometry.pose.pose.position.x,
       .y = odometry.pose.pose.position.y,
       .z = odometry.pose.pose.position.z});
  const auto map_orientation = Multiply(map_from_odom.rotation, *orientation);
  const double map_norm = std::sqrt(
      map_orientation.w * map_orientation.w +
      map_orientation.x * map_orientation.x +
      map_orientation.y * map_orientation.y +
      map_orientation.z * map_orientation.z);
  if (!Finite(translated.x) || !Finite(translated.y) ||
      !Finite(translated.z) || !Finite(map_norm) || map_norm <= 0.0) {
    return Invalid<lunar::incremental_navigation::StateInput>();
  }
  const auto normalized = lunar::incremental_navigation::Quaternion{
      .w = map_orientation.w / map_norm,
      .x = map_orientation.x / map_norm,
      .y = map_orientation.y / map_norm,
      .z = map_orientation.z / map_norm};
  return {
      .value = lunar::incremental_navigation::StateInput{
          .base_link_pose = {
              .position_m = {
                  .x = map_from_odom.translation_m.x + translated.x,
                  .y = map_from_odom.translation_m.y + translated.y},
              .yaw_rad = std::atan2(
                  2.0 * (normalized.w * normalized.z +
                         normalized.x * normalized.y),
                  1.0 - 2.0 * (normalized.y * normalized.y +
                               normalized.z * normalized.z))}},
      .reason_code = {}};
}

}  // namespace lunar::incremental_navigation_ros

#include "hierarchical/frame_transform.hpp"

#include <cmath>
#include <optional>
#include <utility>
#include <variant>

namespace lunar::planning::hierarchical {
namespace {

[[nodiscard]] bool Finite(const Vec3 value) noexcept {
  return std::isfinite(value.x) && std::isfinite(value.y) &&
         std::isfinite(value.z);
}

[[nodiscard]] bool Finite(const Quaternion value) noexcept {
  return std::isfinite(value.w) && std::isfinite(value.x) &&
         std::isfinite(value.y) && std::isfinite(value.z);
}

[[nodiscard]] std::optional<Quaternion>
Normalize(const Quaternion value) noexcept {
  if (!Finite(value)) {
    return std::nullopt;
  }
  const double squared_norm = value.w * value.w + value.x * value.x +
                              value.y * value.y + value.z * value.z;
  if (!std::isfinite(squared_norm) || squared_norm <= 1.0e-24) {
    return std::nullopt;
  }
  const double inverse_norm = 1.0 / std::sqrt(squared_norm);
  return Quaternion{
      .w = value.w * inverse_norm,
      .x = value.x * inverse_norm,
      .y = value.y * inverse_norm,
      .z = value.z * inverse_norm,
  };
}

[[nodiscard]] Quaternion Conjugate(const Quaternion value) noexcept {
  return Quaternion{
      .w = value.w,
      .x = -value.x,
      .y = -value.y,
      .z = -value.z,
  };
}

[[nodiscard]] Quaternion Multiply(const Quaternion left,
                                  const Quaternion right) noexcept {
  return Quaternion{
      .w = left.w * right.w - left.x * right.x - left.y * right.y -
           left.z * right.z,
      .x = left.w * right.x + left.x * right.w + left.y * right.z -
           left.z * right.y,
      .y = left.w * right.y - left.x * right.z + left.y * right.w +
           left.z * right.x,
      .z = left.w * right.z + left.x * right.y - left.y * right.x +
           left.z * right.w,
  };
}

[[nodiscard]] Vec3 Rotate(const Quaternion rotation,
                          const Vec3 point) noexcept {
  const Quaternion vector{.w = 0.0, .x = point.x, .y = point.y, .z = point.z};
  const Quaternion rotated =
      Multiply(Multiply(rotation, vector), Conjugate(rotation));
  return Vec3{rotated.x, rotated.y, rotated.z};
}

[[nodiscard]] std::optional<Quaternion>
DirectedRotation(const RigidTransform &transform,
                 const TransformDirection direction) noexcept {
  const auto normalized = Normalize(transform.rotation);
  if (!normalized || !Finite(transform.translation_m) ||
      transform.parent_frame.empty() || transform.child_frame.empty()) {
    return std::nullopt;
  }
  return direction == TransformDirection::kChildToParent
             ? normalized
             : std::optional<Quaternion>{Conjugate(*normalized)};
}

[[nodiscard]] std::optional<double>
TransformYaw(const double yaw, const RigidTransform &transform,
             const TransformDirection direction) noexcept {
  if (!std::isfinite(yaw)) {
    return std::nullopt;
  }
  const auto rotation = DirectedRotation(transform, direction);
  if (!rotation) {
    return std::nullopt;
  }
  const Vec3 heading =
      Rotate(*rotation, Vec3{std::cos(yaw), std::sin(yaw), 0.0});
  if (!Finite(heading) || std::hypot(heading.x, heading.y) <= 1.0e-12) {
    return std::nullopt;
  }
  const double result = std::atan2(heading.y, heading.x);
  return std::isfinite(result) ? std::optional<double>{result} : std::nullopt;
}

} // namespace

std::optional<Vec3>
TransformPoint(const Vec3 point, const RigidTransform &parent_from_child,
               const TransformDirection direction) noexcept {
  if (!Finite(point)) {
    return std::nullopt;
  }
  const auto rotation = DirectedRotation(parent_from_child, direction);
  if (!rotation) {
    return std::nullopt;
  }
  if (direction == TransformDirection::kChildToParent) {
    const Vec3 rotated = Rotate(*rotation, point);
    const Vec3 transformed{
        rotated.x + parent_from_child.translation_m.x,
        rotated.y + parent_from_child.translation_m.y,
        rotated.z + parent_from_child.translation_m.z,
    };
    return Finite(transformed) ? std::optional<Vec3>{transformed}
                               : std::nullopt;
  }
  const Vec3 translated{
      point.x - parent_from_child.translation_m.x,
      point.y - parent_from_child.translation_m.y,
      point.z - parent_from_child.translation_m.z,
  };
  const Vec3 transformed = Rotate(*rotation, translated);
  return Finite(transformed) ? std::optional<Vec3>{transformed} : std::nullopt;
}

std::optional<Quaternion>
TransformOrientation(const Quaternion orientation,
                     const RigidTransform &parent_from_child,
                     const TransformDirection direction) noexcept {
  const auto pose_rotation = Normalize(orientation);
  const auto frame_rotation = DirectedRotation(parent_from_child, direction);
  if (!pose_rotation || !frame_rotation) {
    return std::nullopt;
  }
  return Normalize(Multiply(*frame_rotation, *pose_rotation));
}

std::optional<Pose3>
TransformPose(const Pose3 &pose, const RigidTransform &parent_from_child,
              const TransformDirection direction) noexcept {
  const auto position =
      TransformPoint(pose.position_m, parent_from_child, direction);
  const auto orientation =
      TransformOrientation(pose.orientation, parent_from_child, direction);
  if (!position || !orientation) {
    return std::nullopt;
  }
  return Pose3{.position_m = *position, .orientation = *orientation};
}

std::optional<GoalRegion>
TransformGoal(const GoalRegion &goal, const RigidTransform &parent_from_child,
              const TransformDirection direction) noexcept {
  GoalRegion transformed{
      .goal_id = goal.goal_id,
      .target = PointGoal{},
      .yaw_rad = std::nullopt,
      .yaw_tolerance_rad = goal.yaw_tolerance_rad,
  };
  if (!std::isfinite(goal.yaw_tolerance_rad)) {
    return std::nullopt;
  }
  if (goal.yaw_rad) {
    transformed.yaw_rad =
        TransformYaw(*goal.yaw_rad, parent_from_child, direction);
    if (!transformed.yaw_rad) {
      return std::nullopt;
    }
  }
  if (const auto *point = std::get_if<PointGoal>(&goal.target)) {
    const auto position =
        TransformPoint(point->position_m, parent_from_child, direction);
    if (!position || !std::isfinite(point->tolerance_m)) {
      return std::nullopt;
    }
    transformed.target = PointGoal{
        .position_m = *position,
        .tolerance_m = point->tolerance_m,
    };
    return transformed;
  }
  const auto *region = std::get_if<PlanarRegionGoal>(&goal.target);
  if (region == nullptr || !std::isfinite(region->normal_tolerance_m)) {
    return std::nullopt;
  }
  PlanarRegionGoal target{
      .boundary_m = {},
      .normal_tolerance_m = region->normal_tolerance_m,
  };
  target.boundary_m.reserve(region->boundary_m.size());
  for (const Vec3 vertex : region->boundary_m) {
    const auto transformed_vertex =
        TransformPoint(vertex, parent_from_child, direction);
    if (!transformed_vertex) {
      return std::nullopt;
    }
    target.boundary_m.push_back(*transformed_vertex);
  }
  transformed.target = std::move(target);
  return transformed;
}

} // namespace lunar::planning::hierarchical

#pragma once

#include <optional>

#include "lunar_planner_core/types/goal.hpp"
#include "lunar_planner_core/types/world_snapshot.hpp"

namespace lunar::planning::hierarchical {

enum class TransformDirection {
  kChildToParent,
  kParentToChild,
};

[[nodiscard]] std::optional<Vec3>
TransformPoint(Vec3 point, const RigidTransform &parent_from_child,
               TransformDirection direction) noexcept;

[[nodiscard]] std::optional<Quaternion>
TransformOrientation(Quaternion orientation,
                     const RigidTransform &parent_from_child,
                     TransformDirection direction) noexcept;

[[nodiscard]] std::optional<Pose3>
TransformPose(const Pose3 &pose, const RigidTransform &parent_from_child,
              TransformDirection direction) noexcept;

[[nodiscard]] std::optional<GoalRegion>
TransformGoal(const GoalRegion &goal, const RigidTransform &parent_from_child,
              TransformDirection direction) noexcept;

} // namespace lunar::planning::hierarchical

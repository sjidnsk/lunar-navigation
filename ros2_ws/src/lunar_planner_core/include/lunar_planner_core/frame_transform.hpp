#pragma once

#include <optional>

#include "lunar_planner_core/types/world_snapshot.hpp"

namespace lunar::planning::hierarchical {

enum class TransformDirection {
  kChildToParent,
  kParentToChild,
};

[[nodiscard]] std::optional<Vec3>
TransformVector(Vec3 vector, const RigidTransform& parent_from_child,
                TransformDirection direction) noexcept;

}  // namespace lunar::planning::hierarchical

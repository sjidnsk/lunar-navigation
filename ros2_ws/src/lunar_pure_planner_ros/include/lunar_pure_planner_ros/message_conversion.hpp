#pragma once

#include <cstdint>
#include <optional>
#include <string>

#include <lunar_planning_msgs/action/plan_motion.hpp>

#include "lunar_pure_planner_core/types/planning_request.hpp"
#include "lunar_pure_planner_core/types/world_snapshot.hpp"

namespace lunar::pure_planner_ros {

struct GoalConversionResult final {
  std::optional<lunar::pure_planning::GoalRegion> goal;
  std::string reason_code;

  [[nodiscard]] bool ok() const noexcept {
    return goal.has_value() && reason_code.empty();
  }
};

[[nodiscard]] GoalConversionResult ConvertGoal(
    const lunar_planning_msgs::action::PlanMotion::Goal& request,
    const lunar::pure_planning::RigidTransform& map_from_odom);

[[nodiscard]] lunar_planning_msgs::action::PlanMotion::Result ConvertResult(
    const lunar::pure_planning::PlanningResult& result,
    std::uint64_t mission_revision);

}  // namespace lunar::pure_planner_ros

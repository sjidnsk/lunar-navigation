#pragma once

#include <cstdint>
#include <optional>
#include <string>

#include <lunar_planning_msgs/action/plan_motion.hpp>
#include <lunar_planning_msgs/msg/goal_region.hpp>

#include "lunar_planner_core/types/planner_io.hpp"

namespace lunar::planning::ros {

struct GoalMessageConversion final {
  std::optional<lunar::planning::GoalRegion> goal;
  std::string reason_code;

  [[nodiscard]] bool ok() const noexcept {
    return goal.has_value() && reason_code.empty();
  }
};

struct PlannerResultContext final {
  lunar::planning::TimePoint global_map_stamp;
  lunar::planning::TimePoint local_map_stamp;
  lunar::planning::TimePoint state_stamp;
  std::uint64_t mission_revision{};
  std::string planning_frame{"odom"};
};

struct ActionResultConversion final {
  std::optional<lunar_planning_msgs::action::PlanMotion::Result> result;
  std::string reason_code;

  [[nodiscard]] bool ok() const noexcept {
    return result.has_value() && reason_code.empty();
  }
};

[[nodiscard]] GoalMessageConversion ConvertGoalMessage(
    const lunar_planning_msgs::msg::GoalRegion& message);

[[nodiscard]] ActionResultConversion ConvertPlannerOutput(
    const lunar::planning::PlannerOutput& output,
    const PlannerResultContext& context);

}  // namespace lunar::planning::ros

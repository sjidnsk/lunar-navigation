#pragma once

#include <optional>
#include <stop_token>
#include <string>

#include "lunar_planner_core/types/execution_context.hpp"
#include "lunar_planner_core/types/goal.hpp"
#include "lunar_planner_core/types/planner_config.hpp"
#include "lunar_planner_core/types/planner_io.hpp"
#include "lunar_planner_core/types/platform_capability.hpp"
#include "lunar_planner_core/types/world_snapshot.hpp"

namespace lunar::planning::hierarchical {

struct LocalPlanningProblem final {
  std::string request_id;
  TimePoint state_time;
  PlatformState current_state;
  GoalRegion goal_odom;
  GridMap local_map_view;
  PlatformCapability capability;
  PlannerConfig config;
  std::optional<ExecutionContext> previous_execution;
  std::stop_token stop_token;
};

} // namespace lunar::planning::hierarchical

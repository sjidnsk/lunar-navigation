#pragma once

#include <string>
#include <string_view>

#include <diagnostic_msgs/msg/diagnostic_array.hpp>

#include "lunar_pure_planner_core/types/planning_request.hpp"

namespace lunar::pure_planner_ros {

[[nodiscard]] diagnostic_msgs::msg::DiagnosticArray MakeRequestDiagnostics(
    std::string_view request_id,
    lunar::pure_planning::PlatformType platform_type,
    lunar::pure_planning::EnvironmentMode environment_mode,
    const lunar::pure_planning::PlanningResult& result);

[[nodiscard]] std::string FindDiagnosticValue(
    const diagnostic_msgs::msg::DiagnosticArray& diagnostics,
    const std::string& key);

}  // namespace lunar::pure_planner_ros

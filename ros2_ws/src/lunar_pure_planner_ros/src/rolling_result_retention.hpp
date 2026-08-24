#pragma once

#include <optional>
#include "lunar_pure_planner_core/planner.hpp"

namespace lunar::pure_planner_ros::detail
{

[[nodiscard]] inline lunar::pure_planning::PlanningResult
MakeRollingIdleCanceledResult(
  std::optional<lunar::pure_planning::PlanningResult> last_segment)
{
  auto canceled = lunar::pure_planning::PlanningResult{
    .status = lunar::pure_planning::PlanningStatus::kCanceled,
    .reason_code = "REQUEST_CANCELED"};
  if (last_segment.has_value()) {
    canceled.wheel_metrics = last_segment->wheel_metrics;
  }
  return canceled;
}

}  // namespace lunar::pure_planner_ros::detail

#include "lunar_pure_planner_ros/request_diagnostics.hpp"

#include <chrono>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <string>
#include <utility>

#include <lunar_planning_msgs/action/plan_motion.hpp>

namespace lunar::pure_planner_ros {
namespace {

[[nodiscard]] std::string Decimal(const double value) {
  std::ostringstream stream;
  stream << std::setprecision(15) << value;
  return stream.str();
}

[[nodiscard]] std::string Milliseconds(const std::chrono::nanoseconds value) {
  return Decimal(std::chrono::duration<double, std::milli>(value).count());
}

[[nodiscard]] std::string PlatformType(const lunar::pure_planning::PlatformType value) {
  switch (value) {
    case lunar::pure_planning::PlatformType::kWheeled:
      return "WHEELED";
    case lunar::pure_planning::PlatformType::kLegged:
      return "LEGGED";
    case lunar::pure_planning::PlatformType::kHopper:
      return "HOPPER";
  }
  return "UNKNOWN";
}

struct StatusFields final {
  std::uint8_t outcome;
  std::uint8_t level;
  const char* reason;
};

[[nodiscard]] StatusFields StatusFor(const lunar::pure_planning::PlanningStatus status) {
  using PlanningStatus = lunar::pure_planning::PlanningStatus;
  using DiagnosticStatus = diagnostic_msgs::msg::DiagnosticStatus;
  using Action = lunar_planning_msgs::action::PlanMotion;
  switch (status) {
    case PlanningStatus::kSuccess:
      return {Action::Result::NEW_REFERENCE_AVAILABLE, DiagnosticStatus::OK, "PLAN_FOUND"};
    case PlanningStatus::kInvalidInput:
      return {Action::Result::INVALID_REQUEST, DiagnosticStatus::ERROR, "INVALID_INPUT"};
    case PlanningStatus::kGoalOutsideLocalMap:
      return {Action::Result::GOAL_INFEASIBLE, DiagnosticStatus::WARN, "GOAL_OUTSIDE_LOCAL_MAP"};
    case PlanningStatus::kNoPath:
      return {Action::Result::GOAL_INFEASIBLE, DiagnosticStatus::WARN, "NO_PATH"};
    case PlanningStatus::kTimedOut:
      return {Action::Result::RESOURCE_EXHAUSTED, DiagnosticStatus::WARN, "TIMEOUT"};
    case PlanningStatus::kCanceled:
      return {Action::Result::CANCELED, DiagnosticStatus::WARN, "REQUEST_CANCELED"};
    case PlanningStatus::kPlannerError:
      return {Action::Result::NUMERICAL_FAILURE, DiagnosticStatus::ERROR, "PLANNER_ERROR"};
  }
  return {Action::Result::NUMERICAL_FAILURE, DiagnosticStatus::ERROR, "PLANNER_ERROR"};
}

}  // namespace

diagnostic_msgs::msg::DiagnosticArray MakeRequestDiagnostics(
    const std::string_view request_id,
    const lunar::pure_planning::PlatformType platform_type,
    const lunar::pure_planning::EnvironmentMode environment_mode,
    const lunar::pure_planning::PlanningResult& result) {
  diagnostic_msgs::msg::DiagnosticStatus status;
  status.name = "lunar_pure_planner/request";
  status.hardware_id = "lunar_pure_planner";
  const StatusFields fields = StatusFor(result.status);
  status.level = fields.level;
  const auto add_value = [&status](std::string key, std::string value) {
    diagnostic_msgs::msg::KeyValue entry;
    entry.key = std::move(key);
    entry.value = std::move(value);
    status.values.push_back(std::move(entry));
  };
  add_value("request_id", std::string{request_id});
  add_value("platform_type", PlatformType(platform_type));
  add_value("environment_mode", std::to_string(static_cast<std::uint8_t>(environment_mode)));
  add_value("planning_outcome", std::to_string(fields.outcome));
  add_value("reason_code", fields.reason);
  add_value("global_elapsed_ms", Milliseconds(result.timing.global_elapsed));
  add_value("global_call_count", std::to_string(result.timing.global_call_count));
  add_value("local_elapsed_ms", Milliseconds(result.timing.local_elapsed));
  add_value("local_call_count", std::to_string(result.timing.local_call_count));
  add_value("total_elapsed_ms", Milliseconds(result.timing.total_elapsed));
  diagnostic_msgs::msg::DiagnosticArray diagnostics;
  diagnostics.status.push_back(std::move(status));
  return diagnostics;
}

std::string FindDiagnosticValue(const diagnostic_msgs::msg::DiagnosticArray& diagnostics,
                                const std::string& key) {
  if (diagnostics.status.size() != 1U) {
    return {};
  }
  for (const auto& value : diagnostics.status.front().values) {
    if (value.key == key) {
      return value.value;
    }
  }
  return {};
}

}  // namespace lunar::pure_planner_ros

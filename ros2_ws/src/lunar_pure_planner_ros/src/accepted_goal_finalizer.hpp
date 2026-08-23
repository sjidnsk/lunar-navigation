#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>

namespace lunar::pure_planner_ros::detail {

enum class AcceptedGoalFinalizationFailure {
  kResultConstruction,
  kTerminalAttempt,
  kDiagnosticConstruction,
  kDiagnosticPublication,
  kDiagnosticFormatting,
};

template <typename FailureReporter>
void ReportAcceptedGoalFinalizationFailure(
    FailureReporter& report_failure,
    const AcceptedGoalFinalizationFailure failure) noexcept {
  try {
    std::invoke(report_failure, failure);
  } catch (...) {
  }
}

template <typename MakeResult, typename DeliverTerminal,
          typename MakeDiagnostics, typename PublishDiagnostics,
          typename FormatDiagnostics, typename FailureReporter>
void FinalizeAcceptedGoal(MakeResult&& make_result,
                          DeliverTerminal&& deliver_terminal,
                          MakeDiagnostics&& make_diagnostics,
                          PublishDiagnostics&& publish_diagnostics,
                          FormatDiagnostics&& format_diagnostics,
                          FailureReporter&& report_failure) noexcept {
  try {
    auto result = std::invoke(make_result);
    try {
      std::invoke(deliver_terminal, result);
    } catch (...) {
      ReportAcceptedGoalFinalizationFailure(
          report_failure,
          AcceptedGoalFinalizationFailure::kTerminalAttempt);
    }

    try {
      auto diagnostics = std::invoke(make_diagnostics);
      try {
        std::invoke(publish_diagnostics, diagnostics);
      } catch (...) {
        ReportAcceptedGoalFinalizationFailure(
            report_failure,
            AcceptedGoalFinalizationFailure::kDiagnosticPublication);
      }
      try {
        std::invoke(format_diagnostics, diagnostics);
      } catch (...) {
        ReportAcceptedGoalFinalizationFailure(
            report_failure,
            AcceptedGoalFinalizationFailure::kDiagnosticFormatting);
      }
    } catch (...) {
      ReportAcceptedGoalFinalizationFailure(
          report_failure,
          AcceptedGoalFinalizationFailure::kDiagnosticConstruction);
    }
  } catch (...) {
    ReportAcceptedGoalFinalizationFailure(
        report_failure,
        AcceptedGoalFinalizationFailure::kResultConstruction);
  }
}

template <typename Action>
[[nodiscard]] std::shared_ptr<typename Action::Result>
MakeMinimalPlannerErrorResult(const std::uint64_t mission_revision,
                              const std::chrono::nanoseconds elapsed) {
  auto result = std::make_shared<typename Action::Result>();
  result->planning_outcome = Action::Result::NUMERICAL_FAILURE;
  result->execution_directive = Action::Result::NO_SAFE_REFERENCE;
  result->reason_code = "PLANNER_ERROR";
  result->mission_revision = mission_revision;
  result->has_reference = false;
  result->diagnostics.planner_name = "lunar_pure_planner";
  result->diagnostics.elapsed_s = std::chrono::duration<double>(elapsed).count();
  return result;
}

}  // namespace lunar::pure_planner_ros::detail

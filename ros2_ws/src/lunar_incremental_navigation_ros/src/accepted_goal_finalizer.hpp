#pragma once

#include <cstdint>
#include <functional>
#include <memory>

namespace lunar::incremental_navigation_ros::detail {

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
MakeMinimalPlannerErrorResult(const std::uint64_t last_segment_revision) {
  auto result = std::make_shared<typename Action::Result>();
  result->outcome = Action::Result::INTERNAL_ERROR;
  result->reason_code = "INTERNAL_ERROR";
  result->last_segment_revision = last_segment_revision;
  return result;
}

}  // namespace lunar::incremental_navigation_ros::detail

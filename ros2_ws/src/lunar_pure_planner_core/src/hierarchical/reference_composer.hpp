#pragma once

#include <optional>
#include <string>

#include "lunar_pure_planner_core/planner.hpp"

namespace lunar::pure_planning::hierarchical {

struct ReferenceComposeResult final {
  std::optional<MotionReference> reference;
  std::string reason_code;

  [[nodiscard]] bool ok() const noexcept {
    return reference.has_value() && reason_code.empty();
  }
};

[[nodiscard]] ReferenceComposeResult ComposeSurfaceReference(
    const PlanningRequest& input, const GlobalRoute& route,
    MotionReferenceData data, SearchControl control);

[[nodiscard]] ReferenceComposeResult ComposeCaveReference(
    const PlanningRequest& input, MotionReferenceData data,
    SearchControl control);

}  // namespace lunar::pure_planning::hierarchical

#pragma once

#include <optional>
#include <string>

#include "hierarchical/global_route.hpp"
#include "lunar_planner_core/types/planner_io.hpp"

namespace lunar::planning::hierarchical {

struct ReferenceComposeResult final {
  std::optional<MotionReference> reference;
  std::string reason_code;

  [[nodiscard]] bool ok() const noexcept {
    return reference.has_value() && reason_code.empty();
  }
};

[[nodiscard]] ReferenceComposeResult
ComposeReference(const PlannerInput &input, const GlobalRoute &global_route,
                 PlannerOutput local_output);

} // namespace lunar::planning::hierarchical

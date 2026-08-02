#pragma once

#include <string>
#include <string_view>
#include <vector>

#include "lunar_path_planner/v3/contracts/planning_response.hpp"

namespace lunar::planning::v3 {

struct ValidationIssue final {
  std::string field_path;
  std::string reason_code;
  std::string message;
};

struct ValidationReport final {
  std::vector<ValidationIssue> issues;

  [[nodiscard]] bool ok() const noexcept { return issues.empty(); }

  [[nodiscard]] bool Contains(
      std::string_view field_path,
      std::string_view reason_code) const;
};

class SemanticValidator final {
 public:
  [[nodiscard]] ValidationReport Validate(
      const PlanningRequest& request) const;

  [[nodiscard]] ValidationReport Validate(
      const PlanningResponse& response) const;

  [[nodiscard]] ValidationReport Validate(
      const PlatformReference& reference) const;

  [[nodiscard]] ValidationReport Validate(
      const ReferenceBundle& bundle) const;

  [[nodiscard]] ValidationReport Validate(
      const SafetyCapabilityProfile& capability) const;

  [[nodiscard]] ValidationReport Validate(
      const PlannerAlgorithmConfig& config) const;

  [[nodiscard]] ValidationReport ValidateForActivation(
      const ReferenceBundle& bundle,
      const ReferenceActivationContext& context) const;

  [[nodiscard]] ValidationReport ValidateForActivation(
      const PlanningResponse& response,
      const ReferenceActivationContext& context) const;
};

}  // namespace lunar::planning::v3

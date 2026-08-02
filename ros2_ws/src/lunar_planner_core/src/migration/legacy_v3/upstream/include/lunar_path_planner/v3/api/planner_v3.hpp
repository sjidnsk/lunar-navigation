#pragma once

#include <memory>

#include "lunar_path_planner/v3/cache/deterministic_cache.hpp"
#include "lunar_path_planner/v3/contracts/planning_response.hpp"
#include "lunar_path_planner/v3/map/safe_projection.hpp"

namespace lunar::planning::v3 {

class ContractObjectRegistry;
class HopperPlanner;
class LeggedPlanner;
class SemanticValidator;
class WheelPlanner;

using SafeProjectionCache =
    DeterministicCache<SafeProjectionCacheKey, SafeProjection>;

class PlannerV3 {
 public:
  virtual ~PlannerV3() = default;

  [[nodiscard]] virtual PlanningResponse Plan(
      const PlanningRequest& request) noexcept = 0;
};

class PlannerV3Impl final : public PlannerV3 {
 public:
  PlannerV3Impl(
      const SemanticValidator& semantic_validator,
      const ContractObjectRegistry& contract_registry,
      SafeProjectionCache& projection_cache,
      std::unique_ptr<WheelPlanner> wheel,
      std::unique_ptr<LeggedPlanner> legged,
      std::unique_ptr<HopperPlanner> hopper) noexcept;

  ~PlannerV3Impl() override;

  PlannerV3Impl(const PlannerV3Impl&) = delete;
  PlannerV3Impl& operator=(const PlannerV3Impl&) = delete;
  PlannerV3Impl(PlannerV3Impl&&) = delete;
  PlannerV3Impl& operator=(PlannerV3Impl&&) = delete;

  [[nodiscard]] PlanningResponse Plan(
      const PlanningRequest& request) noexcept override;

 private:
  const SemanticValidator& semantic_validator_;
  const ContractObjectRegistry& contract_registry_;
  SafeProjectionCache& projection_cache_;
  std::unique_ptr<WheelPlanner> wheel_;
  std::unique_ptr<LeggedPlanner> legged_;
  std::unique_ptr<HopperPlanner> hopper_;
};

[[nodiscard]] std::unique_ptr<PlannerV3> MakeDefaultPlannerV3(
    const SemanticValidator& semantic_validator,
    const ContractObjectRegistry& contract_registry,
    SafeProjectionCache& projection_cache);

}  // namespace lunar::planning::v3

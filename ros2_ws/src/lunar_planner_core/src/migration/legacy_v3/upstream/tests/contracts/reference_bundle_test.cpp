#include <variant>

#include <gtest/gtest.h>

#include "lunar_path_planner/v3/contracts/planning_response.hpp"

namespace lpp = lunar::planning::v3;

TEST(ReferenceBundleContract, PlatformReferenceIsTypeSafeVariant) {
  lpp::ReferenceBundle bundle{};
  bundle.platform_reference = lpp::WheeledReference{};

  EXPECT_TRUE(std::holds_alternative<lpp::WheeledReference>(
      bundle.platform_reference));

  bundle.platform_reference = lpp::LeggedBodyReference{};
  EXPECT_TRUE(std::holds_alternative<lpp::LeggedBodyReference>(
      bundle.platform_reference));

  bundle.platform_reference = lpp::HopperReference{};
  EXPECT_TRUE(std::holds_alternative<lpp::HopperReference>(
      bundle.platform_reference));
}

TEST(PlanningResponseContract, ActivationRequiresBundleAtSemanticLayer) {
  lpp::PlanningResponse response{
      .planning_outcome = lpp::PlanningOutcome::kNewReferenceReady,
      .execution_directive = lpp::ExecutionDirective::kActivateNewBundle};

  EXPECT_FALSE(response.new_reference_bundle.has_value());
}

TEST(ReferenceBundleContract, InvalidationConditionsMatchFrozenSchema) {
  lpp::ReferenceValidity validity{};
  validity.invalidation_conditions = {
      lpp::InvalidationCondition::kMapSafetyRevisionChanged,
      lpp::InvalidationCondition::kStateDeviationExceeded,
      lpp::InvalidationCondition::kCapabilityRevisionChanged,
      lpp::InvalidationCondition::kReferenceHorizonExhausted,
  };

  EXPECT_EQ(validity.invalidation_conditions.size(), 4U);
}

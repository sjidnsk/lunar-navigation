#include <memory>
#include <variant>

#include <gtest/gtest.h>

#include "fixtures/system/planner_v3_system_fixture.hpp"

namespace lunar::planning::v3 {
namespace {

using system_test::MapScenario;
using system_test::RegistryFault;

TEST(PlannerV3FaultMatrixTest,
     MissingRegistryCapabilityIsAnInvalidRequest) {
  auto scenario = system_test::MakeSystemScenario(
      PlatformType::kWheeled,
      MapScenario::kOpenKnown,
      RegistryFault::kMissingCapability);
  auto planner = system_test::MakePlanner(scenario);

  const PlanningResponse response =
      planner->Plan(scenario.request);

  EXPECT_EQ(
      response.planning_outcome,
      PlanningOutcome::kInvalidRequest);
  EXPECT_EQ(
      response.execution_directive,
      ExecutionDirective::kHoldStationary);
  EXPECT_FALSE(response.new_reference_bundle.has_value());
}

TEST(PlannerV3FaultMatrixTest,
     RegistryBindingMismatchFailsBeforePlatformDispatch) {
  auto scenario = system_test::MakeSystemScenario(
      PlatformType::kLegged,
      MapScenario::kOpenKnown,
      RegistryFault::kCapabilityBindingsMismatch);
  auto planner = system_test::MakePlanner(scenario);

  const PlanningResponse response =
      planner->Plan(scenario.request);

  EXPECT_EQ(
      response.planning_outcome,
      PlanningOutcome::kStaleInput);
  EXPECT_EQ(
      response.execution_directive,
      ExecutionDirective::kHoldStationary);
  EXPECT_FALSE(response.new_reference_bundle.has_value());
}

TEST(PlannerV3FaultMatrixTest,
     PlatformCapabilityMismatchIsRejectedAsInvalid) {
  auto wheel = system_test::MakeSystemScenario(
      PlatformType::kWheeled);
  auto legged = system_test::MakeSystemScenario(
      PlatformType::kLegged);
  wheel.request.safety_capability =
      legged.request.safety_capability;
  wheel.request.capability_bindings =
      legged.request.capability_bindings;
  auto planner = system_test::MakePlanner(wheel);

  const PlanningResponse response =
      planner->Plan(wheel.request);

  EXPECT_EQ(
      response.planning_outcome,
      PlanningOutcome::kInvalidRequest);
  EXPECT_EQ(
      response.execution_directive,
      ExecutionDirective::kHoldStationary);
  EXPECT_FALSE(response.new_reference_bundle.has_value());
}

TEST(PlannerV3FaultMatrixTest,
     StaleActiveContextCannotMaskAKnownGoalFailure) {
  auto scenario = system_test::MakeSystemScenario(
      PlatformType::kWheeled,
      MapScenario::kKnownObstacleAtGoal);
  scenario.request.previous_execution_context =
      system_test::MakeGroundExecutionContext(
          scenario, true);
  auto planner = system_test::MakePlanner(scenario);

  const PlanningResponse response =
      planner->Plan(scenario.request);

  EXPECT_EQ(
      response.planning_outcome,
      PlanningOutcome::kActiveReferenceInvalidated);
  EXPECT_EQ(
      response.execution_directive,
      ExecutionDirective::kHoldStationary);
  EXPECT_FALSE(response.active_bundle_ref.has_value());
  EXPECT_FALSE(response.new_reference_bundle.has_value());
}

}  // namespace
}  // namespace lunar::planning::v3

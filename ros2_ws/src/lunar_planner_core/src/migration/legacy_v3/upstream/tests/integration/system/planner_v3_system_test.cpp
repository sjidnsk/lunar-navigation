#include <array>
#include <variant>

#include <gtest/gtest.h>

#include "fixtures/system/planner_v3_system_fixture.hpp"

namespace lunar::planning::v3 {
namespace {

using system_test::MapScenario;

struct HappyPathCase final {
  PlatformType platform_type;
  const char* name;
};

class PlannerV3HappyPathTest
    : public testing::TestWithParam<HappyPathCase> {};

TEST_P(PlannerV3HappyPathTest,
       ProducesAnActivationValidatedPlatformBundle) {
  const HappyPathCase test_case = GetParam();
  auto scenario =
      system_test::MakeSystemScenario(test_case.platform_type);
  const ValidationReport request_report =
      SemanticValidator{}.Validate(scenario.request);
  ASSERT_TRUE(request_report.ok())
      << system_test::DescribeIssues(request_report);
  auto planner = system_test::MakePlanner(scenario);

  const PlanningResponse response =
      planner->Plan(scenario.request);

  ASSERT_EQ(
      response.planning_outcome,
      PlanningOutcome::kNewReferenceReady)
      << JsonCodec::EncodePlanningResponse(response);
  ASSERT_EQ(
      response.execution_directive,
      ExecutionDirective::kActivateNewBundle);
  ASSERT_TRUE(response.new_reference_bundle.has_value());
  const ReferenceBundle& bundle =
      *response.new_reference_bundle;
  EXPECT_EQ(bundle.platform_type, test_case.platform_type);
  EXPECT_EQ(bundle.source_request_id, scenario.request.request_id);
  EXPECT_EQ(
      std::holds_alternative<WheeledReference>(
          bundle.platform_reference),
      test_case.platform_type == PlatformType::kWheeled);
  EXPECT_EQ(
      std::holds_alternative<LeggedBodyReference>(
          bundle.platform_reference),
      test_case.platform_type == PlatformType::kLegged);
  EXPECT_EQ(
      std::holds_alternative<HopperReference>(
          bundle.platform_reference),
      test_case.platform_type == PlatformType::kHopper);

  const ReferenceActivationContext context{
      scenario.request, *scenario.registry};
  const ValidationReport response_report =
      SemanticValidator{}.ValidateForActivation(
          response, context);
  EXPECT_TRUE(response_report.ok())
      << system_test::DescribeIssues(response_report);
}

INSTANTIATE_TEST_SUITE_P(
    AllPlatforms,
    PlannerV3HappyPathTest,
    testing::Values(
        HappyPathCase{PlatformType::kWheeled, "wheeled"},
        HappyPathCase{PlatformType::kLegged, "legged"},
        HappyPathCase{PlatformType::kHopper, "hopper"}),
    [](const testing::TestParamInfo<HappyPathCase>& info) {
      return info.param.name;
    });

TEST(PlannerV3SystemTest,
     FullyKnownObstacleGoalFailsWithoutPublishingAReference) {
  auto scenario = system_test::MakeSystemScenario(
      PlatformType::kWheeled,
      MapScenario::kKnownObstacleAtGoal);
  auto planner = system_test::MakePlanner(scenario);

  const PlanningResponse response =
      planner->Plan(scenario.request);

  EXPECT_EQ(
      response.planning_outcome,
      PlanningOutcome::kGoalInfeasible);
  EXPECT_EQ(
      response.execution_directive,
      ExecutionDirective::kHoldStationary);
  EXPECT_FALSE(response.new_reference_bundle.has_value());
  EXPECT_FALSE(response.active_bundle_ref.has_value());
}

TEST(PlannerV3SystemTest,
     UnknownGoalProducesOnlyANonExecutableFrontierTail) {
  auto scenario = system_test::MakeSystemScenario(
      PlatformType::kWheeled,
      MapScenario::kUnknownGoalWithSafeFrontier);
  auto planner = system_test::MakePlanner(scenario);

  const PlanningResponse response =
      planner->Plan(scenario.request);

  ASSERT_EQ(
      response.planning_outcome,
      PlanningOutcome::kSafeFrontierReferenceReady);
  ASSERT_EQ(
      response.execution_directive,
      ExecutionDirective::kActivateNewBundle);
  ASSERT_TRUE(response.new_reference_bundle.has_value());
  const std::vector<Vec3>& tail =
      response.new_reference_bundle->route_skeleton.content
          .unresolved_tail;
  ASSERT_FALSE(tail.empty());
  EXPECT_EQ(
      response.new_reference_bundle->generation_evidence
          .termination_reason,
      "KNOWN_SAFE_FRONTIER_BEFORE_UNKNOWN_SPACE");
}

}  // namespace
}  // namespace lunar::planning::v3

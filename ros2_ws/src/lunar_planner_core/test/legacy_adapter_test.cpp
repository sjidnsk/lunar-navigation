#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <string>
#include <stop_token>
#include <utility>
#include <variant>

#include <gtest/gtest.h>

#include "migration/legacy_v3_adapter.hpp"
#include "test_fixtures.hpp"

namespace lunar::planning {
namespace {

using InputFactory = PlannerInput (*)();

struct PlatformCase final {
  const char* name;
  PlatformType platform;
  InputFactory make_input;
};

[[nodiscard]] std::string DiagnosticWarnings(const PlannerOutput& output) {
  std::string result;
  for (const std::string& warning : output.diagnostics.warning_codes) {
    if (!result.empty()) {
      result += ',';
    }
    result += warning;
  }
  return result;
}

class LegacyAdapterHappyPathTest
    : public testing::TestWithParam<PlatformCase> {};

TEST_P(LegacyAdapterHappyPathTest, ProducesOnlySimplifiedReferenceValues) {
  const PlatformCase test_case = GetParam();
  LegacyV3Adapter adapter;

  const PlannerOutput output = adapter.Plan(test_case.make_input());

  ASSERT_EQ(output.outcome, PlanningOutcome::kNewReferenceAvailable)
      << output.reason_code << ":" << DiagnosticWarnings(output);
  EXPECT_EQ(output.directive, ExecutionDirective::kActivateNewReference);
  ASSERT_TRUE(output.reference.has_value());
  EXPECT_EQ(output.reference->platform_type, test_case.platform);
  EXPECT_FALSE(output.reference->plan_id.empty());
  EXPECT_EQ(adapter.fallback_invocations(), 0U);
}

INSTANTIATE_TEST_SUITE_P(
    AllPlatforms,
    LegacyAdapterHappyPathTest,
    testing::Values(
        PlatformCase{"wheel", PlatformType::kWheeled, &test::MakeValidWheelInput},
        PlatformCase{"legged", PlatformType::kLegged, &test::MakeValidLeggedInput},
        PlatformCase{"hopper", PlatformType::kHopper, &test::MakeValidHopperInput}),
    [](const testing::TestParamInfo<PlatformCase>& info) {
      return info.param.name;
    });

TEST(LegacyAdapter, MapsNoPathWithoutFallback) {
  auto input = test::MakeValidWheelInput();
  auto& forbidden = std::get<std::vector<std::uint8_t>>(
      input.world.local_map.layers.at("forbidden").values);
  std::fill(forbidden.begin(), forbidden.end(), 1U);
  LegacyV3Adapter adapter;

  const auto output = adapter.Plan(input);

  EXPECT_EQ(output.outcome, PlanningOutcome::kNoKnownSafeRoute)
      << output.reason_code << ":" << DiagnosticWarnings(output);
  EXPECT_EQ(output.directive, ExecutionDirective::kNoSafeReference);
  EXPECT_FALSE(output.reference.has_value());
  EXPECT_EQ(adapter.fallback_invocations(), 0U);
}

TEST(LegacyAdapter, MapsKnownHardGoalToGoalInfeasible) {
  auto input = test::MakeValidWheelInput();
  input.goal.target = PointGoal{
      .position_m = {4.5, 3.5, 0.0},
      .tolerance_m = 0.2,
  };
  auto& forbidden = std::get<std::vector<std::uint8_t>>(
      input.world.local_map.layers.at("forbidden").values);
  forbidden[3U * input.world.local_map.width + 4U] = 1U;
  LegacyV3Adapter adapter;

  const auto output = adapter.Plan(input);

  EXPECT_EQ(output.outcome, PlanningOutcome::kGoalInfeasible)
      << output.reason_code << ":" << DiagnosticWarnings(output);
  EXPECT_EQ(output.directive, ExecutionDirective::kHoldPosition);
  EXPECT_FALSE(output.reference.has_value());
}

TEST(LegacyAdapter, CooperativelyCancelsWithoutPublishingAReference) {
  auto input = test::MakeValidHopperInput();
  std::stop_source source;
  source.request_stop();
  input.stop_token = source.get_token();
  LegacyV3Adapter adapter;

  const auto output = adapter.Plan(input);

  EXPECT_EQ(output.outcome, PlanningOutcome::kCanceled);
  EXPECT_EQ(output.directive, ExecutionDirective::kHoldPosition);
  EXPECT_EQ(output.reason_code, "REQUEST_CANCELED");
  EXPECT_FALSE(output.reference.has_value());
  EXPECT_EQ(adapter.fallback_invocations(), 0U);
}

TEST(LegacyAdapter, MapsInjectedLegacyExceptionToNumericalFailure) {
  LegacyV3Adapter adapter{LegacyV3FaultMode::kThrowingRegistry};

  const auto output = adapter.Plan(test::MakeValidLeggedInput());

  EXPECT_EQ(output.outcome, PlanningOutcome::kNumericalFailure);
  EXPECT_EQ(output.directive, ExecutionDirective::kHoldPosition);
  EXPECT_EQ(output.reason_code, "NUMERICAL_FAILURE");
  EXPECT_FALSE(output.reference.has_value());
  EXPECT_EQ(adapter.fallback_invocations(), 0U);
}

}  // namespace
}  // namespace lunar::planning

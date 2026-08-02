#include <variant>

#include <gtest/gtest.h>

#include "integration/wheel/wheel_planner_test_fixture.hpp"
#include "lunar_path_planner/v3/codec/json_codec.hpp"
#include "lunar_path_planner/v3/codec/semantic_validator.hpp"
#include "lunar_path_planner/v3/wheel/wheel_planner.hpp"

namespace lunar::planning::v3 {
namespace {

[[nodiscard]] CorridorResult AlwaysFallbackCorridor(
    const WheelCorridorRequest&) {
  return {
      .status = CorridorStatus::kFallbackRequired,
      .fallback =
          CorridorFallback::kUseDiscreteValidatedPrimitives,
      .reason_code = "forced_corridor_fallback",
  };
}

TEST(WheelPlannerTest,
     CorridorFailureReturnsOnlyPrimitiveChains) {
  auto fixture = test::MakeFixture();
  WheelPlanner planner{nullptr, &AlwaysFallbackCorridor};

  const auto result =
      planner.Plan(fixture.request, fixture.terminal);

  ASSERT_TRUE(IsOk(result));
  const auto& reference = std::get<WheeledReference>(result);
  ASSERT_FALSE(reference.segments.empty());
  for (const auto& segment : reference.segments) {
    if (const auto* drive =
            std::get_if<DriveSegment>(&segment)) {
      EXPECT_TRUE(
          std::holds_alternative<ValidatedPrimitiveChain>(
              drive->geometric_path));
    }
  }
}

TEST(WheelPlannerTest, RepresentsInPlaceTurnAsSpinSegment) {
  auto fixture = test::MakeFixture(true);
  WheelPlanner planner;

  const auto result =
      planner.Plan(fixture.request, fixture.terminal);

  ASSERT_TRUE(IsOk(result));
  const auto& reference = std::get<WheeledReference>(result);
  ASSERT_EQ(reference.segments.size(), 1U);
  EXPECT_TRUE(std::holds_alternative<SpinSegment>(
      reference.segments.front()));
}

TEST(WheelPlannerTest,
     FinalReferenceHasStableIdentityOriginAndJcsHash) {
  auto fixture = test::MakeFixture();
  WheelPlanner planner;

  const auto first =
      planner.Plan(fixture.request, fixture.terminal);
  const auto second =
      planner.Plan(fixture.request, fixture.terminal);

  ASSERT_TRUE(IsOk(first));
  ASSERT_TRUE(IsOk(second));
  const auto& a = std::get<WheeledReference>(first);
  const auto& b = std::get<WheeledReference>(second);
  EXPECT_FALSE(a.reference_id.empty());
  EXPECT_EQ(a.reference_id, b.reference_id);
  EXPECT_EQ(a.reference_time_origin.clock_id,
            fixture.request.request_time.clock_id);
  EXPECT_EQ(a.reference_time_origin.tick,
            fixture.request.request_time.tick);
  EXPECT_EQ(a.reference_hash, b.reference_hash);
  const auto expected =
      CanonicalReferenceHash(PlatformReference{a});
  ASSERT_TRUE(IsOk(expected));
  EXPECT_EQ(a.reference_hash, std::get<Sha256Digest>(expected));
  EXPECT_TRUE(
      SemanticValidator{}.Validate(PlatformReference{a}).ok());
}

}  // namespace
}  // namespace lunar::planning::v3

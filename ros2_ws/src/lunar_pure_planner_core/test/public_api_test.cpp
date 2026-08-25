#include <cstddef>
#include <functional>
#include <type_traits>
#include <utility>

#include <gtest/gtest.h>

#include "lunar_pure_planner_core/planner.hpp"
#include "lunar_pure_planner_core/traversability_map.hpp"

namespace lunar::pure_planning {
namespace {

TEST(PublicApi, ExposesOnlyTheMinimalDualModePlannerEntrypoint) {
  static_assert(std::is_default_constructible_v<Planner>);
  static_assert(std::is_constructible_v<Planner, PlannerBackends>);
  static_assert(std::is_same_v<
                decltype(std::declval<Planner&>().Plan(
                    std::declval<const PlanningRequest&>())),
                PlanningResult>);
  static_assert(std::is_same_v<
                decltype(std::declval<PlannerBackends&>().global),
                std::function<GlobalStageResult(const PlanningRequest&,
                                                SearchControl)>>);
  static_assert(std::is_same_v<
                decltype(std::declval<PlannerBackends&>().local),
                std::function<LocalStageResult(const PlanningRequest&,
                                               const LocalGoalSet&,
                                               SearchControl)>>);
}

TEST(PublicApi, DefaultsToLegacyAndCarriesNoTraversabilitySnapshot) {
  static_assert(std::is_copy_constructible_v<TraversabilitySnapshot>);
  EXPECT_EQ(WheelPlannerModeName(WheelPlannerMode::kGridTraversabilityV1),
            "grid_traversability_v1");

  PlanningRequest request;

  EXPECT_EQ(request.config.wheel_planner_mode,
            WheelPlannerMode::kLegacyCertified);
  EXPECT_FALSE(request.world.traversability_snapshot);
}

TEST(PublicApi, RejectsPlanarRegionWithoutEnteringInjectedBackends) {
  std::size_t calls{};
  Planner planner(PlannerBackends{
      .global = [&calls](const PlanningRequest&, SearchControl) {
        ++calls;
        return GlobalStageResult{};
      },
      .local = [&calls](const PlanningRequest&, const LocalGoalSet&,
                        SearchControl) {
        ++calls;
        return LocalStageResult{};
      },
  });
  PlanningRequest input;
  input.request_id = "planar-is-out-of-scope";
  input.goal_map.target = PlanarRegionGoal{};

  const PlanningResult output = planner.Plan(input);

  EXPECT_EQ(output.status, PlanningStatus::kInvalidInput);
  EXPECT_EQ(output.reason_code, "INVALID_INPUT");
  EXPECT_EQ(calls, 0U);
}

}  // namespace
}  // namespace lunar::pure_planning

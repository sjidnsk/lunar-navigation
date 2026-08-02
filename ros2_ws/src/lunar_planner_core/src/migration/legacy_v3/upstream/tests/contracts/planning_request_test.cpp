#include <chrono>
#include <concepts>
#include <type_traits>
#include <variant>

#include <gtest/gtest.h>

#include "lunar_path_planner/v3/contracts/planning_request.hpp"

namespace lpp = lunar::planning::v3;

template <class T>
concept HasOrientationBodyToFrame = requires(T state) {
  state.orientation_body_to_frame;
};

TEST(PlanningRequestContract, WheelAndLegStateDoesNotContainQuaternion) {
  static_assert(!HasOrientationBodyToFrame<lpp::WheeledOrLeggedState>);
}

TEST(PlanningRequestContract, HopperCarriesQuaternionAndAngularVelocity) {
  lpp::HopperState state{};
  state.orientation_body_to_frame = {1.0, 0.0, 0.0, 0.0};
  state.angular_velocity_radps = {0.0, 0.0, 0.1};

  EXPECT_DOUBLE_EQ(state.orientation_body_to_frame.w, 1.0);
  EXPECT_DOUBLE_EQ(state.angular_velocity_radps.z, 0.1);
}

TEST(PlanningRequestContract, TimestampDoesNotRoundThroughDouble) {
  const lpp::ClockStamp stamp{
      .clock_id = "mission-clock",
      .tick = std::chrono::nanoseconds{9'007'199'254'740'993LL}};

  EXPECT_EQ(stamp.tick.count(), 9'007'199'254'740'993LL);
}

TEST(PlanningRequestContract, ResultHasOneUniformErrorAlternative) {
  static_assert(std::variant_size_v<lpp::Result<int>> == 2U);
  static_assert(std::same_as<
                std::variant_alternative_t<1, lpp::Result<int>>,
                lpp::Error>);
}

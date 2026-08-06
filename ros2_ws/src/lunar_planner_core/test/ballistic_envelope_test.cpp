#include <chrono>
#include <cmath>
#include <limits>
#include <stop_token>
#include <variant>

#include <gtest/gtest.h>

#include "hierarchical/local_planning_problem.hpp"
#include "hopper/ballistic_envelope.hpp"
#include "hopper/hop_certifier.hpp"
#include "test_fixtures.hpp"

namespace lunar::planning::hopper {
namespace {

template <typename Config>
concept HasSampledBallisticLimits = requires(Config config) {
  config.maximum_nominal_aim_points_per_region;
  config.maximum_certification_attempts;
};

[[nodiscard]] double Norm(const Vec3 value) {
  return std::hypot(std::hypot(value.x, value.y), value.z);
}

[[nodiscard]] HopperCapability NarrowWindowCapability() {
  HopperCapability capability =
      std::get<HopperCapability>(test::MakeValidHopperInput().capability);
  capability.platform_mass_kg = 1.0;
  capability.maximum_launch_speed_mps = 3.94;
  capability.maximum_launch_impulse_newton_seconds = 3.94;
  capability.minimum_flight_time = std::chrono::milliseconds{500};
  capability.maximum_flight_time = std::chrono::seconds{10};
  capability.maximum_landing_speed_mps = 3.94;
  capability.minimum_downward_impact_speed_mps = 0.0;
  return capability;
}

TEST(BallisticEnvelope, FindsNarrowFeasibleWindowBetweenOldSamples) {
  const HopperCapability capability = NarrowWindowCapability();

  const auto result = SolveBallisticEnvelope(
      {0.0, 0.0, 0.5}, {9.55, 0.0, 0.5}, {0.0, 0.0, 0.0}, capability, 0.0, {});

  ASSERT_EQ(result.status, BallisticEnvelopeStatus::kSolved)
      << result.reason_code;
  ASSERT_TRUE(result.arc.has_value());
  EXPECT_NEAR(result.arc->landing_position_m.x, 9.55, 1.0e-12);
  EXPECT_LE(Norm(result.arc->launch_velocity_mps),
            capability.maximum_launch_speed_mps + 1.0e-9);
  EXPECT_LE(Norm(result.arc->landing_velocity_mps),
            capability.maximum_landing_speed_mps + 1.0e-9);
  EXPECT_GT(result.examined_intervals, 0U);
}

TEST(BallisticEnvelope, DistinguishesInfeasibleCanceledAndInvalid) {
  HopperCapability impossible = NarrowWindowCapability();
  impossible.maximum_launch_speed_mps = 1.0;
  impossible.maximum_launch_impulse_newton_seconds = 1.0;
  impossible.maximum_landing_speed_mps = 1.0;
  const auto infeasible = SolveBallisticEnvelope(
      {0.0, 0.0, 0.5}, {9.55, 0.0, 0.5}, {0.0, 0.0, 0.0}, impossible, 0.0, {});
  EXPECT_EQ(infeasible.status, BallisticEnvelopeStatus::kInfeasible);
  EXPECT_EQ(infeasible.reason_code, "HOPPER_BALLISTIC_INFEASIBLE");

  std::stop_source stop_source;
  stop_source.request_stop();
  const auto canceled = SolveBallisticEnvelope(
      {0.0, 0.0, 0.5}, {1.0, 0.0, 0.5}, {0.0, 0.0, 0.0},
      NarrowWindowCapability(), 0.0, stop_source.get_token());
  EXPECT_EQ(canceled.status, BallisticEnvelopeStatus::kCanceled);
  EXPECT_EQ(canceled.reason_code, "REQUEST_CANCELED");

  const auto invalid = SolveBallisticEnvelope(
      {std::numeric_limits<double>::quiet_NaN(), 0.0, 0.5}, {1.0, 0.0, 0.5},
      {0.0, 0.0, 0.0}, NarrowWindowCapability(), 0.0, {});
  EXPECT_EQ(invalid.status, BallisticEnvelopeStatus::kInvalid);
  EXPECT_EQ(invalid.reason_code, "HOPPER_BALLISTIC_INPUT_INVALID");
}

TEST(BallisticEnvelope, EnforcesImpulseAgainstTheActualInitialVelocity) {
  HopperCapability capability = NarrowWindowCapability();
  capability.maximum_launch_speed_mps = 20.0;
  capability.maximum_landing_speed_mps = 20.0;
  capability.maximum_launch_impulse_newton_seconds = 0.1;

  const auto result = SolveBallisticEnvelope(
      {0.0, 0.0, 0.5}, {1.0, 0.0, 0.5}, {10.0, 0.0, 0.0}, capability, 0.0, {});

  EXPECT_EQ(result.status, BallisticEnvelopeStatus::kInfeasible);
  EXPECT_EQ(result.reason_code, "HOPPER_BALLISTIC_INFEASIBLE");
}

TEST(BallisticEnvelope, AppliesAttitudeTimeAsAClosedLowerBound) {
  HopperCapability capability = NarrowWindowCapability();
  capability.maximum_launch_speed_mps = 20.0;
  capability.maximum_launch_impulse_newton_seconds = 20.0;
  capability.maximum_landing_speed_mps = 20.0;

  const auto solved = SolveBallisticEnvelope(
      {0.0, 0.0, 0.5}, {1.0, 0.0, 0.5}, {0.0, 0.0, 0.0}, capability, 4.25, {});
  ASSERT_EQ(solved.status, BallisticEnvelopeStatus::kSolved)
      << solved.reason_code;
  ASSERT_TRUE(solved.arc.has_value());
  EXPECT_GE(solved.arc->flight_time_s, 4.25 - 1.0e-12);

  const auto impossible = SolveBallisticEnvelope(
      {0.0, 0.0, 0.5}, {1.0, 0.0, 0.5}, {0.0, 0.0, 0.0}, capability, 10.01, {});
  EXPECT_EQ(impossible.status, BallisticEnvelopeStatus::kInfeasible);
}

TEST(BallisticEnvelope, CertifyFirstHopUsesTheCompleteImpulseWindow) {
  PlannerInput input = test::MakeValidHopperInput();
  auto &state = std::get<HopperState>(input.current_state);
  auto &capability = std::get<HopperCapability>(input.capability);
  constexpr double kRequiredFlightTimeS = 3.21;
  state.pose.position_m = {2.5, 2.5, 0.5};
  state.velocity.linear_mps = {
      1.0 / kRequiredFlightTimeS,
      0.0,
      -0.5 * capability.gravity_mps2.z * kRequiredFlightTimeS,
  };
  capability.platform_mass_kg = 1.0;
  capability.maximum_launch_speed_mps = 20.0;
  capability.maximum_launch_impulse_newton_seconds = 0.0005;
  capability.maximum_landing_speed_mps = 20.0;
  capability.minimum_downward_impact_speed_mps = 0.0;
  input.world.local_map = test::MakeFlatMap("odom", 12U, 10U, 0.5);

  const CertifiedLandingRegion source{
      .seed_cell = {5, 5},
      .aim_position_on_surface_m = {2.5, 2.5, 0.0},
      .plane_normal = {0.0, 0.0, 1.0},
      .boundary_m = {{2.0, 2.0, 0.0},
                     {3.0, 2.0, 0.0},
                     {3.0, 3.0, 0.0},
                     {2.0, 3.0, 0.0}},
      .area_m2 = 1.0,
      .minimum_clearance_m = 1.0,
  };
  const CertifiedLandingRegion target{
      .seed_cell = {7, 5},
      .aim_position_on_surface_m = {3.5, 2.5, 0.0},
      .plane_normal = {0.0, 0.0, 1.0},
      .boundary_m = {{3.0, 2.0, 0.0},
                     {4.0, 2.0, 0.0},
                     {4.0, 3.0, 0.0},
                     {3.0, 3.0, 0.0}},
      .area_m2 = 1.0,
      .minimum_clearance_m = 1.0,
  };
  const hierarchical::LocalPlanningProblem problem{
      .request_id = "complete-impulse-window",
      .state_time = input.state_time,
      .current_state = input.current_state,
      .goal_odom =
          GoalRegion{
              .goal_id = "narrow-impulse-goal",
              .target =
                  PointGoal{
                      .position_m = {3.5, 2.5, 0.0},
                      .tolerance_m = 0.05,
                  },
          },
      .local_map_view = input.world.local_map,
      .capability = input.capability,
      .config = input.config,
      .previous_execution = std::nullopt,
      .stop_token = {},
  };

  const HopCertificationResult result =
      CertifyFirstHop(problem, source, target);

  ASSERT_TRUE(result.ok()) << result.reason_code;
  ASSERT_TRUE(result.segment.has_value());
  EXPECT_NEAR(
      std::chrono::duration<double>(result.segment->flight_time).count(),
      kRequiredFlightTimeS, 1.0e-3);
}

static_assert(!HasSampledBallisticLimits<HopperPlannerConfig>);

} // namespace
} // namespace lunar::planning::hopper

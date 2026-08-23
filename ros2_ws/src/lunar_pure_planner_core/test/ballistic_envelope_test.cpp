#include <cmath>
#include <limits>

#include <gtest/gtest.h>

#include "hopper/ballistic_envelope.hpp"
#include "hopper/ballistic_kinematics.hpp"

namespace lunar::pure_planning::hopper {
namespace {

constexpr double kTolerance = 1.0e-6;

HopperCapability ReferenceCapability() {
  return HopperCapability{
      .specific_impulse_s = 301.0,
      .reference_total_mass_kg = 20.0,
      .reference_propellant_mass_kg = 0.20,
      .landing_support_radius_m = 0.45,
      .flight_collision_radius_m = 0.55,
      .maximum_landing_plane_residual_m = 0.05,
      .landing_lateral_margin_m = 0.20,
      .flight_map_margin_m = 0.20,
      .reachability_delta_v_margin_ratio = 0.10,
      .standard_gravity_mps2 = 9.80665,
      .maximum_landing_slope_rad = 0.174533,
  };
}

TEST(BallisticEnvelope, ReproducesTheHundredMeterReferenceCase) {
  constexpr double flight_time_s = 100.0 / 9.0;
  const auto solved = SolveBallisticArc(
      Vec3{0.0, 0.0, 0.0}, Vec3{100.0, 0.0, 0.0},
      Vec3{0.0, 0.0, -1.62}, flight_time_s);
  ASSERT_TRUE(solved.ok()) << solved.reason_code;

  const auto evaluated =
      EvaluateSingleHopEnvelope(*solved.arc, ReferenceCapability());

  ASSERT_TRUE(evaluated.ok()) << evaluated.reason_code;
  EXPECT_NEAR(evaluated.evidence->ideal_delta_v_mps, 25.455844, kTolerance);
  EXPECT_NEAR(evaluated.evidence->required_delta_v_mps, 28.001429, kTolerance);
  EXPECT_GE(evaluated.evidence->available_delta_v_mps,
            evaluated.evidence->required_delta_v_mps);
}

TEST(BallisticEnvelope, IsRepeatableAndHasNoRemainingFuelState) {
  const auto solved = SolveBallisticArc(
      Vec3{1.0, -2.0, 0.5}, Vec3{101.0, 3.0, -1.0},
      Vec3{0.0, 0.0, -1.62}, 11.5);
  ASSERT_TRUE(solved.ok());

  const auto first =
      EvaluateSingleHopEnvelope(*solved.arc, ReferenceCapability());
  const auto second =
      EvaluateSingleHopEnvelope(*solved.arc, ReferenceCapability());

  ASSERT_TRUE(first.ok());
  ASSERT_TRUE(second.ok());
  EXPECT_DOUBLE_EQ(first.evidence->available_delta_v_mps,
                   second.evidence->available_delta_v_mps);
  EXPECT_DOUBLE_EQ(first.evidence->required_delta_v_mps,
                   second.evidence->required_delta_v_mps);
}

TEST(BallisticEnvelope, ReportsOnlyTheFixedEnvelopeBoundary) {
  HopperCapability capability = ReferenceCapability();
  capability.reference_propellant_mass_kg = 0.01;
  const auto solved = SolveBallisticArc(
      Vec3{0.0, 0.0, 0.0}, Vec3{100.0, 0.0, 0.0},
      Vec3{0.0, 0.0, -1.62}, 12.0);
  ASSERT_TRUE(solved.ok());

  const auto evaluated =
      EvaluateSingleHopEnvelope(*solved.arc, capability);

  EXPECT_FALSE(evaluated.ok());
  EXPECT_EQ(evaluated.reason_code, "HOPPER_SINGLE_HOP_ENVELOPE_EXCEEDED");
}

TEST(BallisticEnvelope, RejectsInvalidReferenceConditions) {
  HopperCapability capability = ReferenceCapability();
  capability.reference_propellant_mass_kg =
      capability.reference_total_mass_kg;
  EXPECT_EQ(AvailableSingleHopDeltaV(capability).reason_code,
            "HOPPER_CAPABILITY_INVALID");

  capability = ReferenceCapability();
  capability.specific_impulse_s = std::numeric_limits<double>::quiet_NaN();
  EXPECT_EQ(AvailableSingleHopDeltaV(capability).reason_code,
            "HOPPER_CAPABILITY_INVALID");
}

}  // namespace
}  // namespace lunar::pure_planning::hopper

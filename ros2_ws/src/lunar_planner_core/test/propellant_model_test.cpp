#include <cmath>
#include <limits>

#include <gtest/gtest.h>

#include "hopper/ballistic_kinematics.hpp"
#include "hopper/propellant_model.hpp"

namespace lunar::planning::hopper {
namespace {

constexpr double kTolerance = 1.0e-6;

HopperCapability ReferenceCapability() {
  return HopperCapability{
      .specific_impulse_s = 301.0,
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

HopperPropellantState ReferencePropellant() {
  return HopperPropellantState{
      .total_mass_kg = 20.0,
      .remaining_usable_fuel_mass_kg = 0.20,
  };
}

TEST(PropellantModelTest, ReproducesHundredMeterReferenceCase) {
  constexpr double flight_time_s = 100.0 / 9.0;
  const auto solved = SolveBallisticArc(
      Vec3{0.0, 0.0, 0.0}, Vec3{100.0, 0.0, 0.0},
      Vec3{0.0, 0.0, -1.62}, flight_time_s);
  ASSERT_TRUE(solved.ok()) << solved.reason_code;

  const BallisticArc& arc = *solved.arc;
  EXPECT_NEAR(std::hypot(std::hypot(arc.launch_velocity_mps.x,
                                    arc.launch_velocity_mps.y),
                         arc.launch_velocity_mps.z),
              12.727922, kTolerance);
  EXPECT_NEAR(arc.flight_time_s, 11.111111, kTolerance);
  const BallisticState apex = EvaluateBallisticState(arc, BallisticApexTime(arc));
  EXPECT_NEAR(apex.position_m.z, 25.0, kTolerance);

  const auto evaluated = EvaluatePropellant(
      arc, ReferencePropellant(), ReferenceCapability());
  ASSERT_TRUE(evaluated.ok()) << evaluated.reason_code;
  EXPECT_NEAR(evaluated.evidence->ideal_delta_v_mps, 25.455844, kTolerance);
  EXPECT_NEAR(evaluated.evidence->certified_delta_v_mps, 28.001429,
              kTolerance);
  EXPECT_NEAR(evaluated.evidence->certified_fuel_required_kg, 0.188827,
              kTolerance);
  EXPECT_NEAR(evaluated.evidence->expected_remaining_usable_fuel_kg,
              0.20 - 0.188827, kTolerance);
  EXPECT_GE(evaluated.evidence->available_delta_v_mps,
            evaluated.evidence->certified_delta_v_mps);
}

TEST(PropellantModelTest, AvailableDeltaVHasExpectedMonotonicity) {
  const HopperCapability capability = ReferenceCapability();
  const auto low_fuel = AvailableDeltaV(
      HopperPropellantState{.total_mass_kg = 20.0,
                            .remaining_usable_fuel_mass_kg = 0.10},
      capability);
  const auto high_fuel = AvailableDeltaV(ReferencePropellant(), capability);
  const auto high_mass = AvailableDeltaV(
      HopperPropellantState{.total_mass_kg = 30.0,
                            .remaining_usable_fuel_mass_kg = 0.20},
      capability);
  ASSERT_TRUE(low_fuel.ok());
  ASSERT_TRUE(high_fuel.ok());
  ASSERT_TRUE(high_mass.ok());
  EXPECT_GT(*high_fuel.delta_v_mps, *low_fuel.delta_v_mps);
  EXPECT_LT(*high_mass.delta_v_mps, *high_fuel.delta_v_mps);
}

TEST(PropellantModelTest, HandlesElevationAndExactFuelBoundary) {
  const auto solved = SolveBallisticArc(
      Vec3{0.0, 0.0, 0.0}, Vec3{100.0, 0.0, 8.0},
      Vec3{0.0, 0.0, -1.62}, 12.0);
  ASSERT_TRUE(solved.ok());
  const auto nominal = EvaluatePropellant(
      *solved.arc, ReferencePropellant(), ReferenceCapability());
  ASSERT_TRUE(nominal.ok());

  HopperPropellantState exact = ReferencePropellant();
  exact.remaining_usable_fuel_mass_kg =
      nominal.evidence->certified_fuel_required_kg;
  const auto boundary =
      EvaluatePropellant(*solved.arc, exact, ReferenceCapability());
  ASSERT_TRUE(boundary.ok()) << boundary.reason_code;
  EXPECT_NEAR(boundary.evidence->expected_remaining_usable_fuel_kg, 0.0,
              1.0e-12);

  exact.remaining_usable_fuel_mass_kg = std::nextafter(
      exact.remaining_usable_fuel_mass_kg, 0.0);
  const auto insufficient =
      EvaluatePropellant(*solved.arc, exact, ReferenceCapability());
  EXPECT_FALSE(insufficient.ok());
  EXPECT_EQ(insufficient.reason_code, "HOPPER_FUEL_INSUFFICIENT");
}

TEST(PropellantModelTest, RejectsInvalidNumericalDomains) {
  HopperCapability capability = ReferenceCapability();
  HopperPropellantState propellant = ReferencePropellant();

  propellant.total_mass_kg = 0.0;
  EXPECT_EQ(AvailableDeltaV(propellant, capability).reason_code,
            "HOPPER_TOTAL_MASS_INVALID");
  propellant = ReferencePropellant();
  propellant.remaining_usable_fuel_mass_kg = propellant.total_mass_kg;
  EXPECT_EQ(AvailableDeltaV(propellant, capability).reason_code,
            "HOPPER_USABLE_FUEL_INVALID");
  propellant = ReferencePropellant();
  capability.specific_impulse_s =
      std::numeric_limits<double>::quiet_NaN();
  EXPECT_EQ(AvailableDeltaV(propellant, capability).reason_code,
            "HOPPER_CAPABILITY_INVALID");
}

TEST(PropellantModelTest, IsBitwiseDeterministicForIdenticalInputs) {
  const auto solved = SolveBallisticArc(
      Vec3{1.0, -2.0, 0.5}, Vec3{101.0, 3.0, -1.0},
      Vec3{0.0, 0.0, -1.62}, 11.5);
  ASSERT_TRUE(solved.ok());
  const auto first = EvaluatePropellant(
      *solved.arc, ReferencePropellant(), ReferenceCapability());
  const auto second = EvaluatePropellant(
      *solved.arc, ReferencePropellant(), ReferenceCapability());
  ASSERT_TRUE(first.ok());
  ASSERT_TRUE(second.ok());
  EXPECT_DOUBLE_EQ(first.evidence->available_delta_v_mps,
                   second.evidence->available_delta_v_mps);
  EXPECT_DOUBLE_EQ(first.evidence->certified_fuel_required_kg,
                   second.evidence->certified_fuel_required_kg);
}

}  // namespace
}  // namespace lunar::planning::hopper

#include "lunar_pure_exploration_sim/coordinated_steering_plant.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

#include <gtest/gtest.h>

namespace lunar::pure_exploration_sim {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kTolerance = 1.0e-12;

TEST(CoordinatedSteeringPlantTest, MovesForwardAndReverseAlongBodyHeading) {
  const PlantState initial{.x_m = 4.0, .y_m = -2.0, .yaw_rad = kPi / 2.0};

  const auto forward =
      StepPlant(initial, BodyCommand{.longitudinal_velocity_mps = 0.5}, 0.1);
  const auto reverse =
      StepPlant(initial, BodyCommand{.longitudinal_velocity_mps = -0.5}, 0.1);

  EXPECT_NEAR(forward.x_m, 4.0, kTolerance);
  EXPECT_NEAR(forward.y_m, -1.0, kTolerance);
  EXPECT_NEAR(reverse.x_m, 4.0, kTolerance);
  EXPECT_NEAR(reverse.y_m, -3.0, kTolerance);
  EXPECT_DOUBLE_EQ(forward.longitudinal_velocity_mps, 0.5);
  EXPECT_DOUBLE_EQ(reverse.longitudinal_velocity_mps, -0.5);
}

TEST(CoordinatedSteeringPlantTest, IntegratesLeftAndRightArcsAnalytically) {
  const PlantState initial{};

  const auto left = StepPlant(
      initial,
      BodyCommand{.longitudinal_velocity_mps = 1.0, .yaw_rate_rps = 0.5},
      0.05);
  const auto right = StepPlant(
      initial,
      BodyCommand{.longitudinal_velocity_mps = 1.0, .yaw_rate_rps = -0.5},
      0.05);

  EXPECT_NEAR(left.x_m, 2.0 * std::sin(0.5), kTolerance);
  EXPECT_NEAR(left.y_m, 2.0 * (1.0 - std::cos(0.5)), kTolerance);
  EXPECT_NEAR(left.yaw_rad, 0.5, kTolerance);
  EXPECT_NEAR(right.x_m, left.x_m, kTolerance);
  EXPECT_NEAR(right.y_m, -left.y_m, kTolerance);
  EXPECT_NEAR(right.yaw_rad, -0.5, kTolerance);
}

TEST(CoordinatedSteeringPlantTest, ReverseArcKeepsYawSignAndReversesTranslation) {
  const auto reverse_left = StepPlant(
      PlantState{},
      BodyCommand{.longitudinal_velocity_mps = -1.0, .yaw_rate_rps = 0.5},
      0.05);

  EXPECT_NEAR(reverse_left.x_m, -2.0 * std::sin(0.5), kTolerance);
  EXPECT_NEAR(reverse_left.y_m, -2.0 * (1.0 - std::cos(0.5)), kTolerance);
  EXPECT_NEAR(reverse_left.yaw_rad, 0.5, kTolerance);
}

TEST(CoordinatedSteeringPlantTest,
     SpinChangesOnlyYawAndUsesTwentyTimesElapsedTime) {
  const PlantState initial{.x_m = 3.25, .y_m = -7.5, .yaw_rad = 0.1,
                           .simulated_elapsed_s = 8.0};

  const auto result = StepPlant(
      initial, BodyCommand{.yaw_rate_rps = 0.25}, 0.1);

  EXPECT_DOUBLE_EQ(result.x_m, initial.x_m);
  EXPECT_DOUBLE_EQ(result.y_m, initial.y_m);
  EXPECT_NEAR(result.yaw_rad, 0.6, kTolerance);
  EXPECT_DOUBLE_EQ(result.simulated_elapsed_s, 10.0);
  EXPECT_DOUBLE_EQ(result.longitudinal_velocity_mps, 0.0);
  EXPECT_DOUBLE_EQ(result.yaw_rate_rps, 0.25);
}

TEST(CoordinatedSteeringPlantTest, ArcSteeringIsSymmetricFrontToRear) {
  const auto result = StepPlant(
      PlantState{},
      BodyCommand{.longitudinal_velocity_mps = 1.0, .yaw_rate_rps = 0.5},
      0.01);
  const double expected = std::atan(0.8175 * 0.5 / 2.0);

  EXPECT_NEAR(result.front_steering_rad, expected, kTolerance);
  EXPECT_NEAR(result.rear_steering_rad, -expected, kTolerance);
  EXPECT_NEAR(result.wheel_angles.front_left_rad, expected, kTolerance);
  EXPECT_NEAR(result.wheel_angles.front_right_rad, expected, kTolerance);
  EXPECT_NEAR(result.wheel_angles.rear_left_rad, -expected, kTolerance);
  EXPECT_NEAR(result.wheel_angles.rear_right_rad, -expected, kTolerance);
}

TEST(CoordinatedSteeringPlantTest, SpinWheelAnglesAreTangentAboutBodyCenter) {
  const auto positive =
      StepPlant(PlantState{}, BodyCommand{.yaw_rate_rps = 0.5}, 0.01);
  const auto negative =
      StepPlant(PlantState{}, BodyCommand{.yaw_rate_rps = -0.5}, 0.01);
  const double tangent = std::atan(0.8175 / 0.67);

  EXPECT_NEAR(positive.wheel_angles.front_left_rad, -tangent, kTolerance);
  EXPECT_NEAR(positive.wheel_angles.front_right_rad, tangent, kTolerance);
  EXPECT_NEAR(positive.wheel_angles.rear_left_rad, tangent, kTolerance);
  EXPECT_NEAR(positive.wheel_angles.rear_right_rad, -tangent, kTolerance);
  EXPECT_NEAR(negative.wheel_angles.front_left_rad,
              positive.wheel_angles.front_left_rad, kTolerance);
  EXPECT_NEAR(negative.wheel_angles.front_right_rad,
              positive.wheel_angles.front_right_rad, kTolerance);
  EXPECT_NEAR(negative.wheel_angles.rear_left_rad,
              positive.wheel_angles.rear_left_rad, kTolerance);
  EXPECT_NEAR(negative.wheel_angles.rear_right_rad,
              positive.wheel_angles.rear_right_rad, kTolerance);
}

TEST(CoordinatedSteeringPlantTest, ClampsVelocityAndYawRateLimits) {
  const auto result = StepPlant(
      PlantState{},
      BodyCommand{.longitudinal_velocity_mps = 9.0, .yaw_rate_rps = -4.0},
      0.01);

  EXPECT_DOUBLE_EQ(result.longitudinal_velocity_mps, 1.5);
  EXPECT_DOUBLE_EQ(result.yaw_rate_rps, -1.0);
  EXPECT_NEAR(result.yaw_rad, -0.2, kTolerance);
  EXPECT_NEAR(result.x_m, 1.5 * std::sin(0.2), kTolerance);
  EXPECT_NEAR(result.y_m, -1.5 * (1.0 - std::cos(0.2)), kTolerance);
}

TEST(CoordinatedSteeringPlantTest, NonFiniteCommandStopsSafelyButTimeAdvances) {
  const PlantState initial{.x_m = 1.0, .y_m = 2.0, .yaw_rad = 0.3,
                           .longitudinal_velocity_mps = 1.0,
                           .yaw_rate_rps = 0.5,
                           .simulated_elapsed_s = 4.0};

  const auto nan_velocity = StepPlant(
      initial,
      BodyCommand{.longitudinal_velocity_mps =
                      std::numeric_limits<double>::quiet_NaN(),
                  .yaw_rate_rps = 0.5},
      0.1);
  const auto infinite_yaw = StepPlant(
      initial,
      BodyCommand{.longitudinal_velocity_mps = 1.0,
                  .yaw_rate_rps = std::numeric_limits<double>::infinity()},
      0.1);

  for (const auto* result : {&nan_velocity, &infinite_yaw}) {
    EXPECT_DOUBLE_EQ(result->x_m, initial.x_m);
    EXPECT_DOUBLE_EQ(result->y_m, initial.y_m);
    EXPECT_DOUBLE_EQ(result->yaw_rad, initial.yaw_rad);
    EXPECT_DOUBLE_EQ(result->longitudinal_velocity_mps, 0.0);
    EXPECT_DOUBLE_EQ(result->yaw_rate_rps, 0.0);
    EXPECT_DOUBLE_EQ(result->simulated_elapsed_s, 6.0);
  }
}

TEST(CoordinatedSteeringPlantTest, RejectsNegativeOrNonFiniteWallTime) {
  const auto step = [](double wall_dt_s) {
    const auto result = StepPlant(PlantState{}, BodyCommand{}, wall_dt_s);
    static_cast<void>(result);
  };

  EXPECT_THROW(step(-0.01), std::invalid_argument);
  EXPECT_THROW(step(std::numeric_limits<double>::quiet_NaN()),
               std::invalid_argument);
  EXPECT_THROW(step(std::numeric_limits<double>::infinity()),
               std::invalid_argument);
}

TEST(CoordinatedSteeringPlantTest, NormalizesYawToClosedOpenPiInterval) {
  const auto positive_crossing = StepPlant(
      PlantState{.yaw_rad = kPi - 0.1}, BodyCommand{.yaw_rate_rps = 1.0},
      0.01);
  const auto exact_positive_pi = StepPlant(
      PlantState{.yaw_rad = kPi - 0.2}, BodyCommand{.yaw_rate_rps = 1.0},
      0.01);
  const auto negative_crossing = StepPlant(
      PlantState{.yaw_rad = -kPi + 0.1}, BodyCommand{.yaw_rate_rps = -1.0},
      0.01);

  EXPECT_NEAR(positive_crossing.yaw_rad, -kPi + 0.1, kTolerance);
  EXPECT_NEAR(exact_positive_pi.yaw_rad, -kPi, kTolerance);
  EXPECT_NEAR(negative_crossing.yaw_rad, kPi - 0.1, kTolerance);
  EXPECT_GE(positive_crossing.yaw_rad, -kPi);
  EXPECT_LT(positive_crossing.yaw_rad, kPi);
}

}  // namespace
}  // namespace lunar::pure_exploration_sim

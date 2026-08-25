#include "lunar_pure_exploration_sim/coordinated_steering_plant.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace lunar::pure_exploration_sim {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kTwoPi = 2.0 * kPi;
constexpr double kMotionEpsilon = 1.0e-12;

double NormalizeYaw(double yaw_rad) {
  double wrapped = std::fmod(yaw_rad + kPi, kTwoPi);
  if (wrapped < 0.0) {
    wrapped += kTwoPi;
  }
  return wrapped - kPi;
}

void ValidateParameters(const PlantParameters& parameters) {
  if (!std::isfinite(parameters.wheelbase_m) ||
      !std::isfinite(parameters.track_width_m) ||
      !std::isfinite(parameters.speed_multiplier) ||
      !std::isfinite(parameters.max_abs_velocity_mps) ||
      !std::isfinite(parameters.max_abs_yaw_rate_rps) ||
      parameters.wheelbase_m <= 0.0 || parameters.track_width_m <= 0.0 ||
      parameters.speed_multiplier <= 0.0 ||
      parameters.max_abs_velocity_mps <= 0.0 ||
      parameters.max_abs_yaw_rate_rps <= 0.0) {
    throw std::invalid_argument("plant parameters must be finite and positive");
  }
}

WheelDisplayAngles SpinWheelAngles(const PlantParameters& parameters) {
  const double tangent_rad =
      std::atan(parameters.wheelbase_m / parameters.track_width_m);
  return WheelDisplayAngles{
      .front_left_rad = -tangent_rad,
      .front_right_rad = tangent_rad,
      .rear_left_rad = tangent_rad,
      .rear_right_rad = -tangent_rad,
  };
}

}  // namespace

PlantState StepPlant(PlantState state, BodyCommand command, double wall_dt_s,
                     PlantParameters parameters) {
  if (!std::isfinite(wall_dt_s) || wall_dt_s < 0.0) {
    throw std::invalid_argument("wall_dt_s must be finite and non-negative");
  }
  ValidateParameters(parameters);

  if (!std::isfinite(command.longitudinal_velocity_mps) ||
      !std::isfinite(command.yaw_rate_rps)) {
    command = BodyCommand{};
  }

  const double velocity_mps =
      std::clamp(command.longitudinal_velocity_mps,
                 -parameters.max_abs_velocity_mps,
                 parameters.max_abs_velocity_mps);
  const double yaw_rate_rps =
      std::clamp(command.yaw_rate_rps, -parameters.max_abs_yaw_rate_rps,
                 parameters.max_abs_yaw_rate_rps);
  const double simulated_dt_s = wall_dt_s * parameters.speed_multiplier;
  const double initial_yaw_rad = state.yaw_rad;
  const double final_yaw_rad = initial_yaw_rad + yaw_rate_rps * simulated_dt_s;

  state.longitudinal_velocity_mps = velocity_mps;
  state.yaw_rate_rps = yaw_rate_rps;
  state.simulated_elapsed_s += simulated_dt_s;

  if (std::abs(velocity_mps) <= kMotionEpsilon) {
    if (std::abs(yaw_rate_rps) > kMotionEpsilon) {
      state.wheel_angles = SpinWheelAngles(parameters);
      state.front_steering_rad = state.wheel_angles.front_left_rad;
      state.rear_steering_rad = state.wheel_angles.rear_left_rad;
    } else {
      state.front_steering_rad = 0.0;
      state.rear_steering_rad = 0.0;
      state.wheel_angles = WheelDisplayAngles{};
    }
  } else if (std::abs(yaw_rate_rps) <= kMotionEpsilon) {
    const double distance_m = velocity_mps * simulated_dt_s;
    state.x_m += distance_m * std::cos(initial_yaw_rad);
    state.y_m += distance_m * std::sin(initial_yaw_rad);
    state.front_steering_rad = 0.0;
    state.rear_steering_rad = 0.0;
    state.wheel_angles = WheelDisplayAngles{};
  } else {
    const double radius_m = velocity_mps / yaw_rate_rps;
    state.x_m += radius_m *
                 (std::sin(final_yaw_rad) - std::sin(initial_yaw_rad));
    state.y_m -= radius_m *
                 (std::cos(final_yaw_rad) - std::cos(initial_yaw_rad));

    const double curvature_per_m = yaw_rate_rps / velocity_mps;
    state.front_steering_rad =
        std::atan(parameters.wheelbase_m * curvature_per_m / 2.0);
    state.rear_steering_rad = -state.front_steering_rad;
    state.wheel_angles = WheelDisplayAngles{
        .front_left_rad = state.front_steering_rad,
        .front_right_rad = state.front_steering_rad,
        .rear_left_rad = state.rear_steering_rad,
        .rear_right_rad = state.rear_steering_rad,
    };
  }

  state.yaw_rad = NormalizeYaw(final_yaw_rad);
  return state;
}

}  // namespace lunar::pure_exploration_sim

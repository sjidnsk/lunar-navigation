#pragma once

namespace lunar::pure_exploration_sim {

struct BodyCommand {
  double longitudinal_velocity_mps{0.0};
  double yaw_rate_rps{0.0};
};

struct PlantParameters {
  double wheelbase_m{0.8175};
  double track_width_m{0.67};
  double speed_multiplier{20.0};
  double max_abs_velocity_mps{1.5};
  double max_abs_yaw_rate_rps{1.0};
};

struct WheelDisplayAngles {
  double front_left_rad{0.0};
  double front_right_rad{0.0};
  double rear_left_rad{0.0};
  double rear_right_rad{0.0};
};

struct PlantState {
  double x_m{0.0};
  double y_m{0.0};
  double yaw_rad{0.0};
  double longitudinal_velocity_mps{0.0};
  double yaw_rate_rps{0.0};
  double front_steering_rad{0.0};
  double rear_steering_rad{0.0};
  WheelDisplayAngles wheel_angles{};
  double simulated_elapsed_s{0.0};
};

// Advances a constant body twist and normalizes yaw to [-pi, pi).
// A non-finite command is treated as a stop command. Invalid wall time or
// plant parameters are rejected with std::invalid_argument.
[[nodiscard]] PlantState StepPlant(
    PlantState state, BodyCommand command, double wall_dt_s,
    PlantParameters parameters = PlantParameters{});

}  // namespace lunar::pure_exploration_sim
